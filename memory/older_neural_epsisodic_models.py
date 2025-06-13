from typing import Any, Dict, List, Optional, Tuple

import einops
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import einsum
from jaxtyping import Float, Integer
from torch import Tensor


class FastWeightsEpisodicMemory(nn.Module):
    """Fast weights episodic memory from AR conversation.

    Implements the fast-weight matrix F_t that is updated online via outer products:
    F_t = F_{t-1} + η * k_t * v_t^T
    """

    def __init__(self, dim: int = 64, lr: float = 0.01):
        """Initialize fast weights episodic memory.

        Args:
            dim: Feature dimension
            lr: Learning rate for fast weight updates (η)

        """
        super().__init__()

        self.dim = dim
        self.lr = lr

        # Fast weight matrix F_t ∈ R^{d×d}
        self.register_buffer("fast_weights", torch.zeros(dim, dim))

        # Projection matrices for episodic key/value spaces
        self.W_K_fast = nn.Parameter(torch.randn(dim, dim) * 0.1)
        self.W_V_fast = nn.Parameter(torch.randn(dim, dim) * 0.1)

        # Query projection for retrieval
        self.W_Q_fast = nn.Parameter(torch.randn(dim, dim) * 0.1)

        self.current_time = 0

    def forward(self, query: Tensor) -> Tensor:
        """Retrieve from fast weights memory.

        Args:
            query: Query tensor [batch, dim]

        Returns:
            Retrieved values [batch, dim]

        """
        # Project query: q = W_Q_fast * query
        q = query @ self.W_Q_fast

        # Retrieve via: A_fast = softmax(Q * F_t^T / √d)
        attention_scores = q @ self.fast_weights.T / np.sqrt(self.dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Retrieved values (simplified - using fast_weights as both keys and values)
        retrieved = attention_weights @ self.fast_weights

        return retrieved

    def store_episode(self, keys: Tensor, values: Tensor, **kwargs) -> None:
        """Store episode using fast weight outer product update.

        Args:
            keys: Key vectors [batch, dim]
            values: Value vectors [batch, dim]

        """
        batch_size = keys.shape[0]

        for i in range(batch_size):
            # Project to episodic spaces
            k_t = keys[i] @ self.W_K_fast  # k_t = W_K_fast * h_t
            v_t = values[i] @ self.W_V_fast  # v_t = W_V_fast * h_t

            # Fast weight update: F_t = F_{t-1} + η * k_t * v_t^T
            outer_product = torch.outer(k_t, v_t)
            self.fast_weights.data += self.lr * outer_product

            self.current_time += 1


class ContextDriftMemory(nn.Module):
    """Context drift memory with exponential decay.

    Implements: c_t = λ * c_{t-1} + U * h_t
    """

    def __init__(self, dim: int = 64, context_dim: int = 32, decay: float = 0.9):
        """Initialize context drift memory.

        Args:
            dim: Input feature dimension
            context_dim: Context vector dimension
            decay: Decay factor λ ∈ (0,1)

        """
        super().__init__()

        self.dim = dim
        self.context_dim = context_dim
        self.decay = decay

        # Context update matrix U ∈ R^{context_dim × dim}
        self.U = nn.Parameter(torch.randn(context_dim, dim) * 0.1)

        # Context integration matrix W_c ∈ R^{dim × context_dim}
        self.W_c = nn.Parameter(torch.randn(dim, context_dim) * 0.1)

        # Current context state c_t ∈ R^{context_dim}
        self.register_buffer("context_state", torch.zeros(context_dim))

        self.current_time = 0

    def forward(self, query: Tensor, update_context: bool = True) -> Tensor:
        """Retrieve with context-augmented query.

        Args:
            query: Query tensor [batch, dim]
            update_context: Whether to update context state

        Returns:
            Context-augmented query [batch, dim]

        """
        if update_context:
            # Update context: c_t = λ * c_{t-1} + U * h_t
            # Use mean of batch for context update
            h_t = query.mean(dim=0)  # [dim]
            self.context_state.data = self.decay * self.context_state + self.U @ h_t
            self.current_time += 1

        # Augment query with context: h̃ = h + W_c * c_t
        context_contribution = self.context_state @ self.W_c.T  # [dim]
        augmented_query = query + context_contribution.unsqueeze(0)

        return augmented_query

    def store_episode(self, keys: Tensor, values: Tensor, **kwargs) -> None:
        """Store episode by updating context state.

        Args:
            keys: Key vectors [batch, dim]
            values: Value vectors [batch, dim]

        """
        # Context drift memory doesn't explicitly store episodes
        # Storage happens implicitly through context updates during forward pass
        pass

    def reset_context(self):
        """Reset context state to zero."""
        self.context_state.data.zero_()


class GatedEpisodicMemory(nn.Module):
    """Gated episodic memory with selective write gating.

    Implements: g_t = σ(w_g^T [h_t; c_t] + b_g)
    """

    def __init__(
        self,
        dim: int = 64,
        context_dim: int = 32,
        memory_size: int = 100,
        lr: float = 0.01,
    ):
        """Initialize gated episodic memory.

        Args:
            dim: Feature dimension
            context_dim: Context dimension
            memory_size: Number of memory slots
            lr: Learning rate for memory updates

        """
        super().__init__()

        self.dim = dim
        self.context_dim = context_dim
        self.memory_size = memory_size
        self.lr = lr

        # Memory storage
        self.mem_keys = nn.Parameter(torch.randn(memory_size, dim) * 0.1)
        self.mem_values = nn.Parameter(torch.randn(memory_size, dim) * 0.1)

        # Context drift component
        self.context_drift = ContextDriftMemory(dim, context_dim)

        # Write gate network: g_t = σ(w_g^T [h_t; c_t] + b_g)
        self.gate_network = nn.Sequential(
            nn.Linear(dim + context_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

        self.write_pointer = 0
        self.current_time = 0

        # Track which slots are used
        self.used_slots = torch.zeros(memory_size, dtype=torch.bool)

    def forward(self, query: Tensor) -> Tensor:
        """Retrieve from gated episodic memory.

        Args:
            query: Query tensor [batch, dim]

        Returns:
            Retrieved values [batch, dim]

        """
        # Get context-augmented query
        augmented_query = self.context_drift(query, update_context=False)

        if self.used_slots.sum() == 0:
            return torch.zeros_like(query)

        # Standard attention-based retrieval
        used_mask = self.used_slots.float()

        attention_scores = augmented_query @ self.mem_keys.T
        attention_scores = attention_scores * used_mask.unsqueeze(0)
        attention_scores = attention_scores + (1 - used_mask.unsqueeze(0)) * (-1e9)

        attention_weights = F.softmax(attention_scores, dim=-1)
        retrieved = attention_weights @ self.mem_values

        return retrieved

    def store_episode(self, keys: Tensor, values: Tensor, **kwargs) -> None:
        """Store episode with gating mechanism.

        Args:
            keys: Key vectors [batch, dim]
            values: Value vectors [batch, dim]

        """
        batch_size = keys.shape[0]

        for i in range(batch_size):
            # Update context drift
            _ = self.context_drift(keys[i : i + 1], update_context=True)

            # Compute write gate: g_t = σ(w_g^T [h_t; c_t] + b_g)
            h_t = keys[i]  # [dim]
            c_t = self.context_drift.context_state  # [context_dim]

            gate_input = torch.cat([h_t, c_t])  # [dim + context_dim]
            write_gate = self.gate_network(gate_input).item()  # Scalar

            # Only store if gate is open (above threshold)
            if write_gate > 0.5:  # Threshold for writing
                idx = self.write_pointer % self.memory_size
                self.used_slots[idx] = True

                # Store with gate modulation
                self.mem_keys.data[idx] = keys[i].detach()
                self.mem_values.data[idx] = values[i].detach()

                self.write_pointer += 1

            self.current_time += 1


class MetaLearnedFastWeights(nn.Module):
    """Meta-learned fast weights using hypernetwork.

    Implements: ΔF_t = G_φ(h_t), F_t = F_{t-1} + ΔF_t
    """

    def __init__(self, dim: int = 64, hidden_dim: int = 128):
        """Initialize meta-learned fast weights.

        Args:
            dim: Feature dimension
            hidden_dim: Hidden dimension for hypernetwork

        """
        super().__init__()

        self.dim = dim

        # Fast weight matrix F_t ∈ R^{d×d}
        self.register_buffer("fast_weights", torch.zeros(dim, dim))

        # Hypernetwork G_φ that generates ΔF_t
        self.hypernetwork = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim * dim),  # Output ΔF_t flattened
            nn.Tanh(),  # Bound the updates
        )

        # Scale factor for updates
        self.update_scale = nn.Parameter(torch.tensor(0.01))

        self.current_time = 0

    def forward(self, query: Tensor) -> Tensor:
        """Retrieve from meta-learned fast weights.

        Args:
            query: Query tensor [batch, dim]

        Returns:
            Retrieved values [batch, dim]

        """
        # Simple retrieval via matrix multiplication
        retrieved = query @ self.fast_weights
        return retrieved

    def store_episode(self, keys: Tensor, values: Tensor, **kwargs) -> None:
        """Store episode using meta-learned updates.

        Args:
            keys: Key vectors [batch, dim]
            values: Value vectors [batch, dim]

        """
        batch_size = keys.shape[0]

        for i in range(batch_size):
            h_t = keys[i]  # [dim]

            # Generate update via hypernetwork: ΔF_t = G_φ(h_t)
            delta_F_flat = self.hypernetwork(h_t)  # [dim²]
            delta_F = delta_F_flat.view(self.dim, self.dim)  # [dim, dim]

            # Update fast weights: F_t = F_{t-1} + ΔF_t
            self.fast_weights.data += self.update_scale * delta_F

            self.current_time += 1


class RAGEpisodicMemory(nn.Module):
    """Retrieval-Augmented Generation episodic memory.

    Implements external vector database with nearest-neighbor retrieval.
    """

    def __init__(self, dim: int = 64, max_memories: int = 1000, top_k: int = 5):
        """Initialize RAG episodic memory.

        Args:
            dim: Feature dimension
            max_memories: Maximum number of stored memories
            top_k: Number of top memories to retrieve

        """
        super().__init__()

        self.dim = dim
        self.max_memories = max_memories
        self.top_k = top_k

        # External memory storage (not parameters - true external storage)
        self.memory_keys: List[Tensor] = []
        self.memory_values: List[Tensor] = []
        self.memory_metadata: List[Dict[str, Any]] = []

        # Query projection for retrieval
        self.W_q = nn.Parameter(torch.randn(dim, dim) * 0.1)

        # Cross-attention for fusion
        self.cross_attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

        self.current_time = 0

    def forward(self, query: Tensor) -> Tensor:
        """Retrieve using nearest-neighbor search and cross-attention.

        Args:
            query: Query tensor [batch, dim]

        Returns:
            Retrieved and fused values [batch, dim]

        """
        if len(self.memory_keys) == 0:
            return torch.zeros_like(query)

        batch_size = query.shape[0]
        retrieved_batch = []

        for i in range(batch_size):
            # Project query: q_t = W_q * h_t
            q_t = query[i] @ self.W_q  # [dim]

            # Compute similarities with all stored memories
            similarities = []
            for mem_key in self.memory_keys:
                sim = F.cosine_similarity(q_t.unsqueeze(0), mem_key.unsqueeze(0)).item()
                similarities.append(sim)

            # Get top-k nearest neighbors
            if len(similarities) >= self.top_k:
                top_k_indices = torch.topk(
                    torch.tensor(similarities), self.top_k
                ).indices
            else:
                top_k_indices = torch.arange(len(similarities))

            # Retrieve top-k keys and values
            retrieved_keys = torch.stack(
                [self.memory_keys[idx] for idx in top_k_indices]
            )
            retrieved_values = torch.stack(
                [self.memory_values[idx] for idx in top_k_indices]
            )

            # Cross-attention fusion
            q_expanded = q_t.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
            k_expanded = retrieved_keys.unsqueeze(0)  # [1, top_k, dim]
            v_expanded = retrieved_values.unsqueeze(0)  # [1, top_k, dim]

            fused, _ = self.cross_attention(q_expanded, k_expanded, v_expanded)
            retrieved_batch.append(fused.squeeze(0).squeeze(0))  # [dim]

        return torch.stack(retrieved_batch)  # [batch, dim]

    def store_episode(self, keys: Tensor, values: Tensor, **kwargs) -> None:
        """Store episode in external memory.

        Args:
            keys: Key vectors [batch, dim]
            values: Value vectors [batch, dim]

        """
        batch_size = keys.shape[0]

        for i in range(batch_size):
            # Store in external memory (detached from computation graph)
            self.memory_keys.append(keys[i].detach().clone())
            self.memory_values.append(values[i].detach().clone())
            self.memory_metadata.append(
                {"timestamp": self.current_time, "index": len(self.memory_keys) - 1}
            )

            # Maintain maximum memory size (FIFO)
            if len(self.memory_keys) > self.max_memories:
                self.memory_keys.pop(0)
                self.memory_values.pop(0)
                self.memory_metadata.pop(0)

            self.current_time += 1


class GraphStructuredMemory(nn.Module):
    """Graph-structured semantic memory with dynamic node updates.

    Implements dynamic graph with learnable adjacency and graph convolutions.
    """

    def __init__(
        self,
        dim: int = 64,
        num_nodes: int = 50,
        num_heads: int = 4,
        update_lr: float = 0.01,
    ):
        """Initialize graph-structured memory.

        Args:
            dim: Feature dimension
            num_nodes: Number of graph nodes
            num_heads: Number of attention heads
            update_lr: Learning rate for node updates

        """
        super().__init__()

        self.dim = dim
        self.num_nodes = num_nodes
        self.update_lr = update_lr

        # Graph nodes n_j ∈ R^d
        self.nodes = nn.Parameter(torch.randn(num_nodes, dim) * 0.1)

        # Edge weight computation: E_{jk} = σ(n_j^T W_e n_k)
        self.W_e = nn.Parameter(torch.randn(dim, dim) * 0.1)

        # Graph convolution weight: n_j' = Σ_k E_{jk} W_g n_k
        self.W_g = nn.Parameter(torch.randn(dim, dim) * 0.1)

        # Attention for injecting graph memory back
        self.graph_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # Node update network
        self.node_update_net = nn.Sequential(
            nn.Linear(dim * 2, dim),  # [current_node, input]
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

        self.current_time = 0

    def compute_adjacency(self) -> Tensor:
        """Compute dynamic adjacency matrix.

        Returns:
            Adjacency matrix [num_nodes, num_nodes]

        """
        # E_{jk} = σ(n_j^T W_e n_k)
        edge_logits = self.nodes @ self.W_e @ self.nodes.T  # [num_nodes, num_nodes]
        adjacency = torch.sigmoid(edge_logits)
        return adjacency

    def graph_convolution(self) -> Tensor:
        """Perform graph convolution to update node features.

        Returns:
            Updated node features [num_nodes, dim]

        """
        adjacency = self.compute_adjacency()  # [num_nodes, num_nodes]

        # n_j' = Σ_k E_{jk} W_g n_k
        updated_nodes = adjacency @ self.nodes @ self.W_g.T  # [num_nodes, dim]

        return updated_nodes

    def forward(self, query: Tensor) -> Tensor:
        """Retrieve from graph-structured memory.

        Args:
            query: Query tensor [batch, dim]

        Returns:
            Graph-augmented output [batch, dim]

        """
        # Update graph nodes via convolution
        updated_nodes = self.graph_convolution()  # [num_nodes, dim]

        # Inject graph memory via attention: A_graph = softmax(Q N'^T)
        query_expanded = query.unsqueeze(1)  # [batch, 1, dim]
        nodes_expanded = updated_nodes.unsqueeze(0).expand(
            query.shape[0], -1, -1
        )  # [batch, num_nodes, dim]

        graph_output, _ = self.graph_attention(
            query_expanded, nodes_expanded, nodes_expanded
        )  # [batch, 1, dim]

        return graph_output.squeeze(1)  # [batch, dim]

    def store_episode(self, keys: Tensor, values: Tensor, **kwargs) -> None:
        """Store episode by updating graph nodes.

        Args:
            keys: Key vectors [batch, dim]
            values: Value vectors [batch, dim]

        """
        batch_size = keys.shape[0]

        for i in range(batch_size):
            # Find most similar node
            similarities = F.cosine_similarity(
                keys[i].unsqueeze(0), self.nodes, dim=1
            )  # [num_nodes]
            best_node_idx = torch.argmax(similarities).item()

            # Update the most similar node
            current_node = self.nodes[best_node_idx]  # [dim]
            update_input = torch.cat([current_node, values[i]])  # [2*dim]

            node_update = self.node_update_net(update_input)  # [dim]

            # Apply update with learning rate
            self.nodes.data[best_node_idx] += self.update_lr * node_update

            self.current_time += 1


# Test functions for comparing AR models with existing models
def test_ar_memory_comparison(
    dim: int = 32, num_episodes: int = 20
) -> Dict[str, float]:
    """Test AR memory models and compare with existing models.

    Args:
        dim: Feature dimension
        num_episodes: Number of episodes to test

    Returns:
        Dictionary of test results

    """
    print("=== Testing AR Memory Models ===")

    # Initialize AR models
    ar_models = {
        "FastWeights": FastWeightsEpisodicMemory(dim=dim),
        "ContextDrift": ContextDriftMemory(dim=dim),
        "Gated": GatedEpisodicMemory(dim=dim),
        "MetaLearned": MetaLearnedFastWeights(dim=dim),
        "RAG": RAGEpisodicMemory(dim=dim),
        "GraphStructured": GraphStructuredMemory(dim=dim),
    }

    # Generate test data
    test_keys = torch.randn(num_episodes, dim)
    test_values = torch.randn(num_episodes, dim)

    results = {}

    for name, model in ar_models.items():
        print(f"\nTesting {name}...")

        # Store episodes
        model.store_episode(test_keys, test_values)

        # Test retrieval
        with torch.no_grad():
            retrieved = model(test_keys)

            # Compute similarity
            similarities = F.cosine_similarity(retrieved, test_values, dim=1)
            avg_similarity = similarities.mean().item()

            results[name] = avg_similarity
            print(f"{name} average similarity: {avg_similarity:.3f}")

    return results


def test_ar_temporal_interference(dim: int = 32) -> Dict[str, float]:
    """Test temporal interference in AR memory models.

    Args:
        dim: Feature dimension

    Returns:
        Dictionary of interference results

    """
    print("=== Testing AR Temporal Interference ===")

    # Initialize models
    models = {
        "FastWeights": FastWeightsEpisodicMemory(dim=dim),
        "ContextDrift": ContextDriftMemory(dim=dim),
        "Gated": GatedEpisodicMemory(dim=dim),
    }

    # Phase 1: Store initial memories
    initial_keys = torch.randn(5, dim)
    initial_values = torch.randn(5, dim)

    results = {}

    for name, model in models.items():
        print(f"\nTesting {name}...")

        # Store initial episodes
        model.store_episode(initial_keys, initial_values)

        # Test immediate recall
        with torch.no_grad():
            immediate_retrieved = model(initial_keys)
            immediate_similarity = (
                F.cosine_similarity(immediate_retrieved, initial_values, dim=1)
                .mean()
                .item()
            )

        # Add interfering memories
        interfering_keys = initial_keys + 0.1 * torch.randn_like(initial_keys)
        interfering_values = torch.randn_like(initial_values)
        model.store_episode(interfering_keys, interfering_values)

        # Test delayed recall
        with torch.no_grad():
            delayed_retrieved = model(initial_keys)
            delayed_similarity = (
                F.cosine_similarity(delayed_retrieved, initial_values, dim=1)
                .mean()
                .item()
            )

        interference_effect = immediate_similarity - delayed_similarity
        results[name] = interference_effect

        print(f"{name} interference effect: {interference_effect:.3f}")

    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("=== Testing AR Memory Models ===\n")

    # Test basic functionality
    comparison_results = test_ar_memory_comparison()
    print(f"\nComparison Results: {comparison_results}")

    # Test temporal interference
    interference_results = test_ar_temporal_interference()
    print(f"\nInterference Results: {interference_results}")

    print("\n=== AR Memory Models Testing Complete ===")
