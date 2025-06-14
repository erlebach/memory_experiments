from typing import Tuple

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import einsum
from jaxtyping import Float, Integer
from torch import Tensor


@beartype
class TrueEpisodicMemory(nn.Module):
    """Episodic memory with temporal and contextual binding."""

    def __init__(self, dim, memory_size=100, context_dim=16, lr=0.01):
        super().__init__()
        # Initialize with small random values to avoid perfect matches
        self.mem_keys = nn.Parameter(torch.randn(memory_size, dim) * 0.1)
        self.mem_values = nn.Parameter(torch.randn(memory_size, dim) * 0.1)
        self.mem_contexts = nn.Parameter(torch.randn(memory_size, context_dim) * 0.1)
        # Timestamps should not be trainable - they represent actual storage time
        self.register_buffer("mem_timestamps", torch.zeros(memory_size, 1))
        self.mem_keys: Float[Tensor, "memory_size dim"]
        self.mem_values: Float[Tensor, "memory_size dim"]
        self.mem_contexts: Float[Tensor, "memory_size context_dim"]
        self.mem_timestamps: Float[Tensor, "memory_size 1"]

        self.dim = dim
        self.context_dim = context_dim
        self.memory_size = memory_size
        self.current_time = 0
        self.write_pointer = 0

        # Track which slots are actually used
        self.used_slots = torch.zeros(memory_size, dtype=torch.bool)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(
        self,
        query: Float[Tensor, "batch query_dim"],  # noqa: F722
        context: Float[Tensor, "batch context_dim"] | None = None,  # noqa: F722
        time_weight: float = 0.1,
    ):
        """Retrieve with temporal and contextual cues."""
        # Only consider used memory slots
        if self.used_slots.sum() == 0:
            # No memories stored yet, return random
            return torch.randn_like(query)

        # Mask for used slots
        used_mask = self.used_slots.float()

        # Standard content-based attention
        content_scores = einsum(
            query,
            self.mem_keys,
            # query @ self.mem_keys.T
            "batch query_dim, memory_size query_dim -> batch memory_size",
        )

        # Add contextual similarity if context provided
        if context is not None:
            context_scores = einsum(
                context,
                self.mem_contexts,
                # context @ self.mem_contexts.T
                "batch context_dim, memory_size context_dim -> batch memory_size",
            )
            content_scores += +0.5 * context_scores

        # Add temporal recency bias (more recent = higher weight)
        time_decay = torch.exp(
            -time_weight * (self.current_time - self.mem_timestamps.squeeze())
        )
        content_scores = content_scores + 0.3 * time_decay.unsqueeze(0)

        # Mask unused slots
        print(f"{content_scores.shape=}, {used_mask.shape=}")
        print(f"{used_mask=}")
        content_scores = content_scores * used_mask.unsqueeze(0)
        content_scores = content_scores + (1 - used_mask.unsqueeze(0)) * (
            -1e9
        )  # Large negative for unused

        attn_probs = F.softmax(content_scores, dim=-1)
        # retrieved_value = torch.matmul(attn_probs, self.mem_values)
        retrieved_value = einsum(
            attn_probs,
            self.mem_values,
            "batch memory_size, memory_size dim -> batch dim",
        )
        return retrieved_value

    def store_episode(
        self,
        keys: Float[Tensor, "num_episodes key_dim"],  #  noqa: F722
        values: Float[Tensor, "num_episodes value_dim"],  # noqa: F722
        context=None,
        inner_steps=3,  # Learning rate compensation per episode
        outer_steps=2,  # Spaced repetition rounds
    ):
        """Store new episode with learning-based updates."""
        num_episodes = keys.shape[0]

        # Pre-allocate all slots
        episode_slots = []
        for i in range(num_episodes):
            idx = self.write_pointer % self.memory_size
            episode_slots.append(idx)
            self.used_slots[idx] = True
            self.mem_timestamps[idx] = self.current_time + i
            self.write_pointer += 1

        # TRUE HYBRID: Outer loop (spaced repetition) × Inner loop (learning rate compensation)
        for outer_round in range(outer_steps):  # Spaced repetition rounds
            print(f"Outer round {outer_round}")
            for i in range(num_episodes):  # Cycle through all episodes
                for inner_round in range(inner_steps):  # Multiple updates per episode
                    print(f"Inner round {inner_round}, episode {i}")
                    self.optimizer.zero_grad()
                    ctx = context[i : i + 1] if context is not None else None
                    retrieved = self.forward(keys[i : i + 1], ctx)
                    loss = F.mse_loss(retrieved, values[i : i + 1])
                    loss.backward()
                    self.optimizer.step()

        self.current_time += num_episodes


class EpisodicContentClassifier(nn.Module):
    """Classifies types of episodic content to determine optimal learning parameters."""

    def __init__(self, dim: int, context_dim: int, num_content_types: int = 4):
        """Initialize episodic content classifier.

        Args:
            dim: Feature dimension of episodic content
            context_dim: Context vector dimension
            num_content_types: Number of episodic content types to classify

        """
        super().__init__()

        self.content_classifier = nn.Sequential(
            nn.Linear(dim + context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_content_types),
        )

        # Learnable parameters for each content type
        # [visual, social, emotional, routine] episodes
        self.inner_steps_weights = nn.Parameter(torch.tensor([3.0, 4.0, 2.0, 3.0]))
        self.outer_steps_weights = nn.Parameter(torch.tensor([2.0, 3.0, 1.0, 3.0]))
        self.threshold_weights = nn.Parameter(torch.tensor([0.1, 0.08, 0.15, 0.05]))

    def forward(
        self,
        keys: Float[Tensor, "batch dim"],
        context: Float[Tensor, "batch context_dim"] | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Classify content and return adaptive parameters.

        Returns:
            inner_steps: Recommended inner steps per episode
            outer_steps: Recommended outer steps per episode
            thresholds: Convergence thresholds per episode

        """
        if context is not None:
            features = torch.cat([keys, context], dim=-1)
        else:
            # Use zeros for missing context
            batch_size = keys.shape[0]
            zero_context = torch.zeros(
                batch_size, keys.shape[-1] - keys.shape[-1], device=keys.device
            )
            features = torch.cat([keys, zero_context], dim=-1)

        # Get content type probabilities
        type_logits = self.content_classifier(features)
        type_probs = F.softmax(type_logits, dim=-1)

        # Compute weighted adaptive parameters
        inner_steps = torch.sum(type_probs * self.inner_steps_weights, dim=-1)
        outer_steps = torch.sum(type_probs * self.outer_steps_weights, dim=-1)
        thresholds = torch.sum(type_probs * self.threshold_weights, dim=-1)

        return inner_steps, outer_steps, thresholds


class AdaptiveEpisodicMemory(nn.Module):
    """Episodic memory with adaptive learning based on content type."""

    def __init__(
        self, dim: int, memory_size: int = 100, context_dim: int = 16, lr: float = 0.01
    ):
        """Initialize adaptive episodic memory.

        Args:
            dim: Feature dimension
            memory_size: Number of memory slots
            context_dim: Context vector dimension
            lr: Learning rate

        """
        super().__init__()

        # Core episodic memory (unchanged)
        self.episodic_memory = TrueEpisodicMemory(
            dim=dim, memory_size=memory_size, context_dim=context_dim, lr=lr
        )

        # Adaptive content classifier
        self.content_classifier = EpisodicContentClassifier(dim, context_dim)

        # Optimizer for the classifier
        self.classifier_optimizer = torch.optim.Adam(
            self.content_classifier.parameters(), lr=lr
        )

    def forward(
        self,
        query: Float[Tensor, "batch query_dim"],
        context: Float[Tensor, "batch context_dim"] | None = None,
        time_weight: float = 0.1,
    ):
        """Forward pass through core episodic memory."""
        return self.episodic_memory.forward(query, context, time_weight)

    def store_episode_adaptive(
        self,
        keys: Float[Tensor, "num_episodes key_dim"],
        values: Float[Tensor, "num_episodes value_dim"],
        context: Float[Tensor, "num_episodes context_dim"] | None = None,
        use_adaptive: bool = True,
        default_inner_steps: int = 3,
        default_outer_steps: int = 2,
    ):
        """Store episodes with adaptive learning parameters.

        Args:
            keys: Episode keys to store
            values: Episode values to store
            context: Episode contexts
            use_adaptive: Whether to use adaptive parameters
            default_inner_steps: Default inner steps if not adaptive
            default_outer_steps: Default outer steps if not adaptive

        """
        if use_adaptive:
            # Get adaptive parameters for each episode
            inner_steps, outer_steps, thresholds = self.content_classifier(
                keys, context
            )

            # Convert to integers (minimum 1)
            inner_steps = torch.clamp(inner_steps.round().int(), min=1)
            outer_steps = torch.clamp(outer_steps.round().int(), min=1)

            # Store with hybrid approach using adaptive parameters
            self._store_hybrid_adaptive(
                keys, values, context, inner_steps, outer_steps, thresholds
            )
        else:
            # Use fixed parameters with hybrid approach
            self._store_hybrid_fixed(
                keys, values, context, default_inner_steps, default_outer_steps
            )

    def _store_hybrid_fixed(
        self,
        keys: Tensor,
        values: Tensor,
        context: Tensor | None,
        inner_steps: int,
        outer_steps: int,
    ):
        """Store episodes using fixed hybrid approach."""
        num_episodes = keys.shape[0]

        # Pre-allocate all slots
        episode_slots = []
        for i in range(num_episodes):
            idx = self.episodic_memory.write_pointer % self.episodic_memory.memory_size
            episode_slots.append(idx)
            self.episodic_memory.used_slots[idx] = True
            self.episodic_memory.mem_timestamps[idx] = (
                self.episodic_memory.current_time + i
            )
            self.episodic_memory.write_pointer += 1

        # Hybrid learning: outer_steps rounds of all episodes
        for outer_round in range(outer_steps):
            for i in range(num_episodes):
                for inner_round in range(inner_steps):
                    self.episodic_memory.optimizer.zero_grad()
                    ctx = context[i : i + 1] if context is not None else None
                    retrieved = self.episodic_memory.forward(keys[i : i + 1], ctx)
                    loss = F.mse_loss(retrieved, values[i : i + 1])
                    loss.backward()
                    self.episodic_memory.optimizer.step()

        self.episodic_memory.current_time += num_episodes

    def _store_hybrid_adaptive(
        self,
        keys: Tensor,
        values: Tensor,
        context: Tensor | None,
        inner_steps: Tensor,
        outer_steps: Tensor,
        thresholds: Tensor,
    ):
        """Store episodes using adaptive hybrid approach with early stopping."""
        num_episodes = keys.shape[0]

        # Pre-allocate all slots
        episode_slots = []
        for i in range(num_episodes):
            idx = self.episodic_memory.write_pointer % self.episodic_memory.memory_size
            episode_slots.append(idx)
            self.episodic_memory.used_slots[idx] = True
            self.episodic_memory.mem_timestamps[idx] = (
                self.episodic_memory.current_time + i
            )
            self.episodic_memory.write_pointer += 1

        # Adaptive hybrid learning
        max_outer_steps = int(outer_steps.max().item())
        max_inner_steps = int(inner_steps.max().item())

        for outer_round in range(max_outer_steps):
            for i in range(num_episodes):
                # Skip if this episode has completed its outer steps
                if outer_round >= int(outer_steps[i].item()):
                    continue

                episode_converged = False
                for inner_round in range(max_inner_steps):
                    # Skip if this episode has completed its inner steps
                    if inner_round >= int(inner_steps[i].item()):
                        break

                    self.episodic_memory.optimizer.zero_grad()
                    ctx = context[i : i + 1] if context is not None else None
                    retrieved = self.episodic_memory.forward(keys[i : i + 1], ctx)
                    loss = F.mse_loss(retrieved, values[i : i + 1])

                    # Early stopping check
                    if loss.item() < thresholds[i].item():
                        episode_converged = True
                        break

                    loss.backward()
                    self.episodic_memory.optimizer.step()

                # If episode converged, skip remaining outer rounds for this episode
                if episode_converged:
                    continue

        self.episodic_memory.current_time += num_episodes


# Convenience function to create adaptive memory
def create_adaptive_episodic_memory(
    dim: int = 32, memory_size: int = 100, context_dim: int = 16, lr: float = 0.01
) -> AdaptiveEpisodicMemory:
    """Create an adaptive episodic memory system.

    Args:
        dim: Feature dimension
        memory_size: Number of memory slots
        context_dim: Context vector dimension
        lr: Learning rate

    Returns:
        AdaptiveEpisodicMemory instance

    """
    return AdaptiveEpisodicMemory(
        dim=dim, memory_size=memory_size, context_dim=context_dim, lr=lr
    )


# --------------------------------------------------------------------------------
def test_episodic_temporal_interference(
    dim: int = 32, context_dim: int = 16, memory_size: int = 20
) -> dict:
    """Test temporal interference in episodic memory with visualization."""
    memory = TrueEpisodicMemory(
        dim=dim, context_dim=context_dim, memory_size=memory_size, lr=0.1
    )

    print("=== Testing Episodic Temporal Interference ===")

    # Phase 1: Store "morning episodes"
    print("Phase 1: Storing morning episodes...")
    num_episodes = 5  # not the batch size
    morning_context = torch.randn(num_episodes, context_dim)
    morning_keys = torch.randn(num_episodes, dim)
    morning_values = torch.randn(num_episodes, dim)

    memory.store_episode(
        morning_keys, morning_values, morning_context, inner_steps=1, outer_steps=1
    )

    # Test immediate recall
    morning_recall_immediate = []
    with torch.no_grad():
        for i in range(num_episodes):
            retrieved = memory(morning_keys[i : i + 1], morning_context[i : i + 1])
            similarity = F.cosine_similarity(
                retrieved, morning_values[i : i + 1]
            ).item()
            morning_recall_immediate.append(similarity)

    avg_morning_immediate = np.mean(morning_recall_immediate)
    print(f"Morning episodes immediate recall: {avg_morning_immediate:.3f}")

    # Phase 2: Store interfering afternoon episodes
    print("Phase 2: Storing interfering afternoon episodes...")
    num_interferring_episodes = 15
    afternoon_context = torch.randn(num_interferring_episodes, context_dim)
    # Make afternoon keys similar to morning keys (interference)
    afternoon_keys = morning_keys[:3].repeat(num_episodes, 1) + 0.2 * torch.randn(
        num_interferring_episodes, dim
    )
    afternoon_values = torch.randn(num_interferring_episodes, dim)

    memory.store_episode(
        afternoon_keys,
        afternoon_values,
        afternoon_context,
        inner_steps=3,
        outer_steps=2,
    )

    # Phase 3: Test delayed recall
    print("Phase 3: Testing morning recall after interference...")
    morning_recall_delayed = []
    with torch.no_grad():
        for i in range(num_episodes):
            retrieved = memory(morning_keys[i : i + 1], morning_context[i : i + 1])
            similarity = F.cosine_similarity(
                retrieved, morning_values[i : i + 1]
            ).item()
            morning_recall_delayed.append(similarity)

    avg_morning_delayed = np.mean(morning_recall_delayed)
    print(f"Morning episodes delayed recall: {avg_morning_delayed:.3f}")

    interference_effect = avg_morning_immediate - avg_morning_delayed
    print(f"Temporal interference effect: {interference_effect:.3f}")

    # Phase 4: Context-dependent retrieval
    print("Phase 4: Testing context-dependent retrieval...")
    test_key = morning_keys[0:1]

    with torch.no_grad():
        # Retrieve with morning context
        morning_retrieved = memory(test_key, morning_context[0:1])
        morning_sim = F.cosine_similarity(morning_retrieved, morning_values[0:1]).item()

        # Retrieve with afternoon context
        afternoon_retrieved = memory(test_key, afternoon_context[0:1])
        afternoon_sim = F.cosine_similarity(
            afternoon_retrieved, afternoon_values[0:1]
        ).item()

    print(f"Same key, morning context similarity: {morning_sim:.3f}")
    print(f"Same key, afternoon context similarity: {afternoon_sim:.3f}")

    context_selectivity = abs(morning_sim - afternoon_sim)
    print(f"Context selectivity: {context_selectivity:.3f}")

    # Visualization
    plt.figure(figsize=(12, 4))

    # Plot 1: Recall over time
    plt.subplot(1, 3, 1)
    episodes = ["Morning\nImmediate", "Morning\nDelayed"]
    recalls = [avg_morning_immediate, avg_morning_delayed]
    bars = plt.bar(episodes, recalls, color=["lightblue", "lightcoral"])
    plt.ylabel("Cosine Similarity")
    plt.title("Temporal Interference Effect")
    plt.ylim(0, 1)
    for bar, val in zip(bars, recalls):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
        )

    # Plot 2: Individual episode recall
    plt.subplot(1, 3, 2)
    x = range(5)
    plt.plot(x, morning_recall_immediate, "o-", label="Immediate", color="blue")
    plt.plot(x, morning_recall_delayed, "s-", label="Delayed", color="red")
    plt.xlabel("Episode Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Per-Episode Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Context effect
    plt.subplot(1, 3, 3)
    contexts = ["Morning\nContext", "Afternoon\nContext"]
    similarities = [morning_sim, afternoon_sim]
    bars = plt.bar(contexts, similarities, color=["gold", "purple"])
    plt.ylabel("Cosine Similarity")
    plt.title("Context-Dependent Retrieval")
    for bar, val in zip(bars, similarities):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig("episodic_interference_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    results = {
        "morning_immediate_recall": avg_morning_immediate,
        "morning_delayed_recall": avg_morning_delayed,
        "interference_effect": interference_effect,
        "context_selectivity": context_selectivity,
    }

    print("✓ Episodic temporal interference test completed")
    return results


def test_episodic_sequence_recall(
    dim: int = 32,
    context_dim: int = 16,
    sequence_length: int = 6,
) -> dict:
    """Test sequential episodic recall with visualization."""
    memory = TrueEpisodicMemory(
        dim=dim, context_dim=context_dim, memory_size=50, lr=0.1
    )

    print("=== Testing Episodic Sequence Recall ===")

    # Create a coherent sequence
    story_context = torch.randn(1, context_dim).repeat(sequence_length, 1)

    # Create sequence with gradual drift
    sequence_keys = []
    sequence_values = []
    base_key = torch.randn(dim)
    base_value = torch.randn(dim)

    for i in range(sequence_length):
        # Each episode drifts from the base
        key = base_key + 0.1 * i * torch.randn(dim)
        value = base_value + 0.1 * i * torch.randn(dim)
        sequence_keys.append(key)
        sequence_values.append(value)

    sequence_keys = torch.stack(sequence_keys)
    sequence_values = torch.stack(sequence_values)

    print(f"Storing sequence of {sequence_length} episodes...")
    memory.store_episode(
        sequence_keys,
        sequence_values,
        story_context,
        inner_steps=0,
        outer_steps=0,
    )

    # Test sequential vs random recall
    print("Testing sequential vs random recall...")

    # Sequential recall
    sequential_accuracies = []
    with torch.no_grad():
        for i in range(sequence_length):
            retrieved = memory(sequence_keys[i : i + 1], story_context[i : i + 1])
            similarity = F.cosine_similarity(
                retrieved, sequence_values[i : i + 1]
            ).item()
            sequential_accuracies.append(similarity)

    # Random order recall
    random_indices = torch.randperm(sequence_length)
    random_accuracies = []
    with torch.no_grad():
        for idx in random_indices:
            retrieved = memory(
                sequence_keys[idx : idx + 1], story_context[idx : idx + 1]
            )
            similarity = F.cosine_similarity(
                retrieved, sequence_values[idx : idx + 1]
            ).item()
            random_accuracies.append(similarity)

    avg_sequential = np.mean(sequential_accuracies)
    avg_random = np.mean(random_accuracies)

    print(f"Sequential recall accuracy: {avg_sequential:.3f}")
    print(f"Random recall accuracy: {avg_random:.3f}")

    # Visualization
    plt.figure(figsize=(10, 4))

    # Plot 1: Sequential vs Random
    plt.subplot(1, 2, 1)
    x = range(sequence_length)
    plt.plot(x, sequential_accuracies, "o-", label="Sequential Order", color="green")
    plt.plot(
        x,
        [random_accuracies[i] for i in range(sequence_length)],
        "s-",
        label="Random Order",
        color="orange",
    )
    plt.xlabel("Episode Position")
    plt.ylabel("Cosine Similarity")
    plt.title("Sequential vs Random Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Average comparison
    plt.subplot(1, 2, 2)
    methods = ["Sequential", "Random"]
    averages = [avg_sequential, avg_random]
    bars = plt.bar(methods, averages, color=["green", "orange"])
    plt.ylabel("Average Cosine Similarity")
    plt.title("Recall Method Comparison")
    for bar, val in zip(bars, averages):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig("episodic_sequence_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    results = {
        "sequential_accuracy": avg_sequential,
        "random_accuracy": avg_random,
        "sequential_advantage": avg_sequential - avg_random,
    }

    print("✓ Episodic sequence recall test completed")
    return results


# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Set random seeds for repeatability
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=== Testing TrueEpisodicMemory Module ===\n")

    print("1. Testing Episodic Temporal Interference...")
    interference_results = test_episodic_temporal_interference()
    print(f"Results: {interference_results}\n")
    quit()

    print("2. Testing Episodic Sequence Recall...")
    sequence_results = test_episodic_sequence_recall()
    print(f"Results: {sequence_results}\n")

    print("=== All tests completed successfully! ===")
    quit()
    # --------------------------------------------------------------------------------
    # Create adaptive memory system
    memory = create_adaptive_episodic_memory(dim=32, context_dim=16)

    # Store episodes with adaptive learning
    morning_keys = torch.randn(5, 32)
    morning_values = torch.randn(5, 32)
    morning_context = torch.randn(5, 16)

    # Adaptive storage (automatically determines optimal learning parameters)
    memory.store_episode_adaptive(
        morning_keys, morning_values, morning_context, use_adaptive=True
    )

    # Or use fixed hybrid approach
    memory.store_episode_adaptive(
        morning_keys,
        morning_values,
        morning_context,
        use_adaptive=False,
        default_inner_steps=3,
        default_outer_steps=2,
    )

    # Retrieval works the same
    retrieved = memory(morning_keys[0:1], morning_context[0:1])
