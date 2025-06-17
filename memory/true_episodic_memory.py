"""Experiments to test TrueEpisodicMemory."""

import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import einsum
from jaxtyping import Float
from torch import Tensor


@beartype
class TrueEpisodicMemory(nn.Module):
    """Episodic memory with temporal and contextual binding."""

    def __init__(
        self,
        dim: int,
        memory_size: int = 100,
        context_dim: int = 16,
        lr: float = 0.01,
        orthogonality_weight: float = 0.01,
    ) -> None:
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
        self.orthogonality_weight = orthogonality_weight

    def forward(
        self,
        query: Float[Tensor, "batch query_dim"],
        context: Float[Tensor, "batch context_dim"] | None = None,
        time_weight: float = 0.1,
    ) -> Float[Tensor, "batch dim"]:
        """Retrieve with temporal and contextual cues."""
        if self.used_slots.sum() == 0:
            return torch.randn_like(query)

        # Initialize all scores to -1e9 (will become ~0 in softmax)
        content_scores = torch.full(
            (query.shape[0], self.memory_size), -1e9, device=query.device
        )

        # Only compute scores for used slots
        used_mask = self.used_slots
        if used_mask.sum() > 0:
            # Content-based attention for used slots only
            used_content_scores = einsum(
                query,
                self.mem_keys[used_mask],
                "batch query_dim, used_slots query_dim -> batch used_slots",
            )

            # Add contextual similarity if provided
            # This should be parallelized across all memories
            if context is not None:
                used_context_scores = einsum(
                    context,
                    self.mem_contexts[used_mask],
                    "batch context_dim, used_slots context_dim -> batch used_slots",
                )
                used_content_scores += 0.5 * used_context_scores

            # Add temporal recency bias for used slots
            used_time_decay = torch.exp(
                -time_weight
                * (self.mem_timestamps[used_mask].squeeze() - self.current_time)
            )
            # print(f"{used_time_decay=}")
            used_time_decay = torch.clamp(used_time_decay, min=1e-8, max=1.0)
            used_content_scores += 0.3 * used_time_decay.unsqueeze(0)

            # Assign computed scores to used positions
            content_scores[:, used_mask] = used_content_scores

        attn_probs = F.softmax(content_scores, dim=-1)
        retrieved_value = einsum(
            attn_probs,
            self.mem_values,
            "batch memory_size, memory_size dim -> batch dim",
        )
        return retrieved_value

    def store_episode(
        self,
        keys: Float[Tensor, "num_episodes key_dim"],
        values: Float[Tensor, "num_episodes value_dim"],
        context: Float[Tensor, "num_episodes context_dim"] | None = None,
        inner_steps: int = 0,  # Learning rate compensation per episode
        outer_steps: int = 0,  # Spaced repetition rounds
        timestamps: Float[Tensor, "num_episodes 1"] | None = None,
    ) -> None:
        """Store new episode with learning-based updates."""
        num_episodes = keys.shape[0]

        # Pre-allocate all slots
        episode_slots = []
        for i in range(num_episodes):
            idx = self.write_pointer % self.memory_size
            episode_slots.append(idx)
            self.used_slots[idx] = True
            # Use provided timestamp or current_time + i if not provided
            if timestamps is not None:
                self.mem_timestamps[idx] = timestamps[i]
            else:
                self.mem_timestamps[idx] = self.current_time + i
            self.write_pointer += 1

        # TRUE HYBRID: Outer loop (spaced repetition) × Inner loop (learning rate compensation)
        for _ in range(outer_steps):  # Spaced repetition rounds
            for i in range(num_episodes):  # Cycle through all episodes
                for _ in range(inner_steps):  # Multiple updates per episode
                    self.optimizer.zero_grad()
                    ctx = context[i : i + 1] if context is not None else None
                    retrieved = self.forward(keys[i : i + 1], ctx)
                    loss = F.mse_loss(retrieved, values[i : i + 1])
                    # Orthogonality regularization for memory VALUES
                    # Only consider used slots for orthogonality to avoid penalizing empty slots
                    used_mem_values = self.mem_values[self.used_slots]
                    if used_mem_values.shape[0] > 1:
                        # Calculate Gram matrix (dot products between used values)
                        gram_matrix = einsum(
                            used_mem_values, used_mem_values, "n d, m d -> n m"
                        )
                        # Penalize deviation from identity matrix (encourages orthogonality and unit norm)
                        identity_matrix = torch.eye(
                            used_mem_values.shape[0], device=used_mem_values.device
                        )
                        ortho_loss = torch.norm(gram_matrix - identity_matrix) ** 2
                    loss += self.orthogonality_weight * ortho_loss
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


# --------------------------------------------------------------------------------
class AdaptiveEpisodicMemory(nn.Module):
    """Episodic memory with adaptive learning based on content type."""

    def __init__(
        self,
        dim: int,
        memory_size: int = 100,
        context_dim: int = 16,
        lr: float = 0.01,
        orthogonality_weight: float = 0.01,
    ):
        """Initialize adaptive episodic memory.

        Args:
            dim: Feature dimension
            memory_size: Number of memory slots
            context_dim: Context vector dimension
            lr: Learning rate
            orthogonality_weight: Weight for orthogonality regularization

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

        self.orthogonality_weight = orthogonality_weight

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

                    # Orthogonality regularization for memory VALUES
                    # Only consider used slots for orthogonality to avoid penalizing empty slots
                    used_mem_values = self.episodic_memory.mem_values[
                        self.episodic_memory.used_slots
                    ]
                    if used_mem_values.shape[0] > 1:
                        # Calculate Gram matrix (dot products between used values)
                        gram_matrix = einsum(
                            used_mem_values,
                            used_mem_values,
                            "n d, m d -> n m",
                        )
                        # Penalize deviation from identity matrix (encourages orthogonality and unit norm)
                        identity_matrix = torch.eye(
                            used_mem_values.shape[0], device=used_mem_values.device
                        )
                        ortho_loss = torch.norm(gram_matrix - identity_matrix) ** 2
                        loss = loss + self.orthogonality_weight * ortho_loss

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
if __name__ == "__main__":
    # Set random seeds for repeatability
    # seed = 42
    seed = 52
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    # Test fine-grained timestamps + spaced repetition
    def test_optimal_parameters():
        # Fine timestamps
        timestamps = torch.arange(sequence_length) * 0.1

        # Spaced repetition learning
        inner_steps = 1
        outer_steps = 50

        # Test multiple configurations
        configs = [
            {"inner": 1, "outer": 50, "time_inc": 0.1},
            {"inner": 1, "outer": 100, "time_inc": 0.05},
            {"inner": 2, "outer": 25, "time_inc": 0.1},
        ]
