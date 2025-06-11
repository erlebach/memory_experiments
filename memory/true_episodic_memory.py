import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrueEpisodicMemory(nn.Module):
    """Episodic memory with temporal and contextual binding."""

    def __init__(self, dim, memory_size=100, context_dim=16, lr=0.01):
        super().__init__()
        # Initialize with small random values to avoid perfect matches
        self.mem_keys = nn.Parameter(torch.randn(memory_size, dim) * 0.1)
        self.mem_values = nn.Parameter(torch.randn(memory_size, dim) * 0.1)
        self.mem_contexts = nn.Parameter(torch.randn(memory_size, context_dim) * 0.1)
        self.mem_timestamps = nn.Parameter(torch.zeros(memory_size, 1))

        self.dim = dim
        self.context_dim = context_dim
        self.memory_size = memory_size
        self.current_time = 0
        self.write_pointer = 0

        # Track which slots are actually used
        self.used_slots = torch.zeros(memory_size, dtype=torch.bool)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, query, context=None, time_weight=0.1):
        """Retrieve with temporal and contextual cues."""
        # Only consider used memory slots
        if self.used_slots.sum() == 0:
            # No memories stored yet, return random
            return torch.randn_like(query)

        # Mask for used slots
        used_mask = self.used_slots.float()

        # Standard content-based attention
        content_scores = torch.matmul(query, self.mem_keys.T)

        # Add contextual similarity if context provided
        if context is not None:
            context_scores = torch.matmul(context, self.mem_contexts.T)
            content_scores = content_scores + 0.5 * context_scores

        # Add temporal recency bias (more recent = higher weight)
        time_decay = torch.exp(
            -time_weight * (self.current_time - self.mem_timestamps.squeeze())
        )
        content_scores = content_scores + 0.3 * time_decay.unsqueeze(0)

        # Mask unused slots
        content_scores = content_scores * used_mask.unsqueeze(0)
        content_scores = content_scores + (1 - used_mask.unsqueeze(0)) * (
            -1e9
        )  # Large negative for unused

        attn_probs = F.softmax(content_scores, dim=-1)
        retrieved_value = torch.matmul(attn_probs, self.mem_values)
        return retrieved_value

    def store_episode(self, keys, values, context=None, update_steps=3):
        """Store new episode with learning-based updates."""
        batch_size = keys.shape[0]

        for i in range(batch_size):
            # Find slot to update (circular buffer)
            idx = self.write_pointer % self.memory_size
            self.used_slots[idx] = True

            # Learning-based storage (not just copying)
            with torch.enable_grad():
                for _ in range(update_steps):
                    self.optimizer.zero_grad()

                    # Try to retrieve what we want to store
                    ctx = context[i : i + 1] if context is not None else None
                    retrieved = self.forward(keys[i : i + 1], ctx)

                    # Loss: retrieved should match target value
                    loss = F.mse_loss(retrieved, values[i : i + 1])
                    loss.backward()
                    self.optimizer.step()

            # Update timestamp
            self.mem_timestamps.data[idx] = self.current_time
            self.write_pointer += 1
            self.current_time += 1


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
    morning_context = torch.randn(5, context_dim)
    morning_keys = torch.randn(5, dim)
    morning_values = torch.randn(5, dim)

    memory.store_episode(morning_keys, morning_values, morning_context, update_steps=5)

    # Test immediate recall
    morning_recall_immediate = []
    with torch.no_grad():
        for i in range(5):
            retrieved = memory(morning_keys[i : i + 1], morning_context[i : i + 1])
            similarity = F.cosine_similarity(
                retrieved, morning_values[i : i + 1]
            ).item()
            morning_recall_immediate.append(similarity)

    avg_morning_immediate = np.mean(morning_recall_immediate)
    print(f"Morning episodes immediate recall: {avg_morning_immediate:.3f}")

    # Phase 2: Store interfering afternoon episodes
    print("Phase 2: Storing interfering afternoon episodes...")
    afternoon_context = torch.randn(15, context_dim)
    # Make afternoon keys similar to morning keys (interference)
    afternoon_keys = morning_keys[:3].repeat(5, 1) + 0.2 * torch.randn(15, dim)
    afternoon_values = torch.randn(15, dim)

    memory.store_episode(
        afternoon_keys, afternoon_values, afternoon_context, update_steps=5
    )

    # Phase 3: Test delayed recall
    print("Phase 3: Testing morning recall after interference...")
    morning_recall_delayed = []
    with torch.no_grad():
        for i in range(5):
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
    dim: int = 32, context_dim: int = 16, sequence_length: int = 6
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
    memory.store_episode(sequence_keys, sequence_values, story_context, update_steps=5)

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


if __name__ == "__main__":
    print("=== Testing TrueEpisodicMemory Module ===\n")

    print("1. Testing Episodic Temporal Interference...")
    interference_results = test_episodic_temporal_interference()
    print(f"Results: {interference_results}\n")

    print("2. Testing Episodic Sequence Recall...")
    sequence_results = test_episodic_sequence_recall()
    print(f"Results: {sequence_results}\n")

    print("=== All tests completed successfully! ===")
