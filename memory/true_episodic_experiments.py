import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from beartype import beartype
from jaxtyping import Float, Integer
from torch import Tensor

from memory.true_episodic_memory import TrueEpisodicMemory


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

    # Variables needed for plotting
    plot_dict = {
        "avg_morning_immediate": avg_morning_immediate,
        "avg_morning_delayed": avg_morning_delayed,
        "morning_sim": morning_sim,
        "afternoon_sim": afternoon_sim,
        "morning_recall_immediate": morning_recall_immediate,
        "morning_recall_delayed": morning_recall_delayed,
    }

    plot_episodic_interference_results(plot_dict)

    results = {
        "morning_immediate_recall": avg_morning_immediate,
        "morning_delayed_recall": avg_morning_delayed,
        "interference_effect": interference_effect,
        "context_selectivity": context_selectivity,
    }

    print("✓ Episodic temporal interference test completed")
    return results


def plot_episodic_interference_results(plot_dict: dict) -> None:
    """Plot the results of the episodic temporal interference test."""
    # Visualization
    plt.figure(figsize=(12, 4))

    # Plot 1: Recall over time
    plt.subplot(1, 3, 1)
    episodes = ["Morning\nImmediate", "Morning\nDelayed"]
    recalls = [plot_dict["avg_morning_immediate"], plot_dict["avg_morning_delayed"]]
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
    plt.plot(
        x, plot_dict["morning_recall_immediate"], "o-", label="Immediate", color="blue"
    )
    plt.plot(x, plot_dict["morning_recall_delayed"], "s-", label="Delayed", color="red")
    plt.xlabel("Episode Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Per-Episode Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Context effect
    plt.subplot(1, 3, 3)
    contexts = ["Morning\nContext", "Afternoon\nContext"]
    similarities = [plot_dict["morning_sim"], plot_dict["afternoon_sim"]]
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


# --------------------------------------------------------------------------------
def test_episodic_sequence_recall(
    dim: int = 32,
    context_dim: int = 16,
    sequence_length: int = 30,
    delta_time: float = 0.0,
    inner_steps: int = 1,
    outer_steps: int = 50,
) -> dict:
    """Evaluate how well the episodic memory system recalls sequences of related episodes.

    This function tests whether the episodic memory system maintains coherent recall
    for sequences of related episodes, comparing sequential vs. random-order retrieval
    performance.

    """
    print("=== Testing Episodic Sequence Recall ===")

    # Create a coherent sequence (shared data for both tests)
    story_context = torch.randn(1, context_dim).repeat(sequence_length, 1)

    # Create sequence with gradual drift
    sequence_keys = []
    sequence_values = []
    base_key = torch.randn(dim)
    base_value = torch.randn(dim)

    for i in range(sequence_length):
        # Each episode drifts from the base
        # key = base_key + 0.05 * i * torch.randn(dim)
        # value = base_value + 0.05 * i * torch.randn(dim)
        key = base_key + 0.1 * i * torch.ones(dim) / math.sqrt(dim)  # Systematic
        value = base_value + 0.1 * i * torch.ones(dim) / math.sqrt(dim)
        sequence_keys.append(key)
        sequence_values.append(value)

    sequence_keys = torch.stack(sequence_keys)
    sequence_values = torch.stack(sequence_values)

    # TEST 1: Sequential case with fresh memory
    print("Testing sequential recall...")
    memory_sequential = TrueEpisodicMemory(
        dim=dim,
        context_dim=context_dim,
        memory_size=50,
        lr=0.1,
        orthogonality_weight=0.1,
    )

    sequential_timestamps = delta_time * torch.arange(
        sequence_length
    ).float().unsqueeze(1)
    memory_sequential.store_episode(
        sequence_keys,
        sequence_values,
        story_context,
        timestamps=sequential_timestamps,
        inner_steps=inner_steps,
        outer_steps=outer_steps,
    )

    # Sequential recall
    sequential_accuracies = []
    with torch.no_grad():
        for i in range(sequence_length):
            retrieved = memory_sequential(
                sequence_keys[i : i + 1],
                story_context[i : i + 1],
            )
            similarity = F.cosine_similarity(
                retrieved, sequence_values[i : i + 1]
            ).item()
            sequential_accuracies.append(similarity)

    # TEST 2: Random case with fresh memory
    print("Testing random recall...")
    memory_random = TrueEpisodicMemory(
        dim=dim,
        context_dim=context_dim,
        memory_size=50,
        lr=0.1,
        orthogonality_weight=0.1,
    )

    # Create random permutation
    random_indices = torch.randperm(sequence_length)
    random_keys = sequence_keys[random_indices]
    random_values = sequence_values[random_indices]
    random_context = story_context[random_indices]
    random_timestamps = sequential_timestamps[random_indices]
    print(f"{random_timestamps=}")

    memory_random.store_episode(
        random_keys,
        random_values,
        random_context,
        timestamps=random_timestamps,
        inner_steps=inner_steps,
        outer_steps=outer_steps,
    )

    # Random order recall (retrieve in same order as stored)
    random_accuracies = []
    with torch.no_grad():
        for i in range(sequence_length):
            retrieved = memory_random(
                random_keys[i : i + 1],
                random_context[i : i + 1],
            )
            similarity = F.cosine_similarity(
                retrieved,
                random_values[i : i + 1],
            ).item()
            random_accuracies.append(similarity)

    avg_sequential = np.mean(sequential_accuracies)
    avg_random = np.mean(random_accuracies)

    print(f"Sequential recall accuracy: {avg_sequential:.3f}")
    print(f"Random recall accuracy: {avg_random:.3f}")

    # Visualization
    plot_dict = {
        "sequential_accuracies": sequential_accuracies,
        "random_accuracies": random_accuracies,
        "avg_sequential": avg_sequential,
        "avg_random": avg_random,
        "sequence_length": sequence_length,
    }

    plot_episodic_sequence_recall_results(plot_dict)

    results = {
        "sequential_accuracy": avg_sequential,
        "random_accuracy": avg_random,
        "sequential_advantage": avg_sequential - avg_random,
    }

    print("✓ Episodic sequence recall test completed")
    return results


def plot_episodic_sequence_recall_results(plot_dict: dict) -> None:
    """Plot the results of the episodic sequence recall test."""
    plt.figure(figsize=(10, 4))

    # Plot 1: Sequential vs Random
    plt.subplot(1, 2, 1)
    x = range(plot_dict["sequence_length"])
    plt.plot(
        x,
        plot_dict["sequential_accuracies"],
        "o-",
        label="Sequential Order",
        color="green",
    )
    plt.plot(
        x,
        [
            plot_dict["random_accuracies"][i]
            for i in range(plot_dict["sequence_length"])
        ],
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
    averages = [plot_dict["avg_sequential"], plot_dict["avg_random"]]
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


def compute_orthogonality_matrix(self) -> torch.Tensor:
    """Compute the orthogonality matrix between all episodes.

    Returns:
        A tensor of shape (n_episodes, n_episodes) containing the cosine similarity
        between all pairs of episodes.
    """
    return self._compute_orthogonality_matrix()


# --------------------------------------------------------------------------------
def _compute_orthogonality_matrix(self) -> torch.Tensor:
    """Compute the orthogonality matrix between all episodes.

    Returns:
        A tensor of shape (n_episodes, n_episodes) containing the cosine similarity
        between all pairs of episodes.
    """
    n_episodes = len(self.episodes)
    if n_episodes == 0:
        return torch.empty((0, 0))

    # Extract episode vectors
    episode_vectors = torch.stack([ep["episode"] for ep in self.episodes])

    # Normalize vectors to unit length
    normalized_vectors = F.normalize(episode_vectors, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(normalized_vectors, normalized_vectors.t())

    # Apply orthogonalization constraint
    # This ensures that new memories are as orthogonal as possible to existing ones
    for i in range(n_episodes):
        for j in range(i + 1, n_episodes):
            # Compute the projection of vector j onto vector i
            proj = torch.dot(normalized_vectors[i], normalized_vectors[j])
            # Subtract the projection to make vectors more orthogonal
            normalized_vectors[j] -= proj * normalized_vectors[i]
            # Renormalize
            normalized_vectors[j] = F.normalize(normalized_vectors[j], p=2, dim=0)

    # Recompute similarity matrix after orthogonalization
    similarity_matrix = torch.mm(normalized_vectors, normalized_vectors.t())

    return similarity_matrix


def test_multi_scale_temporal_retrieval():
    """Test retrieval at different temporal scales with TrueEpisodicMemory."""
    print("\n=== Testing Multi-Scale Temporal Retrieval ===")

    memory = TrueEpisodicMemory(dim=32, context_dim=16, memory_size=50, lr=0.1)

    # Create multi-scale temporal episodes with BETTER temporal spacing
    keys = []
    values = []
    contexts = []
    timestamps = []
    episode_labels = []

    # Use more realistic temporal spacing (hours)
    all_episodes = [
        ("Wake up", 0.1),
        ("Shower", 0.5),
        ("Breakfast", 1.0),
        ("Commute", 2.0),
        ("Physics lecture", 4.0),
        ("Take notes", 4.5),
        ("Ask question", 5.0),
        ("Library study", 6.0),
        ("Group discussion", 7.0),
        ("Dinner", 8.0),
        ("Friends", 9.0),
        ("Relax", 10.0),
        ("Sleep prep", 11.0),
    ]

    # Create episode vectors with systematic relationships
    base_key = torch.randn(32)
    base_value = torch.randn(32)
    base_context = torch.randn(16)

    # First, generate all initial values
    initial_values = []
    for i, (label, time) in enumerate(all_episodes):
        time_factor = time / 12.0
        value = base_value + 0.3 * time_factor * torch.ones(32) / math.sqrt(32)
        initial_values.append(value)

    # Orthogonalize all values
    values = orthogonalize_vectors(initial_values)

    # Now use the orthogonalized values in the main loop
    for i, (label, time) in enumerate(all_episodes):
        time_factor = time / 12.0

        # Keys drift systematically with time (unchanged)
        key = base_key + 0.3 * time_factor * torch.ones(32) / math.sqrt(32)

        # Context varies by time period (unchanged)
        if time < 4:  # Morning
            context = base_context + 0.1 * torch.ones(16) / math.sqrt(16)
        elif time < 8:  # Afternoon
            context = base_context + 0.2 * torch.ones(16) / math.sqrt(16)
        else:  # Evening
            context = base_context + 0.3 * torch.ones(16) / math.sqrt(16)

        keys.append(key)
        contexts.append(context)
        timestamps.append(time)
        episode_labels.append(label)

    # Convert to tensors
    keys_tensor = torch.stack(keys)
    values_tensor = torch.stack(values)
    contexts_tensor = torch.stack(contexts)
    timestamps_tensor = torch.tensor(timestamps).unsqueeze(1)

    # Create an matrix measuring the orthogonality of each value against all other values
    orthogonality_matrix = torch.zeros(len(values), len(values))
    for i, value in enumerate(values):
        for j, other_value in enumerate(values):
            orthogonality_matrix[i, j] = F.cosine_similarity(
                value, other_value, dim=0
            ).item()
    print(f"Orthogonality matrix:\n{orthogonality_matrix}")

    # Store all episodes with minimal learning to preserve temporal structure
    memory.store_episode(
        keys_tensor,
        values_tensor,
        contexts_tensor,
        timestamps=timestamps_tensor,
        inner_steps=1,
        outer_steps=1,  # Minimal learning
    )

    # CRITICAL: Set current_time to a reasonable query time (end of day)
    memory.current_time = 12.0

    # Test queries at different temporal scales
    test_queries = [
        (
            "Specific moment",
            4.5,
            "Taking notes in physics",
            5,
        ),  # Should match "Take notes"
        (
            "Lecture period",
            4.75,
            "Physics class activities",
            4,
        ),  # Should match afternoon
        ("Afternoon block", 5.5, "Academic activities", 6),  # Should match afternoon
        ("Entire day", 6.0, "University experience", 6),  # Should match multiple
    ]

    results = {}

    for i, (query_name, query_time, description, expected_episode_idx) in enumerate(
        test_queries
    ):
        # Create query that's similar to expected episode
        expected_key = keys[expected_episode_idx]
        expected_context = contexts[expected_episode_idx]

        # Add some noise but keep similarity
        query_key = (expected_key + 0.1 * torch.randn(32)).unsqueeze(0)
        query_context = (expected_context + 0.1 * torch.randn(16)).unsqueeze(0)

        # Test different time weights to simulate different temporal scales
        time_weights = [0.0, 0.1, 0.5, 1.0]  # Different temporal sensitivities
        scale_names = [
            "No temporal bias",
            "Weak temporal",
            "Medium temporal",
            "Strong temporal",
        ]

        similarities = []
        temporal_spans = []

        for time_weight in time_weights:
            # Retrieve with different temporal sensitivities
            with torch.no_grad():
                retrieved = memory(query_key, query_context, time_weight=time_weight)

            # Calculate similarities to all stored episodes
            episode_similarities = []
            for j, (key, value) in enumerate(zip(keys, values)):
                key_sim = F.cosine_similarity(query_key.squeeze(), key, dim=0).item()
                value_sim = F.cosine_similarity(
                    retrieved.squeeze(), value, dim=0
                ).item()
                combined_sim = 0.5 * key_sim + 0.5 * value_sim
                episode_similarities.append(
                    (j, key_sim, value_sim, combined_sim, timestamps[j])
                )

            # Find episodes above similarity threshold
            threshold = 0.3
            relevant_episodes = [ep for ep in episode_similarities if ep[3] > threshold]

            if relevant_episodes:
                avg_similarity = np.mean([ep[3] for ep in relevant_episodes])
                times = [ep[4] for ep in relevant_episodes]
                temporal_span = max(times) - min(times) if len(times) > 1 else 0
                num_relevant = len(relevant_episodes)

                print(f"Query: {query_name}, Time weight: {time_weight:.1f}")
                print(
                    f"  Relevant episodes: {num_relevant}, Avg similarity: {avg_similarity:.3f}, Span: {temporal_span:.1f}"
                )
                print(f"  Episode indices: {[ep[0] for ep in relevant_episodes]}")
                print(f"  Episode times: {[ep[4] for ep in relevant_episodes]}")
            else:
                avg_similarity = 0
                temporal_span = 0
                print(
                    f"Query: {query_name}, Time weight: {time_weight:.1f} - No relevant episodes found"
                )

            similarities.append(avg_similarity)
            temporal_spans.append(temporal_span)

        plot_dict = {
            "similarities": similarities,
            "temporal_spans": temporal_spans,
            "query_time": query_time,
            "query_name": query_name,
            "description": description,
            "scale_names": scale_names,
        }

        plot_test_multi_scale_temporal_retrieval(plot_dict, i)

    plt.tight_layout()
    plt.savefig("multi_scale_temporal_retrieval.png", dpi=150, bbox_inches="tight")
    plt.show()

    results[query_name] = {
        "similarities": similarities,
        "temporal_spans": temporal_spans,
        "query_time": query_time,
    }

    print("✓ Multi-scale temporal retrieval test completed")
    return results


def plot_test_multi_scale_temporal_retrieval(plot_dict: dict, i: int) -> None:
    """Plot the results of the multi-scale temporal retrieval test."""

    similarities = plot_dict["similarities"]
    temporal_spans = plot_dict["temporal_spans"]
    query_time = plot_dict["query_time"]
    query_name = plot_dict["query_name"]
    description = plot_dict["description"]
    scale_names = plot_dict["scale_names"]

    if i == 0:
        plt.figure(figsize=(12, 8))

    # Plot results
    plt.subplot(2, 2, i + 1)
    ax1 = plt.gca()
    bars1 = ax1.bar(
        [x - 0.2 for x in range(4)],
        temporal_spans,
        0.4,
        label="Temporal Span",
        color="skyblue",
    )
    ax1.set_ylabel("Temporal Span (hours)", color="blue")
    ax1.set_ylim(0, max(temporal_spans) * 1.2 if max(temporal_spans) > 0 else 1)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(
        [x + 0.2 for x in range(4)],
        similarities,
        0.4,
        label="Avg Similarity",
        color="lightcoral",
    )
    ax2.set_ylabel("Average Similarity", color="red")
    ax2.set_ylim(0, max(similarities) * 1.2 if max(similarities) > 0 else 1)

    plt.title(f'{query_name}\n"{description}"')
    ax1.set_xlabel("Temporal Scale")
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(scale_names, rotation=45, ha="right")

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    return None


def test_multi_scale_temporal_retrieval_debug():
    """Debug version to understand what's happening."""
    print("\n=== Testing Multi-Scale Temporal Retrieval (Debug) ===")

    memory = TrueEpisodicMemory(dim=32, context_dim=16, memory_size=50, lr=0.1)

    # Create episodes with REASONABLE time differences (not exponential)
    timestamps = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]  # Linear spacing in hours
    episode_labels = [f"Episode_{i}" for i in range(len(timestamps))]

    # Create more distinct episodes
    keys = []
    values = []
    contexts = []

    base_key = torch.randn(32)
    base_value = torch.randn(32)
    base_context = torch.randn(16)

    for i, time in enumerate(timestamps):
        # Make episodes more distinct but still related
        key = base_key + 0.3 * i * torch.randn(32) / math.sqrt(32)
        value = base_value + 0.3 * i * torch.randn(32) / math.sqrt(32)
        context = base_context + 0.2 * i * torch.randn(16) / math.sqrt(16)

        keys.append(key)
        values.append(value)
        contexts.append(context)

    # Convert to tensors
    keys_tensor = torch.stack(keys)
    values_tensor = torch.stack(values)
    contexts_tensor = torch.stack(contexts)
    timestamps_tensor = torch.tensor(timestamps, dtype=torch.float).unsqueeze(1)

    # Store with MINIMAL learning to preserve temporal effects
    memory.store_episode(
        keys_tensor,
        values_tensor,
        contexts_tensor,
        timestamps=timestamps_tensor,
        inner_steps=0,  # No learning
        outer_steps=0,
    )

    # Set current_time to a reasonable query time
    memory.current_time = max(timestamps) + 2  # 18 hours

    print(f"Current time: {memory.current_time}")
    print(f"Stored timestamps: {timestamps}")
    print(f"Time differences: {[memory.current_time - t for t in timestamps]}")

    # Test one specific query
    query_key = (keys[5] + 0.1 * torch.randn(32)).unsqueeze(0)  # Similar to episode 5
    query_context = (contexts[5] + 0.1 * torch.randn(16)).unsqueeze(0)

    print("\n=== Testing Different Time Weights ===")

    for time_weight in [0.0, 0.01, 0.1, 1.0]:
        print(f"\nTime weight: {time_weight}")

        with torch.no_grad():
            retrieved = memory(query_key, query_context, time_weight=time_weight)

        # Calculate time decay values manually (CORRECTED FORMULA)
        time_decays = torch.exp(-time_weight * torch.tensor(timestamps))

        print(f"Time decays: {time_decays.numpy()}")

        # Calculate similarities
        similarities = []
        for i, (key, value) in enumerate(zip(keys, values)):
            key_sim = F.cosine_similarity(query_key.squeeze(), key, dim=0).item()
            value_sim = F.cosine_similarity(retrieved.squeeze(), value, dim=0).item()
            similarities.append((i, key_sim, value_sim, timestamps[i]))

        print("Episode similarities (idx, key_sim, value_sim, timestamp):")
        for sim in similarities:
            print(f"  {sim}")

        # Find best matches
        best_key_match = max(similarities, key=lambda x: x[1])
        best_value_match = max(similarities, key=lambda x: x[2])

        print(
            f"Best key match: Episode {best_key_match[0]} (sim={best_key_match[1]:.3f})"
        )
        print(
            f"Best value match: Episode {best_value_match[0]} (sim={best_value_match[2]:.3f})"
        )

    return None


def test_temporal_granularity_effects():
    """Test how temporal granularity affects retrieval."""

    granularities = [
        ("Fine", [0.1, 0.2, 0.3, 0.4, 0.5]),  # 0.1 hour differences
        ("Medium", [1, 2, 3, 4, 5]),  # 1 hour differences
        ("Coarse", [10, 20, 30, 40, 50]),  # 10 hour differences
        ("Very Coarse", [100, 200, 300, 400, 500]),  # 100 hour differences
    ]

    results = {}

    for gran_name, timestamps in granularities:
        print(f"\n=== Testing {gran_name} Granularity ===")

        memory = TrueEpisodicMemory(dim=32, context_dim=16, memory_size=50, lr=0.1)

        # Create episodes
        keys = [torch.randn(32) for _ in timestamps]
        values = [torch.randn(32) for _ in timestamps]
        contexts = [torch.randn(16) for _ in timestamps]

        keys_tensor = torch.stack(keys)
        values_tensor = torch.stack(values)
        contexts_tensor = torch.stack(contexts)
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float).unsqueeze(1)

        # Minimal learning
        memory.store_episode(
            keys_tensor,
            values_tensor,
            contexts_tensor,
            timestamps=timestamps_tensor,
            inner_steps=0,
            outer_steps=0,
        )

        memory.current_time = max(timestamps) + max(timestamps) * 0.1

        # Test query similar to middle episode
        mid_idx = len(keys) // 2
        query_key = (keys[mid_idx] + 0.1 * torch.randn(32)).unsqueeze(0)
        query_context = (contexts[mid_idx] + 0.1 * torch.randn(16)).unsqueeze(0)

        # Test different time weights
        time_weight_effects = []
        for time_weight in [0.0, 0.1, 1.0]:
            with torch.no_grad():
                retrieved = memory(query_key, query_context, time_weight=time_weight)

            # Check if it retrieved the right episode
            best_match_idx = -1
            best_similarity = -1
            for i, value in enumerate(values):
                sim = F.cosine_similarity(retrieved.squeeze(), value, dim=0).item()
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_idx = i

            time_weight_effects.append((time_weight, best_match_idx, best_similarity))

        results[gran_name] = time_weight_effects
        print(f"Results: {time_weight_effects}")

    return results


def orthogonalize_vectors(vectors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Orthogonalize a list of vectors using QR decomposition.

    Args:
        vectors: List of vectors to orthogonalize.

    Returns:
        List of orthogonalized vectors.
    """
    # Stack vectors into a matrix and transpose to get shape (n_vectors, vector_dim)
    matrix = torch.stack(vectors).T
    # QR decomposition gives us orthogonal vectors in Q
    Q, _ = torch.linalg.qr(matrix)
    # Transpose back and convert to list of vectors
    return [Q.T[i] for i in range(Q.shape[1])]


# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # Set random seeds for repeatability
    # seed = 42
    seed = 52
    torch.manual_seed(seed)
    np.random.seed(seed)

    # print("=== Testing TrueEpisodicMemory Module ===\n")

    # print("1. Testing Episodic Temporal Interference...")
    # interference_results = test_episodic_temporal_interference()
    # print(f"Results: {interference_results}\n")

    # print("2. Testing Episodic Sequence Recall...")
    # sequence_results = test_episodic_sequence_recall(
    #     inner_steps=1,
    #     outer_steps=5,
    #     delta_time=1,
    #     sequence_length=30,
    # )
    # print(f"Results: {sequence_results}\n")

    print("3. Testing Multi-Scale Temporal Retrieval...")
    multi_scale_results = test_multi_scale_temporal_retrieval()
    print(f"Results: {multi_scale_results}\n")

    print("4. Testing Multi-Scale Temporal Retrieval (Debug)...")
    debug_results = test_multi_scale_temporal_retrieval_debug()
    print(f"Results: {debug_results}\n")

    print("5. Testing Temporal Granularity Effects...")
    granularity_results = test_temporal_granularity_effects()
    print(f"Results: {granularity_results}\n")

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
