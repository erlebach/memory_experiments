import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrueEpisodicMemory(nn.Module):
    """Episodic memory with temporal and contextual binding."""

    def __init__(self, dim, memory_size=100, context_dim=16, lr=0.01):
        super().__init__()
        self.mem_keys = nn.Parameter(torch.randn(memory_size, dim))
        self.mem_values = nn.Parameter(torch.randn(memory_size, dim))
        # Add temporal and contextual dimensions
        self.mem_contexts = nn.Parameter(torch.randn(memory_size, context_dim))
        self.mem_timestamps = nn.Parameter(torch.zeros(memory_size, 1))

        self.dim = dim
        self.context_dim = context_dim
        self.memory_size = memory_size
        self.current_time = 0
        self.write_pointer = 0  # Circular buffer for episodic storage

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, query, context=None, time_weight=0.1):
        """Retrieve with temporal and contextual cues."""
        # Standard content-based attention
        content_scores = torch.matmul(query, self.mem_keys.T)

        # Add contextual similarity if context provided
        if context is not None:
            context_scores = torch.matmul(context, self.mem_contexts.T)
            content_scores = content_scores + 0.5 * context_scores

        # Add temporal recency bias (more recent = higher weight)
        time_decay = torch.exp(-time_weight * (self.current_time - self.mem_timestamps.squeeze()))
        content_scores = content_scores + time_decay.unsqueeze(0)

        attn_probs = F.softmax(content_scores, dim=-1)
        retrieved_value = torch.matmul(attn_probs, self.mem_values)
        return retrieved_value

    def store_episode(self, keys, values, context=None):
        """Store new episode with temporal and contextual binding."""
        batch_size = keys.shape[0]

        with torch.enable_grad():
            for i in range(batch_size):
                # Store in circular buffer (episodic replacement)
                idx = self.write_pointer % self.memory_size

                # Update memory slot
                self.mem_keys.data[idx] = keys[i]
                self.mem_values.data[idx] = values[i]
                self.mem_timestamps.data[idx] = self.current_time

                if context is not None:
                    self.mem_contexts.data[idx] = context[i]

                self.write_pointer += 1
                self.current_time += 1


def test_episodic_temporal_interference(dim: int = 32, context_dim: int = 16, 
                                      memory_size: int = 20) -> dict:
    """Test temporal interference in episodic memory.
    
    This tests a key property of episodic memory: recent episodes can interfere 
    with recall of older episodes, especially when they share similar content.
    
    Args:
        dim: Content dimension
        context_dim: Context dimension  
        memory_size: Number of memory slots
        
    Returns:
        Dictionary with test results
        
    """
    memory = TrueEpisodicMemory(dim=dim, context_dim=context_dim, 
                               memory_size=memory_size, lr=0.01)

    print("=== Testing Episodic Temporal Interference ===")

    # Phase 1: Store "morning episodes" 
    print("Phase 1: Storing morning episodes...")
    morning_context = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(5, 1)  # Morning context
    morning_keys = torch.randn(5, dim) + torch.tensor([1.0] + [0.0]*(dim-1))  # Similar content
    morning_values = torch.randn(5, dim) + torch.tensor([2.0] + [0.0]*(dim-1))  # Morning-specific values

    memory.store_episode(morning_keys, morning_values, morning_context[:, :context_dim])

    # Test immediate recall of morning episodes
    morning_recall_immediate = []
    with torch.no_grad():
        for i in range(5):
            retrieved = memory(morning_keys[i:i+1], morning_context[i:i+1, :context_dim])
            similarity = F.cosine_similarity(retrieved, morning_values[i:i+1]).item()
            morning_recall_immediate.append(similarity)

    avg_morning_immediate = np.mean(morning_recall_immediate)
    print(f"Morning episodes immediate recall: {avg_morning_immediate:.3f}")

    # Phase 2: Store many "afternoon episodes" (interference)
    print("Phase 2: Storing interfering afternoon episodes...")
    afternoon_context = torch.tensor([0.0, 1.0, 0.0, 0.0]).repeat(15, 1)  # Different context
    # Similar keys to morning (content interference) but different context
    afternoon_keys = torch.randn(15, dim) + torch.tensor([1.0] + [0.0]*(dim-1))  
    afternoon_values = torch.randn(15, dim) + torch.tensor([-2.0] + [0.0]*(dim-1))  # Different values

    memory.store_episode(afternoon_keys, afternoon_values, afternoon_context[:, :context_dim])

    # Phase 3: Test recall of original morning episodes (should show interference)
    print("Phase 3: Testing morning recall after interference...")
    morning_recall_delayed = []
    with torch.no_grad():
        for i in range(5):
            retrieved = memory(morning_keys[i:i+1], morning_context[i:i+1, :context_dim])
            similarity = F.cosine_similarity(retrieved, morning_values[i:i+1]).item()
            morning_recall_delayed.append(similarity)

    avg_morning_delayed = np.mean(morning_recall_delayed)
    print(f"Morning episodes delayed recall: {avg_morning_delayed:.3f}")

    # Calculate interference effect
    interference_effect = avg_morning_immediate - avg_morning_delayed
    print(f"Temporal interference effect: {interference_effect:.3f}")

    # Phase 4: Test context-dependent retrieval
    print("Phase 4: Testing context-dependent retrieval...")

    # Query with morning content but afternoon context (should retrieve afternoon)
    test_key = morning_keys[0:1]  # Morning content
    afternoon_ctx = afternoon_context[0:1, :context_dim]  # Afternoon context

    with torch.no_grad():
        retrieved_mixed = memory(test_key, afternoon_ctx)

        # Check which it's more similar to
        morning_sim = F.cosine_similarity(retrieved_mixed, morning_values[0:1]).item()
        afternoon_sim = F.cosine_similarity(retrieved_mixed, afternoon_values[0]).item()

    print(f"Mixed query - Morning similarity: {morning_sim:.3f}, Afternoon similarity: {afternoon_sim:.3f}")

    context_selectivity = afternoon_sim - morning_sim
    print(f"Context selectivity: {context_selectivity:.3f}")

    results = {
        'morning_immediate_recall': avg_morning_immediate,
        'morning_delayed_recall': avg_morning_delayed, 
        'interference_effect': interference_effect,
        'context_selectivity': context_selectivity
    }

    # Assertions for episodic memory properties
    assert interference_effect > 0, "Should show temporal interference (delayed < immediate)"
    assert context_selectivity > 0, "Should show context-dependent retrieval"

    print("✓ Episodic temporal interference test passed")
    return results


def test_episodic_sequence_recall(dim: int = 32, context_dim: int = 16, 
                                sequence_length: int = 8) -> dict:
    """Test sequential episodic recall - a hallmark of episodic memory.
    
    Tests whether the memory can recall events in temporal order and whether
    recalling one episode cues recall of temporally adjacent episodes.
    
    Args:
        dim: Content dimension
        context_dim: Context dimension
        sequence_length: Length of episode sequence
        
    Returns:
        Dictionary with test results
        
    """
    memory = TrueEpisodicMemory(dim=dim, context_dim=context_dim, 
                               memory_size=50, lr=0.01)

    print("=== Testing Episodic Sequence Recall ===")

    # Create a sequence of related episodes (like a story)
    story_context = torch.randn(1, context_dim).repeat(sequence_length, 1)  # Shared story context

    # Create sequence where each episode relates to the next
    sequence_keys = []
    sequence_values = []

    base_pattern = torch.randn(dim)
    for i in range(sequence_length):
        # Each episode is similar to previous but with some drift
        key = base_pattern + 0.3 * torch.randn(dim) + 0.1 * i * torch.randn(dim)
        value = base_pattern + 0.5 * torch.randn(dim) + 0.2 * i * torch.ones(dim)

        sequence_keys.append(key)
        sequence_values.append(value)

    sequence_keys = torch.stack(sequence_keys)
    sequence_values = torch.stack(sequence_values)

    # Store the sequence
    print(f"Storing sequence of {sequence_length} episodes...")
    memory.store_episode(sequence_keys, sequence_values, story_context)

    # Test 1: Forward sequence recall
    print("Testing forward sequence recall...")
    forward_accuracies = []

    with torch.no_grad():
        for i in range(sequence_length - 1):
            # Query with episode i, see if it helps recall episode i+1
            current_retrieved = memory(sequence_keys[i:i+1], story_context[i:i+1])

            # Use retrieved content as cue for next episode
            next_retrieved = memory(current_retrieved, story_context[i+1:i+2])

            # Check similarity to actual next episode
            similarity = F.cosine_similarity(next_retrieved, sequence_values[i+1:i+2]).item()
            forward_accuracies.append(similarity)

            print(f"Episode {i} -> {i+1} similarity: {similarity:.3f}")

    avg_forward_accuracy = np.mean(forward_accuracies)
    print(f"Average forward sequence accuracy: {avg_forward_accuracy:.3f}")

    # Test 2: Random access vs sequential access
    print("Testing random vs sequential access...")

    # Random access: query episodes in random order
    random_indices = torch.randperm(sequence_length)
    random_accuracies = []

    with torch.no_grad():
        for idx in random_indices:
            retrieved = memory(sequence_keys[idx:idx+1], story_context[idx:idx+1])
            similarity = F.cosine_similarity(retrieved, sequence_values[idx:idx+1]).item()
            random_accuracies.append(similarity)

    avg_random_accuracy = np.mean(random_accuracies)
    print(f"Average random access accuracy: {avg_random_accuracy:.3f}")

    # Sequential advantage
    sequential_advantage = avg_forward_accuracy - avg_random_accuracy
    print(f"Sequential advantage: {sequential_advantage:.3f}")

    results = {
        'forward_sequence_accuracy': avg_forward_accuracy,
        'random_access_accuracy': avg_random_accuracy,
        'sequential_advantage': sequential_advantage
    }

    # Assertions for episodic properties
    assert avg_forward_accuracy > 0.3, "Should show some sequential structure"

    print("✓ Episodic sequence recall test passed")
    return results


# Add to __main__ section:
if __name__ == "__main__":
    print("=== Testing EpisodicMemory Module ===\n")
    print("4. Testing Episodic Temporal Interference...")
    interference_results = test_episodic_temporal_interference()
    print("✓ Episodic temporal interference test passed\n")

    print("5. Testing Episodic Sequence Recall...")
    sequence_results = test_episodic_sequence_recall()
    print("✓ Episodic sequence recall test passed\n")

    print("=== All tests completed successfully! ===")
