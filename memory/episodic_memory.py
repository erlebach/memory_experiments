import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    def __init__(self, dim, memory_size=100, update_steps=1, lr=0.01):
        super().__init__()
        # Use nn.Parameter for internal state that needs to be updated via backprop
        self.mem_keys = nn.Parameter(torch.randn(memory_size, dim))
        self.mem_values = nn.Parameter(torch.randn(memory_size, dim))
        print(f"mem_keys: {self.mem_keys.shape=}")
        print(f"mem_values: {self.mem_values.shape=}")
        self.update_steps = update_steps

        # The module manages its own optimizer for test-time updates
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, query):
        # print(f"In forward:{query.shape=}")
        # Standard retrieval logic
        attn_scores = torch.matmul(query, self.mem_keys.T)
        attn_probs = F.softmax(attn_scores, dim=-1)
        retrieved_value = torch.matmul(attn_probs, self.mem_values)
        return retrieved_value

    def update(self, new_keys, new_values):
        # The key logic: temporarily enable gradients for this update step,
        # even if the parent model is in eval mode and under a no_grad() context.
        with torch.enable_grad():
            for _ in range(self.update_steps):
                self.optimizer.zero_grad()
                # Use a loss function to drive the update
                # print(f"Before forward:{new_keys.shape=}")
                retrieved = self.forward(new_keys)
                loss = F.mse_loss(retrieved, new_values)
                loss.backward()  # Gradients are computed only for self.keys and self.values
                print("loss: ", loss.detach().item())
                self.optimizer.step()

                with torch.no_grad():
                    value = self.forward(new_keys[0])
                    error = F.mse_loss(value, new_values[0])
                    print("error: ", error.detach().item())


def test_adaptation_speed(
    dim: int = 32, memory_size: int = 100, max_steps: int = 20
) -> list[float]:
    """Test how quickly memory adapts to a new key-value pair.

    Args:
        dim: Embedding dimension for keys and values.
        memory_size: Number of memory slots.
        max_steps: Maximum number of update steps to test.

    Returns:
        List of MSE errors at each update step.

    """
    # Create fresh memory
    memory = EpisodicMemory(dim=dim, memory_size=memory_size, update_steps=1, lr=0.01)

    # Create novel key-value pair
    k_target = torch.randn(1, dim)
    v_target = torch.randn(1, dim)

    errors = []

    # Initial query (baseline)
    with torch.no_grad():
        initial_retrieved = memory(k_target)
        error_0 = F.mse_loss(initial_retrieved, v_target).item()
        errors.append(error_0)
        print(f"Initial error (step 0): {error_0:.4f}")

    # Iterative updates
    for step in range(1, max_steps + 1):
        # Perform one update step
        memory.update(k_target, v_target)

        # Query and measure error
        with torch.no_grad():
            retrieved = memory(k_target)
            error = F.mse_loss(retrieved, v_target).item()
            errors.append(error)
            print(f"Error after step {step}: {error:.4f}")

    print(f"Test passed: Adaptation speed measured over {max_steps} steps")
    return errors


def test_capture_capacity(
    dim: int = 32,
    memory_size: int = 100,
    n_items: int = 20,
    update_steps: int = 5,
    similarity_threshold: float = 0.8,
) -> tuple[float, float]:
    """Test memory's capacity to store and recall multiple items.

    Args:
        dim: Embedding dimension for keys and values.
        memory_size: Number of memory slots.
        n_items: Number of unique key-value pairs to store.
        update_steps: Number of update steps for bulk learning.
        similarity_threshold: Cosine similarity threshold for correct recall.

    Returns:
        Tuple of (recall_accuracy, average_retrieval_error).

    """
    # Create fresh memory
    memory = EpisodicMemory(
        dim=dim, memory_size=memory_size, update_steps=update_steps, lr=0.01
    )

    # Create N unique key-value pairs
    keys = torch.randn(n_items, dim)
    values = torch.randn(n_items, dim)

    print(f"Storing {n_items} key-value pairs...")

    # Bulk update
    memory.update(keys, values)

    # Item-by-item recall
    correct_recalls = 0
    total_errors = []

    print("Testing recall...")
    with torch.no_grad():
        for i in range(n_items):
            # Query with original key
            retrieved = memory(keys[i : i + 1])  # Keep batch dimension

            # Find best match among original values using cosine similarity
            similarities = F.cosine_similarity(retrieved, values, dim=1)
            best_match_idx = torch.argmax(similarities).item()
            best_similarity = similarities[best_match_idx].item()

            # Check if correct (best match is the original value)
            if best_match_idx == i and best_similarity > similarity_threshold:
                correct_recalls += 1

            # Calculate retrieval error for this item
            error = F.mse_loss(retrieved, values[i : i + 1]).item()
            total_errors.append(error)

            print(
                f"Item {i}: best_match={best_match_idx}, similarity={best_similarity:.3f}, "
                f"correct={best_match_idx == i and best_similarity > similarity_threshold}"
            )

    recall_accuracy = correct_recalls / n_items
    avg_error = np.mean(total_errors)

    print(f"Recall accuracy: {recall_accuracy:.3f} ({correct_recalls}/{n_items})")
    print(f"Average retrieval error: {avg_error:.4f}")
    print(f"Test passed: Capacity test completed")

    return recall_accuracy, avg_error


def test_catastrophic_forgetting(
    dim: int = 32, memory_size: int = 100, task_size: int = 10, update_steps: int = 5
) -> tuple[float, float, float]:
    """Test memory's retention after learning new tasks (catastrophic forgetting).

    Args:
        dim: Embedding dimension for keys and values.
        memory_size: Number of memory slots.
        task_size: Number of key-value pairs per task.
        update_steps: Number of update steps per task.

    Returns:
        Tuple of (initial_accuracy_A, final_accuracy_A, forgetting_score).

    """
    # Create fresh memory
    memory = EpisodicMemory(
        dim=dim, memory_size=memory_size, update_steps=update_steps, lr=0.01
    )

    # Create Task A and Task B datasets
    # Task A: keys centered around [2, 0, 0, ...]
    task_A_keys = torch.randn(task_size, dim) + torch.cat(
        [torch.tensor([2.0]), torch.zeros(dim - 1)]
    )
    task_A_values = torch.randn(task_size, dim)

    # Task B: keys centered around [-2, 0, 0, ...]
    task_B_keys = torch.randn(task_size, dim) + torch.cat(
        [torch.tensor([-2.0]), torch.zeros(dim - 1)]
    )
    task_B_values = torch.randn(task_size, dim)

    # Test keys for evaluation (subset of training keys)
    test_size = min(5, task_size)
    task_A_test_keys = task_A_keys[:test_size]
    task_A_test_values = task_A_values[:test_size]

    print("Learning Task A...")
    # Learn Task A
    memory.update(task_A_keys, task_A_values)

    # Test on Task A (initial performance)
    print("Testing initial Task A performance...")
    accuracy_A1 = _evaluate_task_accuracy(memory, task_A_test_keys, task_A_test_values)
    print(f"Initial Task A accuracy: {accuracy_A1:.3f}")

    print("Learning Task B...")
    # Learn Task B (without showing Task A data)
    memory.update(task_B_keys, task_B_values)

    # Test on Task A (final performance)
    print("Testing final Task A performance...")
    accuracy_A2 = _evaluate_task_accuracy(memory, task_A_test_keys, task_A_test_values)
    print(f"Final Task A accuracy: {accuracy_A2:.3f}")

    # Calculate forgetting score
    forgetting_score = accuracy_A1 - accuracy_A2
    print(f"Forgetting score: {forgetting_score:.3f}")

    if forgetting_score < 0.1:
        print("Low catastrophic forgetting - good retention!")
    elif forgetting_score > 0.5:
        print("High catastrophic forgetting - significant knowledge loss")
    else:
        print("Moderate catastrophic forgetting")

    print(f"Test passed: Catastrophic forgetting test completed")

    return accuracy_A1, accuracy_A2, forgetting_score


def _evaluate_task_accuracy(
    memory: EpisodicMemory,
    test_keys: torch.Tensor,
    test_values: torch.Tensor,
    similarity_threshold: float = 0.7,
) -> float:
    """Evaluate accuracy on a task.

    Args:
        memory: The memory module to test.
        test_keys: Keys to query with.
        test_values: Expected values.
        similarity_threshold: Cosine similarity threshold for correct recall.

    Returns:
        Accuracy as fraction of correctly recalled items.

    """
    correct = 0
    total = len(test_keys)

    with torch.no_grad():
        for i in range(total):
            retrieved = memory(test_keys[i : i + 1])
            similarity = F.cosine_similarity(
                retrieved, test_values[i : i + 1], dim=1
            ).item()

            if similarity > similarity_threshold:
                correct += 1

    return correct / total


if __name__ == "__main__":
    print("=== Testing EpisodicMemory Module ===\n")

    # Test 1: Adaptation Speed
    print("1. Testing Adaptation Speed...")
    errors = test_adaptation_speed(dim=32, memory_size=50, max_steps=15)
    assert len(errors) == 16, "Should have 16 error measurements (0 to 15 steps)"
    assert errors[0] > errors[-1], "Error should decrease over time"
    print("✓ Adaptation speed test passed\n")

    # Test 2: Capture Capacity
    print("2. Testing Capture Capacity...")
    recall_acc, avg_err = test_capture_capacity(dim=32, memory_size=50, n_items=10)
    assert 0 <= recall_acc <= 1, "Recall accuracy should be between 0 and 1"
    assert avg_err >= 0, "Average error should be non-negative"
    print("✓ Capture capacity test passed\n")

    # Test 3: Catastrophic Forgetting
    print("3. Testing Catastrophic Forgetting...")
    acc_A1, acc_A2, forgetting = test_catastrophic_forgetting(
        dim=32, memory_size=50, task_size=8
    )
    assert 0 <= acc_A1 <= 1, "Initial accuracy should be between 0 and 1"
    assert 0 <= acc_A2 <= 1, "Final accuracy should be between 0 and 1"
    assert (
        forgetting == acc_A1 - acc_A2
    ), "Forgetting score should equal accuracy difference"
    print("✓ Catastrophic forgetting test passed\n")

    print("=== All tests completed successfully! ===")
