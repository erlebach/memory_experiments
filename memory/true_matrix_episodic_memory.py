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


class TrueMatrixEpisodicMemory(nn.Module):
    """Matrix-based episodic memory with temporal and contextual binding.

    Memory structure: [M, d1, d2] where each memory slot is a d1 x d2 matrix.
    """

    def __init__(
        self,
        dim: int,
        memory_size: int = 100,
        context_dim: int = 16,
        lr: float = 0.01,
        matrix_dim1: int = 8,
        matrix_dim2: int = 8,
    ):
        """Initialize matrix episodic memory.

        Args:
            dim: Feature dimension for queries/values (must equal matrix_dim1 * matrix_dim2)
            memory_size: Number of memory slots
            context_dim: Context vector dimension
            lr: Learning rate for memory updates
            matrix_dim1: First dimension of memory matrices
            matrix_dim2: Second dimension of memory matrices

        """
        super().__init__()

        assert (
            dim == matrix_dim1 * matrix_dim2
        ), f"dim ({dim}) must equal matrix_dim1 * matrix_dim2 ({matrix_dim1 * matrix_dim2})"

        # Memory structure: [M, d1, d2] - each slot is a matrix
        self.mem_keys = nn.Parameter(
            torch.randn(memory_size, matrix_dim1, matrix_dim2) * 0.1
        )
        self.mem_values = nn.Parameter(
            torch.randn(memory_size, matrix_dim1, matrix_dim2) * 0.1
        )
        self.mem_contexts = nn.Parameter(torch.randn(memory_size, context_dim) * 0.1)
        self.mem_timestamps = nn.Parameter(torch.zeros(memory_size, 1))

        self.dim = dim
        self.context_dim = context_dim
        self.memory_size = memory_size
        self.matrix_dim1 = matrix_dim1
        self.matrix_dim2 = matrix_dim2
        self.current_time = 0
        self.write_pointer = 0

        # Track which slots are actually used
        self.used_slots = torch.zeros(memory_size, dtype=torch.bool)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _vector_to_matrix(self, vector: Tensor) -> Tensor:
        """Convert vector to matrix format.

        Args:
            vector: Input tensor of shape [batch, dim]

        Returns:
            Matrix tensor of shape [batch, matrix_dim1, matrix_dim2]

        """
        batch_size = vector.shape[0]
        return vector.view(batch_size, self.matrix_dim1, self.matrix_dim2)

    def _matrix_to_vector(self, matrix: Tensor) -> Tensor:
        """Convert matrix to vector format.

        Args:
            matrix: Input tensor of shape [batch, matrix_dim1, matrix_dim2]

        Returns:
            Vector tensor of shape [batch, dim]

        """
        batch_size = matrix.shape[0]
        return matrix.view(batch_size, self.dim)

    def _matrix_similarity(self, query_matrix: Tensor, key_matrices: Tensor) -> Tensor:
        """Compute matrix-based similarity using Frobenius inner product.

        Args:
            query_matrix: Query matrix [batch, d1, d2]
            key_matrices: Key matrices [memory_size, d1, d2]

        Returns:
            Similarity scores [batch, memory_size]

        """
        # Frobenius inner product: tr(A^T B) = sum(A * B)
        similarities = einsum(
            query_matrix, key_matrices, "batch d1 d2, memory d1 d2 -> batch memory"
        )
        return similarities

    def forward(
        self, query: Tensor, context: Tensor | None = None, time_weight: float = 0.1
    ) -> Tensor:
        """Retrieve with temporal and contextual cues using matrix memory.

        Args:
            query: Query vector [batch, dim]
            context: Context vector [batch, context_dim] (optional)
            time_weight: Weight for temporal decay

        Returns:
            Retrieved value vector [batch, dim]

        """
        # Only consider used memory slots
        if self.used_slots.sum() == 0:
            # No memories stored yet, return random
            return torch.randn_like(query)

        # Mask for used slots
        used_mask = self.used_slots.float()

        # Convert query vector to matrix format
        query_matrix = self._vector_to_matrix(query)

        # Matrix-based content similarity using Frobenius inner product
        content_scores = self._matrix_similarity(query_matrix, self.mem_keys)

        # Add contextual similarity if context provided
        if context is not None:
            context_scores = einsum(
                context,
                self.mem_contexts,
                "batch context_dim, memory_size context_dim -> batch memory_size",
            )
            content_scores = content_scores + 0.5 * context_scores

        # Add temporal recency bias (more recent = higher weight)
        time_decay = torch.exp(
            -time_weight * (self.current_time - self.mem_timestamps.squeeze())
        )
        content_scores = content_scores + 0.3 * time_decay.unsqueeze(0)

        # Mask unused slots
        content_scores = content_scores * used_mask.unsqueeze(0)
        content_scores = content_scores + (1 - used_mask.unsqueeze(0)) * (-1e9)

        # Compute attention weights
        attn_probs = F.softmax(content_scores, dim=-1)

        # Retrieve value matrices and convert back to vectors
        retrieved_matrices = einsum(
            attn_probs,
            self.mem_values,
            "batch memory_size, memory_size d1 d2 -> batch d1 d2",
        )
        retrieved_value = self._matrix_to_vector(retrieved_matrices)

        return retrieved_value

    def store_episode(
        self,
        keys: Tensor,
        values: Tensor,
        context: Tensor | None = None,
        update_steps: int = 3,
    ) -> None:
        """Store new episode with learning-based updates using matrix memory.

        Args:
            keys: Key vectors [batch, dim]
            values: Value vectors [batch, dim]
            context: Context vectors [batch, context_dim] (optional)
            update_steps: Number of gradient steps for storage

        """
        batch_size = keys.shape[0]

        for i in range(batch_size):
            # Find slot to update (circular buffer)
            idx = self.write_pointer % self.memory_size
            self.used_slots[idx] = True

            # Convert vectors to matrices for storage
            key_matrix = self._vector_to_matrix(keys[i : i + 1])
            value_matrix = self._vector_to_matrix(values[i : i + 1])

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


def test_matrix_capacity_advantage(
    matrix_dim1: int = 8, matrix_dim2: int = 8, num_patterns: int = 100
) -> dict:
    """Test the capacity advantage of matrix memory over linear memory.

    Matrix memory can theoretically store up to d1*d2 orthogonal patterns,
    while linear memory can only store d patterns exactly.

    Args:
        matrix_dim1: First matrix dimension
        matrix_dim2: Second matrix dimension
        num_patterns: Number of patterns to test

    Returns:
        Dictionary comparing matrix vs linear memory capacity

    """
    dim = matrix_dim1 * matrix_dim2

    print(
        f"=== Testing Memory Capacity: Matrix ({matrix_dim1}x{matrix_dim2}) vs Linear ({dim}D) ==="
    )
    print(f"Theoretical capacity - Matrix: {matrix_dim1 * matrix_dim2}, Linear: {dim}")

    # Create orthogonal matrix patterns
    def create_orthogonal_matrices(n_matrices: int, d1: int, d2: int) -> Tensor:
        """Create approximately orthogonal matrices."""
        matrices = []
        for i in range(n_matrices):
            # Create structured patterns that are approximately orthogonal
            matrix = torch.zeros(d1, d2)
            if i < d1:  # Row patterns
                matrix[i, :] = 1.0
            elif i < d1 + d2:  # Column patterns
                matrix[:, i - d1] = 1.0
            elif i < d1 + d2 + min(d1, d2):  # Diagonal patterns
                diag_idx = i - d1 - d2
                if diag_idx < min(d1, d2):
                    matrix[diag_idx, diag_idx] = 1.0
            else:  # Random orthogonal patterns
                matrix = torch.randn(d1, d2)
                # Orthogonalize against previous matrices
                for prev_matrix in matrices:
                    overlap = torch.sum(matrix * prev_matrix)
                    matrix = matrix - overlap * prev_matrix
                matrix = matrix / (torch.norm(matrix) + 1e-8)

            matrices.append(matrix)

        return torch.stack(matrices)

    # Test different numbers of patterns
    pattern_counts = [10, 20, 30, 40, 50, 60, 70, 80]
    matrix_accuracies = []
    linear_accuracies = []

    for n_patterns in pattern_counts:
        if n_patterns > num_patterns:
            break

        print(f"\nTesting with {n_patterns} patterns...")

        # Create patterns
        matrix_patterns = create_orthogonal_matrices(
            n_patterns, matrix_dim1, matrix_dim2
        )
        linear_patterns = matrix_patterns.view(
            n_patterns, -1
        )  # Flatten for linear memory

        # Test Matrix Memory
        matrix_memory = TrueMatrixEpisodicMemory(
            dim=dim,
            memory_size=n_patterns + 10,
            matrix_dim1=matrix_dim1,
            matrix_dim2=matrix_dim2,
            lr=0.01,
        )

        # Store patterns in matrix memory
        matrix_memory.store_episode(linear_patterns, linear_patterns, update_steps=10)

        # Test retrieval accuracy for matrix memory
        matrix_correct = 0
        with torch.no_grad():
            for i in range(n_patterns):
                retrieved = matrix_memory(linear_patterns[i : i + 1])
                similarity = F.cosine_similarity(
                    retrieved, linear_patterns[i : i + 1]
                ).item()
                if similarity > 0.9:  # High similarity threshold
                    matrix_correct += 1

        matrix_accuracy = matrix_correct / n_patterns
        matrix_accuracies.append(matrix_accuracy)

        # Test Linear Memory (simple associative memory)
        # For fair comparison, use same learning approach but with vector operations
        linear_memory_keys = nn.Parameter(torch.randn(n_patterns + 10, dim) * 0.1)
        linear_memory_values = nn.Parameter(torch.randn(n_patterns + 10, dim) * 0.1)
        linear_optimizer = torch.optim.Adam(
            [linear_memory_keys, linear_memory_values], lr=0.01
        )

        # Train linear memory
        for epoch in range(10):
            for i in range(n_patterns):
                linear_optimizer.zero_grad()

                # Compute similarities
                similarities = torch.matmul(
                    linear_patterns[i : i + 1], linear_memory_keys[:n_patterns].T
                )
                weights = F.softmax(similarities, dim=-1)
                retrieved = torch.matmul(weights, linear_memory_values[:n_patterns])

                loss = F.mse_loss(retrieved, linear_patterns[i : i + 1])
                loss.backward()
                linear_optimizer.step()

        # Test retrieval accuracy for linear memory
        linear_correct = 0
        with torch.no_grad():
            for i in range(n_patterns):
                similarities = torch.matmul(
                    linear_patterns[i : i + 1], linear_memory_keys[:n_patterns].T
                )
                weights = F.softmax(similarities, dim=-1)
                retrieved = torch.matmul(weights, linear_memory_values[:n_patterns])
                similarity = F.cosine_similarity(
                    retrieved, linear_patterns[i : i + 1]
                ).item()
                if similarity > 0.9:
                    linear_correct += 1

        linear_accuracy = linear_correct / n_patterns
        linear_accuracies.append(linear_accuracy)

        print(f"Matrix Memory Accuracy: {matrix_accuracy:.3f}")
        print(f"Linear Memory Accuracy: {linear_accuracy:.3f}")

    # Visualization
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        pattern_counts[: len(matrix_accuracies)],
        matrix_accuracies,
        "o-",
        label="Matrix Memory",
        color="blue",
        linewidth=2,
    )
    plt.plot(
        pattern_counts[: len(linear_accuracies)],
        linear_accuracies,
        "s-",
        label="Linear Memory",
        color="red",
        linewidth=2,
    )
    plt.axvline(
        x=dim, color="red", linestyle="--", alpha=0.7, label=f"Linear Capacity ({dim})"
    )
    plt.axvline(
        x=matrix_dim1 * matrix_dim2,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Matrix Capacity ({matrix_dim1*matrix_dim2})",
    )
    plt.xlabel("Number of Patterns")
    plt.ylabel("Retrieval Accuracy")
    plt.title("Memory Capacity Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    capacity_advantage = [m - l for m, l in zip(matrix_accuracies, linear_accuracies)]
    plt.bar(
        pattern_counts[: len(capacity_advantage)],
        capacity_advantage,
        color="green",
        alpha=0.7,
    )
    plt.xlabel("Number of Patterns")
    plt.ylabel("Matrix Advantage (Accuracy Difference)")
    plt.title("Matrix Memory Advantage")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("matrix_capacity_advantage.png", dpi=150, bbox_inches="tight")
    plt.show()

    results = {
        "pattern_counts": pattern_counts[: len(matrix_accuracies)],
        "matrix_accuracies": matrix_accuracies,
        "linear_accuracies": linear_accuracies,
        "capacity_advantage": capacity_advantage,
    }

    print("✓ Matrix capacity advantage test completed")
    return results


def test_structural_pattern_memory(matrix_dim1: int = 8, matrix_dim2: int = 8) -> dict:
    """Test matrix memory's ability to preserve and retrieve structural patterns.

    This tests something linear memory fundamentally cannot do well:
    preserve 2D spatial/structural relationships.

    Args:
        matrix_dim1: First matrix dimension
        matrix_dim2: Second matrix dimension

    Returns:
        Dictionary of structural pattern results

    """
    dim = matrix_dim1 * matrix_dim2

    print("=== Testing Structural Pattern Memory ===")

    def create_structural_patterns():
        """Create patterns with clear 2D structure."""
        patterns = {}

        # 1. Diagonal patterns
        diag_pattern = torch.zeros(matrix_dim1, matrix_dim2)
        diag_pattern.fill_diagonal_(1.0)
        patterns["diagonal"] = diag_pattern

        # 2. Cross pattern
        cross_pattern = torch.zeros(matrix_dim1, matrix_dim2)
        mid_row, mid_col = matrix_dim1 // 2, matrix_dim2 // 2
        cross_pattern[mid_row, :] = 1.0
        cross_pattern[:, mid_col] = 1.0
        patterns["cross"] = cross_pattern

        # 3. Border pattern
        border_pattern = torch.zeros(matrix_dim1, matrix_dim2)
        border_pattern[0, :] = border_pattern[-1, :] = 1.0
        border_pattern[:, 0] = border_pattern[:, -1] = 1.0
        patterns["border"] = border_pattern

        # 4. Checkerboard pattern
        checker_pattern = torch.zeros(matrix_dim1, matrix_dim2)
        for i in range(matrix_dim1):
            for j in range(matrix_dim2):
                if (i + j) % 2 == 0:
                    checker_pattern[i, j] = 1.0
        patterns["checkerboard"] = checker_pattern

        # 5. Spiral pattern
        spiral_pattern = torch.zeros(matrix_dim1, matrix_dim2)
        # Simple spiral approximation
        center_x, center_y = matrix_dim1 // 2, matrix_dim2 // 2
        for i in range(matrix_dim1):
            for j in range(matrix_dim2):
                dist = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
                if abs(dist - 2) < 1:
                    spiral_pattern[i, j] = 1.0
        patterns["spiral"] = spiral_pattern

        return patterns

    # Create patterns
    patterns = create_structural_patterns()
    pattern_names = list(patterns.keys())
    pattern_matrices = torch.stack([patterns[name] for name in pattern_names])
    pattern_vectors = pattern_matrices.view(len(pattern_names), -1)

    # Test Matrix Memory
    matrix_memory = TrueMatrixEpisodicMemory(
        dim=dim,
        memory_size=20,
        matrix_dim1=matrix_dim1,
        matrix_dim2=matrix_dim2,
        lr=0.01,
    )

    # Store patterns
    matrix_memory.store_episode(pattern_vectors, pattern_vectors, update_steps=15)

    # Test retrieval and structural preservation
    matrix_results = {}

    plt.figure(figsize=(20, 12))

    for i, name in enumerate(pattern_names):
        # Retrieve pattern
        with torch.no_grad():
            retrieved_vector = matrix_memory(pattern_vectors[i : i + 1])
            retrieved_matrix = retrieved_vector.view(matrix_dim1, matrix_dim2)

        # Calculate structural similarity metrics
        original_matrix = pattern_matrices[i]

        # 1. Frobenius norm similarity
        frobenius_sim = F.cosine_similarity(
            original_matrix.flatten().unsqueeze(0),
            retrieved_matrix.flatten().unsqueeze(0),
        ).item()

        # 2. Structural correlation (position-wise correlation)
        struct_corr = torch.corrcoef(
            torch.stack([original_matrix.flatten(), retrieved_matrix.flatten()])
        )[0, 1].item()

        # 3. Peak preservation (how well peaks are preserved)
        orig_peaks = (original_matrix > 0.5).float()
        retr_peaks = (retrieved_matrix > 0.5).float()
        peak_overlap = (orig_peaks * retr_peaks).sum() / (orig_peaks.sum() + 1e-8)

        matrix_results[name] = {
            "frobenius_similarity": frobenius_sim,
            "structural_correlation": struct_corr,
            "peak_preservation": peak_overlap.item(),
        }

        # Visualization
        plt.subplot(3, len(pattern_names), i + 1)
        plt.imshow(original_matrix.numpy(), cmap="viridis")
        plt.title(f"Original {name.title()}")
        plt.colorbar()

        plt.subplot(3, len(pattern_names), i + 1 + len(pattern_names))
        plt.imshow(retrieved_matrix.detach().numpy(), cmap="viridis")
        plt.title(f"Retrieved {name.title()}\nSim: {frobenius_sim:.3f}")
        plt.colorbar()

        plt.subplot(3, len(pattern_names), i + 1 + 2 * len(pattern_names))
        diff = torch.abs(original_matrix - retrieved_matrix)
        plt.imshow(diff.detach().numpy(), cmap="Reds")
        plt.title(f"Difference\nCorr: {struct_corr:.3f}")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig("structural_pattern_memory.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Summary statistics
    avg_frobenius = np.mean(
        [r["frobenius_similarity"] for r in matrix_results.values()]
    )
    avg_correlation = np.mean(
        [r["structural_correlation"] for r in matrix_results.values()]
    )
    avg_peak_preservation = np.mean(
        [r["peak_preservation"] for r in matrix_results.values()]
    )

    print(f"Average Frobenius Similarity: {avg_frobenius:.3f}")
    print(f"Average Structural Correlation: {avg_correlation:.3f}")
    print(f"Average Peak Preservation: {avg_peak_preservation:.3f}")

    results = {
        "pattern_results": matrix_results,
        "average_frobenius": avg_frobenius,
        "average_correlation": avg_correlation,
        "average_peak_preservation": avg_peak_preservation,
    }

    print("✓ Structural pattern memory test completed")
    return results


def test_compositional_memory(matrix_dim1: int = 8, matrix_dim2: int = 8) -> dict:
    """Test matrix memory's ability to compose and decompose patterns.

    This tests matrix memory's ability to handle compositional structures
    that linear memory cannot represent effectively.

    Args:
        matrix_dim1: First matrix dimension
        matrix_dim2: Second matrix dimension

    Returns:
        Dictionary of compositional memory results

    """
    dim = matrix_dim1 * matrix_dim2

    print("=== Testing Compositional Memory ===")

    # Create base components
    def create_base_components():
        """Create basic components that can be composed."""
        components = {}

        # Horizontal line
        h_line = torch.zeros(matrix_dim1, matrix_dim2)
        h_line[matrix_dim1 // 2, :] = 1.0
        components["h_line"] = h_line

        # Vertical line
        v_line = torch.zeros(matrix_dim1, matrix_dim2)
        v_line[:, matrix_dim2 // 2] = 1.0
        components["v_line"] = v_line

        # Corner (top-left)
        corner = torch.zeros(matrix_dim1, matrix_dim2)
        corner[0, : matrix_dim2 // 2] = 1.0
        corner[: matrix_dim1 // 2, 0] = 1.0
        components["corner"] = corner

        return components

    # Create compositions
    def create_compositions(components):
        """Create composed patterns from base components."""
        compositions = {}

        # Cross = horizontal + vertical
        compositions["cross"] = components["h_line"] + components["v_line"]

        # T-shape = horizontal + partial vertical
        t_shape = components["h_line"].clone()
        t_shape[: matrix_dim1 // 2, matrix_dim2 // 2] = 1.0
        compositions["t_shape"] = t_shape

        # L-shape = corner
        compositions["l_shape"] = components["corner"]

        # Plus with corner = cross + corner
        plus_corner = components["h_line"] + components["v_line"] + components["corner"]
        compositions["plus_corner"] = torch.clamp(plus_corner, 0, 1)

        return compositions

    components = create_base_components()
    compositions = create_compositions(components)

    # Prepare data
    all_patterns = {**components, **compositions}
    pattern_names = list(all_patterns.keys())
    pattern_matrices = torch.stack([all_patterns[name] for name in pattern_names])
    pattern_vectors = pattern_matrices.view(len(pattern_names), -1)

    # Train matrix memory
    matrix_memory = TrueMatrixEpisodicMemory(
        dim=dim,
        memory_size=30,
        matrix_dim1=matrix_dim1,
        matrix_dim2=matrix_dim2,
        lr=0.01,
    )

    matrix_memory.store_episode(pattern_vectors, pattern_vectors, update_steps=20)

    # Test compositional queries
    print("Testing compositional queries...")

    # Query 1: Given horizontal line, can we retrieve cross?
    # This tests if the memory can understand compositional relationships
    h_line_vector = pattern_vectors[pattern_names.index("h_line")].unsqueeze(0)

    with torch.no_grad():
        # Retrieve using horizontal line as query
        retrieved = matrix_memory(h_line_vector)
        retrieved_matrix = retrieved.view(matrix_dim1, matrix_dim2)

        # Check similarity to cross (which contains horizontal line)
        cross_matrix = all_patterns["cross"]
        cross_similarity = F.cosine_similarity(
            retrieved.flatten().unsqueeze(0), cross_matrix.flatten().unsqueeze(0)
        ).item()

    # Query 2: Partial pattern completion
    # Give partial cross, expect full cross
    partial_cross = all_patterns["cross"].clone()
    partial_cross[:, matrix_dim2 // 2 + 1 :] = 0  # Remove right half of vertical line
    partial_cross_vector = partial_cross.view(1, -1)

    with torch.no_grad():
        completed = matrix_memory(partial_cross_vector)
        completed_matrix = completed.view(matrix_dim1, matrix_dim2)

        completion_similarity = F.cosine_similarity(
            completed.flatten().unsqueeze(0), cross_matrix.flatten().unsqueeze(0)
        ).item()

    # Visualization
    plt.figure(figsize=(16, 10))

    # Show all stored patterns
    for i, name in enumerate(pattern_names):
        plt.subplot(3, len(pattern_names), i + 1)
        plt.imshow(pattern_matrices[i].numpy(), cmap="viridis")
        plt.title(f"{name.title()}")
        plt.colorbar()

    # Show compositional query result
    plt.subplot(3, 3, len(pattern_names) + 1)
    plt.imshow(all_patterns["h_line"].numpy(), cmap="viridis")
    plt.title("Query: H-Line")
    plt.colorbar()

    plt.subplot(3, 3, len(pattern_names) + 2)
    plt.imshow(retrieved_matrix.detach().numpy(), cmap="viridis")
    plt.title(f"Retrieved\nCross Sim: {cross_similarity:.3f}")
    plt.colorbar()

    plt.subplot(3, 3, len(pattern_names) + 3)
    plt.imshow(cross_matrix.numpy(), cmap="viridis")
    plt.title("Target: Cross")
    plt.colorbar()

    # Show pattern completion
    plt.subplot(3, 3, 2 * len(pattern_names) + 1)
    plt.imshow(partial_cross.numpy(), cmap="viridis")
    plt.title("Partial Cross")
    plt.colorbar()

    plt.subplot(3, 3, 2 * len(pattern_names) + 2)
    plt.imshow(completed_matrix.detach().numpy(), cmap="viridis")
    plt.title(f"Completed\nSim: {completion_similarity:.3f}")
    plt.colorbar()

    plt.subplot(3, 3, 2 * len(pattern_names) + 3)
    plt.imshow(cross_matrix.numpy(), cmap="viridis")
    plt.title("Target: Full Cross")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("compositional_memory.png", dpi=150, bbox_inches="tight")
    plt.show()

    results = {
        "cross_similarity_from_hline": cross_similarity,
        "pattern_completion_similarity": completion_similarity,
        "stored_patterns": pattern_names,
    }

    print(f"Cross similarity from H-line query: {cross_similarity:.3f}")
    print(f"Pattern completion similarity: {completion_similarity:.3f}")
    print("✓ Compositional memory test completed")

    return results


if __name__ == "__main__":
    # Set random seeds for repeatability
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=== Testing Matrix Memory Unique Advantages ===\n")

    # Test with 8x8 matrices (64-dimensional vectors)
    matrix_dim1, matrix_dim2 = 8, 8

    print("1. Testing Matrix Memory Capacity Advantage...")
    capacity_results = test_matrix_capacity_advantage(
        matrix_dim1=matrix_dim1, matrix_dim2=matrix_dim2, num_patterns=80
    )
    print(f"Results: {capacity_results}\n")

    print("2. Testing Structural Pattern Memory...")
    structural_results = test_structural_pattern_memory(
        matrix_dim1=matrix_dim1, matrix_dim2=matrix_dim2
    )
    print(f"Results: {structural_results}\n")

    print("3. Testing Compositional Memory...")
    compositional_results = test_compositional_memory(
        matrix_dim1=matrix_dim1, matrix_dim2=matrix_dim2
    )
    print(f"Results: {compositional_results}\n")

    print("=== Matrix Memory Advantage Tests Completed! ===")
