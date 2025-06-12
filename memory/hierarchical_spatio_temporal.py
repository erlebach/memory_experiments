from dataclasses import dataclass
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


@dataclass
class MemoryNode:
    """A node in the hierarchical memory structure."""

    content: Tensor  # The actual memory content
    level: int  # Hierarchy level (0=atomic, higher=more abstract)
    timespan: Tuple[float, float]  # (start_time, end_time)
    spatial_extent: Tuple[int, int]  # (spatial_start, spatial_end)
    children: List["MemoryNode"]  # Sub-memories
    parent: Optional["MemoryNode"]  # Parent memory
    context: Optional[Tensor]  # Contextual information
    importance: float  # Memory importance/salience


class HierarchicalSpatioTemporalMemory(nn.Module):
    """Hierarchical episodic memory with spatial and temporal structure.

    Memory is organized as a tree where:
    - Leaves are atomic memories (moments, locations)
    - Internal nodes are composite memories (events, scenes, episodes)
    - Each node has spatial and temporal extent
    - Higher levels represent longer timespans and larger spatial areas
    """

    def __init__(
        self,
        dim: int = 64,
        max_levels: int = 5,
        spatial_dim: int = 32,
        temporal_dim: int = 32,
        context_dim: int = 16,
        lr: float = 0.01,
    ):
        """Initialize hierarchical spatio-temporal memory.

        Args:
            dim: Base feature dimension
            max_levels: Maximum hierarchy levels
            spatial_dim: Spatial encoding dimension
            temporal_dim: Temporal encoding dimension
            context_dim: Context dimension
            lr: Learning rate

        """
        super().__init__()

        self.dim = dim
        self.max_levels = max_levels
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.context_dim = context_dim

        # Hierarchical memory storage
        self.memory_trees: List[MemoryNode] = []
        self.current_time = 0.0
        self.current_location = 0

        # Encoding networks for different scales
        self.spatial_encoders = nn.ModuleList(
            [nn.Linear(spatial_dim, dim) for _ in range(max_levels)]
        )
        self.temporal_encoders = nn.ModuleList(
            [nn.Linear(temporal_dim, dim) for _ in range(max_levels)]
        )

        # Hierarchical composition networks
        self.composition_networks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim * 2, dim),  # Combine child memories
                    nn.ReLU(),
                    nn.Linear(dim, dim),
                )
                for _ in range(max_levels - 1)
            ]
        )

        # Attention mechanisms for each level
        self.attention_networks = nn.ModuleList(
            [
                nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
                for _ in range(max_levels)
            ]
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def encode_spatiotemporal_position(
        self, spatial_pos: int, temporal_pos: float, level: int
    ) -> Tensor:
        """Encode spatial and temporal position with level-specific scaling.

        Args:
            spatial_pos: Spatial position
            temporal_pos: Temporal position
            level: Hierarchy level

        Returns:
            Position encoding tensor

        """
        # Convert to tensors
        spatial_pos = torch.tensor(spatial_pos, dtype=torch.float32)
        temporal_pos = torch.tensor(temporal_pos, dtype=torch.float32)

        spatial_encoding = torch.zeros(self.spatial_dim)
        temporal_encoding = torch.zeros(self.temporal_dim)

        # Spatial encoding (scale varies by level)
        spatial_scale = 2**level
        for i in range(0, self.spatial_dim, 2):
            spatial_encoding[i] = torch.sin(
                spatial_pos / (10000 ** (i / self.spatial_dim)) / spatial_scale
            )
            if i + 1 < self.spatial_dim:
                spatial_encoding[i + 1] = torch.cos(
                    spatial_pos / (10000 ** (i / self.spatial_dim)) / spatial_scale
                )

        # Temporal encoding (scale varies by level)
        temporal_scale = 2**level
        for i in range(0, self.temporal_dim, 2):
            temporal_encoding[i] = torch.sin(
                temporal_pos / (10000 ** (i / self.temporal_dim)) / temporal_scale
            )
            if i + 1 < self.temporal_dim:
                temporal_encoding[i + 1] = torch.cos(
                    temporal_pos / (10000 ** (i / self.temporal_dim)) / temporal_scale
                )

        # Encode through level-specific networks
        spatial_encoded = self.spatial_encoders[level](spatial_encoding)
        temporal_encoded = self.temporal_encoders[level](temporal_encoding)

        return spatial_encoded + temporal_encoded

    def create_memory_node(
        self,
        content: Tensor,
        level: int,
        timespan: Tuple[float, float],
        spatial_extent: Tuple[int, int],
        context: Optional[Tensor] = None,
        importance: float = 1.0,
    ) -> MemoryNode:
        """Create a new memory node.

        Args:
            content: Memory content
            level: Hierarchy level
            timespan: Temporal extent
            spatial_extent: Spatial extent
            context: Optional context
            importance: Memory importance

        Returns:
            New memory node

        """
        return MemoryNode(
            content=content,
            level=level,
            timespan=timespan,
            spatial_extent=spatial_extent,
            children=[],
            parent=None,
            context=context,
            importance=importance,
        )

    def compose_memories(self, child_memories: List[MemoryNode], level: int) -> Tensor:
        """Compose child memories into parent memory at higher level.

        Args:
            child_memories: List of child memory nodes
            level: Target level for composition

        Returns:
            Composed memory content

        """
        if not child_memories:
            return torch.zeros(self.dim)

        if len(child_memories) == 1:
            return child_memories[0].content

        # Attention-based composition
        child_contents = torch.stack([child.content for child in child_memories])
        child_contents = child_contents.unsqueeze(0)  # Add batch dimension

        # Self-attention to find important relationships
        attended, _ = self.attention_networks[level](
            child_contents, child_contents, child_contents
        )

        # Hierarchical composition
        composed = attended.mean(dim=1).squeeze(0)  # Average attended representations

        # Apply composition network
        if level > 0:
            # Combine with temporal structure
            temporal_info = self.encode_spatiotemporal_position(
                spatial_pos=(
                    child_memories[0].spatial_extent[0]
                    + child_memories[-1].spatial_extent[1]
                )
                // 2,
                temporal_pos=(
                    child_memories[0].timespan[0] + child_memories[-1].timespan[1]
                )
                / 2,
                level=level,
            )
            combined = torch.cat([composed, temporal_info])
            composed = self.composition_networks[level - 1](combined)

        return composed

    def store_atomic_memory(
        self,
        content: Tensor,
        spatial_pos: int,
        temporal_pos: float,
        context: Optional[Tensor] = None,
        importance: float = 1.0,
    ) -> MemoryNode:
        """Store an atomic (level 0) memory.

        Args:
            content: Memory content
            spatial_pos: Spatial position
            temporal_pos: Temporal position
            context: Optional context
            importance: Memory importance

        Returns:
            Created memory node

        """
        # Add positional encoding to content
        position_encoding = self.encode_spatiotemporal_position(
            spatial_pos, temporal_pos, 0
        )
        encoded_content = content + position_encoding

        # Create atomic memory node
        memory_node = self.create_memory_node(
            content=encoded_content,
            level=0,
            timespan=(temporal_pos, temporal_pos),
            spatial_extent=(spatial_pos, spatial_pos),
            context=context,
            importance=importance,
        )

        return memory_node

    def build_hierarchy(self, atomic_memories: List[MemoryNode]) -> MemoryNode:
        """Build hierarchical structure from atomic memories.

        Args:
            atomic_memories: List of atomic memory nodes

        Returns:
            Root of memory hierarchy

        """
        current_level_memories = atomic_memories

        for level in range(1, self.max_levels):
            next_level_memories = []

            # Group memories by spatial and temporal proximity
            groups = self._group_memories_by_proximity(current_level_memories, level)

            for group in groups:
                if len(group) > 1:  # Only create parent if multiple children
                    # Compute extent of group
                    min_time = min(mem.timespan[0] for mem in group)
                    max_time = max(mem.timespan[1] for mem in group)
                    min_space = min(mem.spatial_extent[0] for mem in group)
                    max_space = max(mem.spatial_extent[1] for mem in group)

                    # Compose memories
                    composed_content = self.compose_memories(group, level)

                    # Create parent memory
                    parent_memory = self.create_memory_node(
                        content=composed_content,
                        level=level,
                        timespan=(min_time, max_time),
                        spatial_extent=(min_space, max_space),
                        importance=sum(mem.importance for mem in group) / len(group),
                    )

                    # Set parent-child relationships
                    for child in group:
                        child.parent = parent_memory
                        parent_memory.children.append(child)

                    next_level_memories.append(parent_memory)
                else:
                    # Single memory, promote to next level
                    next_level_memories.extend(group)

            current_level_memories = next_level_memories

            if len(current_level_memories) <= 1:
                break

        return current_level_memories[0] if current_level_memories else None

    def _group_memories_by_proximity(
        self, memories: List[MemoryNode], level: int
    ) -> List[List[MemoryNode]]:
        """Group memories by spatial and temporal proximity.

        Args:
            memories: List of memory nodes to group
            level: Current hierarchy level

        Returns:
            List of memory groups

        """
        if not memories:
            return []

        # Sort by time first, then by space
        sorted_memories = sorted(
            memories, key=lambda m: (m.timespan[0], m.spatial_extent[0])
        )

        groups = []
        current_group = [sorted_memories[0]]

        # Group based on temporal and spatial windows (scale with level)
        temporal_window = 2**level
        spatial_window = 2**level

        for memory in sorted_memories[1:]:
            last_memory = current_group[-1]

            # Check if memory is within temporal and spatial windows
            temporal_gap = memory.timespan[0] - last_memory.timespan[1]
            spatial_gap = abs(memory.spatial_extent[0] - last_memory.spatial_extent[1])

            if temporal_gap <= temporal_window and spatial_gap <= spatial_window:
                current_group.append(memory)
            else:
                groups.append(current_group)
                current_group = [memory]

        groups.append(current_group)
        return groups

    def retrieve_hierarchical(
        self, query: Tensor, spatial_pos: int, temporal_pos: float, level: int = None
    ) -> Tuple[Tensor, List[MemoryNode]]:
        """Retrieve memories hierarchically.

        Args:
            query: Query tensor
            spatial_pos: Query spatial position
            temporal_pos: Query temporal position
            level: Specific level to query (None for all levels)

        Returns:
            Retrieved content and relevant memory nodes

        """
        if not self.memory_trees:
            return torch.zeros_like(query), []

        # Add positional encoding to query
        query_levels = []
        if level is None:
            levels_to_search = range(self.max_levels)
        else:
            levels_to_search = [level]

        for l in levels_to_search:
            position_encoding = self.encode_spatiotemporal_position(
                spatial_pos, temporal_pos, l
            )
            encoded_query = query + position_encoding
            query_levels.append((l, encoded_query))

        # Search through memory trees
        relevant_nodes = []
        similarities = []

        for tree_root in self.memory_trees:
            nodes_at_levels = self._collect_nodes_at_levels(tree_root, levels_to_search)

            for level_nodes in nodes_at_levels:
                for node in level_nodes:
                    # Check spatial-temporal overlap
                    if self._check_spatiotemporal_overlap(
                        node, spatial_pos, temporal_pos
                    ):
                        # Compute similarity
                        for l, encoded_query in query_levels:
                            if node.level == l:
                                similarity = F.cosine_similarity(
                                    encoded_query.unsqueeze(0),
                                    node.content.unsqueeze(0),
                                ).item()
                                similarities.append(similarity * node.importance)
                                relevant_nodes.append(node)

        if not relevant_nodes:
            return torch.zeros_like(query), []

        # Weight by similarity and importance
        similarities = torch.tensor(similarities)
        weights = F.softmax(similarities, dim=0)

        # Compose retrieved content
        retrieved_content = torch.zeros_like(query)
        for weight, node in zip(weights, relevant_nodes):
            retrieved_content += weight * node.content

        return retrieved_content, relevant_nodes

    def _collect_nodes_at_levels(
        self, root: MemoryNode, levels: List[int]
    ) -> List[List[MemoryNode]]:
        """Collect nodes at specified levels from tree.

        Args:
            root: Root of memory tree
            levels: Levels to collect

        Returns:
            List of node lists for each level

        """
        nodes_by_level = {level: [] for level in levels}

        def traverse(node):
            if node.level in levels:
                nodes_by_level[node.level].append(node)
            for child in node.children:
                traverse(child)

        traverse(root)
        return [nodes_by_level[level] for level in levels]

    def _check_spatiotemporal_overlap(
        self, node: MemoryNode, spatial_pos: int, temporal_pos: float
    ) -> bool:
        """Check if query position overlaps with memory node.

        Args:
            node: Memory node to check
            spatial_pos: Query spatial position
            temporal_pos: Query temporal position

        Returns:
            True if there's overlap

        """
        spatial_overlap = (
            node.spatial_extent[0] <= spatial_pos <= node.spatial_extent[1]
        )
        temporal_overlap = node.timespan[0] <= temporal_pos <= node.timespan[1]

        return spatial_overlap and temporal_overlap

    def store_episode(
        self, atomic_memories: List[Tuple[Tensor, int, float, Optional[Tensor], float]]
    ) -> MemoryNode:
        """Store a complete episode as hierarchical memory.

        Args:
            atomic_memories: List of (content, spatial_pos, temporal_pos, context, importance)

        Returns:
            Root of created memory hierarchy

        """
        # Create atomic memory nodes
        memory_nodes = []
        for content, spatial_pos, temporal_pos, context, importance in atomic_memories:
            node = self.store_atomic_memory(
                content, spatial_pos, temporal_pos, context, importance
            )
            memory_nodes.append(node)

        # Build hierarchy
        root = self.build_hierarchy(memory_nodes)

        if root:
            self.memory_trees.append(root)

        return root


def test_hierarchical_memory_structure():
    """Test the hierarchical structure of spatio-temporal memory."""
    print("=== Testing Hierarchical Spatio-Temporal Memory ===")

    memory = HierarchicalSpatioTemporalMemory(
        dim=64, max_levels=4, spatial_dim=32, temporal_dim=32
    )

    # Create a story: "Walking through a park"
    # This will have multiple levels:
    # Level 0: Individual moments (seeing flower, hearing bird, etc.)
    # Level 1: Scenes (flower garden, pond area, playground)
    # Level 2: Segments (morning walk, afternoon rest)
    # Level 3: Complete episode (day at the park)

    atomic_memories = []

    # Morning walk segment (time 0-10, space 0-20)
    morning_memories = [
        (torch.randn(64), 0, 0.0, None, 1.0),  # Enter park
        (torch.randn(64), 2, 1.0, None, 0.8),  # See flowers
        (torch.randn(64), 4, 2.0, None, 0.9),  # Smell roses
        (torch.randn(64), 6, 3.0, None, 0.7),  # Hear birds
        (torch.randn(64), 8, 4.0, None, 0.6),  # Walk on path
    ]

    # Pond area segment (time 5-15, space 10-30)
    pond_memories = [
        (torch.randn(64), 10, 5.0, None, 0.9),  # Approach pond
        (torch.randn(64), 12, 6.0, None, 1.0),  # See ducks
        (torch.randn(64), 14, 7.0, None, 0.8),  # Feed ducks
        (torch.randn(64), 16, 8.0, None, 0.7),  # Sit on bench
        (torch.randn(64), 18, 9.0, None, 0.6),  # Watch water
    ]

    # Playground segment (time 10-20, space 20-40)
    playground_memories = [
        (torch.randn(64), 20, 10.0, None, 0.8),  # See children
        (torch.randn(64), 22, 11.0, None, 0.9),  # Hear laughter
        (torch.randn(64), 24, 12.0, None, 0.7),  # Watch swings
        (torch.randn(64), 26, 13.0, None, 0.6),  # Remember childhood
        (torch.randn(64), 28, 14.0, None, 0.5),  # Feel nostalgic
    ]

    # Combine all atomic memories
    all_memories = morning_memories + pond_memories + playground_memories
    atomic_memories.extend(all_memories)

    # Store the episode
    print("Storing hierarchical episode...")
    root = memory.store_episode(atomic_memories)

    if root:
        print(f"Created memory hierarchy with root at level {root.level}")
        print(f"Root timespan: {root.timespan}")
        print(f"Root spatial extent: {root.spatial_extent}")

        # Visualize hierarchy
        visualize_memory_hierarchy(root)

        # Test retrieval at different levels
        print("\nTesting retrieval at different levels...")

        # Query for specific moment (level 0)
        query = torch.randn(64)
        retrieved, nodes = memory.retrieve_hierarchical(query, 12, 6.0, level=0)
        print(f"Level 0 retrieval found {len(nodes)} atomic memories")

        # Query for scene (level 1)
        retrieved, nodes = memory.retrieve_hierarchical(query, 15, 7.5, level=1)
        print(f"Level 1 retrieval found {len(nodes)} scene memories")

        # Query for segment (level 2)
        retrieved, nodes = memory.retrieve_hierarchical(query, 20, 10.0, level=2)
        print(f"Level 2 retrieval found {len(nodes)} segment memories")

        # Query for episode (level 3)
        retrieved, nodes = memory.retrieve_hierarchical(query, 15, 10.0, level=3)
        print(f"Level 3 retrieval found {len(nodes)} episode memories")

    print("✓ Hierarchical memory structure test completed")


def visualize_memory_hierarchy(root: MemoryNode, level: int = 0):
    """Visualize the hierarchical memory structure."""
    indent = "  " * level
    print(
        f"{indent}Level {root.level}: "
        f"Time {root.timespan[0]:.1f}-{root.timespan[1]:.1f}, "
        f"Space {root.spatial_extent[0]}-{root.spatial_extent[1]}, "
        f"Children: {len(root.children)}"
    )

    for child in root.children:
        visualize_memory_hierarchy(child, level + 1)


def test_hierarchical_retrieval_dynamics():
    """Test how retrieval works across different hierarchy levels."""
    print("\n=== Testing Hierarchical Retrieval Dynamics ===")

    memory = HierarchicalSpatioTemporalMemory(dim=32, max_levels=3)

    # Create nested episode: "Learning Einstein's equation"
    # Level 0: Individual concepts (E, =, m, c, ²)
    # Level 1: Equation components (E=mc², physics context)
    # Level 2: Complete learning episode

    learning_episode = [
        # Introduction to energy concept
        (torch.tensor([1.0] + [0.0] * 31), 0, 0.0, None, 1.0),  # "Energy"
        (torch.tensor([0.0, 1.0] + [0.0] * 30), 1, 1.0, None, 0.8),  # "is"
        (torch.tensor([0.0, 0.0, 1.0] + [0.0] * 29), 2, 2.0, None, 0.9),  # "conserved"
        # Introduction to mass concept
        (torch.tensor([0.0] * 3 + [1.0] + [0.0] * 28), 3, 3.0, None, 1.0),  # "Mass"
        (torch.tensor([0.0] * 4 + [1.0] + [0.0] * 27), 4, 4.0, None, 0.8),  # "has"
        (torch.tensor([0.0] * 5 + [1.0] + [0.0] * 26), 5, 5.0, None, 0.9),  # "inertia"
        # The equation itself
        (
            torch.tensor([1.0, 0.0, 0.0, 1.0] + [0.0] * 28),
            6,
            6.0,
            None,
            1.0,
        ),  # "E = mc²"
        (
            torch.tensor([0.0] * 7 + [1.0] + [0.0] * 24),
            7,
            7.0,
            None,
            0.9,
        ),  # "revolutionary"
        (torch.tensor([0.0] * 8 + [1.0] + [0.0] * 23), 8, 8.0, None, 0.8),  # "insight"
    ]

    root = memory.store_episode(learning_episode)

    # Test different types of queries
    queries = [
        ("Energy concept", torch.tensor([1.0] + [0.0] * 31), 0, 1.0),
        ("Mass concept", torch.tensor([0.0] * 3 + [1.0] + [0.0] * 28), 3, 4.0),
        ("Full equation", torch.tensor([1.0, 0.0, 0.0, 1.0] + [0.0] * 28), 6, 6.5),
        ("Physics context", torch.tensor([0.5, 0.0, 0.0, 0.5] + [0.0] * 28), 4, 5.0),
    ]

    plt.figure(figsize=(15, 10))

    for i, (query_name, query_vec, spatial_pos, temporal_pos) in enumerate(queries):
        similarities_by_level = []

        for level in range(3):
            retrieved, nodes = memory.retrieve_hierarchical(
                query_vec, spatial_pos, temporal_pos, level=level
            )

            if len(nodes) > 0:
                avg_similarity = sum(
                    F.cosine_similarity(
                        query_vec.unsqueeze(0), node.content.unsqueeze(0)
                    ).item()
                    for node in nodes
                ) / len(nodes)
            else:
                avg_similarity = 0.0

            similarities_by_level.append(avg_similarity)

        plt.subplot(2, 2, i + 1)
        plt.bar(range(3), similarities_by_level, color=["red", "green", "blue"])
        plt.title(f"Query: {query_name}")
        plt.xlabel("Hierarchy Level")
        plt.ylabel("Average Similarity")
        plt.xticks(range(3), ["Atomic", "Component", "Episode"])

    plt.tight_layout()
    plt.savefig("hierarchical_retrieval_dynamics.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✓ Hierarchical retrieval dynamics test completed")


if __name__ == "__main__":
    # Set random seeds for repeatability
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=== Testing Hierarchical Spatio-Temporal Episodic Memory ===\n")

    test_hierarchical_memory_structure()
    test_hierarchical_retrieval_dynamics()

    print("\n=== All hierarchical memory tests completed! ===")
