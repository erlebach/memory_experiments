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

from memory.hierarchical_spatio_temporal import HierarchicalSpatioTemporalMemory

def test_multi_scale_temporal_retrieval():
    """Test retrieval at different temporal scales and hierarchy levels."""
    print("\n=== Testing Multi-Scale Temporal Retrieval ===")

    memory = HierarchicalSpatioTemporalMemory(dim=32, max_levels=4)

    # Create multi-scale temporal episode: "A day at university"
    # Seconds scale: Individual actions
    # Minutes scale: Lectures, breaks
    # Hours scale: Morning, afternoon, evening
    # Day scale: Complete university day

    university_day = []

    # Morning (0-4 hours): Getting ready and commuting
    morning_actions = [
        (torch.randn(32), 0, 0.1, None, 0.8),   # Wake up
        (torch.randn(32), 1, 0.5, None, 0.6),   # Shower
        (torch.randn(32), 2, 1.0, None, 0.7),   # Breakfast
        (torch.randn(32), 3, 2.0, None, 0.9),   # Commute
    ]

    # Afternoon (4-8 hours): Classes and studying
    afternoon_actions = [
        (torch.randn(32), 10, 4.0, None, 1.0),  # Physics lecture
        (torch.randn(32), 11, 4.5, None, 0.9),  # Take notes
        (torch.randn(32), 12, 5.0, None, 0.8),  # Ask question
        (torch.randn(32), 15, 6.0, None, 0.7),  # Library study
        (torch.randn(32), 16, 7.0, None, 0.8),  # Group discussion
    ]

    # Evening (8-12 hours): Social and rest
    evening_actions = [
        (torch.randn(32), 20, 8.0, None, 0.6),  # Dinner
        (torch.randn(32), 21, 9.0, None, 0.8),  # Friends
        (torch.randn(32), 22, 10.0, None, 0.5), # Relax
        (torch.randn(32), 23, 11.0, None, 0.4), # Sleep prep
    ]

    university_day.extend(morning_actions + afternoon_actions + evening_actions)
    root = memory.store_episode(university_day)

    # Test queries at different temporal scales
    test_queries = [
        ("Specific moment", torch.randn(32), 11, 4.5, "Taking notes in physics"),
        ("Lecture period", torch.randn(32), 11, 4.75, "Physics class"),
        ("Afternoon block", torch.randn(32), 13, 5.5, "Academic activities"),
        ("Entire day", torch.randn(32), 12, 6.0, "University experience"),
    ]

    plt.figure(figsize=(12, 8))

    for i, (query_name, query_vec, spatial_pos, temporal_pos, description) in enumerate(test_queries):
        level_results = []

        for level in range(4):
            retrieved, nodes = memory.retrieve_hierarchical(
                query_vec, spatial_pos, temporal_pos, level=level
            )

            # Measure temporal span of retrieved memories
            if nodes:
                min_time = min(node.timespan[0] for node in nodes)
                max_time = max(node.timespan[1] for node in nodes)
                temporal_span = max_time - min_time
                num_memories = len(nodes)
            else:
                temporal_span = 0
                num_memories = 0

            level_results.append((temporal_span, num_memories))

        plt.subplot(2, 2, i + 1)
        spans, counts = zip(*level_results)

        ax1 = plt.gca()
        bars1 = ax1.bar([x - 0.2 for x in range(4)], spans, 0.4, 
                       label='Temporal Span', color='skyblue')
        ax1.set_ylabel('Temporal Span (hours)', color='blue')

        ax2 = ax1.twinx()
        bars2 = ax2.bar([x + 0.2 for x in range(4)], counts, 0.4, 
                       label='Memory Count', color='lightcoral')
        ax2.set_ylabel('Number of Memories', color='red')

        plt.title(f'{query_name}\n"{description}"')
        ax1.set_xlabel('Hierarchy Level')
        ax1.set_xticks(range(4))
        ax1.set_xticklabels(['Atomic', 'Scene', 'Segment', 'Episode'])

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig("multi_scale_temporal_retrieval.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✓ Multi-scale temporal retrieval test completed")


def test_hierarchical_generalization():
    """Test if higher levels can extract abstract patterns across episodes."""
    print("\n=== Testing Hierarchical Generalization ===")

    memory = HierarchicalSpatioTemporalMemory(dim=32, max_levels=3)

    # Create multiple similar episodes with common abstract structure
    # Pattern: Problem → Analysis → Solution → Verification

    episodes = []

    # Episode 1: Math problem solving
    math_episode = [
        (torch.tensor([1.0, 0.0] + [0.0] * 30), 0, 0.0, None, 1.0),  # Problem
        (torch.tensor([0.0, 1.0] + [0.0] * 30), 1, 1.0, None, 0.9),  # Analysis
        (torch.tensor([0.0, 0.0, 1.0] + [0.0] * 29), 2, 2.0, None, 1.0),  # Solution
        (torch.tensor([0.0, 0.0, 0.0, 1.0] + [0.0] * 28), 3, 3.0, None, 0.8),  # Check
    ]

    # Episode 2: Coding problem solving (same abstract pattern)
    coding_episode = [
        (torch.tensor([1.0, 0.0] + [0.1] * 30), 10, 10.0, None, 1.0),  # Problem
        (torch.tensor([0.0, 1.0] + [0.1] * 30), 11, 11.0, None, 0.9),  # Analysis
        (torch.tensor([0.0, 0.0, 1.0] + [0.1] * 29), 12, 12.0, None, 1.0),  # Solution
        (torch.tensor([0.0, 0.0, 0.0, 1.0] + [0.1] * 28), 13, 13.0, None, 0.8),  # Check
    ]

    # Episode 3: Physics problem solving (same abstract pattern)
    physics_episode = [
        (torch.tensor([1.0, 0.0] + [0.2] * 30), 20, 20.0, None, 1.0),  # Problem
        (torch.tensor([0.0, 1.0] + [0.2] * 30), 21, 21.0, None, 0.9),  # Analysis
        (torch.tensor([0.0, 0.0, 1.0] + [0.2] * 29), 22, 22.0, None, 1.0),  # Solution
        (torch.tensor([0.0, 0.0, 0.0, 1.0] + [0.2] * 28), 23, 23.0, None, 0.8),  # Check
    ]

    episodes = [math_episode, coding_episode, physics_episode]
    roots = []

    for i, episode in enumerate(episodes):
        root = memory.store_episode(episode)
        roots.append(root)
        print(f"Stored episode {i+1} with {len(episode)} atomic memories")

    # Test generalization: Query with abstract pattern
    abstract_queries = [
        ("Problem phase", torch.tensor([1.0, 0.0] + [0.0] * 30)),
        ("Analysis phase", torch.tensor([0.0, 1.0] + [0.0] * 30)),
        ("Solution phase", torch.tensor([0.0, 0.0, 1.0] + [0.0] * 29)),
        ("Verification phase", torch.tensor([0.0, 0.0, 0.0, 1.0] + [0.0] * 28)),
    ]

    plt.figure(figsize=(12, 8))

    for i, (phase_name, query_vec) in enumerate(abstract_queries):
        cross_episode_similarities = []

        for level in range(3):
            level_similarities = []

            # Query each episode at this level
            for episode_idx in range(3):
                # Query at the center of each episode
                spatial_center = episode_idx * 10 + 1.5
                temporal_center = episode_idx * 10 + 1.5

                retrieved, nodes = memory.retrieve_hierarchical(
                    query_vec, int(spatial_center), temporal_center, level=level
                )

                if nodes:
                    similarities = [
                        F.cosine_similarity(
                            query_vec.unsqueeze(0), node.content.unsqueeze(0)
                        ).item() for node in nodes
                    ]
                    avg_similarity = sum(similarities) / len(similarities)
                else:
                    avg_similarity = 0.0

                level_similarities.append(avg_similarity)

            # Measure consistency across episodes (lower std = better generalization)
            consistency = 1.0 - np.std(level_similarities) if level_similarities else 0.0
            cross_episode_similarities.append(consistency)

        plt.subplot(2, 2, i + 1)
        plt.bar(range(3), cross_episode_similarities, 
                color=['red', 'green', 'blue'])
        plt.title(f'{phase_name}\nCross-Episode Consistency')
        plt.xlabel('Hierarchy Level')
        plt.ylabel('Generalization Score')
        plt.xticks(range(3), ['Atomic', 'Component', 'Episode'])
        plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig("hierarchical_generalization.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✓ Hierarchical generalization test completed")


def test_temporal_chunking_segmentation():
    """Test automatic discovery of episode boundaries."""
    print("\n=== Testing Temporal Chunking and Segmentation ===")

    memory = HierarchicalSpatioTemporalMemory(dim=32, max_levels=4)

    # Create continuous sequence with natural boundaries
    # Sequence: Morning routine → Commute → Work → Lunch → Work → Commute → Evening

    continuous_sequence = []

    # Morning routine (high similarity within, distinct from others)
    morning_base = torch.tensor([1.0, 0.0, 0.0, 0.0] + [0.0] * 28)
    for i in range(5):
        noise = torch.randn(32) * 0.1
        continuous_sequence.append((morning_base + noise, i, float(i), None, 1.0))

    # Commute 1 (different pattern)
    commute1_base = torch.tensor([0.0, 1.0, 0.0, 0.0] + [0.0] * 28)
    for i in range(3):
        noise = torch.randn(32) * 0.1
        continuous_sequence.append((commute1_base + noise, i+5, float(i+5), None, 0.8))

    # Work morning (another distinct pattern)
    work_am_base = torch.tensor([0.0, 0.0, 1.0, 0.0] + [0.0] * 28)
    for i in range(6):
        noise = torch.randn(32) * 0.1
        continuous_sequence.append((work_am_base + noise, i+8, float(i+8), None, 0.9))

    # Lunch (brief, distinct)
    lunch_base = torch.tensor([0.0, 0.0, 0.0, 1.0] + [0.0] * 28)
    for i in range(2):
        noise = torch.randn(32) * 0.1
        continuous_sequence.append((lunch_base + noise, i+14, float(i+14), None, 0.7))

    # Work afternoon (similar to morning work but slightly different)
    work_pm_base = torch.tensor([0.0, 0.0, 0.8, 0.2] + [0.0] * 28)
    for i in range(5):
        noise = torch.randn(32) * 0.1
        continuous_sequence.append((work_pm_base + noise, i+16, float(i+16), None, 0.9))

    # Commute 2 (similar to commute 1)
    commute2_base = torch.tensor([0.0, 0.9, 0.1, 0.0] + [0.0] * 28)
    for i in range(3):
        noise = torch.randn(32) * 0.1
        continuous_sequence.append((commute2_base + noise, i+21, float(i+21), None, 0.8))

    # Evening (distinct pattern)
    evening_base = torch.tensor([0.2, 0.0, 0.0, 0.8] + [0.0] * 28)
    for i in range(4):
        noise = torch.randn(32) * 0.1
        continuous_sequence.append((evening_base + noise, i+24, float(i+24), None, 0.6))

    root = memory.store_episode(continuous_sequence)

    # Analyze discovered segments
    def analyze_segmentation(node, level=0):
        segments = []
        if node.children:
            for child in node.children:
                segments.append({
                    'level': child.level,
                    'timespan': child.timespan,
                    'spatial_extent': child.spatial_extent,
                    'duration': child.timespan[1] - child.timespan[0],
                    'size': child.spatial_extent[1] - child.spatial_extent[0] + 1
                })
        return segments

    # Collect segments at each level
    all_segments = {}

    def collect_segments(node):
        if node.level not in all_segments:
            all_segments[node.level] = []

        if node.children:
            for child in node.children:
                all_segments[node.level].append({
                    'timespan': child.timespan,
                    'spatial_extent': child.spatial_extent,
                    'duration': child.timespan[1] - child.timespan[0] + 1,
                    'children_count': len(child.children)
                })
                collect_segments(child)

    if root:
        collect_segments(root)

    # Visualize segmentation
    plt.figure(figsize=(15, 10))

    # Plot 1: Timeline with discovered segments
    plt.subplot(2, 2, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']

    # Ground truth segments (what we expect)
    true_segments = [
        (0, 4, "Morning"),
        (5, 7, "Commute1"), 
        (8, 13, "Work AM"),
        (14, 15, "Lunch"),
        (16, 20, "Work PM"),
        (21, 23, "Commute2"),
        (24, 27, "Evening")
    ]

    for i, (start, end, label) in enumerate(true_segments):
        plt.barh(0, end-start+1, left=start, height=0.3, 
                color=colors[i % len(colors)], alpha=0.7, label=f"True: {label}")

    plt.xlabel('Time')
    plt.title('Ground Truth Segmentation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: Discovered segments at level 1
    plt.subplot(2, 2, 2)
    if 1 in all_segments:
        for i, seg in enumerate(all_segments[1]):
            start_time = seg['timespan'][0]
            duration = seg['duration']
            plt.barh(0, duration, left=start_time, height=0.3,
                    color=colors[i % len(colors)], alpha=0.7)

    plt.xlabel('Time')
    plt.title('Discovered Segments (Level 1)')

    # Plot 3: Segment size distribution
    plt.subplot(2, 2, 3)
    level_sizes = {}
    for level, segments in all_segments.items():
        sizes = [seg['duration'] for seg in segments]
        level_sizes[level] = sizes

    for level, sizes in level_sizes.items():
        plt.hist(sizes, alpha=0.6, label=f'Level {level}', bins=5)

    plt.xlabel('Segment Duration')
    plt.ylabel('Count')
    plt.title('Segment Size Distribution')
    plt.legend()

    # Plot 4: Hierarchy depth
    plt.subplot(2, 2, 4)
    level_counts = [len(segments) for level, segments in all_segments.items()]
    levels = list(all_segments.keys())

    plt.bar(levels, level_counts, color='skyblue')
    plt.xlabel('Hierarchy Level')
    plt.ylabel('Number of Segments')
    plt.title('Segments per Level')

    plt.tight_layout()
    plt.savefig("temporal_chunking_segmentation.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print analysis
    print(f"Discovered {len(all_segments)} hierarchy levels")
    for level, segments in all_segments.items():
        avg_duration = np.mean([seg['duration'] for seg in segments])
        print(f"Level {level}: {len(segments)} segments, avg duration: {avg_duration:.1f}")

    print("✓ Temporal chunking and segmentation test completed")


def test_cross_level_interference():
    """Test how interference at different levels affects retrieval."""
    print("\n=== Testing Cross-Level Interference ===")

    memory = HierarchicalSpatioTemporalMemory(dim=32, max_levels=3)

    # Create original episode
    original_episode = [
        (torch.tensor([1.0, 0.0] + [0.0] * 30), 0, 0.0, None, 1.0),  # Concept A
        (torch.tensor([0.0, 1.0] + [0.0] * 30), 1, 1.0, None, 1.0),  # Concept B
        (torch.tensor([1.0, 1.0] + [0.0] * 30), 2, 2.0, None, 1.0),  # A+B combination
    ]

    root1 = memory.store_episode(original_episode)

    # Test baseline retrieval
    query_A = torch.tensor([1.0, 0.0] + [0.0] * 30)
    query_B = torch.tensor([0.0, 1.0] + [0.0] * 30)
    query_AB = torch.tensor([1.0, 1.0] + [0.0] * 30)

    baseline_results = {}
    for level in range(3):
        baseline_results[level] = {}

        ret_A, nodes_A = memory.retrieve_hierarchical(query_A, 0, 0.5, level=level)
        ret_B, nodes_B = memory.retrieve_hierarchical(query_B, 1, 1.5, level=level)
        ret_AB, nodes_AB = memory.retrieve_hierarchical(query_AB, 2, 2.5, level=level)

        baseline_results[level]['A'] = F.cosine_similarity(query_A.unsqueeze(0), ret_A.unsqueeze(0)).item() if ret_A.norm() > 0 else 0
        baseline_results[level]['B'] = F.cosine_similarity(query_B.unsqueeze(0), ret_B.unsqueeze(0)).item() if ret_B.norm() > 0 else 0
        baseline_results[level]['AB'] = F.cosine_similarity(query_AB.unsqueeze(0), ret_AB.unsqueeze(0)).item() if ret_AB.norm() > 0 else 0

    # Create interfering episodes at different levels
    interference_types = [
        ("Atomic", [
            (torch.tensor([1.0, 0.0] + [0.1] * 30), 10, 10.0, None, 1.0),  # Similar to A
            (torch.tensor([0.0, 1.0] + [0.1] * 30), 11, 11.0, None, 1.0),  # Similar to B
        ]),
        ("Component", [
            (torch.tensor([0.8, 0.2] + [0.0] * 30), 20, 20.0, None, 1.0),  # Mixed A+B
            (torch.tensor([0.2, 0.8] + [0.0] * 30), 21, 21.0, None, 1.0),  # Mixed B+A
            (torch.tensor([0.5, 0.5] + [0.0] * 30), 22, 22.0, None, 1.0),  # Balanced mix
        ]),
        ("Episode", [
            (torch.tensor([1.0, 0.0] + [0.2] * 30), 30, 30.0, None, 1.0),  # A variant
            (torch.tensor([0.0, 1.0] + [0.2] * 30), 31, 31.0, None, 1.0),  # B variant
            (torch.tensor([1.0, 1.0] + [0.2] * 30), 32, 32.0, None, 1.0),  # AB variant
            (torch.tensor([0.5, 0.5] + [0.2] * 30), 33, 33.0, None, 1.0),  # New combination
        ])
    ]

    interference_results = {}

    for interference_name, interference_episode in interference_types:
        # Add interfering episode
        memory.store_episode(interference_episode)

        # Test retrieval after interference
        interference_results[interference_name] = {}

        for level in range(3):
            interference_results[interference_name][level] = {}

            ret_A, _ = memory.retrieve_hierarchical(query_A, 0, 0.5, level=level)
            ret_B, _ = memory.retrieve_hierarchical(query_B, 1, 1.5, level=level)
            ret_AB, _ = memory.retrieve_hierarchical(query_AB, 2, 2.5, level=level)

            interference_results[interference_name][level]['A'] = F.cosine_similarity(query_A.unsqueeze(0), ret_A.unsqueeze(0)).item() if ret_A.norm() > 0 else 0
            interference_results[interference_name][level]['B'] = F.cosine_similarity(query_B.unsqueeze(0), ret_B.unsqueeze(0)).item() if ret_B.norm() > 0 else 0
            interference_results[interference_name][level]['AB'] = F.cosine_similarity(query_AB.unsqueeze(0), ret_AB.unsqueeze(0)).item() if ret_AB.norm() > 0 else 0

    # Visualize interference effects
    plt.figure(figsize=(15, 10))

    concepts = ['A', 'B', 'AB']
    interference_names = list(interference_results.keys())

    for i, concept in enumerate(concepts):
        plt.subplot(2, 2, i + 1)

        # Baseline performance
        baseline_vals = [baseline_results[level][concept] for level in range(3)]

        # Performance after each type of interference
        interference_vals = {}
        for interference_name in interference_names:
            interference_vals[interference_name] = [
                interference_results[interference_name][level][concept] for level in range(3)
            ]

        x = np.arange(3)
        width = 0.2

        plt.bar(x - width, baseline_vals, width, label='Baseline', color='gray', alpha=0.7)

        colors = ['red', 'green', 'blue']
        for j, interference_name in enumerate(interference_names):
            plt.bar(x + j * width, interference_vals[interference_name], width, 
                   label=f'{interference_name} Interference', color=colors[j], alpha=0.7)

        plt.xlabel('Hierarchy Level')
        plt.ylabel('Retrieval Similarity')
        plt.title(f'Concept {concept} - Interference Effects')
        plt.xticks(x, ['Atomic', 'Component', 'Episode'])
        plt.legend()
        plt.ylim(0, 1)

    # Overall interference summary
    plt.subplot(2, 2, 4)

    # Calculate average interference effect per level
    avg_interference = {}
    for level in range(3):
        level_effects = []
        for interference_name in interference_names:
            for concept in concepts:
                baseline_val = baseline_results[level][concept]
                interference_val = interference_results[interference_name][level][concept]
                effect = abs(baseline_val - interference_val)
                level_effects.append(effect)
        avg_interference[level] = np.mean(level_effects)

    levels = list(avg_interference.keys())
    effects = list(avg_interference.values())

    plt.bar(levels, effects, color='orange', alpha=0.7)
    plt.xlabel('Hierarchy Level')
    plt.ylabel('Average Interference Effect')
    plt.title('Cross-Level Interference Summary')
    plt.xticks(levels, ['Atomic', 'Component', 'Episode'])

    plt.tight_layout()
    plt.savefig("cross_level_interference.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✓ Cross-level interference test completed")


def test_compositional_memory():
    """Test ability to compose separate experiences into coherent wholes."""
    print("\n=== Testing Compositional Memory ===")

    memory = HierarchicalSpatioTemporalMemory(dim=32, max_levels=4)

    # Store separate components that should compose into larger concepts

    # Component 1: "Einstein" concept
    einstein_components = [
        (torch.tensor([1.0, 0.0, 0.0, 0.0] + [0.0] * 28), 0, 0.0, None, 1.0),  # Person
        (torch.tensor([0.0, 1.0, 0.0, 0.0] + [0.0] * 28), 1, 1.0, None, 0.9),  # Scientist
        (torch.tensor([0.0, 0.0, 1.0, 0.0] + [0.0] * 28), 2, 2.0, None, 0.8),  # Genius
        (torch.tensor([0.0, 0.0, 0.0, 1.0] + [0.0] * 28), 3, 3.0, None, 0.7),  # German
    ]

    # Component 2: "Equation" concept  
    equation_components = [
        (torch.tensor([0.0] * 4 + [1.0, 0.0, 0.0, 0.0] + [0.0] * 24), 10, 10.0, None, 1.0),  # Formula
        (torch.tensor([0.0] * 4 + [0.0, 1.0, 0.0, 0.0] + [0.0] * 24), 11, 11.0, None, 0.9),  # Mathematics
        (torch.tensor([0.0] * 4 + [0.0, 0.0, 1.0, 0.0] + [0.0] * 24), 12, 12.0, None, 0.8),  # Energy
        (torch.tensor([0.0] * 4 + [0.0, 0.0, 0.0, 1.0] + [0.0] * 24), 13, 13.0, None, 0.7),  # Mass
    ]

    # Component 3: "Relativity" concept
    relativity_components = [
        (torch.tensor([0.0] * 8 + [1.0, 0.0, 0.0, 0.0] + [0.0] * 20), 20, 20.0, None, 1.0),  # Theory
        (torch.tensor([0.0] * 8 + [0.0, 1.0, 0.0, 0.0] + [0.0] * 20), 21, 21.0, None, 0.9),  # Physics
        (torch.tensor([0.0] * 8 + [0.0, 0.0, 1.0, 0.0] + [0.0] * 20), 22, 22.0, None, 0.8),  # Space
        (torch.tensor([0.0] * 8 + [0.0, 0.0, 0.0, 1.0] + [0.0] * 20), 23, 23.0, None, 0.7),  # Time
    ]

    # Store components separately
    root1 = memory.store_episode(einstein_components)
    root2 = memory.store_episode(equation_components)  
    root3 = memory.store_episode(relativity_components)

    # Now store a composed episode that combines all three
    composed_episode = [
        # Einstein's equation in relativity context
        (torch.tensor([0.5, 0.5, 0.0, 0.0] + [0.5, 0.5, 0.0, 0.0] + [0.5, 0.5, 0.0, 0.0] + [0.0] * 20), 
         30, 30.0, None, 1.0),  # Einstein + Equation
        (torch.tensor([0.3, 0.3, 0.0, 0.0] + [0.3, 0.3, 0.0, 0.0] + [0.3, 0.3, 0.0, 0.0] + [0.0] * 20), 
         31, 31.0, None, 1.0),  # Full composition
        (torch.tensor([0.2, 0.2, 0.0, 0.0] + [0.2, 0.2, 0.0, 0.0] + [0.2, 0.2, 0.2, 0.2] + [0.0] * 20), 
         32, 32.0, None, 1.0),  # Complete theory
    ]

    root4 = memory.store_episode(composed_episode)

    # Test compositional queries
    test_queries = [
        ("Einstein only", torch.tensor([1.0, 0.5, 0.0, 0.0] + [0.0] * 28)),
        ("Equation only", torch.tensor([0.0] * 4 + [1.0, 0.5, 0.0, 0.0] + [0.0] * 24)),
        ("Relativity only", torch.tensor([0.0] * 8 + [1.0, 0.5, 0.0, 0.0] + [0.0] * 20)),
        ("Einstein + Equation", torch.tensor([0.5, 0.3, 0.0, 0.0] + [0.5, 0.3, 0.0, 0.0] + [0.0] * 24)),
        ("Einstein + Relativity", torch.tensor([0.5, 0.3, 0.0, 0.0] + [0.0] * 4 + [0.5, 0.3, 0.0, 0.0] + [0.0] * 20)),
        ("Equation + Relativity", torch.tensor([0.0] * 4 + [0.5, 0.3, 0.0, 0.0] + [0.5, 0.3, 0.0, 0.0] + [0.0] * 20)),
        ("Full composition", torch.tensor([0.3, 0.2, 0.0, 0.0] + [0.3, 0.2, 0.0, 0.0] + [0.3, 0.2, 0.0, 0.0] + [0.0] * 20)),
    ]

    # Test retrieval at different levels
    plt.figure(figsize=(15, 12))

    for i, (query_name, query_vec) in enumerate(test_queries):
        plt.subplot(3, 3, i + 1)

        level_similarities = []
        level_compositions = []  # Measure how well it composes vs. retrieves parts

        for level in range(4):
            # Query at center of composition space
            retrieved, nodes = memory.retrieve_hierarchical(query_vec, 31, 31.0, level=level)

            if retrieved.norm() > 0:
                similarity = F.cosine_similarity(query_vec.unsqueeze(0), retrieved.unsqueeze(0)).item()
            else:
                similarity = 0.0

            level_similarities.append(similarity)

            # Measure composition: how many different episodes contribute?
            episode_sources = set()
            for node in nodes:
                # Determine which original episode this came from based on spatial location
                if node.spatial_extent[0] < 10:
                    episode_sources.add("Einstein")
                elif node.spatial_extent[0] < 20:
                    episode_sources.add("Equation")
                elif node.spatial_extent[0] < 30:
                    episode_sources.add("Relativity")
                else:
                    episode_sources.add("Composed")

            composition_score = len(episode_sources) / 4.0  # Normalize by max possible sources
            level_compositions.append(composition_score)

        # Plot similarity and composition scores
        x = np.arange(4)
        width = 0.35

        bars1 = plt.bar(x - width/2, level_similarities, width, label='Similarity', alpha=0.7, color='blue')
        bars2 = plt.bar(x + width/2, level_compositions, width, label='Composition', alpha=0.7, color='red')

        plt.xlabel('Hierarchy Level')
        plt.ylabel('Score')
        plt.title(f'{query_name}')
        plt.xticks(x, ['Atomic', 'Scene', 'Segment', 'Episode'])
        plt.legend()
        plt.ylim(0, 1)

    # Summary plot
    plt.subplot(3, 3, 8)

    # Average composition ability per level
    avg_composition_per_level = []
    for level in range(4):
        level_scores = []
        for query_name, query_vec in test_queries:
            retrieved, nodes = memory.retrieve_hierarchical(query_vec, 31, 31.0, level=level)

            episode_sources = set()
            for node in nodes:
                if node.spatial_extent[0] < 10:
                    episode_sources.add("Einstein")
                elif node.spatial_extent[0] < 20:
                    episode_sources.add("Equation")
                elif node.spatial_extent[0] < 30:
                    episode_sources.add("Relativity")
                else:
                    episode_sources.add("Composed")

            composition_score = len(episode_sources) / 4.0
            level_scores.append(composition_score)

        avg_composition_per_level.append(np.mean(level_scores))

    plt.bar(range(4), avg_composition_per_level, color='green', alpha=0.7)
    plt.xlabel('Hierarchy Level')
    plt.ylabel('Average Composition Score')
    plt.title('Compositional Ability by Level')
    plt.xticks(range(4), ['Atomic', 'Scene', 'Segment', 'Episode'])

    plt.tight_layout()
    plt.savefig("compositional_memory.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✓ Compositional memory test completed")


def test_memory_consolidation_simulation():
    """Test memory consolidation: important memories promoted, details fade."""
    print("\n=== Testing Memory Consolidation Simulation ===")

    memory = HierarchicalSpatioTemporalMemory(dim=32, max_levels=4)

    # Create initial detailed episode with varying importance
    detailed_episode = []

    # High importance: Key moments
    key_moments = [
        (torch.randn(32), 0, 0.0, None, 1.0),   # Very important
        (torch.randn(32), 5, 5.0, None, 0.9),   # Important  
        (torch.randn(32), 10, 10.0, None, 1.0), # Very important
    ]

    # Medium importance: Supporting details
    supporting_details = [
        (torch.randn(32), 1, 1.0, None, 0.6),
        (torch.randn(32), 2, 2.0, None, 0.5),
        (torch.randn(32), 6, 6.0, None, 0.6),
        (torch.randn(32), 7, 7.0, None, 0.5),
        (torch.randn(32), 11, 11.0, None, 0.6),
    ]

    # Low importance: Minor details
    minor_details = [
        (torch.randn(32), 3, 3.0, None, 0.3),
        (torch.randn(32), 4, 4.0, None, 0.2),
        (torch.randn(32), 8, 8.0, None, 0.3),
        (torch.randn(32), 9, 9.0, None, 0.2),
        (torch.randn(32), 12, 12.0, None, 0.3),
        (torch.randn(32), 13, 13.0, None, 0.2),
    ]

    detailed_episode.extend(key_moments + supporting_details + minor_details)

    # Store initial episode
    root = memory.store_episode(detailed_episode)

    # Function to simulate consolidation by importance-based filtering
    def simulate_consolidation(memory_tree, importance_threshold):
        """Simulate consolidation by filtering out low-importance memories."""
        consolidated_memories = []

        def collect_important_memories(node):
            if node.importance >= importance_threshold:
                # Convert back to atomic memory format for re-storage
                # Use center of spatial/temporal extent as position
                spatial_pos = (node.spatial_extent[0] + node.spatial_extent[1]) // 2
                temporal_pos = (node.timespan[0] + node.timespan[1]) / 2
                consolidated_memories.append((
                    node.content, spatial_pos, temporal_pos, node.context, node.importance
                ))

            for child in node.children:
                collect_important_memories(child)

        if memory_tree:
            collect_important_memories(memory_tree)

        return consolidated_memories

    # Simulate consolidation at different time points (increasing thresholds)
    consolidation_stages = [
        ("Immediate", 0.0),    # All memories
        ("1 day", 0.3),        # Filter very low importance
        ("1 week", 0.5),       # Filter low importance  
        ("1 month", 0.7),      # Only medium+ importance
        ("1 year", 0.9),       # Only high importance
    ]

    consolidation_results = {}

    for stage_name, threshold in consolidation_stages:
        consolidated = simulate_consolidation(root, threshold)

        # Re-create memory with consolidated memories
        temp_memory = HierarchicalSpatioTemporalMemory(dim=32, max_levels=4)
        if consolidated:
            consolidated_root = temp_memory.store_episode(consolidated)

            # Analyze consolidated memory
            def analyze_memory_structure(node):
                structure = {
                    'total_nodes': 0,
                    'nodes_by_level': {},
                    'avg_importance': 0,
                    'importance_by_level': {}
                }

                importances = []

                def traverse(n):
                    structure['total_nodes'] += 1
                    if n.level not in structure['nodes_by_level']:
                        structure['nodes_by_level'][n.level] = 0
                        structure['importance_by_level'][n.level] = []

                    structure['nodes_by_level'][n.level] += 1
                    structure['importance_by_level'][n.level].append(n.importance)
                    importances.append(n.importance)

                    for child in n.children:
                        traverse(child)

                if node:
                    traverse(node)
                    structure['avg_importance'] = np.mean(importances) if importances else 0

                    # Average importance by level
                    for level in structure['importance_by_level']:
                        structure['importance_by_level'][level] = np.mean(structure['importance_by_level'][level])

                return structure

            analysis = analyze_memory_structure(consolidated_root)
            consolidation_results[stage_name] = analysis
        else:
            consolidation_results[stage_name] = {
                'total_nodes': 0,
                'nodes_by_level': {},
                'avg_importance': 0,
                'importance_by_level': {}
            }

    # Visualize consolidation process
    plt.figure(figsize=(15, 10))

    # Plot 1: Total memory nodes over time
    plt.subplot(2, 3, 1)
    stages = list(consolidation_results.keys())
    total_nodes = [consolidation_results[stage]['total_nodes'] for stage in stages]

    plt.plot(range(len(stages)), total_nodes, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Consolidation Stage')
    plt.ylabel('Total Memory Nodes')
    plt.title('Memory Decay Over Time')
    plt.xticks(range(len(stages)), stages, rotation=45)

    # Plot 2: Average importance over time
    plt.subplot(2, 3, 2)
    avg_importances = [consolidation_results[stage]['avg_importance'] for stage in stages]

    plt.plot(range(len(stages)), avg_importances, 'o-', color='red', linewidth=2, markersize=8)
    plt.xlabel('Consolidation Stage')
    plt.ylabel('Average Importance')
    plt.title('Importance Concentration')
    plt.xticks(range(len(stages)), stages, rotation=45)

    # Plot 3: Memory distribution by level over time
    plt.subplot(2, 3, 3)

    max_level = max([max(result['nodes_by_level'].keys()) if result['nodes_by_level'] else 0 
                     for result in consolidation_results.values()])

    for level in range(max_level + 1):
        level_counts = []
        for stage in stages:
            count = consolidation_results[stage]['nodes_by_level'].get(level, 0)
            level_counts.append(count)

        plt.plot(range(len(stages)), level_counts, 'o-', label=f'Level {level}', linewidth=2, markersize=6)

    plt.xlabel('Consolidation Stage')
    plt.ylabel('Nodes per Level')
    plt.title('Hierarchical Distribution Over Time')
    plt.xticks(range(len(stages)), stages, rotation=45)
    plt.legend()

    # Plot 4: Importance by level heatmap
    plt.subplot(2, 3, 4)

    # Create importance matrix
    importance_matrix = []
    level_labels = []

    for stage in stages:
        stage_importances = []
        for level in range(max_level + 1):
            importance = consolidation_results[stage]['importance_by_level'].get(level, 0)
            stage_importances.append(importance)
        importance_matrix.append(stage_importances)

    importance_matrix = np.array(importance_matrix)

    if importance_matrix.size > 0:
        im = plt.imshow(importance_matrix, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, label='Average Importance')
        plt.xlabel('Hierarchy Level')
        plt.ylabel('Consolidation Stage')
        plt.title('Importance by Level Over Time')
        plt.xticks(range(max_level + 1), [f'L{i}' for i in range(max_level + 1)])
        plt.yticks(range(len(stages)), stages)

    # Plot 5: Memory efficiency (importance per node)
    plt.subplot(2, 3, 5)

    efficiency = []
    for stage in stages:
        total = consolidation_results[stage]['total_nodes']
        avg_imp = consolidation_results[stage]['avg_importance']
        eff = avg_imp * total if total > 0 else 0  # Total importance preserved
        efficiency.append(eff)

    plt.bar(range(len(stages)), efficiency, color='green', alpha=0.7)
    plt.xlabel('Consolidation Stage')
    plt.ylabel('Total Importance Preserved')
    plt.title('Memory Efficiency')
    plt.xticks(range(len(stages)), stages, rotation=45)

    # Plot 6: Retrieval test over consolidation stages
    plt.subplot(2, 3, 6)

    # Test retrieval of key moments vs details across consolidation stages
    key_query = key_moments[0][0]  # Query for a key moment
    detail_query = minor_details[0][0]  # Query for a minor detail

    key_retrievals = []
    detail_retrievals = []

    for stage_name, threshold in consolidation_stages:
        consolidated = simulate_consolidation(root, threshold)
        temp_memory = HierarchicalSpatioTemporalMemory(dim=32, max_levels=4)

        if consolidated:
            temp_memory.store_episode(consolidated)

            # Test key moment retrieval
            key_ret, key_nodes = temp_memory.retrieve_hierarchical(key_query, 0, 0.0)
            key_sim = F.cosine_similarity(key_query.unsqueeze(0), key_ret.unsqueeze(0)).item() if key_ret.norm() > 0 else 0
            key_retrievals.append(key_sim)

            # Test detail retrieval
            detail_ret, detail_nodes = temp_memory.retrieve_hierarchical(detail_query, 3, 3.0)
            detail_sim = F.cosine_similarity(detail_query.unsqueeze(0), detail_ret.unsqueeze(0)).item() if detail_ret.norm() > 0 else 0
            detail_retrievals.append(detail_sim)
        else:
            key_retrievals.append(0)
            detail_retrievals.append(0)

    plt.plot(range(len(stages)), key_retrievals, 'o-', label='Key Moments', linewidth=2, color='blue')
    plt.plot(range(len(stages)), detail_retrievals, 'o-', label='Minor Details', linewidth=2, color='orange')
    plt.xlabel('Consolidation Stage')
    plt.ylabel('Retrieval Similarity')
    plt.title('Selective Memory Preservation')
    plt.xticks(range(len(stages)), stages, rotation=45)
    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig("memory_consolidation_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print consolidation summary
    print("Memory Consolidation Analysis:")
    for stage in stages:
        result = consolidation_results[stage]
        print(f"{stage:>10}: {result['total_nodes']:>3} nodes, "
              f"avg importance: {result['avg_importance']:.3f}")

    print("✓ Memory consolidation simulation test completed")


def run_all_hierarchical_experiments():
    """Run all hierarchical memory experiments."""
    print("=== Running All Hierarchical Episodic Memory Experiments ===\n")

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    experiments = [
        test_multi_scale_temporal_retrieval,
        test_hierarchical_generalization,
        test_temporal_chunking_segmentation,
        test_cross_level_interference,
        test_compositional_memory,
        test_memory_consolidation_simulation
    ]

    for i, experiment in enumerate(experiments, 1):
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(experiments)}: {experiment.__name__}")
        print(f"{'='*60}")

        try:
            experiment()
            print(f"✅ {experiment.__name__} completed successfully")
        except Exception as e:
            print(f"❌ {experiment.__name__} failed: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All hierarchical experiments completed!")
    print(f"{'='*60}")


# Add this to the main execution block
if __name__ == "__main__":
    # Set random seeds for repeatability
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=== Testing Hierarchical Spatio-Temporal Episodic Memory ===\n")

    # Original tests
    # test_hierarchical_memory_structure()
    # test_hierarchical_retrieval_dynamics()

    # New comprehensive experiments
    run_all_hierarchical_experiments()

    print("\n=== All hierarchical memory tests completed! ===")
