"""
Set Cover Instance Generator
Generates random Set Cover instances for empirical analysis.
"""

import random
import json
import os
from typing import Set, List, Tuple


def generate_set_cover_instance(num_elements: int, num_sets: int, 
                                density: float = 0.2, seed: int = None,
                                cost_range: Tuple[float, float] = (1.0, 10.0)) -> Tuple[Set[int], List[Set[int]], List[float]]:
    """
    Generate a random Set Cover instance.
    
    Args:
        num_elements: Size of the universe to cover
        num_sets: Number of subsets in the collection
        density: Approximate fraction of universe each set covers (on average)
        seed: Random seed for reproducibility
        cost_range: (min, max) for uniform random subset costs
        
    Returns:
        (universe, subsets, costs):
            - universe: Set of all elements {0, 1, ..., num_elements-1}
            - subsets: List of sets, each containing some elements
            - costs: List of costs for each subset
    """
    if seed is not None:
        random.seed(seed)
    
    universe = set(range(num_elements))
    subsets = []
    
    # Generate random subsets
    for _ in range(num_sets):
        # Each set covers approximately 'density' fraction of universe
        set_size = max(1, int(num_elements * density))
        # Add some randomness to set size
        set_size = max(1, int(set_size * random.uniform(0.5, 1.5)))
        set_size = min(set_size, num_elements)
        
        subset = set(random.sample(list(universe), k=set_size))
        subsets.append(subset)
    
    # Ensure all elements are coverable (feasibility)
    covered = set().union(*subsets)
    uncovered = universe - covered
    
    # Add singleton sets for uncovered elements
    for elem in uncovered:
        subsets.append({elem})
    
    # Generate random costs
    min_cost, max_cost = cost_range
    costs = [random.uniform(min_cost, max_cost) for _ in range(len(subsets))]
    
    return universe, subsets, costs


def generate_vertex_cover_as_set_cover(n: int, p: float = 0.3, seed: int = None,
                                      cost_range: Tuple[float, float] = (1.0, 10.0)) -> Tuple[Set[int], List[Set[int]], List[float]]:
    """
    Generate a Set Cover instance from a Vertex Cover problem.
    Each vertex becomes a set containing its incident edges.
    
    Args:
        n: Number of vertices in the graph
        p: Edge probability
        seed: Random seed
        cost_range: Vertex weight range
        
    Returns:
        (universe, subsets, costs): universe = edges, subsets = vertex neighborhoods
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate random edges
    edges = []
    edge_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                edges.append((u, v))
                edge_set.add((u, v))
    
    # Map each edge to a unique element ID
    edge_to_id = {edge: i for i, edge in enumerate(edges)}
    universe = set(range(len(edges)))
    
    # For each vertex, create a set of its incident edges
    subsets = []
    for v in range(n):
        incident_edges = set()
        for (u, w) in edges:
            if u == v or w == v:
                incident_edges.add(edge_to_id[(u, w)])
        subsets.append(incident_edges)
    
    # Generate random costs
    min_cost, max_cost = cost_range
    costs = [random.uniform(min_cost, max_cost) for _ in range(n)]
    
    return universe, subsets, costs


def generate_facility_location_as_set_cover(num_locations: int, num_clients: int,
                                            coverage_radius: int = 3, seed: int = None,
                                            cost_range: Tuple[float, float] = (5.0, 20.0)) -> Tuple[Set[int], List[Set[int]], List[float]]:
    """
    Generate a Set Cover instance modeling a facility location problem.
    Each facility (set) can cover clients within a certain radius.
    
    Args:
        num_locations: Number of potential facility locations
        num_clients: Number of clients to cover
        coverage_radius: Number of clients each facility can cover
        seed: Random seed
        cost_range: Facility opening cost range
        
    Returns:
        (universe, subsets, costs): universe = clients, subsets = facility coverages
    """
    if seed is not None:
        random.seed(seed)
    
    universe = set(range(num_clients))
    subsets = []
    
    for _ in range(num_locations):
        # Each facility covers a random subset of clients
        num_covered = random.randint(1, min(coverage_radius, num_clients))
        covered_clients = set(random.sample(list(universe), k=num_covered))
        subsets.append(covered_clients)
    
    # Ensure feasibility
    covered = set().union(*subsets)
    uncovered = universe - covered
    for elem in uncovered:
        subsets.append({elem})
    
    # Facility costs (generally higher than element-wise costs)
    min_cost, max_cost = cost_range
    costs = [random.uniform(min_cost, max_cost) for _ in range(len(subsets))]
    
    return universe, subsets, costs


def save_sc_instance(filename: str, universe: Set[int], subsets: List[Set[int]], costs: List[float]):
    """
    Save a Set Cover instance to a JSON file.
    
    Args:
        filename: Output file path
        universe: Set of all elements
        subsets: List of subsets
        costs: List of subset costs
    """
    data = {
        'num_elements': len(universe),
        'num_sets': len(subsets),
        'universe': sorted(list(universe)),
        'subsets': [sorted(list(s)) for s in subsets],
        'costs': costs
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_sc_instance(filename: str) -> Tuple[Set[int], List[Set[int]], List[float]]:
    """
    Load a Set Cover instance from a JSON file.
    
    Args:
        filename: Input file path
        
    Returns:
        (universe, subsets, costs)
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    universe = set(data['universe'])
    subsets = [set(s) for s in data['subsets']]
    costs = data['costs']
    
    return universe, subsets, costs


def generate_sc_benchmark_suite(output_dir: str = 'datasets/sc'):
    """
    Generate a comprehensive benchmark suite of Set Cover instances.
    Covers various sizes and densities for both small n and large n experiments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Small instances - for comparison with exact solver
    # (num_elements, num_sets, density)
    small_configs = [
        (20, 10, 0.3),   # sparse
        (30, 15, 0.25),  # medium
        (40, 20, 0.2),   # larger
    ]
    seeds_small = range(100, 105)  # 5 instances per config
    
    # Large instances - for scalability testing
    large_configs = [
        (100, 40, 0.2),
        (200, 60, 0.15),
        (300, 80, 0.15),
        (500, 100, 0.1),
    ]
    seeds_large = range(1000, 1003)  # 3 instances per config
    
    print("Generating Set Cover benchmark suite...")
    print(f"Output directory: {output_dir}")
    
    # Generate small instances
    print("\nSmall instances (for exact solver comparison):")
    for (n_elem, n_sets, density) in small_configs:
        for seed in seeds_small:
            universe, subsets, costs = generate_set_cover_instance(n_elem, n_sets, density, seed=seed)
            filename = os.path.join(output_dir, f"sc_e{n_elem}_s{n_sets}_d{density:.2f}_seed{seed}.json")
            save_sc_instance(filename, universe, subsets, costs)
        print(f"  elements={n_elem}, sets={n_sets}, density={density}: {len(list(seeds_small))} instances")
    
    # Generate large instances
    print("\nLarge instances (for scalability testing):")
    for (n_elem, n_sets, density) in large_configs:
        for seed in seeds_large:
            universe, subsets, costs = generate_set_cover_instance(n_elem, n_sets, density, seed=seed)
            filename = os.path.join(output_dir, f"sc_e{n_elem}_s{n_sets}_d{density:.2f}_seed{seed}.json")
            save_sc_instance(filename, universe, subsets, costs)
        print(f"  elements={n_elem}, sets={n_sets}, density={density}: {len(list(seeds_large))} instances")
    
    total = len(small_configs) * len(list(seeds_small)) + len(large_configs) * len(list(seeds_large))
    print(f"\n✓ Generated {total} Set Cover instances")


if __name__ == "__main__":
    # Generate benchmark suite
    generate_sc_benchmark_suite()
    
    # Test individual generators
    print("\n" + "="*60)
    print("Testing instance generators...")
    
    # Random instance
    universe, subsets, costs = generate_set_cover_instance(50, 20, density=0.2, seed=42)
    print(f"\nRandom Set Cover: {len(universe)} elements, {len(subsets)} sets")
    print(f"  Coverage: {sum(len(s) for s in subsets) / len(universe):.2f}x redundancy")
    print(f"  Sample costs: {costs[:3]}")
    
    # Vertex Cover as Set Cover
    universe, subsets, costs = generate_vertex_cover_as_set_cover(20, p=0.3, seed=42)
    print(f"\nVertex Cover as Set Cover: {len(universe)} edges, {len(subsets)} vertices")
    
    # Facility Location
    universe, subsets, costs = generate_facility_location_as_set_cover(30, 50, seed=42)
    print(f"\nFacility Location: {len(universe)} clients, {len(subsets)} facilities")
