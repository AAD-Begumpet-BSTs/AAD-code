"""
Vertex Cover Instance Generator
Generates random weighted graph instances for Vertex Cover experiments.
"""

import sys
import os
# Add parent directories to path to import from VC module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'VC'))

import random
import json
import networkx as nx
from typing import Dict, Tuple, List
from src.graph import Graph


def generate_random_graph_vc(n: int, p: float = 0.3, seed: int = None, 
                             weight_range: Tuple[float, float] = (1.0, 10.0)) -> Tuple[Graph, Dict[int, float]]:
    """
    Generate a random Erdős-Rényi graph for Vertex Cover with random vertex weights.
    
    Args:
        n: Number of vertices
        p: Edge probability (density)
        seed: Random seed for reproducibility
        weight_range: (min, max) for uniform random vertex weights
        
    Returns:
        (graph, weights): Graph instance and dictionary of vertex weights
    """
    if seed is not None:
        random.seed(seed)
        nx_seed = seed
    else:
        nx_seed = None
    
    # Generate Erdős-Rényi random graph
    G_nx = nx.erdos_renyi_graph(n, p, seed=nx_seed)
    
    # Convert to custom Graph format
    G = Graph(n)
    for (u, v) in G_nx.edges():
        G.add_edge(u, v)
    
    # Generate random vertex weights
    min_w, max_w = weight_range
    weights = {v: random.uniform(min_w, max_w) for v in range(n)}
    
    return G, weights


def generate_barabasi_albert_vc(n: int, m: int = 2, seed: int = None,
                                weight_range: Tuple[float, float] = (1.0, 10.0)) -> Tuple[Graph, Dict[int, float]]:
    """
    Generate a Barabási-Albert scale-free graph for Vertex Cover.
    These graphs have power-law degree distributions (realistic for many networks).
    
    Args:
        n: Number of vertices
        m: Number of edges to attach from a new node to existing nodes
        seed: Random seed for reproducibility
        weight_range: (min, max) for uniform random vertex weights
        
    Returns:
        (graph, weights): Graph instance and dictionary of vertex weights
    """
    if seed is not None:
        random.seed(seed)
        nx_seed = seed
    else:
        nx_seed = None
    
    # Generate Barabási-Albert graph
    G_nx = nx.barabasi_albert_graph(n, m, seed=nx_seed)
    
    # Convert to custom Graph format
    G = Graph(n)
    for (u, v) in G_nx.edges():
        G.add_edge(u, v)
    
    # Generate random vertex weights
    min_w, max_w = weight_range
    weights = {v: random.uniform(min_w, max_w) for v in range(n)}
    
    return G, weights


def generate_geometric_graph_vc(n: int, radius: float = 0.3, seed: int = None,
                                weight_range: Tuple[float, float] = (1.0, 10.0)) -> Tuple[Graph, Dict[int, float]]:
    """
    Generate a random geometric graph for Vertex Cover.
    Vertices are placed randomly in unit square, edges connect vertices within radius.
    
    Args:
        n: Number of vertices
        radius: Distance threshold for edge creation
        seed: Random seed for reproducibility
        weight_range: (min, max) for uniform random vertex weights
        
    Returns:
        (graph, weights): Graph instance and dictionary of vertex weights
    """
    if seed is not None:
        random.seed(seed)
        nx_seed = seed
    else:
        nx_seed = None
    
    # Generate random geometric graph
    G_nx = nx.random_geometric_graph(n, radius, seed=nx_seed)
    
    # Convert to custom Graph format
    G = Graph(n)
    for (u, v) in G_nx.edges():
        G.add_edge(u, v)
    
    # Generate random vertex weights
    min_w, max_w = weight_range
    weights = {v: random.uniform(min_w, max_w) for v in range(n)}
    
    return G, weights


def save_vc_instance(filename: str, G: Graph, weights: Dict[int, float]):
    """
    Save a Vertex Cover instance to a JSON file.
    
    Args:
        filename: Output file path
        G: Graph instance
        weights: Vertex weights dictionary
    """
    data = {
        'n': G.number_of_vertices(),
        'm': G.number_of_edges(),
        'edges': [[u, v] for (u, v) in G.edges],
        'weights': weights
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_vc_instance(filename: str) -> Tuple[Graph, Dict[int, float]]:
    """
    Load a Vertex Cover instance from a JSON file.
    
    Args:
        filename: Input file path
        
    Returns:
        (graph, weights): Graph instance and vertex weights
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    n = data['n']
    G = Graph(n)
    
    for (u, v) in data['edges']:
        G.add_edge(u, v)
    
    # Convert weights keys from string to int (JSON limitation)
    weights = {int(k): v for k, v in data['weights'].items()}
    
    return G, weights


def generate_vc_benchmark_suite(output_dir: str = 'datasets/vc'):
    """
    Generate a comprehensive benchmark suite of Vertex Cover instances.
    Covers various sizes and densities for both small n and large n experiments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Small instances (n <= 24) - for comparison with exact solver
    small_configs = [
        (12, 0.3),   # sparse
        (18, 0.35),  # medium
        (24, 0.4),   # denser
    ]
    seeds_small = range(100, 105)  # 5 instances per config
    
    # Large instances (n >= 50) - for scalability testing
    large_configs = [
        (50, 0.3),
        (75, 0.35),
        (100, 0.4),
        (150, 0.35),
        (200, 0.3),
    ]
    seeds_large = range(1000, 1003)  # 3 instances per config
    
    print("Generating Vertex Cover benchmark suite...")
    print(f"Output directory: {output_dir}")
    
    # Generate small instances
    print("\nSmall instances (for exact solver comparison):")
    for (n, p) in small_configs:
        for seed in seeds_small:
            G, weights = generate_random_graph_vc(n, p, seed=seed)
            filename = os.path.join(output_dir, f"vc_n{n}_p{p:.2f}_s{seed}.json")
            save_vc_instance(filename, G, weights)
        print(f"  n={n}, p={p}: {len(list(seeds_small))} instances ({G.number_of_edges()} avg edges)")
    
    # Generate large instances
    print("\nLarge instances (for scalability testing):")
    for (n, p) in large_configs:
        for seed in seeds_large:
            G, weights = generate_random_graph_vc(n, p, seed=seed)
            filename = os.path.join(output_dir, f"vc_n{n}_p{p:.2f}_s{seed}.json")
            save_vc_instance(filename, G, weights)
        print(f"  n={n}, p={p}: {len(list(seeds_large))} instances ({G.number_of_edges()} avg edges)")
    
    total = len(small_configs) * len(list(seeds_small)) + len(large_configs) * len(list(seeds_large))
    print(f"\n✓ Generated {total} Vertex Cover instances")


if __name__ == "__main__":
    # Generate benchmark suite
    generate_vc_benchmark_suite()
    
    # Test individual generators
    print("\n" + "="*60)
    print("Testing instance generators...")
    
    # Random graph
    G, weights = generate_random_graph_vc(20, p=0.3, seed=42)
    print(f"\nRandom graph (n=20, p=0.3): {G.number_of_vertices()} vertices, {G.number_of_edges()} edges")
    print(f"  Sample weights: {dict(list(weights.items())[:3])}")
    
    # Barabási-Albert graph
    G, weights = generate_barabasi_albert_vc(30, m=3, seed=42)
    print(f"\nBarabási-Albert (n=30, m=3): {G.number_of_vertices()} vertices, {G.number_of_edges()} edges")
    
    # Geometric graph
    G, weights = generate_geometric_graph_vc(25, radius=0.3, seed=42)
    print(f"\nGeometric graph (n=25, r=0.3): {G.number_of_vertices()} vertices, {G.number_of_edges()} edges")
