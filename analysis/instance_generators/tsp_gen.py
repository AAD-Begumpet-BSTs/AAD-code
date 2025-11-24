"""
TSP Instance Generator
Generates random Euclidean TSP instances for empirical analysis.
"""

import numpy as np
from scipy.spatial import distance_matrix
from typing import Tuple, List
import json
import os


def generate_euclidean_tsp(n: int, seed: int = None, bounds: Tuple[float, float] = (0.0, 100.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random Euclidean TSP instance with n cities.
    
    Args:
        n: Number of cities
        seed: Random seed for reproducibility
        bounds: (min, max) coordinates for the 2D plane
        
    Returns:
        (points, dist_matrix): 
            - points: np.array of shape (n, 2) with (x, y) coordinates
            - dist_matrix: np.array of shape (n, n) with Euclidean distances
    """
    if seed is not None:
        np.random.seed(seed)
    
    min_bound, max_bound = bounds
    points = np.random.uniform(min_bound, max_bound, size=(n, 2))
    dist = distance_matrix(points, points)
    
    return points, dist


def generate_clustered_tsp(n: int, num_clusters: int = 3, seed: int = None, 
                          cluster_spread: float = 10.0, 
                          bounds: Tuple[float, float] = (0.0, 100.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a clustered Euclidean TSP instance (harder for some algorithms).
    
    Args:
        n: Number of cities
        num_clusters: Number of clusters to create
        seed: Random seed for reproducibility
        cluster_spread: Standard deviation of points within each cluster
        bounds: (min, max) coordinates for cluster centers
        
    Returns:
        (points, dist_matrix)
    """
    if seed is not None:
        np.random.seed(seed)
    
    min_bound, max_bound = bounds
    
    # Generate cluster centers
    centers = np.random.uniform(min_bound, max_bound, size=(num_clusters, 2))
    
    # Assign cities to clusters
    points_per_cluster = n // num_clusters
    remainder = n % num_clusters
    
    points = []
    for i, center in enumerate(centers):
        # Add extra point to first 'remainder' clusters
        num_points = points_per_cluster + (1 if i < remainder else 0)
        cluster_points = np.random.normal(center, cluster_spread, size=(num_points, 2))
        points.extend(cluster_points)
    
    points = np.array(points)
    dist = distance_matrix(points, points)
    
    return points, dist


def generate_grid_tsp(grid_size: int, seed: int = None, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a TSP instance on a grid with optional noise.
    
    Args:
        grid_size: Size of the grid (total cities = grid_size^2)
        seed: Random seed for reproducibility
        noise: Amount of random noise to add to grid positions (0 = perfect grid)
        
    Returns:
        (points, dist_matrix)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = grid_size * grid_size
    points = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = i + np.random.uniform(-noise, noise)
            y = j + np.random.uniform(-noise, noise)
            points.append([x, y])
    
    points = np.array(points)
    dist = distance_matrix(points, points)
    
    return points, dist


def save_tsp_instance(filename: str, points: np.ndarray, dist_matrix: np.ndarray = None):
    """
    Save a TSP instance to a JSON file.
    
    Args:
        filename: Output file path
        points: Array of city coordinates
        dist_matrix: Optional distance matrix (computed if not provided)
    """
    if dist_matrix is None:
        dist_matrix = distance_matrix(points, points)
    
    data = {
        'n': len(points),
        'points': points.tolist(),
        'distances': dist_matrix.tolist()
    }
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_tsp_instance(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a TSP instance from a JSON file.
    
    Args:
        filename: Input file path
        
    Returns:
        (points, dist_matrix)
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    points = np.array(data['points'])
    dist_matrix = np.array(data['distances'])
    
    return points, dist_matrix


def generate_tsp_benchmark_suite(output_dir: str = 'datasets/tsp'):
    """
    Generate a comprehensive benchmark suite of TSP instances.
    Covers various sizes for both small n (with exact solver) and large n experiments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Small instances (n <= 20) - for comparison with exact solver
    small_n_sizes = [5, 8, 10, 12, 15, 18, 20]
    seeds_small = range(100, 105)  # 5 instances per size
    
    # Large instances (n >= 30) - for scalability testing
    large_n_sizes = [30, 40, 50, 75, 100, 150, 200]
    seeds_large = range(1000, 1003)  # 3 instances per size
    
    print("Generating TSP benchmark suite...")
    print(f"Output directory: {output_dir}")
    
    # Generate small instances
    print("\nSmall instances (for exact solver comparison):")
    for n in small_n_sizes:
        for seed in seeds_small:
            points, dist = generate_euclidean_tsp(n, seed=seed)
            filename = os.path.join(output_dir, f"tsp_n{n}_s{seed}.json")
            save_tsp_instance(filename, points, dist)
        print(f"  n={n}: {len(list(seeds_small))} instances")
    
    # Generate large instances
    print("\nLarge instances (for scalability testing):")
    for n in large_n_sizes:
        for seed in seeds_large:
            points, dist = generate_euclidean_tsp(n, seed=seed)
            filename = os.path.join(output_dir, f"tsp_n{n}_s{seed}.json")
            save_tsp_instance(filename, points, dist)
        print(f"  n={n}: {len(list(seeds_large))} instances")
    
    print(f"\n✓ Generated {len(small_n_sizes) * len(list(seeds_small)) + len(large_n_sizes) * len(list(seeds_large))} TSP instances")


if __name__ == "__main__":
    # Generate benchmark suite
    generate_tsp_benchmark_suite()
    
    # Test individual generators
    print("\n" + "="*60)
    print("Testing instance generators...")
    
    # Random instance
    points, dist = generate_euclidean_tsp(10, seed=42)
    print(f"\nRandom TSP (n=10): {points.shape}, distances: {dist.shape}")
    
    # Clustered instance
    points, dist = generate_clustered_tsp(15, num_clusters=3, seed=42)
    print(f"Clustered TSP (n=15, 3 clusters): {points.shape}")
    
    # Grid instance
    points, dist = generate_grid_tsp(4, seed=42)
    print(f"Grid TSP (4x4 = 16 cities): {points.shape}")
