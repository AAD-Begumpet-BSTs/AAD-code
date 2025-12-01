import random
import networkx as nx
import numpy as np

class DataGenerator:
    """
    Generates synthetic datasets for TSP, Vertex Cover, and Set Cover.
    Optimized for large-scale generation.
    """

    @staticmethod
    def generate_euclidean_tsp(n, seed=None):
        """Generates random (x,y) coordinates in a 2D plane."""
        if seed is not None:
            np.random.seed(seed)
        
        points = np.random.rand(n, 2)
        # Fast vectorized distance matrix calculation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        
        return dist_matrix, points

    @staticmethod
    def generate_random_graph_vc(n, p=0.3, seed=None):
        """Generates an Erdos-Renyi graph for Vertex Cover."""
        return nx.erdos_renyi_graph(n, p, seed=seed)

    @staticmethod
    def generate_set_cover_instance(num_elements, num_sets, density=0.2, seed=None):
        """
        Generates a universe and a collection of subsets.
        
        OPTIMIZATION:
        Uses random.sample on a range() object to avoid O(N) list conversion 
        overhead inside the loop. This reduces complexity from O(M*N) to O(M*k).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        universe = set(range(num_elements))
        subsets = []
        costs = []
        
        # Pre-calculate set size (fixed density for consistency)
        # Ensure at least 1 element per set
        k = max(1, int(num_elements * density))
        
        # Use a range object for sampling (O(1) memory, O(k) sampling)
        population = range(num_elements)
        
        # Track coverage to ensure feasibility
        covered_elements = set()

        for _ in range(num_sets):
            # Fast sampling without replacement
            indices = random.sample(population, k)
            new_set = set(indices)
            
            subsets.append(new_set)
            costs.append(random.uniform(1.0, 10.0))
            
            # Update coverage (set update is fast)
            covered_elements.update(new_set)

        # --- Feasibility Repair ---
        # Identify elements not covered by the random generation
        missing = universe - covered_elements
        
        if missing:
            # Create singleton sets for missing elements to guarantee feasibility
            # This is standard practice to prevent the instance from being invalid
            for m in missing:
                subsets.append({m})
                costs.append(random.uniform(10.0, 20.0)) # Higher cost for repair sets

        return universe, subsets, costs