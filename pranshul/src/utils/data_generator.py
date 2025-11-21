import random

import networkx as nx
import numpy as np


class DataGenerator:
    """
    Generates synthetic datasets for TSP, Vertex Cover, and Set Cover.
    Ref: Empirical protocol[cite: 46, 47].
    """

    @staticmethod
    def generate_euclidean_tsp(n, seed=None):
        """Generates random (x,y) coordinates in a 2D plane."""
        if seed:
            np.random.seed(seed)
        # Returns a distance matrix and the raw points
        points = np.random.rand(n, 2)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = np.linalg.norm(points[i] - points[j])
        return dist_matrix, points

    @staticmethod
    def generate_random_graph_vc(n, p=0.3, seed=None):
        """Generates an Erdos-Renyi graph for Vertex Cover."""
        return nx.erdos_renyi_graph(n, p, seed=seed)

    @staticmethod
    def generate_set_cover_instance(num_elements, num_sets, density=0.2, seed=None):
        """
        Generates a universe and a collection of subsets.
        Returns: universe (set), subsets (list of sets), costs (list of floats)
        """
        if seed:
            random.seed(seed)
        universe = set(range(num_elements))
        subsets = []
        for _ in range(num_sets):
            # Create random subset
            s = set(
                random.sample(list(universe), k=max(1, int(num_elements * density)))
            )
            subsets.append(s)

        # Ensure feasibility (simple fix: add singleton sets for uncovered elements)
        covered = set().union(*subsets)
        uncovered = universe - covered
        for elem in uncovered:
            subsets.append({elem})

        costs = [random.uniform(1, 10) for _ in range(len(subsets))]
        return universe, subsets, costs
