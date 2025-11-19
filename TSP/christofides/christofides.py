"""
Christofides Algorithm for Metric TSP
Achieves 1.5-approximation guarantee
"""

import numpy as np
import networkx as nx
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import time

class ChristofidesAlgorithm:
    def __init__(self, points):
        """
        Initialize with points in 2D Euclidean space
        points: np.array of shape (n, 2)
        """
        self.points = points
        self.n = len(points)
        self.dist_matrix = distance_matrix(points, points)
        
    def compute_mst(self):
        """
        Step 1: Compute Minimum Spanning Tree using Prim's algorithm
        Returns: MST as list of edges
        """
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i+1, self.n):
                G.add_edge(i, j, weight=self.dist_matrix[i, j])
        
        mst = nx.minimum_spanning_tree(G)
        self.mst = mst
        return list(mst.edges())
    
    def find_odd_degree_vertices(self):
        """
        Step 2: Find vertices with odd degree in MST
        Returns: list of vertices with odd degree
        """
        degrees = dict(self.mst.degree())
        odd_vertices = [v for v, d in degrees.items() if d % 2 == 1]
        return odd_vertices
    
    def minimum_weight_matching(self, odd_vertices):
        """
        Step 3: Find minimum weight perfect matching on odd degree vertices
        Uses greedy approach (optimal would use Blossom algorithm)
        Returns: list of matched edges
        """
        G_odd = nx.Graph()
        for i, u in enumerate(odd_vertices):
            for j, v in enumerate(odd_vertices):
                if i < j:
                    G_odd.add_edge(u, v, weight=self.dist_matrix[u, v])
        
        # NetworkX minimum weight matching
        matching = nx.min_weight_matching(G_odd)
        self.matching = matching
        return list(matching)
    
    def create_eulerian_circuit(self, mst_edges, matching_edges):
        """
        Step 4: Combine MST and matching to create Eulerian multigraph
        Returns: Eulerian circuit
        """
        G = nx.MultiGraph()
        G.add_edges_from(mst_edges)
        G.add_edges_from(matching_edges)
        
        # Verify it's Eulerian (all vertices have even degree)
        assert all(d % 2 == 0 for v, d in G.degree()), "Graph is not Eulerian!"
        
        # Find Eulerian circuit
        eulerian_circuit = list(nx.eulerian_circuit(G, source=0))
        return eulerian_circuit
    
    def shortcut_to_hamiltonian(self, eulerian_circuit):
        """
        Step 5: Convert Eulerian circuit to Hamiltonian by skipping repeated vertices
        Returns: Hamiltonian tour
        """
        visited = set()
        tour = []
        
        for u, v in eulerian_circuit:
            if u not in visited:
                tour.append(u)
                visited.add(u)
        
        # Add first vertex to complete the cycle
        if tour[-1] != tour[0]:
            tour.append(tour[0])
            
        return tour
    
    def tour_cost(self, tour):
        """Calculate total cost of a tour"""
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.dist_matrix[tour[i], tour[i+1]]
        return cost
    
    def run(self, verbose=True):
        """
        Execute complete Christofides algorithm
        Returns: tour and cost
        """
        start_time = time.time()
        
        # Step 1: MST
        if verbose: print("Step 1: Computing MST...")
        mst_edges = self.compute_mst()
        mst_cost = sum(self.dist_matrix[u, v] for u, v in mst_edges)
        if verbose: print(f"  MST cost: {mst_cost:.2f}")
        
        # Step 2: Odd degree vertices
        if verbose: print("Step 2: Finding odd degree vertices...")
        odd_vertices = self.find_odd_degree_vertices()
        if verbose: print(f"  Found {len(odd_vertices)} odd degree vertices")
        
        # Step 3: Minimum matching
        if verbose: print("Step 3: Computing minimum weight matching...")
        matching_edges = self.minimum_weight_matching(odd_vertices)
        matching_cost = sum(self.dist_matrix[u, v] for u, v in matching_edges)
        if verbose: print(f"  Matching cost: {matching_cost:.2f}")
        
        # Step 4: Eulerian circuit
        if verbose: print("Step 4: Creating Eulerian circuit...")
        eulerian = self.create_eulerian_circuit(mst_edges, matching_edges)
        
        # Step 5: Shortcut to Hamiltonian
        if verbose: print("Step 5: Shortcutting to Hamiltonian tour...")
        tour = self.shortcut_to_hamiltonian(eulerian)
        
        cost = self.tour_cost(tour)
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Christofides Algorithm Complete")
            print(f"  Tour cost: {cost:.2f}")
            print(f"  Runtime: {elapsed_time:.4f} seconds")
            print(f"  Theoretical bound: ≤ 1.5 × OPT")
        
        return {
            'tour': tour,
            'cost': cost,
            'mst_edges': mst_edges,
            'matching_edges': matching_edges,
            'runtime': elapsed_time
        }
    
    def visualize(self, result, title="Christofides Algorithm"):
        """Visualize the algorithm steps"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Original points
        ax = axes[0, 0]
        ax.scatter(self.points[:, 0], self.points[:, 1], c='red', s=100, zorder=5)
        for i, (x, y) in enumerate(self.points):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        ax.set_title("Input Points")
        ax.grid(True, alpha=0.3)
        
        # MST
        ax = axes[0, 1]
        ax.scatter(self.points[:, 0], self.points[:, 1], c='red', s=100, zorder=5)
        for u, v in result['mst_edges']:
            ax.plot([self.points[u, 0], self.points[v, 0]], 
                   [self.points[u, 1], self.points[v, 1]], 'g-', linewidth=2)
        ax.set_title("Step 1: Minimum Spanning Tree")
        ax.grid(True, alpha=0.3)
        
        # MST + Matching
        ax = axes[1, 0]
        ax.scatter(self.points[:, 0], self.points[:, 1], c='red', s=100, zorder=5)
        for u, v in result['mst_edges']:
            ax.plot([self.points[u, 0], self.points[v, 0]], 
                   [self.points[u, 1], self.points[v, 1]], 'g-', linewidth=2, label='MST' if u == result['mst_edges'][0][0] else '')
        for u, v in result['matching_edges']:
            ax.plot([self.points[u, 0], self.points[v, 0]], 
                   [self.points[u, 1], self.points[v, 1]], 'orange', linewidth=2, linestyle='--', 
                   label='Matching' if u == result['matching_edges'][0][0] else '')
        ax.set_title("Step 3: MST + Matching")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Final Tour
        ax = axes[1, 1]
        tour = result['tour']
        ax.scatter(self.points[:, 0], self.points[:, 1], c='red', s=100, zorder=5)
        for i in range(len(tour) - 1):
            ax.plot([self.points[tour[i], 0], self.points[tour[i+1], 0]], 
                   [self.points[tour[i], 1], self.points[tour[i+1], 1]], 'b-', linewidth=2)
        ax.set_title(f"Final Tour (Cost: {result['cost']:.2f})")
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig


# Example usage and testing
if __name__ == "__main__":
    # Generate random Euclidean points
    np.random.seed(42)
    n_points = 15
    points = np.random.rand(n_points, 2) * 100
    
    print(f"Running Christofides Algorithm on {n_points} points\n")
    print("=" * 60)
    
    # Run algorithm
    christofides = ChristofidesAlgorithm(points)
    result = christofides.run(verbose=True)
    
    print("\n" + "=" * 60)
    print("\nTour sequence:", result['tour'])
    
    # Visualize
    fig = christofides.visualize(result)
    plt.savefig('christofides_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'christofides_visualization.png'")
    plt.show()