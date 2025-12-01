"""
2-Opt Local Search for TSP
Iteratively improves tour by reversing segments
Guaranteed to converge to 2-local optimum
"""

import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import time

class TwoOptLocalSearch:
    def __init__(self, points):
        """
        Initialize with points in 2D Euclidean space
        points: np.array of shape (n, 2)
        """
        self.points = points
        self.n = len(points)
        self.dist_matrix = distance_matrix(points, points)
        
    def tour_cost(self, tour):
        """Calculate total cost of a tour"""
        cost = 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            cost += self.dist_matrix[tour[i], tour[j]]
        return cost
    
    def two_opt_swap(self, tour, i, k):
        """
        Perform 2-opt swap: reverse tour segment between i and k
        tour[0:i] + reverse(tour[i:k+1]) + tour[k+1:n]
        """
        new_tour = tour[:i] + tour[i:k+1][::-1] + tour[k+1:]
        return new_tour
    
    def run(self, initial_tour=None, max_iterations=10000, verbose=True):
        """
        Execute 2-Opt local search
        
        Args:
            initial_tour: starting tour (if None, uses greedy nearest neighbor)
            max_iterations: maximum number of iterations
            verbose: print progress
            
        Returns:
            dict with tour, cost, and convergence history
        """
        start_time = time.time()
        
        # Initialize tour
        if initial_tour is None:
            if verbose: print("Generating initial tour using Nearest Neighbor...")
            tour = self._nearest_neighbor_tour()
        else:
            tour = initial_tour.copy()
        
        initial_cost = self.tour_cost(tour)
        if verbose: 
            print(f"Initial tour cost: {initial_cost:.2f}")
            print("Starting 2-Opt optimization...\n")
        
        # Track progress
        cost_history = [initial_cost]
        improvement = True
        iteration = 0
        total_improvements = 0
        
        while improvement and iteration < max_iterations:
            improvement = False
            best_delta = 0
            best_i, best_k = -1, -1
            
            # Try all possible 2-opt swaps
            for i in range(1, self.n - 1):
                for k in range(i + 1, self.n):
                    # Calculate change in cost for this swap
                    # Before: (i-1, i) and (k, k+1)
                    # After:  (i-1, k) and (i, k+1)
                    
                    i_prev = (i - 1) % self.n
                    k_next = (k + 1) % self.n
                    
                    # Cost of edges being removed
                    old_cost = (self.dist_matrix[tour[i_prev], tour[i]] + 
                               self.dist_matrix[tour[k], tour[k_next]])
                    
                    # Cost of edges being added
                    new_cost = (self.dist_matrix[tour[i_prev], tour[k]] + 
                               self.dist_matrix[tour[i], tour[k_next]])
                    
                    delta = new_cost - old_cost
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_k = i, k
                        improvement = True
            
            # Apply best improvement
            if improvement:
                tour = self.two_opt_swap(tour, best_i, best_k)
                new_cost = self.tour_cost(tour)
                cost_history.append(new_cost)
                total_improvements += 1
                
                if verbose and total_improvements % 10 == 0:
                    print(f"  Improvement {total_improvements}: Cost = {new_cost:.2f} (Δ = {best_delta:.2f})")
            
            iteration += 1
        
        final_cost = self.tour_cost(tour)
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n✓ 2-Opt Local Search Complete")
            print(f"  Final tour cost: {final_cost:.2f}")
            print(f"  Initial cost: {initial_cost:.2f}")
            print(f"  Improvement: {initial_cost - final_cost:.2f} ({(1 - final_cost/initial_cost)*100:.1f}%)")
            print(f"  Total improvements: {total_improvements}")
            print(f"  Iterations: {iteration}")
            print(f"  Runtime: {elapsed_time:.4f} seconds")
        
        return {
            'tour': tour,
            'cost': final_cost,
            'initial_cost': initial_cost,
            'cost_history': cost_history,
            'improvements': total_improvements,
            'iterations': iteration,
            'runtime': elapsed_time
        }
    
    def _nearest_neighbor_tour(self):
        """Generate initial tour using greedy nearest neighbor heuristic"""
        unvisited = set(range(self.n))
        tour = [0]
        unvisited.remove(0)
        
        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda x: self.dist_matrix[current, x])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        return tour
    
    def visualize_convergence(self, result, title="2-Opt Convergence"):
        """Plot convergence of 2-Opt algorithm"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Convergence plot
        ax = axes[0]
        ax.plot(result['cost_history'], linewidth=2, color='blue')
        ax.axhline(result['cost'], color='red', linestyle='--', label=f"Final: {result['cost']:.2f}")
        ax.axhline(result['initial_cost'], color='green', linestyle='--', label=f"Initial: {result['initial_cost']:.2f}")
        ax.set_xlabel('Improvement Step', fontsize=12)
        ax.set_ylabel('Tour Cost', fontsize=12)
        ax.set_title('Cost vs Improvement Steps', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Tour visualization
        ax = axes[1]
        tour = result['tour']
        ax.scatter(self.points[:, 0], self.points[:, 1], c='red', s=100, zorder=5)
        
        # Draw tour
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            ax.plot([self.points[tour[i], 0], self.points[tour[j], 0]], 
                   [self.points[tour[i], 1], self.points[tour[j], 1]], 
                   'b-', linewidth=2, alpha=0.7)
        
        # Annotate vertices
        for i, (x, y) in enumerate(self.points):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
        
        ax.set_title(f'Optimized Tour (Cost: {result["cost"]:.2f})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def compare_with_initial(self, result):
        """Visualize initial vs optimized tour side by side"""
        # Reconstruct initial tour
        initial_tour = self._nearest_neighbor_tour()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Initial tour
        ax = axes[0]
        ax.scatter(self.points[:, 0], self.points[:, 1], c='red', s=100, zorder=5)
        for i in range(len(initial_tour)):
            j = (i + 1) % len(initial_tour)
            ax.plot([self.points[initial_tour[i], 0], self.points[initial_tour[j], 0]], 
                   [self.points[initial_tour[i], 1], self.points[initial_tour[j], 1]], 
                   'gray', linewidth=2, alpha=0.7)
        ax.set_title(f'Initial Tour (Nearest Neighbor)\nCost: {result["initial_cost"]:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Optimized tour
        ax = axes[1]
        tour = result['tour']
        ax.scatter(self.points[:, 0], self.points[:, 1], c='red', s=100, zorder=5)
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            ax.plot([self.points[tour[i], 0], self.points[tour[j], 0]], 
                   [self.points[tour[i], 1], self.points[tour[j], 1]], 
                   'blue', linewidth=2, alpha=0.7)
        improvement_pct = (1 - result['cost']/result['initial_cost']) * 100
        ax.set_title(f'After 2-Opt\nCost: {result["cost"]:.2f} ({improvement_pct:.1f}% improvement)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Example usage and testing
if __name__ == "__main__":
    # Generate random Euclidean points
    np.random.seed(42)
    n_points = 20
    points = np.random.rand(n_points, 2) * 100
    
    print(f"Running 2-Opt Local Search on {n_points} points\n")
    print("=" * 60)
    
    # Run algorithm
    two_opt = TwoOptLocalSearch(points)
    result = two_opt.run(verbose=True)
    
    print("\n" + "=" * 60)
    print("\nOptimized tour sequence:", result['tour'])
    
    # Visualize convergence
    fig1 = two_opt.visualize_convergence(result)
    plt.savefig('two_opt_convergence.png', dpi=150, bbox_inches='tight')
    print("\n✓ Convergence plot saved as 'two_opt_convergence.png'")
    
    # Compare initial vs optimized
    fig2 = two_opt.compare_with_initial(result)
    plt.savefig('two_opt_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Comparison plot saved as 'two_opt_comparison.png'")
    
    plt.show()