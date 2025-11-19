"""
TSP LP Relaxation with Subtour Elimination Constraints
Provides lower bound on optimal TSP solution
Can be used with randomized rounding for hybrid approach
"""

import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linprog
import pulp
import matplotlib.pyplot as plt
import time
from itertools import combinations

class TSPLPRelaxation:
    def __init__(self, points):
        """
        Initialize with points in 2D Euclidean space
        points: np.array of shape (n, 2)
        """
        self.points = points
        self.n = len(points)
        self.dist_matrix = distance_matrix(points, points)
        
    def solve_basic_lp(self, verbose=True):
        """
        Solve basic LP relaxation without subtour elimination
        This will likely produce fractional solutions with subtours
        
        min: sum(c_ij * x_ij) for all edges
        s.t: sum(x_ij) for j != i = 2 for all i (degree constraint)
             0 <= x_ij <= 1 for all i, j
        """
        if verbose: print("Solving basic LP relaxation (no subtour elimination)...")
        start_time = time.time()
        
        # Create problem
        prob = pulp.LpProblem("TSP_LP_Basic", pulp.LpMinimize)
        
        # Variables: x_ij for each edge (i < j)
        x = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1)
        
        # Objective: minimize total distance
        prob += pulp.lpSum([self.dist_matrix[i, j] * x[i, j] 
                           for i in range(self.n) 
                           for j in range(i + 1, self.n)])
        
        # Constraints: each vertex has degree 2
        for i in range(self.n):
            edges = []
            for j in range(self.n):
                if i < j and (i, j) in x:
                    edges.append(x[i, j])
                elif j < i and (j, i) in x:
                    edges.append(x[j, i])
            prob += pulp.lpSum(edges) == 2
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        runtime = time.time() - start_time
        objective_value = pulp.value(prob.objective)
        
        # Extract solution
        solution = {}
        for (i, j), var in x.items():
            val = var.varValue
            if val > 1e-6:  # Only keep non-zero edges
                solution[i, j] = val
        
        if verbose:
            print(f"  LP Objective: {objective_value:.2f}")
            print(f"  Runtime: {runtime:.4f} seconds")
            print(f"  Non-zero edges: {len(solution)}")
        
        return {
            'objective': objective_value,
            'solution': solution,
            'runtime': runtime,
            'status': pulp.LpStatus[prob.status]
        }
    
    def solve_with_subtour_elimination(self, max_subtours=10, verbose=True):
        """
        Solve LP with iterative subtour elimination
        Start with basic LP, add subtour constraints as violations are found
        
        This is the Dantzig-Fulkerson-Johnson formulation
        """
        if verbose: 
            print("\nSolving LP with subtour elimination constraints...")
            print("(Iterative constraint generation)")
        
        start_time = time.time()
        iteration = 0
        added_constraints = 0
        
        # Create problem
        prob = pulp.LpProblem("TSP_LP_Subtour", pulp.LpMinimize)
        
        # Variables
        x = {}
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1)
        
        # Objective
        prob += pulp.lpSum([self.dist_matrix[i, j] * x[i, j] 
                           for i in range(self.n) 
                           for j in range(i + 1, self.n)])
        
        # Degree constraints
        for i in range(self.n):
            edges = []
            for j in range(self.n):
                if i < j and (i, j) in x:
                    edges.append(x[i, j])
                elif j < i and (j, i) in x:
                    edges.append(x[j, i])
            prob += pulp.lpSum(edges) == 2
        
        # Iteratively add subtour elimination constraints
        while iteration < max_subtours:
            # Solve current LP
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            
            if pulp.LpStatus[prob.status] != 'Optimal':
                break
            
            # Extract current solution
            solution = {}
            for (i, j), var in x.items():
                val = var.varValue
                if val > 1e-6:
                    solution[i, j] = val
            
            # Find subtours
            subtours = self._find_subtours(solution)
            
            if len(subtours) == 1:
                # Single tour found - optimal!
                if verbose: print(f"  ✓ Found connected tour in iteration {iteration + 1}")
                break
            
            # Add subtour elimination constraints
            for subtour in subtours:
                if len(subtour) < self.n:  # Don't constrain the full tour
                    edges_in_subtour = []
                    for i in subtour:
                        for j in subtour:
                            if i < j and (i, j) in x:
                                edges_in_subtour.append(x[i, j])
                    
                    # Constraint: sum of edges in subtour <= |subtour| - 1
                    prob += pulp.lpSum(edges_in_subtour) <= len(subtour) - 1
                    added_constraints += 1
            
            if verbose:
                print(f"  Iteration {iteration + 1}: Found {len(subtours)} components, added {len(subtours) - 1} constraints")
            
            iteration += 1
        
        runtime = time.time() - start_time
        objective_value = pulp.value(prob.objective)
        
        # Final solution
        solution = {}
        for (i, j), var in x.items():
            val = var.varValue
            if val > 1e-6:
                solution[i, j] = val
        
        if verbose:
            print(f"\n✓ LP with Subtour Elimination Complete")
            print(f"  LP Lower Bound: {objective_value:.2f}")
            print(f"  Iterations: {iteration + 1}")
            print(f"  Constraints added: {added_constraints}")
            print(f"  Runtime: {runtime:.4f} seconds")
        
        return {
            'objective': objective_value,
            'solution': solution,
            'runtime': runtime,
            'iterations': iteration + 1,
            'constraints_added': added_constraints,
            'status': pulp.LpStatus[prob.status]
        }
    
    def _find_subtours(self, solution):
        """
        Find connected components (subtours) in LP solution
        Uses DFS to identify components
        """
        # Build adjacency list from fractional solution
        adj = {i: [] for i in range(self.n)}
        for (i, j), val in solution.items():
            if val > 0.5:  # Consider edge if x_ij > 0.5
                adj[i].append(j)
                adj[j].append(i)
        
        # Find connected components
        visited = [False] * self.n
        components = []
        
        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    dfs(neighbor, component)
        
        for i in range(self.n):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)
        
        return components
    
    def randomized_rounding(self, lp_solution, seed=42, verbose=True):
        """
        Apply randomized rounding to LP solution to get integer tour
        
        This is a simple approach - more sophisticated methods exist
        """
        if verbose: print("\nApplying randomized rounding...")
        
        np.random.seed(seed)
        
        # Build graph with edges weighted by LP values
        edges = []
        for (i, j), val in lp_solution['solution'].items():
            if val > 0.1:  # Include edges with substantial LP value
                edges.append((i, j, val))
        
        # Sample edges proportional to LP values
        selected_edges = []
        adj = {i: [] for i in range(self.n)}
        
        # Sort edges by LP value (prioritize higher values)
        edges.sort(key=lambda e: e[2], reverse=True)
        
        for i, j, val in edges:
            # Add edge if it doesn't violate degree constraints too much
            if len(adj[i]) < 2 and len(adj[j]) < 2:
                selected_edges.append((i, j))
                adj[i].append(j)
                adj[j].append(i)
        
        # Try to complete into a tour using nearest neighbor
        tour = self._complete_tour(selected_edges)
        cost = self._tour_cost(tour)
        
        if verbose:
            print(f"  Rounded tour cost: {cost:.2f}")
            print(f"  LP lower bound: {lp_solution['objective']:.2f}")
            print(f"  Approximation ratio: {cost / lp_solution['objective']:.3f}")
        
        return {
            'tour': tour,
            'cost': cost,
            'lp_bound': lp_solution['objective'],
            'ratio': cost / lp_solution['objective']
        }
    
    def _complete_tour(self, edges):
        """Complete partial tour using nearest neighbor"""
        # Build adjacency from selected edges
        adj = {i: [] for i in range(self.n)}
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # Start from vertex 0
        tour = [0]
        visited = {0}
        
        while len(tour) < self.n:
            current = tour[-1]
            
            # Try to follow existing edges first
            next_vertex = None
            for neighbor in adj[current]:
                if neighbor not in visited:
                    next_vertex = neighbor
                    break
            
            # If no edge, use nearest unvisited neighbor
            if next_vertex is None:
                min_dist = float('inf')
                for i in range(self.n):
                    if i not in visited:
                        dist = self.dist_matrix[current, i]
                        if dist < min_dist:
                            min_dist = dist
                            next_vertex = i
            
            tour.append(next_vertex)
            visited.add(next_vertex)
        
        return tour
    
    def _tour_cost(self, tour):
        """Calculate tour cost"""
        cost = 0
        for i in range(len(tour)):
            j = (i + 1) % len(tour)
            cost += self.dist_matrix[tour[i], tour[j]]
        return cost
    
    def visualize_lp_solution(self, lp_result, title="LP Relaxation Solution"):
        """Visualize fractional LP solution"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw edges with thickness proportional to LP value
        for (i, j), val in lp_result['solution'].items():
            alpha = min(val, 1.0)
            linewidth = val * 4
            ax.plot([self.points[i, 0], self.points[j, 0]], 
                   [self.points[i, 1], self.points[j, 1]], 
                   'b-', linewidth=linewidth, alpha=alpha)
        
        # Draw vertices
        ax.scatter(self.points[:, 0], self.points[:, 1], c='red', s=150, zorder=5, edgecolors='black', linewidths=2)
        
        # Annotate
        for i, (x, y) in enumerate(self.points):
            ax.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax.set_title(f'{title}\nObjective: {lp_result["objective"]:.2f}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Example usage
if __name__ == "__main__":
    # Generate random points
    np.random.seed(42)
    n_points = 10  # Keep small for LP solver
    points = np.random.rand(n_points, 2) * 100
    
    print(f"Running TSP LP Relaxation on {n_points} points\n")
    print("=" * 60)
    
    # Initialize
    tsp_lp = TSPLPRelaxation(points)
    
    # Solve basic LP
    basic_result = tsp_lp.solve_basic_lp(verbose=True)
    
    # Solve with subtour elimination
    subtour_result = tsp_lp.solve_with_subtour_elimination(max_subtours=20, verbose=True)
    
    # Apply randomized rounding
    rounded = tsp_lp.randomized_rounding(subtour_result, verbose=True)
    
    print("\n" + "=" * 60)
    print("\n✓ All LP computations complete!")
    print(f"\nSummary:")
    print(f"  LP Lower Bound: {subtour_result['objective']:.2f}")
    print(f"  Rounded Tour Cost: {rounded['cost']:.2f}")
    print(f"  Gap: {rounded['cost'] - subtour_result['objective']:.2f}")
    
    # Visualize
    fig = tsp_lp.visualize_lp_solution(subtour_result)
    plt.savefig('tsp_lp_solution.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'tsp_lp_solution.png'")
    plt.show()