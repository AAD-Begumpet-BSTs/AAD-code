"""
Quick Start Script for TSP Algorithms
Run this single file to execute all three algorithms and generate visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix
import networkx as nx
import pulp
import time
from datetime import datetime

print("""
╔═══════════════════════════════════════════════════════════════╗
║         TSP ALGORITHMS - COMPREHENSIVE TEST SUITE             ║
║                                                               ║
║  Implementations:                                             ║
║  • Christofides Algorithm (1.5-approximation)                 ║
║  • 2-Opt Local Search                                         ║
║  • LP Relaxation with Subtour Elimination                     ║
╚═══════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# CONFIGURATION
# ============================================================================

PROBLEM_SIZE = 12  # Number of cities (keep ≤15 for LP)
RANDOM_SEED = 42
SAVE_PLOTS = True
VERBOSE = True

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def tour_cost(tour, dist_matrix):
    """Calculate total tour cost"""
    cost = 0
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        cost += dist_matrix[tour[i], tour[j]]
    return cost

def format_time(seconds):
    """Format time for display"""
    if seconds < 0.001:
        return f"{seconds*1000000:.1f} μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f} ms"
    else:
        return f"{seconds:.3f} s"

# ============================================================================
# ALGORITHM 1: CHRISTOFIDES
# ============================================================================

def run_christofides(points, dist_matrix, verbose=True):
    """Execute Christofides Algorithm"""
    if verbose:
        print("\n" + "─"*70)
        print("1. CHRISTOFIDES ALGORITHM")
        print("─"*70)
    
    start = time.time()
    n = len(points)
    
    # Step 1: MST
    if verbose: print("   → Computing Minimum Spanning Tree...")
    G = nx.Graph()
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j, weight=dist_matrix[i, j])
    mst = nx.minimum_spanning_tree(G)
    
    # Step 2: Odd degree vertices
    if verbose: print("   → Finding odd-degree vertices...")
    degrees = dict(mst.degree())
    odd_vertices = [v for v, d in degrees.items() if d % 2 == 1]
    
    # Step 3: Minimum matching
    if verbose: print("   → Computing minimum weight matching...")
    G_odd = nx.Graph()
    for i, u in enumerate(odd_vertices):
        for j, v in enumerate(odd_vertices):
            if i < j:
                G_odd.add_edge(u, v, weight=dist_matrix[u, v])
    matching = nx.min_weight_matching(G_odd)
    
    # Step 4: Eulerian circuit
    if verbose: print("   → Creating Eulerian circuit...")
    G_euler = nx.MultiGraph()
    G_euler.add_edges_from(mst.edges())
    G_euler.add_edges_from(matching)
    eulerian = list(nx.eulerian_circuit(G_euler, source=0))
    
    # Step 5: Shortcut to tour
    if verbose: print("   → Shortcutting to Hamiltonian tour...")
    visited = set()
    tour = []
    for u, v in eulerian:
        if u not in visited:
            tour.append(u)
            visited.add(u)
    
    cost = tour_cost(tour, dist_matrix)
    runtime = time.time() - start
    
    if verbose:
        print(f"\n   ✓ Complete!")
        print(f"     Tour cost: {cost:.2f}")
        print(f"     Runtime: {format_time(runtime)}")
        print(f"     Theoretical guarantee: ≤ 1.5 × OPT")
    
    return {
        'tour': tour,
        'cost': cost,
        'runtime': runtime,
        'mst': list(mst.edges()),
        'matching': list(matching)
    }

# ============================================================================
# ALGORITHM 2: 2-OPT LOCAL SEARCH
# ============================================================================

def nearest_neighbor(dist_matrix):
    """Generate initial tour using nearest neighbor"""
    n = len(dist_matrix)
    unvisited = set(range(n))
    tour = [0]
    unvisited.remove(0)
    
    while unvisited:
        current = tour[-1]
        nearest = min(unvisited, key=lambda x: dist_matrix[current, x])
        tour.append(nearest)
        unvisited.remove(nearest)
    
    return tour

def run_2opt(points, dist_matrix, initial_tour=None, max_iter=10000, verbose=True):
    """Execute 2-Opt Local Search"""
    if verbose:
        print("\n" + "─"*70)
        print("2. 2-OPT LOCAL SEARCH")
        print("─"*70)
    
    start = time.time()
    n = len(points)
    
    # Initialize tour
    if initial_tour is None:
        if verbose: print("   → Generating initial tour (Nearest Neighbor)...")
        tour = nearest_neighbor(dist_matrix)
    else:
        tour = initial_tour.copy()
    
    initial_cost = tour_cost(tour, dist_matrix)
    if verbose: print(f"     Initial cost: {initial_cost:.2f}")
    
    # 2-Opt optimization
    if verbose: print("   → Running 2-Opt optimization...")
    improved = True
    iterations = 0
    improvements = 0
    cost_history = [initial_cost]
    
    while improved and iterations < max_iter:
        improved = False
        best_delta = 0
        best_i, best_k = -1, -1
        
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                i_prev = (i - 1) % n
                k_next = (k + 1) % n
                
                old_cost = (dist_matrix[tour[i_prev], tour[i]] + 
                           dist_matrix[tour[k], tour[k_next]])
                new_cost = (dist_matrix[tour[i_prev], tour[k]] + 
                           dist_matrix[tour[i], tour[k_next]])
                
                delta = new_cost - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_i, best_k = i, k
                    improved = True
        
        if improved:
            tour = tour[:best_i] + tour[best_i:best_k+1][::-1] + tour[best_k+1:]
            improvements += 1
            cost_history.append(tour_cost(tour, dist_matrix))
        
        iterations += 1
    
    final_cost = tour_cost(tour, dist_matrix)
    runtime = time.time() - start
    improvement_pct = ((initial_cost - final_cost) / initial_cost) * 100
    
    if verbose:
        print(f"\n   ✓ Complete!")
        print(f"     Final cost: {final_cost:.2f}")
        print(f"     Improvement: {improvement_pct:.1f}%")
        print(f"     Iterations: {improvements}")
        print(f"     Runtime: {format_time(runtime)}")
    
    return {
        'tour': tour,
        'cost': final_cost,
        'initial_cost': initial_cost,
        'runtime': runtime,
        'improvements': improvements,
        'cost_history': cost_history
    }

# ============================================================================
# ALGORITHM 3: LP RELAXATION
# ============================================================================

def run_lp_relaxation(points, dist_matrix, verbose=True):
    """Execute LP Relaxation with Subtour Elimination"""
    if verbose:
        print("\n" + "─"*70)
        print("3. LP RELAXATION (Lower Bound)")
        print("─"*70)
    
    start = time.time()
    n = len(points)
    
    # Create LP problem
    if verbose: print("   → Formulating linear program...")
    prob = pulp.LpProblem("TSP_LP", pulp.LpMinimize)
    
    # Variables
    x = {}
    for i in range(n):
        for j in range(i + 1, n):
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1)
    
    # Objective
    prob += pulp.lpSum([dist_matrix[i, j] * x[i, j] 
                       for i in range(n) 
                       for j in range(i + 1, n)])
    
    # Degree constraints
    for i in range(n):
        edges = []
        for j in range(n):
            if i < j and (i, j) in x:
                edges.append(x[i, j])
            elif j < i and (j, i) in x:
                edges.append(x[j, i])
        prob += pulp.lpSum(edges) == 2
    
    # Solve
    if verbose: print("   → Solving linear program...")
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    runtime = time.time() - start
    objective = pulp.value(prob.objective)
    
    if verbose:
        print(f"\n   ✓ Complete!")
        print(f"     LP Lower Bound: {objective:.2f}")
        print(f"     Runtime: {format_time(runtime)}")
        print(f"     Status: {pulp.LpStatus[prob.status]}")
    
    return {
        'objective': objective,
        'runtime': runtime,
        'status': pulp.LpStatus[prob.status]
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_all_results(points, results, save=True):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(18, 12))
    
    colors = {
        'nn': '#95a5a6',
        'christofides': '#3498db',
        '2opt_nn': '#2ecc71',
        '2opt_chris': '#9b59b6'
    }
    
    # 1. Nearest Neighbor
    ax1 = plt.subplot(2, 3, 1)
    tour = results['nearest_neighbor']['tour']
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        ax1.plot([points[tour[i], 0], points[tour[j], 0]], 
                [points[tour[i], 1], points[tour[j], 1]], 
                color=colors['nn'], linewidth=2, alpha=0.7)
    ax1.scatter(points[:, 0], points[:, 1], c='red', s=100, zorder=5, edgecolors='black')
    ax1.set_title(f'Nearest Neighbor\nCost: {results["nearest_neighbor"]["cost"]:.2f}', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Christofides
    ax2 = plt.subplot(2, 3, 2)
    # Draw MST
    for u, v in results['christofides']['mst']:
        ax2.plot([points[u, 0], points[v, 0]], [points[u, 1], points[v, 1]], 
                'g-', linewidth=1.5, alpha=0.5, label='MST' if u == results['christofides']['mst'][0][0] else '')
    # Draw matching
    for u, v in results['christofides']['matching']:
        ax2.plot([points[u, 0], points[v, 0]], [points[u, 1], points[v, 1]], 
                'orange', linewidth=1.5, linestyle='--', alpha=0.5, 
                label='Matching' if u == results['christofides']['matching'][0][0] else '')
    # Draw tour
    tour = results['christofides']['tour']
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        ax2.plot([points[tour[i], 0], points[tour[j], 0]], 
                [points[tour[i], 1], points[tour[j], 1]], 
                color=colors['christofides'], linewidth=2.5, alpha=0.8)
    ax2.scatter(points[:, 0], points[:, 1], c='red', s=100, zorder=5, edgecolors='black')
    ax2.set_title(f'Christofides\nCost: {results["christofides"]["cost"]:.2f}', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 2-Opt from NN
    ax3 = plt.subplot(2, 3, 3)
    tour = results['2opt_nn']['tour']
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        ax3.plot([points[tour[i], 0], points[tour[j], 0]], 
                [points[tour[i], 1], points[tour[j], 1]], 
                color=colors['2opt_nn'], linewidth=2, alpha=0.7)
    ax3.scatter(points[:, 0], points[:, 1], c='red', s=100, zorder=5, edgecolors='black')
    improvement = ((results['nearest_neighbor']['cost'] - results['2opt_nn']['cost']) / 
                   results['nearest_neighbor']['cost']) * 100
    ax3.set_title(f'2-Opt from NN\nCost: {results["2opt_nn"]["cost"]:.2f} ({improvement:.1f}% better)', 
                 fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. 2-Opt from Christofides
    ax4 = plt.subplot(2, 3, 4)
    tour = results['2opt_chris']['tour']
    for i in range(len(tour)):
        j = (i + 1) % len(tour)
        ax4.plot([points[tour[i], 0], points[tour[j], 0]], 
                [points[tour[i], 1], points[tour[j], 1]], 
                color=colors['2opt_chris'], linewidth=2, alpha=0.7)
    ax4.scatter(points[:, 0], points[:, 1], c='red', s=100, zorder=5, edgecolors='black')
    improvement = ((results['christofides']['cost'] - results['2opt_chris']['cost']) / 
                   results['christofides']['cost']) * 100
    ax4.set_title(f'2-Opt from Christofides\nCost: {results["2opt_chris"]["cost"]:.2f} ({improvement:.1f}% better)', 
                 fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. 2-Opt Convergence
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(results['2opt_nn']['cost_history'], linewidth=2, label='2-Opt from NN', color=colors['2opt_nn'])
    ax5.plot(results['2opt_chris']['cost_history'], linewidth=2, label='2-Opt from Chris', color=colors['2opt_chris'])
    ax5.set_xlabel('Improvement Step', fontsize=11)
    ax5.set_ylabel('Tour Cost', fontsize=11)
    ax5.set_title('2-Opt Convergence', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Comparison
    ax6 = plt.subplot(2, 3, 6)
    algorithms = ['NN', 'Christofides', '2-Opt\n(from NN)', '2-Opt\n(from Chris)']
    costs = [
        results['nearest_neighbor']['cost'],
        results['christofides']['cost'],
        results['2opt_nn']['cost'],
        results['2opt_chris']['cost']
    ]
    if results.get('lp'):
        algorithms.append('LP\nBound')
        costs.append(results['lp']['objective'])
    
    bars = ax6.bar(range(len(algorithms)), costs, 
                   color=[colors['nn'], colors['christofides'], colors['2opt_nn'], colors['2opt_chris']] + (['red'] if results.get('lp') else []),
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    ax6.set_xticks(range(len(algorithms)))
    ax6.set_xticklabels(algorithms, fontsize=10)
    ax6.set_ylabel('Tour Cost', fontsize=11)
    ax6.set_title('Algorithm Comparison', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{cost:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.suptitle(f'TSP Algorithms Comparison - {len(points)} Cities', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save:
        filename = f'tsp_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n   ✓ Saved visualization: {filename}")
    
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Generate test points
    print(f"Generating {PROBLEM_SIZE} random Euclidean points...")
    np.random.seed(RANDOM_SEED)
    points = np.random.rand(PROBLEM_SIZE, 2) * 100
    dist_matrix = distance_matrix(points, points)
    print(f"✓ Points generated\n")
    
    results = {}
    
    # Run all algorithms
    print("="*70)
    print("EXECUTING ALGORITHMS")
    print("="*70)
    
    # Nearest Neighbor (baseline)
    results['nearest_neighbor'] = {
        'tour': nearest_neighbor(dist_matrix),
        'cost': None,
        'runtime': None
    }
    results['nearest_neighbor']['cost'] = tour_cost(results['nearest_neighbor']['tour'], dist_matrix)
    
    # Christofides
    results['christofides'] = run_christofides(points, dist_matrix, VERBOSE)
    
    # 2-Opt from NN
    results['2opt_nn'] = run_2opt(points, dist_matrix, 
                                  results['nearest_neighbor']['tour'], 
                                  verbose=VERBOSE)
    
    # 2-Opt from Christofides
    results['2opt_chris'] = run_2opt(points, dist_matrix, 
                                     results['christofides']['tour'], 
                                     verbose=VERBOSE)
    
    # LP Relaxation (if problem size allows)
    if PROBLEM_SIZE <= 15:
        results['lp'] = run_lp_relaxation(points, dist_matrix, VERBOSE)
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    data = []
    for name, result in results.items():
        if result:
            cost = result.get('cost', result.get('objective', 'N/A'))
            runtime = result.get('runtime', 'N/A')
            data.append({
                'Algorithm': name.replace('_', ' ').title(),
                'Cost': f"{cost:.2f}" if isinstance(cost, float) else cost,
                'Runtime': format_time(runtime) if isinstance(runtime, float) else runtime
            })
    
    df = pd.DataFrame(data)
    print("\n" + df.to_string(index=False))
    
    # Best solution
    best = min([(k, v['cost']) for k, v in results.items() if 'cost' in v], key=lambda x: x[1])
    print(f"\n🏆 Best Solution: {best[0].replace('_', ' ').title()} with cost {best[1]:.2f}")
    
    if results.get('lp'):
        gap = best[1] - results['lp']['objective']
        gap_pct = (gap / results['lp']['objective']) * 100
        print(f"📊 Optimality Gap: {gap:.2f} ({gap_pct:.1f}% above LP bound)")
    
    print("\n" + "="*70)
    
    # Visualization
    print("\nGenerating visualizations...")
    visualize_all_results(points, results, SAVE_PLOTS)
    plt.show()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
