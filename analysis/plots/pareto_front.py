"""
Pareto Front Plotting
Visualizes trade-offs between solution quality and runtime.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from matplotlib.patches import FancyBboxPatch


def plot_pareto_front(algorithms_data: Dict[str, Tuple[float, float]],
                     problem_name: str = "Problem",
                     output_file: str = None,
                     opt_value: float = None,
                     log_time: bool = False):
    """
    Plot Pareto front showing quality vs. runtime trade-off.
    
    Args:
        algorithms_data: Dict mapping algorithm name to (runtime, cost) tuple
        problem_name: Name of the problem for labeling
        output_file: Optional path to save the plot
        opt_value: Optimal value (if known) to show approximation ratio
        log_time: Use logarithmic scale for time axis
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    alg_names = list(algorithms_data.keys())
    times = np.array([algorithms_data[alg][0] for alg in alg_names])
    costs = np.array([algorithms_data[alg][1] for alg in alg_names])
    
    # Normalize costs to ratios if OPT is known
    if opt_value and opt_value > 0:
        ratios = costs / opt_value
        y_label = 'Approximation Ratio (ALG/OPT)'
        y_data = ratios
    else:
        y_label = 'Solution Cost'
        y_data = costs
    
    # Plot points
    colors = plt.cm.tab10(np.linspace(0, 1, len(alg_names)))
    
    for i, alg in enumerate(alg_names):
        ax.scatter(times[i], y_data[i], s=200, alpha=0.7, color=colors[i],
                  edgecolors='black', linewidths=2, zorder=3, label=alg)
        
        # Annotate points
        ax.annotate(alg, (times[i], y_data[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Identify Pareto-optimal points
    # A point is Pareto-optimal if no other point is better in both dimensions
    pareto_indices = []
    for i in range(len(alg_names)):
        is_dominated = False
        for j in range(len(alg_names)):
            if i != j:
                # Point j dominates point i if it's faster AND better quality
                if times[j] <= times[i] and y_data[j] <= y_data[i]:
                    if times[j] < times[i] or y_data[j] < y_data[i]:
                        is_dominated = True
                        break
        if not is_dominated:
            pareto_indices.append(i)
    
    # Draw Pareto front
    if len(pareto_indices) > 1:
        pareto_times = times[pareto_indices]
        pareto_costs = y_data[pareto_indices]
        
        # Sort by time
        sorted_indices = np.argsort(pareto_times)
        pareto_times = pareto_times[sorted_indices]
        pareto_costs = pareto_costs[sorted_indices]
        
        ax.plot(pareto_times, pareto_costs, 'r--', linewidth=2, alpha=0.5,
               label='Pareto Front', zorder=2)
        
        # Highlight Pareto-optimal points
        ax.scatter(pareto_times, pareto_costs, s=300, facecolors='none',
                  edgecolors='red', linewidths=3, zorder=4)
    
    # Add OPT line if available
    if opt_value and opt_value > 0:
        ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
                  alpha=0.5, label='OPT (ratio=1.0)')
    
    ax.set_xlabel('Runtime (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
    ax.set_title(f'{problem_name}: Quality vs. Runtime Trade-off', 
                fontsize=14, fontweight='bold')
    
    if log_time:
        ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig


def plot_multi_instance_pareto(instance_results: List[Dict[str, Tuple[float, float]]],
                               problem_name: str = "Problem",
                               output_file: str = None):
    """
    Plot Pareto fronts for multiple problem instances on the same graph.
    
    Args:
        instance_results: List of dicts, each mapping algorithm to (runtime, cost)
        problem_name: Name of the problem
        output_file: Optional path to save plot
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    instance_colors = plt.cm.viridis(np.linspace(0, 1, len(instance_results)))
    
    for idx, (inst_data, color) in enumerate(zip(instance_results, instance_colors)):
        alg_names = list(inst_data.keys())
        times = np.array([inst_data[alg][0] for alg in alg_names])
        costs = np.array([inst_data[alg][1] for alg in alg_names])
        
        # Plot points for this instance
        ax.scatter(times, costs, s=100, alpha=0.6, color=color,
                  edgecolors='black', linewidths=1, label=f'Instance {idx+1}')
        
        # Connect with lines to show Pareto front
        sorted_indices = np.argsort(times)
        ax.plot(times[sorted_indices], costs[sorted_indices], '--', 
               color=color, alpha=0.3, linewidth=1.5)
    
    ax.set_xlabel('Runtime (seconds)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Solution Cost', fontsize=13, fontweight='bold')
    ax.set_title(f'{problem_name}: Quality vs. Runtime (Multiple Instances)', 
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2, loc='best')
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig


def analyze_vc_pareto(json_file: str, output_dir: str = 'collected_results'):
    """
    Create Pareto front plots for Vertex Cover experiments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(json_file):
        print(f"Warning: {json_file} not found")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Group by instance parameters
    instances = {}
    for entry in data:
        key = (entry.get('n', 0), entry.get('p', 0), entry.get('seed', 0))
        if key not in instances:
            instances[key] = {}
        
        opt_cost = entry.get('exact_cost', None)
        
        # Primal-Dual
        if 'primal_dual_cost' in entry and 'timings' in entry:
            pd_time = entry['timings'].get('primal_dual', 0)
            instances[key]['Primal-Dual'] = (pd_time, entry['primal_dual_cost'])
        
        # Hybrid
        if 'hybrid_cost' in entry and 'timings' in entry:
            hybrid_time = entry['timings'].get('hybrid_total', 0)
            instances[key]['Hybrid'] = (hybrid_time, entry['hybrid_cost'])
        
        # Exact (for reference)
        if opt_cost and 'timings' in entry:
            exact_time = entry['timings'].get('exact', 0)
            instances[key]['Exact'] = (exact_time, opt_cost)
        
        instances[key]['_opt'] = opt_cost
    
    # Create plots for a few representative instances
    sample_instances = list(instances.items())[:3]
    
    for (n, p, seed), inst_data in sample_instances:
        opt_value = inst_data.pop('_opt', None)
        
        plot_pareto_front(
            inst_data,
            problem_name=f"Vertex Cover (n={n}, p={p}, seed={seed})",
            output_file=os.path.join(output_dir, f'vc_pareto_n{n}_s{seed}.png'),
            opt_value=opt_value,
            log_time=True
        )
    
    print(f"✓ Generated {len(sample_instances)} Pareto front plots for VC")
    plt.close('all')


def create_all_pareto_plots():
    """
    Generate all Pareto front plots from experimental results.
    """
    print("="*60)
    print("GENERATING PARETO FRONT PLOTS")
    print("="*60)
    
    # Vertex Cover
    vc_file = "../../VC/results/vc_small_experiments.json"
    if os.path.exists(vc_file):
        print("\nCreating VC Pareto fronts...")
        analyze_vc_pareto(vc_file)
    else:
        print("\nWarning: VC experiments not found")
    
    # TODO: Add TSP and SC when data is available
    
    print("\n" + "="*60)
    print("✓ Pareto front plot generation complete")
    print("="*60)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Pareto Front Plotting Demo")
    print("="*60)
    
    # Create demo data
    demo_algorithms = {
        'Exact': (1.5, 100.0),       # Slow, optimal
        'Greedy': (0.01, 150.0),     # Fast, lower quality
        'LP-Round': (0.2, 120.0),    # Medium speed, good quality
        'Hybrid': (0.5, 105.0),      # Slower, better quality
        'Local Search': (0.05, 140.0) # Fast, moderate quality
    }
    
    plot_pareto_front(
        demo_algorithms,
        problem_name="Demo Problem",
        output_file="collected_results/demo_pareto.png",
        opt_value=100.0,
        log_time=True
    )
    
    print("\n✓ Demo plot generated")
    
    # Try to generate plots from real data
    print("\n" + "="*60)
    create_all_pareto_plots()
