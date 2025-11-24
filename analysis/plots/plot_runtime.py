"""
Runtime Plotting Scripts
Generates runtime vs. n plots for TSP, Vertex Cover, and Set Cover algorithms.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Dict, Tuple
import glob


def plot_runtime_scaling(results: List[Tuple[int, float]], 
                         algorithm_name: str,
                         output_file: str = None,
                         log_scale: bool = True,
                         show_fit: bool = True,
                         color: str = 'blue'):
    """
    Plot runtime vs. problem size with optional log-log scaling and complexity fit.
    
    Args:
        results: List of (n, runtime) tuples
        algorithm_name: Name of the algorithm for labeling
        output_file: Optional path to save the plot
        log_scale: Use log-log scale (good for exponential/polynomial growth)
        show_fit: Fit and display a power-law curve
    """
    if not results:
        print(f"Warning: No results to plot for {algorithm_name}")
        return
    
    results = sorted(results)
    n_values = np.array([r[0] for r in results])
    times = np.array([r[1] for r in results])
    
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(n_values, times, s=100, alpha=0.6, color=color, 
               edgecolors='black', linewidths=1.5, label='Measured', zorder=3)
    
    # Fit power law if requested (log-log linear fit)
    if show_fit and len(n_values) > 2:
        # Fit log(time) = a * log(n) + b => time = exp(b) * n^a
        log_n = np.log(n_values)
        log_t = np.log(times)
        coeffs = np.polyfit(log_n, log_t, 1)
        a, b = coeffs[0], coeffs[1]
        
        # Generate fit line
        n_fit = np.linspace(n_values.min(), n_values.max(), 100)
        t_fit = np.exp(b) * (n_fit ** a)
        
        plt.plot(n_fit, t_fit, '--', color='red', linewidth=2, 
                label=f'Fit: O(n^{a:.2f})', alpha=0.7, zorder=2)
    
    plt.xlabel('Problem Size (n)', fontsize=13, fontweight='bold')
    plt.ylabel('Runtime (seconds)', fontsize=13, fontweight='bold')
    plt.title(f'{algorithm_name}: Runtime vs. Problem Size', 
             fontsize=14, fontweight='bold')
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, which='both', linestyle='--')
    else:
        plt.grid(True, alpha=0.3)
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return plt.gcf()


def plot_multiple_algorithms_runtime(results_dict: Dict[str, List[Tuple[int, float]]],
                                     title: str = "Algorithm Runtime Comparison",
                                     output_file: str = None,
                                     log_scale: bool = True):
    """
    Plot runtime comparison for multiple algorithms on the same graph.
    
    Args:
        results_dict: Dictionary mapping algorithm names to list of (n, runtime) tuples
        title: Plot title
        output_file: Optional path to save the plot
        log_scale: Use log-log scale
    """
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, (alg_name, results) in enumerate(results_dict.items()):
        if not results:
            continue
        
        results = sorted(results)
        n_values = np.array([r[0] for r in results])
        times = np.array([r[1] for r in results])
        
        marker = markers[i % len(markers)]
        plt.plot(n_values, times, marker=marker, markersize=8, linewidth=2,
                alpha=0.7, label=alg_name, color=colors[i])
    
    plt.xlabel('Problem Size (n)', fontsize=13, fontweight='bold')
    plt.ylabel('Runtime (seconds)', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    
    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3, which='both', linestyle='--')
    else:
        plt.grid(True, alpha=0.3)
    
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return plt.gcf()


def load_and_plot_vc_results(small_json: str = None, large_json: str = None,
                             output_dir: str = 'collected_results'):
    """
    Load Vertex Cover experiment results and generate runtime plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    results_dict = {}
    
    if small_json and os.path.exists(small_json):
        with open(small_json, 'r') as f:
            small_data = json.load(f)
        
        # Extract runtime data for each algorithm
        for alg in ['exact', 'primal_dual', 'hybrid_total']:
            results_dict[f'VC_{alg}'] = []
        
        for entry in small_data:
            n = entry['n']
            if 'timings' in entry:
                for alg in ['exact', 'primal_dual', 'hybrid_total']:
                    if alg in entry['timings']:
                        results_dict[f'VC_{alg}'].append((n, entry['timings'][alg]))
    
    if large_json and os.path.exists(large_json):
        with open(large_json, 'r') as f:
            large_data = json.load(f)
        
        for entry in large_data:
            n = entry['n']
            if 'timings' in entry:
                for alg in ['primal_dual', 'hybrid_total']:
                    if alg in entry['timings']:
                        if f'VC_{alg}' in results_dict:
                            results_dict[f'VC_{alg}'].append((n, entry['timings'][alg]))
    
    # Generate plots
    if results_dict:
        # Individual plots
        for alg_name, results in results_dict.items():
            if results:
                plot_runtime_scaling(
                    results,
                    alg_name,
                    output_file=os.path.join(output_dir, f'{alg_name}_runtime.png')
                )
        
        # Combined plot
        plot_multiple_algorithms_runtime(
            results_dict,
            title="Vertex Cover: Runtime Comparison",
            output_file=os.path.join(output_dir, 'vc_runtime_comparison.png')
        )
        
        plt.close('all')
        print("✓ VC runtime plots generated")


def create_tsp_runtime_plots(data_dir: str = 'collected_results/tsp',
                             output_dir: str = 'collected_results'):
    """
    Create runtime plots for TSP algorithms from experimental data.
    Expected data format: JSON files with runtime measurements.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # This is a template - actual implementation depends on data format
    # You would load TSP experimental results here
    print("TSP runtime plotting: Waiting for experimental data...")
    
    # Example structure:
    # results_dict = {
    #     'Christofides': [(n1, t1), (n2, t2), ...],
    #     '2-Opt': [...],
    #     'Hybrid': [...],
    #     'Exact': [...]
    # }


def create_sc_runtime_plots(data_dir: str = 'collected_results/sc',
                            output_dir: str = 'collected_results'):
    """
    Create runtime plots for Set Cover algorithms from experimental data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("SC runtime plotting: Waiting for experimental data...")


def create_all_runtime_plots():
    """
    Generate all runtime plots from collected experimental results.
    """
    print("="*60)
    print("GENERATING RUNTIME PLOTS")
    print("="*60)
    
    # Vertex Cover plots
    vc_small = "../../VC/results/vc_small_experiments.json"
    vc_large = "../../VC/results/vc_large_experiments.json"
    
    if os.path.exists(vc_small) or os.path.exists(vc_large):
        print("\nGenerating Vertex Cover runtime plots...")
        load_and_plot_vc_results(vc_small, vc_large)
    else:
        print("\nWarning: VC experiment results not found")
    
    # TSP plots
    print("\nChecking for TSP results...")
    create_tsp_runtime_plots()
    
    # SC plots
    print("\nChecking for SC results...")
    create_sc_runtime_plots()
    
    print("\n" + "="*60)
    print("✓ Runtime plot generation complete")
    print("="*60)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Runtime Plotting Demo")
    print("="*60)
    
    # Example: Plot exponential growth (exact solver)
    exact_results = [(n, 2**(n/3) * 0.001) for n in range(5, 21)]
    plot_runtime_scaling(exact_results, "Exact Solver (Exponential)", 
                        output_file="collected_results/demo_exact.png")
    
    # Example: Plot polynomial growth (approximation)
    approx_results = [(n, (n**2) * 0.0001) for n in range(10, 101, 10)]
    plot_runtime_scaling(approx_results, "Approximation (Polynomial)",
                        output_file="collected_results/demo_approx.png")
    
    # Example: Multiple algorithms
    multi_results = {
        'Exact': exact_results[:10],
        'Approx 1': [(n, (n**1.5) * 0.0001) for n in range(10, 101, 10)],
        'Approx 2': [(n, (n**1.8) * 0.00008) for n in range(10, 101, 10)]
    }
    plot_multiple_algorithms_runtime(multi_results,
                                    title="Algorithm Comparison Demo",
                                    output_file="collected_results/demo_multi.png")
    
    print("\n✓ Demo plots generated in collected_results/")
    
    # Try to generate actual plots from real data
    print("\n" + "="*60)
    create_all_runtime_plots()
