"""
Approximation Ratio Plotting Scripts
Generates plots of approximation ratios (ALG/OPT) for comparison with exact solutions.
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from collections import defaultdict


def plot_approximation_ratios(ratios_dict: Dict[str, List[float]],
                              problem_name: str = "Problem",
                              output_file: str = None,
                              theoretical_bounds: Dict[str, float] = None):
    """
    Plot approximation ratios for multiple algorithms with box plots.
    
    Args:
        ratios_dict: Dict mapping algorithm name to list of approximation ratios
        problem_name: Name of the problem for labeling
        output_file: Optional path to save the plot
        theoretical_bounds: Dict of theoretical approximation guarantees
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    alg_names = list(ratios_dict.keys())
    ratios_data = [ratios_dict[alg] for alg in alg_names]
    
    bp = ax1.boxplot(ratios_data, labels=alg_names, patch_artist=True,
                     notch=True, showmeans=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(alg_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add theoretical bounds if provided
    if theoretical_bounds:
        for i, alg in enumerate(alg_names):
            if alg in theoretical_bounds:
                bound = theoretical_bounds[alg]
                ax1.axhline(y=bound, color='red', linestyle='--', alpha=0.5, 
                           linewidth=1.5)
                ax1.text(i+1, bound, f' {bound}', verticalalignment='bottom',
                        fontsize=9, color='red')
    
    ax1.set_ylabel('Approximation Ratio (ALG/OPT)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax1.set_title(f'{problem_name}: Approximation Ratio Distribution', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='x', rotation=15)
    
    # Scatter plot with mean line
    for i, alg in enumerate(alg_names):
        ratios = ratios_dict[alg]
        x = np.random.normal(i+1, 0.04, size=len(ratios))
        ax2.scatter(x, ratios, alpha=0.5, s=50, color=colors[i], label=alg)
        
        # Add mean line
        mean_ratio = np.mean(ratios)
        ax2.hlines(mean_ratio, i+0.75, i+1.25, colors='black', 
                  linewidth=2, linestyles='solid', alpha=0.7)
    
    ax2.set_ylabel('Approximation Ratio (ALG/OPT)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax2.set_title(f'{problem_name}: Individual Measurements', 
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(range(1, len(alg_names)+1))
    ax2.set_xticklabels(alg_names, rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return fig


def plot_ratio_vs_n(results: Dict[str, List[Tuple[int, float]]],
                    problem_name: str = "Problem",
                    output_file: str = None,
                    theoretical_bounds: Dict[str, float] = None):
    """
    Plot how approximation ratio varies with problem size n.
    
    Args:
        results: Dict mapping algorithm to list of (n, ratio) tuples
        problem_name: Name of the problem
        output_file: Optional path to save plot
        theoretical_bounds: Theoretical approximation guarantees
    """
    plt.figure(figsize=(12, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, (alg_name, data) in enumerate(results.items()):
        if not data:
            continue
        
        data = sorted(data)
        n_values = [d[0] for d in data]
        ratios = [d[1] for d in data]
        
        marker = markers[i % len(markers)]
        plt.plot(n_values, ratios, marker=marker, markersize=8, linewidth=2,
                alpha=0.7, label=alg_name, color=colors[i])
        
        # Add theoretical bound as horizontal line
        if theoretical_bounds and alg_name in theoretical_bounds:
            bound = theoretical_bounds[alg_name]
            plt.axhline(y=bound, color=colors[i], linestyle='--', alpha=0.3,
                       linewidth=1.5)
    
    plt.xlabel('Problem Size (n)', fontsize=13, fontweight='bold')
    plt.ylabel('Approximation Ratio (ALG/OPT)', fontsize=13, fontweight='bold')
    plt.title(f'{problem_name}: Approximation Ratio vs. Problem Size', 
             fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='best')
    plt.tight_layout()
    
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
    
    return plt.gcf()


def analyze_vc_ratios(json_file: str, output_dir: str = 'collected_results'):
    """
    Analyze and plot approximation ratios for Vertex Cover experiments.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(json_file):
        print(f"Warning: {json_file} not found")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract ratios
    ratios_dict = defaultdict(list)
    ratios_vs_n = defaultdict(list)
    
    for entry in data:
        n = entry.get('n', 0)
        
        # Primal-Dual ratio
        if 'primal_dual_ratio' in entry:
            ratios_dict['Primal-Dual'].append(entry['primal_dual_ratio'])
            ratios_vs_n['Primal-Dual'].append((n, entry['primal_dual_ratio']))
        
        # Hybrid ratio
        if 'hybrid_ratio' in entry:
            ratios_dict['Hybrid'].append(entry['hybrid_ratio'])
            ratios_vs_n['Hybrid'].append((n, entry['hybrid_ratio']))
    
    # Theoretical bounds
    theoretical_bounds = {
        'Primal-Dual': 2.0,  # 2-approximation for VC
        'Hybrid': 2.0        # Still 2-approx guaranteed
    }
    
    # Generate plots
    if ratios_dict:
        # Box plot
        plot_approximation_ratios(
            dict(ratios_dict),
            problem_name="Vertex Cover",
            output_file=os.path.join(output_dir, 'vc_approximation_ratios.png'),
            theoretical_bounds=theoretical_bounds
        )
        
        # Ratio vs n
        plot_ratio_vs_n(
            dict(ratios_vs_n),
            problem_name="Vertex Cover",
            output_file=os.path.join(output_dir, 'vc_ratio_vs_n.png'),
            theoretical_bounds=theoretical_bounds
        )
        
        # Print statistics
        print("\n" + "="*60)
        print("VERTEX COVER APPROXIMATION RATIO STATISTICS")
        print("="*60)
        for alg, ratios in ratios_dict.items():
            print(f"\n{alg}:")
            print(f"  Mean:   {np.mean(ratios):.4f}")
            print(f"  Median: {np.median(ratios):.4f}")
            print(f"  Std:    {np.std(ratios):.4f}")
            print(f"  Min:    {np.min(ratios):.4f}")
            print(f"  Max:    {np.max(ratios):.4f}")
            if alg in theoretical_bounds:
                worst_case = theoretical_bounds[alg]
                print(f"  Theoretical bound: {worst_case:.4f}")
                violations = sum(1 for r in ratios if r > worst_case + 0.01)
                print(f"  Violations: {violations}/{len(ratios)}")
        print("="*60)
        
        plt.close('all')


def create_all_ratio_plots():
    """
    Generate all approximation ratio plots from experimental results.
    """
    print("="*60)
    print("GENERATING APPROXIMATION RATIO PLOTS")
    print("="*60)
    
    # Vertex Cover
    vc_file = "../../VC/results/vc_small_experiments.json"
    if os.path.exists(vc_file):
        print("\nAnalyzing Vertex Cover ratios...")
        analyze_vc_ratios(vc_file)
    else:
        print("\nWarning: VC small experiments not found")
    
    # TODO: Add TSP and SC analysis when data is available
    
    print("\n" + "="*60)
    print("✓ Approximation ratio plot generation complete")
    print("="*60)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Approximation Ratio Plotting Demo")
    print("="*60)
    
    # Create demo data
    np.random.seed(42)
    demo_ratios = {
        'Greedy': np.random.uniform(1.2, 1.8, 20),
        'LP-Rounding': np.random.uniform(1.1, 1.5, 20),
        'Hybrid': np.random.uniform(1.05, 1.3, 20)
    }
    
    plot_approximation_ratios(
        demo_ratios,
        problem_name="Demo Problem",
        output_file="collected_results/demo_ratios.png",
        theoretical_bounds={'Greedy': 2.0, 'LP-Rounding': 2.0, 'Hybrid': 2.0}
    )
    
    # Ratio vs n demo
    demo_vs_n = {
        'Greedy': [(n, 1.5 + 0.1*np.random.randn()) for n in range(10, 30, 2)],
        'LP-Rounding': [(n, 1.3 + 0.08*np.random.randn()) for n in range(10, 30, 2)],
        'Hybrid': [(n, 1.15 + 0.05*np.random.randn()) for n in range(10, 30, 2)]
    }
    
    plot_ratio_vs_n(
        demo_vs_n,
        problem_name="Demo Problem",
        output_file="collected_results/demo_ratio_vs_n.png",
        theoretical_bounds={'Greedy': 2.0}
    )
    
    print("\n✓ Demo plots generated")
    
    # Try to generate plots from real data
    print("\n" + "="*60)
    create_all_ratio_plots()
