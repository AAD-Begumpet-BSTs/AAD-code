"""
Generate All Required Plots for TSP Experiments
Author: Saharsh (TSP)
Date: December 2025

This script generates:
1. Approximation ratio plots (small n)
2. Runtime scaling plots (large n, log-log)
3. Pareto-front plots (quality vs time)
4. Statistical comparison plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class TSPPlotGenerator:
    """Generate all plots for TSP experiments"""
    
    def __init__(self, results_file='tsp_experiments_results.csv'):
        """Load experimental results"""
        self.df = pd.read_csv(os.path.join('experiments', results_file))
        self.output_dir = 'experiments/plots'
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✓ Loaded {len(self.df)} experimental results")
        print(f"  Problem sizes: {sorted(self.df['n'].unique())}")
        print(f"  Output directory: {self.output_dir}")
    
    def plot_approximation_ratios_small_n(self):
        """
        Plot 1: Approximation Ratios for Small n
        Shows how close each algorithm gets to optimal/lower bound
        """
        print("\n[1/7] Generating approximation ratio plot (small n)...")
        
        # Filter small instances with valid ratios
        small_df = self.df[self.df['n'] <= 15].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        algorithms = [
            ('christofides_ratio', 'Christofides', 'blue'),
            ('2opt_ratio', '2-Opt', 'orange'),
            ('hybrid_ratio', 'Hybrid', 'green')
        ]
        
        # Plot 1a: Box plots by algorithm
        ax = axes[0]
        data_to_plot = []
        labels = []
        
        for ratio_col, label, color in algorithms:
            if ratio_col in small_df.columns:
                valid_data = small_df[ratio_col].dropna()
                if len(valid_data) > 0:
                    data_to_plot.append(valid_data)
                    labels.append(label)
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, (_, _, color) in zip(bp['boxes'], algorithms):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax.axhline(y=1.5, color='purple', linestyle='--', linewidth=1, 
                   label='Christofides Guarantee', alpha=0.7)
        ax.set_ylabel('Approximation Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Approximation Ratios (Small Instances)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 1b: Ratio vs problem size
        ax = axes[1]
        
        for ratio_col, label, color in algorithms:
            if ratio_col in small_df.columns:
                grouped = small_df.groupby('n')[ratio_col].agg(['mean', 'std'])
                sizes = grouped.index
                means = grouped['mean']
                stds = grouped['std']
                
                ax.plot(sizes, means, 'o-', label=label, color=color, linewidth=2, markersize=8)
                ax.fill_between(sizes, means - stds, means + stds, alpha=0.2, color=color)
        
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax.axhline(y=1.5, color='purple', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Problem Size (n)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Approximation Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Approximation Ratio vs Problem Size', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/1_approximation_ratios_small_n.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 1_approximation_ratios_small_n.png")
    
    def plot_runtime_scaling_large_n(self):
        """
        Plot 2: Runtime Scaling for Large n (log-log)
        Shows how runtime grows with problem size
        """
        print("\n[2/7] Generating runtime scaling plot (large n)...")
        
        large_df = self.df[self.df['n'] >= 20].copy()
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        algorithms = [
            ('christofides_time', 'Christofides', 'blue'),
            ('2opt_time', '2-Opt', 'orange'),
            ('hybrid_time', 'Hybrid', 'green')
        ]
        
        # Plot 2a: Log-log runtime scaling
        ax = axes[0]
        
        for time_col, label, color in algorithms:
            if time_col in large_df.columns:
                grouped = large_df.groupby('n')[time_col].agg(['mean', 'std'])
                sizes = grouped.index
                means = grouped['mean']
                
                ax.loglog(sizes, means, 'o-', label=label, color=color, linewidth=2, markersize=8)
        
        # Add reference lines for complexity classes
        n_ref = np.array([20, 200])
        ax.loglog(n_ref, n_ref**2 / 1000, '--', color='gray', alpha=0.5, label='O(n²)')
        ax.loglog(n_ref, n_ref * np.log(n_ref) / 10, '--', color='purple', alpha=0.5, label='O(n log n)')
        
        ax.set_xlabel('Problem Size (n)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Runtime Scaling (Log-Log)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, which='both')
        
        # Plot 2b: Linear scale for clarity
        ax = axes[1]
        
        for time_col, label, color in algorithms:
            if time_col in large_df.columns:
                grouped = large_df.groupby('n')[time_col].agg(['mean', 'std'])
                sizes = grouped.index
                means = grouped['mean']
                stds = grouped['std']
                
                ax.plot(sizes, means, 'o-', label=label, color=color, linewidth=2, markersize=8)
                ax.fill_between(sizes, means - stds, means + stds, alpha=0.2, color=color)
        
        ax.set_xlabel('Problem Size (n)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Runtime Scaling (Linear)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/2_runtime_scaling_large_n.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 2_runtime_scaling_large_n.png")
    
    def plot_pareto_fronts(self):
        """
        Plot 3: Pareto Fronts (Quality vs Time)
        Shows trade-offs between solution quality and runtime
        """
        print("\n[3/7] Generating Pareto front plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        problem_sizes = [10, 20, 30, 50, 75, 100]
        
        for idx, n in enumerate(problem_sizes):
            ax = axes[idx // 3, idx % 3]
            
            subset = self.df[self.df['n'] == n].copy()
            
            if len(subset) == 0:
                ax.text(0.5, 0.5, f'No data for n={n}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'n = {n}')
                continue
            
            # Plot each algorithm
            algorithms = [
                ('christofides', 'Christofides', 'blue', 'o'),
                ('2opt', '2-Opt', 'orange', 's'),
                ('hybrid', 'Hybrid', 'green', '^')
            ]
            
            for algo, label, color, marker in algorithms:
                time_col = f'{algo}_time'
                cost_col = f'{algo}_cost'
                
                if time_col in subset.columns and cost_col in subset.columns:
                    times = subset[time_col].dropna()
                    costs = subset[cost_col].dropna()
                    
                    if len(times) > 0 and len(costs) > 0:
                        ax.scatter(times, costs, label=label, color=color, 
                                 marker=marker, s=100, alpha=0.6, edgecolors='black')
            
            ax.set_xlabel('Runtime (seconds)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Tour Cost', fontsize=10, fontweight='bold')
            ax.set_title(f'n = {n}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Pareto Fronts: Solution Quality vs Runtime', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/3_pareto_fronts.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 3_pareto_fronts.png")
    
    def plot_algorithm_comparison_detailed(self):
        """
        Plot 4: Detailed Algorithm Comparison
        Side-by-side comparison of all metrics
        """
        print("\n[4/7] Generating detailed algorithm comparison...")
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        algorithms = [
            ('christofides', 'Christofides', 'blue'),
            ('2opt', '2-Opt', 'orange'),
            ('hybrid', 'Hybrid', 'green')
        ]
        
        # 1. Average approximation ratio
        ax1 = fig.add_subplot(gs[0, 0])
        small_df = self.df[self.df['n'] <= 15].copy()
        ratios = []
        labels = []
        colors = []
        
        for algo, label, color in algorithms:
            ratio_col = f'{algo}_ratio'
            if ratio_col in small_df.columns:
                valid = small_df[ratio_col].dropna()
                if len(valid) > 0:
                    ratios.append(valid.mean())
                    labels.append(label)
                    colors.append(color)
        
        bars = ax1.bar(range(len(labels)), ratios, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('Avg Approximation Ratio', fontweight='bold')
        ax1.set_title('Average Approximation Ratio\n(small instances)', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Average runtime (medium instances)
        ax2 = fig.add_subplot(gs[0, 1])
        medium_df = self.df[(self.df['n'] >= 20) & (self.df['n'] <= 50)].copy()
        times = []
        labels = []
        colors = []
        
        for algo, label, color in algorithms:
            time_col = f'{algo}_time'
            if time_col in medium_df.columns:
                valid = medium_df[time_col].dropna()
                if len(valid) > 0:
                    times.append(valid.mean())
                    labels.append(label)
                    colors.append(color)
        
        bars = ax2.bar(range(len(labels)), times, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Avg Runtime (seconds)', fontweight='bold')
        ax2.set_title('Average Runtime\n(medium instances, n=20-50)', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Success rate
        ax3 = fig.add_subplot(gs[0, 2])
        success_rates = []
        labels = []
        colors = []
        
        for algo, label, color in algorithms:
            cost_col = f'{algo}_cost'
            if cost_col in self.df.columns:
                total = len(self.df)
                successful = self.df[cost_col].notna().sum()
                success_rate = (successful / total) * 100
                success_rates.append(success_rate)
                labels.append(label)
                colors.append(color)
        
        bars = ax3.bar(range(len(labels)), success_rates, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels)
        ax3.set_ylabel('Success Rate (%)', fontweight='bold')
        ax3.set_title('Success Rate\n(all instances)', fontweight='bold')
        ax3.set_ylim([0, 105])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Distribution of approximation ratios
        ax4 = fig.add_subplot(gs[1, :])
        small_df = self.df[self.df['n'] <= 15].copy()
        
        data_to_plot = []
        labels_to_plot = []
        
        for algo, label, color in algorithms:
            ratio_col = f'{algo}_ratio'
            if ratio_col in small_df.columns:
                valid = small_df[ratio_col].dropna()
                if len(valid) > 0:
                    data_to_plot.append(valid)
                    labels_to_plot.append(label)
        
        positions = range(1, len(data_to_plot) + 1)
        parts = ax4.violinplot(data_to_plot, positions=positions, showmeans=True, showmedians=True)
        
        for i, (pc, (_, _, color)) in enumerate(zip(parts['bodies'], algorithms)):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Optimal')
        ax4.set_xticks(positions)
        ax4.set_xticklabels(labels_to_plot)
        ax4.set_ylabel('Approximation Ratio', fontweight='bold')
        ax4.set_title('Distribution of Approximation Ratios (Violin Plot)', fontweight='bold', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Comprehensive Algorithm Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f'{self.output_dir}/4_algorithm_comparison_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 4_algorithm_comparison_detailed.png")
    
    def plot_2opt_improvement_analysis(self):
        """
        Plot 5: 2-Opt Improvement Analysis
        Shows how 2-Opt improves different starting solutions
        """
        print("\n[5/7] Generating 2-Opt improvement analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Filter instances where we have both initial and improved solutions
        comparison_df = self.df[
            (self.df['christofides_cost'].notna()) & 
            (self.df['hybrid_cost'].notna()) &
            (self.df['2opt_cost'].notna())
        ].copy()
        
        # Plot 5a: Improvement from Christofides
        ax = axes[0]
        comparison_df['chris_improvement'] = (
            (comparison_df['christofides_cost'] - comparison_df['hybrid_cost']) / 
            comparison_df['christofides_cost'] * 100
        )
        
        grouped = comparison_df.groupby('n')['chris_improvement'].agg(['mean', 'std'])
        sizes = grouped.index
        means = grouped['mean']
        stds = grouped['std']
        
        ax.plot(sizes, means, 'o-', color='green', linewidth=2, markersize=8, label='Mean Improvement')
        ax.fill_between(sizes, means - stds, means + stds, alpha=0.3, color='green')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Problem Size (n)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
        ax.set_title('2-Opt Improvement over Christofides', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5b: Number of 2-Opt iterations
        ax = axes[1]
        
        if 'hybrid_iterations' in comparison_df.columns:
            grouped = comparison_df.groupby('n')['hybrid_iterations'].agg(['mean', 'std'])
            sizes = grouped.index
            means = grouped['mean']
            stds = grouped['std']
            
            ax.plot(sizes, means, 'o-', color='purple', linewidth=2, markersize=8)
            ax.fill_between(sizes, means - stds, means + stds, alpha=0.3, color='purple')
            
            ax.set_xlabel('Problem Size (n)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Number of Iterations', fontsize=12, fontweight='bold')
            ax.set_title('2-Opt Iterations until Convergence', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/5_2opt_improvement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 5_2opt_improvement_analysis.png")
    
    def plot_distribution_comparison(self):
        """
        Plot 6: Compare performance across different instance distributions
        """
        print("\n[6/7] Generating distribution comparison...")
        
        if 'distribution' not in self.df.columns:
            print("  ⚠ No distribution data, skipping...")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        distributions = self.df['distribution'].unique()
        small_df = self.df[self.df['n'] <= 15].copy()
        
        # Plot 6a: Approximation ratios by distribution
        ax = axes[0]
        x_pos = np.arange(len(distributions))
        width = 0.25
        
        algorithms = [
            ('christofides_ratio', 'Christofides', 'blue'),
            ('2opt_ratio', '2-Opt', 'orange'),
            ('hybrid_ratio', 'Hybrid', 'green')
        ]
        
        for i, (ratio_col, label, color) in enumerate(algorithms):
            means = []
            stds = []
            for dist in distributions:
                subset = small_df[small_df['distribution'] == dist][ratio_col].dropna()
                means.append(subset.mean() if len(subset) > 0 else 0)
                stds.append(subset.std() if len(subset) > 0 else 0)
            
            ax.bar(x_pos + i*width, means, width, label=label, color=color, 
                   alpha=0.7, yerr=stds, capsize=5)
        
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Instance Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Approximation Ratio', fontsize=12, fontweight='bold')
        ax.set_title('Performance by Instance Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(distributions)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 6b: Runtime by distribution (large instances)
        ax = axes[1]
        large_df = self.df[self.df['n'] >= 50].copy()
        
        for i, (time_col, label, color) in enumerate([
            ('christofides_time', 'Christofides', 'blue'),
            ('2opt_time', '2-Opt', 'orange'),
            ('hybrid_time', 'Hybrid', 'green')
        ]):
            means = []
            stds = []
            for dist in distributions:
                subset = large_df[large_df['distribution'] == dist][time_col].dropna()
                means.append(subset.mean() if len(subset) > 0 else 0)
                stds.append(subset.std() if len(subset) > 0 else 0)
            
            ax.bar(x_pos + i*width, means, width, label=label, color=color, 
                   alpha=0.7, yerr=stds, capsize=5)
        
        ax.set_xlabel('Instance Distribution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Runtime by Instance Type (Large n)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(distributions)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/6_distribution_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 6_distribution_comparison.png")
    
    def generate_summary_table(self):
        """
        Plot 7: Generate summary table as image
        """
        print("\n[7/7] Generating summary table...")
        
        # Create summary statistics
        summary_data = []
        
        algorithms = [
            ('christofides', 'Christofides'),
            ('2opt', '2-Opt'),
            ('hybrid', 'Hybrid (Chris+2Opt)')
        ]
        
        for algo, name in algorithms:
            small_df = self.df[self.df['n'] <= 15].copy()
            large_df = self.df[self.df['n'] >= 50].copy()
            
            row = {'Algorithm': name}
            
            # Small instance metrics
            ratio_col = f'{algo}_ratio'
            if ratio_col in small_df.columns:
                ratios = small_df[ratio_col].dropna()
                row['Avg Ratio\n(small n)'] = f"{ratios.mean():.3f}"
                row['Max Ratio\n(small n)'] = f"{ratios.max():.3f}"
            else:
                row['Avg Ratio\n(small n)'] = 'N/A'
                row['Max Ratio\n(small n)'] = 'N/A'
            
            # Large instance metrics
            time_col = f'{algo}_time'
            if time_col in large_df.columns:
                times = large_df[time_col].dropna()
                row['Avg Time\n(large n)'] = f"{times.mean():.2f}s"
            else:
                row['Avg Time\n(large n)'] = 'N/A'
            
            summary_data.append(row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        
        df_summary = pd.DataFrame(summary_data)
        table = ax.table(cellText=df_summary.values, colLabels=df_summary.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(df_summary.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        colors = ['#E8F5E9', '#C8E6C9', '#A5D6A7']
        for i in range(len(summary_data)):
            for j in range(len(df_summary.columns)):
                table[(i+1, j)].set_facecolor(colors[i])
        
        plt.title('TSP Algorithms: Summary Statistics', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(f'{self.output_dir}/7_summary_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: 7_summary_table.png")
    
    def generate_all_plots(self):
        """Generate all plots"""
        print("\n" + "="*70)
        print("GENERATING ALL PLOTS")
        print("="*70)
        
        self.plot_approximation_ratios_small_n()
        self.plot_runtime_scaling_large_n()
        self.plot_pareto_fronts()
        self.plot_algorithm_comparison_detailed()
        self.plot_2opt_improvement_analysis()
        self.plot_distribution_comparison()
        self.generate_summary_table()
        
        print("\n" + "="*70)
        print("✓ ALL PLOTS GENERATED!")
        print(f"Check {self.output_dir}/ for all visualizations")
        print("="*70)


def main():
    """Main execution"""
    plotter = TSPPlotGenerator()
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()
