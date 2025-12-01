"""
Comprehensive Experimental Suite for TSP Algorithms
Author: Saharsh (TSP)
Date: December 2025

This script runs experiments across multiple problem sizes and generates
all required data for analysis and plotting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
import time
import json
from datetime import datetime
import sys
import os

# Add parent directory and algorithm directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'christofides'))
sys.path.insert(0, os.path.join(parent_dir, '2-opt'))
sys.path.insert(0, os.path.join(parent_dir, 'LP'))

# Import algorithm classes
from christofides import ChristofidesAlgorithm
from two_opt import TwoOptLocalSearch
from tsp_lp import TSPLPRelaxation

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class TSPExperimentRunner:
    """
    Comprehensive experiment runner for TSP algorithms
    """
    
    def __init__(self, seed=42):
        """Initialize experiment runner"""
        self.seed = seed
        np.random.seed(seed)
        self.results = []
        
    def generate_random_instance(self, n, distribution='uniform'):
        """
        Generate random TSP instance
        
        Args:
            n: Number of cities
            distribution: 'uniform', 'clustered', or 'grid'
        """
        if distribution == 'uniform':
            points = np.random.rand(n, 2) * 100
        elif distribution == 'clustered':
            # Create 3-5 clusters
            num_clusters = np.random.randint(3, 6)
            points = []
            cities_per_cluster = n // num_clusters
            for i in range(num_clusters):
                center = np.random.rand(2) * 100
                cluster_points = np.random.randn(cities_per_cluster, 2) * 5 + center
                points.append(cluster_points)
            points = np.vstack(points)[:n]
        elif distribution == 'grid':
            side = int(np.sqrt(n))
            x = np.linspace(0, 100, side)
            y = np.linspace(0, 100, side)
            xx, yy = np.meshgrid(x, y)
            points = np.column_stack([xx.ravel(), yy.ravel()])[:n]
            # Add small noise
            points += np.random.randn(n, 2) * 2
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
            
        return points
    
    def compute_optimal_or_lower_bound(self, points, n):
        """
        Compute optimal solution for small n, or LP lower bound for larger n
        """
        if n <= 10:
            # Use LP with subtour elimination as proxy for optimal
            try:
                lp_solver = TSPLPRelaxation(points)
                result = lp_solver.solve_with_subtour_elimination(
                    max_subtours=20, 
                    verbose=False
                )
                return result['objective'], 'LP_subtour'
            except:
                return None, 'failed'
        elif n <= 15:
            # Use LP relaxation as lower bound
            try:
                lp_solver = TSPLPRelaxation(points)
                result = lp_solver.solve_basic_lp(verbose=False)
                return result['objective'], 'LP_basic'
            except:
                return None, 'failed'
        else:
            # For large n, use best algorithm result as proxy
            return None, 'none'
    
    def run_single_experiment(self, n, instance_num, distribution='uniform'):
        """
        Run all algorithms on a single instance
        """
        print(f"\n{'='*60}")
        print(f"Experiment: n={n}, instance={instance_num}, dist={distribution}")
        print(f"{'='*60}")
        
        # Generate instance
        points = self.generate_random_instance(n, distribution)
        dist_matrix = distance_matrix(points, points)
        
        # Get optimal/lower bound
        opt_value, opt_type = self.compute_optimal_or_lower_bound(points, n)
        print(f"Reference value ({opt_type}): {opt_value:.2f}" if opt_value else "No reference value")
        
        experiment_data = {
            'n': n,
            'instance': instance_num,
            'distribution': distribution,
            'opt_value': opt_value,
            'opt_type': opt_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Run Christofides
        print("\n1. Running Christofides Algorithm...")
        try:
            chris_solver = ChristofidesAlgorithm(points)
            chris_result = chris_solver.run(verbose=False)
            
            experiment_data['christofides_cost'] = chris_result['cost']
            experiment_data['christofides_time'] = chris_result['runtime']
            experiment_data['christofides_ratio'] = (
                chris_result['cost'] / opt_value if opt_value else None
            )
            print(f"   ✓ Cost: {chris_result['cost']:.2f}, Time: {chris_result['runtime']:.4f}s")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            experiment_data['christofides_cost'] = None
            experiment_data['christofides_time'] = None
            experiment_data['christofides_ratio'] = None
        
        # 2. Run 2-Opt from random start
        print("\n2. Running 2-Opt (random start)...")
        try:
            two_opt_solver = TwoOptLocalSearch(points)
            two_opt_result = two_opt_solver.run(initial_tour=None, verbose=False)
            
            experiment_data['2opt_cost'] = two_opt_result['cost']
            experiment_data['2opt_time'] = two_opt_result['runtime']
            experiment_data['2opt_ratio'] = (
                two_opt_result['cost'] / opt_value if opt_value else None
            )
            experiment_data['2opt_iterations'] = two_opt_result['iterations']
            print(f"   ✓ Cost: {two_opt_result['cost']:.2f}, Time: {two_opt_result['runtime']:.4f}s")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            experiment_data['2opt_cost'] = None
            experiment_data['2opt_time'] = None
            experiment_data['2opt_ratio'] = None
            experiment_data['2opt_iterations'] = None
        
        # 3. Run Hybrid (2-Opt from Christofides)
        print("\n3. Running Hybrid (Christofides + 2-Opt)...")
        try:
            if chris_result:
                hybrid_solver = TwoOptLocalSearch(points)
                hybrid_result = hybrid_solver.run(
                    initial_tour=chris_result['tour'], 
                    verbose=False
                )
                
                experiment_data['hybrid_cost'] = hybrid_result['cost']
                experiment_data['hybrid_time'] = (
                    chris_result['runtime'] + hybrid_result['runtime']
                )
                experiment_data['hybrid_ratio'] = (
                    hybrid_result['cost'] / opt_value if opt_value else None
                )
                experiment_data['hybrid_iterations'] = hybrid_result['iterations']
                print(f"   ✓ Cost: {hybrid_result['cost']:.2f}, Time: {experiment_data['hybrid_time']:.4f}s")
            else:
                raise Exception("Christofides failed, cannot run hybrid")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            experiment_data['hybrid_cost'] = None
            experiment_data['hybrid_time'] = None
            experiment_data['hybrid_ratio'] = None
            experiment_data['hybrid_iterations'] = None
        
        # 4. Run LP Relaxation (only for small n)
        if n <= 15:
            print("\n4. Running LP Relaxation...")
            try:
                lp_solver = TSPLPRelaxation(points)
                lp_result = lp_solver.solve_basic_lp(verbose=False)
                
                experiment_data['lp_objective'] = lp_result['objective']
                experiment_data['lp_time'] = lp_result['runtime']
                experiment_data['lp_status'] = lp_result.get('status', 'N/A')
                print(f"   ✓ Lower Bound: {lp_result['objective']:.2f}, Time: {lp_result['runtime']:.4f}s")
            except Exception as e:
                print(f"   ✗ Failed: {e}")
                experiment_data['lp_objective'] = None
                experiment_data['lp_time'] = None
                experiment_data['lp_status'] = 'Failed'
        
        self.results.append(experiment_data)
        return experiment_data
    
    def run_experiments(self, 
                       small_sizes=[5, 8, 10, 12, 15],
                       large_sizes=[20, 30, 50, 75, 100, 150, 200],
                       instances_per_size=10,
                       distributions=['uniform', 'clustered']):
        """
        Run comprehensive experiments across problem sizes
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE TSP EXPERIMENTAL SUITE")
        print("="*70)
        print(f"Small problem sizes: {small_sizes}")
        print(f"Large problem sizes: {large_sizes}")
        print(f"Instances per size: {instances_per_size}")
        print(f"Distributions: {distributions}")
        print("="*70)
        
        total_experiments = (len(small_sizes) + len(large_sizes)) * instances_per_size * len(distributions)
        experiment_count = 0
        
        # Run experiments on small instances (with LP)
        print("\n" + "="*70)
        print("PHASE 1: SMALL INSTANCES (with LP lower bounds)")
        print("="*70)
        
        for n in small_sizes:
            for dist in distributions:
                for instance in range(instances_per_size):
                    experiment_count += 1
                    print(f"\nProgress: {experiment_count}/{total_experiments}")
                    self.run_single_experiment(n, instance, dist)
        
        # Run experiments on large instances (no LP)
        print("\n" + "="*70)
        print("PHASE 2: LARGE INSTANCES (runtime scaling)")
        print("="*70)
        
        for n in large_sizes:
            for dist in distributions:
                for instance in range(instances_per_size):
                    experiment_count += 1
                    print(f"\nProgress: {experiment_count}/{total_experiments}")
                    self.run_single_experiment(n, instance, dist)
        
        print("\n" + "="*70)
        print("✓ ALL EXPERIMENTS COMPLETE!")
        print(f"Total experiments run: {len(self.results)}")
        print("="*70)
    
    def save_results(self, filename='tsp_experiments_results.csv'):
        """Save results to CSV"""
        df = pd.DataFrame(self.results)
        filepath = os.path.join('experiments', filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Results saved to: {filepath}")
        return df
    
    def generate_summary_statistics(self):
        """Generate summary statistics"""
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        # Group by problem size
        for n in sorted(df['n'].unique()):
            subset = df[df['n'] == n]
            print(f"\nProblem size n={n} ({len(subset)} instances)")
            print("-" * 60)
            
            for algo in ['christofides', '2opt', 'hybrid']:
                cost_col = f'{algo}_cost'
                time_col = f'{algo}_time'
                ratio_col = f'{algo}_ratio'
                
                if cost_col in subset.columns:
                    valid_costs = subset[cost_col].dropna()
                    valid_times = subset[time_col].dropna()
                    valid_ratios = subset[ratio_col].dropna()
                    
                    if len(valid_costs) > 0:
                        print(f"\n{algo.upper()}:")
                        print(f"  Cost:  Mean={valid_costs.mean():.2f}, Std={valid_costs.std():.2f}")
                        print(f"  Time:  Mean={valid_times.mean():.4f}s, Std={valid_times.std():.4f}s")
                        if len(valid_ratios) > 0:
                            print(f"  Ratio: Mean={valid_ratios.mean():.4f}, Max={valid_ratios.max():.4f}")


def main():
    """Main execution function"""
    
    # Create experiments directory if it doesn't exist
    os.makedirs('experiments', exist_ok=True)
    
    # Initialize experiment runner
    runner = TSPExperimentRunner(seed=42)
    
    # Run experiments
    # For testing, use smaller numbers:
    runner.run_experiments(
        small_sizes=[5, 8, 10, 12, 15],           # With LP lower bounds
        large_sizes=[20, 30, 50, 75, 100],        # Runtime scaling
        instances_per_size=10,                     # 10 instances per size
        distributions=['uniform', 'clustered']     # 2 distributions
    )
    
    # Save results
    df = runner.save_results()
    
    # Generate summary statistics
    runner.generate_summary_statistics()
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Run: python experiments/generate_plots.py")
    print("   (to generate all required visualizations)")
    print("\n2. Check experiments/ folder for:")
    print("   - tsp_experiments_results.csv (raw data)")
    print("   - All generated plots")
    print("="*70)


if __name__ == "__main__":
    main()
