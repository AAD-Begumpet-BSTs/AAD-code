"""
Run TSP Algorithms on Benchmark Instances
Author: Saharsh (TSP)
Date: December 2025

This script runs all TSP algorithms (Exact, Christofides, 2-Opt, Hybrid)
on the benchmark instances and saves results in the required JSON format.
"""

import json
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add paths to TSP algorithm implementations
current_dir = Path(__file__).parent
tsp_dir = current_dir.parent / 'TSP'
sys.path.insert(0, str(tsp_dir / 'Algorithms'))
sys.path.insert(0, str(current_dir.parent / 'pranshul' / 'src'))

# Import TSP algorithms
try:
    from christofides import ChristofidesAlgorithm
    from two_opt import TwoOptLocalSearch
    from tsp_lp import TSPLPRelaxation
    from algorithms.exact_solvers import TSPExact
    print("✓ All TSP algorithms imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please ensure all TSP algorithm files are in place")
    sys.exit(1)


class TSPBenchmarkRunner:
    """Run TSP algorithms on benchmark instances"""
    
    def __init__(self):
        self.datasets_dir = Path(__file__).parent / 'datasets' / 'tsp'
        self.results_dir = Path(__file__).parent / 'collected_results' / 'tsp'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'small_n').mkdir(exist_ok=True)
        (self.results_dir / 'large_n').mkdir(exist_ok=True)
        
    def load_instance(self, filename):
        """Load a TSP instance from JSON file"""
        filepath = self.datasets_dir / filename
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        points = np.array(data['points'])
        dist_matrix = np.array(data['distances'])
        n = data['n']
        
        return n, points, dist_matrix
    
    def run_exact_solver(self, dist_matrix, n, timeout=300):
        """Run Held-Karp exact solver (only for small n)"""
        if n > 12:  # Too slow for n > 12
            return None, None, 'skipped'
        
        try:
            exact_solver = TSPExact()
            start_time = time.time()
            cost = exact_solver.solve(dist_matrix.tolist())
            runtime = time.time() - start_time
            
            if runtime > timeout:
                return None, None, 'timeout'
            
            return cost, runtime, 'optimal'
        except Exception as e:
            print(f"    Exact solver failed: {e}")
            return None, None, 'failed'
    
    def run_lp_relaxation(self, points, n):
        """Run LP relaxation for lower bound"""
        if n > 20:  # LP too slow for large n
            return None, None, 'skipped'
        
        try:
            lp_solver = TSPLPRelaxation(points)
            
            if n <= 15:
                # Use subtour elimination for tight bounds
                result = lp_solver.solve_with_subtour_elimination(
                    max_subtours=20, 
                    verbose=False
                )
            else:
                # Use basic LP for larger instances
                result = lp_solver.solve_basic_lp(verbose=False)
            
            return result['objective'], result['runtime'], result['status']
        except Exception as e:
            print(f"    LP solver failed: {e}")
            return None, None, 'failed'
    
    def run_christofides(self, points):
        """Run Christofides algorithm"""
        try:
            chris_solver = ChristofidesAlgorithm(points)
            result = chris_solver.run(verbose=False)
            return result['cost'], result['runtime'], 'success'
        except Exception as e:
            print(f"    Christofides failed: {e}")
            return None, None, 'failed'
    
    def run_two_opt(self, points):
        """Run 2-Opt from random start"""
        try:
            two_opt_solver = TwoOptLocalSearch(points)
            result = two_opt_solver.run(initial_tour=None, verbose=False, max_iterations=5000)
            return result['cost'], result['runtime'], result['iterations'], 'success'
        except Exception as e:
            print(f"    2-Opt failed: {e}")
            return None, None, None, 'failed'
    
    def run_hybrid(self, points):
        """Run Hybrid (Christofides + 2-Opt)"""
        try:
            # Step 1: Christofides
            chris_solver = ChristofidesAlgorithm(points)
            chris_result = chris_solver.run(verbose=False)
            
            # Step 2: 2-Opt refinement
            two_opt_solver = TwoOptLocalSearch(points)
            two_opt_result = two_opt_solver.run(
                initial_tour=chris_result['tour'], 
                verbose=False,
                max_iterations=5000
            )
            
            total_time = chris_result['runtime'] + two_opt_result['runtime']
            
            return (
                two_opt_result['cost'], 
                total_time,
                chris_result['runtime'],
                two_opt_result['runtime'],
                two_opt_result['iterations'],
                'success'
            )
        except Exception as e:
            print(f"    Hybrid failed: {e}")
            return None, None, None, None, None, 'failed'
    
    def run_single_instance(self, filename):
        """Run all algorithms on a single instance"""
        print(f"\n{'='*70}")
        print(f"Processing: {filename}")
        print(f"{'='*70}")
        
        # Load instance
        n, points, dist_matrix = self.load_instance(filename)
        print(f"Instance size: n={n}")
        
        # Prepare result structure
        result = {
            'instance_file': filename,
            'n': n,
            'algorithms': {}
        }
        
        # Determine if small or large instance
        is_small = n <= 20
        
        # 1. Run Exact Solver (only for very small instances)
        if n <= 12:
            print("  1. Running Exact Solver (Held-Karp)...")
            exact_cost, exact_time, exact_status = self.run_exact_solver(dist_matrix, n)
            result['exact_cost'] = exact_cost
            result['exact_time'] = exact_time
            result['exact_status'] = exact_status
            if exact_cost:
                print(f"     ✓ Exact: {exact_cost:.2f} (time: {exact_time:.4f}s)")
        else:
            result['exact_cost'] = None
            result['exact_time'] = None
            result['exact_status'] = 'skipped'
        
        # 2. Run LP Relaxation (for lower bounds)
        if is_small:
            print("  2. Running LP Relaxation...")
            lp_value, lp_time, lp_status = self.run_lp_relaxation(points, n)
            result['lp_value'] = lp_value
            result['lp_time'] = lp_time
            result['lp_status'] = lp_status
            if lp_value:
                print(f"     ✓ LP Lower Bound: {lp_value:.2f} (time: {lp_time:.4f}s)")
        else:
            result['lp_value'] = None
            result['lp_time'] = None
            result['lp_status'] = 'skipped'
        
        # 3. Run Christofides
        print("  3. Running Christofides Algorithm...")
        chris_cost, chris_time, chris_status = self.run_christofides(points)
        result['christofides_cost'] = chris_cost
        result['christofides_time'] = chris_time
        result['christofides_status'] = chris_status
        if chris_cost:
            print(f"     ✓ Christofides: {chris_cost:.2f} (time: {chris_time:.4f}s)")
        
        # 4. Run 2-Opt
        print("  4. Running 2-Opt Local Search...")
        two_opt_cost, two_opt_time, two_opt_iters, two_opt_status = self.run_two_opt(points)
        result['two_opt_cost'] = two_opt_cost
        result['two_opt_time'] = two_opt_time
        result['two_opt_iterations'] = two_opt_iters
        result['two_opt_status'] = two_opt_status
        if two_opt_cost:
            print(f"     ✓ 2-Opt: {two_opt_cost:.2f} (time: {two_opt_time:.4f}s, iters: {two_opt_iters})")
        
        # 5. Run Hybrid
        print("  5. Running Hybrid (Christofides + 2-Opt)...")
        hybrid_result = self.run_hybrid(points)
        if hybrid_result[5] == 'success':
            hybrid_cost, hybrid_total_time, hybrid_chris_time, hybrid_2opt_time, hybrid_iters, _ = hybrid_result
            result['hybrid_cost'] = hybrid_cost
            result['hybrid_total_time'] = hybrid_total_time
            result['hybrid_christofides_time'] = hybrid_chris_time
            result['hybrid_2opt_time'] = hybrid_2opt_time
            result['hybrid_iterations'] = hybrid_iters
            result['hybrid_status'] = 'success'
            print(f"     ✓ Hybrid: {hybrid_cost:.2f} (time: {hybrid_total_time:.4f}s, iters: {hybrid_iters})")
        else:
            result['hybrid_cost'] = None
            result['hybrid_total_time'] = None
            result['hybrid_status'] = 'failed'
        
        # Calculate approximation ratios (if we have reference values)
        reference = None
        if result['exact_cost']:
            reference = result['exact_cost']
            ref_type = 'exact'
        elif result['lp_value']:
            reference = result['lp_value']
            ref_type = 'lp'
        
        if reference:
            result['reference_value'] = reference
            result['reference_type'] = ref_type
            
            if chris_cost:
                result['christofides_ratio'] = chris_cost / reference
                print(f"     Christofides ratio: {result['christofides_ratio']:.4f}")
            if two_opt_cost:
                result['two_opt_ratio'] = two_opt_cost / reference
                print(f"     2-Opt ratio: {result['two_opt_ratio']:.4f}")
            if hybrid_result[5] == 'success':
                result['hybrid_ratio'] = hybrid_cost / reference
                print(f"     Hybrid ratio: {result['hybrid_ratio']:.4f}")
        
        # Save result
        output_dir = self.results_dir / ('small_n' if is_small else 'large_n')
        output_file = output_dir / f"{filename.replace('.json', '_results.json')}"
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✓ Results saved to: {output_file}")
        
        return result
    
    def run_all_instances(self, pattern='*.json', small_only=False, large_only=False):
        """Run all algorithms on all benchmark instances"""
        
        # Get all instance files
        instance_files = sorted(self.datasets_dir.glob(pattern))
        
        if not instance_files:
            print(f"No instance files found matching pattern: {pattern}")
            return
        
        # Filter by size if requested
        if small_only:
            instance_files = [f for f in instance_files if self._get_n_from_filename(f.name) <= 20]
        elif large_only:
            instance_files = [f for f in instance_files if self._get_n_from_filename(f.name) > 20]
        
        print(f"\n{'='*70}")
        print(f"TSP BENCHMARK EXPERIMENTS")
        print(f"{'='*70}")
        print(f"Total instances to process: {len(instance_files)}")
        print(f"Algorithms: Exact (n≤12), LP, Christofides, 2-Opt, Hybrid")
        print(f"{'='*70}")
        
        results = []
        failed_instances = []
        
        for i, instance_file in enumerate(instance_files, 1):
            print(f"\n[{i}/{len(instance_files)}] ", end='')
            
            try:
                result = self.run_single_instance(instance_file.name)
                results.append(result)
            except Exception as e:
                print(f"✗ Failed to process {instance_file.name}: {e}")
                failed_instances.append(instance_file.name)
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'='*70}")
        print(f"Total instances processed: {len(results)}")
        print(f"Failed instances: {len(failed_instances)}")
        if failed_instances:
            print("Failed files:")
            for f in failed_instances:
                print(f"  - {f}")
        print(f"{'='*70}")
        
        # Create summary file
        summary = {
            'total_instances': len(instance_files),
            'successful': len(results),
            'failed': len(failed_instances),
            'failed_files': failed_instances,
            'results_summary': self._create_summary_stats(results)
        }
        
        summary_file = self.results_dir / 'tsp_experiments_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        
        return results
    
    def _get_n_from_filename(self, filename):
        """Extract n from filename like 'tsp_n20_s100.json'"""
        try:
            parts = filename.split('_')
            n_part = [p for p in parts if p.startswith('n')][0]
            return int(n_part[1:])
        except:
            return 0
    
    def _create_summary_stats(self, results):
        """Create summary statistics from results"""
        small_n = [r for r in results if r['n'] <= 20]
        large_n = [r for r in results if r['n'] > 20]
        
        def avg_ratio(results, key):
            ratios = [r.get(key) for r in results if r.get(key)]
            return sum(ratios) / len(ratios) if ratios else None
        
        def avg_time(results, key):
            times = [r.get(key) for r in results if r.get(key)]
            return sum(times) / len(times) if times else None
        
        return {
            'small_n': {
                'count': len(small_n),
                'avg_christofides_ratio': avg_ratio(small_n, 'christofides_ratio'),
                'avg_hybrid_ratio': avg_ratio(small_n, 'hybrid_ratio'),
                'avg_christofides_time': avg_time(small_n, 'christofides_time'),
                'avg_hybrid_time': avg_time(small_n, 'hybrid_total_time')
            },
            'large_n': {
                'count': len(large_n),
                'avg_christofides_time': avg_time(large_n, 'christofides_time'),
                'avg_hybrid_time': avg_time(large_n, 'hybrid_total_time')
            }
        }


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TSP algorithms on benchmark instances')
    parser.add_argument('--small-only', action='store_true', help='Process only small instances (n≤20)')
    parser.add_argument('--large-only', action='store_true', help='Process only large instances (n>20)')
    parser.add_argument('--pattern', default='*.json', help='File pattern to match (default: *.json)')
    parser.add_argument('--single', help='Process a single instance file')
    
    args = parser.parse_args()
    
    runner = TSPBenchmarkRunner()
    
    if args.single:
        # Run single instance
        runner.run_single_instance(args.single)
    else:
        # Run all instances
        runner.run_all_instances(
            pattern=args.pattern,
            small_only=args.small_only,
            large_only=args.large_only
        )
    
    print("\n✓ TSP benchmark experiments complete!")
    print("Results saved in: analysis/collected_results/tsp/")


if __name__ == '__main__':
    main()
