#!/usr/bin/env python3
"""
Comprehensive Statistical Analysis Runner
Performs statistical significance tests on all experimental results.
"""

import sys
import os
import json
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_tests import (
    wilcoxon_signed_rank_test,
    mann_whitney_u_test,
    paired_t_test,
    effect_size_cohens_d
)


def load_json_safely(filepath: str) -> List[Dict]:
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"  Warning: File not found - {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"  Warning: Invalid JSON - {filepath}")
        return []


def analyze_vc_results():
    """Analyze Vertex Cover experimental results."""
    print("\n" + "="*80)
    print("VERTEX COVER STATISTICAL ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Load small-n data (with exact comparisons)
    small_data = load_json_safely("../../VC/results/vc_small_experiments.json")
    if small_data:
        print(f"\n✓ Loaded {len(small_data)} small-n instances")
        
        # Extract data
        pd_ratios = [d['primal_dual_ratio'] for d in small_data if 'primal_dual_ratio' in d]
        hybrid_ratios = [d['hybrid_ratio'] for d in small_data if 'hybrid_ratio' in d]
        pd_times = [d['timings']['primal_dual'] for d in small_data if 'timings' in d and 'primal_dual' in d['timings']]
        hybrid_times = [d['timings']['hybrid_total'] for d in small_data if 'timings' in d and 'hybrid_total' in d['timings']]
        
        print(f"  Primal-Dual avg ratio: {np.mean(pd_ratios):.4f} (n={len(pd_ratios)})")
        print(f"  Hybrid avg ratio: {np.mean(hybrid_ratios):.4f} (n={len(hybrid_ratios)})")
        
        # Test 1: Quality comparison
        if len(pd_ratios) == len(hybrid_ratios) and len(pd_ratios) >= 3:
            quality_test = wilcoxon_signed_rank_test(pd_ratios, hybrid_ratios)
            quality_effect = effect_size_cohens_d(pd_ratios, hybrid_ratios)
            
            results['small_n_quality'] = {
                'test': quality_test,
                'effect_size': quality_effect,
                'mean_pd': float(np.mean(pd_ratios)),
                'mean_hybrid': float(np.mean(hybrid_ratios)),
                'improvement': float((np.mean(pd_ratios) - np.mean(hybrid_ratios)) / np.mean(pd_ratios) * 100)
            }
            
            print("\n  Quality Comparison (Primal-Dual vs Hybrid):")
            print(f"    p-value: {quality_test['p_value']:.6f}")
            print(f"    Significant: {'YES' if quality_test['significant'] else 'NO'} (α=0.05)")
            print(f"    Cohen's d: {quality_effect:.4f} ({'LARGE' if abs(quality_effect) >= 0.8 else 'MEDIUM' if abs(quality_effect) >= 0.5 else 'SMALL' if abs(quality_effect) >= 0.2 else 'NEGLIGIBLE'})")
            print(f"    Hybrid improvement: {results['small_n_quality']['improvement']:.2f}%")
        
        # Test 2: Runtime comparison
        if len(pd_times) == len(hybrid_times) and len(pd_times) >= 3:
            runtime_test = wilcoxon_signed_rank_test(pd_times, hybrid_times)
            runtime_effect = effect_size_cohens_d(pd_times, hybrid_times)
            
            results['small_n_runtime'] = {
                'test': runtime_test,
                'effect_size': runtime_effect,
                'mean_pd': float(np.mean(pd_times)),
                'mean_hybrid': float(np.mean(hybrid_times))
            }
            
            print("\n  Runtime Comparison (Primal-Dual vs Hybrid):")
            print(f"    PD avg: {np.mean(pd_times)*1000:.4f} ms")
            print(f"    Hybrid avg: {np.mean(hybrid_times)*1000:.4f} ms")
            print(f"    p-value: {runtime_test['p_value']:.6f}")
            print(f"    Significant: {'YES' if runtime_test['significant'] else 'NO'}")
    
    # Load large-n data
    large_data = load_json_safely("../../VC/results/vc_large_experiments.json")
    if large_data:
        print(f"\n✓ Loaded {len(large_data)} large-n instances")
        
        pd_vs_lp = [d['primal_dual_vs_lp'] for d in large_data if 'primal_dual_vs_lp' in d]
        hybrid_vs_lp = [d['hybrid_vs_lp'] for d in large_data if 'hybrid_vs_lp' in d]
        
        print(f"  Primal-Dual / LP avg: {np.mean(pd_vs_lp):.4f}")
        print(f"  Hybrid / LP avg: {np.mean(hybrid_vs_lp):.4f}")
        
        if len(pd_vs_lp) == len(hybrid_vs_lp) and len(pd_vs_lp) >= 3:
            large_test = wilcoxon_signed_rank_test(pd_vs_lp, hybrid_vs_lp)
            large_effect = effect_size_cohens_d(pd_vs_lp, hybrid_vs_lp)
            
            results['large_n_quality'] = {
                'test': large_test,
                'effect_size': large_effect,
                'mean_pd': float(np.mean(pd_vs_lp)),
                'mean_hybrid': float(np.mean(hybrid_vs_lp)),
                'improvement': float((np.mean(pd_vs_lp) - np.mean(hybrid_vs_lp)) / np.mean(pd_vs_lp) * 100)
            }
            
            print("\n  Quality vs LP Bound (Primal-Dual vs Hybrid):")
            print(f"    p-value: {large_test['p_value']:.6f}")
            print(f"    Significant: {'YES' if large_test['significant'] else 'NO'}")
            print(f"    Hybrid improvement: {results['large_n_quality']['improvement']:.2f}%")
    
    return results


def analyze_tsp_results():
    """Analyze TSP experimental results."""
    print("\n" + "="*80)
    print("TSP STATISTICAL ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Try to load TSP results CSV
    csv_file = "../../TSP/Results/tsp_experiments_results.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        print(f"\n✓ Loaded TSP results with {len(df)} instances")
        
        # Filter small instances with exact values
        small_df = df[df['n'] <= 12].copy()
        if len(small_df) > 0:
            print(f"  Small instances (n≤12): {len(small_df)}")
            
            # Check for exact column (might be opt_value)
            exact_col = 'exact_cost' if 'exact_cost' in small_df.columns else 'opt_value'
            
            # Compare Christofides vs Exact
            chris_ratios = (small_df['christofides_cost'] / small_df[exact_col]).dropna()
            if len(chris_ratios) >= 3:
                print(f"\n  Christofides vs OPT:")
                print(f"    Mean ratio: {chris_ratios.mean():.4f}")
                print(f"    Std dev: {chris_ratios.std():.4f}")
                print(f"    Min: {chris_ratios.min():.4f}, Max: {chris_ratios.max():.4f}")
                print(f"    Finds optimal: {(chris_ratios == 1.0).sum()}/{len(chris_ratios)} instances")
                
            results['christofides_quality'] = {
                'mean_ratio': float(chris_ratios.mean()),
                'std_ratio': float(chris_ratios.std()),
                'min_ratio': float(chris_ratios.min()),
                'max_ratio': float(chris_ratios.max()),
                'optimal_count': int((chris_ratios == 1.0).sum()),
                'total_count': int(len(chris_ratios))
            }            # Compare Hybrid vs Exact
            hybrid_ratios = (small_df['hybrid_cost'] / small_df[exact_col]).dropna()
            if len(hybrid_ratios) >= 3:
                print(f"\n  Hybrid vs OPT:")
                print(f"    Mean ratio: {hybrid_ratios.mean():.4f}")
                print(f"    Finds optimal: {(hybrid_ratios == 1.0).sum()}/{len(hybrid_ratios)} instances")
                
                results['hybrid_quality'] = {
                    'mean_ratio': float(hybrid_ratios.mean()),
                    'optimal_count': int((hybrid_ratios == 1.0).sum())
                }
            
            # Compare Christofides vs Hybrid (paired test)
            if len(chris_ratios) == len(hybrid_ratios) and len(chris_ratios) >= 3:
                chris_list = chris_ratios.tolist()
                hybrid_list = hybrid_ratios.tolist()
                comparison = wilcoxon_signed_rank_test(chris_list, hybrid_list)
                effect = effect_size_cohens_d(chris_list, hybrid_list)
                
                results['chris_vs_hybrid'] = {
                    'test': comparison,
                    'effect_size': effect
                }
                
                print(f"\n  Christofides vs Hybrid Quality:")
                print(f"    p-value: {comparison['p_value']:.6f}")
                print(f"    Significant: {'YES' if comparison['significant'] else 'NO'}")
                print(f"    Cohen's d: {effect:.4f}")
        
        # Analyze large instances
        large_df = df[df['n'] > 12].copy()
        if len(large_df) > 0:
            print(f"\n  Large instances (n>12): {len(large_df)}")
            
            # Compare using LP bound
            chris_vs_lp = (large_df['christofides_cost'] / large_df['lp_objective']).dropna()
            hybrid_vs_lp = (large_df['hybrid_cost'] / large_df['lp_objective']).dropna()
            
            if len(chris_vs_lp) >= 3:
                print(f"\n  Christofides / LP: {chris_vs_lp.mean():.4f}")
            if len(hybrid_vs_lp) >= 3:
                print(f"  Hybrid / LP: {hybrid_vs_lp.mean():.4f}")
    
    return results


def analyze_sc_results():
    """Analyze Set Cover experimental results."""
    print("\n" + "="*80)
    print("SET COVER STATISTICAL ANALYSIS")
    print("="*80)
    
    results = {}
    
    # Load small-n CSV
    small_csv = "../../SC/results/sc_small_n_raw.csv"
    if os.path.exists(small_csv):
        df = pd.read_csv(small_csv)
        print(f"\n✓ Loaded {len(df)} small-n instances")
        
        # Compare algorithms
        if 'greedy_cost' in df.columns and 'Hybrid_Full_cost' in df.columns and 'opt_cost' in df.columns:
            greedy_ratios = (df['greedy_cost'] / df['opt_cost']).dropna()
            hybrid_ratios = (df['Hybrid_Full_cost'] / df['opt_cost']).dropna()
            
            print(f"\n  Greedy avg ratio: {greedy_ratios.mean():.4f}")
            print(f"  Hybrid avg ratio: {hybrid_ratios.mean():.4f}")
            
            if len(greedy_ratios) == len(hybrid_ratios) and len(greedy_ratios) >= 3:
                comparison = wilcoxon_signed_rank_test(
                    greedy_ratios.tolist(),
                    hybrid_ratios.tolist()
                )
                effect = effect_size_cohens_d(
                    greedy_ratios.tolist(),
                    hybrid_ratios.tolist()
                )
                
                results['greedy_vs_hybrid'] = {
                    'test': comparison,
                    'effect_size': effect,
                    'improvement': float((greedy_ratios.mean() - hybrid_ratios.mean()) / greedy_ratios.mean() * 100)
                }
                
                print(f"\n  Greedy vs Hybrid:")
                print(f"    p-value: {comparison['p_value']:.6f}")
                print(f"    Significant: {'YES' if comparison['significant'] else 'NO'}")
                print(f"    Hybrid improvement: {results['greedy_vs_hybrid']['improvement']:.2f}%")
    
    # Load large-n CSV
    large_csv = "../../SC/results/sc_large_n_raw.csv"
    if os.path.exists(large_csv):
        df = pd.read_csv(large_csv)
        print(f"\n✓ Loaded {len(df)} large-n instances")
        
        # Analyze efficiency metrics
        if 'mode' in df.columns and 'ratio' in df.columns:
            print(f"\n  Large-n performance (scalability test):")
            
            # Group by mode
            modes = df['mode'].unique()
            for mode in sorted(modes):
                subset = df[df['mode'] == mode]
                mean_ratio = subset['ratio'].mean()
                mean_time = subset['time'].mean()
                print(f"    {mode}: ratio={mean_ratio:.4f}, time={mean_time:.6f}s")
    
    return results


def make_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def main():
    """Run all statistical tests."""
    print("="*80)
    print("COMPREHENSIVE STATISTICAL SIGNIFICANCE ANALYSIS")
    print("Running tests on all experimental data")
    print("="*80)
    
    all_results = {}
    
    # Analyze each problem
    all_results['vertex_cover'] = analyze_vc_results()
    all_results['tsp'] = analyze_tsp_results()
    all_results['set_cover'] = analyze_sc_results()
    
    # Save comprehensive results
    output_file = "../collected_results/statistical_tests_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Make all results JSON-serializable
    all_results = make_json_serializable(all_results)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✓ Statistical analysis complete!")
    print(f"✓ Results saved to: {output_file}")
    
    # Print key findings
    print("\nKEY FINDINGS:")
    
    # VC findings
    if 'small_n_quality' in all_results.get('vertex_cover', {}):
        vc_qual = all_results['vertex_cover']['small_n_quality']
        print(f"\n  Vertex Cover (Small-n):")
        print(f"    - Hybrid improves over Primal-Dual by {vc_qual['improvement']:.2f}%")
        print(f"    - Difference is {'SIGNIFICANT' if vc_qual['test']['significant'] else 'NOT significant'} (p={vc_qual['test']['p_value']:.4f})")
        print(f"    - Effect size: {vc_qual['effect_size']:.3f}")
    
    # TSP findings
    if 'christofides_quality' in all_results.get('tsp', {}):
        tsp_chris = all_results['tsp']['christofides_quality']
        print(f"\n  TSP (Small-n with Exact):")
        print(f"    - Christofides mean ratio: {tsp_chris['mean_ratio']:.4f}")
        print(f"    - Finds optimal: {tsp_chris['optimal_count']}/{tsp_chris['total_count']} instances ({tsp_chris['optimal_count']/tsp_chris['total_count']*100:.1f}%)")
    
    # SC findings
    if 'greedy_vs_hybrid' in all_results.get('set_cover', {}):
        sc_comp = all_results['set_cover']['greedy_vs_hybrid']
        print(f"\n  Set Cover (Small-n):")
        print(f"    - Hybrid improves over Greedy by {sc_comp['improvement']:.2f}%")
        print(f"    - Difference is {'SIGNIFICANT' if sc_comp['test']['significant'] else 'NOT significant'} (p={sc_comp['test']['p_value']:.4f})")
    
    print("\n" + "="*80)
    print("For detailed results, see:", output_file)
    print("="*80)


if __name__ == "__main__":
    main()
