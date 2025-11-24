"""
Statistical Significance Testing
Performs hypothesis tests to compare algorithm performance.
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple
import json


def wilcoxon_signed_rank_test(data1: List[float], data2: List[float], 
                               alpha: float = 0.05) -> Dict:
    """
    Wilcoxon signed-rank test for paired samples.
    Tests if two related samples have different distributions.
    
    Use case: Compare two algorithms on the same set of instances.
    
    Args:
        data1: First algorithm's results
        data2: Second algorithm's results (paired with data1)
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if len(data1) != len(data2):
        raise ValueError("Data arrays must have same length for paired test")
    
    if len(data1) < 3:
        return {
            'test': 'Wilcoxon Signed-Rank',
            'statistic': None,
            'p_value': None,
            'significant': None,
            'note': 'Too few samples (need at least 3)'
        }
    
    # Perform test
    statistic, p_value = stats.wilcoxon(data1, data2, alternative='two-sided')
    
    return {
        'test': 'Wilcoxon Signed-Rank',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'alpha': alpha,
        'n_samples': len(data1),
        'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis (p={p_value:.4f})"
    }


def mann_whitney_u_test(data1: List[float], data2: List[float],
                        alpha: float = 0.05) -> Dict:
    """
    Mann-Whitney U test for independent samples.
    Tests if two independent samples have different distributions.
    
    Use case: Compare two algorithms on different instance sets.
    
    Args:
        data1: First algorithm's results
        data2: Second algorithm's results (independent)
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if len(data1) < 3 or len(data2) < 3:
        return {
            'test': 'Mann-Whitney U',
            'statistic': None,
            'p_value': None,
            'significant': None,
            'note': 'Too few samples (need at least 3 per group)'
        }
    
    # Perform test
    statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    
    return {
        'test': 'Mann-Whitney U',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'alpha': alpha,
        'n1': len(data1),
        'n2': len(data2),
        'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis (p={p_value:.4f})"
    }


def paired_t_test(data1: List[float], data2: List[float],
                  alpha: float = 0.05) -> Dict:
    """
    Paired t-test for normally distributed paired samples.
    
    Note: Requires normality assumption. Use Wilcoxon for non-normal data.
    
    Args:
        data1: First algorithm's results
        data2: Second algorithm's results (paired)
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    if len(data1) != len(data2):
        raise ValueError("Data arrays must have same length for paired test")
    
    if len(data1) < 3:
        return {
            'test': 'Paired t-test',
            'statistic': None,
            'p_value': None,
            'significant': None,
            'note': 'Too few samples'
        }
    
    # Perform test
    statistic, p_value = stats.ttest_rel(data1, data2)
    
    # Check normality of differences
    differences = np.array(data1) - np.array(data2)
    if len(differences) >= 3:
        _, normality_p = stats.shapiro(differences)
        normality_ok = normality_p > 0.05
    else:
        normality_ok = None
    
    return {
        'test': 'Paired t-test',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'alpha': alpha,
        'n_samples': len(data1),
        'normality_assumption_met': normality_ok,
        'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis (p={p_value:.4f})"
    }


def kruskal_wallis_test(data_groups: Dict[str, List[float]],
                        alpha: float = 0.05) -> Dict:
    """
    Kruskal-Wallis H-test for comparing multiple independent groups.
    Non-parametric version of one-way ANOVA.
    
    Use case: Compare 3+ algorithms on same instances.
    
    Args:
        data_groups: Dictionary mapping algorithm name to results
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    groups = list(data_groups.values())
    
    if len(groups) < 2:
        return {
            'test': 'Kruskal-Wallis',
            'note': 'Need at least 2 groups'
        }
    
    # Perform test
    statistic, p_value = stats.kruskal(*groups)
    
    return {
        'test': 'Kruskal-Wallis H',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'alpha': alpha,
        'n_groups': len(groups),
        'group_sizes': [len(g) for g in groups],
        'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis (p={p_value:.4f})"
    }


def friedman_test(data_groups: Dict[str, List[float]],
                  alpha: float = 0.05) -> Dict:
    """
    Friedman test for comparing multiple related groups.
    Non-parametric version of repeated measures ANOVA.
    
    Use case: Compare 3+ algorithms on the same set of instances (paired).
    
    Args:
        data_groups: Dictionary mapping algorithm name to paired results
        alpha: Significance level
        
    Returns:
        Dictionary with test results
    """
    groups = list(data_groups.values())
    
    if len(groups) < 2:
        return {
            'test': 'Friedman',
            'note': 'Need at least 2 groups'
        }
    
    # Check all groups have same length (paired data)
    lengths = [len(g) for g in groups]
    if len(set(lengths)) > 1:
        return {
            'test': 'Friedman',
            'note': 'All groups must have same size (paired data required)'
        }
    
    # Perform test
    statistic, p_value = stats.friedmanchisquare(*groups)
    
    return {
        'test': 'Friedman',
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'alpha': alpha,
        'n_groups': len(groups),
        'n_samples': lengths[0],
        'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} null hypothesis (p={p_value:.4f})"
    }


def effect_size_cohens_d(data1: List[float], data2: List[float]) -> float:
    """
    Compute Cohen's d effect size for two samples.
    
    Interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large
    
    Args:
        data1: First sample
        data2: Second sample
        
    Returns:
        Cohen's d value
    """
    n1, n2 = len(data1), len(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(data1) - np.mean(data2)) / pooled_std
    
    return float(d)


def analyze_vc_statistical_significance(json_file: str, output_file: str = None):
    """
    Perform comprehensive statistical analysis of Vertex Cover experiments.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract paired data for each algorithm
    primal_dual_ratios = []
    hybrid_ratios = []
    primal_dual_times = []
    hybrid_times = []
    
    for entry in data:
        if 'primal_dual_ratio' in entry:
            primal_dual_ratios.append(entry['primal_dual_ratio'])
        if 'hybrid_ratio' in entry:
            hybrid_ratios.append(entry['hybrid_ratio'])
        if 'timings' in entry:
            if 'primal_dual' in entry['timings']:
                primal_dual_times.append(entry['timings']['primal_dual'])
            if 'hybrid_total' in entry['timings']:
                hybrid_times.append(entry['timings']['hybrid_total'])
    
    results = {}
    
    # Test 1: Compare approximation ratios (Primal-Dual vs Hybrid)
    if len(primal_dual_ratios) == len(hybrid_ratios) and len(primal_dual_ratios) > 0:
        results['ratio_comparison'] = wilcoxon_signed_rank_test(
            primal_dual_ratios, hybrid_ratios
        )
        results['ratio_effect_size'] = effect_size_cohens_d(
            primal_dual_ratios, hybrid_ratios
        )
    
    # Test 2: Compare runtimes (Primal-Dual vs Hybrid)
    if len(primal_dual_times) == len(hybrid_times) and len(primal_dual_times) > 0:
        results['runtime_comparison'] = wilcoxon_signed_rank_test(
            primal_dual_times, hybrid_times
        )
        results['runtime_effect_size'] = effect_size_cohens_d(
            primal_dual_times, hybrid_times
        )
    
    # Print results
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS: VERTEX COVER")
    print("="*70)
    
    if 'ratio_comparison' in results:
        print("\n1. Approximation Ratio Comparison (Primal-Dual vs Hybrid):")
        print(f"   Test: {results['ratio_comparison']['test']}")
        print(f"   p-value: {results['ratio_comparison']['p_value']:.6f}")
        print(f"   Significant (α=0.05): {results['ratio_comparison']['significant']}")
        print(f"   Cohen's d: {results['ratio_effect_size']:.4f}")
        print(f"   {results['ratio_comparison']['interpretation']}")
    
    if 'runtime_comparison' in results:
        print("\n2. Runtime Comparison (Primal-Dual vs Hybrid):")
        print(f"   Test: {results['runtime_comparison']['test']}")
        print(f"   p-value: {results['runtime_comparison']['p_value']:.6f}")
        print(f"   Significant (α=0.05): {results['runtime_comparison']['significant']}")
        print(f"   Cohen's d: {results['runtime_effect_size']:.4f}")
        print(f"   {results['runtime_comparison']['interpretation']}")
    
    print("\n" + "="*70)
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


def run_all_statistical_tests():
    """
    Run all statistical tests on collected experimental data.
    """
    print("="*70)
    print("RUNNING STATISTICAL SIGNIFICANCE TESTS")
    print("="*70)
    
    # Vertex Cover
    vc_file = "../../VC/results/vc_small_experiments.json"
    if os.path.exists(vc_file):
        analyze_vc_statistical_significance(
            vc_file, 
            output_file="collected_results/vc_statistical_tests.json"
        )
    else:
        print("\nWarning: VC results not found")
    
    # TODO: Add TSP and SC when data is available
    
    print("\n" + "="*70)
    print("✓ Statistical testing complete")
    print("="*70)


if __name__ == "__main__":
    import os
    
    # Demo with synthetic data
    print("Statistical Testing Demo")
    print("="*70)
    
    np.random.seed(42)
    
    # Create two sets of results
    alg1_results = np.random.normal(1.3, 0.1, 20)  # Mean ratio 1.3
    alg2_results = np.random.normal(1.15, 0.08, 20)  # Mean ratio 1.15
    
    print("\nDemo: Comparing two algorithms on paired instances")
    print(f"Algorithm 1 mean: {np.mean(alg1_results):.4f}")
    print(f"Algorithm 2 mean: {np.mean(alg2_results):.4f}")
    
    # Wilcoxon test
    result = wilcoxon_signed_rank_test(alg1_results, alg2_results)
    print(f"\n{result['test']}:")
    print(f"  p-value: {result['p_value']:.6f}")
    print(f"  Significant: {result['significant']}")
    print(f"  {result['interpretation']}")
    
    # Effect size
    d = effect_size_cohens_d(alg1_results, alg2_results)
    print(f"\nCohen's d: {d:.4f}")
    if abs(d) >= 0.8:
        print("  Effect size: LARGE")
    elif abs(d) >= 0.5:
        print("  Effect size: MEDIUM")
    elif abs(d) >= 0.2:
        print("  Effect size: SMALL")
    else:
        print("  Effect size: NEGLIGIBLE")
    
    # Try to run tests on real data
    print("\n" + "="*70)
    run_all_statistical_tests()
