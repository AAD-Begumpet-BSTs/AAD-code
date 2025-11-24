"""
Master Experiment Runner
Coordinates all experimental evaluations for TSP, VC, and SC.
"""

import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pranshul', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'VC'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TSP'))

import subprocess
import json
from datetime import datetime


def run_instance_generation():
    """Generate all benchmark instances."""
    print("="*70)
    print("STEP 1: GENERATING BENCHMARK INSTANCES")
    print("="*70)
    
    generators = [
        ('TSP', 'instance_generators/tsp_gen.py'),
        ('Vertex Cover', 'instance_generators/vc_gen.py'),
        ('Set Cover', 'instance_generators/sc_gen.py')
    ]
    
    for name, script in generators:
        print(f"\nGenerating {name} instances...")
        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=os.path.dirname(__file__),
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                print(f"✓ {name} instances generated")
                if result.stdout:
                    print(result.stdout)
            else:
                print(f"✗ Error generating {name} instances:")
                print(result.stderr)
        except Exception as e:
            print(f"✗ Failed to run {script}: {e}")


def run_visualization():
    """Generate all plots and visualizations."""
    print("\n" + "="*70)
    print("STEP 2: GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_scripts = [
        ('Runtime Plots', 'plots/plot_runtime.py'),
        ('Approximation Ratio Plots', 'plots/plot_ratio.py'),
        ('Pareto Fronts', 'plots/pareto_front.py')
    ]
    
    for name, script in plot_scripts:
        print(f"\nGenerating {name}...")
        try:
            result = subprocess.run(
                [sys.executable, script],
                cwd=os.path.dirname(__file__),
                capture_output=True,
                text=True,
                timeout=180
            )
            if result.returncode == 0:
                print(f"✓ {name} generated")
            else:
                print(f"Note: {name} may require experimental data")
        except Exception as e:
            print(f"Note: {script} - {e}")


def run_statistical_tests():
    """Run statistical significance tests."""
    print("\n" + "="*70)
    print("STEP 3: STATISTICAL ANALYSIS")
    print("="*70)
    
    print("\nRunning statistical tests...")
    try:
        result = subprocess.run(
            [sys.executable, 'stats/stats_tests.py'],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            print("✓ Statistical tests complete")
            if result.stdout:
                print(result.stdout)
        else:
            print("Note: Statistical tests may require experimental data")
    except Exception as e:
        print(f"Note: Statistical tests - {e}")


def generate_summary_report():
    """Generate a summary of all results."""
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'instances_generated': {},
        'results_available': {},
        'plots_generated': []
    }
    
    # Check for generated instances
    instance_dirs = {
        'TSP': 'datasets/tsp',
        'Vertex Cover': 'datasets/vc',
        'Set Cover': 'datasets/sc'
    }
    
    for name, dir_path in instance_dirs.items():
        full_path = os.path.join(os.path.dirname(__file__), dir_path)
        if os.path.exists(full_path):
            count = len([f for f in os.listdir(full_path) if f.endswith('.json')])
            summary['instances_generated'][name] = count
            print(f"{name}: {count} instances")
    
    # Check for results
    vc_results_small = os.path.join(os.path.dirname(__file__), '..', 'VC', 'results', 'vc_small_experiments.json')
    vc_results_large = os.path.join(os.path.dirname(__file__), '..', 'VC', 'results', 'vc_large_experiments.json')
    
    if os.path.exists(vc_results_small):
        with open(vc_results_small, 'r') as f:
            vc_small = json.load(f)
            summary['results_available']['VC_small'] = len(vc_small)
            print(f"\nVertex Cover (small n): {len(vc_small)} experiments")
    
    if os.path.exists(vc_results_large):
        with open(vc_results_large, 'r') as f:
            vc_large = json.load(f)
            summary['results_available']['VC_large'] = len(vc_large)
            print(f"Vertex Cover (large n): {len(vc_large)} experiments")
    
    # Check for plots
    results_dir = os.path.join(os.path.dirname(__file__), 'collected_results')
    if os.path.exists(results_dir):
        plots = [f for f in os.listdir(results_dir) if f.endswith('.png')]
        summary['plots_generated'] = plots
        print(f"\nGenerated plots: {len(plots)}")
        for plot in plots[:5]:  # Show first 5
            print(f"  - {plot}")
        if len(plots) > 5:
            print(f"  ... and {len(plots) - 5} more")
    
    # Save summary
    summary_file = os.path.join(os.path.dirname(__file__), 'collected_results', 'experiment_summary.json')
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_file}")


def main():
    """Run complete experimental pipeline."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                  EXPERIMENTAL EVALUATION PIPELINE                ║
║                                                                  ║
║  Problems: TSP, Vertex Cover, Set Cover                         ║
║  Algorithms: Exact, Approximation, Hybrid LP+Local              ║
║                                                                  ║
║  Pipeline:                                                       ║
║    1. Generate benchmark instances                              ║
║    2. Generate visualizations from existing results             ║
║    3. Run statistical tests                                     ║
║    4. Generate summary report                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Instance Generation
    run_instance_generation()
    
    # Step 2: Visualizations
    run_visualization()
    
    # Step 3: Statistical Tests
    run_statistical_tests()
    
    # Step 4: Summary
    generate_summary_report()
    
    print("\n" + "="*70)
    print("✓ EXPERIMENTAL PIPELINE COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. Review generated instances in analysis/datasets/")
    print("2. Run team-specific experiment scripts (TSP/, VC/, etc.)")
    print("3. Check plots in analysis/collected_results/")
    print("4. Review statistical tests output")
    print("\nFor detailed methodology, see: analysis/protocol/empirical_protocol.md")


if __name__ == "__main__":
    main()
