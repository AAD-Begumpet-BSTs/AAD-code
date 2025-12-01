# TSP Experiments - README

## Overview
This directory contains experimental code for comprehensive TSP algorithm evaluation.

**Author**: Saharsh (TSP)  
**Date**: December 2025

## Files
- `run_experiments.py` - Main experimental runner
- `generate_plots.py` - Plot generation script
- `tsp_experiments_results.csv` - Raw experimental data (generated)
- `plots/` - All generated visualizations (generated)

## Running the Experiments

### Step 1: Install dependencies
```bash
# Activate your virtual environment
source ../tsp_env/bin/activate

# Install seaborn (for plotting)
pip install seaborn
```

### Step 2: Run experiments
```bash
python run_experiments.py
```

This will:
- Test 5 small problem sizes (5, 8, 10, 12, 15) with LP lower bounds
- Test 5 large problem sizes (20, 30, 50, 75, 100) for runtime scaling
- Run 10 instances per size
- Test on 2 distributions (uniform, clustered)
- Total: ~100 experiments

**Expected runtime**: 15-30 minutes

### Step 3: Generate plots
```bash
python generate_plots.py
```

This will generate:
1. `1_approximation_ratios_small_n.png` - Approximation ratios for small instances
2. `2_runtime_scaling_large_n.png` - Runtime scaling (log-log)
3. `3_pareto_fronts.png` - Quality vs runtime trade-offs
4. `4_algorithm_comparison_detailed.png` - Comprehensive comparison
5. `5_2opt_improvement_analysis.png` - 2-Opt improvement analysis
6. `6_distribution_comparison.png` - Performance by instance type
7. `7_summary_table.png` - Summary statistics table

## Output Files
- `tsp_experiments_results.csv` - Raw data with all metrics
- `plots/*.png` - All generated visualizations

## For Your Report
Use the generated plots in your TSP section. The plots cover:
- ✓ Approximation ratio analysis (small n)
- ✓ Runtime scaling analysis (large n, log-log)
- ✓ Pareto-front trade-offs
- ✓ Statistical comparisons
- ✓ Algorithm performance across different instance types

## Troubleshooting

If you get import errors:
```bash
pip install networkx scipy pandas matplotlib numpy pulp seaborn
```

If experiments are too slow, reduce in `run_experiments.py`:
- `instances_per_size=5` (instead of 10)
- Remove large sizes: `large_sizes=[20, 30, 50]`
