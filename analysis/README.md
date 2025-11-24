# Analysis Framework for Approximation Algorithms

**Owner:** Durga (Analysis Lead)  
**Project:** Hybrid LP+Local Meta-Approximation Algorithms  
**Team:** Begumpet BSTs (Group 9)

---

## Overview

This directory contains the complete empirical evaluation framework for analyzing approximation algorithms for three NP-hard optimization problems:

- **TSP** (Traveling Salesman Problem)
- **VC** (Vertex Cover)
- **SC** (Set Cover)

## Directory Structure

```
analysis/
├── README.md                    # This file
├── run_all_experiments.py       # Master experiment runner
│
├── instance_generators/         # Benchmark instance generators
│   ├── tsp_gen.py              # TSP instance generator
│   ├── vc_gen.py               # Vertex Cover instance generator
│   └── sc_gen.py               # Set Cover instance generator
│
├── plots/                       # Visualization scripts
│   ├── plot_runtime.py         # Runtime vs. n plots
│   ├── plot_ratio.py           # Approximation ratio plots
│   └── pareto_front.py         # Quality vs. runtime trade-offs
│
├── stats/                       # Statistical analysis
│   └── stats_tests.py          # Hypothesis tests (Wilcoxon, etc.)
│
├── protocol/                    # Methodology documentation
│   └── empirical_protocol.md   # Detailed experimental protocol
│
├── datasets/                    # Generated benchmark instances
│   ├── tsp/                    # TSP instances (JSON)
│   ├── vc/                     # VC instances (JSON)
│   └── sc/                     # SC instances (JSON)
│
└── collected_results/           # Results and plots
    ├── small_n/                # Small instance results
    ├── large_n/                # Large instance results
    └── *.png                   # Generated plots
```

---

## Quick Start

### 1. Generate All Benchmark Instances

```bash
cd analysis
python run_all_experiments.py
```

This will:
- Generate TSP, VC, and SC benchmark instances
- Create visualizations from existing experimental data
- Run statistical significance tests
- Generate a summary report

### 2. Generate Individual Problem Instances

```bash
# TSP instances
python instance_generators/tsp_gen.py

# Vertex Cover instances
python instance_generators/vc_gen.py

# Set Cover instances
python instance_generators/sc_gen.py
```

### 3. Generate Plots

```bash
# Runtime plots
python plots/plot_runtime.py

# Approximation ratio plots
python plots/plot_ratio.py

# Pareto front plots
python plots/pareto_front.py
```

### 4. Run Statistical Tests

```bash
python stats/stats_tests.py
```

---

## Instance Generators

### TSP Generator (`tsp_gen.py`)

Generates Euclidean TSP instances with random points in 2D plane.

**Features:**
- Random Euclidean instances
- Clustered instances (realistic scenarios)
- Grid-based instances
- Configurable size, seed, and bounds

**Usage:**
```python
from instance_generators.tsp_gen import generate_euclidean_tsp

points, dist_matrix = generate_euclidean_tsp(n=20, seed=42)
```

**Benchmark Suite:**
- Small n: {5, 8, 10, 12, 15, 18, 20} × 5 instances = 35 instances
- Large n: {30, 40, 50, 75, 100, 150, 200} × 3 instances = 21 instances

### Vertex Cover Generator (`vc_gen.py`)

Generates weighted graph instances for Vertex Cover.

**Features:**
- Erdős-Rényi random graphs
- Barabási-Albert scale-free graphs
- Random geometric graphs
- Random vertex weights

**Usage:**
```python
from instance_generators.vc_gen import generate_random_graph_vc

graph, weights = generate_random_graph_vc(n=50, p=0.3, seed=42)
```

**Benchmark Suite:**
- Small n: 3 configs × 5 instances = 15 instances
- Large n: 5 sizes × 3 instances = 15 instances

### Set Cover Generator (`sc_gen.py`)

Generates weighted Set Cover instances.

**Features:**
- Random set systems with configurable density
- Vertex Cover reduction to Set Cover
- Facility location scenarios
- Random subset costs

**Usage:**
```python
from instance_generators.sc_gen import generate_set_cover_instance

universe, subsets, costs = generate_set_cover_instance(
    num_elements=100, 
    num_sets=40, 
    density=0.2, 
    seed=42
)
```

**Benchmark Suite:**
- Small: 3 configs × 5 instances = 15 instances
- Large: 4 configs × 3 instances = 12 instances

---

## Plotting Scripts

### Runtime Plots (`plot_runtime.py`)

Generates runtime vs. problem size plots with log-log scaling.

**Features:**
- Individual algorithm runtime curves
- Multi-algorithm comparison
- Power-law fitting (O(nᵃ) estimation)
- Separate plots for small/large instances

**Example:**
```python
from plots.plot_runtime import plot_runtime_scaling

results = [(10, 0.05), (20, 0.15), (30, 0.35)]  # (n, time) pairs
plot_runtime_scaling(results, "Algorithm Name", output_file="runtime.png")
```

### Approximation Ratio Plots (`plot_ratio.py`)

Analyzes approximation quality (ALG/OPT ratios).

**Features:**
- Box plots showing ratio distributions
- Individual measurement scatter plots
- Comparison with theoretical bounds
- Ratio vs. n trend analysis

**Example:**
```python
from plots.plot_ratio import plot_approximation_ratios

ratios = {
    'Greedy': [1.2, 1.3, 1.25, 1.4],
    'Hybrid': [1.05, 1.1, 1.08, 1.12]
}
plot_approximation_ratios(ratios, "Problem Name", theoretical_bounds={'Greedy': 2.0})
```

### Pareto Front Plots (`pareto_front.py`)

Visualizes quality vs. runtime trade-offs.

**Features:**
- Identifies Pareto-optimal algorithms
- Quality-speed scatter plots
- Multi-instance overlays
- Optimal solution reference lines

**Example:**
```python
from plots.pareto_front import plot_pareto_front

algorithms = {
    'Exact': (1.5, 100.0),    # (runtime, cost)
    'Greedy': (0.01, 150.0),
    'Hybrid': (0.5, 105.0)
}
plot_pareto_front(algorithms, "Problem", opt_value=100.0)
```

---

## Statistical Tests

### Available Tests (`stats_tests.py`)

1. **Wilcoxon Signed-Rank Test** (paired samples)
   - Compare two algorithms on same instances
   - Non-parametric (no normality assumption)

2. **Mann-Whitney U Test** (independent samples)
   - Compare algorithms on different instance sets

3. **Paired t-test** (assumes normality)
   - Parametric version of Wilcoxon

4. **Kruskal-Wallis Test** (3+ independent groups)
   - Non-parametric multi-algorithm comparison

5. **Friedman Test** (3+ related groups)
   - Paired multi-algorithm comparison

6. **Cohen's d Effect Size**
   - Quantify magnitude of differences

**Example:**
```python
from stats.stats_tests import wilcoxon_signed_rank_test, effect_size_cohens_d

alg1_ratios = [1.2, 1.3, 1.25, 1.4]
alg2_ratios = [1.05, 1.1, 1.08, 1.12]

result = wilcoxon_signed_rank_test(alg1_ratios, alg2_ratios)
print(f"p-value: {result['p_value']}")
print(f"Significant: {result['significant']}")

d = effect_size_cohens_d(alg1_ratios, alg2_ratios)
print(f"Effect size: {d}")
```

---

## Experimental Protocol

See [`protocol/empirical_protocol.md`](protocol/empirical_protocol.md) for:

- Detailed research questions
- Instance generation specifications
- Performance metrics definitions
- Experimental procedures
- Statistical analysis plan
- Visualization requirements
- Result interpretation guidelines
- Reproducibility standards

---

## Integration with Team Member Code

### Using TSP Algorithms (Saharsh's work)

```python
# From TSP/christofides/christofides.py
from christofides import ChristofidesAlgorithm

points, dist = generate_euclidean_tsp(20, seed=42)
christofides = ChristofidesAlgorithm(points)
result = christofides.run()
print(f"Tour cost: {result['cost']}")
```

### Using VC Algorithms (Lohith's work)

```python
# From VC/algorithms/hybrid.py
from algorithms.hybrid import run_hybrid_vc

graph, weights = generate_random_graph_vc(50, p=0.3, seed=42)
result = run_hybrid_vc(graph, weights, num_rounds=64, k_exchange=2)
print(f"Hybrid cost: {result.hybrid_cost}")
print(f"Approximation ratio: {result.hybrid_cost / result.lp_value}")
```

### Using Exact Solvers (Pranshul's work)

```python
# From pranshul/src/algorithms/exact_solvers.py
from src.algorithms.exact_solvers import VCExact

graph, weights = generate_random_graph_vc(20, p=0.3, seed=42)
exact = VCExact()
opt_cost = exact.solve(graph)
print(f"Optimal cost: {opt_cost}")
```

---

## Results Format

### Small n Experiments (JSON)

```json
{
  "n": 20,
  "p": 0.3,
  "seed": 100,
  "num_edges": 58,
  "exact_cost": 45.67,
  "lp_value": 42.31,
  "primal_dual_cost": 78.92,
  "hybrid_cost": 48.23,
  "primal_dual_ratio": 1.729,
  "hybrid_ratio": 1.056,
  "timings": {
    "exact": 0.234,
    "primal_dual": 0.001,
    "hybrid_total": 0.089,
    "lp_solve": 0.023,
    "rounding": 0.004,
    "local_search": 0.062
  }
}
```

### Large n Experiments (JSON)

```json
{
  "n": 100,
  "p": 0.4,
  "seed": 1000,
  "num_edges": 1982,
  "lp_value": 234.56,
  "primal_dual_cost": 456.78,
  "hybrid_cost": 289.34,
  "hybrid_vs_lp": 1.234,
  "primal_dual_vs_lp": 1.948,
  "timings": {
    "primal_dual": 0.008,
    "hybrid_total": 0.876,
    "lp_solve": 0.045,
    "rounding": 0.006,
    "local_search": 0.825
  }
}
```

---

## Dependencies

```
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.4
networkx >= 2.6
pulp >= 2.5
```

Install with:
```bash
pip install numpy scipy matplotlib networkx pulp
```

---

## Troubleshooting

### Import Errors

If you get import errors when running scripts:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'VC'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pranshul', 'src'))
```

### Missing Data

If plots show "no data available":
1. Ensure experimental results exist in `../VC/results/` or similar
2. Run team-specific experiment scripts first
3. Check JSON file paths in plotting scripts

### Solver Timeouts

For exact solvers on large instances:
- Set time limits in experimental code
- Focus exact solvers on small n only (n ≤ 24 for VC)

---

## Contributing

When adding new analysis scripts:

1. Follow naming conventions: `snake_case.py`
2. Include docstrings for all functions
3. Add usage examples in `__main__` block
4. Update this README with new features
5. Ensure reproducibility with fixed seeds

---

## Contact

**Analysis Lead:** Durga  
**Project:** Advanced Algorithm Design, IIIT Hyderabad  
**Team:** Begumpet BSTs (Group 9)

For questions about:
- Instance generators → Durga
- TSP experiments → Saharsh
- VC experiments → Lohith
- SC experiments → Aayush
- Exact solvers → Pranshul

---

## License

This is academic work for Advanced Algorithm Design course.  
All code is for educational purposes.
