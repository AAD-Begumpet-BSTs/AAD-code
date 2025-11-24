# Empirical Evaluation Protocol

## Project: Hybrid LP+Local Meta-Approximation Algorithms
**Team:** Begumpet BSTs (Group 9)  
**Course:** Advanced Algorithm Design  
**Date:** November 2025

---

## 1. Overview

This document describes the empirical methodology for evaluating approximation algorithms for three NP-hard optimization problems:
- **Traveling Salesman Problem (TSP)** - Metric/Euclidean variant
- **Vertex Cover (VC)** - Weighted variant
- **Set Cover (SC)** - Weighted variant

### 1.1 Research Questions

1. **Quality vs. Optimality**: How do approximation algorithms compare to exact optimal solutions on small instances?
2. **Scalability**: How does runtime scale with problem size for approximation vs. exact algorithms?
3. **Hybrid Performance**: Does the LP-based hybrid approach (LP + Randomized Rounding + Local Search) outperform classical approximations?
4. **Quality-Runtime Trade-offs**: What are the Pareto-optimal solutions in the quality-speed space?

---

## 2. Algorithms Under Evaluation

### 2.1 TSP Algorithms

| Algorithm | Type | Theoretical Guarantee | Implementation Owner |
|-----------|------|----------------------|---------------------|
| Held-Karp | Exact | Optimal | Pranshul |
| Christofides | Approximation | 1.5-approx | Saharsh |
| 2-Opt Local Search | Local Search | 2-local optimum | Saharsh |
| LP + Rounding + 2-Opt | Hybrid | Empirical | Saharsh |

### 2.2 Vertex Cover Algorithms

| Algorithm | Type | Theoretical Guarantee | Implementation Owner |
|-----------|------|----------------------|---------------------|
| ILP (Branch & Bound) | Exact | Optimal | Pranshul |
| Primal-Dual | Approximation | 2-approx | Lohith |
| LP + Rounding + k-exchange | Hybrid | Empirical | Lohith |

### 2.3 Set Cover Algorithms

| Algorithm | Type | Theoretical Guarantee | Implementation Owner |
|-----------|------|----------------------|---------------------|
| ILP | Exact | Optimal | Pranshul |
| Greedy | Approximation | ln(n)-approx | Aayush |
| LP + Rounding + Local Search | Hybrid | Empirical | Aayush |

---

## 3. Instance Generation

### 3.1 TSP Instances

**Small Instances** (for exact solver comparison):
- **Sizes**: n ∈ {5, 8, 10, 12, 15, 18, 20}
- **Type**: Random Euclidean (points in [0, 100]² plane)
- **Replicates**: 5 instances per size (seeds: 100-104)
- **Total**: 35 instances

**Large Instances** (for scalability):
- **Sizes**: n ∈ {30, 40, 50, 75, 100, 150, 200}
- **Type**: Random Euclidean
- **Replicates**: 3 instances per size (seeds: 1000-1002)
- **Total**: 21 instances

**Rationale**: Euclidean metric ensures triangle inequality, making Christofides applicable. Small n limited by Held-Karp's O(n² 2ⁿ) complexity.

### 3.2 Vertex Cover Instances

**Small Instances**:
- **Configurations**: (n, p) ∈ {(12, 0.3), (18, 0.35), (24, 0.4)}
- **Graph Model**: Erdős-Rényi G(n, p)
- **Weights**: Uniform random ∈ [1.0, 10.0]
- **Replicates**: 5 instances per config (seeds: 100-104)
- **Total**: 15 instances

**Large Instances**:
- **Sizes**: n ∈ {50, 75, 100, 150, 200}
- **Density**: p ∈ {0.3, 0.35, 0.4}
- **Replicates**: 3 instances per config (seeds: 1000-1002)
- **Total**: 15 instances

**Rationale**: Random graphs with moderate density (p ~ 0.3) typical for VC benchmarks. Vertex weights test weighted case.

### 3.3 Set Cover Instances

**Small Instances**:
- **Configurations**: (elements, sets, density) ∈ {(20, 10, 0.3), (30, 15, 0.25), (40, 20, 0.2)}
- **Costs**: Uniform random ∈ [1.0, 10.0]
- **Replicates**: 5 instances per config (seeds: 100-104)
- **Total**: 15 instances

**Large Instances**:
- **Sizes**: (elements, sets) ∈ {(100, 40), (200, 60), (300, 80), (500, 100)}
- **Density**: ~0.15-0.2
- **Replicates**: 3 instances per config (seeds: 1000-1002)
- **Total**: 12 instances

**Rationale**: Moderate set overlap (density 0.2-0.3) ensures non-trivial covering instances.

---

## 4. Performance Metrics

### 4.1 Primary Metrics

1. **Solution Quality**
   - Absolute cost: Total cost/distance of solution
   - Approximation ratio ρ = ALG / OPT (for small instances only)

2. **Runtime**
   - Wall-clock time (seconds) using `time.perf_counter()`
   - Measured for each algorithm phase (LP solve, rounding, local search)

3. **Scalability**
   - Runtime growth rate (fit to O(nᵃ) or O(2ⁿ))

### 4.2 Secondary Metrics

1. **LP Gap**: (OPT - LP_bound) / LP_bound
2. **Iterations**: Number of local search improvements
3. **Pareto Efficiency**: Quality vs. runtime trade-off position

---

## 5. Experimental Procedure

### 5.1 Small n Experiments (Comparison with Exact)

**For each instance:**

1. Run **Exact Solver** → Record OPT value and runtime
2. Run **Classical Approximation** → Record cost and runtime
3. Run **Hybrid Algorithm** → Record cost, runtime, LP bound
4. Compute **approximation ratios**: ALG_cost / OPT

**Output Format** (JSON):
```json
{
  "n": 20,
  "instance_id": "tsp_n20_s100",
  "exact_cost": 245.67,
  "exact_time": 15.234,
  "approx_cost": 312.45,
  "approx_time": 0.045,
  "hybrid_cost": 268.91,
  "hybrid_time": 1.234,
  "lp_bound": 240.12,
  "approx_ratio": 1.272,
  "hybrid_ratio": 1.095
}
```

### 5.2 Large n Experiments (Scalability Only)

**For each instance:**

1. Run **Classical Approximation** → Record cost and runtime
2. Run **Hybrid Algorithm** → Record cost and runtime
3. (Skip exact solver - too slow)

**Output Format**:
```json
{
  "n": 100,
  "instance_id": "vc_n100_p0.4_s1000",
  "approx_cost": 1234.5,
  "approx_time": 0.234,
  "hybrid_cost": 1156.3,
  "hybrid_time": 5.678,
  "lp_bound": 1089.2
}
```

### 5.3 Execution Environment

- **Hardware**: Document CPU, RAM for reproducibility
- **Software**: Python 3.10+, PuLP (CBC solver), NetworkX, NumPy, SciPy
- **Repetitions**: Each instance run 1 time (use multiple seeds for statistical power)
- **Time Limit**: 300 seconds per exact solver run (timeout = no solution)

---

## 6. Statistical Analysis

### 6.1 Hypothesis Tests

**Research Hypothesis 1**: Hybrid algorithms achieve better approximation ratios than classical approximations.

- **Test**: Wilcoxon signed-rank test (paired samples)
- **Null Hypothesis**: median(ρ_hybrid - ρ_classical) = 0
- **Significance Level**: α = 0.05

**Research Hypothesis 2**: Runtime growth differs between exact and approximation algorithms.

- **Test**: Log-log regression fit
- **Metrics**: R² goodness of fit, estimated exponent

### 6.2 Effect Size

- **Cohen's d** for mean differences in approximation ratios
- Interpretation: |d| ≥ 0.8 (large), 0.5 ≤ |d| < 0.8 (medium), 0.2 ≤ |d| < 0.5 (small)

---

## 7. Visualization

### 7.1 Required Plots

1. **Approximation Ratio Box Plots** (small n)
   - X-axis: Algorithm
   - Y-axis: ρ = ALG/OPT
   - Show theoretical bounds as horizontal lines

2. **Runtime vs. n (log-log scale)**
   - Separate plots for small n and large n
   - Fit power-law curves

3. **Pareto Fronts** (Quality vs. Runtime)
   - X-axis: Runtime (log scale)
   - Y-axis: Solution quality (or approximation ratio)
   - Identify Pareto-optimal algorithms

4. **LP Gap Analysis**
   - Compare (OPT - LP) gap across instances

### 7.2 Plot Specifications

- **Format**: PNG, 300 DPI minimum
- **Fonts**: 12pt labels, 14pt titles
- **Colors**: Colorblind-friendly palette (tab10 or Set2)
- **Grid**: Light gray, alpha=0.3
- **Legends**: Clear algorithm names

---

## 8. Data Collection Workflow

1. **Generate Instances** (using generators in `analysis/instance_generators/`)
   ```bash
   python tsp_gen.py
   python vc_gen.py
   python sc_gen.py
   ```

2. **Run Small n Experiments**
   ```bash
   python run_experiments_small.py --problem tsp
   python run_experiments_small.py --problem vc
   python run_experiments_small.py --problem sc
   ```

3. **Run Large n Experiments**
   ```bash
   python run_experiments_large.py --problem tsp
   python run_experiments_large.py --problem vc
   python run_experiments_large.py --problem sc
   ```

4. **Generate Plots**
   ```bash
   python plots/plot_runtime.py
   python plots/plot_ratio.py
   python plots/pareto_front.py
   ```

5. **Run Statistical Tests**
   ```bash
   python stats/stats_tests.py
   ```

---

## 9. Result Interpretation Guidelines

### 9.1 Approximation Quality

- **Excellent**: ρ < 1.1 (within 10% of OPT)
- **Good**: 1.1 ≤ ρ < 1.3
- **Acceptable**: 1.3 ≤ ρ < theoretical bound
- **Poor**: ρ > theoretical bound (investigate why)

### 9.2 Runtime Scaling

- **Exact**: Expected O(2ⁿ) or worse
- **Approximation**: Expected O(n²) to O(n³)
- **Hybrid**: Expected O(n³) (dominated by LP solve)

### 9.3 Success Criteria

A hybrid algorithm is considered **successful** if:
1. ρ_hybrid < ρ_classical (statistically significant, α=0.05)
2. Runtime_hybrid ≤ 10 × Runtime_classical
3. No violations of theoretical guarantees

---

## 10. Reporting Standards

### 10.1 Tables

Each results table must include:
- Instance parameters (n, density, seed)
- OPT value (if available)
- Algorithm costs
- Approximation ratios
- Runtimes
- Statistical summaries (mean, std, median)

### 10.2 Figures

Each figure must have:
- Descriptive caption explaining what is shown
- Axis labels with units
- Legend identifying all data series
- Reference to relevant section in report

### 10.3 Reproducibility

All experiments must be reproducible via:
- Documented random seeds
- Fixed instance generator parameters
- Version-controlled code
- Recorded software dependencies

---

## 11. Validation Checks

Before finalizing results, verify:

1. ✓ All approximation ratios ρ ≥ 1.0
2. ✓ LP bounds ≤ OPT (when known)
3. ✓ Exact solver solutions are valid (satisfy all constraints)
4. ✓ No timeout instances excluded without documentation
5. ✓ Statistical tests have sufficient sample size (n ≥ 10)

---

## 12. Timeline

| Week | Task | Owner |
|------|------|-------|
| 1-2 | Implement algorithms | Saharsh, Lohith, Aayush |
| 2-3 | Generate benchmark instances | Durga |
| 3-4 | Run small n experiments | All |
| 4-5 | Run large n experiments | All |
| 5 | Statistical analysis | Durga |
| 5-6 | Generate all plots | Durga |
| 6 | Write final report | All |

---

## 13. References

Key papers influencing this protocol:

1. **Approximation Algorithms** (Vazirani, 2001)
2. **LP Relaxation Methods** (Williamson & Shmoys, 2011)
3. **Empirical Evaluation Guidelines** (Johnson, 2002)
4. **Benchmark Instance Design** (Reinelt, 1991 - TSPLIB)

---

## Appendix A: File Structure

```
analysis/
├── datasets/
│   ├── tsp/          # Generated TSP instances
│   ├── vc/           # Generated VC instances
│   └── sc/           # Generated SC instances
├── collected_results/
│   ├── small_n/      # Small instance results
│   └── large_n/      # Large instance results
├── instance_generators/
│   ├── tsp_gen.py
│   ├── vc_gen.py
│   └── sc_gen.py
├── plots/
│   ├── plot_runtime.py
│   ├── plot_ratio.py
│   └── pareto_front.py
├── stats/
│   └── stats_tests.py
└── protocol/
    └── empirical_protocol.md (this file)
```

---

**Document Version:** 1.0  
**Last Updated:** November 25, 2025  
**Authors:** Durga (Analysis Lead), with input from all team members
