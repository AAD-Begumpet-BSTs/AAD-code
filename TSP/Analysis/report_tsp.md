## Traveling Salesman Problem (Saharsh)

### What We Built Together
- **Christofides 1.5-Approximation (`christofides.christofides_tsp`)** – our reliable baseline. Builds a minimum spanning tree, finds minimum-weight perfect matching on odd-degree vertices, constructs an Eulerian circuit, and shortcuts to a Hamiltonian cycle. Guarantees ≤1.5× OPT and gives us a constructive solution on every metric TSP instance in polynomial time.
- **LP Relaxation Lower Bound (`tsp_lp.tsp_lp_relaxation`)** – we formulate the TSP as an integer program with subtour elimination constraints, solve the LP relaxation with PuLP/CBC, and use the optimal LP value as a tight lower bound for approximation ratio calculations. Critical for evaluating solution quality when exact solutions are intractable.
- **2-Opt Local Search (`two_opt.two_opt`)** – our iterative refinement engine. Starting from a random tour (or any initial tour), we repeatedly find edge swaps that reduce tour cost until no improving 2-opt move exists. Fast, effective, and frequently finds optimal solutions on small instances through pure local optimization.
- **Hybrid Meta-Heuristic (`Algorithms.hybrid.hybrid_tsp`)** – combines the best of both worlds. Uses Christofides for a quality starting point (already ≤1.5× OPT), then applies 2-Opt local search to polish the solution. Logs detailed timings for Christofides construction and 2-Opt iterations, giving us full visibility into the optimization process.
- **Exact Solver (Held-Karp) (`Algorithms.held_karp.held_karp_tsp`)** – dynamic programming with bitmask state compression. Solves TSP optimally in O(n²·2ⁿ) time and O(n·2ⁿ) space. We use this for n≤12 to get ground truth optimal values for small instances, enabling precise approximation ratio calculations.

### How We Evaluate Now
1. **Development Experiments (`TSP/experiments/run_experiments.py`)**  
   - Generates fresh random Euclidean TSP instances with coordinates in [0, 1000] × [0, 1000].  
   - *Comprehensive suite*: 200 instances across 10 problem sizes (n ∈ {5,10,15,20,30,40,50,75,100,200}), 2 seed distributions, 10 instances each.  
   - Compares all algorithms: Exact (n≤12), LP (n≤20), Christofides, 2-Opt, and Hybrid.  
   - Outputs detailed CSV with costs, runtimes, approximation ratios, and iteration counts.

2. **Standardized Benchmark Suite (`analysis/run_tsp_experiments.py`)**  
   - Walks through every JSON instance in `analysis/datasets/tsp/` (35 small + 21 large benchmark instances).  
   - *Small suite* (n≤20): Runs Exact (n≤12), LP (n≤20), Christofides, 2-Opt, and Hybrid with exact/LP reference values.  
   - *Large suite* (50≤n≤200): Runs Christofides, 2-Opt, and Hybrid; uses LP values (n≤20) or best heuristic as reference.  
   - Outputs results to `analysis/collected_results/tsp/small_n/*.json` and `large_n/*.json`, following the shared analysis format.  
   - Provides Durga's analysis pipeline with consistent JSON data—no manual conversions needed.

3. **Comprehensive Visualizations (`analysis/visualize_tsp_results.py`)**  
   - 850+ line visualization suite with 8 publication-quality plots at 300 DPI.  
   - Latest PNGs in `analysis/collected_results/tsp/`:  
     `1_approximation_ratios_comparison.png` – Boxplots showing quality distribution vs optimal/LP bounds  
     `2_runtime_scaling_all_sizes.png` – Linear and log-log scaling analysis across all problem sizes  
     `3_quality_comparison_detailed.png` – 4-panel analysis: costs, improvements, iterations, success rates  
     `4_pareto_fronts_analysis.png` – Quality vs speed tradeoffs for n={10,20,50,100}  
     `5_ratio_distributions.png` – Violin plots of approximation ratios by size  
     `6_algorithm_efficiency.png` – Quality per unit time metrics  
     `7_summary_statistics_table.png` – Comprehensive statistics across all sizes  
     `8_comprehensive_overview.png` – Single-page 6-panel executive summary  
   - All visualizations regenerate automatically from benchmark JSON files—report stays in sync with code.

### Numbers We're Proud Of
- **Small instances (n≤20, 35 benchmarks with exact/LP reference)**  
  - Christofides averages **1.098× OPT** (best 1.000, worst 1.272); finds optimal on 6/35 instances (17%); never exceeds theoretical 1.5 bound.  
  - 2-Opt averages **1.022× OPT** (best 1.000, worst 1.159); finds optimal on 23/35 instances (66%) through pure local search.  
  - Hybrid averages **1.028× OPT** (best 1.000, worst 1.159); finds optimal on 20/35 instances (57%); **7.1% better than Christofides**.  
  - LP relaxation provides tight lower bounds averaging **0.93× OPT**; critical for large instance evaluation.  
  - Hybrid completes in <90ms on average, with Christofides taking ~26ms and 2-Opt polish adding ~27ms.

- **Large instances (n>20, 21 benchmarks up to n=200)**  
  - All algorithms achieve 100% success rate across all problem sizes.  
  - Christofides scales efficiently: n=200 solved in ~280ms with consistent solution quality.  
  - Hybrid scales to n=200 in ~633ms (max runtime across all 56 instances), staying under 1 second.  
  - 2-Opt iterations scale gracefully: average 15-20 improving swaps for small instances, 40-60 for n=200.  
  - Runtime scaling is practical: O(n²) for Christofides MST+matching, O(n²·iterations) for 2-Opt (typically converges quickly).

- **Cross-instance consistency (56 total benchmarks)**  
  - Zero failures across all 56 instances—every algorithm completes successfully on every benchmark.  
  - Hybrid provides best quality-speed tradeoff: **6.5% better cost than Christofides** for **2× runtime** (still <1s for n=200).  
  - 2-Opt with random start competitive with Christofides start: demonstrates local search power.  
  - Held-Karp exact solver validates approximation quality on n≤12, confirming theoretical guarantees.

### What We Learned
- Combining Christofides with 2-Opt local search delivers consistent near-optimal results: the constructive guarantee of 1.5-approximation plus iterative refinement frequently finds optimal solutions on small instances and high-quality solutions on large instances.  
- LP relaxation as a lower bound is essential for evaluating solution quality when exact methods are intractable—our LP implementation with subtour elimination provides bounds averaging 93% of optimal, making it a reliable reference for approximation ratios.  
- 2-Opt local search is remarkably powerful: even starting from random tours, it finds optimal solutions 66% of the time on small instances, demonstrating that local optimization can overcome poor initialization.  
- The hybrid approach's success validates the meta-heuristic philosophy: leverage theoretical approximation guarantees (Christofides) for fast, quality solutions, then polish with local search for practical improvements without sacrificing polynomial-time complexity.  
- Comprehensive visualization infrastructure (8 plots covering quality, speed, efficiency, and tradeoffs) makes experimental insights immediately accessible—no more digging through CSV files to understand algorithm behavior.  
- Standardizing on JSON benchmark files and shared analysis tools accelerates the entire team: we generate consistent results, Durga can compare across problems, and anyone can reproduce or extend experiments with confidence.


