## Vertex Cover (Lohith)

### What We Built Together
- **Primal–Dual 2-Approx (`primal_dual_vc_weighted`)** – our “always on” baseline. For every uncovered edge we raise the dual until one endpoint becomes tight and drop it into the cover. Handles arbitrary non‑negative weights, stays linear in |E|, and gives us instant feasibility on every instance.
- **LP Relaxation + Randomized Rounding (`vc_lp_relaxation.solve_vc_lp_relaxation`)** – we solve the VC LP with PuLP/CBC, treat each `x_v` as an inclusion probability the way Pranshul’s `HybridLPRounding` does, and greedily repair uncovered edges by adding the cheaper endpoint.
- **k-Exchange Local Search (`local_search_vc_k_exchange`)** – after a 1-exchange prune we keep sampling subsets of size ≤ k (we pin k=2 everywhere), remove them, and greedily re-cover edges if the total cost drops. This is the polishing pass that makes rounding results shine.
- **Hybrid Meta-Approx (`algorithms.hybrid.run_hybrid_vc`)** – glues all of the above. We run the LP once, spin through 24–64 randomized rounding trials, polish each candidate via 2-exchange local search, and log detailed timings for LP solve, rounding, and local search phases.

### How We Evaluate Now
1. **Scripted Random Runs (`VC/analysis/run_experiments.py`)**  
   - Generates new Erdős–Rényi `G(n,p)` graphs with weights drawn uniformly in [1,10].  
   - *Small suite* (8 instances): `n ∈ {12,18,24}`; we compare Exact ILP, Primal–Dual, and Hybrid.  
   - *Large suite* (8 instances): `n ∈ {50,75,100}`; we drop exact ILP, compare Primal–Dual vs. Hybrid, and normalize by the LP lower bound.  
   - We explicitly removed the old “massive n” branch—solving LPs on n ≫ 100 inside this driver wasn’t worth the wall-clock time.

2. **Benchmark Sweeps (`analysis/run_vc_benchmarks.py`)**  
   - Newly added tool that walks through every JSON graph in `analysis/datasets/vc/` (15 small + 15 large benchmark instances).  
   - Runs Exact (for n≤30), Primal–Dual, and Hybrid on those fixed inputs and drops the results into `analysis/collected_results/vc_small.json` and `vc_large.json`, following the format documented in `analysis/README.md`.  
   - Gives Durga’s analysis stack plug-and-play data—no more ad-hoc conversions.

3. **Visualizations Living in Version Control**  
   - `analysis/plots/plot_runtime.py` + `plot_ratio.py` now ingest the collected benchmark JSON directly.  
   - Latest PNGs in `analysis/collected_results/`:  
     `VC_exact_runtime.png`, `VC_primal_dual_runtime.png`, `VC_hybrid_total_runtime.png`, `vc_runtime_comparison.png`, `vc_approximation_ratios.png`, `vc_ratio_vs_n.png`.  
   - Everyone can regenerate them via the benchmark runner + plotting helpers, so our slides and report stay in sync with the code.

### Numbers We’re Proud Of
- **Synthetic runs (`VC/results/*.json`)**  
  - Small n: Primal–Dual averages **1.26× OPT** (worst 1.50); Hybrid is **1.00× OPT** on all eight instances; LP lower bound averages **0.85× OPT**. Hybrid finishes <90 ms even with LP + local search; the ILP dominates runtime.  
  - Large n: Primal–Dual averages **1.87× LP**; Hybrid trims that to **1.57× LP** (best 1.40, worst 1.73) while staying below a second at `n=100`.

- **Benchmark datasets (`analysis/collected_results/*.json`)**  
  - 15 “small” graphs (n≤30): Hybrid matches OPT everywhere again; Primal–Dual mean ratio **1.34** (std 0.12).  
  - 15 “large” graphs (50≤n≤200): Hybrid/LP averages **1.58**, Primal–Dual/LP averages **1.84**; hybrid takes ~15‑20 % cost off the primal–dual solution while running under 0.3 s even for `n=200`.

### What We Learned
- Leaning on Pranshul’s LP relaxation plus our randomized rounding and 2-exchange polish gives us a dependable hybrid that hits OPT on small graphs and stays close to the LP bound on larger ones.  
- Killing the massive-n mode was freeing: instead we use the shared dataset files when we need big graphs, and the new benchmark runner keeps runtimes sane.  
- Recording every run (synthetic + benchmark) as JSON and pushing the plots alongside them means we can rerun, audit, or extend the study without guesswork.

