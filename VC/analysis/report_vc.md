## Vertex Cover (Lohith)

### Algorithms & Hybrid Pipeline
- **Primal–Dual 2-Approx** (`primal_dual_vc_weighted`): Covers every uncovered edge by adding both endpoints, supporting arbitrary non-negative weights. Serves as the fast baseline.
- **LP Relaxation + Round** (`vc_lp_relaxation.solve_vc_lp_relaxation` + Pranshul-style randomized rounding): Solve the fractional relaxation, then sample each vertex with probability `x_v` followed by a greedy repair on uncovered edges (mirrors `HybridLPRounding` in Pranshul's repo).
- **k-Exchange Local Search** (`local_search_vc_k_exchange`): Starts from any feasible cover, performs 1-exchange pruning, then repeatedly samples subsets up to size `k` and greedily re-covers uncovered edges. We fix `k=2` during experiments for tractability on large instances.
- **Hybrid Meta-Approx** (`algorithms.hybrid.run_hybrid_vc`): Runs the LP solver once, executes 64 randomized-rounding trials, and refines each candidate with 2-exchange local search. Keeps the best rounded cover and the best locally-search-improved cover, reporting their costs and timing breakdowns.

### Experimental Protocol
- **Instances**: Erdős–Rényi `G(n, p)` graphs generated via `analysis/run_experiments.py`. Vertex weights sampled i.i.d. `U[1, 10]`.
- **Small n (exactly comparable)**: `n ∈ {12, 18, 24}` with densities `{0.30, 0.35, 0.40}`; 3,3,2 seeds respectively → 8 instances. We compare Primal–Dual, Hybrid, and the exact ILP solver (`vc_exact_solver.exact_vertex_cover_ilp`).
- **Large n (scalability study)**: `n ∈ {50, 75, 100}` with densities `{0.30, 0.35, 0.40}`; 3,3,2 seeds respectively → 8 instances. Exact ILP is omitted; we benchmark Primal–Dual vs. Hybrid and normalize by the LP value as a common lower bound.
- **Tuning**: Hybrid uses 64 randomized rounds, `k=2`, identical seeds per instance for reproducibility. Raw measurements are stored in `VC/results/vc_small_experiments.json` and `VC/results/vc_large_experiments.json`.

### Results (Small n)
- Avg primal–dual / OPT ratio: **1.26** (worst 1.50).
- Avg hybrid / OPT ratio: **1.00** (all 8 instances matched the exact optimum).
- Avg LP / OPT ratio: **0.85**, confirming the relaxation is a tight lower bound on these densities.
- Runtime: ILP dominates wall-clock (up to 1.52 s for `n=12`, ~0.09 s for `n=24`). Hybrid stays <0.09 s including LP + local search.

### Results (Large n)
- Avg primal–dual / LP ratio: **1.87**, with worst case 1.95 on `n=100`.
- Avg hybrid / LP ratio: **1.58**, improving cost by ~16% over primal–dual on average.
- Hybrid’s best LP-relative ratio: **1.40** on `n=50`, worst: **1.73** on `n=100`.
- Runtime: Hybrid remains sub-second even for `n=100` (≈0.87 s). Primal–Dual solves in milliseconds, so we retain it as a fallback for extremely large or streaming cases.

### Takeaways
- The randomized LP rounding inspired by Pranshul’s implementation, when paired with our 2-exchange local search, consistently recovers optimal covers on small graphs and materially outperforms the pure primal–dual heuristic on larger ones.
- LP values provide practical lower bounds for reporting approximation ratios when exact ILP is infeasible; hybrid’s gap to the bound stays within ~1.4–1.7 on the tested large instances.
- The experiment harness (`analysis/run_experiments.py`) is deterministic, generates JSON logs for analysis, and can be re-run with alternative `--rounds`, `--k`, or `--skip-*` flags for ablation studies or expanded datasets.

