## Exact Solver Report (Pranshul)

We own the exact baselines for all three problems so that everyone else has a concrete OPT to compare against. All solvers live in `pranshul/src/algorithms/exact_solvers.py`, share the same PuLP/CBC backend where possible, and are wired into the team pipelines via straightforward imports (`from src.algorithms.exact_solvers import VCExact`, etc.).

### Vertex Cover – `VCExact`

- **Formulation:** binary ILP with `min ∑ x_v`, subject to `x_u + x_v ≥ 1` for every edge.
- **Usage:** Lohith calls this for any instance with `n ≤ 30` (see `analysis/run_vc_benchmarks.py`) to log the ground-truth cost next to Primal–Dual and Hybrid.
- **Runtime notes:** CBC finishes the benchmark graphs (n up to 30, |E| ≤ 70) in under 0.2 s. For synthetic runs larger than that we intentionally skip the ILP and fall back to LP lower bounds, but the solver remains the reference for plots and ratio calculations.

### Set Cover – `SCExact`

- **Formulation:** binary ILP with decision variables per subset, `min ∑ y_j * c_j`, subject to `∑_{j : e ∈ S_j} y_j ≥ 1` for every element.
- **Integration:** Aayush plugs this into the small-n experiments in `SC/sc_experiments.py` to report OPT alongside the greedy, LP, and hybrid costs.
- **Scaling:** Works comfortably for the 15-element, 25-set regime used in Set Cover’s “exact comparison” suite; anything larger gets relegated to approximation-only runs.

### Traveling Salesman – `TSPExact`

- **Algorithm:** Held–Karp dynamic program (O(n²·2ⁿ)), implemented with memoization over `(subset, last)` states.
- **Purpose:** Provides the OPT tour cost for Saharsh’s smaller Euclidean instances so his Christofides + 2‑opt hybrids have a real benchmark.
- **Practical limit:** n ≈ 18–20 on our machines; beyond that we rely on approximation-only charts.

### How We Use These Results

1. **Ground-truth JSON:** Every experiment driver that can afford the ILP/Dynamic Program writes the exact cost directly into its output JSON (e.g., Lohith’s `vc_small_experiments.json`, analysis’ `vc_small.json`). That makes approximation-ratio plots trivial to regenerate.
2. **Fair comparison plots:** The new VC runtime and ratio figures (`analysis/collected_results/*.png`) show Exact vs. Primal–Dual vs. Hybrid using our solvers’ timings/costs. The same pattern is being replicated for SC/TSP once their benchmark sweeps land.
3. **Regression safety net:** Because all solvers sit in this single module, any change to data schemas or graph loaders only has to be updated once for every team to keep getting OPT values.

If you need to reproduce the numbers from scratch:

```bash
cd pranshul
uv run main.py          # Sanity check the drivers + visualizations
cd ../analysis
python run_vc_benchmarks.py  # Re-run benchmark sweeps (calls VCExact internally)
```
