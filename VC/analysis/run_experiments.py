from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.exact_solver import exact_cover_cost, exact_vertex_cover_ilp
from algorithms.hybrid import HybridVCResult, run_hybrid_vc, run_primal_dual
from src.graph import Graph

RESULTS_DIR = (Path(__file__).resolve().parent.parent / "results").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SMALL_SETTINGS = [
    {"n": 12, "p": 0.30, "instances": 3, "seed": 100},
    {"n": 18, "p": 0.35, "instances": 3, "seed": 200},
    {"n": 24, "p": 0.40, "instances": 2, "seed": 300},
]

LARGE_SETTINGS = [
    {"n": 50, "p": 0.30, "instances": 3, "seed": 1000},
    {"n": 75, "p": 0.35, "instances": 3, "seed": 2000},
    {"n": 100, "p": 0.40, "instances": 2, "seed": 3000},
]


def generate_random_graph(n: int, p: float, seed: int) -> Graph:
    rng = random.Random(seed)
    G = Graph(n)
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                G.add_edge(u, v)
    return G


def generate_weights(n: int, seed: int, low: float = 1.0, high: float = 10.0) -> Dict[int, float]:
    rng = random.Random(seed)
    return {v: rng.uniform(low, high) for v in range(n)}


def serialize_hybrid_result(result: HybridVCResult) -> Dict[str, float]:
    return {
        "lp_value": result.lp_value,
        "threshold_cost": result.threshold_cost,
        "best_rounded_cost": result.best_rounded_cost,
        "hybrid_cost": result.hybrid_cost,
        "timings": result.timings,
        "rounds": result.rounds_evaluated,
    }


def run_small_suite(
    settings: Sequence[Dict],
    *,
    num_rounds: int,
    k_exchange: int,
) -> List[Dict]:
    rows: List[Dict] = []

    for cfg in settings:
        n, p, instances, base_seed = cfg["n"], cfg["p"], cfg["instances"], cfg["seed"]
        for offset in range(instances):
            seed = base_seed + offset
            weight_seed = seed + 13

            G = generate_random_graph(n, p, seed)
            weights = generate_weights(n, weight_seed)

            start = time.perf_counter()
            pd_cover, pd_cost = run_primal_dual(G, weights)
            pd_time = time.perf_counter() - start

            start = time.perf_counter()
            exact_cover = exact_vertex_cover_ilp(G, weights)
            exact_time = time.perf_counter() - start
            exact_cost = exact_cover_cost(G, exact_cover, weights)

            hybrid_result = run_hybrid_vc(
                G,
                weights,
                num_rounds=num_rounds,
                k_exchange=k_exchange,
                seed=seed,
            )

            hybrid_time = sum(hybrid_result.timings.values())

            rows.append(
                {
                    "n": n,
                    "p": p,
                    "seed": seed,
                    "num_edges": len(G.edges),
                    "exact_cost": exact_cost,
                    "lp_value": hybrid_result.lp_value,
                    "primal_dual_cost": pd_cost,
                    "hybrid_cost": hybrid_result.hybrid_cost,
                    "primal_dual_ratio": pd_cost / exact_cost if exact_cost else 1.0,
                    "hybrid_ratio": hybrid_result.hybrid_cost / exact_cost if exact_cost else 1.0,
                    "lp_lower_bound_ratio": hybrid_result.lp_value / exact_cost if exact_cost else 1.0,
                    "timings": {
                        "primal_dual": pd_time,
                        "exact": exact_time,
                        "hybrid_total": hybrid_time,
                        **hybrid_result.timings,
                    },
                }
            )
    return rows


def run_large_suite(
    settings: Sequence[Dict],
    *,
    num_rounds: int,
    k_exchange: int,
) -> List[Dict]:
    rows: List[Dict] = []

    for cfg in settings:
        n, p, instances, base_seed = cfg["n"], cfg["p"], cfg["instances"], cfg["seed"]
        for offset in range(instances):
            seed = base_seed + offset
            weight_seed = seed + 31

            G = generate_random_graph(n, p, seed)
            weights = generate_weights(n, weight_seed)

            start = time.perf_counter()
            pd_cover, pd_cost = run_primal_dual(G, weights)
            pd_time = time.perf_counter() - start

            hybrid_result = run_hybrid_vc(
                G,
                weights,
                num_rounds=num_rounds,
                k_exchange=k_exchange,
                seed=seed,
            )
            hybrid_time = sum(hybrid_result.timings.values())

            rows.append(
                {
                    "n": n,
                    "p": p,
                    "seed": seed,
                    "num_edges": len(G.edges),
                    "lp_value": hybrid_result.lp_value,
                    "primal_dual_cost": pd_cost,
                    "hybrid_cost": hybrid_result.hybrid_cost,
                    "hybrid_vs_lp": hybrid_result.hybrid_cost / hybrid_result.lp_value
                    if hybrid_result.lp_value
                    else 1.0,
                    "primal_dual_vs_lp": pd_cost / hybrid_result.lp_value if hybrid_result.lp_value else 1.0,
                    "timings": {
                        "primal_dual": pd_time,
                        "hybrid_total": hybrid_time,
                        **hybrid_result.timings,
                    },
                }
            )
    return rows


def summarize_small(rows: Sequence[Dict]) -> str:
    ratios_pd = [row["primal_dual_ratio"] for row in rows]
    ratios_hybrid = [row["hybrid_ratio"] for row in rows]
    lp_bounds = [row["lp_lower_bound_ratio"] for row in rows]

    return (
        f"Small-n summary over {len(rows)} instances:\n"
        f"  Avg primal-dual ratio : {statistics.mean(ratios_pd):.3f}\n"
        f"  Avg hybrid ratio      : {statistics.mean(ratios_hybrid):.3f}\n"
        f"  Avg LP/OPT ratio      : {statistics.mean(lp_bounds):.3f}\n"
        f"  Best hybrid ratio     : {min(ratios_hybrid):.3f}\n"
        f"  Worst hybrid ratio    : {max(ratios_hybrid):.3f}"
    )


def summarize_large(rows: Sequence[Dict]) -> str:
    ratios_hybrid = [row["hybrid_vs_lp"] for row in rows]
    ratios_pd = [row["primal_dual_vs_lp"] for row in rows]

    return (
        f"Large-n summary over {len(rows)} instances:\n"
        f"  Avg primal-dual / LP : {statistics.mean(ratios_pd):.3f}\n"
        f"  Avg hybrid / LP      : {statistics.mean(ratios_hybrid):.3f}\n"
        f"  Hybrid best (vs LP)  : {min(ratios_hybrid):.3f}\n"
        f"  Hybrid worst (vs LP) : {max(ratios_hybrid):.3f}"
    )


def write_json(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vertex Cover experiment runner (Lohith / VC track)")
    parser.add_argument("--skip-small", action="store_true", help="Skip the small-n experiment suite")
    parser.add_argument("--skip-large", action="store_true", help="Skip the large-n experiment suite")
    parser.add_argument("--rounds", type=int, default=64, help="Randomized rounding trials for hybrid algo")
    parser.add_argument("--k", type=int, default=2, help="k parameter for local search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_small:
        small_rows = run_small_suite(SMALL_SETTINGS, num_rounds=args.rounds, k_exchange=args.k)
        small_path = RESULTS_DIR / "vc_small_experiments.json"
        write_json(small_path, small_rows)
        print(summarize_small(small_rows))
        print(f"  -> Saved detailed rows to {small_path}")

    if not args.skip_large:
        large_rows = run_large_suite(LARGE_SETTINGS, num_rounds=args.rounds, k_exchange=args.k)
        large_path = RESULTS_DIR / "vc_large_experiments.json"
        write_json(large_path, large_rows)
        print(summarize_large(large_rows))
        print(f"  -> Saved detailed rows to {large_path}")


if __name__ == "__main__":
    main()

