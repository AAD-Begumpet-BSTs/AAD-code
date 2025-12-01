from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ANALYSIS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_ROOT.parent
VC_ROOT = PROJECT_ROOT / "VC"
PRANSHUL_ROOT = PROJECT_ROOT / "pranshul" / "src"

for path in (VC_ROOT, PRANSHUL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from algorithms.exact_solver import exact_cover_cost, exact_vertex_cover_ilp
from algorithms.hybrid import run_hybrid_vc, run_primal_dual
from src.graph import Graph

FILENAME_RE = re.compile(r"vc_n(?P<n>\d+)_p(?P<p>\d+\.\d+)_s(?P<seed>\d+)\.json$")


@dataclass
class InstanceMeta:
    path: Path
    n: int
    p: float
    seed: int


def parse_instance_metadata(path: Path) -> InstanceMeta:
    match = FILENAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Filename does not match expected pattern: {path}")
    n = int(match.group("n"))
    p = float(match.group("p"))
    seed = int(match.group("seed"))
    return InstanceMeta(path=path, n=n, p=p, seed=seed)


def load_instance_graph(meta: InstanceMeta) -> tuple[Graph, Dict[int, float], int]:
    with meta.path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    n = data["n"]
    edges = [tuple(edge) for edge in data["edges"]]
    weights = {int(k): float(v) for k, v in data["weights"].items()}

    graph = Graph.from_edge_list(n, edges)
    num_edges = data.get("m", len(edges))
    return graph, weights, num_edges


def evaluate_instance(
    meta: InstanceMeta,
    *,
    num_rounds: int,
    k_exchange: int,
    exact_limit: int,
) -> dict:
    graph, weights, num_edges = load_instance_graph(meta)
    run_exact = meta.n <= exact_limit

    exact_cost: Optional[float] = None
    exact_time: Optional[float] = None

    if run_exact:
        start = time.perf_counter()
        exact_cover = exact_vertex_cover_ilp(graph, weights)
        exact_time = time.perf_counter() - start
        exact_cost = exact_cover_cost(graph, exact_cover, weights)

    start = time.perf_counter()
    _, primal_dual_cost = run_primal_dual(graph, weights)
    primal_dual_time = time.perf_counter() - start

    hybrid_result = run_hybrid_vc(
        graph,
        weights,
        num_rounds=num_rounds,
        k_exchange=k_exchange,
        seed=meta.seed,
    )
    hybrid_time = sum(hybrid_result.timings.values())

    common_payload = {
        "instance": meta.path.name,
        "n": meta.n,
        "p": meta.p,
        "seed": meta.seed,
        "num_edges": num_edges,
        "lp_value": hybrid_result.lp_value,
        "primal_dual_cost": primal_dual_cost,
        "hybrid_cost": hybrid_result.hybrid_cost,
    }

    if run_exact and exact_cost is not None and exact_cost > 0:
        return {
            **common_payload,
            "exact_cost": exact_cost,
            "primal_dual_ratio": primal_dual_cost / exact_cost,
            "hybrid_ratio": hybrid_result.hybrid_cost / exact_cost,
            "lp_lower_bound_ratio": hybrid_result.lp_value / exact_cost,
            "timings": {
                "exact": exact_time,
                "primal_dual": primal_dual_time,
                "hybrid_total": hybrid_time,
                **hybrid_result.timings,
            },
        }

    lp_value = hybrid_result.lp_value if hybrid_result.lp_value else float("inf")
    return {
        **common_payload,
        "hybrid_vs_lp": hybrid_result.hybrid_cost / lp_value,
        "primal_dual_vs_lp": primal_dual_cost / lp_value,
        "timings": {
            "primal_dual": primal_dual_time,
            "hybrid_total": hybrid_time,
            **hybrid_result.timings,
        },
    }


def collect_instances(dataset_dir: Path) -> List[InstanceMeta]:
    return sorted(
        (parse_instance_metadata(path) for path in dataset_dir.glob("vc_n*_p*_s*.json")),
        key=lambda meta: (meta.n, meta.seed),
    )


def split_results(
    metas: Sequence[InstanceMeta],
    *,
    num_rounds: int,
    k_exchange: int,
    exact_limit: int,
    small_threshold: int,
) -> tuple[List[dict], List[dict]]:
    small_rows: List[dict] = []
    large_rows: List[dict] = []

    for meta in metas:
        row = evaluate_instance(
            meta,
            num_rounds=num_rounds,
            k_exchange=k_exchange,
            exact_limit=exact_limit,
        )
        if meta.n <= small_threshold:
            small_rows.append(row)
        else:
            large_rows.append(row)
    return small_rows, large_rows


def write_json(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VC algorithms on benchmark datasets.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ANALYSIS_ROOT / "datasets" / "vc",
        help="Directory containing VC benchmark JSON instances.",
    )
    parser.add_argument(
        "--output-small",
        type=Path,
        default=ANALYSIS_ROOT / "collected_results" / "vc_small.json",
        help="Output JSON for small-n instances.",
    )
    parser.add_argument(
        "--output-large",
        type=Path,
        default=ANALYSIS_ROOT / "collected_results" / "vc_large.json",
        help="Output JSON for large-n instances.",
    )
    parser.add_argument("--rounds", type=int, default=32, help="Randomized rounding trials for hybrid.")
    parser.add_argument("--k", type=int, default=2, help="k parameter for local search polishing.")
    parser.add_argument(
        "--exact-limit",
        type=int,
        default=30,
        help="Maximum n for which to run the exact ILP solver.",
    )
    parser.add_argument(
        "--small-threshold",
        type=int,
        default=30,
        help="n threshold separating small and large instance summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metas = collect_instances(args.dataset_dir)
    if not metas:
        raise SystemExit(f"No VC instances found in {args.dataset_dir}")

    small_rows, large_rows = split_results(
        metas,
        num_rounds=args.rounds,
        k_exchange=args.k,
        exact_limit=args.exact_limit,
        small_threshold=args.small_threshold,
    )

    if small_rows:
        write_json(args.output_small, small_rows)
        print(f"Wrote {len(small_rows)} small-n rows to {args.output_small}")

    if large_rows:
        write_json(args.output_large, large_rows)
        print(f"Wrote {len(large_rows)} large-n rows to {args.output_large}")


if __name__ == "__main__":
    main()

