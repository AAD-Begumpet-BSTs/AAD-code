from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from algorithms.local_search import local_search_vc_k_exchange
from algorithms.primal_dual import cover_cost, is_vertex_cover, primal_dual_vc_weighted
from algorithms.lp_relaxation import round_lp_solution_to_cover, solve_vc_lp_relaxation
from src.graph import Graph


@dataclass
class HybridVCResult:
    """
    Container for the different stages of the VC hybrid pipeline.
    """

    lp_value: float
    lp_solution: Dict[int, float]
    threshold_cover: Set[int]
    threshold_cost: float
    best_rounded_cover: Set[int]
    best_rounded_cost: float
    hybrid_cover: Set[int]
    hybrid_cost: float
    timings: Dict[str, float]
    rounds_evaluated: int


def _ensure_weights(G: Graph, weights: Optional[Dict[int, float]]) -> Dict[int, float]:
    if weights is not None:
        return weights
    return {v: 1.0 for v in range(G.number_of_vertices())}


def randomized_rounding_with_repair(
    G: Graph,
    lp_solution: Dict[int, float],
    weights: Dict[int, float],
    rng: random.Random,
) -> Set[int]:
    """
    Adaptation of Pranshul's randomized rounding routine
    (see pranshul/src/algorithms/approximation.py).

    Each vertex v is added independently with probability x_v.
    Any uncovered edge is repaired by adding the cheaper endpoint.
    """

    cover: Set[int] = set()

    for v, prob in lp_solution.items():
        if rng.random() < prob:
            cover.add(v)

    for (u, v) in G.edges:
        if u in cover or v in cover:
            continue
        if weights[u] <= weights[v]:
            cover.add(u)
        else:
            cover.add(v)

    if not is_vertex_cover(G, cover):
        raise RuntimeError("Randomized rounding failed to produce a valid cover.")
    return cover


def run_hybrid_vc(
    G: Graph,
    weights: Optional[Dict[int, float]] = None,
    *,
    num_rounds: int = 64,
    k_exchange: int = 2,
    seed: int = 0,
) -> HybridVCResult:
    """
    Execute the LP + randomized rounding + k-exchange local search pipeline.

    Args:
        G: Graph instance.
        weights: optional vertex weights (default 1.0 each).
        num_rounds: number of randomized rounding trials.
        k_exchange: parameter for local_search_vc_k_exchange.
        seed: RNG seed for reproducibility.
    """

    if num_rounds <= 0:
        raise ValueError("num_rounds must be positive.")
    if k_exchange < 1:
        raise ValueError("k_exchange must be >= 1.")

    weights = _ensure_weights(G, weights)
    rng = random.Random(seed)

    timings = {"lp_solve": 0.0, "rounding": 0.0, "local_search": 0.0}

    start = time.perf_counter()
    lp_solution, lp_value = solve_vc_lp_relaxation(G, weights)
    timings["lp_solve"] = time.perf_counter() - start

    threshold_cover = round_lp_solution_to_cover(G, lp_solution, threshold=0.5)
    threshold_cost = cover_cost(threshold_cover, weights)

    best_rounded_cover: Set[int] = set(threshold_cover)
    best_rounded_cost = threshold_cost
    best_hybrid_cover: Set[int] = set(threshold_cover)
    best_hybrid_cost = threshold_cost

    for _ in range(num_rounds):
        start = time.perf_counter()
        rounded_cover = randomized_rounding_with_repair(G, lp_solution, weights, rng)
        timings["rounding"] += time.perf_counter() - start

        rounded_cost = cover_cost(rounded_cover, weights)
        if rounded_cost + 1e-9 < best_rounded_cost:
            best_rounded_cover = rounded_cover
            best_rounded_cost = rounded_cost

        start = time.perf_counter()
        hybrid_cover = local_search_vc_k_exchange(
            G,
            rounded_cover,
            weights,
            k=k_exchange,
        )
        timings["local_search"] += time.perf_counter() - start

        hybrid_cost = cover_cost(hybrid_cover, weights)
        if hybrid_cost + 1e-9 < best_hybrid_cost:
            best_hybrid_cover = hybrid_cover
            best_hybrid_cost = hybrid_cost

    return HybridVCResult(
        lp_value=lp_value,
        lp_solution=lp_solution,
        threshold_cover=threshold_cover,
        threshold_cost=threshold_cost,
        best_rounded_cover=best_rounded_cover,
        best_rounded_cost=best_rounded_cost,
        hybrid_cover=best_hybrid_cover,
        hybrid_cost=best_hybrid_cost,
        timings=timings,
        rounds_evaluated=num_rounds,
    )


def run_primal_dual(G: Graph, weights: Optional[Dict[int, float]] = None) -> Tuple[Set[int], float]:
    """Convenience wrapper returning the cover and its cost."""

    weights = _ensure_weights(G, weights)
    cover = primal_dual_vc_weighted(G, weights)
    return cover, cover_cost(cover, weights)

