# run_vc_demo.py
from __future__ import annotations
from typing import Dict
from graph import Graph
from primal_dual_vc import (
    primal_dual_vc_unweighted,
    primal_dual_vc_weighted,
    is_vertex_cover,
    cover_size,
    cover_cost,
)
from vc_exact_solver import exact_vertex_cover_ilp, exact_cover_cost


def main():
    # Small demo graph: 5-vertex graph with some structure
    G = Graph(5)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]
    for (u, v) in edges:
        G.add_edge(u, v)

    # Weights: make vertex 2 a bit cheaper, vertex 0 more expensive
    weights: Dict[int, float] = {0: 4.0, 1: 2.0, 2: 1.0, 3: 2.0, 4: 3.0}

    print("Graph edges:", G.edges)
    print("Weights:", weights)

    # 1) Unweighted primal-dual (treat all weights as 1)
    C_unw = primal_dual_vc_unweighted(G)
    print("\n[UNWEIGHTED 2-APPROX]")
    print("Cover:", sorted(C_unw))
    print("Size:", cover_size(C_unw), "Valid?:", is_vertex_cover(G, C_unw))

    # 2) Weighted primal-dual 2-approx
    C_w = primal_dual_vc_weighted(G, weights)
    print("\n[WEIGHTED PRIMAL-DUAL 2-APPROX]")
    print("Cover:", sorted(C_w))
    print("Cost:", cover_cost(C_w, weights), "Valid?:", is_vertex_cover(G, C_w))

    # 3) Exact solution via ILP
    C_opt = exact_vertex_cover_ilp(G, weights)
    print("\n[EXACT ILP SOLVER]")
    print("Optimal cover:", sorted(C_opt))
    print("Optimal cost:", exact_cover_cost(G, C_opt, weights),
          "Valid?:", is_vertex_cover(G, C_opt))

    # Quick approximation ratio check (for this instance)
    approx_ratio = cover_cost(C_w, weights) / exact_cover_cost(G, C_opt, weights)
    print("\nApprox ratio (weighted primal-dual vs OPT):", approx_ratio)


if __name__ == "__main__":
    main()
