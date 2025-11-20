# vc_exact_solver.py
from __future__ import annotations
from typing import Dict, Set
from graph import Graph

import pulp


def exact_vertex_cover_ilp(G: Graph, weights: Dict[int, float]) -> Set[int]:
    """
    Solve WEIGHTED Vertex Cover EXACTLY using an ILP:

        min  sum_v w_v x_v
        s.t. x_u + x_v >= 1   for all (u, v) in E
             x_v in {0,1}

    Returns:
        A set of vertices representing an optimal vertex cover.

    Dependencies:
        pip install pulp
    """
    n = G.number_of_vertices()
    # Validate weights
    for v in range(n):
        if v not in weights:
            raise ValueError(f"Missing weight for vertex {v}")
        if weights[v] < 0:
            raise ValueError(f"Negative weight for vertex {v}: {weights[v]}")

    # Define ILP problem
    prob = pulp.LpProblem("Exact_Vertex_Cover", pulp.LpMinimize)

    # Binary variable x_v for each vertex v
    x = {
        v: pulp.LpVariable(f"x_{v}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for v in range(n)
    }

    # Objective: minimize sum w_v x_v
    prob += pulp.lpSum(weights[v] * x[v] for v in range(n))

    # Constraints: for every edge (u, v), ensure coverage
    for (u, v) in G.edges:
        prob += x[u] + x[v] >= 1, f"edge_{u}_{v}"

    # Solve (CBC by default)
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"ILP solver did not find an optimal solution. Status: {status}")

    # Extract solution
    cover: Set[int] = set()
    for v in range(n):
        val = x[v].value()
        if val is None:
            raise RuntimeError(f"Solver returned None for x_{v}")
        if val > 0.5:
            cover.add(v)

    return cover


def exact_cover_cost(G: Graph, cover: Set[int], weights: Dict[int, float]) -> float:
    return sum(weights[v] for v in cover)


if __name__ == "__main__":
    from primal_dual_vc import is_vertex_cover, cover_cost

    # Example graph: 4-cycle 0-1-2-3-0
    G = Graph(4)
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 0)

    # Make vertex 0 expensive, others cheap
    weights = {0: 10.0, 1: 1.0, 2: 1.0, 3: 1.0}

    opt_cover = exact_vertex_cover_ilp(G, weights)
    print("Exact cover:", opt_cover)
    print("Exact cost:", exact_cover_cost(G, opt_cover, weights))
    print("Valid VC?:", is_vertex_cover(G, opt_cover))
