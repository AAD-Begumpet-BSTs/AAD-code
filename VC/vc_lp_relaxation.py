# vc_lp_relaxation.py
from __future__ import annotations
from typing import Dict, Tuple, Set
from graph import Graph
from primal_dual_vc import is_vertex_cover
import pulp


def solve_vc_lp_relaxation(
    G: Graph,
    weights: Dict[int, float] | None = None,
    solver: pulp.LpSolver | None = None,
) -> Tuple[Dict[int, float], float]:
    """
    Solve the LP relaxation of (weighted) Vertex Cover:

        min   sum_v w_v x_v
        s.t.  x_u + x_v >= 1     for all edges (u, v) in E
              0 <= x_v <= 1      for all v in V

    If weights is None, all weights are set to 1.0 (unweighted case).

    Returns:
        (x, obj_value)
          x: dict v -> fractional x_v
          obj_value: optimal LP objective value
    """
    n = G.number_of_vertices()

    # Default to unit weights if not provided
    if weights is None:
        weights = {v: 1.0 for v in range(n)}

    # Sanity check
    for v in range(n):
        if v not in weights:
            raise ValueError(f"Missing weight for vertex {v}")
        if weights[v] < 0:
            raise ValueError(f"Negative weight for vertex {v}: {weights[v]}")

    # Define LP problem
    prob = pulp.LpProblem("VC_LP_Relaxation", pulp.LpMinimize)

    # Variables: 0 <= x_v <= 1
    x = {
        v: pulp.LpVariable(f"x_{v}", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)
        for v in range(n)
    }

    # Objective: minimize sum_v w_v x_v
    prob += pulp.lpSum(weights[v] * x[v] for v in range(n))

    # Constraints: x_u + x_v >= 1 for every edge (u, v)
    for (u, v) in G.edges:
        prob += x[u] + x[v] >= 1, f"edge_{u}_{v}"

    # Solve
    if solver is None:
        # default CBC solver, silent
        solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise RuntimeError(f"LP solver did not find an optimal solution. Status: {status}")

    # Extract solution
    x_val: Dict[int, float] = {}
    for v in range(n):
        val = x[v].value()
        if val is None:
            raise RuntimeError(f"LP returned None for x_{v}")
        x_val[v] = float(val)

    obj_value = float(pulp.value(prob.objective))
    return x_val, obj_value


def round_lp_solution_to_cover(
    G: Graph,
    x: Dict[int, float],
    threshold: float = 0.5,
) -> Set[int]:
    """
    Simple deterministic rounding of LP solution:
        C = { v | x_v >= threshold }

    This ALWAYS gives a valid vertex cover (since x_u + x_v >= 1
    for every edge), but is generally a 2-approximation in the
    unweighted case when threshold = 0.5.

    This is mainly a helper for:
      - debugging
      - plugging into your Hybrid algorithm (before local search).
    """
    cover: Set[int] = {v for v, val in x.items() if val >= threshold}
    # Optional sanity check
    if not is_vertex_cover(G, cover):
        raise RuntimeError("Rounded LP solution did not produce a valid vertex cover.")
    return cover


if __name__ == "__main__":
    # Small demo: compare LP lower bound with exact ILP and a rounded cover
    from primal_dual_vc import cover_cost
    from vc_exact_solver import exact_vertex_cover_ilp, exact_cover_cost

    # Build a small graph
    G = Graph(5)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]
    for (u, v) in edges:
        G.add_edge(u, v)

    # Weights (can be arbitrary non-negative)
    weights = {0: 4.0, 1: 2.0, 2: 1.0, 3: 2.0, 4: 3.0}

    print("Graph edges:", G.edges)
    print("Weights:", weights)

    # 1) Solve LP relaxation
    x_lp, lp_obj = solve_vc_lp_relaxation(G, weights)
    print("\n[VC LP RELAXATION]")
    print("Fractional solution x_v:")
    for v in range(G.number_of_vertices()):
        print(f"  v={v}: x_v={x_lp[v]:.4f}")
    print("LP objective value (lower bound on OPT):", lp_obj)

    # 2) Round LP solution to a cover (simple threshold)
    C_rounded = round_lp_solution_to_cover(G, x_lp, threshold=0.5)
    print("\n[ROUNDED LP SOLUTION]")
    print("Rounded cover:", sorted(C_rounded))
    print("Rounded cost:", cover_cost(C_rounded, weights),
          "Valid VC?:", is_vertex_cover(G, C_rounded))

    # 3) Compare to exact ILP
    C_opt = exact_vertex_cover_ilp(G, weights)
    print("\n[EXACT ILP SOLUTION]")
    print("Optimal cover:", sorted(C_opt))
    print("Optimal cost:", exact_cover_cost(G, C_opt, weights))

    print("\nRelationships:")
    print(f"  LP <= OPT <= Rounded:")
    print(f"    LP    = {lp_obj}")
    print(f"    OPT   = {exact_cover_cost(G, C_opt, weights)}")
    print(f"    Round = {cover_cost(C_rounded, weights)}")
