# primal_dual_vc.py
from __future__ import annotations
from typing import Set, Dict, Tuple, List
from src.graph import Graph


def primal_dual_vc_unweighted(G: Graph) -> Set[int]:
    """
    2-approximation algorithm for UNWEIGHTED Vertex Cover
    using a simple primal-dual style rule:

      Initialize cover C = ∅
      For each edge (u, v) in arbitrary order:
          if (u, v) is not yet covered by C:
              add both u and v to C

    Guarantees |C| <= 2 * OPT for unweighted VC.
    """
    cover: Set[int] = set()

    for (u, v) in G.edges:
        # If edge is uncovered, add both endpoints
        if u not in cover and v not in cover:
            cover.add(u)
            cover.add(v)

    return cover


def primal_dual_vc_weighted(G: Graph, weights: Dict[int, float]) -> Set[int]:
    """
    Primal–Dual 2-approximation algorithm for WEIGHTED Vertex Cover.

    LP relaxation (primal):
        min  sum_v w_v x_v
        s.t. x_u + x_v >= 1  for all edges (u, v)
             x_v >= 0

    Dual:
        max  sum_e y_e
        s.t. sum_{e incident to v} y_e <= w_v  for all v
             y_e >= 0

    Algorithm sketch:
      - Start with y_e = 0 for all edges, C = ∅, all vertices 'not tight'.
      - While there exists an uncovered edge (u, v):
          * Increase y_{u,v} uniformly until one of u or v becomes tight:
                sum_{e incident to v} y_e == w_v
          * Add every vertex that becomes tight to the cover C.
      - Return C.

    This yields a 2-approximation for weighted VC.
    """
    n = G.number_of_vertices()
    # Validate weights
    for v in range(n):
        if v not in weights:
            raise ValueError(f"Missing weight for vertex {v}")
        if weights[v] < 0:
            raise ValueError(f"Negative weight for vertex {v}: {weights[v]}")

    m = len(G.edges)
    # Dual variables y_e indexed by edge index
    y: List[float] = [0.0] * m
    # sum_dual[v] = sum of y_e over e incident to v
    sum_dual = [0.0] * n
    tight = [False] * n
    cover: Set[int] = set()

    # Helper: check if edge is covered by current cover
    def edge_covered(u: int, v: int) -> bool:
        return (u in cover) or (v in cover)

    # While some edge is still uncovered, run the dual update step
    # We always pick the first uncovered edge we find.
    while True:
        # Find an uncovered edge
        edge_index = -1
        for i, (u, v) in enumerate(G.edges):
            if not edge_covered(u, v):
                edge_index = i
                break

        if edge_index == -1:
            # all edges are covered
            break

        u, v = G.edges[edge_index]

        # Compute remaining slack for each endpoint:
        # slack[v] = w_v - sum_{e incident to v} y_e
        slack_u = weights[u] - sum_dual[u]
        slack_v = weights[v] - sum_dual[v]

        # We can only increase y_e until one of these hits zero.
        delta = min(slack_u, slack_v)
        if delta < 1e-12:
            # Numerical safety: if delta is ~0, just mark any
            # non-tight endpoint as tight and move on.
            if not tight[u]:
                tight[u] = True
                cover.add(u)
            if not tight[v]:
                tight[v] = True
                cover.add(v)
            continue

        # Raise y for this edge
        y[edge_index] += delta
        sum_dual[u] += delta
        sum_dual[v] += delta

        # Check for vertices that became tight
        for vertex in (u, v):
            if (not tight[vertex]) and (sum_dual[vertex] >= weights[vertex] - 1e-9):
                tight[vertex] = True
                cover.add(vertex)

    return cover


def is_vertex_cover(G: Graph, cover: Set[int]) -> bool:
    """
    Check if 'cover' is a valid vertex cover of graph G.
    """
    for (u, v) in G.edges:
        if u not in cover and v not in cover:
            return False
    return True


def cover_size(cover: Set[int]) -> int:
    return len(cover)


def cover_cost(cover: Set[int], weights: Dict[int, float]) -> float:
    return sum(weights[v] for v in cover)


if __name__ == "__main__":
    # Small sanity tests
    from src.graph import Graph

    # Unweighted example: path on 4 vertices: 0-1-2-3
    G1 = Graph(4)
    G1.add_edge(0, 1)
    G1.add_edge(1, 2)
    G1.add_edge(2, 3)

    C1 = primal_dual_vc_unweighted(G1)
    print("UNWEIGHTED TEST")
    print("G1 edges:", G1.edges)
    print("Cover C1:", C1, "size =", cover_size(C1),
          "valid:", is_vertex_cover(G1, C1))

    # Weighted example: star centered at 0 with different weights
    G2 = Graph(5)
    for v in range(1, 5):
        G2.add_edge(0, v)
    weights2 = {0: 5.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}

    C2 = primal_dual_vc_weighted(G2, weights2)
    print("\nWEIGHTED TEST")
    print("G2 edges:", G2.edges)
    print("Weights:", weights2)
    print("Cover C2:", C2, "cost =", cover_cost(C2, weights2),
          "valid:", is_vertex_cover(G2, C2))
