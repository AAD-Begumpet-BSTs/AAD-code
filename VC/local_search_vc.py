# local_search_vc.py
from __future__ import annotations
from typing import Set, Dict, List, Tuple, Optional
from random import sample, shuffle, randint

from graph import Graph
from primal_dual_vc import is_vertex_cover, cover_cost


def local_search_vc_1exchange(
    G: Graph,
    initial_cover: Set[int],
    weights: Optional[Dict[int, float]] = None,
) -> Set[int]:
    """
    Simple 1-exchange local search for (possibly weighted) Vertex Cover.

    Repeatedly tries to remove a single vertex from the cover if the
    remaining set is still a vertex cover. In the weighted case, we
    only ever *remove* vertices, so cost never increases.

    This returns a 1-local-optimum: no single verex can be removed
    without violating the vertex-cover property.
    """
    cover = set(initial_cover)

    # If no weights given, treat as unweighted (all weights = 1)
    if weights is None:
        weights = {v: 1.0 for v in range(G.number_of_vertices())}

    improved = True
    while improved:
        improved = False
        # iterate over snapshot since we modify 'cover'
        for v in list(cover):
            candidate = cover - {v}
            if is_vertex_cover(G, candidate):
                # Always improves cost since we removed a vertex
                cover = candidate
                improved = True
                break  # restart scanning

    return cover


def _uncovered_edges(G: Graph, cover: Set[int]) -> List[Tuple[int, int]]:
    """Return list of edges that are NOT covered by the vertex set 'cover'."""
    uncovered = []
    for (u, v) in G.edges:
        if u not in cover and v not in cover:
            uncovered.append((u, v))
    return uncovered


def _greedy_fix_uncovered(
    G: Graph,
    current_cover: Set[int],
    uncovered: List[Tuple[int, int]],
    weights: Dict[int, float],
    max_new_vertices: int,
) -> Optional[Set[int]]:
    """
    Given a set of uncovered edges and a current partial cover, try to
    greedily add up to 'max_new_vertices' vertices that cover all
    uncovered edges with minimal weight (heuristic).

    Returns:
        A new cover (current_cover ∪ added_vertices) if successful,
        or None if we failed to cover everything within the limit.
    """
    cover = set(current_cover)
    U = list(uncovered)

    while U and max_new_vertices > 0:
        # Score each vertex by (# of uncovered edges it covers) / weight
        score: Dict[int, float] = {}
        for (u, v) in U:
            for x in (u, v):
                if x in cover:
                    continue
                # initialize
                if x not in score:
                    score[x] = 0.0
                # each uncovered edge incident on x contributes 1 / w_x
                score[x] += 1.0 / max(weights.get(x, 1.0), 1e-12)

        if not score:
            # no available vertex can help
            return None

        # pick best vertex by score
        best_v = max(score, key=score.get)
        cover.add(best_v)
        max_new_vertices -= 1

        # recompute uncovered edges
        U = _uncovered_edges(G, cover)

    if not U:
        return cover
    return None


def local_search_vc_k_exchange(
    G: Graph,
    initial_cover: Set[int],
    weights: Optional[Dict[int, float]] = None,
    k: int = 2,
    max_iters: int = 1000,
    samples_per_iter: int = 50,
) -> Set[int]:
    """
    Heuristic k-exchange local search for (possibly weighted) Vertex Cover.

    High-level idea:
      - Start from an initial cover C (e.g., from primal-dual or LP-rounding).
      - First run 1-exchange local search to remove any redundant vertices.
      - Then, for up to 'max_iters':
            * Randomly sample up to 'samples_per_iter' subsets S of C,
              with 1 <= |S| <= k.
            * Let C_minus = C \ S.
            * If C_minus is still a cover and has lower cost, accept it.
            * Otherwise, see which edges become uncovered, and try to
              greedily re-cover them by adding <= |S| vertices with
              minimal additional weight. If this yields a valid cover
              with lower cost, accept it.
      - Stop when no improving move is found or iteration limit is hit.

    Notes:
      - For k = 1, this just calls local_search_vc_1exchange().
      - For k >= 2, this is a randomized heuristic (not exhaustive).
      - In unweighted VC, set weights[v] = 1 for all v.
    """
    if k <= 1:
        return local_search_vc_1exchange(G, initial_cover, weights)

    n = G.number_of_vertices()
    if weights is None:
        weights = {v: 1.0 for v in range(n)}

    # Start from a 1-local-optimal cover
    cover = local_search_vc_1exchange(G, initial_cover, weights)
    current_cost = cover_cost(cover, weights)

    vertices_in_cover: List[int]
    iters = 0

    while iters < max_iters:
        iters += 1
        improved_in_this_iter = False

        vertices_in_cover = list(cover)
        if len(vertices_in_cover) == 0:
            # empty cover only possible if graph has no edges
            break

        # If cover smaller than k, we limit subset size
        max_subset_size = min(k, len(vertices_in_cover))

        # Try a bunch of random subsets S of the current cover
        for _ in range(samples_per_iter):
            subset_size = randint(1, max_subset_size)
            if subset_size > len(vertices_in_cover):
                continue
            S = set(sample(vertices_in_cover, subset_size))

            C_minus = cover - S
            # Case 1: C_minus is still a cover and cheaper -> accept
            if is_vertex_cover(G, C_minus):
                new_cost = cover_cost(C_minus, weights)
                if new_cost + 1e-9 < current_cost:
                    cover = C_minus
                    current_cost = new_cost
                    improved_in_this_iter = True
                    break  # restart outer loop

            # Case 2: C_minus is not a cover, try to fix uncovered edges
            uncovered = _uncovered_edges(G, C_minus)
            # We allow adding up to |S| vertices as replacements.
            fixed_cover = _greedy_fix_uncovered(
                G,
                C_minus,
                uncovered,
                weights,
                max_new_vertices=len(S),
            )
            if fixed_cover is None:
                continue

            if not is_vertex_cover(G, fixed_cover):
                continue

            new_cost = cover_cost(fixed_cover, weights)
            if new_cost + 1e-9 < current_cost:
                cover = fixed_cover
                current_cost = new_cost
                improved_in_this_iter = True
                break  # restart outer loop

        if not improved_in_this_iter:
            # no improving move found in this iteration
            break

    return cover


if __name__ == "__main__":
    # Small demo: see k-exchange improving a naive cover
    from primal_dual_vc import primal_dual_vc_weighted

    # Build a small graph
    G = Graph(5)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4), (1, 3)]
    for (u, v) in edges:
        G.add_edge(u, v)

    # Weights
    weights = {0: 4.0, 1: 2.0, 2: 1.0, 3: 2.0, 4: 3.0}

    print("Graph edges:", G.edges)
    print("Weights:", weights)

    # Start from weighted primal-dual solution
    C0 = primal_dual_vc_weighted(G, weights)
    cost0 = cover_cost(C0, weights)
    print("\nInitial cover (primal-dual):", sorted(C0), "cost =", cost0)

    # 1-exchange local search
    C1 = local_search_vc_1exchange(G, C0, weights)
    cost1 = cover_cost(C1, weights)
    print("After 1-exchange local search:", sorted(C1), "cost =", cost1)

    # k-exchange (k=2) local search
    C2 = local_search_vc_k_exchange(G, C0, weights, k=2, max_iters=200, samples_per_iter=50)
    cost2 = cover_cost(C2, weights)
    print("After 2-exchange local search:", sorted(C2), "cost =", cost2)
