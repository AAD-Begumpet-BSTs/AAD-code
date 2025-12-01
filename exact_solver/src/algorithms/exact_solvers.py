import pulp


class TSPExact:
    """
    Solves Metric TSP using the Held-Karp Dynamic Programming algorithm.
    Complexity: O(n^2 * 2^n). Suitable for n <= 20.
    """

    def solve(self, dist_matrix):
        n = len(dist_matrix)
        # Maps (subset_mask, last_vertex) -> cost
        # We use frozenset for python dictionary keys to represent subsets
        memo = {}

        def get_cost(subset, last):
            if len(subset) == 1:  # Only the start node (0)
                return dist_matrix[0][last] if last != 0 else 0

            state = (subset, last)
            if state in memo:
                return memo[state]

            prev_subset = subset - {last}
            min_dist = float("inf")

            for prev_node in prev_subset:
                if prev_node == 0 and len(prev_subset) > 1:
                    continue  # 0 must be start
                new_dist = (
                    get_cost(prev_subset, prev_node) + dist_matrix[prev_node][last]
                )
                if new_dist < min_dist:
                    min_dist = new_dist

            memo[state] = min_dist
            return min_dist

        # Final calculation: Min cost to visit all nodes, ending at k, then returning to 0
        all_nodes = frozenset(range(n))
        min_tour_cost = float("inf")

        for k in range(1, n):
            cost = get_cost(all_nodes, k) + dist_matrix[k][0]
            if cost < min_tour_cost:
                min_tour_cost = cost

        return min_tour_cost


class VCExact:
    """
    Solves Minimum Vertex Cover using Exact ILP (Branch-and-Bound via Solver).
    Ref: Section 2.2 Primal-Dual context.
    """

    def solve(self, graph):
        prob = pulp.LpProblem("VertexCoverExact", pulp.LpMinimize)
        nodes = list(graph.nodes())

        # Decision variables: x_i = 1 if node i is in cover
        x = pulp.LpVariable.dicts("x", nodes, cat="Binary")

        # Objective: Minimize size of cover
        prob += pulp.lpSum([x[i] for i in nodes])

        # Constraints: For every edge (u,v), at least one must be chosen
        for u, v in graph.edges():
            prob += x[u] + x[v] >= 1

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return pulp.value(prob.objective)


class SCExact:
    """
    Solves Minimum Set Cover using Exact ILP.
    """

    def solve(self, universe, subsets, costs):
        prob = pulp.LpProblem("SetCoverExact", pulp.LpMinimize)

        # Decision variables: y_j = 1 if subset j is picked
        subset_indices = range(len(subsets))
        y = pulp.LpVariable.dicts("y", subset_indices, cat="Binary")

        # Objective: Minimize total cost
        prob += pulp.lpSum([y[j] * costs[j] for j in subset_indices])

        # Constraints: Every element in universe must be covered by at least one chosen set
        for element in universe:
            prob += (
                pulp.lpSum([y[j] for j in subset_indices if element in subsets[j]]) >= 1
            )

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        return pulp.value(prob.objective)
