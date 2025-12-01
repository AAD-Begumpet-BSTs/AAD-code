import numpy as np
from scipy.optimize import linprog
import time

def solve_lp_sc(universe, subsets, costs=None):
    """
    Solves the Linear Programming Relaxation of Set Cover.
    
    Args:
        universe (set): Set of all elements.
        subsets (dict): Dict {set_id: set_of_elements}.
        costs (dict): Dict {set_id: cost}.
        
    Returns:
        lp_solution (dict): Dict {set_id: fractional_value} (e.g., {'S1': 0.5}).
        lp_obj_val (float): The objective value (lower bound cost).
        time_taken (float): Execution time.
    """
    start_time = time.time()
    
    # 1. Data Preparation
    # We need to map Elements and Sets to integer indices to build the matrix
    sorted_universe = sorted(list(universe))
    sorted_set_ids = sorted(list(subsets.keys()))
    
    elem_to_idx = {elem: i for i, elem in enumerate(sorted_universe)}
    
    n_sets = len(sorted_set_ids)
    n_elements = len(sorted_universe)
    
    # Handle default costs
    if costs is None:
        cost_vector = [1.0] * n_sets
    else:
        cost_vector = [costs[sid] for sid in sorted_set_ids]
        
    # 2. Build the Constraint Matrix A
    # A has dimensions (Num_Elements x Num_Sets)
    # A[i][j] = 1 if Element i is in Set j, else 0
    # Note: This is a dense matrix. For massive data, we'd use scipy.sparse.
    
    A = np.zeros((n_elements, n_sets))
    
    for j, set_id in enumerate(sorted_set_ids):
        for elem in subsets[set_id]:
            if elem in elem_to_idx:
                row_idx = elem_to_idx[elem]
                A[row_idx][j] = 1
    
    # 3. Prepare for linprog
    # Linprog expects: A_ub * x <= b_ub
    # We have:         A * x >= 1
    # Multiply by -1:  -A * x <= -1
    
    A_ub = -1 * A
    b_ub = -1 * np.ones(n_elements)
    
    # Bounds: 0 <= x <= 1 (Relaxation means x is continuous between 0 and 1)
    x_bounds = [(0, 1) for _ in range(n_sets)]
    
    # 4. Solve
    # 'highs' is the modern, fast solver in scipy
    res = linprog(cost_vector, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')
    
    end_time = time.time()
    
    if res.success:
        # Map the result vector back to Set IDs
        lp_solution = {sorted_set_ids[i]: res.x[i] for i in range(n_sets)}
        return lp_solution, res.fun, (end_time - start_time)
    else:
        print("LP Solver Failed:", res.message)
        return None, float('inf'), (end_time - start_time)

# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Testing LP Relaxation for Set Cover ---")
    
    U = {1, 2, 3}
    # A classic case where LP gives a fraction
    # Sets: {1,2}, {2,3}, {1,3}. All cost 1.
    # Optimal Integer Solution: Pick any 2 sets (Cost 2).
    # Optimal LP Solution: Pick 0.5 of each set (0.5+0.5+0.5 = 1.5 Cost).
    
    S = {
        'S1': {1, 2},
        'S2': {2, 3},
        'S3': {1, 3}
    }
    
    print(f"Universe: {U}")
    print("Sets (All cost 1): S1:{1,2}, S2:{2,3}, S3:{1,3}")
    
    solution, val, t = solve_lp_sc(U, S)
    
    print("\n--- Results ---")
    print(f"LP Objective Value (Cost): {val}")
    print("Fractional Solution:")
    for sid, val in solution.items():
        print(f"  {sid}: {val:.4f}")
        
    print(f"\nNote: Integer OPT is 2.0. LP Lower Bound is {val}.")