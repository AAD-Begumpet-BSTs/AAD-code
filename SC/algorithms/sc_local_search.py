import time
from typing import List, Set, Dict, Tuple

# Helper function to check if a set of selected IDs covers the universe
def _check_coverage(selected_ids: List[str], universe: Set, subsets: Dict[str, Set]) -> bool:
    """Checks if the union of selected subsets covers the entire universe."""
    
    # Calculate the union of all elements in the selected sets
    current_coverage = set()
    for sid in selected_ids:
        current_coverage.update(subsets.get(sid, set()))
        
    return current_coverage == universe

def local_search_sc(initial_cover_ids: List[str], universe: Set, subsets: Dict[str, Set], costs: Dict[str, float]) -> Tuple[List[str], float, float]:
    """
    Performs Local Search via Reverse Deletion (Pruning) to reduce the cost 
    of an existing feasible Set Cover solution.
    
    Args:
        initial_cover_ids: A list of set IDs representing a feasible cover.
        universe, subsets, costs: The original problem data.
        
    Returns:
        optimized_cover_ids, final_cost, time_taken.
    """
    start_time = time.time()
    
    # 1. Start with a copy of the initial solution
    current_solution = list(initial_cover_ids)
    
    # Sort the current solution by cost in descending order (Heuristic: Try to remove expensive sets first)
    current_solution.sort(key=lambda sid: costs.get(sid, 1.0), reverse=True)
    
    i = 0
    improvement_made = True
    
    # 2. Main Pruning Loop: Repeat until no improvements are made in a full pass
    while improvement_made:
        improvement_made = False
        i = 0
        
        # Iterate through the solution list
        while i < len(current_solution):
            set_to_test = current_solution[i]
            
            # Create a tentative solution without the set_to_test
            # Using slicing/copy for the temporary list
            tentative_solution = current_solution[:i] + current_solution[i+1:]
            
            # Check if the tentative solution is still feasible
            if _check_coverage(tentative_solution, universe, subsets):
                
                # Success! The set was redundant. Remove it permanently.
                current_solution = tentative_solution
                improvement_made = True
                
                # DO NOT increment i, as the next element has moved into position i
                continue
            
            # Failure. The set was essential. Move to the next set.
            i += 1
    
    # 3. Calculate Final Cost and return
    final_cost = sum(costs.get(sid, 1.0) for sid in current_solution)
    end_time = time.time()
    
    return current_solution, final_cost, (end_time - start_time)

# ==========================================
# TEST BLOCK
# ==========================================
if __name__ == "__main__":
    print("--- Testing Local Search (Reverse Deletion) ---")
    
    U = {1, 2, 3, 4}
    S = {
        'S1': {1, 2, 3, 4}, # Full cover, high cost
        'S2': {1, 2},       # Half cover, low cost
        'S3': {3, 4}        # Half cover, low cost
    }
    C = {'S1': 10.0, 'S2': 1.0, 'S3': 1.0}
    
    # Scenario: A rounding algorithm picked all three sets (S1, S2, S3)
    # The actual optimal cover is just {S2, S3} (Cost 2.0)
    initial_solution = ['S1', 'S2', 'S3']
    initial_cost = sum(C[sid] for sid in initial_solution) # 10 + 1 + 1 = 12.0
    
    print(f"Initial Feasible Solution: {initial_solution}")
    print(f"Initial Cost: {initial_cost:.1f}")
    
    # Run Local Search
    optimized_sol, final_cost, t = local_search_sc(initial_solution, U, S, C)
    
    print("\n--- Optimized Results ---")
    print(f"Optimized Solution: {optimized_sol}")
    print(f"Final Cost: {final_cost:.1f}")
    print(f"Time Taken: {t:.6f}s")
    
    if final_cost < initial_cost:
        print(f"Verification: SUCCESS! Cost reduced by {initial_cost - final_cost:.1f}")
    else:
        print("Verification: No improvement made.")