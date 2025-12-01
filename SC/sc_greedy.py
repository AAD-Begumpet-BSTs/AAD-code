import time

def greedy_set_cover(universe, subsets, costs=None):
    """
    Implements the Greedy Approximation Algorithm for Set Cover.
    
    Logic:
        At each step, pick the set that minimizes: (Cost of Set) / (Number of new items covered)
    
    Args:
        universe (set): The set of all elements to be covered.
        subsets (dict): A dictionary where key is the Set ID (e.g., 'S1', 0) 
                        and value is the set of elements (e.g., {1, 2, 3}).
        costs (dict): Optional. A dictionary mapping Set ID -> Cost (float).
                      If None, assumes Unweighted Set Cover (all costs = 1).
    
    Returns:
        final_solution (list): List of selected Set IDs.
        total_cost (float): The total cost of the solution.
        execution_time (float): Time taken in seconds.
    """
    
    start_time = time.time()
    
    # 1. Validate Inputs
    # Make a copy of the universe to track what is still uncovered
    elements_to_cover = set(universe)
    
    # Handle unweighted case (default cost = 1.0)
    if costs is None:
        costs = {key: 1.0 for key in subsets.keys()}
        
    final_solution = []
    total_cost = 0.0
    
    # 2. The Main Greedy Loop
    # We loop until there are no elements left to cover
    while elements_to_cover:
        
        best_set_id = None
        best_ratio = float('inf') # We want to minimize Cost/Coverage
        
        # Iterate over all sets to find the "best" one for this step
        for set_id, elements in subsets.items():
            
            # Optimization: Skip sets we have already picked
            if set_id in final_solution:
                continue
            
            # Calculate intersection: What NEW things does this set cover?
            # This is the critical efficient Python set operation
            newly_covered = elements.intersection(elements_to_cover)
            num_new = len(newly_covered)
            
            # If the set covers nothing new, it is useless right now
            if num_new > 0:
                # The Greedy Metric: Cost per new element
                current_ratio = costs[set_id] / num_new
                
                # Check if this is the best set so far
                if current_ratio < best_ratio:
                    best_ratio = current_ratio
                    best_set_id = set_id
        
        # 3. Feasibility Check
        # If we finished the for-loop and found nothing, it means the remaining
        # sets cannot cover the remaining elements. The instance is infeasible.
        if best_set_id is None:
            print(f"CRITICAL ERROR: Universe cannot be covered! Remaining elements: {len(elements_to_cover)}")
            return None, float('inf'), 0
            
        # 4. Update State
        # Add best set to solution
        final_solution.append(best_set_id)
        total_cost += costs[best_set_id]
        
        # Remove the covered elements from our 'to do' list
        # efficient in-place update
        elements_to_cover.difference_update(subsets[best_set_id])
        
    end_time = time.time()
    
    return final_solution, total_cost, (end_time - start_time)

# ==========================================
# TEST BLOCK (Runs only if you run this file directly)
# ==========================================
if __name__ == "__main__":
    print("--- Testing Greedy Set Cover ---")
    
    # 1. Define a Universe (e.g., integers 1 to 10)
    U = set(range(1, 11))
    
    # 2. Define Subsets (Some overlapping)
    # Let's make a tricky case where Greedy might not be optimal
    S = {
        'S1': {1, 2, 3, 4, 5},       # Covers 5 items, Cost 10
        'S2': {6, 7, 8, 9, 10},      # Covers 5 items, Cost 10
        'S3': {1, 3, 5, 7, 9},       # Covers mixed, Cost 2
        'S4': {2, 4, 6, 8, 10},      # Covers mixed, Cost 2
        'S5': {1, 2},                # Small, Cost 1
    }
    
    # 3. Define Costs
    C = {
        'S1': 10.0, 
        'S2': 10.0, 
        'S3': 2.0, 
        'S4': 2.0, 
        'S5': 1.0
    }
    
    print(f"Universe Size: {len(U)}")
    print(f"Number of Sets: {len(S)}")
    
    # 4. Run Algorithm
    solution, cost, time_taken = greedy_set_cover(U, S, C)
    
    # 5. Output Results
    print("\n--- Results ---")
    print(f"Selected Sets: {solution}")
    print(f"Total Cost: {cost}")
    print(f"Time Taken: {time_taken:.6f}s")
    
    # Verification (Optional)
    covered_check = set()
    for sid in solution:
        covered_check.update(S[sid])
    
    if covered_check == U:
        print("Verification: VALID (All elements covered)")
    else:
        print(f"Verification: INVALID (Missing {U - covered_check})")