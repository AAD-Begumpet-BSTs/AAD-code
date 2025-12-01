import time
from typing import Set, Dict, List, Tuple
from collections import defaultdict

def primal_dual_fast(universe: Set, subsets: Dict, costs: Dict) -> Tuple[List[str], float, float]:
    """
    The 'Speed King' Primal-Dual implementation.
    Logic: Single Pass, Sorted by Density. Zero overhead.
    """
    start_time = time.perf_counter()
    
    # 1. Pre-processing (Minimal)
    elem_to_sets = defaultdict(list)
    set_densities = {}
    
    for sid, elems in subsets.items():
        c = costs.get(sid, 1.0)
        size = len(elems)
        # Pre-calc density
        set_densities[sid] = c / size if size > 0 else float('inf')
        for e in elems:
            elem_to_sets[e].append(sid)
            
    # Sort elements by the density of their best set (Greedy-like proxy)
    # Using a list comprehension for the key is faster than a loop here
    universe_list = sorted(list(universe), key=lambda e: min([set_densities[sid] for sid in elem_to_sets[e]]))
    
    # 2. Execution
    uncovered = set(universe)
    solution_ids = [] 
    solution_set = set()
    set_payment = {sid: 0.0 for sid in subsets}
    
    for e in universe_list:
        if e not in uncovered:
            continue
        
        relevant_sets = elem_to_sets[e]
        min_slack = float('inf')
        tight_candidates = []
        
        for sid in relevant_sets:
            current_slack = costs.get(sid, 1.0) - set_payment[sid]
            if current_slack < min_slack:
                min_slack = current_slack
                tight_candidates = [sid]
            elif abs(current_slack - min_slack) < 1e-9:
                tight_candidates.append(sid)
        
        if min_slack > 1e-9:
            for sid in relevant_sets:
                set_payment[sid] += min_slack
        
        for sid in tight_candidates:
            if sid not in solution_set:
                solution_set.add(sid)
                solution_ids.append(sid)
                uncovered.difference_update(subsets[sid])

    # 3. Simple Reverse Deletion (Fast cleanup)
    cover_counts = defaultdict(int)
    for sid in solution_ids:
        for e in subsets[sid]:
            cover_counts[e] += 1
            
    final_solution = []
    for sid in reversed(solution_ids):
        needed = False
        for e in subsets[sid]:
            if cover_counts[e] == 1:
                needed = True
                break
        if needed:
            final_solution.append(sid)
        else:
            for e in subsets[sid]:
                cover_counts[e] -= 1
                
    end_time = time.perf_counter()
    final_cost = sum(costs[sid] for sid in final_solution)
    
    return final_solution, final_cost, (end_time - start_time)