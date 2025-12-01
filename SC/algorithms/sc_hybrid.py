import random
import time
from typing import Set, Dict, List, Tuple

# --- Import your components ---
import sc_lp
import sc_greedy
import sc_local_search
import sc_pd_fast      # The Speed King
import sc_pd_balanced  # The Accuracy Contender

# --- Helper Functions ---
def _get_fast_lp_proxy(subsets: Dict, universe: Set) -> Tuple[Dict, float, float]:
    """Provides a uniform fractional guide (x_j = 0.5) for ablation testing."""
    start_time = time.perf_counter()
    proxy_solution = {sid: 0.5 for sid in subsets.keys()}
    proxy_cost = sum(0.5 * 1.0 for sid in subsets.keys()) 
    return proxy_solution, proxy_cost, 1e-6 

def sc_randomized_rounding_and_repair(universe: Set, subsets: Dict, costs: Dict, lp_solution_dict: Dict, num_trials: int = 10) -> List[str]:
    """Standard Randomized Rounding with Greedy Repair."""
    best_cover_ids = []
    min_cost = float('inf')
    
    for _ in range(num_trials):
        current_cover_ids = []
        # 1. Randomized Selection
        for set_id, fraction in lp_solution_dict.items():
            if random.random() < fraction:
                current_cover_ids.append(set_id)
        
        # 2. Repair Infeasibility
        current_coverage = set().union(*(subsets.get(sid, set()) for sid in current_cover_ids))
        missing_elements = universe - current_coverage
        
        if missing_elements:
            repair_solution, _, _ = sc_greedy.greedy_set_cover(missing_elements, subsets, costs)
            current_cover_ids.extend(repair_solution)
        
        # 3. Keep Best
        current_cost = sum(costs.get(sid, 1.0) for sid in current_cover_ids)
        if current_cost < min_cost:
            min_cost = current_cost
            best_cover_ids = list(current_cover_ids) 
            
    return best_cover_ids


def hybrid_sc_solver(
    universe: Set, 
    subsets: Dict, 
    costs: Dict, 
    lp_mode: str = 'full', 
    enable_local_search: bool = True, 
    num_rounding_trials: int = 10,
    pd_strategy: str = 'balanced'
) -> Tuple[List[str], float, float]:
    """
    Master Hybrid Solver.
    Modes:
      - 'full': Uses SC LP Relaxation (PuLP) -> Rounding -> Local Search
      - 'primal_dual': Uses Combinatorial Primal-Dual (Fast or Balanced) -> Local Search
      - 'proxy': Uses Dummy LP -> Rounding -> Local Search (Ablation)
    """
    start_time = time.perf_counter()
    
    final_cover_ids = []
    
    # 1. Guidance System Selection
    if lp_mode == 'primal_dual':
        # Route to the specific module based on strategy
        if pd_strategy == 'fast':
            initial_feasible_cover, initial_cost, guidance_time = sc_pd_fast.primal_dual_fast(
                universe, subsets, costs
            )
        else:
            # Default to balanced (3-Pass + Swap)
            initial_feasible_cover, initial_cost, guidance_time = sc_pd_balanced.primal_dual_balanced(
                universe, subsets, costs
            )
        
        final_cover_ids = initial_feasible_cover
        
    else: # lp_mode is 'full' or 'proxy'
        if lp_mode == 'proxy':
            lp_solution, lp_cost_lb, guidance_time = _get_fast_lp_proxy(subsets, universe)
        else:
            lp_solution, lp_cost_lb, guidance_time = sc_lp.solve_lp_sc(universe, subsets, costs)
        
        if not lp_solution:
             return [], float('inf'), 0.0

        # Apply Randomized Rounding to the fractional guide
        initial_feasible_cover = sc_randomized_rounding_and_repair(
            universe, subsets, costs, lp_solution, num_trials=num_rounding_trials
        )
        final_cover_ids = initial_feasible_cover

    # 2. Local Search Phase (Polishing)
    # Note: Primal-Dual modules perform their own internal cleanup, 
    # but running this shared LS ensures maximum fairness and accuracy.
    if enable_local_search:
        optimized_cover, final_cost, ls_time = sc_local_search.local_search_sc(
            final_cover_ids, universe, subsets, costs
        )
    else:
        optimized_cover = final_cover_ids
        final_cost = sum(costs.get(sid, 1.0) for sid in optimized_cover)
    
    total_time = (time.perf_counter() - start_time)
    
    return optimized_cover, final_cost, total_time