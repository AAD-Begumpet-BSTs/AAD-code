import numpy as np
import random
import time
import csv
from typing import Set, Dict, List, Tuple, Any

# --- Import your core logic ---
import sc_greedy
import sc_lp
import sc_hybrid 

# --- Import Pranshul's and visualization tools ---
from src.algorithms.exact_solvers import SCExact
from src.utils.data_generator import DataGenerator 

# --- Global Definitions ---
TEST_MODES = [
    {'name': 'Hybrid_Full',       'ls': True,  'trials': 10, 'lp_mode': 'full', 'pd_strat': 'balanced'},
    {'name': 'Hybrid_NoLS',       'ls': False, 'trials': 10, 'lp_mode': 'full', 'pd_strat': 'balanced'}, 
    {'name': 'Hybrid_Trials20',   'ls': True,  'trials': 20, 'lp_mode': 'full', 'pd_strat': 'balanced'},
    {'name': 'Hybrid_LP_Skip',    'ls': True,  'trials': 10, 'lp_mode': 'proxy', 'pd_strat': 'balanced'},
    {'name': 'Hybrid_PD_Fast',    'ls': True,  'trials': 1,  'lp_mode': 'primal_dual', 'pd_strat': 'fast'},
    {'name': 'Hybrid_PD_Balanced','ls': True,  'trials': 1,  'lp_mode': 'primal_dual', 'pd_strat': 'balanced'}
]

MASSIVE_MODES = [
    {'name': 'Greedy_Baseline',   'ls': False, 'trials': 0, 'lp_mode': 'none', 'pd_strat': 'none'}, 
    {'name': 'Hybrid_PD_Fast',    'ls': True, 'trials': 1, 'lp_mode': 'primal_dual', 'pd_strat': 'fast'},
    {'name': 'Hybrid_PD_Balanced','ls': True, 'trials': 1, 'lp_mode': 'primal_dual', 'pd_strat': 'balanced'},
    {'name': 'Hybrid_LP_Skip',    'ls': True, 'trials': 10, 'lp_mode': 'proxy', 'pd_strat': 'balanced'} 
]

# --- Helper Functions ---
def _format_generator_output(universe: Set, subsets_list: List[Set], costs_list: List[float]) -> Tuple[Dict, Dict]:
    subsets_dict = {}
    costs_dict = {}
    for i, s in enumerate(subsets_list):
        set_id = f'S{i}'
        subsets_dict[set_id] = s
        costs_dict[set_id] = costs_list[i]
    return subsets_dict, costs_dict

def _prepare_exact_inputs(subsets: Dict[str, Set], costs: Dict[str, float]) -> Tuple[List[Set], List[float]]:
    sorted_set_ids = sorted(subsets.keys())
    subsets_list = [subsets[sid] for sid in sorted_set_ids]
    costs_list = [costs[sid] for sid in sorted_set_ids]
    return subsets_list, costs_list

def save_to_csv(data: List[Dict[str, Any]], filename: str):
    if not data: return
    fieldnames = list(data[0].keys())
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Saved {len(data)} rows to: {filename}")
    except Exception as e:
        print(f"Warning: Could not save CSV {filename}. Error: {e}")

# --- Experiment Runners ---

def run_small_n_experiment(num_trials=10, seed=42):
    random.seed(seed); np.random.seed(seed)
    results = []
    print(f"\n--- Starting Small N Experiment (n=15, m=25, {num_trials} trials) ---")
    for i in range(num_trials):
        universe, subsets_list, costs_list = DataGenerator.generate_set_cover_instance(15, 25, 0.25, seed=seed+i)
        subsets_dict, costs_dict = _format_generator_output(universe, subsets_list, costs_list)
        subsets_list_exact, costs_list_exact = _prepare_exact_inputs(subsets_dict, costs_dict)
        
        row = {'id': f'Inst-{i+1}'}
        
        # Exact Solver
        exact_solver = SCExact()
        try:
            row['opt_cost'] = exact_solver.solve(universe, subsets_list_exact, costs_list_exact)
        except: continue
        
        # Greedy
        _, greedy_cost, _ = sc_greedy.greedy_set_cover(universe, subsets_dict, costs_dict)
        row['greedy_cost'] = greedy_cost
        
        # Hybrids
        for m in TEST_MODES:
            _, c, _ = sc_hybrid.hybrid_sc_solver(universe, subsets_dict, costs_dict, enable_local_search=m['ls'], num_rounding_trials=m['trials'], lp_mode=m['lp_mode'], pd_strategy=m.get('pd_strat', 'balanced'))
            row[m['name'] + '_cost'] = c
            
        results.append(row)
        print(f"Instance {i+1}: OPT={row.get('opt_cost',0):.1f}, PD_Bal={row.get('Hybrid_PD_Balanced_cost',0):.1f}")
    return results

def run_large_n_scaling_experiment(num_trials=5, n_sizes=[50, 100, 150, 200], seed=42):
    random.seed(seed); np.random.seed(seed)
    results = []
    print("\n--- Starting Medium N Scaling (LP vs PD) ---")
    for n in n_sizes:
        print(f"Running N={n}...")
        for i in range(num_trials):
            univ, sets_list, costs = DataGenerator.generate_set_cover_instance(n, int(1.5*n), 0.20, seed=seed+n+i)
            s_dict, c_dict = _format_generator_output(univ, sets_list, costs)
            
            # LP Lower Bound
            _, lp_lb, _ = sc_lp.solve_lp_sc(univ, s_dict, c_dict)
            if not lp_lb: continue
            
            # Greedy
            _, g_cost, g_time = sc_greedy.greedy_set_cover(univ, s_dict, c_dict)
            results.append({'n': n, 'mode': 'Greedy_Baseline', 'time': g_time, 'ratio': g_cost/lp_lb})
            
            # Hybrids
            for m in TEST_MODES:
                _, c, t = sc_hybrid.hybrid_sc_solver(univ, s_dict, c_dict, enable_local_search=m['ls'], num_rounding_trials=m['trials'], lp_mode=m['lp_mode'], pd_strategy=m.get('pd_strat', 'balanced'))
                results.append({'n': n, 'mode': m['name'], 'time': t, 'ratio': c/lp_lb})
    return results

def run_massive_scale_experiment(n_sizes=[1000, 5000, 10000, 20000], num_trials=3, seed=100):
    random.seed(seed); np.random.seed(seed)
    results = []
    print("\n--- Starting MASSIVE Scale Experiment ---")
    for n in n_sizes:
        print(f"Running N={n}...")
        density = max(0.0001, 100.0 / n) 
        m = int(1.2 * n) 
        for i in range(num_trials):
            univ, sets_list, costs = DataGenerator.generate_set_cover_instance(n, m, density, seed=seed+n+i)
            s_dict, c_dict = _format_generator_output(univ, sets_list, costs)
            
            # Greedy
            _, g_cost, g_time = sc_greedy.greedy_set_cover(univ, s_dict, c_dict)
            results.append({'n': n, 'mode': 'Greedy_Baseline', 'time': g_time, 'cost': g_cost, 'rel_cost': 1.0})
            
            # Primal-Duals
            for m_conf in MASSIVE_MODES:
                if m_conf['name'] == 'Greedy_Baseline': continue
                _, h_cost, h_time = sc_hybrid.hybrid_sc_solver(
                    univ, s_dict, c_dict, 
                    enable_local_search=m_conf['ls'], 
                    num_rounding_trials=m_conf['trials'], 
                    lp_mode=m_conf['lp_mode'], 
                    pd_strategy=m_conf['pd_strat']
                )
                results.append({'n': n, 'mode': m_conf['name'], 'time': h_time, 'cost': h_cost, 'rel_cost': h_cost / g_cost})
            print(f"  Trial {i+1}: Greedy={g_time:.4f}s")
    return results

if __name__ == "__main__":
    # 1. Small N (Accuracy Raw Data)
    small_data = run_small_n_experiment(num_trials=10)
    save_to_csv(small_data, "sc_small_n_raw.csv")

    # 2. Large N (Scaling Raw Data - VITAL for Stats)
    large_data = run_large_n_scaling_experiment(num_trials=10, n_sizes=[50, 100, 150, 200])
    save_to_csv(large_data, "sc_large_n_raw.csv")

    # 3. Massive N (Scalability Raw Data)
    massive_data = run_massive_scale_experiment(n_sizes=[1000, 5000, 10000, 20000])
    save_to_csv(massive_data, "sc_massive_raw.csv")
    
    print("\nALL EXPERIMENTS COMPLETE. Raw data saved to CSVs.")
    print("Run 'sc_statistics.py' to generate plots and analysis.")