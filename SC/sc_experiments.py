import numpy as np
import random
import time
import csv
import matplotlib.pyplot as plt
from typing import Set, Dict, List, Tuple, Any

# --- Import your core logic ---
import sc_greedy
import sc_lp
import sc_hybrid 

# --- Import Pranshul's and visualization tools ---
from src.algorithms.exact_solvers import SCExact
from src.analysis.visualizer import Visualizer
from src.utils.data_generator import DataGenerator 

# --- Global Definitions ---
# Full suite for Small/Medium N
TEST_MODES = [
    {'name': 'Hybrid_Full',       'ls': True,  'trials': 10, 'lp_mode': 'full', 'pd_strat': 'balanced'},
    {'name': 'Hybrid_NoLS',       'ls': False, 'trials': 10, 'lp_mode': 'full', 'pd_strat': 'balanced'}, 
    {'name': 'Hybrid_Trials20',   'ls': True,  'trials': 20, 'lp_mode': 'full', 'pd_strat': 'balanced'},
    {'name': 'Hybrid_LP_Skip',    'ls': True,  'trials': 10, 'lp_mode': 'proxy', 'pd_strat': 'balanced'},
    {'name': 'Hybrid_PD_Fast',    'ls': True,  'trials': 1,  'lp_mode': 'primal_dual', 'pd_strat': 'fast'},
    {'name': 'Hybrid_PD_Balanced','ls': True,  'trials': 1,  'lp_mode': 'primal_dual', 'pd_strat': 'balanced'}
]

# Scalable algorithms for Massive N (Added LP_Skip)
MASSIVE_MODES = [
    {'name': 'Greedy_Baseline',   'ls': False, 'trials': 0, 'lp_mode': 'none', 'pd_strat': 'none'}, # Special handling in loop
    {'name': 'Hybrid_PD_Fast',    'ls': True, 'trials': 1, 'lp_mode': 'primal_dual', 'pd_strat': 'fast'},
    {'name': 'Hybrid_PD_Balanced','ls': True, 'trials': 1, 'lp_mode': 'primal_dual', 'pd_strat': 'balanced'},
    # ADDED: To show that Rounding is scalable but inaccurate without LP
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

def _save_summary_to_csv(summary_data: List[Dict[str, Any]], filename="sc_summary_metrics.csv"):
    fieldnames = list(summary_data[0].keys())
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_data)
        print(f"\nSaved summary metrics to: {filename}")
    except Exception as e:
        print(f"\nWarning: Could not save CSV file. Error: {e}")

def calculate_phi_micro(ratio, time_sec):
    if time_sec <= 0: return float('inf')
    time_micro = time_sec * 1_000_000
    return ratio * np.log10(1 + time_micro)

# --- Experiment Runners ---

def run_small_n_experiment(num_trials=10, seed=42):
    random.seed(seed); np.random.seed(seed)
    results = []
    print(f"Starting Small N Experiment (n=15, m=25, {num_trials} trials)...")
    for i in range(num_trials):
        universe, subsets_list, costs_list = DataGenerator.generate_set_cover_instance(15, 25, 0.25, seed=seed+i)
        subsets_dict, costs_dict = _format_generator_output(universe, subsets_list, costs_list)
        exact = SCExact()
        try: opt = exact.solve(universe, *_prepare_exact_inputs(subsets_dict, costs_dict))
        except: continue
        
        _, greedy_cost, _ = sc_greedy.greedy_set_cover(universe, subsets_dict, costs_dict)
        
        row = {'id': f'Inst-{i+1}', 'opt': opt, 'greedy': greedy_cost}
        for m in TEST_MODES:
            _, c, _ = sc_hybrid.hybrid_sc_solver(universe, subsets_dict, costs_dict, enable_local_search=m['ls'], num_rounding_trials=m['trials'], lp_mode=m['lp_mode'], pd_strategy=m.get('pd_strat', 'balanced'))
            row[m['name']] = c
        results.append(row)
        print(f"Instance {i+1}: OPT={opt:.1f}, Greedy={greedy_cost:.1f}, PD_Bal={row['Hybrid_PD_Balanced']:.1f}")
    return results

def run_large_n_scaling_experiment(num_trials=5, n_sizes=[50, 100, 150, 200], seed=42):
    random.seed(seed); np.random.seed(seed)
    results = []
    print("\nStarting Medium N Scaling (LP vs PD)...")
    for n in n_sizes:
        for i in range(num_trials):
            univ, sets_list, costs = DataGenerator.generate_set_cover_instance(n, int(1.5*n), 0.20, seed=seed+n+i)
            s_dict, c_dict = _format_generator_output(univ, sets_list, costs)
            
            # LP Bound for Ratio
            _, lp_lb, _ = sc_lp.solve_lp_sc(univ, s_dict, c_dict)
            if not lp_lb: continue
            
            # Greedy
            _, g_cost, g_time = sc_greedy.greedy_set_cover(univ, s_dict, c_dict)
            results.append({'n': n, 'mode': 'Greedy_Baseline', 'time': g_time, 'ratio': g_cost/lp_lb})
            
            # Hybrids
            for m in TEST_MODES:
                _, c, t = sc_hybrid.hybrid_sc_solver(univ, s_dict, c_dict, enable_local_search=m['ls'], num_rounding_trials=m['trials'], lp_mode=m['lp_mode'], pd_strategy=m.get('pd_strat', 'balanced'))
                results.append({'n': n, 'mode': m['name'], 'time': t, 'ratio': c/lp_lb})
        print(f"N={n} complete.")
    return results

def run_massive_scale_experiment(n_sizes=[1000, 5000, 10000, 20000], num_trials=3, seed=100):
    random.seed(seed); np.random.seed(seed)
    results = []
    print("\nStarting MASSIVE Scale Experiment (Scalable Algos Only)...")
    
    for n in n_sizes:
        print(f"--- Running N={n} (x{num_trials} trials) ---")
        
        # Adjust density to keep problem feasible but sparse
        density = max(0.0001, 100.0 / n) 
        m = int(1.2 * n) 
        
        for i in range(num_trials):
            univ, sets_list, costs = DataGenerator.generate_set_cover_instance(n, m, density, seed=seed+n+i)
            s_dict, c_dict = _format_generator_output(univ, sets_list, costs)
            
            # 1. Greedy (Reference)
            _, g_cost, g_time = sc_greedy.greedy_set_cover(univ, s_dict, c_dict)
            results.append({
                'n': n, 'mode': 'Greedy_Baseline', 
                'time': g_time, 'cost': g_cost, 'rel_cost': 1.0
            })
            
            # 2. Scalable Hybrids
            for m_conf in MASSIVE_MODES:
                if m_conf['name'] == 'Greedy_Baseline': continue
                
                _, h_cost, h_time = sc_hybrid.hybrid_sc_solver(
                    univ, s_dict, c_dict, 
                    enable_local_search=m_conf['ls'], 
                    num_rounding_trials=m_conf['trials'], 
                    lp_mode=m_conf['lp_mode'], 
                    pd_strategy=m_conf['pd_strat']
                )
                
                results.append({
                    'n': n, 'mode': m_conf['name'], 
                    'time': h_time, 'cost': h_cost, 'rel_cost': h_cost / g_cost
                })
            
            print(f"  Trial {i+1}: Greedy={g_time:.4f}s")
            
    return results

# --- Main ---
if __name__ == "__main__":
    
    # 1. Medium Scale (Existing Logic)
    medium_results = run_large_n_scaling_experiment(num_trials=3, n_sizes=[100, 200])
    
    # Save Medium Summary
    summary_med = []
    all_modes = ['Greedy_Baseline'] + [m['name'] for m in TEST_MODES]
    for mode in all_modes:
        data = [r for r in medium_results if r['mode'] == mode]
        if not data: continue
        avg_t = np.mean([d['time'] for d in data])
        avg_r = np.mean([d['ratio'] for d in data])
        phi = calculate_phi_micro(avg_r, avg_t)
        summary_med.append({'Mode': mode, 'Avg_Time': avg_t, 'Avg_Ratio': avg_r, 'Avg_Phi': phi})
    _save_summary_to_csv(summary_med, "sc_summary_metrics.csv") # Original CSV for report consistency

    # 2. Massive Scale
    massive_results = run_massive_scale_experiment(n_sizes=[1000, 5000, 10000, 20000])
    
    # Analyze Best/Avg/Worst for Massive (Largest N)
    summary_mass = []
    massive_modes = [m['name'] for m in MASSIVE_MODES]
    
    print("\n--- Massive Scale Analysis (N=20,000) ---")
    print(f"{'Mode':<20} | {'Time (Avg)':<10} | {'Time (Best)':<10} | {'Cost vs Grd (Avg)':<15}")
    print("-" * 65)
    
    for mode in massive_modes:
        max_n = 20000
        data = [r for r in massive_results if r['mode'] == mode and r['n'] == max_n]
        if not data: continue
        
        times = [d['time'] for d in data]
        rels = [d['rel_cost'] for d in data]
        
        print(f"{mode:<20} | {np.mean(times):.4f}s    | {np.min(times):.4f}s    | {np.mean(rels):.3f}x")
        
        summary_mass.append({
            'Mode': mode, 'Max_N': max_n,
            'Time_Avg': np.mean(times), 'Time_Min': np.min(times), 'Time_Max': np.max(times),
            'RelCost_Avg': np.mean(rels), 'RelCost_Min': np.min(rels), 'RelCost_Max': np.max(rels)
        })
        
    _save_summary_to_csv(summary_mass, "sc_massive_stats.csv")
    
    # Plot Massive Scaling
    plt.figure(figsize=(10, 6))
    for mode in massive_modes:
        data = [r for r in massive_results if r['mode'] == mode]
        ns = sorted(list(set(r['n'] for r in data)))
        avg_times = []
        for n in ns:
            t = np.mean([r['time'] for r in data if r['n'] == n])
            avg_times.append(t)
        
        plt.plot(ns, avg_times, marker='o', label=mode)
        
    plt.title('Massive Scale Runtime: Primal-Dual vs Greedy vs Skip')
    plt.xlabel('Input Size (N)')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.grid(True)
    plt.savefig("visualizations/SC_Massive_Scaling.png")
    print("Saved plot: SC_Massive_Scaling.png")