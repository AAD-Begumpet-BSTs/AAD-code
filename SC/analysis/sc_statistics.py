import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import csv

# --- Configuration ---
RAW_SMALL_FILE = "sc_small_n_raw.csv"
RAW_LARGE_FILE = "sc_large_n_raw.csv"
RAW_MASSIVE_FILE = "sc_massive_raw.csv"
VIZ_DIR = "visualizations"

# --- I/O Helper ---
def load_csv(filename):
    if not os.path.exists(filename):
        if os.path.exists(os.path.basename(filename)):
            filename = os.path.basename(filename)
        else:
            print(f"ERROR: {filename} not found. Run sc_experiments.py first.")
            return []
            
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {}
            for k, v in row.items():
                try:
                    clean_row[k] = float(v) if '.' in v else int(v)
                except:
                    clean_row[k] = v 
            data.append(clean_row)
    return data

def save_csv(data, filename):
    if not data: return
    keys = list(data[0].keys())
    try:
        with open(filename, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(data)
        print(f"Saved data: {filename}")
    except Exception as e:
        print(f"Error saving {filename}: {e}")

# --- Metric Helper ---
def calculate_phi_micro(ratio, time_sec):
    if time_sec <= 0: return float('inf')
    return ratio * np.log10(1 + time_sec * 1_000_000)

# --- Statistical Functions ---
def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    if not data: return (0.0, 0.0)
    means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    return np.percentile(means, [(100-ci)/2, 100-(100-ci)/2])

def analyze_scaling(results, mode):
    data = [r for r in results if r['mode'] == mode]
    if not data: return 0.0, 0.0
    n_map = {}
    for r in data:
        n_map.setdefault(r['n'], []).append(r['time'])
    
    ns = np.array(sorted(n_map.keys()))
    times = np.array([np.median(n_map[n]) for n in ns])
    
    if len(ns) < 2: return 0.0, 0.0
    slope, _, r2, _, _ = stats.linregress(np.log(ns), np.log(times))
    return slope, r2

# --- Plotting Functions ---
def plot_runtime_scaling(results):
    plt.figure(figsize=(12, 7))
    modes = sorted(list(set(r['mode'] for r in results)))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    
    for idx, mode in enumerate(modes):
        data = [r for r in results if r['mode'] == mode]
        ns = sorted(list(set(r['n'] for r in data)))
        times = [np.median([x['time'] for x in data if x['n'] == n]) for n in ns]
        label = mode.replace('Hybrid_', '').replace('_Baseline', '')
        plt.plot(ns, times, marker=markers[idx % len(markers)], linestyle='-', label=label, alpha=0.8)

    plt.title('Runtime Scaling: All Algorithms (Medium N)')
    plt.xlabel('Input Size (N)')
    plt.ylabel('Time (s)')
    plt.yscale('log'); plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig(f"{VIZ_DIR}/SC_Combined_Runtime_Scaling.png")
    plt.close()

def plot_efficiency_bar(summary_data):
    plt.figure(figsize=(12, 6))
    modes = [d['Mode'].replace('Hybrid_', '').replace('_Baseline', '') for d in summary_data]
    phis = [d['Avg_Phi'] for d in summary_data]
    
    colors = []
    for m in modes:
        if 'Primal' in m or 'PD' in m: colors.append('green')
        elif 'Greedy' in m: colors.append('blue')
        elif 'Skip' in m: colors.append('red')
        else: colors.append('gray')
        
    bars = plt.bar(modes, phis, color=colors, alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')

    plt.title(r'Final Efficiency Comparison (Lower $\Phi_{\mu}$ is Better)')
    plt.ylabel(r'Phi Metric')
    plt.xlabel('Solver Mode')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(f"{VIZ_DIR}/SC_Final_Efficiency_Comparison.png")
    plt.close()

# --- NEW: Small N Accuracy Plotter ---
def plot_accuracy_bar(results):
    if not results: return
    print("Generating Accuracy Plot...")
    ids = [r['id'] for r in results]
    
    # Dynamically find keys for costs
    modes = ['opt_cost', 'greedy_cost'] 
    for k in results[0].keys():
        if k.endswith('_cost') and k not in modes:
            modes.append(k)
            
    # Filter: only plot key modes to keep chart readable
    # We want OPT, Greedy, Full Hybrid, PD Balanced, PD Fast
    priority = ['opt_cost', 'greedy_cost', 'Hybrid_Full_cost', 'Hybrid_PD_Balanced_cost', 'Hybrid_PD_Fast_cost']
    final_modes = [m for m in priority if m in modes]
    
    labels = {
        'opt_cost': 'Exact', 'greedy_cost': 'Greedy',
        'Hybrid_Full_cost': 'Full', 'Hybrid_PD_Balanced_cost': 'PD(Bal)', 
        'Hybrid_PD_Fast_cost': 'PD(Fast)'
    }
    
    x = np.arange(len(ids))
    width = 0.8 / len(final_modes)
    
    plt.figure(figsize=(14, 7))
    for i, m in enumerate(final_modes):
        costs = [r[m] for r in results]
        offset = (i - len(final_modes)/2) * width + (width/2)
        lbl = labels.get(m, m.replace('_cost',''))
        plt.bar(x + offset, costs, width, label=lbl)
        
    plt.xlabel('Instance ID')
    plt.ylabel('Total Cost')
    plt.title('Accuracy Comparison (Small N)')
    plt.xticks(x, ids)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig(f"{VIZ_DIR}/SC_Accuracy_Comparison_All.png")
    plt.close()
    print("Generated SC_Accuracy_Comparison_All.png")

# --- Main Analysis Controller ---
def run_full_analysis():
    print("\n=== FINAL PROJECT ANALYSIS: SET COVER ===")
    os.makedirs(VIZ_DIR, exist_ok=True)
    
    # 1. SMALL N (Accuracy)
    print("Loading Small N Data...")
    small_data = load_csv(RAW_SMALL_FILE)
    if small_data:
        plot_accuracy_bar(small_data) # Calls the new function
    else:
        print("Warning: Small N data not found. Skipping Accuracy Plot.")

    # 2. MEDIUM N (Scaling & Efficiency)
    print(f"Loading Large N Data from {RAW_LARGE_FILE}...")
    large_data = load_csv(RAW_LARGE_FILE)
    
    if large_data:
        modes = sorted(list(set([r['mode'] for r in large_data])))
        target_n = 200
        
        print("\n--- Large N Stats (N=200) ---")
        print(f"{'Mode':<20} | {'Slope':<6} | {'Mean Ratio':<10} | {'Avg Phi':<8}")
        print("-" * 65)
        
        summary_data = []
        for mode in modes:
            d = [r for r in large_data if r['mode'] == mode]
            if not d: continue
            
            # Stats at Target N
            d_target = [x for x in d if x['n'] == target_n]
            
            slope, r2 = analyze_scaling(large_data, mode)
            
            avg_t = 0.0
            avg_r = 0.0
            avg_phi = 0.0
            
            if d_target:
                avg_t = np.mean([x['time'] for x in d_target])
                avg_r = np.mean([x['ratio'] for x in d_target])
                avg_phi = calculate_phi_micro(avg_r, avg_t)
            
            summary_data.append({'Mode': mode, 'Avg_Time': avg_t, 'Avg_Ratio': avg_r, 'Avg_Phi': avg_phi})
            print(f"{mode:<20} | {slope:.3f}  | {avg_r:.4f}     | {avg_phi:.3f}")

        save_csv(summary_data, "sc_summary_metrics.csv")

        # Generate Analysis Plots
        plot_runtime_scaling(large_data)
        plot_efficiency_bar(summary_data)
        
        # Plot C: Pareto
        print("\nGenerating Pareto Front...")
        plt.figure(figsize=(10, 6))
        for mode in modes:
            d = [r for r in large_data if r['mode'] == mode and r['n'] == target_n]
            if not d: continue
            t = np.mean([x['time'] for x in d])
            r = np.mean([x['ratio'] for x in d])
            
            color = 'blue' if 'Greedy' in mode else ('green' if 'Primal' in mode or 'PD' in mode else 'red')
            if 'Full' in mode or 'NoLS' in mode or 'Trials' in mode: color = 'gray'
            
            plt.scatter(t, r, s=100, c=color, edgecolors='black', label=mode)
            plt.annotate(mode.replace('Hybrid_', '').replace('_Baseline', ''), (t, r), xytext=(5,5), textcoords='offset points')
            
        plt.title(f'Pareto Front (N={target_n})'); plt.xlabel('Time (s)'); plt.ylabel('Ratio'); plt.grid(True, linestyle='--')
        plt.savefig(f"{VIZ_DIR}/SC_Pareto_Front.png"); plt.close()
        print("Generated SC_Pareto_Front.png")

    # 3. MASSIVE N (Scalability)
    massive_data = load_csv(RAW_MASSIVE_FILE)
    if massive_data:
        print("\n--- Massive Scale Analysis ---")
        mass_modes = sorted(list(set([r['mode'] for r in massive_data])))
        
        plt.figure(figsize=(10, 6))
        for mode in mass_modes:
            d = [r for r in massive_data if r['mode'] == mode]
            ns = sorted(list(set([x['n'] for x in d])))
            ts = [np.mean([x['time'] for x in d if x['n'] == n]) for n in ns]
            plt.plot(ns, ts, marker='o', label=mode)
            
            # Print max stats
            d_max = [x for x in d if x['n'] == 20000]
            if d_max:
                 avg_t = np.mean([x['time'] for x in d_max])
                 avg_rel = np.mean([x['rel_cost'] for x in d_max])
                 print(f"{mode:<20} | Time: {avg_t:.4f}s | RelCost: {avg_rel:.3f}x")
        
        plt.legend(); plt.title('Massive Scale Runtime'); plt.xlabel('N'); plt.ylabel('Time (s)'); plt.grid(True)
        plt.savefig(f"{VIZ_DIR}/SC_Massive_Scaling.png"); plt.close()
        print("Generated SC_Massive_Scaling.png")
        
    print("\nALL ANALYSIS COMPLETE.")

if __name__ == "__main__":
    run_full_analysis()