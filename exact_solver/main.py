import time

from src.algorithms.approximation import HybridLPRounding
from src.algorithms.exact_solvers import TSPExact, VCExact
from src.analysis.visualizer import Visualizer
from src.utils.data_generator import DataGenerator


def run_tsp_analysis():
    print("\n--- Running TSP Exact Solver Analysis ---")
    solver = TSPExact()
    results = []

    # Run for n=5 to n=16 (Warning: n=20 takes very long for Held-Karp)
    for n in range(5, 17):
        dist, _ = DataGenerator.generate_euclidean_tsp(n, seed=42)

        start_time = time.perf_counter()
        solver.solve(dist)
        duration = time.perf_counter() - start_time

        results.append((n, duration))
        print(f"n={n}: {duration:.4f}s")

    Visualizer.plot_runtime_scaling(results, "TSP_HeldKarp")


def run_vc_comparison():
    print("\n--- Running Vertex Cover: Exact vs LP+Rounding ---")
    exact_solver = VCExact()
    approx_solver = HybridLPRounding()

    exact_costs = []
    approx_costs = []
    labels = []

    # Compare on fixed size n=20, varying density
    for i in range(5):
        graph = DataGenerator.generate_random_graph_vc(n=20, p=0.3, seed=i)

        # Exact
        c_exact = exact_solver.solve(graph)

        # Approx (LP + Rounding)
        _, c_approx = approx_solver.solve_vc_rounding(graph)

        exact_costs.append(c_exact)
        approx_costs.append(c_approx)
        labels.append(f"Inst_{i}")

        print(f"Instance {i}: OPT={c_exact}, Rounded={c_approx}")

    Visualizer.plot_cost_comparison(labels, exact_costs, approx_costs)


if __name__ == "__main__":
    # 1. Analyze TSP Runtime
    run_tsp_analysis()

    # 2. Compare VC Exact vs Approx
    run_vc_comparison()

    print("\nDone. Check .png files for visualizations.")
