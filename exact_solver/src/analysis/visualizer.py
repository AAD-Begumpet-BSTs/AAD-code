import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """
    Handles plotting for Runtime Analysis and Objective Value Comparisons.
    Ref: Section 3.3 Statistical testing and visualization[cite: 57].
    """

    @staticmethod
    def plot_runtime_scaling(results, algorithm_name):
        """
        Plots Input Size (n) vs Runtime (s).
        results: list of tuples (n, time)
        """
        ns, times = zip(*results)

        plt.figure(figsize=(10, 6))
        plt.plot(
            ns,
            times,
            marker="o",
            linestyle="-",
            color="b",
            label=f"{algorithm_name} Exact",
        )

        plt.title(f"{algorithm_name}: Runtime Scaling")
        plt.xlabel("Input Size (n)")
        plt.ylabel("Time (seconds)")
        plt.grid(True)
        plt.legend()
        plt.yscale("log")  # Log scale often better for exponential algos
        plt.savefig(f"visualizations/{algorithm_name}_runtime.png")
        print(f"Saved plot: {algorithm_name}_runtime.png")
        plt.close()

    @staticmethod
    def plot_cost_comparison(instance_ids, exact_costs, approx_costs):
        """
        Bar chart comparing Exact OPT vs LP-Rounded solution.
        """
        x = np.arange(len(instance_ids))
        width = 0.35

        plt.figure(figsize=(10, 6))
        plt.bar(x - width / 2, exact_costs, width, label="Exact (OPT)")
        plt.bar(x + width / 2, approx_costs, width, label="LP Rounding")

        plt.xlabel("Instance ID")
        plt.ylabel("Cost")
        plt.title("Exact vs. LP Relaxation+Rounding Cost Gap")
        plt.xticks(x, instance_ids)
        plt.legend()
        plt.savefig("visualizations/cost_comparison.png")
        print("Saved plot: cost_comparison.png")
        plt.close()
