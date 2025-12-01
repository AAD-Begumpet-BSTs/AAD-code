4\. Set Cover (Aayush)
----------------------

### What We Built Together

To rigorously evaluate "Beyond Greedy" approaches for Set Cover, we implemented a comprehensive suite of algorithms, ranging from standard combinatorial baselines to advanced hybrid meta-approximations designed to push the boundaries of speed and accuracy.

*   **Greedy Baseline (sc\_greedy.py)** – Our efficiency standard. This algorithm iteratively selects the set that offers the best "cost-per-new-element" density. We implemented an optimized version with a complexity of $O(mn)$, serving as the "Gold Standard" for judging efficiency and a reliable baseline for accuracy comparisons.
    
*   **Hybrid LP Relaxation (sc\_lp.py & sc\_hybrid.py)** – A sophisticated pipeline that solves the full Linear Program using scipy.optimize. It interprets the fractional values from the LP relaxation as probabilities for **Randomized Rounding** to select sets. Crucially, it includes a feasibility repair step to ensure valid covers, providing our theoretical accuracy ceiling and a benchmark for solution quality.
    
*   **Optimized Primal–Dual Suite (sc\_pd\_balanced.py & sc\_pd\_fast.py)** – The core engineering achievement of this vertical. We addressed the computational bottleneck of the LP solver by replacing it with a custom combinatorial Dual-Ascent algorithm. This suite was split into two specialized strategies to explore the trade-off space:
    
    *   **Fast Mode (sc\_pd\_fast.py):** A single-pass heuristic with $O(N)$ complexity, designed specifically for ultra-low latency applications where speed is the only priority.
        
    *   **Balanced Mode (sc\_pd\_balanced.py):** An "Ensemble" strategy that runs 3 deterministic passes (Weighted Density, Frequency, Cost) to explore different structural advantages. This is combined with an iterative **Swap Optimization** cleanup step to maximize accuracy without sacrificing the speed advantage.
        
*   **Reverse Deletion Local Search (sc\_local\_search.py)** – A pruning heuristic applied to Hybrid solutions. Through ablation studies, we confirmed that this step contributes a critical ~2-3% accuracy boost by removing redundant sets, with negligible runtime overhead.
    

### How We Evaluate Now

1.  **Unified Experiment Engine (sc\_experiments.py)**
    
    *   A single master script that orchestrates Data Generation, Execution, and Data Logging. It automatically detects available modes and generates the 3 critical CSVs (small\_n, summary, massive) and 4 distinct plots, ensuring reproducibility and ease of analysis.
        
2.  **Three-Tiered Protocol**
    
    *   **Accuracy Validation ($N=15$):** We compare all algorithms against an **Exact Branch-and-Bound Solver** (SCExact) to measure the true approximation ratio $\\rho = Cost / OPT$. This validates the fundamental correctness of our approaches.
        
    *   **Efficiency Scaling ($N=50 \\to 200$):** We compare results against the **LP Lower Bound**. We introduced and utilized the monotonic efficiency metric $\\Phi\_{\\mu} = \\rho \\cdot \\log\_{10}(1 + T\_{\\mu s})$ to fairly quantify the trade-off between solution quality and speed, especially for sub-second runtimes.
        
    *   **Massive Stress Test ($N=1,000 \\to 20,000$):** We filter out the non-scalable LP methods and pit **Greedy vs. Primal-Dual** on sparse instances to identify asymptotic runtime limits. This highlights the practical difference between $O(N^2)$ and $O(N)$ scaling.
        
3.  **Automated Statistical Rigor (sc\_statistics.py logic)**
    
    *   We integrated rigorous statistical tests directly into our analysis pipeline. This includes Wilcoxon signed-rank tests to compare algorithm performance and Log-Log regression to mathematically confirm scaling exponents ($p$) and significance ($p < 0.05$), moving beyond visual estimation to statistical proof.
        

### Numbers We’re Proud Of

*   **Small Scale Accuracy ($N=15$)**
    
    *   **Hybrid (Full LP)** achieved near-perfect optimality, averaging **1.018× OPT**. It frequently found the exact optimal solution, validating the power of LP guidance.
        
    *   **Greedy** averaged **1.086× OPT**, confirming that the Hybrid approach successfully tightens the empirical approximation gap on small instances.
        
*   **Efficiency & Scaling ($N=200$)**
    
    *   The **Hybrid Primal-Dual (Balanced)** achieved the best efficiency score ($\\Phi = $ \*\*4.62\*\*), beating the Greedy baseline ($\\Phi = $ **4.76**). This validates our optimization strategy.
        
    *   We maintained a competitive accuracy ratio of **1.37× LP** while running **25% faster** than Greedy (0.0026s vs 0.0035s).
        
*   **Massive Scale Scalability ($N=20,000$)**
    
    *   **Greedy Hit the Wall:** The baseline took **175 seconds**, confirming quadratic scaling behavior ($p \\approx 1.9$) likely due to repeated set scanning.
        
    *   **Primal-Dual Flew:** Our optimized algorithm finished in just **9 seconds**, demonstrating linear scaling ($p \\approx 1.0$).
        
    *   **The Trade-off:** We achieved a **19x speedup** for a cost penalty of only **6.5%** (1.065× Greedy cost).
        

### What We Learned and Inferred

*   **The "No Free Lunch" Theorem:** Our results clearly demonstrate that there is no single "best" algorithm for all scenarios. Hybrid (LP) wins on cost but fails on scale. Primal-Dual wins on scale and efficiency but trades a small amount of accuracy. Greedy remains the balanced generalist for mid-sized problems.
    
*   **Software Overhead is the Bottleneck:** The standard LP-based approximation is mathematically sound but practically limited by the solver (PuLP/SciPy). Implementing the logic combinatorially (Primal-Dual) removed the $O(N^3)$ bottleneck, proving that implementation details are as critical as algorithmic theory for performance.
    
*   **Heuristics Matter:** Our "Ensemble" strategy for Primal-Dual (mixing Min-Freq, Max-Freq, and Density passes) was critical. It allowed a fast heuristic to approach the accuracy of the greedy selection without the quadratic cost of re-scanning the matrix at every step.
    
*   **Density-Dependent Performance:** We observed a crossover in performance. For dense instances ($N=200$), Primal-Dual outperformed Greedy in both accuracy and speed. However, for massive sparse instances ($N=20,000$), Greedy's global re-optimization proved slightly more robust in accuracy, albeit at a massive time cost. This suggests Primal-Dual is particularly well-suited for dense, complex problem spaces.
    

