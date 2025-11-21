### 3. Testing Instructions

1. **File Placement:** Ensure the directory structure matches the imports (e.g., `src/algorithms/exact_solvers.py`).

2. **Execution:** Run the driver script via `uv`:

```bash
uv run main.py
```

---

### 4. Expected Outcomes

- **TSP Runtime (Line Graph):** You will see an exponential curve.
  - For \( n = 10 \), it should be instantaneous (< 0.01s).
  - For \( n = 16 \) or \( 17 \), it might take several seconds to a minute.
  - _Conceptual visualization of what the TSP output will look like._

- **VC Comparison (Bar Chart):**
  - The **Exact** bars will represent the optimal Vertex Cover size (\( OPT \)).
  - The **LP Rounding** bars will be slightly higher or equal to Exact.  
    Since the approximation guarantee for VC is 2, the rounded value will never be more than \( 2 \times OPT \), but typically it is much closer (often \( 1.1\times \) or equal).

- **Console Output:**
  - You will see logs verifying that the ILP solver (`pulp`) is finding solutions.
  - `"Saved plot: ..."` messages indicating the visualizer worked.
