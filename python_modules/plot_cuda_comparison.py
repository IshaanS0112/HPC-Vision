import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

naive = pd.read_csv(RESULTS_DIR / "cuda_naive_results.csv")
tiled = pd.read_csv(RESULTS_DIR / "cuda_tiled_results.csv")

# sort for safety
naive = naive.sort_values("size")
tiled = tiled.sort_values("size")

plt.figure(figsize=(10,6))

plt.plot(naive["size"], naive["time"],
         marker='o', label="CUDA Naive")

plt.plot(tiled["size"], tiled["time"],
         marker='o', label="CUDA Tiled")

plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Execution Time (seconds)")
plt.title("CUDA Optimization: Naive vs Tiled")
plt.legend()
plt.grid(True)

output = RESULTS_DIR / "cuda_naive_vs_tiled.png"
plt.savefig(output)

plt.show()

print("Saved:", output)