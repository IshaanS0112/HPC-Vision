import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Load data
serial = pd.read_csv(RESULTS_DIR / "serial_results.csv")
openmp = pd.read_csv(RESULTS_DIR / "openmp_results.csv")
cuda = pd.read_csv(RESULTS_DIR / "cuda_naive_results.csv")

# Merge only common sizes
merged = serial.merge(openmp, on="size", suffixes=("_serial", "_openmp"))
merged = merged.merge(cuda, on="size")

# Rename columns
merged.columns = ["size", "serial", "openmp", "cuda"]

# Compute speedups
openmp_speedup = merged["serial"] / merged["openmp"]
cuda_speedup = merged["serial"] / merged["cuda"]

# Plot
plt.figure(figsize=(10,6))

plt.plot(merged["size"], openmp_speedup, marker='o', label="OpenMP Speedup")
plt.plot(merged["size"], cuda_speedup, marker='o', label="CUDA Speedup")

plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Speedup (vs Serial)")
plt.title("Speedup Comparison Across Architectures")
plt.legend()
plt.grid(True)

# Save
out_path = RESULTS_DIR / "speedup_architecture.png"
plt.savefig(out_path)

# Show
plt.show()

print("Saved:", out_path)