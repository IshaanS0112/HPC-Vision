import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

def safe_load(file):
    path = RESULTS_DIR / file
    if path.exists():
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"{file} is empty")
                return None
            return df
        except:
            print(f"{file} invalid format")
            return None
    else:
        print(f"{file} not found")
        return None

opencl = safe_load("opencl_results.csv")
cuda_naive = safe_load("cuda_naive_results.csv")
cuda_tiled = safe_load("cuda_tiled_results.csv")

# ❗ if no GPU data → fallback
if opencl is None or cuda_naive is None or cuda_tiled is None:
    print("GPU data missing → skipping GPU plot")
    exit()

merged = opencl.merge(cuda_naive, on="size")
merged = merged.merge(cuda_tiled, on="size")

merged.columns = ["size", "opencl", "cuda_naive", "cuda_tiled"]

plt.figure(figsize=(10,6))

plt.plot(merged["size"], merged["opencl"], marker='o', label="OpenCL GPU")
plt.plot(merged["size"], merged["cuda_naive"], marker='o', label="CUDA Naive GPU")
plt.plot(merged["size"], merged["cuda_tiled"], marker='o', label="CUDA Tiled GPU")

plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Execution Time (seconds)")
plt.title("GPU Architecture Comparison (OpenCL vs CUDA)")
plt.legend()
plt.grid(True)

output_path = RESULTS_DIR / "gpu_architecture_comparison.png"
plt.savefig(output_path)

plt.show()

print("Saved:", output_path)