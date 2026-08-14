import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

def safe_load(name):
    path = RESULTS_DIR / name
    if path.exists():
        try:
            df = pd.read_csv(path)
            if df.empty:
                return None
            return df
        except:
            return None
    return None

serial = safe_load("serial_results.csv")
openmp = safe_load("openmp_results.csv")

plt.figure(figsize=(10, 6))

if serial is not None:
    plt.plot(serial["size"], serial["time"], marker='o', label="Serial CPU")

if openmp is not None:
    plt.plot(openmp["size"], openmp["time"], marker='o', label="OpenMP CPU")

plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Execution Time (seconds)")
plt.title("CPU Performance Comparison")
plt.legend()
plt.grid(True)

out_path = RESULTS_DIR / "final_architecture_comparison.png"
plt.savefig(out_path)

plt.show()