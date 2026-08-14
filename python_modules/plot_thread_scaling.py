import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
df = pd.read_csv(RESULTS_DIR / "thread_scaling_results.csv")

# Ensure correct column names
df.columns = ["threads", "time"]

# Convert threads to numeric (fix string issue)
df["threads"] = pd.to_numeric(df["threads"], errors="coerce")
df["time"] = pd.to_numeric(df["time"], errors="coerce")

# Drop invalid rows
df = df.dropna()

# Sort by threads
df = df.sort_values("threads")

# Runtime vs Threads
plt.plot(df["threads"], df["time"], marker="o")
plt.xlabel("Threads")
plt.ylabel("Execution Time (seconds)")
plt.title("Runtime vs Threads")
plt.savefig(RESULTS_DIR / "runtime_vs_threads.png")
plt.clf()

# Speedup vs Threads
# Get baseline (threads = 1 or smallest available)
if (df["threads"] == 1).any():
    baseline = df[df["threads"] == 1]["time"].values[0]
else:
    baseline = df.iloc[0]["time"]

speedup = baseline / df["time"]

plt.plot(df["threads"], speedup, marker="o")
plt.xlabel("Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Threads")
plt.savefig(RESULTS_DIR / "speedup_vs_threads.png")
plt.clf()

# Efficiency vs Threads
efficiency = speedup / df["threads"]

plt.plot(df["threads"], efficiency, marker="o")
plt.xlabel("Threads")
plt.ylabel("Efficiency")
plt.title("Parallel Efficiency vs Threads")
plt.savefig(RESULTS_DIR / "efficiency_vs_threads.png")
plt.clf()

print("Thread scaling plots generated successfully.")