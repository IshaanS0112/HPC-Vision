#!/bin/bash

echo "🚀 Running HPC-Vision Full Pipeline..."

# go to project root
cd "$(dirname "$0")"

echo "📊 Generating plots..."

cd python_modules

# Thread scaling plots
python3 plot_thread_scaling.py

# CPU comparison
python3 plot_results.py

# CUDA comparison
python3 plot_cuda_comparison.py

# GPU comparison (safe)
python3 plot_gpu_comparison.py || echo "Skipping GPU comparison"

# Speedup architecture
python3 plot_speedup_architecture.py

cd ..

echo "✅ All plots generated!"

echo "📂 Results:"
ls results