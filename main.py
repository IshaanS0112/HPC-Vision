"""Build and run the serial and OpenMP benchmarks, then plot.

Usage:
    python main.py               # every size that has generated input matrices
    python main.py 256 512 1024  # just these sizes

The C programs take the matrix size as their only argument and use paths
relative to openmp/ (../datasets for input, ../results for output), so they are
compiled into openmp/ and run from there. Each run appends a row to the CSVs
in results/.
"""

import os
import platform
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OPENMP_DIR = os.path.join(ROOT, "openmp")
DATASETS_DIR = os.path.join(ROOT, "datasets")
THREAD_COUNTS = [4, 8]


def available_sizes():
    sizes = []
    for name in os.listdir(DATASETS_DIR):
        match = re.fullmatch(r"matrixA_(\d+)\.txt", name)
        if match and os.path.exists(os.path.join(DATASETS_DIR, f"matrixB_{match.group(1)}.txt")):
            sizes.append(int(match.group(1)))
    return sorted(sizes)


def openmp_flags():
    if platform.system() != "Darwin":
        return ["-fopenmp"]
    # Apple Clang ships no OpenMP runtime; use Homebrew's libomp.
    try:
        prefix = subprocess.run(
            ["brew", "--prefix", "libomp"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        prefix = "/opt/homebrew/opt/libomp"
    return ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include", f"-L{prefix}/lib", "-lomp"]


def compile_programs():
    print("\nCompiling...")
    subprocess.run(["gcc", "matrix_serial.c", "-o", "serial_run"], cwd=OPENMP_DIR, check=True)
    subprocess.run(
        ["gcc", "matrix_openmp.c", *openmp_flags(), "-o", "openmp_run"],
        cwd=OPENMP_DIR,
        check=True,
    )


def run_serial(sizes):
    print("\nRunning Serial Version...")
    for n in sizes:
        subprocess.run(["./serial_run", str(n)], cwd=OPENMP_DIR, check=True)


def run_openmp(sizes):
    print("\nRunning OpenMP Version...")
    for threads in THREAD_COUNTS:
        print(f"\nRunning with {threads} threads...")
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        for n in sizes:
            subprocess.run(["./openmp_run", str(n)], cwd=OPENMP_DIR, env=env, check=True)


def run_cuda_note():
    print("\nCUDA versions already executed inside Google Colab.")
    print("Refer stored results for CUDA Naive and CUDA Tiled.")


def plot_results():
    print("\nGenerating performance graphs...")
    subprocess.run([sys.executable, "python_modules/plot_results.py"], cwd=ROOT)


if __name__ == "__main__":
    sizes = [int(arg) for arg in sys.argv[1:]] or available_sizes()
    missing = [n for n in sizes if n not in available_sizes()]
    if not sizes or missing:
        print(
            "No input matrices for: " + (", ".join(map(str, missing)) or "any size") + ".\n"
            "Generate them first, e.g.  python datasets/generate_matrices.py --sizes 256 512 1024"
        )
        sys.exit(1)

    compile_programs()
    run_serial(sizes)
    run_openmp(sizes)
    run_cuda_note()
    plot_results()
