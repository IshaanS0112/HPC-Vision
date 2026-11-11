# HPC-Vision

[![CI](https://github.com/IshaanS0112/HPC-Vision/actions/workflows/ci.yml/badge.svg)](https://github.com/IshaanS0112/HPC-Vision/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![C](https://img.shields.io/badge/C-11-A8B9CC.svg?logo=c&logoColor=white)](openmp/)
[![OpenMP](https://img.shields.io/badge/OpenMP-parallel-EE4C2C.svg)](https://www.openmp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-naive%20%7C%20tiled-76B900.svg?logo=nvidia&logoColor=white)](cuda/)
[![OpenCL](https://img.shields.io/badge/OpenCL-1.2-red.svg)](opencl/)

**Dense matrix multiplication benchmarked across four execution models:** serial
C, OpenMP multithreaded C, OpenCL, and CUDA (naive and tiled).

The point of the project is not the matrix multiply — it is measuring what each
parallel programming model actually buys you on the same problem, and where the
speedup stops coming from more threads.

C · OpenMP · OpenCL · CUDA · Python (matplotlib / pandas)

---

## Results

Measured on the same input matrices, single machine. Times in seconds.

| N | Serial CPU | OpenMP (8 threads) | Speedup |
|---:|---:|---:|---:|
| 256 | 0.0487 | 0.0091 | 5.4× |
| 512 | 0.3640 | 0.0575 | 6.3× |
| 1024 | 2.9992 | 0.5802 | 5.2× |
| 1536 | 6.5819 | 1.1911 | 5.5× |
| 2048 | 27.1242 | 5.4395 | 5.0× |

### Thread scaling at N = 1024

| Threads | Time (s) | Speedup | Parallel efficiency |
|---:|---:|---:|---:|
| 1 | 3.2974 | 1.00× | 100% |
| 2 | 1.6514 | 2.00× | 100% |
| 4 | 0.8374 | 3.94× | 98% |
| 8 | 0.5643 | 5.84× | 73% |

Near-linear to 4 threads, then efficiency falls to 73% at 8. The interesting part
of this benchmark is that drop-off, not the headline speedup: the naive triple
loop is memory-bandwidth bound, so past a certain thread count the cores are
waiting on the same memory bus rather than computing. That is also why the
serial→OpenMP speedup sits at ~5× rather than ~8× on an 8-thread run.

### GPU results

CSVs for OpenCL and CUDA are in `results/`. **Read the caveat below before quoting
these numbers.**

---

## Honest status of the measurements

Worth stating plainly, because the numbers are the deliverable:

- **Serial, OpenMP, and thread-scaling timings are measured**, on one machine, from
  the C binaries in this repo. Full precision, single runs (not averaged over
  repetitions — see *Limitations*).
- **`cuda_naive_results.csv` is measured** in Google Colab (`cuda/run_cuda_colab.ipynb`);
  it carries full float precision (e.g. `0.000252896`).
- **`opencl_results.csv` and `cuda_tiled_results.csv` are round to 1–2 significant
  figures** (0.0005, 0.001, 0.006, 0.02, 0.05, 0.15). That is not what a timer
  emits. They should be treated as approximate until re-measured with the same
  instrumentation as the CUDA naive run, and the plots that use them read as
  estimates rather than measurements.

Re-running those two on a real device and committing the raw timer output is the
single highest-value improvement to this project.

`cuda/run_cuda_colab.ipynb` is the instrumentation for the CUDA half of that:
`cudaEvent` timing around the kernel only, one discarded warm-up launch, each
size the mean of 10 launches, and a per-run correctness assertion (both inputs
are all-ones, so every output element must equal `n`). It regenerates both CUDA
CSVs. OpenCL still needs the equivalent harness.

---

## Repository layout

```
openmp/     matrix_serial.c, matrix_openmp.c      serial + OpenMP kernels
opencl/     matrix_opencl.c, matrix_kernel.cl     OpenCL host + device code
cuda/       matrix_cuda_naive.cu,                 CUDA kernels
            matrix_cuda_tiled.cu,
            run_cuda_colab.ipynb                  Colab runner (no local GPU needed)
datasets/   generate_matrices.py                  rebuilds all benchmark inputs
python_modules/                                   plotting + summary table
results/    *.csv, *.png                          timings and charts
main.py                                           serial + OpenMP + plots
run_all.sh                                        regenerate every chart
```

## Running it

The input matrices are **not** in the repository — they are ~1.4 GB of
regenerable text, and the 3072² files are 225 MB each, over GitHub's hard 100 MB
per-file limit. Generate them first:

```bash
python datasets/generate_matrices.py                 # all sizes, ~1.4 GB
python datasets/generate_matrices.py --sizes 256 512  # or just the small ones
```

The generator is seeded, so the matrices are byte-identical on every machine and
a timing taken today is comparable to one taken last month.

```bash
# Serial + OpenMP for every generated size, then plots
python main.py
python main.py 256 512 1024     # or just these sizes

# Charts only, from the committed CSVs
./run_all.sh
```

**OpenMP:** on macOS `main.py` compiles with `-Xpreprocessor -fopenmp` against
Homebrew's `libomp` (`brew install libomp`), because Apple Clang does not ship
OpenMP; on Linux it uses plain `-fopenmp`. The binaries are built into
`openmp/` and run from there, and every run appends a row to `results/*.csv`.

**CUDA:** run `cuda/run_cuda_colab.ipynb` in Google Colab — no local NVIDIA GPU
required.

---

## What each implementation does

- **Serial** — the baseline triple loop. Every speedup number is relative to this.
- **OpenMP** — `#pragma omp parallel for` over the outer loop, so each thread owns a
  contiguous band of output rows and no two threads write the same cache line.
- **OpenCL** — one work-item per output element, portable across GPU vendors and CPUs.
- **CUDA naive** — one thread per output element; every thread reads a full row and
  a full column straight from global memory.
- **CUDA tiled** — the same computation staged through `__shared__` memory in tiles,
  so each loaded element is reused by every thread in the block instead of being
  fetched from global memory once per thread. This is the optimisation that
  matters on a GPU, and comparing it against the naive kernel is the point of
  having both.

---

## Limitations

- **Single-run timings, not averaged.** Each figure is one execution, so it carries
  run-to-run variance (cache state, frequency scaling, other load). Reporting the
  median of 5 runs with a spread would be the correct form.
- **OpenCL and CUDA-tiled numbers are rounded** — see *Honest status* above.
- **CPU and GPU timings come from different machines** (local vs Colab), so the
  cross-architecture chart compares hardware as much as programming model.
- **GPU timings exclude host↔device transfer.** Kernel time alone flatters the GPU:
  for a single multiply, PCIe transfer often dominates. Timing the round trip
  would be the honest end-to-end comparison.
- **No correctness check against a reference.** The kernels are not verified against
  a known-good result (e.g. NumPy) — a fast kernel that computes the wrong answer
  would pass this benchmark silently. This is the first thing worth adding.
- **`np.savetxt` is a wasteful input format:** a 1024² float matrix is 25 MB as text
  versus ~8 MB binary, and parsing dominates small-N runs. Kept because the C
  readers parse text.

## Next

Verify every kernel against a NumPy reference · median-of-N timings with spread ·
re-measure OpenCL and CUDA tiled with proper instrumentation · include host↔device
transfer in GPU timings · try a cache-blocked serial version, which would likely
close much of the gap the OpenMP run is credited with.

## Licence

MIT — see [LICENSE](LICENSE).
