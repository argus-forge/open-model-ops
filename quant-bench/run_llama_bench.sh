#!/usr/bin/env bash
# Benchmark every GGUF in a directory with llama-bench, on GPU (all layers offloaded)
# and CPU only. Reports prompt throughput (pp512) and generation throughput (tg128).
#
# Usage:
#   ./run_llama_bench.sh /path/to/llama.cpp /path/to/gguf_dir [cpu_threads]
#
# This is the script that reproduces the throughput tables in the repo README.
# Run it on whatever hardware you want to measure (a rented H100, a local 3090, CPU).
set -euo pipefail

LLAMA_DIR="${1:?usage: run_llama_bench.sh <llama.cpp dir> <gguf dir> [threads]}"
GGUF_DIR="${2:?missing dir of gguf files}"
THREADS="${3:-8}"
BENCH="${LLAMA_DIR}/llama-bench"

for M in "${GGUF_DIR}"/*.gguf; do
  echo "=== ${M} :: GPU, all layers (-ngl 99) ==="
  "${BENCH}" -m "${M}" -ngl 99 -p 512 -n 128
  echo "=== ${M} :: CPU only (-ngl 0, ${THREADS} threads) ==="
  "${BENCH}" -m "${M}" -ngl 0 -t "${THREADS}" -p 512 -n 128
done
