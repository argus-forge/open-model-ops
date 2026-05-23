#!/usr/bin/env bash
# Measure perplexity across a GGUF quant ladder on a public text (wikitext-2).
# Pairs with run_llama_bench.sh: throughput on one axis, quality on the other.
# Lower perplexity is better; the gap from q8_0 down to q4_K_M is the real
# quality cost of quantization.
#
# Usage:
#   ./run_perplexity.sh /path/to/llama.cpp/build/bin /path/to/gguf_dir /path/to/wiki.test.raw
set -euo pipefail

BIN_DIR="${1:?usage: run_perplexity.sh <llama.cpp bin dir> <gguf dir> <wiki.test.raw>}"
GGUF_DIR="${2:?missing dir of gguf files}"
WIKI="${3:?missing path to wiki.test.raw}"
PPL="${BIN_DIR}/llama-perplexity"

for M in "${GGUF_DIR}"/*.gguf; do
  echo "=== ${M} ==="
  "${PPL}" -m "${M}" -f "${WIKI}" -ngl 99 2>&1 | tail -4
done
