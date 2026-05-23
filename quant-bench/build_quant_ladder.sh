#!/usr/bin/env bash
# Build a GGUF quant ladder from an f16 base GGUF using llama.cpp.
#
# Usage:
#   ./build_quant_ladder.sh /path/to/llama.cpp /path/to/model-f16.gguf /path/to/out_dir
#
# Produces model-q8_0.gguf, model-q6_K.gguf, model-q5_K_M.gguf, model-q4_K_M.gguf.
set -euo pipefail

LLAMA_DIR="${1:?usage: build_quant_ladder.sh <llama.cpp dir> <f16 gguf> <out dir>}"
BASE="${2:?missing path to f16 base gguf}"
OUT="${3:?missing output dir}"
QUANT_BIN="${LLAMA_DIR}/llama-quantize"

mkdir -p "${OUT}"
for Q in q8_0 q6_K q5_K_M q4_K_M; do
  echo "[*] quantizing ${Q}"
  "${QUANT_BIN}" "${BASE}" "${OUT}/model-${Q}.gguf" "${Q}"
done

echo "[*] ladder built in ${OUT}"
ls -lh "${OUT}"
