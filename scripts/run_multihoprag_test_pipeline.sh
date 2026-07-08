#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${1:-.venv-rag/bin/python}"
LOAD_QUANT="${2:-model/qwen3-4b-w4-g128-awq-v2.pt}"
MODEL_PATH="${3:-qwen3-4b-awq-runtime}"

bash scripts/build_multihoprag_eval_full.sh "$PYTHON_BIN"
bash scripts/build_multihoprag_test_30.sh "$PYTHON_BIN"
bash scripts/build_multihoprag_full_corpus.sh "$PYTHON_BIN"
bash scripts/build_multihoprag_faiss_index.sh "$PYTHON_BIN"
bash scripts/build_multihoprag_minirag_index.sh "$PYTHON_BIN" "$LOAD_QUANT" "$MODEL_PATH"
bash scripts/run_multihoprag_compare.sh "$PYTHON_BIN" "$LOAD_QUANT" "$MODEL_PATH"
