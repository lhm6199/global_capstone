#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${1:-.venv-rag/bin/python}"
LOAD_QUANT="${2:-model/qwen3-4b-w4-g128-awq-v2.pt}"
MODEL_PATH="${3:-qwen3-4b-awq-runtime}"

"$PYTHON_BIN" compare_rag_backends.py \
  --dataset hotpotqa \
  --eval-file data/eval/hotpot_stratified_30.json \
  --rag-index-dir data/indexes/hotpotqa_dev_bge_base \
  --minirag-working-dir data/indexes/minirag_hotpotqa_dev_bge_base \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --backends faiss_naive minirag_naive minirag_light minirag_mini \
  --output outputs/hotpotqa_backend_compare.json \
  --summary-output outputs/hotpotqa_backend_compare_summary.md \
  --model_path "$MODEL_PATH" \
  --load_quant "$LOAD_QUANT" \
  --awq_backend auto \
  --local-files-only
