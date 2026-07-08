#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${1:-.venv-rag/bin/python}"
LOAD_QUANT="${2:-model/qwen3-4b-w4-g128-awq-v2.pt}"
MODEL_PATH="${3:-qwen3-4b-awq-runtime}"

"$PYTHON_BIN" build_minirag_index.py \
  --chunks-jsonl data/indexes/multihoprag_bge_base/chunks.jsonl \
  --working-dir data/indexes/minirag_multihoprag_bge_base \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --embedding-batch-size 16 \
  --rag-top-k 3 \
  --model_path "$MODEL_PATH" \
  --load_quant "$LOAD_QUANT" \
  --awq_backend auto \
  --local-files-only
