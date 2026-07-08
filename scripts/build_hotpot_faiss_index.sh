#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${1:-.venv-rag/bin/python}"

"$PYTHON_BIN" build_rag_index.py \
  --chunks-jsonl data/indexes/hotpotqa_dev_bge_base/chunks.jsonl \
  --output-dir data/indexes/hotpotqa_dev_bge_base \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --batch-size 16 \
  --local-files-only
