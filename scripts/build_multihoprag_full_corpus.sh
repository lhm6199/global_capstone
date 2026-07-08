#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${1:-.venv-rag/bin/python}"

"$PYTHON_BIN" build_multihoprag_corpus.py \
  --source data/raw/multihoprag/corpus.json \
  --output data/indexes/multihoprag_bge_base/chunks.jsonl
