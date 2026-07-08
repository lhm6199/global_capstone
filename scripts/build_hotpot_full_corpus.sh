#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${1:-.venv-rag/bin/python}"

"$PYTHON_BIN" build_hotpot_corpus.py \
  --source data/raw/hotpot/hotpot_dev_fullwiki_v1.json \
  --output data/indexes/hotpotqa_dev_bge_base/chunks.jsonl
