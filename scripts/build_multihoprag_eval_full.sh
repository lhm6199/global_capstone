#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${1:-.venv-rag/bin/python}"

"$PYTHON_BIN" build_multihoprag_eval.py \
  --source data/raw/multihoprag/MultiHopRAG.json \
  --output data/eval/multihoprag_eval_full.json
