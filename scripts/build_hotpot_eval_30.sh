#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${1:-.venv-rag/bin/python}"

"$PYTHON_BIN" build_hotpot_stratified_eval.py \
  --source data/raw/hotpot/hotpot_dev_fullwiki_v1.json \
  --output data/eval/hotpot_stratified_30.json \
  --sample-size 30 \
  --seed 7
