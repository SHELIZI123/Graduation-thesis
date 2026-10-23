#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
OUT="${1:-smoke_v173}"
export PYTHONDONTWRITEBYTECODE=1 TABPFN_DISABLE_TELEMETRY=1
python -B code/check_package_v173.py --foundation
python -B code/test_v172.py
python -B code/test_v173.py
python -u -B code/train_v173.py --data data --extras extras --output "$OUT" --device "${DEVICE:-cuda:0}" \
  --folds 0 --seeds 20261023 --profile smoke
python -u -B code/evaluate_v173.py --data data --extras extras --runs "$OUT" --output "${OUT}_evaluation" \
  --replay --allow-partial --bootstrap 2000 --device "${DEVICE:-cuda:0}"
