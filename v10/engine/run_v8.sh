#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ "$#" -ne 4 ]; then
  echo 'Usage: bash run_v8.sh /data/features /data/emotion/plan.json /data/psqi/plan.json /results/NEW_v8_run'
  exit 2
fi
python -B verify_package.py
python -B build_improved.py --features "$1" --plans "$2" "$3" --output "$4" \
  --devices cuda:0 cuda:1 --batch-size 16 --threads 2 --allow-unreviewed
