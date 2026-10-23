#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
OUT="${1:-server_v173}"
THREADS="${THREADS:-2}"
FOUNDATION="${FOUNDATION:-1}"
read -r -a SEED_ARRAY <<< "${SEEDS:-20261023 20261024 20261025}"
export PYTHONDONTWRITEBYTECODE=1 TABPFN_DISABLE_TELEMETRY=1
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Two visible GPUs required"'
EXTRA=()
if [[ "$FOUNDATION" == 1 ]]; then
  python -B code/check_package_v173.py --foundation
else
  python -B code/check_package_v173.py
  EXTRA+=(--without-foundation)
fi
python -B code/test_v172.py
python -B code/test_v173.py
if [[ "${RESUME:-0}" == 1 ]]; then
  [[ -d "$OUT" ]] || { echo "Resume directory missing" >&2; exit 1; }
  EXTRA+=(--resume)
else
  [[ ! -e "$OUT" ]] || { echo "Output already exists; choose a new name or RESUME=1" >&2; exit 1; }
  mkdir -p -- "$OUT"
fi
python -m pip freeze > "$OUT/environment.freeze.txt"
python -c 'from pathlib import Path; import sys,json; sys.path.insert(0,"code"); from foundation_v173 import runtime; Path(sys.argv[1]).write_text(json.dumps(runtime(),indent=2),encoding="utf-8")' "$OUT/runtime.json"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT
CUDA_VISIBLE_DEVICES=0 python -u -B code/train_v173.py --data data --extras extras --output "$OUT/gpu0" --device cuda:0 \
  --folds 0 2 4 --threads "$THREADS" --seeds "${SEED_ARRAY[@]}" "${EXTRA[@]}" > "$OUT/gpu0.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=1 python -u -B code/train_v173.py --data data --extras extras --output "$OUT/gpu1" --device cuda:0 \
  --folds 1 3 --threads "$THREADS" --seeds "${SEED_ARRAY[@]}" "${EXTRA[@]}" > "$OUT/gpu1.log" 2>&1 &
pids+=("$!")
set +e
wait "${pids[0]}"; status0=$?
wait "${pids[1]}"; status1=$?
set -e
[[ $status0 -eq 0 && $status1 -eq 0 ]] || { echo "Worker failed; inspect logs. Completed folds retained." >&2; exit 1; }
python -u -B code/evaluate_v173.py --data data --extras extras --runs "$OUT/gpu0" "$OUT/gpu1" \
  --output "${REPORT:-$OUT/evaluation}" --replay --device cuda:0 --threads "$THREADS"
if [[ "${REFIT:-1}" == 1 ]]; then
  python -u -B code/refit_v173.py --data data --extras extras --output "${REFIT_OUT:-$OUT/final_models}" \
    --device cuda:0 --threads "$THREADS" --seeds "${SEED_ARRAY[@]}"
fi
echo "Completed: ${REPORT:-$OUT/evaluation}/结果说明.md"
