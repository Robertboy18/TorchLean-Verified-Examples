#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 BASE_PARAMETERS [RUN_NAME]" >&2
  exit 2
fi

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
week="$repo/week-03-gpt-training"
base="$1"
run_name="${2:-sft-reproduction}"
stage1="$week/artifacts/${run_name}-smoltalk-10m"
stage2="$week/artifacts/${run_name}-smoltalk2-50m"

required_files=(
  "$base"
  "$week/data/smol-smoltalk-10m/train.bin"
  "$week/data/smol-smoltalk-10m/val.bin"
  "$week/data/smol-smoltalk-10m/train.mask"
  "$week/data/smol-smoltalk-10m/val.mask"
  "$week/data/smol-smoltalk-10m/train.records"
  "$week/data/smol-smoltalk-10m/val.records"
  "$week/data/smoltalk2-balanced-50m/train.bin"
  "$week/data/smoltalk2-balanced-50m/val.bin"
  "$week/data/smoltalk2-balanced-50m/train.mask"
  "$week/data/smoltalk2-balanced-50m/val.mask"
  "$week/data/smoltalk2-balanced-50m/train.records"
  "$week/data/smoltalk2-balanced-50m/val.records"
)

for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
done

select_best_checkpoint() {
  python3 - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    points = json.load(handle)["points"]

evaluations = [
    point for point in points
    if point["phase"] == "validation" and point["step"] > 0
]
if not evaluations:
    raise SystemExit("training produced no post-initialization validation point")

best = min(evaluations, key=lambda point: point["loss"])
print(best["step"], best["loss"])
PY
}

write_selection() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import json
import sys

output, step, loss, checkpoint = sys.argv[1:]
with open(output, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "schema": "torchlean.gpt-training.selected-checkpoint.v1",
            "selection": "lowest post-initialization sampled validation loss",
            "step": int(step),
            "loss": float(loss),
            "checkpoint": checkpoint,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
PY
}

mkdir -p "$stage1/checkpoints" "$stage2/checkpoints"
cd "$repo"

lake -R -K cuda=true build train_torchlean_gpt

echo "Stage 1: 10M scheduled tokens from dialogue-bounded SmolTalk records"
LEAN_PROFILE=1 lake -R -K cuda=true exe train_torchlean_gpt --device cuda \
  --preset gpt2-small \
  --train-bin "$week/data/smol-smoltalk-10m/train.bin" \
  --val-bin "$week/data/smol-smoltalk-10m/val.bin" \
  --train-mask "$week/data/smol-smoltalk-10m/train.mask" \
  --val-mask "$week/data/smol-smoltalk-10m/val.mask" \
  --train-records "$week/data/smol-smoltalk-10m/train.records" \
  --val-records "$week/data/smol-smoltalk-10m/val.records" \
  --load-params "$base" \
  --batch 6 \
  --token-budget 10000000 \
  --lr 0.00003 \
  --min-lr 0.000003 \
  --warmup-steps 100 \
  --weight-decay 0 \
  --eval-every 250 \
  --eval-batches 8 \
  --checkpoint-dir "$stage1/checkpoints" \
  --checkpoint-every 250 \
  --save-params "$stage1/final-parameters.tlf32" \
  --metrics "$stage1/training-metrics.json" \
  --passport "$stage1/run-passport.json"

read -r best1_step best1_loss < <(select_best_checkpoint "$stage1/training-metrics.json")
best1="$stage1/checkpoints/step-$best1_step/parameters.tlf32"
if [[ ! -f "$best1" ]]; then
  echo "best Stage 1 evaluation has no checkpoint: step $best1_step" >&2
  exit 1
fi
write_selection "$stage1/selected-checkpoint.json" "$best1_step" "$best1_loss" "$best1"

echo "Stage 2: 50M scheduled tokens from balanced dialogue-bounded SmolTalk2 records"
LEAN_PROFILE=1 lake -R -K cuda=true exe train_torchlean_gpt --device cuda \
  --preset gpt2-small \
  --train-bin "$week/data/smoltalk2-balanced-50m/train.bin" \
  --val-bin "$week/data/smoltalk2-balanced-50m/val.bin" \
  --train-mask "$week/data/smoltalk2-balanced-50m/train.mask" \
  --val-mask "$week/data/smoltalk2-balanced-50m/val.mask" \
  --train-records "$week/data/smoltalk2-balanced-50m/train.records" \
  --val-records "$week/data/smoltalk2-balanced-50m/val.records" \
  --load-params "$best1" \
  --batch 6 \
  --token-budget 50000000 \
  --lr 0.00002 \
  --min-lr 0.000002 \
  --warmup-steps 250 \
  --weight-decay 0 \
  --eval-every 500 \
  --eval-batches 8 \
  --checkpoint-dir "$stage2/checkpoints" \
  --checkpoint-every 500 \
  --save-params "$stage2/final-parameters.tlf32" \
  --metrics "$stage2/training-metrics.json" \
  --passport "$stage2/run-passport.json"

read -r best2_step best2_loss < <(select_best_checkpoint "$stage2/training-metrics.json")
best2="$stage2/checkpoints/step-$best2_step/parameters.tlf32"
if [[ ! -f "$best2" ]]; then
  echo "best Stage 2 evaluation has no checkpoint: step $best2_step" >&2
  exit 1
fi
write_selection "$stage2/selected-checkpoint.json" "$best2_step" "$best2_loss" "$best2"

echo "Stage 1 best: step $best1_step, validation loss $best1_loss"
echo "Stage 2 best: step $best2_step, validation loss $best2_loss"
echo "Selected parameters: $best2"
