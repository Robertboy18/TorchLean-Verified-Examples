#!/usr/bin/env bash
set -euo pipefail

# Run both implementations sequentially so they see the same physical GPU.
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
week="$repo/week-03-gpt-training"
output="${1:-$week/results/matched-runtime-50-step}"
checkpoint="${2:-$week/artifacts/gpt2-small-2.5b-continuation-20260806/final-parameters.tlf32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
train_bin="$week/data/smol-smoltalk-10m/train.bin"
train_mask="$week/data/smol-smoltalk-10m/train.mask"
train_records="$week/data/smol-smoltalk-10m/train.records"
val_bin="$week/data/smol-smoltalk-10m/val.bin"
val_mask="$week/data/smol-smoltalk-10m/val.mask"
val_records="$week/data/smol-smoltalk-10m/val.records"

cd "$repo"
mkdir -p "$output"
lake -R -K cuda=true build train_torchlean_gpt

./.lake/build/bin/train_torchlean_gpt --device cuda \
  --preset gpt2-small \
  --train-bin "$train_bin" --train-mask "$train_mask" --train-records "$train_records" \
  --val-bin "$val_bin" --val-mask "$val_mask" --val-records "$val_records" \
  --load-params "$checkpoint" \
  --batch 6 --steps 50 --seed 0 \
  --lr 0.00003 --min-lr 0.000003 --warmup-steps 10 --weight-decay 0 \
  --eval-every 10 --eval-batches 2 \
  --metrics "$output/torchlean-metrics.json" \
  --passport "$output/torchlean-passport.json"

python "$week/tools/matched_pytorch_run.py" \
  --checkpoint "$checkpoint" \
  --train-bin "$train_bin" --train-mask "$train_mask" --train-records "$train_records" \
  --val-bin "$val_bin" --val-mask "$val_mask" --val-records "$val_records" \
  --device cuda:0 --batch 6 --steps 50 --seed 0 \
  --lr 0.00003 --min-lr 0.000003 --warmup-steps 10 --weight-decay 0 \
  --eval-every 10 --eval-batches 2 \
  --output "$output/pytorch-metrics.json"

python "$week/tools/compare_matched_runs.py" \
  --torchlean "$output/torchlean-metrics.json" \
  --pytorch "$output/pytorch-metrics.json" \
  --tokens-per-update 6144 \
  --output "$output"
