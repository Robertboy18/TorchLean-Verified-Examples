#!/bin/bash
# spotcheck_all.sh: the independent exact-arithmetic check (spotcheck.py) on every replayed
# family: all cases of the small families, a seeded sample of the others.
set -u
here=$(cd "$(dirname "$0")" && pwd); suite=$(dirname "$here"); week=$(dirname "$suite")
run=$(realpath -m "${RUN:-$suite/run}")          # downloads (dl/), translated inputs (work/), raw results (results/)
replay=${REPLAY:-$week/.lake/build/bin/replay}       # `lake build replay` in the week's Lake project
cd "$run" || exit 2
for pair in acasxu_2023:0 test:0 sat_relu:0 safenlp_2024:600 collins_rul_cnn_2022:30 tllverifybench_2023:12 \
            cora_2024:20 malbeware:6 relusplitter:6 soundnessbench:3; do
  family=${pair%%:*}; count=${pair##*:}
  python3 -I "$here/spotcheck.py" $family work results/replay-$family.tsv results/spotcheck-$family.tsv $count \
    || echo "$family: spot check failed"
done
