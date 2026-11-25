"""Choose the counterexamples that Lean's kernel replays for a family: one per instance.

    python3 -I scripts/witness_ids.py RESULTS_DIR FAMILY OUT_IDS

For every instance of `RESULTS_DIR/files-FAMILY.csv` with an exact counterexample this writes the
case id of one witness: a file of class `exact` if there is one and otherwise of class `projected`,
with the fewest points, from the first verifier in the fixed order of the tables. The choice does
not look at any margin.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.dont_write_bytecode = True
TOOLS = ['alpha_beta_crown', 'cora', 'neuralsat', 'nnenum', 'nnv', 'pyrat', 'rover', 'sobolbox']
RANK = {'exact': 0, 'projected': 1}


def main() -> None:
    results, family, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    best = {}
    with (results / f'files-{family}.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['class'] not in RANK:
                continue
            key = (row['net'], row['prop'])
            rank = (RANK[row['class']], int(row['points']), TOOLS.index(row['tool']))
            if key not in best or rank < best[key][0]:
                best[key] = (rank, row['best_point'])
    with open(out, 'w') as stream:
        for key in sorted(best):
            stream.write(best[key][1] + '\n')
    print(f'{family}: {len(best)} witnesses')


if __name__ == '__main__':
    main()
