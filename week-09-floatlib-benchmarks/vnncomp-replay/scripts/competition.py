"""What the competition's own scoring said about every counterexample file and every result.

    python3 -I scripts/competition.py RESULTS_REPO OUT_DIR FAMILY...

Reads `RESULTS_REPO/SCORING-SMALL-TOL/results.txt` and `RESULTS_REPO/SCORING-ZERO-TOL/results.txt`
(the logs of the competition's `process_results.py`, which checks counterexamples with
onnxruntime and tolerances, `counterexamples.py`) as text and writes

    OUT_DIR/competition-ce-FAMILY.tsv      scoring, tool, net, prop, checker verdict, the reported
                                           difference between onnxruntime and the file's output
    OUT_DIR/competition-score-FAMILY.tsv   scoring, instance, the result the scoring took as true,
                                           tool, score (10 correct, 0 none, -150 penalty)

Nothing is evaluated here.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

CE_PATH = re.compile(r'^Checking ce path: \.\./([a-z_]+)/2025_([A-Za-z0-9_]+)/(.+)\.counterexample\.gz, ')
CE_RESULT = re.compile(r'^CE result ([a-z_]+): (.*)$')
DIFF = re.compile(r'onnx execution and CE file output: ([^ ]+) \(rel error: ([^)]+)\)')
ROW = re.compile(r"^Row: \['([^']+)'")
TRUE = re.compile(r'^True Result: (\S+)')
SCORE = re.compile(r'^(\d+): ([a-z_]+) score: (-?\d+),')
CATEGORY = re.compile(r'^Category 2025_([A-Za-z0-9_]+):\s*$')


def main() -> None:
    repo, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    families = sys.argv[3:]
    ce_rows = {f: [] for f in families}
    score_rows = {f: [] for f in families}
    for scoring in ('SCORING-SMALL-TOL', 'SCORING-ZERO-TOL'):
        label = 'small' if 'SMALL' in scoring else 'zero'
        category = None
        pending = None
        instance = true_result = None
        seen_categories = set()
        with (repo / scoring / 'results.txt').open(encoding='utf-8', errors='replace') as stream:
            for line in stream:
                line = line.rstrip('\n')
                match = CATEGORY.match(line)
                if match:
                    # the first block of a category is the scoring log; later ones are summaries
                    category = match.group(1) if match.group(1) not in seen_categories else None
                    seen_categories.add(match.group(1))
                    continue
                match = CE_PATH.match(line)
                if match:
                    pending = match.groups()
                    continue
                match = CE_RESULT.match(line)
                if match and pending:
                    tool, family, name = pending
                    pending = None
                    if family in ce_rows:
                        diff = DIFF.search(match.group(2))
                        ce_rows[family].append([label, tool, name, match.group(1),
                                                diff.group(1) if diff else '', diff.group(2) if diff else ''])
                    continue
                if category not in score_rows:
                    continue
                match = ROW.match(line)
                if match:
                    instance = match.group(1)
                    continue
                match = TRUE.match(line)
                if match:
                    true_result = match.group(1)
                    continue
                match = SCORE.match(line)
                if match and instance is not None:
                    score_rows[category].append([label, match.group(1), instance, true_result,
                                                 match.group(2), match.group(3)])
    for family in families:
        with (out_dir / f'competition-ce-{family}.tsv').open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
            writer.writerow(['scoring', 'tool', 'file', 'checker', 'abs_difference', 'rel_difference'])
            writer.writerows(ce_rows[family])
        with (out_dir / f'competition-score-{family}.tsv').open('w', newline='') as stream:
            writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
            writer.writerow(['scoring', 'index', 'instance', 'true_result', 'tool', 'score'])
            writer.writerows(score_rows[family])
        print(f'{family}: {len(ce_rows[family])} checker verdicts, {len(score_rows[family])} scores')


if __name__ == '__main__':
    main()
