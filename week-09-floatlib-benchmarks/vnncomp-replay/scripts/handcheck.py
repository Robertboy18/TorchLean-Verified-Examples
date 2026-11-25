"""A few refuted unsat results checked independently of the replay's readers.

    python3 -I scripts/handcheck.py FAMILY INDEX...        (FAMILY: safenlp_2024 or soundnessbench)

For the INDEX-th refuted unsat result of `RUN/results/refuted-FAMILY.csv` this takes the witness the replay
found, reads the VNNLIB file again with regular expressions (not with scripts/common.py), tests the
box with `fractions.Fraction` at the binary32 point, runs the original ONNX file with onnxruntime at
that point, and decides the output property on onnxruntime's outputs with fractions. Nothing here
feeds a result.
"""
import csv
import gzip
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

SUITE = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get('RUN', SUITE / 'run')).resolve()      # downloads, translated inputs, raw results
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUN / 'py'))       # an optional `pip --target` directory with onnxruntime
import onnxruntime as ort  # noqa: E402

ort.set_default_logger_severity(3)
BENCH = RUN / 'dl' / 'vnncomp2025_benchmarks' / 'benchmarks'


def main() -> None:
    family = sys.argv[1]
    refuted = [r for r in csv.DictReader(open(RUN / 'results' / f'refuted-{family}.csv')) if r['refuted'] == '1']
    files = list(csv.DictReader(open(RUN / 'results' / f'files-{family}.csv')))
    replay = {r['id']: r for r in csv.DictReader(open(RUN / 'results' / f'replay-{family}.tsv'), delimiter='\t')}
    for index in sys.argv[2:]:
        ref = refuted[int(index)]
        witness = next(f for f in files if f['net'] == ref['net'] and f['prop'] == ref['prop']
                       and f['class'] in ('exact', 'projected'))
        row = replay[witness['best_point']]
        prefix = '' if witness['class'] == 'exact' else 'proj_'
        x = np.array([int(t, 16) for t in row[prefix + 'x_bits'].split(',')], dtype=np.uint32).view(np.float32)
        if family == 'safenlp_2024':
            folder, name = ref['prop'].split('_', 1)
            vnnlib = BENCH / family / 'vnnlib' / folder / f'{name}.vnnlib.gz'
            onnx = BENCH / family / 'onnx' / folder / 'perturbations_0.onnx.gz'
        else:
            vnnlib = BENCH / family / 'vnnlib' / f'{ref["prop"]}.vnnlib.gz'
            onnx = BENCH / family / 'onnx' / f'{ref["net"]}.onnx.gz'
        text = gzip.open(vnnlib, 'rt').read()
        lower = {int(i): Fraction(v) for i, v in re.findall(r'\(assert \(>= X_(\d+) ([^\s)]+)\)\)', text)}
        upper = {int(i): Fraction(v) for i, v in re.findall(r'\(assert \(<= X_(\d+) ([^\s)]+)\)\)', text)}
        assert len(lower) == len(upper) == x.size
        inside = all(lower[i] <= Fraction(float(v)) <= upper[i] for i, v in enumerate(x))
        runtime = ort.InferenceSession(gzip.open(onnx, 'rb').read(), providers=['CPUExecutionProvider'])
        inp = runtime.get_inputs()[0]
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        y = [Fraction(float(v)) for v in runtime.run(None, {inp.name: x.reshape(shape)})[0].reshape(-1)]

        def value(term: str) -> Fraction:
            return y[int(term[2:])] if term.startswith('Y_') else Fraction(term)

        def holds(atoms) -> bool:
            return all(value(a) <= value(b) if op == '<=' else value(a) >= value(b) for op, a, b in atoms)

        atom = r'\(([<>]=) (Y_\d+|[^\s()]+) (Y_\d+|[^\s()]+)\)'
        plain = r'\((?:[<>]=) (?:Y_\d+|[^\s()]+) (?:Y_\d+|[^\s()]+)\)'
        groups = re.findall(r'\(and((?:\s*' + plain + r')+)\s*\)', text)
        if groups:
            verdicts = [holds(re.findall(atom, group)) for group in groups]
            violated = any(verdicts)
            shape_text = f'{len(groups)} disjuncts, {sum(verdicts)} of them hold'
        else:
            atoms = [a for a in re.findall(atom, text) if 'Y_' in a[1] or 'Y_' in a[2]]
            violated = holds(atoms)
            shape_text = f'{len(atoms)} output atoms'
        print(f'{family} {ref["net"]} {ref["prop"]}: unsat by {ref["unsat_by"]}; witness of {witness["tool"]} '
              f'({witness["class"]}); {x.size} inputs inside the box: {inside}; property on onnxruntime\'s output '
              f'({shape_text}) violated: {violated}; replay: violated with slack {witness["out32_slack"]}')


if __name__ == '__main__':
    main()
