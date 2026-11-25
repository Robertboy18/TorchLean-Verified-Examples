"""What onnxruntime, the runtime of the competition's own checker, computes at the replayed points.

    python3 -I scripts/ort_compare.py FAMILY BENCHMARKS_DIR WORK_DIR REPLAY_TSV OUT_TSV

For every line of REPLAY_TSV this runs the original ONNX file with onnxruntime (CPU provider, one
thread, default graph optimizations as in the competition's `counterexamples.py`, and once more
with all graph optimizations disabled) on the binary32 input words that the Lean replay used, at
the reported point and at the point moved into the box. It writes how many output words differ
from FloatLib's reference-order result, the largest difference, and whether the output part of
the property holds at onnxruntime's output, decided on exact rationals (`fractions.Fraction`),
and the time of one evaluation (`ort_us`: the median of three calls of `InferenceSession.run` after
the one whose output is compared, in microseconds, one thread).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

SUITE = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get('RUN', SUITE / 'run')).resolve()      # downloads, translated inputs, raw results
sys.dont_write_bytecode = True
sys.path.insert(0, str(RUN / 'py'))       # an optional `pip --target` directory with onnxruntime
sys.path.insert(0, str(Path(__file__).resolve().parent))
import onnxruntime as ort  # noqa: E402
ort.set_default_logger_severity(3)      # errors only
import common  # noqa: E402


def parse_formula(tokens, position=0):
    head = tokens[position]
    if head in ('and', 'or'):
        count = int(tokens[position + 1])
        position += 2
        children = []
        for _ in range(count):
            child, position = parse_formula(tokens, position)
            children.append(child)
        return (head, children), position
    a = (tokens[position + 1], tokens[position + 2])
    b = (tokens[position + 3], tokens[position + 4])
    return ((head, a, b) if head == 'le' else ('le', b, a)), position + 5


def slack(formula, xs, ys, use_in: bool, use_out: bool):
    """None is plus infinity; the formula holds iff the slack is not negative."""
    head = formula[0]
    if head == 'le':
        a, b = formula[1], formula[2]
        is_input = a[0] != 'Y' and b[0] != 'Y'
        if (is_input and not use_in) or (not is_input and not use_out):
            return None

        def value(t):
            if t[0] == 'X':
                return xs[int(t[1])]
            if t[0] == 'Y':
                return ys[int(t[1])]
            return Fraction(t[1])
        return value(b) - value(a)
    parts = [slack(child, xs, ys, use_in, use_out) for child in formula[1]]
    if head == 'and':
        finite = [p for p in parts if p is not None]
        return min(finite) if finite else None
    if any(p is None for p in parts):
        return None
    return max(parts)


def words(text: str) -> np.ndarray:
    return np.array([int(t, 16) for t in text.split(',')], dtype=np.uint32)


def main() -> None:
    family, benchmarks, work, replay_path, out_path = sys.argv[1:6]
    specs, nets = {}, {}
    with open(Path(work) / 'cases' / f'{family}.cases') as stream:
        block = {}
        for line in stream:
            key, _, rest = line.rstrip('\n').partition(' ')
            if key == 'end':
                specs[block['case']] = block['spec']
            else:
                block[key] = rest
    loaded = {}
    onnx_files = common.net_files(Path(benchmarks) / family)

    def loaded_net(net: str, optimized: bool):
        if (net, optimized) not in loaded:
            if len(loaded) >= 4:          # keep at most two networks loaded
                loaded.clear()
            data = common.read_bytes(onnx_files[net])
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            if not optimized:
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            sess = ort.InferenceSession(data, options, providers=['CPUExecutionProvider'])
            inp = sess.get_inputs()[0]
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            loaded[(net, optimized)] = (sess, inp.name, shape)
        return loaded[(net, optimized)]

    fields = ['id', 'point', 'optimized', 'outputs', 'differing_words', 'max_abs_difference', 'out_ort',
              'out_ort_slack', 'out_floatlib', 'verdict_agrees', 'ort_us']
    counts = {}
    with open(replay_path) as stream, open(out_path, 'w', newline='') as out:
        writer = csv.writer(out, delimiter='\t', lineterminator='\n')
        writer.writerow(fields)
        for row in csv.DictReader(stream, delimiter='\t'):
            if row['status'] != 'ok':
                continue
            net = row['id'].split('|')[2]
            formula, _ = parse_formula(specs[row['id']].split())
            variants = [('reported', '')]
            if row['moved'] != '0':
                variants.append(('projected', 'proj_'))
            for which, prefix in variants:
                x = words(row[prefix + 'x_bits']).view(np.float32)
                reference = words(row[prefix + 'y_bits']).view(np.float32)
                xs = [Fraction(float(v)) for v in x]
                for optimized in (True, False):
                    sess, name, shape = loaded_net(net, optimized)
                    feed = {name: x.reshape(shape)}
                    y = sess.run(None, feed)[0].reshape(-1).astype(np.float32)
                    laps = []
                    for _ in range(3):
                        start = time.perf_counter()
                        sess.run(None, feed)
                        laps.append(time.perf_counter() - start)
                    micros = sorted(laps)[1] * 1e6
                    differing = int(np.sum(y != reference))
                    worst = float(np.max(np.abs(y.astype(np.float64) - reference.astype(np.float64))))
                    s = slack(formula, xs, [Fraction(float(v)) for v in y], False, True)
                    holds = s is None or s >= 0
                    agrees = int(holds == (row[prefix + 'out32'] == '1'))
                    writer.writerow([row['id'], which, int(optimized), y.size, differing, f'{worst:.6e}',
                                     int(holds), 'inf' if s is None else f'{float(s):.9e}',
                                     row[prefix + 'out32'], agrees, f'{micros:.1f}'])
                    key = (which, optimized)
                    c = counts.setdefault(key, [0, 0, 0, 0.0])
                    c[0] += 1
                    c[1] += differing == 0
                    c[2] += agrees
                    c[3] = max(c[3], worst)
    for (which, optimized), (total, same, agree, worst) in sorted(counts.items()):
        print(f'{family} {which} optimized={optimized}: {total} points, all output words equal in {same}, '
              f'verdict agrees in {agree}, largest difference {worst:.3e}')


if __name__ == '__main__':
    main()
