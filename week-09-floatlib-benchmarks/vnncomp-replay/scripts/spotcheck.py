"""An independent check of the exact part of the Lean replay on a sample of cases.

    python3 -I scripts/spotcheck.py FAMILY WORK_DIR REPLAY_TSV OUT_TSV COUNT [SEED]

For COUNT cases of REPLAY_TSV chosen with a seeded shuffle (all when COUNT is 0) this evaluates the
`.net` network in exact arithmetic with Python integers (a value is `m * 2 ** e`; no rounding
anywhere) at the binary32 input words the Lean replay used, at the reported point and at the point
moved into the box, decides the VNNLIB property with `fractions.Fraction` (decimal constants as
the fractions they are) and compares with the replay's columns:

    in_exact, out_real, full_real          the verdicts
    in_exact_slack, out_real_slack         the slacks, to nine significant digits

It also evaluates in binary32 with numpy in the reference order and compares the verdict `out32`.
Nothing here feeds the results: it only says whether two implementations agree.
"""
from __future__ import annotations

import csv
import random
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crosscheck  # noqa: E402


def dyadic(word: int):
    sign = word >> 31
    exponent = (word >> 23) & 0xff
    fraction = word & 0x7fffff
    assert exponent != 255, 'infinity or NaN'
    mantissa = fraction if exponent == 0 else fraction + (1 << 23)
    return (-mantissa if sign else mantissa, -149 if exponent == 0 else exponent - 150)


def add(a, b):
    if a[0] == 0:
        return b
    if b[0] == 0:
        return a
    if a[1] <= b[1]:
        return (a[0] + (b[0] << (b[1] - a[1])), a[1])
    return ((a[0] << (a[1] - b[1])) + b[0], b[1])


def mul(a, b):
    return (a[0] * b[0], a[1] + b[1])


def to_fraction(a) -> Fraction:
    return Fraction(a[0] * (1 << a[1])) if a[1] >= 0 else Fraction(a[0], 1 << -a[1])


def words_of(array) -> list[int]:
    return [int(w) for w in np.ascontiguousarray(array, dtype=np.float32).reshape(-1).view(np.uint32)]


def prepare_exact(ops):
    """The operations with their constants as dyadic pairs, converted once per network."""
    prepared = []
    for op in ops:
        if op[0] == 'relu':
            prepared.append(op)
        elif op[0] in ('sub', 'mulc', 'addc'):
            prepared.append((op[0], [dyadic(w) for w in words_of(op[1])]))
        elif op[0] == 'dense':
            _, weight, bias = op
            prepared.append(('dense', [[dyadic(w) for w in words_of(row)] for row in weight],
                             [dyadic(w) for w in words_of(bias)] if bias is not None else None))
        else:
            _, params, weight, bias = op
            prepared.append(('conv', params, [dyadic(v) for v in words_of(weight)],
                             [dyadic(v) for v in words_of(bias)] if bias is not None else None))
    return prepared


def evaluate_exact(ops, x):
    """ops: the result of `prepare_exact`; x: list of dyadic pairs."""
    for op in ops:
        if op[0] == 'relu':
            x = [(0, 0) if v[0] < 0 else v for v in x]
        elif op[0] in ('sub', 'mulc', 'addc'):
            consts = op[1]
            if op[0] == 'sub':
                x = [add(v, (-c[0], c[1])) for v, c in zip(x, consts)]
            elif op[0] == 'mulc':
                x = [mul(v, c) for v, c in zip(x, consts)]
            else:
                x = [add(v, c) for v, c in zip(x, consts)]
        elif op[0] == 'dense':
            _, rows, biases = op
            out = []
            for j, row in enumerate(rows):
                acc = (0, 0)
                for v, w in zip(x, row):
                    if v[0] != 0 and w[0] != 0:
                        acc = add(acc, mul(v, w))
                out.append(add(acc, biases[j]) if biases is not None else acc)
            x = out
        else:
            _, params, wd, bd = op
            cin, h, w, cout, kh, kw, sh, sw, pt, pl, pb, pr, _ = params
            oh = (h + pt + pb - kh) // sh + 1
            ow = (w + pl + pr - kw) // sw + 1
            out = []
            for co in range(cout):
                for i in range(oh):
                    for j in range(ow):
                        acc = (0, 0)
                        for ci in range(cin):
                            for r in range(kh):
                                ih = i * sh + r - pt
                                if not 0 <= ih < h:
                                    continue
                                for c in range(kw):
                                    iw = j * sw + c - pl
                                    if not 0 <= iw < w:
                                        continue
                                    v = x[ci * h * w + ih * w + iw]
                                    k = wd[((co * cin + ci) * kh + r) * kw + c]
                                    if v[0] != 0 and k[0] != 0:
                                        acc = add(acc, mul(v, k))
                        out.append(add(acc, bd[co]) if bd is not None else acc)
            x = out
    return x


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


def close(mine, text: str) -> bool:
    if mine is None:
        return text == 'inf'
    if text == 'inf':
        return False
    theirs = float(text)
    if mine == 0 or theirs == 0:
        return mine == 0 and theirs == 0
    return abs(float(mine) - theirs) <= 2e-9 * abs(theirs)


def holds(value) -> str:
    return '1' if value is None or value >= 0 else '0'


def main() -> None:
    family, work, replay_path, out_path, count = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 20261121
    rows = [r for r in csv.DictReader(open(replay_path), delimiter='\t') if r['status'] == 'ok']
    chosen = list(rows)
    random.Random(seed).shuffle(chosen)
    if count:
        chosen = chosen[:count]
    wanted = {r['id']: r for r in chosen}
    blocks = {}
    with (work / 'cases' / f'{family}.cases').open() as stream:
        block = {}
        for line in stream:
            key, _, rest = line.rstrip('\n').partition(' ')
            if key == 'end':
                if block['case'] in wanted:
                    blocks[block['case']] = (block['net'], block['spec'])
            else:
                block[key] = rest
    nets = {}
    checked = disagreements = evaluations = 0
    with open(out_path, 'w', newline='') as out:
        writer = csv.writer(out, delimiter='\t', lineterminator='\n')
        writer.writerow(['id', 'point', 'in_exact', 'in_exact_slack', 'out_real', 'out_real_slack', 'full_real',
                         'out32', 'agrees'])
        for ident in sorted(wanted):
            row = wanted[ident]
            net_path, spec = blocks[ident]
            if net_path not in nets:
                loaded = crosscheck.load_net(net_path)
                nets = {net_path: (loaded, prepare_exact(loaded))}
            ops, exact_ops = nets[net_path]
            formula, _ = parse_formula(spec.split())
            variants = [('reported', '')]
            if row['moved'] != '0':
                variants.append(('projected', 'proj_'))
            for which, prefix in variants:
                words = [int(t, 16) for t in row[prefix + 'x_bits'].split(',')]
                x = [dyadic(w) for w in words]
                xs = [to_fraction(v) for v in x]
                ys = [to_fraction(v) for v in evaluate_exact(exact_ops, x)]
                s_in = slack(formula, xs, ys, True, False)
                s_out = slack(formula, xs, ys, False, True)
                s_full = slack(formula, xs, ys, True, True)
                with np.errstate(all='ignore'):
                    y32 = crosscheck.evaluate(ops, np.array(words, dtype=np.uint32).view(np.float32))
                s_32 = slack(formula, xs, [Fraction(float(v)) for v in y32], False, True)
                ok = (holds(s_in) == row[prefix + 'in_exact'] and close(s_in, row[prefix + 'in_exact_slack'])
                      and holds(s_out) == row[prefix + 'out_real'] and close(s_out, row[prefix + 'out_real_slack'])
                      and holds(s_full) == row[prefix + 'full_real'] and holds(s_32) == row[prefix + 'out32'])
                evaluations += 1
                disagreements += not ok
                writer.writerow([ident, which, holds(s_in), 'inf' if s_in is None else f'{float(s_in):.9e}',
                                 holds(s_out), 'inf' if s_out is None else f'{float(s_out):.9e}', holds(s_full),
                                 holds(s_32), int(ok)])
            checked += 1
    print(f'{family}: {checked} cases, {evaluations} evaluations in exact integer arithmetic, '
          f'{disagreements} disagreements with the Lean replay')


if __name__ == '__main__':
    main()
