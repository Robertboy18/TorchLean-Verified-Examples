#!/usr/bin/env python3
"""selftest_gen.py: a self-test of the translator outside the benchmark.

The SMT-LIB benchmark exercises one format (binary64, and binary32 in ten instances), four rounding modes and
fourteen operations. This script writes single-operation instances of the same shape for seven formats, the five
rounding modes of SMT-LIB and every function the translator knows, with the expected result taken from Z3:

1. for each case, ask Z3 for the value of the operation at the fixed operands (`get-value`);
2. write the instance `x = ..., y = ..., r = <Z3's value>, (assert (= (op x y) r))` with status `sat`
   (for a predicate: `(assert (= (p x y) <Z3's truth value>))`).

FloatLib must then prove every instance through the same translator, and cvc5 must answer `sat`. A case where
they differ is a bug in Z3, in cvc5, in FloatLib or in the translation. These instances are not part of
SMT-LIB and are not counted in any benchmark number.

Usage: selftest_gen.py Z3 OUTDIR [--seed N]     (queries in OUTDIR/queries, instances in OUTDIR/instances)
"""
import argparse
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qffp2lean import parse  # noqa: E402

FORMATS = [(2, 2), (3, 5), (5, 11), (6, 20), (8, 24), (11, 53), (15, 113)]
MODES = ['RNE', 'RNA', 'RTP', 'RTN', 'RTZ']
ROUNDED_BINARY = ['fp.add', 'fp.sub', 'fp.mul', 'fp.div']
COMPARISONS = ['fp.eq', 'fp.lt', 'fp.leq', 'fp.gt', 'fp.geq']
CLASSES = ['fp.isNormal', 'fp.isSubnormal', 'fp.isZero', 'fp.isInfinite', 'fp.isNaN', 'fp.isNegative',
           'fp.isPositive']


def literal(eb, sb, bits):
    fb = sb - 1
    sign = bits >> (eb + fb)
    exponent = (bits >> fb) & ((1 << eb) - 1)
    fraction = bits & ((1 << fb) - 1)
    return f'(fp #b{sign} #b{exponent:0{eb}b} #b{fraction:0{fb}b})'


def specials(eb, sb):
    fb = sb - 1
    top = ((1 << eb) - 1) << fb
    sign = 1 << (eb + fb)
    bias = (1 << (eb - 1)) - 1
    one = bias << fb
    values = [0, sign, top, sign | top, top | (1 << (fb - 1)), top | 1, 1, sign | 1, (1 << fb) - 1, 1 << fb,
              top - 1, sign | (top - 1), one, sign | one, one | (1 << (fb - 1)), (bias + 1) << fb,
              (bias - 1) << fb, ((bias + 1) << fb) | (1 << (fb - 1))]
    return [v for v in values if v < (1 << (eb + sb))]


def operands(rng, eb, sb, count, arity):
    """`count` operand tuples: special values, random patterns, and patterns with close exponents."""
    width = eb + sb
    fb = sb - 1
    special = specials(eb, sb)
    tuples = []
    for index in range(count):
        kind = index % 4
        if kind == 0:
            tuples.append(tuple(rng.choice(special) for _ in range(arity)))
        elif kind == 1:
            tuples.append(tuple(rng.getrandbits(width) for _ in range(arity)))
        elif kind == 2:
            base = rng.randrange(1, (1 << eb) - 1)
            row = []
            for _ in range(arity):
                exponent = min(max(base + rng.randrange(-2, 3), 0), (1 << eb) - 2)
                row.append((rng.getrandbits(1) << (eb + fb)) | (exponent << fb) | rng.getrandbits(fb))
            tuples.append(tuple(row))
        else:
            row = [rng.choice(special) if rng.random() < 0.5 else rng.getrandbits(width) for _ in range(arity)]
            tuples.append(tuple(row))
    return tuples


def cases(rng):
    """Every case: (format, function, rounding mode or None, operand tuple, result is Boolean)."""
    for eb, sb in FORMATS:
        for function in ROUNDED_BINARY:
            for mode in MODES:
                for xs in operands(rng, eb, sb, 12, 2):
                    yield (eb, sb), function, mode, xs, False
        for mode in MODES:
            for xs in operands(rng, eb, sb, 12, 3):
                yield (eb, sb), 'fp.fma', mode, xs, False
            for xs in operands(rng, eb, sb, 10, 1):
                yield (eb, sb), 'fp.sqrt', mode, xs, False
            for xs in operands(rng, eb, sb, 10, 1):
                yield (eb, sb), 'fp.roundToIntegral', mode, xs, False
        for function in ('fp.rem', 'fp.min', 'fp.max'):
            for xs in operands(rng, eb, sb, 16, 2):
                yield (eb, sb), function, None, xs, False
        for function in ('fp.abs', 'fp.neg'):
            for xs in operands(rng, eb, sb, 8, 1):
                yield (eb, sb), function, None, xs, False
        for function in COMPARISONS:
            for xs in operands(rng, eb, sb, 12, 2):
                yield (eb, sb), function, None, xs, True
        for function in CLASSES:
            for xs in operands(rng, eb, sb, 10, 1):
                yield (eb, sb), function, None, xs, True


def application(function, mode, names):
    arguments = ([mode] if mode else []) + names
    return f'({function} {" ".join(arguments)})'


def render(value):
    if isinstance(value, list):
        return '(' + ' '.join(render(v) for v in value) + ')'
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('z3')
    parser.add_argument('outdir')
    parser.add_argument('--seed', type=int, default=20261121)
    arguments = parser.parse_args()
    rng = random.Random(arguments.seed)
    everything = list(cases(rng))
    queries = os.path.join(arguments.outdir, 'queries')
    os.makedirs(queries, exist_ok=True)
    names = ['x', 'y', 'z']
    written = 0
    chunk = 250
    for begin in range(0, len(everything), chunk):
        part = everything[begin:begin + chunk]
        query = ['(set-logic QF_FP)']
        for (eb, sb), function, mode, xs, boolean in part:
            query.append('(push 1)')
            for name, bits in zip(names, xs):
                query.append(f'(declare-const {name} (_ FloatingPoint {eb} {sb}))')
                query.append(f'(assert (= {name} {literal(eb, sb, bits)}))')
            query.append(f'(declare-const r {"Bool" if boolean else f"(_ FloatingPoint {eb} {sb})"})')
            query.append(f'(assert (= r {application(function, mode, names[:len(xs)])}))')
            query.append('(check-sat)')
            query.append('(get-value (r))')
            query.append('(pop 1)')
        path = os.path.join(queries, f'query-{begin // chunk:03d}.smt2')
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write('\n'.join(query) + '\n')
        output = subprocess.run([arguments.z3, '-smt2', path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, timeout=3600).stdout.decode('utf-8', errors='replace')
        answers = parse(output)
        if len(answers) != 2 * len(part):
            raise SystemExit(f'{path}: {len(answers)} answers for {len(part)} cases: {output[:400]}')
        for position, ((eb, sb), function, mode, xs, boolean) in enumerate(part):
            status, value = answers[2 * position], answers[2 * position + 1]
            if status != 'sat' or value[0][0] != 'r':
                raise SystemExit(f'{path}: case {position}: {status} {value}')
            result = render(value[0][1])
            number = begin + position
            lines = ['(set-info :smt-lib-version 2.6)', '(set-logic QF_FP)',
                     '(set-info :source |Self-test of qffp2lean.py: the expected result is the value that Z3 '
                     'gives for the operation. Not part of SMT-LIB.|)',
                     '(set-info :status sat)', f'(define-sort FPN () (_ FloatingPoint {eb} {sb}))']
            for name, bits in zip(names, xs):
                lines.append(f'(declare-fun {name} () FPN)')
            if not boolean:
                lines.append('(declare-fun r () FPN)')
            for name, bits in zip(names, xs):
                lines.append(f'(assert (= {name} {literal(eb, sb, bits)}))')
            term = application(function, mode, names[:len(xs)])
            if boolean:
                lines.append(f'(assert (= {term} {result}))')
            else:
                lines.append(f'(assert (= r {result}))')
                lines.append(f'(assert (= {term} r))')
            lines += ['(check-sat)', '(exit)']
            directory = os.path.join(arguments.outdir, 'instances', 'selftest', f'{function[3:]}-{eb}-{sb}')
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, f'{function[3:]}-{eb}-{sb}-{mode or "none"}-{number:05d}.smt2'), 'w',
                      encoding='utf-8') as handle:
                handle.write('\n'.join(lines) + '\n')
            written += 1
    print(f'{written} instances under {arguments.outdir}/instances')


if __name__ == '__main__':
    main()
