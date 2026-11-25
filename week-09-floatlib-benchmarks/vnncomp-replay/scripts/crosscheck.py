"""An independent check of the Lean replay, with the host processor's float32 arithmetic.

    python3 -I scripts/crosscheck.py CASES LEAN_TSV OUT_TSV

For every case of CASES that LEAN_TSV has a line for, this
  1. rounds each decimal input to binary32 with exact rational arithmetic (`fractions.Fraction`,
     nearest, ties to even) and compares the word with FloatLib's (`x_bits`);
  2. evaluates the `.net` network with numpy float32 operations in the reference order of
     `Replay.lean` (left to right, one multiplication and one addition per term, bias last; each
     numpy call is one IEEE operation per element, so no fused multiply-add can occur) and compares
     every output word with FloatLib's (`y_bits`), where +0 and -0 count as equal.

Nothing here feeds the results: it only says whether two implementations agree.
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True


def dec_to_fraction(text: str) -> Fraction:
    return Fraction(text)


def round_f32(q: Fraction) -> int:
    """The binary32 word nearest to q, ties to even (q is far from the overflow threshold here)."""
    guess = np.float32(float(q))
    candidates = {float(guess), float(np.nextafter(guess, np.float32(np.inf))),
                  float(np.nextafter(guess, np.float32(-np.inf)))}
    best = None
    for value in candidates:
        distance = abs(Fraction(value) - q)
        word = int(np.float32(value).view(np.uint32))
        key = (distance, word & 1)
        if best is None or key < best[0]:
            best = (key, word)
    word = best[1]
    if q == 0:
        word = 0
    return word


def load_net(path: str):
    tokens = Path(path).read_text().split()
    assert tokens[0] == 'net'
    position = 3
    ops = []

    def words(count):
        nonlocal position
        chunk = tokens[position:position + count]
        position += count
        return np.array([int(t, 16) for t in chunk], dtype=np.uint32).view(np.float32)

    while tokens[position] != 'end':
        kind = tokens[position]
        position += 1
        if kind == 'relu':
            ops.append(('relu',))
        elif kind in ('sub', 'mulc', 'addc'):
            count = int(tokens[position]); position += 1
            ops.append((kind, words(count)))
        elif kind == 'dense':
            nin, nout, has_bias = (int(t) for t in tokens[position:position + 3]); position += 3
            weight = words(nin * nout).reshape(nout, nin)
            bias = words(nout) if has_bias else None
            ops.append(('dense', weight, bias))
        elif kind == 'conv':
            params = [int(t) for t in tokens[position:position + 13]]; position += 13
            cin, h, w, cout, kh, kw = params[:6]
            weight = words(cout * cin * kh * kw).reshape(cout, cin, kh, kw)
            bias = words(cout) if params[12] else None
            ops.append(('conv', params, weight, bias))
        else:
            raise ValueError(kind)
    return ops


def evaluate(ops, x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    for op in ops:
        if op[0] == 'relu':
            x = np.where(x < np.float32(0), np.float32(0), x).astype(np.float32)
        elif op[0] == 'sub':
            x = (x - op[1]).astype(np.float32)
        elif op[0] == 'mulc':
            x = np.multiply(x, op[1], dtype=np.float32)
        elif op[0] == 'addc':
            x = np.add(x, op[1], dtype=np.float32)
        elif op[0] == 'dense':
            _, weight, bias = op
            acc = np.zeros(weight.shape[0], dtype=np.float32)
            for k in range(weight.shape[1]):
                product = np.multiply(x[k], weight[:, k], dtype=np.float32)
                acc = np.add(acc, product, dtype=np.float32)
            if bias is not None:
                acc = np.add(acc, bias, dtype=np.float32)
            x = acc
        else:
            _, params, weight, bias = op
            cin, h, w, cout, kh, kw, sh, sw, pt, pl, pb, pr, _ = params
            oh = (h + pt + pb - kh) // sh + 1
            ow = (w + pl + pr - kw) // sw + 1
            image = np.zeros((cin, h + pt + pb, w + pl + pr), dtype=np.float32)
            image[:, pt:pt + h, pl:pl + w] = x.reshape(cin, h, w)
            acc = np.zeros((cout, oh, ow), dtype=np.float32)
            for ci in range(cin):
                for r in range(kh):
                    for c in range(kw):
                        patch = image[ci, r:r + sh * oh:sh, c:c + sw * ow:sw]
                        product = np.multiply(weight[:, ci, r, c][:, None, None], patch[None, :, :],
                                              dtype=np.float32)
                        acc = np.add(acc, product, dtype=np.float32)
            if bias is not None:
                acc = np.add(acc, bias[:, None, None], dtype=np.float32)
            x = acc.reshape(-1)
    return x


def main() -> None:
    cases_path, lean_path, out_path = sys.argv[1:4]
    lean = {}
    with open(lean_path) as stream:
        header = stream.readline().rstrip('\n').split('\t')
        for line in stream:
            fields = line.rstrip('\n').split('\t')
            if len(fields) == len(header) and fields[1] == 'ok':
                lean[fields[0]] = dict(zip(header, fields))
    nets = {}
    checked = x_agree = y_agree = 0
    worst = []
    with open(cases_path) as stream, open(out_path, 'w') as out:
        out.write('id\tx_words_agree\ty_values_agree\tdiffering_outputs\n')
        block = {}
        for line in stream:
            key, _, rest = line.rstrip('\n').partition(' ')
            if key != 'end':
                block[key] = rest
                continue
            ident = block['case']
            if ident not in lean:
                continue
            row = lean[ident]
            xs = block['x'].split()
            words = [round_f32(dec_to_fraction(t)) for t in xs]
            lean_x = [int(t, 16) for t in row['x_bits'].split(',')]
            same_x = all((a == b) or ((a | b) & 0x7fffffff) == 0 for a, b in zip(words, lean_x))
            if block['net'] not in nets:
                nets = {block['net']: load_net(block['net'])}
            x = np.array(lean_x, dtype=np.uint32).view(np.float32)
            with np.errstate(all='ignore'):
                y = evaluate(nets[block['net']], x)
            lean_y = np.array([int(t, 16) for t in row['y_bits'].split(',')], dtype=np.uint32).view(np.float32)
            differing = int(np.sum(~((y == lean_y) | (np.isnan(y) & np.isnan(lean_y)))))
            checked += 1
            x_agree += same_x
            y_agree += differing == 0
            if differing:
                worst.append(ident)
            out.write(f'{ident}\t{int(same_x)}\t{int(differing == 0)}\t{differing}\n')
    print(f'{cases_path}: {checked} cases checked, inputs agree in {x_agree}, outputs agree in {y_agree}')
    for ident in worst[:10]:
        print('  differs:', ident)


if __name__ == '__main__':
    main()
