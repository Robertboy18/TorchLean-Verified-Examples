#!/usr/bin/env python3
"""zero_pairs.py: the fp.min and fp.max instances whose operands are two zeros of opposite signs.

SMT-LIB leaves `fp.min` and `fp.max` open there. For each such instance: the operands, the benchmark's result
`r`, and the result of IEEE 754-2019 `minimumNumber` or `maximumNumber` (-0 is less than +0).

Usage: zero_pairs.py ROOT OUT.csv
"""
import csv, os, re, sys

root, out = sys.argv[1], sys.argv[2]
LITERAL = re.compile(r'\(assert \(= ([xyr]) (.*)\)\)\s*$')


def bits(literal):
    m = re.match(r'\(fp #b([01]) #b([01]+) #(b|x)([0-9a-fA-F]+)\)', literal)
    if m:
        frac = m.group(4)
        width = len(frac) if m.group(3) == 'b' else 4 * len(frac)
        value = int(frac, 2 if m.group(3) == 'b' else 16)
        return (int(m.group(1)) << (len(m.group(2)) + width)) | (int(m.group(2), 2) << width) | value
    m = re.match(r'\(_ ([+-]zero) (\d+) (\d+)\)', literal)
    if m:
        return 0 if m.group(1) == '+zero' else 1 << (int(m.group(2)) + int(m.group(3)) - 1)
    return None


rows = []
for operation in ('min', 'max'):
    directory = os.path.join(root, 'wintersteiger', operation)
    for name in sorted(os.listdir(directory)):
        values = {}
        with open(os.path.join(directory, name), encoding='utf-8') as handle:
            for line in handle:
                m = LITERAL.match(line)
                if m:
                    values[m.group(1)] = bits(m.group(2))
        x, y = values.get('x'), values.get('y')
        sign = 1 << 63
        if x in (0, sign) and y in (0, sign) and x != y:
            ieee = sign if operation == 'min' else 0
            r = values.get('r')
            rows.append([f'wintersteiger/{operation}/{name}', f'{x:#018x}', f'{y:#018x}',
                         '' if r is None else f'{r:#018x}', f'{ieee:#018x}',
                         '' if r is None else int(r == ieee)])
with open(out, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['instance', 'x', 'y', 'benchmark_r', 'ieee2019_result', 'benchmark_r_is_ieee2019'])
    writer.writerows(rows)
for row in rows:
    print(*row)
print(len(rows), 'instances with zeros of opposite signs')
