#!/usr/bin/env python3
"""make_smt_batches.py: the instances of each Lean batch file as one SMT-LIB script, for timing a solver the way
Lean is timed in batches: one process answers every instance of the file.

The instances follow one another with `(reset)` in between and without their `(exit)`. The solver prints one
answer per instance; `NAME.expected` holds the benchmark's statuses in the same order.

Usage: make_smt_batches.py INDEX.jsonl ROOT OUTDIR
"""
import json, os, re, sys

index, root, outdir = sys.argv[1:4]
os.makedirs(outdir, exist_ok=True)
files = {}
for line in open(index, encoding='utf-8'):
    record = json.loads(line)
    instances = files.setdefault(record['file'], [])
    if not instances or instances[-1][0] != record['instance']:
        instances.append((record['instance'], record['status']))
for name, instances in files.items():
    stem = os.path.splitext(name)[0]
    with open(os.path.join(outdir, stem + '.smt2'), 'w', encoding='utf-8') as out, \
            open(os.path.join(outdir, stem + '.expected'), 'w', encoding='utf-8') as expected:
        for position, (instance, status) in enumerate(instances):
            text = open(os.path.join(root, instance), encoding='utf-8').read()
            text = re.sub(r'^\(exit\)\s*$', '', text, flags=re.M)
            if position:
                out.write('(reset)\n')
            out.write(text.rstrip('\n') + '\n')
            expected.write(status + '\n')
    print(stem, len(instances))
