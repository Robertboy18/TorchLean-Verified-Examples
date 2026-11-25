#!/usr/bin/env python3
"""SMT-COMP 2025, single-query track, logic QF_FP: one CSV row per (solver, benchmark), joined with the translator's
classification of the benchmark (fixed inputs or search).

Usage: smtcomp_qffp.py RESULTS.json.gz BENCHMARKS.json.gz CLASSIFY.jsonl OUT.csv
"""
import collections, csv, gzip, json, sys

results_path, benchmarks_path, classify_path, out_path = sys.argv[1:5]
reach = {}
status = {}
with open(classify_path, encoding='utf-8') as handle:
    for line in handle:
        record = json.loads(line)
        reach[record['instance']] = record['reach']
        status[record['instance']] = record.get('status')

with gzip.open(results_path, 'rt', encoding='utf-8') as handle:
    results = json.load(handle)['results']
rows = []
for r in results:
    f = r['file']
    if f['logic'] != 'QF_FP' or r['track'] != 'SingleQuery':
        continue
    instance = '/'.join(f['family'] + [f['name']])
    rows.append((r['solver'], instance, r['result'], r['cpu_time'], r['wallclock_time'], r['memory_usage']))
solvers = sorted({row[0] for row in rows})
benchmarks = sorted({row[1] for row in rows})
print('solvers:', solvers)
print('benchmarks in the track:', len(benchmarks))
missing = [b for b in benchmarks if b not in reach]
print('not in the SMT-LIB 2025 archive:', len(missing), missing[:5])
by_family = collections.Counter(b.split('/')[0] for b in benchmarks)
print('by family:', dict(by_family))
by_reach = collections.Counter((b.split('/')[0], reach.get(b)) for b in benchmarks)
print('by family and reach:', dict(by_reach))
with open(out_path, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['solver', 'instance', 'family', 'reach', 'status', 'result', 'cpu_seconds', 'wall_seconds', 'memory_bytes'])
    for solver, instance, result, cpu, wall, memory in sorted(rows):
        writer.writerow([solver, instance, instance.split('/')[0], reach.get(instance, ''), status.get(instance, ''),
                         result, cpu, wall, memory])
print('result values:', dict(collections.Counter(row[2] for row in rows)))
for solver in solvers:
    mine = [row for row in rows if row[0] == solver]
    solved = [row for row in mine if row[2] in ('sat', 'unsat')]
    print(f'{solver}: {len(mine)} benchmarks, {len(solved)} sat/unsat, cpu of solved {sum(r[3] for r in solved):.1f}s, '
          f'wall of solved {sum(r[4] for r in solved):.1f}s, results {dict(collections.Counter(r[2] for r in mine))}')

# all QF_FP benchmarks of the competition's benchmark list
with gzip.open(benchmarks_path, 'rt', encoding='utf-8') as handle:
    listed = json.load(handle)['non_incremental']
qffp = ['/'.join(b['file']['family'] + [b['file']['name']]) for b in listed if b['file']['logic'] == 'QF_FP']
print('QF_FP benchmarks in benchmarks-2025:', len(qffp), dict(collections.Counter(b.split('/')[0] for b in qffp)))
