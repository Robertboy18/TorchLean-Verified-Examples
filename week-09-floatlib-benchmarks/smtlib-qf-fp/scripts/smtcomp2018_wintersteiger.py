#!/usr/bin/env python3
"""SMT-COMP 2018, main track: the published result of every solver on every Wintersteiger QF_FP benchmark.

2018 is the last year in which the competition ran the whole family (from 2019 on it selects benchmarks and
leaves out those that every solver answers within a second).

Usage: smtcomp2018_wintersteiger.py RESULTS-2018.json.gz CLASSIFY.jsonl OUT.csv SUMMARY.csv
"""
import collections, csv, gzip, json, statistics, sys

results_path, classify_path, out_path, summary_path = sys.argv[1:5]
expected = {}
with open(classify_path, encoding='utf-8') as handle:
    for line in handle:
        record = json.loads(line)
        expected[record['instance']] = record.get('status')
with gzip.open(results_path, 'rt', encoding='utf-8') as handle:
    results = json.load(handle)['results']
rows = []
for r in results:
    f = r['file']
    if f['logic'] != 'QF_FP' or not f['family'] or f['family'][0] != 'wintersteiger':
        continue
    instance = '/'.join(f['family'] + [f['name']])
    rows.append((r['solver'], instance, r['result'], float(r['cpu_time']), float(r['wallclock_time'])))
names = {'COLIBRI 10_06_18 v2038_default': 'COLIBRI v2038',
         'master-2018-06-10-b19c840-competition-default_default': 'CVC4 2018-06-10',
         'z3-4.7.1_default': 'Z3 4.7.1'}
with open(out_path, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['solver', 'instance', 'operation', 'expected', 'result', 'cpu_seconds', 'wall_seconds'])
    for solver, instance, result, cpu, wall in sorted(rows):
        writer.writerow([names.get(solver, solver), instance, instance.split('/')[1], expected.get(instance, ''),
                         result, cpu, wall])
groups = collections.defaultdict(list)
for solver, instance, result, cpu, wall in rows:
    groups[(names.get(solver, solver), instance.split('/')[1])].append((instance, result, cpu, wall))
    groups[(names.get(solver, solver), 'all')].append((instance, result, cpu, wall))
with open(summary_path, 'w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['solver', 'operation', 'benchmarks', 'correct', 'wrong', 'unknown', 'cpu_median_s', 'cpu_max_s',
                     'cpu_total_s', 'wall_median_s', 'wall_max_s', 'wall_total_s'])
    for (solver, operation), group in sorted(groups.items()):
        correct = sum(1 for i, r, _, _ in group if r == expected.get(i))
        unknown = sum(1 for i, r, _, _ in group if r not in ('sat', 'unsat'))
        wrong = len(group) - correct - unknown
        cpu = [c for _, _, c, _ in group]
        wall = [w for _, _, _, w in group]
        writer.writerow([solver, operation, len(group), correct, wrong, unknown, f'{statistics.median(cpu):.4f}',
                         f'{max(cpu):.3f}', f'{sum(cpu):.1f}', f'{statistics.median(wall):.4f}', f'{max(wall):.3f}',
                         f'{sum(wall):.1f}'])
        if operation == 'all':
            print(solver, len(group), 'correct', correct, 'wrong', wrong, 'unknown', unknown,
                  f'cpu median {statistics.median(cpu):.4f}s max {max(cpu):.2f}s total {sum(cpu):.0f}s; '
                  f'wall median {statistics.median(wall):.4f}s total {sum(wall):.0f}s')
            for i, r, c, w in group:
                if r != expected.get(i):
                    print('   ', i, 'expected', expected.get(i), 'got', r, c)
