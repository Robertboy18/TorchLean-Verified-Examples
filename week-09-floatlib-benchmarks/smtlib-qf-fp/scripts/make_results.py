#!/usr/bin/env python3
"""make_results.py: the small result files behind the figures and the tables of the README.

Reads what the runs left (nothing is measured here):

  results/wintersteiger-instances.csv   collect.py on the main run (every Wintersteiger instance)
  results/side-instances.csv            collect.py on the other families and the controls
  results/timing-sample.csv             time_solvers.py: Z3 and cvc5, one process per instance, 3 repetitions
  results/timing-sample-leanbatch.csv   time_solvers.py: Lean on the same sample in batch files
  lean/sample/*.repN.out                Lean's output of those batch runs (kernel time and clock per instance)
  results/timing-process.csv            time_solvers.py: Z3, cvc5 and Lean with one instance per process
  results/smtcomp2025-qffp.csv          smtcomp_qffp.py: SMT-COMP 2025, QF_FP, one row per solver and benchmark
  results/smtcomp2018-wintersteiger-summary.csv   smtcomp2018_wintersteiger.py
  meta/classify.jsonl                   qffp2lean.py classify

Writes results/fig-time.csv, results/fig-coverage.csv, results/fig-smtcomp2025.csv, results/facts.json and
the Markdown tables under tables/.

Usage: make_results.py RUN        (the directory with `results/`, `lean/` and `meta/` of the runs)
"""
import collections
import csv
import json
import os
import statistics
import sys

run_dir = sys.argv[1]
R = os.path.join(run_dir, 'results')
T = os.path.join(run_dir, 'tables')
os.makedirs(T, exist_ok=True)
OPERATIONS = ['add', 'sub', 'mul', 'div', 'fma', 'sqrt', 'rem', 'toIntegral', 'min', 'max', 'abs', 'eq', 'lt', 'gt']
SMT_NAME = {'add': 'fp.add', 'sub': 'fp.sub', 'mul': 'fp.mul', 'div': 'fp.div', 'fma': 'fp.fma', 'sqrt': 'fp.sqrt',
            'rem': 'fp.rem', 'toIntegral': 'fp.roundToIntegral', 'min': 'fp.min', 'max': 'fp.max', 'abs': 'fp.abs',
            'eq': 'fp.eq', 'lt': 'fp.lt', 'gt': 'fp.gt'}
FLOATLIB_NAME = {'add': 'add', 'sub': 'sub', 'mul': 'mul', 'div': 'div', 'fma': 'fma', 'sqrt': 'sqrt',
                 'rem': 'remainder', 'toIntegral': 'roundToIntegral', 'min': 'minimumNumber',
                 'max': 'maximumNumber', 'abs': 'abs', 'eq': 'compareEqual', 'lt': 'compareLess',
                 'gt': 'compareGreater'}
STANDARD = {'propext', 'Classical.choice', 'Quot.sound'}


def read(name):
    with open(os.path.join(R, name), newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def spread(values):
    """Median, 5th and 95th percentile."""
    values = sorted(values)
    cuts = statistics.quantiles(values, n=20, method='inclusive')
    return statistics.median(values), cuts[0], cuts[-1]


def write(name, header, rows):
    with open(os.path.join(R, name), 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def table(name, lines):
    with open(os.path.join(T, name), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def operation_of(instance):
    parts = instance.split('/')
    return parts[1] if parts[0] == 'wintersteiger' else None


facts = {}

# ---------------------------------------------------------------------------------------------------------------
# 1. The main run: every Wintersteiger instance

main = read('wintersteiger-instances.csv')
by_operation = collections.defaultdict(list)
for row in main:
    by_operation[row['operation']].append(row)
lines = ['| operation | FloatLib operation | instances | sat | unsat | proved | disagree | kernel total (s) | '
         'kernel median (ms) | of which decision (ms) | 95th percentile (ms) | slowest (ms) |',
         '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']


def kernel_cells(rows):
    kernel = [int(r['kernel_ns']) / 1e6 for r in rows if r['proved'] == '1']
    decision = [int(r['kernel_auxiliary_ns']) / 1e6 for r in rows if r['proved'] == '1']
    median, _, high = spread(kernel)
    return (f'{sum(kernel) / 1e3:.2f} | {median:.2f} | {statistics.median(decision):.2f} | {high:.2f} | '
            f'{max(kernel):.1f} |')


for operation in OPERATIONS:
    rows = by_operation[operation]
    lines.append(f'| `{SMT_NAME[operation]}` | `{FLOATLIB_NAME[operation]}` | {len(rows):,} | '
                 f'{sum(r["status"] == "sat" for r in rows):,} | {sum(r["status"] == "unsat" for r in rows):,} | '
                 f'{sum(r["proved"] == "1" for r in rows):,} | {sum(r["proved"] != "1" for r in rows)} | '
                 + kernel_cells(rows))
all_kernel = [int(r['kernel_ns']) / 1e6 for r in main if r['proved'] == '1']
all_decision = [int(r['kernel_auxiliary_ns']) / 1e6 for r in main if r['proved'] == '1']
all_statement = [int(r['kernel_statement_ns']) / 1e6 for r in main if r['proved'] == '1']
median, low, high = spread(all_kernel)
lines.append(f'| all | | {len(main):,} | {sum(r["status"] == "sat" for r in main):,} | '
             f'{sum(r["status"] == "unsat" for r in main):,} | {sum(r["proved"] == "1" for r in main):,} | '
             f'{sum(r["proved"] != "1" for r in main)} | ' + kernel_cells(main))
table('wintersteiger-by-operation.md', lines)
summary = json.load(open(os.path.join(R, 'wintersteiger-summary.json'), encoding='utf-8'))
facts['main'] = {
    'instances': len(main), 'proved': sum(r['proved'] == '1' for r in main),
    'kernel_seconds': sum(all_kernel) / 1e3, 'kernel_median_ms': median, 'kernel_p05_ms': low,
    'kernel_decision_seconds': sum(all_decision) / 1e3, 'kernel_rewrite_seconds': sum(all_statement) / 1e3,
    'kernel_decision_median_ms': statistics.median(all_decision),
    'kernel_rewrite_median_ms': statistics.median(all_statement),
    'elaboration_seconds': sum(int(r['elaboration_ns']) for r in main if r['elaboration_ns'] != '') / 1e9,
    'kernel_p95_ms': high, 'kernel_max_ms': max(all_kernel),
    'files': summary['files'], 'process_wall_seconds': summary['file_wall_seconds'],
    'process_user_seconds': summary['file_user_seconds'], 'process_system_seconds': summary['file_system_seconds'],
    'peak_rss_gb': summary['peak_rss_kb'] / 1e6,
    'axioms': sorted({a for r in main for a in r['axioms'].split()}),
    'per_instance_process_ms': 1e3 * summary['file_wall_seconds'] / len(main),
}

# ---------------------------------------------------------------------------------------------------------------
# 2. Families: what is in reach

classify = [json.loads(line) for line in open(os.path.join(run_dir, 'meta', 'classify.jsonl'), encoding='utf-8')]
side = read('side-instances.csv')
proved_side = {r['instance'] for r in side
               if r['proved'] == '1' and not r['theorem'].endswith('#mutated»')
               and (r['status'] == 'unknown' or r['claims'] == r['status'])}
family_label = {'wintersteiger': 'wintersteiger', 'griggio': 'griggio', 'schanda': 'schanda',
                '20210211-Vector': 'Vector', 'ramalho': 'ramalho'}
families = collections.OrderedDict()
for record in classify:
    label = family_label.get(record['family'], 'UltimateAutomizer')
    entry = families.setdefault(label, {'total': 0, 'fixed': 0, 'proved': 0, 'search': 0})
    entry['total'] += 1
    if record['reach'] == 'search':
        entry['search'] += 1
    else:
        entry['fixed'] += 1
proved_main = {r['instance'] for r in main if r['proved'] == '1'}
for record in classify:
    if record['reach'] != 'search' and record['family'] == 'wintersteiger' and record['instance'] in proved_main:
        families['wintersteiger']['proved'] += 1
    if record['reach'] != 'search' and record['family'] != 'wintersteiger' and record['instance'] in proved_side:
        families[family_label.get(record['family'], 'UltimateAutomizer')]['proved'] += 1
order = ['wintersteiger', 'griggio', 'schanda', 'Vector', 'ramalho', 'UltimateAutomizer']
write('fig-coverage.csv', ['family', 'benchmarks', 'fixed_inputs', 'proved', 'search'],
      [[name, families[name]['total'], families[name]['fixed'], families[name]['proved'], families[name]['search']]
       for name in order])
lines = ['| family | benchmarks | fixed inputs | translated | proved by `float_decide` | search problems |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
for name in order:
    f = families[name]
    lines.append(f'| {name} | {f["total"]:,} | {f["fixed"]:,} | {f["fixed"]:,} | {f["proved"]:,} | {f["search"]:,} |')
lines.append(f'| all | {sum(f["total"] for f in families.values()):,} | {sum(f["fixed"] for f in families.values()):,} | '
             f'{sum(f["fixed"] for f in families.values()):,} | {sum(f["proved"] for f in families.values()):,} | '
             f'{sum(f["search"] for f in families.values()):,} |')
table('families.md', lines)
facts['families'] = families

# ---------------------------------------------------------------------------------------------------------------
# 3. SMT-COMP 2025, QF_FP single query

smtcomp = read('smtcomp2025-qffp.csv')
solvers = collections.OrderedDict()
for row in smtcomp:
    entry = solvers.setdefault(row['solver'], {'benchmarks': 0, 'solved': 0, 'sat': 0, 'unsat': 0, 'cpu': 0.0,
                                               'wall': 0.0, 'fixed': 0})
    entry['benchmarks'] += 1
    entry['fixed'] += row['reach'] != 'search'
    if row['result'] in ('sat', 'unsat'):
        entry['solved'] += 1
        entry[row['result']] += 1
        entry['cpu'] += float(row['cpu_seconds'])
        entry['wall'] += float(row['wall_seconds'])
ranked = sorted(solvers.items(), key=lambda item: -item[1]['solved'])
write('fig-smtcomp2025.csv', ['solver', 'benchmarks', 'solved', 'sat', 'unsat', 'cpu_seconds_solved',
                              'wall_seconds_solved', 'benchmarks_with_fixed_inputs'],
      [[name, e['benchmarks'], e['solved'], e['sat'], e['unsat'], f'{e["cpu"]:.1f}', f'{e["wall"]:.1f}', e['fixed']]
       for name, e in ranked])
lines = ['| solver | benchmarks | solved | sat | unsat | not solved | CPU time on the solved (s) |',
         '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
for name, e in ranked:
    lines.append(f'| {name} | {e["benchmarks"]} | {e["solved"]} | {e["sat"]} | {e["unsat"]} | '
                 f'{e["benchmarks"] - e["solved"]} | {e["cpu"]:,.0f} |')
benchmarks_2025 = sorted({row['instance'] for row in smtcomp})
fixed_2025 = sorted({row['instance'] for row in smtcomp if row['reach'] != 'search'})
lines.append(f'| FloatLib `float_decide` | {len(benchmarks_2025)} | {len(fixed_2025)} in reach | | | | |')
table('smtcomp2025.md', lines)
facts['smtcomp2025'] = {'benchmarks': len(benchmarks_2025), 'fixed': len(fixed_2025),
                        'by_family': dict(collections.Counter(b.split('/')[0] for b in benchmarks_2025))}

# ---------------------------------------------------------------------------------------------------------------
# 4. Times on the sample, one row per series and operation

time_rows = []
per_series = {}


def add_series(name, values_by_instance):
    """`values_by_instance`: instance -> milliseconds (already the median over repetitions)."""
    per_series[name] = values_by_instance
    groups = collections.defaultdict(list)
    for instance, value in values_by_instance.items():
        operation = operation_of(instance)
        if operation:
            groups[operation].append(value)
            groups['all'].append(value)
    for operation in OPERATIONS + ['all']:
        if groups[operation]:
            median, low, high = spread(groups[operation])
            time_rows.append([name, operation, len(groups[operation]), f'{median:.4f}', f'{low:.4f}', f'{high:.4f}',
                              f'{min(groups[operation]):.4f}', f'{max(groups[operation]):.4f}'])


timing_path = os.path.join(R, 'timing-sample.csv')
if os.path.exists(timing_path):
    timing = read('timing-sample.csv')
    answers = collections.defaultdict(collections.Counter)
    for tool in ('z3', 'cvc5'):
        values = collections.defaultdict(list)
        cpu = collections.defaultdict(list)
        for row in timing:
            if row['tool'] == tool:
                values[row['instance']].append(int(row['wall_ns']) / 1e6)
                cpu[row['instance']].append(int(row['cpu_ns']) / 1e6)
                answers[tool][(row['instance'], row['answer'])] += 1
        add_series(f'{tool} process wall', {k: min(v) for k, v in values.items()})
        add_series(f'{tool} process wall median3', {k: statistics.median(v) for k, v in values.items()})
        add_series(f'{tool} process cpu', {k: min(v) for k, v in cpu.items()})
    expected = {r['instance']: r['status'] for r in classify}
    facts['sample'] = {'instances': len({row['instance'] for row in timing}),
                       'repetitions': len({row['repetition'] for row in timing})}
    for tool in ('z3', 'cvc5'):
        runs = sum(answers[tool].values())
        agree = sum(count for (instance, answer), count in answers[tool].items() if answer == expected[instance])
        unknown_status = sum(count for (instance, answer), count in answers[tool].items()
                             if expected[instance] == 'unknown')
        other = {f'{instance}: {answer}': count for (instance, answer), count in answers[tool].items()
                 if answer != expected[instance] and expected[instance] != 'unknown'}
        facts['sample'][tool] = {'runs': runs, 'agree_with_status': agree, 'status_unknown_runs': unknown_status,
                                 'other': other,
                                 'answers_on_unknown': {instance: answer for (instance, answer) in answers[tool]
                                                        if expected[instance] == 'unknown'}}

    # Lean on the same sample in batch files: kernel time and whole time per instance, median over repetitions.
    batch = read('timing-sample-leanbatch.csv')
    kernel = collections.defaultdict(list)
    decision = collections.defaultdict(list)
    rewrite = collections.defaultdict(list)
    whole = collections.defaultdict(list)
    file_stats = []
    for row in batch:
        path = os.path.join(run_dir, 'lean', 'sample', f'{row["file"]}.rep{row["repetition"]}.out')
        checks, file_clock = [], None
        with open(path, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                if line.startswith('QFFPCLOCK|'):
                    file_clock = int(line.split('|')[1])
                elif line.startswith('QFFP|'):
                    fields = line.rstrip('\n').split('|')
                    name, nanoseconds, axioms, clock = fields[1:5]
                    start, stop = (int(x) for x in clock.split())
                    ok = nanoseconds != 'FAILED' and set(axioms.split(',')) - {''} <= STANDARD
                    checks.append((name, int(nanoseconds) if ok else None, start, stop,
                                   tuple(int(x) for x in fields[5].split())))
        wall = int(row['wall_ns'])
        inside = checks[-1][3] - file_clock          # from the first theorem to the last report
        share = (wall - inside) / len(checks)        # loading the library, the prelude and exit, shared equally
        file_stats.append((row['file'], int(row['repetition']), wall / 1e9, int(row['cpu_ns']) / 1e9,
                           len(checks), int(row['accepted']), int(row['kernel_ns']) / 1e9, (wall - inside) / 1e9))
        for position, (name, nanoseconds, start, stop, (statement, auxiliary)) in enumerate(checks):
            instance = name[len('QFFP.«'):-1]
            if '#' in instance.rsplit('/', 1)[-1]:
                continue                              # the two candidates of an instance of unknown status
            if nanoseconds is not None:
                kernel[instance].append(nanoseconds / 1e6)
                decision[instance].append(auxiliary / 1e6)
                rewrite[instance].append(statement / 1e6)
            previous = checks[position - 1][3] if position > 0 else file_clock
            whole[instance].append((stop - previous + share) / 1e6)
    add_series('lean kernel', {k: min(v) for k, v in kernel.items()})
    add_series('lean kernel median3', {k: statistics.median(v) for k, v in kernel.items()})
    add_series('lean kernel decision', {k: min(v) for k, v in decision.items()})
    add_series('lean kernel rewrite', {k: min(v) for k, v in rewrite.items()})
    add_series('lean batch whole', {k: min(v) for k, v in whole.items()})
    add_series('lean batch whole median3', {k: statistics.median(v) for k, v in whole.items()})
    write('timing-sample-leanbatch-files.csv',
          ['file', 'repetition', 'wall_seconds', 'cpu_seconds', 'theorems', 'accepted', 'kernel_seconds',
           'seconds_outside_the_theorems'], [[f, r, f'{w:.3f}', f'{c:.3f}', n, a, f'{k:.3f}', f'{o:.3f}']
                                             for f, r, w, c, n, a, k, o in file_stats])
    by_rep = collections.defaultdict(lambda: [0.0, 0, 0.0, 0.0])
    for f, r, w, c, n, a, k, o in file_stats:
        by_rep[r][0] += w
        by_rep[r][1] += n
        by_rep[r][2] += k
        by_rep[r][3] += o
    facts['sample']['lean_batch'] = {str(r): {'wall_seconds': v[0], 'theorems': v[1], 'kernel_seconds': v[2],
                                              'outside_seconds': v[3], 'ms_per_theorem': 1e3 * v[0] / v[1]}
                                     for r, v in sorted(by_rep.items())}

load_path = os.path.join(R, 'timing-sample.csv.load.csv')
launch_path = os.path.join(R, 'timing-sample.csv.launch.json')
if os.path.exists(load_path) and os.path.exists(launch_path):
    launch = json.load(open(launch_path, encoding='utf-8'))
    load = read('timing-sample.csv.load.csv')
    busy = [1 - float(row['machine_idle']) for row in load]
    mine = [float(row['cpu_not_niced']) + float(row['cpu_niced']) for row in load]
    sibling = [float(row['sibling_not_niced']) + float(row['sibling_niced']) for row in load]
    facts['sample_load'] = {
        'cpu': launch['cpu'], 'cookie': launch['core_scheduling_cookie'], 'nice': launch['nice'],
        'started_utc': launch['started_utc'], 'samples': len(load),
        'machine_busy_min': min(busy), 'machine_busy_median': statistics.median(busy), 'machine_busy_max': max(busy),
        'own_cpu_busy_median': statistics.median(mine), 'sibling_busy_median': statistics.median(sibling),
        'load1_min': min(float(row['load1']) for row in load), 'load1_max': max(float(row['load1']) for row in load),
        'sentence': (f'over the run {100 * min(busy):.0f} to {100 * max(busy):.0f} percent of its 96 hardware '
                     f'threads were in use (median {100 * statistics.median(busy):.0f}, load average '
                     f'{min(float(row["load1"]) for row in load):.0f} to '
                     f'{max(float(row["load1"]) for row in load):.0f}). The run was pinned to the least busy core '
                     f'(CPU {launch["cpu"]}) at nice {launch["nice"]} with a core-scheduling cookie, so that '
                     'nothing else ran on the second hardware thread of that core while a process was timed; '
                     'other tasks still shared the first.')}

process_path = os.path.join(R, 'timing-process.csv')
if os.path.exists(process_path):
    process = read('timing-process.csv')
    facts['process'] = {}
    for tool in ('z3', 'cvc5', 'lean'):
        values = collections.defaultdict(list)
        cpu = collections.defaultdict(list)
        answers = collections.Counter()
        for row in process:
            if row['tool'] == tool:
                values[row['instance']].append(int(row['wall_ns']) / 1e6)
                cpu[row['instance']].append(int(row['cpu_ns']) / 1e6)
                answers[row['answer']] += 1
        if not values:
            continue
        medians = {k: min(v) for k, v in values.items()}
        add_series(f'{tool} one process per instance', medians)
        everything = list(medians.values())
        facts['process'][tool] = {'instances': len(medians), 'median_ms': statistics.median(everything),
                                  'min_ms': min(everything), 'max_ms': max(everything),
                                  'cpu_median_ms': statistics.median(min(v) for v in cpu.values()),
                                  'answers': dict(answers)}

# The solvers on the same batches as Lean: one process answers every instance of a file.
smt_batch_path = os.path.join(R, 'timing-sample-smtbatch.csv')
if os.path.exists(smt_batch_path):
    facts['smt_batch'] = {}
    rows = read('timing-sample-smtbatch.csv')
    for tool in sorted({row['tool'] for row in rows}):
        mine = [row for row in rows if row['tool'] == tool]
        per_rep = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
        for row in mine:
            entry = per_rep[row['repetition']]
            entry[0] += int(row['wall_ns'])
            entry[1] += int(row['cpu_ns'])
            entry[2] += int(row['instances'])
            entry[3] += int(row['answers'])
            entry[4] += int(row['agree_with_status'])
        per_instance = sorted(1e-6 * v[0] / v[2] for v in per_rep.values())
        facts['smt_batch'][tool] = {
            'repetitions': len(per_rep), 'instances': next(iter(per_rep.values()))[2],
            'ms_per_instance_median': statistics.median(per_instance),
            'ms_per_instance_min': per_instance[0], 'ms_per_instance_max': per_instance[-1],
            'answers': sum(v[3] for v in per_rep.values()), 'agree': sum(v[4] for v in per_rep.values()),
            'expected': sum(v[2] for v in per_rep.values())}

# Z3 and cvc5 once on every Wintersteiger instance.
full_path = os.path.join(R, 'timing-full.csv')
if os.path.exists(full_path):
    facts['full'] = {}
    rows = read('timing-full.csv')
    expected_status = {r['instance']: r['status'] for r in classify}
    by_tool = collections.defaultdict(list)
    for row in rows:
        by_tool[row['tool']].append(row)
    full_lines = ['| operation | Z3 5.4.1 median (ms) | Z3 5.4.1 slowest (ms) | cvc5 1.4.2 median (ms) | '
                  'cvc5 1.4.2 slowest (ms) |', '| --- | ---: | ---: | ---: | ---: |']
    per_operation = collections.defaultdict(dict)
    for tool, mine in sorted(by_tool.items()):
        wall = [int(r['wall_ns']) / 1e6 for r in mine]
        cpu = [int(r['cpu_ns']) / 1e6 for r in mine]
        agree = sum(r['answer'] == expected_status[r['instance']] for r in mine)
        facts['full'][tool] = {
            'instances': len(mine), 'agree_with_status': agree,
            'other_answers': dict(collections.Counter(r['answer'] for r in mine
                                                      if r['answer'] != expected_status[r['instance']])),
            'wall_seconds': sum(wall) / 1e3, 'cpu_seconds': sum(cpu) / 1e3,
            'wall_median_ms': statistics.median(wall), 'wall_max_ms': max(wall),
            'cpu_median_ms': statistics.median(cpu)}
        groups = collections.defaultdict(list)
        for r in mine:
            groups[operation_of(r['instance'])].append(int(r['wall_ns']) / 1e6)
        for operation, values in groups.items():
            per_operation[operation][tool] = (statistics.median(values), max(values))
    for operation in OPERATIONS:
        cells = per_operation[operation]
        full_lines.append(f'| `{SMT_NAME[operation]}` | {cells["z3"][0]:.1f} | {cells["z3"][1]:.1f} | '
                          f'{cells["cvc5"][0]:.1f} | {cells["cvc5"][1]:.1f} |')
    table('solvers-full-by-operation.md', full_lines)

# The translator's self-test (not part of the benchmark).
selftest_path = os.path.join(R, 'selftest-instances.csv')
if os.path.exists(selftest_path):
    rows = read('selftest-instances.csv')
    groups = collections.defaultdict(lambda: [0, 0])
    formats = collections.defaultdict(lambda: [0, 0])
    for row in rows:
        function, eb, sb = row['operation'].rsplit('-', 2)
        groups[function][0] += 1
        groups[function][1] += row['proved'] == '1'
        formats[f'{eb}/{sb}'][0] += 1
        formats[f'{eb}/{sb}'][1] += row['proved'] == '1'
    facts['selftest'] = {'instances': len(rows), 'proved': sum(r['proved'] == '1' for r in rows),
                         'not_proved': [r['instance'] for r in rows if r['proved'] != '1'],
                         'by_function': {k: v for k, v in sorted(groups.items())},
                         'by_format': {k: v for k, v in sorted(formats.items(), key=lambda item: int(item[0].split('/')[0]))}}
    solver_path = os.path.join(R, 'selftest-solvers.csv')
    if os.path.exists(solver_path):
        solver_rows = read('selftest-solvers.csv')
        for tool in sorted({r['tool'] for r in solver_rows}):
            mine = [r for r in solver_rows if r['tool'] == tool]
            facts['selftest'][tool] = {'runs': len(mine), 'sat': sum(r['answer'] == 'sat' for r in mine),
                                       'other': {r['instance']: r['answer'].replace('\n', ' ')[-140:]
                                                 for r in mine if r['answer'] != 'sat'}}
    fpexp_path = os.path.join(R, 'selftest-cvc5-fpexp.csv')
    if os.path.exists(fpexp_path):
        mine = read('selftest-cvc5-fpexp.csv')
        facts['selftest']['cvc5 --fp-exp'] = {
            'runs': len(mine), 'sat': sum(r['answer'] == 'sat' for r in mine),
            'other': {r['instance']: r['answer'][:80] for r in mine if r['answer'] != 'sat'}}
    if 'cvc5' in facts['selftest']:
        # cvc5 1.4.2 refuses formats other than binary32 and binary64 without --fp-exp ("FP term ... with type
        # whose size is eb/sb is not supported"); time_solvers.py keeps the first 120 characters of the message.
        solver_rows = read('selftest-solvers.csv')
        refused = [r['instance'] for r in solver_rows
                   if r['tool'] == 'cvc5' and r['answer'].startswith('error: (error "FP term')]
        facts['selftest']['cvc5']['refused_format'] = len(refused)
        facts['selftest']['cvc5']['other'] = {
            r['instance']: r['answer'][:80] for r in solver_rows
            if r['tool'] == 'cvc5' and r['answer'] != 'sat' and r['instance'] not in set(refused)}
    lines = ['| function | instances | proved by `float_decide` |', '| --- | ---: | ---: |']
    for function, (count, proved) in sorted(groups.items()):
        lines.append(f'| `fp.{function}` | {count} | {proved} |')
    lines.append(f'| all | {len(rows)} | {sum(r["proved"] == "1" for r in rows)} |')
    table('selftest-by-function.md', lines)
    lines = ['| format (exponent/significand bits) | instances | proved by `float_decide` |', '| --- | ---: | ---: |']
    for name, (count, proved) in facts['selftest']['by_format'].items():
        lines.append(f'| {name} | {count} | {proved} |')
    table('selftest-by-format.md', lines)

# The main run's kernel time as a series too (every instance, eight processes at a time).
add_series('lean kernel main run', {r['instance']: int(r['kernel_ns']) / 1e6 for r in main if r['proved'] == '1'})
write('fig-time.csv', ['series', 'operation', 'instances', 'median_ms', 'p05_ms', 'p95_ms', 'min_ms', 'max_ms'],
      time_rows)

# SMT-COMP 2018 next to it, for the table
published = {(r['solver'], r['operation']): r for r in read('smtcomp2018-wintersteiger-summary.csv')}
cell = {(series, operation): (median, low, high) for series, operation, _, median, low, high, _, _ in time_rows}


def value(series, operation, digits=1):
    if (series, operation) not in cell:
        return ''
    return f'{float(cell[(series, operation)][0]):.{digits}f}'


lines = ['| operation | FloatLib kernel | FloatLib whole run, batched | Z3 5.4.1 | cvc5 1.4.2 | Z3 4.7.1 (2018) | '
         'CVC4 (2018) | COLIBRI (2018) |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for operation in OPERATIONS + ['all']:
    old = [f'{1e3 * float(published[(solver, operation)]["cpu_median_s"]):.1f}'
           for solver in ('Z3 4.7.1', 'CVC4 2018-06-10', 'COLIBRI v2038')]
    name = f'`{SMT_NAME[operation]}`' if operation != 'all' else 'all'
    lines.append(f'| {name} | {value("lean kernel", operation, 2)} | {value("lean batch whole", operation)} | '
                 f'{value("z3 process wall", operation)} | {value("cvc5 process wall", operation)} | '
                 + ' | '.join(old) + ' |')
table('time-by-operation.md', lines)

with open(os.path.join(R, 'facts.json'), 'w', encoding='utf-8') as handle:
    json.dump(facts, handle, indent=1, sort_keys=True, default=str)
print(json.dumps(facts, indent=1, sort_keys=True, default=str)[:6000])
