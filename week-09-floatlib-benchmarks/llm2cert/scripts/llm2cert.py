"""LLM2Cert (DynaCertBench) obligations as goals for `interval`.

    llm2cert.py gen OBLIGATIONS.json OUT [--tactic 'interval'] [--defaults]   one Lean file per obligation
    llm2cert.py run OUT [--jobs 6] [--timeout 300] [--only KEYFILE] check every file, write results.tsv
    llm2cert.py table OUT [RECORDS.json]                           coverage and time, against their records

The statement of each obligation is LLM2Cert's, verbatim (their locked source from the RQ1 All-Lean
records). Only the proof is ours: one call of the tactic given by `--tactic`. Every file ends with
`#print axioms`, and an obligation counts as closed only when Lean accepts the file and prints
exactly `propext`, `Classical.choice` and `Quot.sound`.

The files set `trace.interval.constraints`, so the output of each says which step of the route
closed it; `table` counts the obligations by that step. They also raise `maxHeartbeats` to
4,000,000 and `maxRecDepth` to 100,000 (the raised limits of the comparison);
`gen --defaults` leaves Lean's defaults for both, which is what a call of `interval` in an
ordinary file gets.

Nothing here is used by the library: this is the glue between their benchmark and the general
constraint route of `interval`.
"""
import argparse, collections, json, os, re, statistics, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HEADER = """import FloatLib.Tactic.Interval.Constraints

namespace LeanBarrier
abbrev State (n : ℕ) := Fin n → ℝ
/-- LLM2Cert's definition, from `LeanBarrier/Lyapunov.lean` of their formalization. -/
def Lyapunov.RadiallyUnbounded {n : ℕ} (V : State n → ℝ) : Prop :=
  ∀ K : ℝ, ∃ R : ℝ, ∀ x : State n, R ≤ (∑ i : Fin n, (x i) ^ 2) → K ≤ V x
end LeanBarrier

set_option maxHeartbeats 4000000
set_option maxRecDepth 100000
set_option trace.interval.constraints true
"""
LIMITS = "set_option maxHeartbeats 4000000\nset_option maxRecDepth 100000\n"
AXIOMS = "depends on axioms: [propext, Classical.choice, Quot.sound]"


def base_of(key):
    return re.sub(r'[^A-Za-z0-9]+', '_', key)


def lean_file(src, proof, header=HEADER):
    """Their definitions and statement, then our proof."""
    start = src.index('theorem verify_')
    theorem = src[start:].strip()
    name = re.match(r'theorem (\S+)', theorem).group(1)
    statement = theorem.split(':=')[0].rstrip()
    body = '\n'.join('  ' + line for line in proof.strip('\n').split('\n'))
    return f"{header}\n{src[:start]}\n{statement} := by\n{body}\n\n#print axioms {name}\n"


def gen(args):
    obligations = json.load(open(args.obligations))
    os.makedirs(args.out, exist_ok=True)
    index = []
    header = HEADER.replace(LIMITS, '') if args.defaults else HEADER
    for key, src in sorted(obligations.items()):
        base = base_of(key)
        open(os.path.join(args.out, base + '.lean'), 'w').write(lean_file(src, args.tactic, header))
        barrier = re.search(r'def barrier_([0-9a-f]{12})', src).group(1)
        number, kind = key.split('|')[1].split('-', 1)
        index.append({'key': key, 'base': base, 'kind': kind, 'cert': key.split('|')[0] + ':' + barrier,
                      'condition': f'{barrier}:{number}:{kind}'})
    json.dump(index, open(os.path.join(args.out, 'index.json'), 'w'), indent=1)
    print(f'{len(index)} files in {args.out}')


def check(out, base, timeout, lean):
    path = os.path.join(out, base + '.lean')
    start = time.time()
    try:
        done = subprocess.run([lean, path], capture_output=True, text=True, timeout=timeout)
        code, text = done.returncode, done.stdout + done.stderr
    except subprocess.TimeoutExpired as expired:
        code, text = 124, (expired.stdout or b'').decode(errors='replace') + '\n(timeout)'
    seconds = time.time() - start
    open(os.path.join(out, base + '.out'), 'w').write(text)
    ok = code == 0 and 'error' not in text and AXIOMS in text
    return base, code, seconds, ok


def run(args):
    index = json.load(open(os.path.join(args.out, 'index.json')))
    bases = [entry['base'] for entry in index]
    if args.only:
        wanted = {base_of(line.strip()) for line in open(args.only) if line.strip()}
        bases = [base for base in bases if base in wanted]
    lean = args.lean or 'lean'
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(lambda base: check(args.out, base, args.timeout, lean), bases):
            rows.append(row)
    with open(os.path.join(args.out, 'results.tsv'), 'a' if args.only else 'w') as results:
        for base, code, seconds, ok in rows:
            results.write(f'{base}\t{code}\t{seconds:.2f}\t{"yes" if ok else "no"}\n')
    print(f'{sum(row[3] for row in rows)}/{len(rows)} closed')


def table(args):
    index = json.load(open(os.path.join(args.out, 'index.json')))
    ours = {}
    for line in open(os.path.join(args.out, 'results.tsv')):
        base, code, seconds, ok = line.rstrip('\n').split('\t')
        ours[base] = (ok == 'yes', float(seconds))
    theirs = {}
    if args.records:
        for record in json.load(open(args.records)):
            for condition in record['conditions']:
                if not condition.get('active'):
                    continue
                passed = bool((condition.get('formal_result') or {}).get('passed'))
                old = theirs.get(condition['condition_id'])
                if old is None or (passed and not old[0]):
                    theirs[condition['condition_id']] = (
                        passed, (condition.get('timing') or {}).get('wall_seconds'),
                        condition.get('total_tokens') or 0)
    kinds = collections.defaultdict(lambda: [0, 0, 0])
    certs = collections.defaultdict(list)
    rows = []
    for entry in index:
        mine = ours.get(entry['base'], (False, 0.0))
        other = theirs.get(entry['condition'], (False, None, 0))
        rows.append({**entry, 'ours': mine[0], 'ours_s': mine[1], 'theirs': other[0], 'theirs_s': other[1],
                     'theirs_tokens': other[2]})
        kinds[entry['kind']][0] += 1
        kinds[entry['kind']][1] += mine[0]
        kinds[entry['kind']][2] += other[0]
        certs[entry['cert']].append((mine[0], other[0]))
    json.dump(rows, open(os.path.join(args.out, 'compare.json'), 'w'), indent=1)
    closed = [row for row in rows if row['ours']]
    print(f'obligations {len(rows)}: ours {len(closed)}, theirs {sum(row["theirs"] for row in rows)}, '
          f'both {sum(row["ours"] and row["theirs"] for row in rows)}')
    if closed:
        times = [row['ours_s'] for row in closed]
        print(f'ours: median {statistics.median(times):.1f} s, mean {statistics.mean(times):.1f} s per closed '
              f'obligation, {sum(row["ours_s"] for row in rows) / 60:.1f} Lean-minutes over all')
    print('kind: ours / theirs / total')
    for kind, (total, mine, other) in sorted(kinds.items(), key=lambda item: -item[1][0]):
        print(f'  {kind:28s} {mine:3d} / {other:3d} / {total:3d}')
    print(f'certificates {len(certs)}: ours {sum(all(m for m, _ in v) for v in certs.values())}, '
          f'theirs {sum(all(t for _, t in v) for v in certs.values())}')
    closing, steps = collections.Counter(), collections.Counter()
    for row in closed:
        path = os.path.join(args.out, row['base'] + '.out')
        trace = re.findall(r'\[interval\.constraints\] ([^\n]*)', open(path).read()) if os.path.exists(path) else []
        last = [line for line in trace if line.startswith('closed by')]
        closing[STEP(last[-1]) if last else 'the route of single inequalities'] += 1
        for step in {STEP(line) for line in trace if not line.startswith('closed by')}:
            steps[step] += 1
    print('closed by (the last step of the route, from the trace of each file):')
    for step, count in closing.most_common():
        print(f'  {count:3d}  {step}')
    print('obligations closed in which the route also took the step:')
    for step, count in steps.most_common():
        print(f'  {count:3d}  {step}')


def STEP(line):
    """The kind of a trace line of the route, without its numbers and terms."""
    line = re.sub(r'^closed by ', '', line)
    kinds = [('a certificate of', 'a certificate of boxes'), ('derived the bounds', 'bounds derived from comparisons'),
             ('combination with', 'a combination of two comparisons'), ('the search stopped', 'a search that stopped'),
             ('a sum of nonnegative terms', 'terms of a sum that must vanish'),
             ('cleared the denominators', 'denominators cleared'), ('slopes for', 'slopes of function values')]
    for prefix, kind in kinds:
        if line.startswith(prefix):
            return kind + (' with rational endpoints' if 'rational endpoints' in line else '')
    if 'is pinned to' in line:
        return 'an atom pinned to a value'
    if 'becomes a variable' in line or line.startswith('Real.'):
        return 'a bounded value of an argument without bounds'
    return re.sub(r'^\d+ ', '', line)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    g = commands.add_parser('gen')
    g.add_argument('obligations'); g.add_argument('out'); g.add_argument('--tactic', default='interval')
    g.add_argument('--defaults', action='store_true', help="keep Lean's default maxHeartbeats and maxRecDepth")
    r = commands.add_parser('run')
    r.add_argument('out'); r.add_argument('--jobs', type=int, default=6)
    r.add_argument('--timeout', type=int, default=300); r.add_argument('--only'); r.add_argument('--lean')
    t = commands.add_parser('table')
    t.add_argument('out'); t.add_argument('records', nargs='?')
    args = parser.parse_args()
    {'gen': gen, 'run': run, 'table': table}[args.command](args)


if __name__ == '__main__':
    main()
