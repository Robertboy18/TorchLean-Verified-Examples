"""Tables of the replay: what exact evaluation says about every published result.

    python3 -I scripts/analyze.py WORK_DIR RESULTS_DIR FAMILY...

Reads `WORK_DIR/claims-FAMILY.csv` (published results), `RESULTS_DIR/replay-FAMILY.tsv` (the
compiled Lean replay) and, when present, `RESULTS_DIR/kernel-FAMILY.tsv` (which theorems Lean's
kernel accepted) and `RESULTS_DIR/competition-{ce,score}-FAMILY.tsv` (the competition's own
scoring, scripts/competition.py), and writes

    RESULTS_DIR/files-FAMILY.csv       one row per counterexample file (a "sat" claim)
    RESULTS_DIR/instances-FAMILY.csv   one row per instance
    RESULTS_DIR/verifiers-FAMILY.csv   one row per verifier
    RESULTS_DIR/refuted-FAMILY.csv     every "unsat" on an instance with an exact counterexample
    RESULTS_DIR/table-FAMILY.md        the table in Markdown
    RESULTS_DIR/summary.csv            the verifier rows of all families given

A point is judged where it can be a counterexample: at the reported point, read to binary32, when
that lies in the input box, and otherwise at the point moved into the box (each coordinate outside
becomes the nearest binary32 number inside; in a coordinate where the box holds no binary32 number,
for example an input fixed to a decimal fraction, the nearest binary32 number to its bounds). There
the network is evaluated in binary32 in four orders of summation and in exact arithmetic, and the
output part of the property is decided on exact rationals each time. Classes of a point:

    exact               the reported point is in the box and every evaluation violates the property
    projected           the reported point is outside the box, the moved point is in the box, and
                        there every evaluation violates the property
    rounded_box         the box holds no binary32 number in some coordinate; the moved point is in
                        the box whose bounds are rounded to binary32, and there every evaluation
                        violates the property
    rounding_dependent  the evaluations disagree: whether the property is violated depends on the
                        order of summation or on rounding at all (`reference` gives the verdict of
                        the reference order, left to right)
    rejected            no evaluation violates the property
    outside             the point cannot be moved into the box (the property has no top-level
                        bounds); counted with the rejected files

A file has the best class of its points, in this order.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
TOOLS = ['alpha_beta_crown', 'cora', 'neuralsat', 'nnenum', 'nnv', 'pyrat', 'rover', 'sobolbox']
NAMES = {'alpha_beta_crown': 'alpha-beta-CROWN', 'cora': 'CORA', 'neuralsat': 'NeuralSAT', 'nnenum': 'nnenum',
         'nnv': 'NNV', 'pyrat': 'PyRAT', 'rover': 'Rover', 'sobolbox': 'SobolBox'}
ORDER = ['exact', 'projected', 'rounded_box', 'rounding_dependent', 'rejected', 'outside']
EVALUATIONS = ('out32', 'rev', 'pair', 'fma', 'out_real')


def number(text: str) -> float:
    return float('inf') if text == 'inf' else float(text)


def point_class(row):
    """(class, prefix of the columns of the deciding point)."""
    if row['in_exact'] == '1':
        base, prefix = 'exact', ''
    elif row['proj_in_exact'] == '1':
        base, prefix = 'projected', 'proj_'
    elif row['proj_in_rounded'] == '1':
        base, prefix = 'rounded_box', 'proj_'
    else:
        return 'outside', ''
    known = [row[prefix + c] for c in EVALUATIONS if row[prefix + c] != 'NA']
    if all(v == '1' for v in known):
        return base, prefix
    if all(v == '0' for v in known):
        return 'rejected', prefix
    return 'rounding_dependent', prefix


def read_tsv(path: Path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def analyze(work: Path, results: Path, family: str):
    claims = list(csv.DictReader((work / f'claims-{family}.csv').open()))
    replay = {row['id']: row for row in read_tsv(results / f'replay-{family}.tsv')}
    kernel = defaultdict(list)
    for row in read_tsv(results / f'kernel-{family}.tsv'):
        kernel[row['id']].append(row['accepted'] == '1')
    checker = {}
    for row in read_tsv(results / f'competition-ce-{family}.tsv'):
        checker[(row['scoring'], row['tool'], row['file'])] = row['checker']
    # the competition's scores: an instance is found by its name when the name is unique in the
    # family, and otherwise by its position in instances.csv when the name at that position agrees
    score_rows = read_tsv(results / f'competition-score-{family}.tsv')
    names_at = defaultdict(dict)
    for row in score_rows:
        names_at[row['scoring']][row['index']] = row['instance']
    name_count = {s: Counter(names.values()) for s, names in names_at.items()}
    by_name, by_index, truth_name, truth_index = {}, {}, {}, {}
    for row in score_rows:
        by_name[(row['scoring'], row['instance'], row['tool'])] = row
        by_index[(row['scoring'], row['index'], row['tool'])] = row
        truth_name[(row['scoring'], row['instance'])] = row['true_result']
        truth_index[(row['scoring'], row['index'])] = row['true_result']
    have_competition = bool(score_rows)

    def scored(scoring: str, claim, tool: str):
        """The scoring's line for a claim: (score, result taken as true), or ('', '')."""
        name = claim.get('score_name') or f'{claim["net"]}-{claim["prop"]}'
        if name_count.get(scoring, {}).get(name) == 1:
            row = by_name.get((scoring, name, tool))
            return (row['score'] if row else '', truth_name.get((scoring, name), ''))
        index = claim.get('index', '')
        if index != '' and names_at.get(scoring, {}).get(index) == name:
            row = by_index.get((scoring, index, tool))
            return (row['score'] if row else '', truth_index.get((scoring, index), ''))
        return ('', '')

    files = []            # one per counterexample file
    by_instance = defaultdict(dict)
    not_replayed = Counter()
    for claim in claims:
        key = (claim['net'], claim['prop'])
        by_instance[key][claim['tool']] = claim
        points = int(claim['ce_points'])
        if points == 0:
            if claim['ce_file'] not in ('none', 'empty'):
                not_replayed[(claim['tool'], claim['ce_file'][:40])] += 1
            continue
        rows = []
        for k in range(points):
            ident = f'{family}|{claim["tool"]}|{claim["net"]}|{claim["prop"]}|{k}'
            row = replay.get(ident)
            if row is None or row['status'] != 'ok':
                raise SystemExit(f'no replay line for {ident}: {row and row.get("nin")}')
            rows.append(row)
        judged = [point_class(r) for r in rows]
        classes = [c for c, _ in judged]
        best = min(classes, key=ORDER.index)
        row = rows[classes.index(best)]
        prefix = judged[classes.index(best)][1]
        kernel_ok = ''
        if kernel:
            accepted = [flag for r in rows for flag in kernel.get(r['id'], [False])]
            kernel_ok = int(all(accepted))
        name = claim.get('ce_name') or f'{claim["net"]}_{claim["prop"]}'
        files.append({
            'family': family, 'tool': claim['tool'], 'net': claim['net'], 'prop': claim['prop'],
            'result': claim['result'], 'points': points, 'class': best,
            'reference': row[prefix + 'out32'] if best != 'outside' else '0',
            'best_point': row['id'], 'decided_at': 'reported' if prefix == '' else 'projected',
            'points_exact': classes.count('exact'), 'points_projected': classes.count('projected'),
            'points_rounded_box': classes.count('rounded_box'),
            'points_rounding_dependent': classes.count('rounding_dependent'),
            'points_rejected': classes.count('rejected') + classes.count('outside'),
            'in_exact': row['in_exact'], 'in_exact_slack': row['in_exact_slack'],
            'in_rounded': row['in_rounded'], 'in_decimal': row['in_decimal'],
            'in_decimal_slack': row['in_decimal_slack'],
            'out32': row[prefix + 'out32'], 'out32_slack': row[prefix + 'out32_slack'],
            'out_real': row[prefix + 'out_real'], 'out_real_slack': row[prefix + 'out_real_slack'],
            'rev': row[prefix + 'rev'], 'rev_slack': row[prefix + 'rev_slack'],
            'pair': row[prefix + 'pair'], 'pair_slack': row[prefix + 'pair_slack'],
            'fma': row[prefix + 'fma'], 'fma_slack': row[prefix + 'fma_slack'],
            'reported_out32': row['out32'], 'reported_out32_slack': row['out32_slack'],
            'reported_out_real': row['out_real'],
            'max_abs_32_minus_real': row[prefix + 'max_abs_32_minus_real'],
            'max_abs_32_minus_reported': row['max_abs_32_minus_reported'],
            'reported_norm': row['reported_norm'], 'out_reported': row['out_reported'],
            'moved': row['moved'], 'max_move': row['max_move'], 'kernel': kernel_ok,
            'checker_small': checker.get(('small', claim['tool'], name), ''),
            'checker_zero': checker.get(('zero', claim['tool'], name), ''),
        })
    # exact counterexamples per instance, and the unsat claims on those instances
    # witnesses: counterexamples in the box; weak: counterexamples for the rounded box only, and
    # rounding-dependent files
    witnesses, weak = defaultdict(list), defaultdict(list)
    for f in files:
        if f['class'] in ('exact', 'projected'):
            witnesses[(f['net'], f['prop'])].append(f)
        elif f['class'] in ('rounded_box', 'rounding_dependent'):
            weak[(f['net'], f['prop'])].append(f)
    for f in files:
        key = (f['net'], f['prop'])
        f['instance_has_exact_witness'] = int(key in witnesses)
        f['instance_unsat_by'] = ' '.join(t for t in TOOLS if by_instance[key].get(t, {}).get('result') == 'unsat')
    if files:
        with (results / f'files-{family}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(files[0].keys()), lineterminator='\n')
            writer.writeheader()
            writer.writerows(files)
    refuted = []
    instance_rows = []
    for key, tools in sorted(by_instance.items()):
        ws = witnesses.get(key, [])
        any_claim = next(iter(tools.values()))
        verdict = ''
        if ws:
            verdict = 'sat'
        elif any(w['class'] == 'rounded_box' for w in weak.get(key, [])):
            verdict = 'sat for the rounded box'
        elif weak.get(key):
            verdict = 'depends on rounding'
        elif any(t.get('result') == 'sat' for t in tools.values()):
            verdict = 'no counterexample confirmed'
        instance_rows.append({
            'family': family, 'net': key[0], 'prop': key[1],
            'sat_by': ' '.join(t for t in TOOLS if tools.get(t, {}).get('result') == 'sat'),
            'unsat_by': ' '.join(t for t in TOOLS if tools.get(t, {}).get('result') == 'unsat'),
            'exact_witnesses': sum(w['class'] == 'exact' for w in ws),
            'projected_witnesses': sum(w['class'] == 'projected' for w in ws),
            'rounded_box_witnesses': sum(w['class'] == 'rounded_box' for w in weak.get(key, [])),
            'rounding_dependent_files': sum(w['class'] == 'rounding_dependent' for w in weak.get(key, [])),
            'exact_replay': verdict,
            'competition_small': scored('small', any_claim, any_claim['tool'])[1],
            'competition_zero': scored('zero', any_claim, any_claim['tool'])[1],
        })
        for tool, claim in tools.items():
            if claim['result'] != 'unsat' or not (ws or weak.get(key)):
                continue
            both = ws or weak[key]
            refuted.append({
                'family': family, 'net': key[0], 'prop': key[1], 'unsat_by': tool,
                'refuted': int(bool(ws)),
                'rounded_box_only': int(not ws and any(w['class'] == 'rounded_box' for w in weak[key])),
                'witnesses': ' '.join(sorted({w['tool'] + ':' + w['class'] for w in both})),
                'some_witness_as_reported': int(any(w['class'] == 'exact' for w in ws)),
                'some_witness_kernel': int(any(w['kernel'] == 1 for w in ws)) if kernel else '',
                'least_slack_binary32': min((w['out32_slack'] for w in both), key=number),
                'greatest_slack_binary32': max((w['out32_slack'] for w in both), key=number),
                'competition_score_small': scored('small', claim, tool)[0],
                'competition_score_zero': scored('zero', claim, tool)[0],
            })
    for name, rows in (('refuted', refuted), ('instances', instance_rows)):
        target = results / f'{name}-{family}.csv'
        if rows:
            with target.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), lineterminator='\n')
                writer.writeheader()
                writer.writerows(rows)
        elif target.exists():
            target.unlink()

    # per verifier
    instances = len(by_instance)
    table = []
    for tool in TOOLS:
        mine = [c for c in claims if c['tool'] == tool]
        results_count = Counter(c['result'] for c in mine)
        if results_count.get('missing', 0) == len(mine):
            continue
        my_files = [f for f in files if f['tool'] == tool]
        classes = Counter(f['class'] for f in my_files)
        sat = results_count.get('sat', 0)
        unsat = results_count.get('unsat', 0)
        my_scores = {s: [scored(s, c, tool)[0] for c in mine] for s in ('small', 'zero')}
        my_refuted = [r for r in refuted if r['unsat_by'] == tool]
        table.append({
            'family': family, 'tool': tool, 'instances': len(mine), 'sat': sat, 'unsat': unsat,
            'other': len(mine) - sat - unsat, 'files': len(my_files),
            'points': sum(f['points'] for f in my_files),
            'exact': classes.get('exact', 0), 'projected': classes.get('projected', 0),
            'rounded_box': classes.get('rounded_box', 0),
            'rounding_dependent': classes.get('rounding_dependent', 0),
            'rounding_dependent_reference_violated': sum(f['class'] == 'rounding_dependent' and f['reference'] == '1'
                                                         for f in my_files),
            'rejected': classes.get('rejected', 0) + classes.get('outside', 0),
            'rejected_instance_sat_by_another_file': sum(
                f['class'] in ('rejected', 'outside') and f['instance_has_exact_witness'] == 1 for f in my_files),
            'rejected_instance_unsat_by_others': sum(
                f['class'] in ('rejected', 'outside') and f['instance_has_exact_witness'] == 0
                and f['instance_unsat_by'] != '' for f in my_files),
            'files_not_replayed': sum(1 for c in mine if int(c['ce_points']) == 0
                                      and c['ce_file'] not in ('none', 'empty')),
            'sat_without_file': sum(1 for c in mine if c['result'] == 'sat' and c['ce_file'] in ('none', 'empty')),
            'kernel_accepted': sum(f['kernel'] == 1 for f in my_files) if kernel else '',
            'unsat_refuted': sum(r['refuted'] == 1 for r in my_refuted),
            'unsat_refuted_kernel': sum(r['refuted'] == 1 and r['some_witness_kernel'] == 1 for r in my_refuted)
            if kernel else '',
            'unsat_against_rounded_box_only': sum(r['refuted'] == 0 and r['rounded_box_only'] == 1
                                                  for r in my_refuted),
            'unsat_against_rounding_dependent_only': sum(r['refuted'] == 0 and r['rounded_box_only'] == 0
                                                         for r in my_refuted),
            'checker_small_correct': sum(f['checker_small'] == 'correct' for f in my_files) if have_competition else '',
            'checker_small_tolerance': sum(f['checker_small'] == 'correct_up_to_tolerance' for f in my_files)
            if have_competition else '',
            'checker_small_other': sum(f['checker_small'] not in ('correct', 'correct_up_to_tolerance')
                                       for f in my_files) if have_competition else '',
            'checker_zero_correct': sum(f['checker_zero'] == 'correct' for f in my_files) if have_competition else '',
            'checker_zero_tolerance': sum(f['checker_zero'] == 'correct_up_to_tolerance' for f in my_files)
            if have_competition else '',
            'penalties_small': my_scores['small'].count('-150') if have_competition else '',
            'penalties_zero': my_scores['zero'].count('-150') if have_competition else '',
        })
    with (results / f'verifiers-{family}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0].keys()), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)

    with_witness = sum(1 for r in instance_rows if r['exact_replay'] == 'sat')
    only_weak = sum(1 for r in instance_rows if r['exact_replay'] == 'depends on rounding')
    only_rounded = sum(1 for r in instance_rows if r['exact_replay'] == 'sat for the rounded box')
    truth_counts = {s: Counter(r[f'competition_{s}'] for r in instance_rows) for s in ('small', 'zero')}
    with (results / f'table-{family}.md').open('w') as out:
        out.write(f'### {family}: {instances} instances, {with_witness} with an exact binary32 counterexample\n\n')
        head = ['verifier', 'sat', 'files', 'exact', 'projected', 'rounded box', 'rounding-dependent', 'rejected',
                'kernel',
                'checker 1e-4: correct / tolerance', 'checker 0: correct / tolerance', 'unsat', 'unsat refuted',
                'penalties 1e-4 / 0']
        out.write('| ' + ' | '.join(head) + ' |\n')
        out.write('| --- |' + ' ---: |' * (len(head) - 1) + '\n')
        for t in table:
            cells = [NAMES[t['tool']], t['sat'], t['files'], t['exact'], t['projected'], t['rounded_box'],
                     t['rounding_dependent'], t['rejected'], t['kernel_accepted'] if kernel else 'not run',
                     f'{t["checker_small_correct"]} / {t["checker_small_tolerance"]}' if have_competition else '',
                     f'{t["checker_zero_correct"]} / {t["checker_zero_tolerance"]}' if have_competition else '',
                     t['unsat'], t['unsat_refuted'],
                     f'{t["penalties_small"]} / {t["penalties_zero"]}' if have_competition else '']
            out.write('| ' + ' | '.join(str(c) for c in cells) + ' |\n')
        out.write('\n')
        if have_competition:
            out.write(f'Instances with a file that the competition\'s checker calls correct: '
                      f'{truth_counts["small"].get("sat", 0)} with tolerance '
                      f'1e-4, {truth_counts["zero"].get("sat", 0)} with tolerance 0; with an exact binary32 '
                      f'counterexample: {with_witness}; with a counterexample for the rounded box only: '
                      f'{only_rounded}; with rounding-dependent files only: {only_weak}.\n\n')
    print(f'== {family}: {instances} instances, {with_witness} with an exact counterexample, {only_rounded} for '
          f'the rounded box only, {only_weak} with rounding-dependent files only, {sum(r["refuted"] for r in refuted)} unsat claims refuted; competition '
          f'sat: small {truth_counts["small"].get("sat", 0)}, zero {truth_counts["zero"].get("sat", 0)}')
    for t in table:
        print('  ', NAMES[t['tool']].ljust(17), ' '.join(f'{k}={v}' for k, v in t.items()
                                                        if k not in ('family', 'tool', 'instances')))
    for key, count in sorted(not_replayed.items()):
        print('   counterexample files not replayed:', key, count)
    return table


def main() -> None:
    work, results = Path(sys.argv[1]), Path(sys.argv[2])
    summary = []
    for family in sys.argv[3:]:
        summary += analyze(work, results, family)
    with (results / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary)


if __name__ == '__main__':
    main()
