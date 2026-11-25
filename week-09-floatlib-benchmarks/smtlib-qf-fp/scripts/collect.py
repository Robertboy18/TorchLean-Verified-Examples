#!/usr/bin/env python3
"""collect.py: the outcome of a directory of Lean batch files written by qffp2lean.py.

Reads DIR/index.jsonl and, next to every batch file, its `.out` (Lean's output with one `QFFP|...` line per
theorem), `.time` (wall, user, system seconds and peak resident kilobytes) and `.rc` (Lean's exit code).
Writes one CSV row per theorem and prints a summary per family and operation.

A theorem counts as proved when Lean printed its `QFFP` line with a kernel time (the kernel accepted the proof
again), its axioms are among `propext`, `Classical.choice` and `Quot.sound`, and Lean reported no error between
the theorem's first line and the next theorem.

Usage: collect.py DIR OUT.csv [SUMMARY.json]
"""
import collections
import csv
import json
import os
import re
import statistics
import sys

STANDARD = {'propext', 'Classical.choice', 'Quot.sound'}


def main():
    directory, out_path = sys.argv[1], sys.argv[2]
    summary_path = sys.argv[3] if len(sys.argv) > 3 else None
    index = collections.defaultdict(list)
    with open(os.path.join(directory, 'index.jsonl'), encoding='utf-8') as handle:
        for line in handle:
            record = json.loads(line)
            index[record['file']].append(record)
    rows, files = [], []
    for file_name in sorted(index):
        records = index[file_name]
        path = os.path.join(directory, file_name)
        checks, errors, file_clock = {}, [], None
        missing = not os.path.exists(path + '.rc')
        if not missing:
            with open(path + '.out', encoding='utf-8', errors='replace') as handle:
                for line in handle:
                    if line.startswith('QFFPCLOCK|'):
                        file_clock = int(line.split('|')[1])
                    elif line.startswith('QFFP|'):
                        fields = line.rstrip('\n').split('|')
                        name, time, axioms, clock = fields[1:5]
                        parts = fields[5].split() if len(fields) > 5 else ['', '']
                        start, stop = clock.split()
                        checks[name] = (time, axioms.split(',') if axioms else [], int(start), int(stop), parts)
                    else:
                        match = re.match(r'^(.*?):(\d+):(\d+): (error|warning): (.*)$', line)
                        if match:
                            errors.append((int(match.group(2)), match.group(4), match.group(5)))
            with open(path + '.time', encoding='utf-8') as handle:
                wall, user, system, rss = handle.read().split()[-4:]
            with open(path + '.rc', encoding='utf-8') as handle:
                rc = int(handle.read().strip())
            files.append({'file': file_name, 'wall': float(wall), 'user': float(user), 'system': float(system),
                          'rss_kb': int(rss), 'rc': rc, 'theorems': len(records)})
        previous_stop = file_clock
        for position, record in enumerate(records):
            first = record['line']
            last = records[position + 1]['line'] if position + 1 < len(records) else 10 ** 9
            mine = [(kind, text) for line, kind, text in errors if first <= line < last]
            check = checks.get(record['theorem'])
            proved = (check is not None and check[0] != 'FAILED' and set(check[1]) <= STANDARD
                      and not any(kind == 'error' for kind, _ in mine))
            parts = record['instance'].split('/')
            family = parts[0]
            operation = parts[1] if len(parts) > 2 else ''
            elaboration = ''
            if check is not None and previous_stop is not None:
                elaboration = check[2] - previous_stop
            if check is not None:
                previous_stop = check[3]
            rows.append({
                'instance': record['instance'], 'theorem': record['theorem'], 'family': family, 'operation': operation,
                'status': record['status'], 'claims': record['claims'], 'proved': int(proved),
                'kernel_ns': check[0] if check else '', 'axioms': ' '.join(check[1]) if check else '',
                'kernel_statement_ns': check[4][0] if check else '',
                'kernel_auxiliary_ns': check[4][1] if check else '',
                'elaboration_ns': elaboration, 'file': file_name,
                'message': ' / '.join(text for _, text in mine)[:300] if mine else ('' if not missing else 'not run'),
            })
    with open(out_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['family'], row['operation'])].append(row)
    summary = {'files': len(files), 'file_wall_seconds': sum(f['wall'] for f in files),
               'file_user_seconds': sum(f['user'] for f in files), 'file_system_seconds': sum(f['system'] for f in files),
               'peak_rss_kb': max((f['rss_kb'] for f in files), default=0),
               'nonzero_rc': [f['file'] for f in files if f['rc'] != 0], 'groups': []}
    print(f'{"family":14s} {"operation":11s} {"theorems":>8s} {"proved":>7s} {"not":>5s} {"kernel s":>9s} '
          f'{"median ms":>9s} {"max ms":>8s} {"elab s":>8s}')
    for (family, operation), group in sorted(groups.items()):
        kernel = [int(r['kernel_ns']) for r in group if r['kernel_ns'] not in ('', 'FAILED') and r['proved']]
        elaboration = [int(r['elaboration_ns']) for r in group if r['elaboration_ns'] != '']
        entry = {'family': family, 'operation': operation, 'theorems': len(group),
                 'proved': sum(r['proved'] for r in group),
                 'kernel_seconds': sum(kernel) / 1e9,
                 'kernel_median_ms': statistics.median(kernel) / 1e6 if kernel else None,
                 'kernel_max_ms': max(kernel) / 1e6 if kernel else None,
                 'elaboration_seconds': sum(elaboration) / 1e9}
        summary['groups'].append(entry)
        print(f'{family[:14]:14s} {operation:11s} {entry["theorems"]:8d} {entry["proved"]:7d} '
              f'{entry["theorems"] - entry["proved"]:5d} {entry["kernel_seconds"]:9.2f} '
              f'{(entry["kernel_median_ms"] or 0):9.3f} {(entry["kernel_max_ms"] or 0):8.2f} '
              f'{entry["elaboration_seconds"]:8.1f}')
    proved = sum(r['proved'] for r in rows)
    kernel_total = sum(int(r['kernel_ns']) for r in rows if r['kernel_ns'] not in ('', 'FAILED') and r['proved'])
    summary.update({'theorems': len(rows), 'proved': proved, 'kernel_seconds_proved': kernel_total / 1e9})
    print(f'total: {len(rows)} theorems, {proved} proved, {len(rows) - proved} not; kernel {kernel_total / 1e9:.2f} s; '
          f'{len(files)} files, wall {summary["file_wall_seconds"]:.1f} s, user {summary["file_user_seconds"]:.1f} s, '
          f'system {summary["file_system_seconds"]:.1f} s, peak RSS {summary["peak_rss_kb"] / 1e6:.2f} GB, '
          f'files with a nonzero exit code: {len(summary["nonzero_rc"])}')
    axioms = collections.Counter(r['axioms'] for r in rows)
    print('axioms:', dict(axioms))
    summary['axioms'] = dict(axioms)
    if summary_path:
        with open(summary_path, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=1, sort_keys=True)
    for row in rows:
        if not row['proved']:
            print('NOT PROVED', row['instance'], row['claims'], row['kernel_ns'], row['axioms'], row['message'][:160])


if __name__ == '__main__':
    main()
