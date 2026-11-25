"""Which generated kernel theorems Lean accepted, and with which axioms.

    python3 -I scripts/kernel_table.py KERNEL_DIR OUT_TSV OUT_FILES_TSV

Reads `KERNEL_DIR/manifest.tsv` (scripts/gen_kernel.py) and, for every module, Lean's output
`KERNEL_DIR/logs/MODULE.log` and `/usr/bin/time -v` report `KERNEL_DIR/logs/MODULE.time`.

A manifest row (one point of one counterexample file) is accepted when Lean exited with status 0,
its log has no error, and `#print axioms` lists only `propext`, `Classical.choice` and
`Quot.sound` for the four theorems of the row: `pK_eval` (the network at the point), and
`pK_PROP_in`, `pK_PROP_out`, `pK_PROP_full` (the property at the point).

OUT_TSV: one line per manifest row. OUT_FILES_TSV: one line per module (points, theorems, time,
peak memory).
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
STANDARD = {'propext', 'Classical.choice', 'Quot.sound'}
AXIOMS = re.compile(r"^'([^']+)' depends on axioms: \[([^\]]*)\]")
NO_AXIOMS = re.compile(r"^'([^']+)' does not depend on any axioms")


def seconds(text: str) -> float:
    parts = [float(p) for p in text.split(':')]
    value = 0.0
    for part in parts:
        value = value * 60 + part
    return value


def main() -> None:
    kernel_dir, out_path, files_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    manifest = list(csv.DictReader((kernel_dir / 'manifest.tsv').open(), delimiter='\t'))
    modules = {}
    for module in sorted({row['module'] for row in manifest}):
        info = {'axioms': {}, 'errors': 0, 'exit': '', 'wall': '', 'rss_kb': '', 'user': '', 'kernel_seconds': 0.0,
                'steps': 0}
        log = kernel_dir / 'logs' / f'{module}.log'
        if log.exists():
            # Lean wraps a long axiom list over several lines: join continuation lines
            text = re.sub(r',\n\s+', ', ', log.read_text(errors='replace'))
            for line in text.splitlines():
                match = AXIOMS.match(line)
                if match:
                    info['axioms'][match.group(1)] = {a.strip() for a in match.group(2).split(',') if a.strip()}
                    continue
                match = NO_AXIOMS.match(line)
                if match:
                    info['axioms'][match.group(1)] = set()
                    continue
                if re.search(r'(^|:\s)error[:(]', line) or line.startswith('error'):
                    info['errors'] += 1
                match = re.match(r'^type checking took ([0-9.]+)(ms|s)$', line)
                if match:
                    info['kernel_seconds'] += float(match.group(1)) / (1000 if match.group(2) == 'ms' else 1)
                    info['steps'] += 1
        time_file = kernel_dir / 'logs' / f'{module}.time'
        if time_file.exists():
            for line in time_file.read_text().splitlines():
                line = line.strip()
                if line.startswith('Elapsed'):
                    info['wall'] = f'{seconds(line.split()[-1]):.1f}'
                elif line.startswith('Maximum resident'):
                    info['rss_kb'] = line.split()[-1]
                elif line.startswith('User time'):
                    info['user'] = line.split()[-1]
                elif line.startswith('Exit status'):
                    info['exit'] = line.split()[-1]
        source = kernel_dir / f'{module}.lean'
        info['theorems'] = len(re.findall(r'^theorem ', source.read_text(), flags=re.M)) if source.exists() else 0
        modules[module] = info

    accepted_rows = 0
    with open(out_path, 'w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['id', 'point', 'module', 'accepted', 'in', 'out', 'full', 'axioms', 'why_not'])
        for row in manifest:
            info = modules[row['module']]
            names = [row['eval_theorem']] + [f'{row["verdict_prefix"]}_{part}' for part in ('in', 'out', 'full')]
            used = set()
            why = ''
            if info['exit'] != '0':
                why = f'lean exit {info["exit"] or "missing"}'
            elif info['errors']:
                why = f'{info["errors"]} errors in the log'
            else:
                for name in names:
                    if name not in info['axioms']:
                        why = f'no axiom report for {name}'
                        break
                    used |= info['axioms'][name]
                if not why and not used <= STANDARD:
                    why = 'axioms beyond the three standard ones'
            accepted = int(not why)
            accepted_rows += accepted
            writer.writerow([row['id'], row['point'], row['module'], accepted, row['in'], row['out'], row['full'],
                             ' '.join(sorted(used)), why])
    with open(files_path, 'w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['module', 'manifest_rows', 'theorems', 'audited_theorems', 'lean_exit', 'errors',
                         'wall_seconds', 'user_seconds', 'kernel_seconds', 'kernel_steps', 'peak_rss_kb'])
        for module, info in modules.items():
            rows = sum(row['module'] == module for row in manifest)
            writer.writerow([module, rows, info['theorems'], len(info['axioms']), info['exit'], info['errors'],
                             info['wall'], info['user'], f'{info["kernel_seconds"]:.1f}', info['steps'],
                             info['rss_kb']])
    done = sum(info['exit'] == '0' and not info['errors'] for info in modules.values())
    print(f'{len(manifest)} manifest rows, {accepted_rows} accepted; {done} of {len(modules)} modules without error; '
          f'{sum(i["theorems"] for i in modules.values())} theorems, '
          f'{sum(len(i["axioms"]) for i in modules.values())} axiom reports')


if __name__ == '__main__':
    main()
