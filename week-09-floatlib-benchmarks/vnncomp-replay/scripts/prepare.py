"""Translate the benchmark families and the published verifier results into the inputs of the replay.

    python3 -I scripts/prepare.py BENCHMARKS_DIR RESULTS_DIR WORK_DIR FAMILY...

BENCHMARKS_DIR is `.../vnncomp2025_benchmarks/benchmarks`, RESULTS_DIR the checkout of
`vnncomp2025_results`. For every FAMILY (for example `acasxu_2023`) this writes

    WORK_DIR/nets/FAMILY/NAME.net        the network, lowered (scripts/common.py)
    WORK_DIR/cases/FAMILY.cases          one block per counterexample to replay
    WORK_DIR/claims-FAMILY.csv           one row per (instance, verifier) with the published result
    WORK_DIR/untranslated-FAMILY.csv     the networks and properties that are not translated, and why

A network or property in a subfolder (`onnx/medical/perturbations_0.onnx`) is named with `_` for
`/` (`medical_perturbations_0`), which is also how the result repository names its counterexample
files. Nothing is evaluated here.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TOOLS = ['alpha_beta_crown', 'cora', 'neuralsat', 'nnenum', 'nnv', 'pyrat', 'rover', 'sobolbox']
MAX_WEIGHTS = 5_000_000      # larger networks are not replayed (time and memory of one run)


relative, find = common.relative, common.find


def main() -> None:
    benchmarks, results, work = (Path(p) for p in sys.argv[1:4])
    for family in sys.argv[4:]:
        root = benchmarks / family
        instances = []
        listed = set()
        with (root / 'instances.csv').open(newline='') as stream:
            for row in csv.reader(stream):
                if row:
                    net, prop = relative(row[0], 'onnx', '.onnx'), relative(row[1], 'vnnlib', '.vnnlib')
                    if (net, prop) in listed:
                        print(f'{family}: instance {net} {prop} is listed twice; counted once', file=sys.stderr)
                        continue
                    listed.add((net, prop))
                    instances.append((net, prop, row[2].strip()))
        nets_dir = work / 'nets' / family
        nets_dir.mkdir(parents=True, exist_ok=True)
        (work / 'cases').mkdir(parents=True, exist_ok=True)
        untranslated = []

        # networks
        shapes = {}
        for net in sorted({i[0] for i in instances}):
            path = find(root, 'onnx', net, '.onnx')
            try:
                if path is None:
                    raise ValueError('ONNX file not found')
                data = common.read_bytes(path)
                weights = common.count_weights(data)
                if weights > MAX_WEIGHTS:
                    raise ValueError(f'{weights} weights, more than the limit of {MAX_WEIGHTS}')
                nin, nout, ops, names = common.lower_onnx(data)
            except Exception as error:               # noqa: BLE001 (every refusal is recorded)
                untranslated.append(['network', net, f'{type(error).__name__}: {str(error)[:150]}'])
                continue
            common.write_net(nets_dir / f'{net.replace("/", "_")}.net', nin, nout, ops)
            shapes[net] = (nin, nout)
            print(f'net {family}/{net}: {nin} -> {nout}, ONNX nodes {"/".join(names)[:120]}', file=sys.stderr)

        # properties
        specs = {}
        for prop in sorted({i[1] for i in instances}):
            path = find(root, 'vnnlib', prop, '.vnnlib')
            try:
                if path is None:
                    raise ValueError('VNNLIB file not found')
                specs[prop] = common.parse_vnnlib(common.read_text(path))
            except Exception as error:               # noqa: BLE001
                untranslated.append(['property', prop, f'{type(error).__name__}: {str(error)[:150]}'])

        # published results
        claims = {}
        duplicates = []
        for tool in TOOLS:
            path = results / tool / 'results.csv'
            if not path.exists():
                continue
            with path.open(newline='') as stream:
                for row in csv.reader(stream):
                    # a tool's file may hold a row of another family under this family's name
                    if len(row) < 6 or row[0] != family or f'/{family}/onnx/' not in row[1].replace('/./', '/'):
                        continue
                    key = (tool, relative(row[1], 'onnx', '.onnx'), relative(row[2], 'vnnlib', '.vnnlib'))
                    if key in claims:
                        duplicates.append((key, claims[key][0], row[4]))
                        continue          # the first row of an instance is kept
                    claims[key] = (row[4], row[3], row[5])
        for key, first, second in duplicates:
            print(f'duplicate row {key}: kept {first!r}, dropped {second!r}', file=sys.stderr)

        cases_path = work / 'cases' / f'{family}.cases'
        rows = []
        with cases_path.open('w', encoding='utf-8') as cases:
            for index, (net, prop, timeout) in enumerate(instances):
                net_id, prop_id = net.replace('/', '_'), prop.replace('/', '_')
                base_net, base_prop = net.rsplit('/', 1)[-1], prop.rsplit('/', 1)[-1]
                translated = net in shapes and prop in specs
                if translated:
                    nin, nout, formula = specs[prop]
                    if (nin, nout) != shapes[net]:
                        untranslated.append(['instance', f'{net} {prop}',
                                             f'property has {nin} inputs and {nout} outputs, network {shapes[net]}'])
                        translated = False
                    else:
                        spec = ' '.join(common.formula_tokens(formula))
                for tool in TOOLS:
                    result, prepare, runtime = claims.get((tool, net, prop), ('missing', '', ''))
                    ce_name, ce_path = '', None
                    for name in dict.fromkeys([f'{net_id}_{base_prop}', f'{base_net}_{base_prop}',
                                               f'{net_id}_{prop_id}']):
                        candidate = results / tool / f'2025_{family}' / f'{name}.counterexample.gz'
                        if candidate.exists():
                            ce_name, ce_path = name, candidate
                            break
                    status, points = 'none', 0
                    if ce_path is not None:
                        text = common.read_text(ce_path)
                        if not text.strip():
                            status = 'empty'
                        elif not translated:
                            status = 'not translated'
                        else:
                            try:
                                xs, ys = common.parse_counterexample(text)
                            except ValueError as error:
                                status = f'unparsed: {error}'
                            else:
                                if len(xs) == nin and len(ys) in (0, nout):
                                    status, points = 'ok', 1
                                elif xs and len(xs) % nin == 0 and len(ys) == len(xs) // nin * nout:
                                    points = len(xs) // nin
                                    status = f'batch of {points}'
                                else:
                                    status = f'wrong shape: {len(xs)} inputs, {len(ys)} outputs'
                                for k in range(points):
                                    cases.write(f'case {family}|{tool}|{net_id}|{prop_id}|{k}\n')
                                    cases.write(f'net {nets_dir / (net_id + ".net")}\n')
                                    cases.write('x ' + ' '.join(xs[k * nin:(k + 1) * nin]) + '\n')
                                    ypart = ys[k * nout:(k + 1) * nout]
                                    cases.write('y ' + ' '.join(ypart) + '\n')
                                    cases.write('spec ' + spec + '\n')
                                    cases.write('end\n')
                    rows.append([family, net_id, prop_id, timeout, tool, result, prepare, runtime, status, points,
                                 index, f'{base_net}-{base_prop}', ce_name])
        with (work / f'claims-{family}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['family', 'net', 'prop', 'timeout', 'tool', 'result', 'prepare_time',
                             'run_time', 'ce_file', 'ce_points', 'index', 'score_name', 'ce_name'])
            writer.writerows(rows)
        with (work / f'untranslated-{family}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['kind', 'name', 'reason'])
            writer.writerows(untranslated)
        print(f'{family}: {len(instances)} instances, {len(shapes)} of {len({i[0] for i in instances})} networks and '
              f'{len(specs)} of {len({i[1] for i in instances})} properties translated, {len(rows)} claim rows, '
              f'{sum(r[9] for r in rows)} counterexample points, '
              f'{sum(r[8] == "not translated" for r in rows)} counterexample files on untranslated instances',
              file=sys.stderr)


if __name__ == '__main__':
    main()
