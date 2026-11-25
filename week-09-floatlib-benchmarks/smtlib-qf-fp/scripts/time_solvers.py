#!/usr/bin/env python3
"""time_solvers.py: run tools on instances one process at a time and time every process the same way.

Each run is one process started with the instance's file as its last argument. The wall time is the monotonic
clock around the process (start to exit), the CPU time is the user plus system time of the process and its
children (`getrusage(RUSAGE_CHILDREN)` before and after), and the answer is

* for a solver, the first line of its output that is `sat`, `unsat` or `unknown`;
* for Lean (a tool named `lean`), the status claimed by the theorem that Lean accepted in the instance's file:
  a `QFFP|...` line with a kernel time and no axiom beyond `propext`, `Classical.choice` and `Quot.sound`.

In every repetition the tools run in a rotated order on each instance, so that no tool is always first.

Usage:
  time_solvers.py ROOT LIST OUT.csv --reps 3 --timeout 60 --tool NAME=COMMAND ... [--lean-single DIR]
                  [--lean-batch DIR --batch-out BATCH.csv]

  NAME=COMMAND      the command, split on spaces; the instance's `.smt2` path is appended. The tool `lean` gets
                    the one-instance Lean file of the instance from `--lean-single DIR` (DIR/index.jsonl).
  --lean-batch DIR  after the instances, in every repetition, also run `lean` once on each batch file of DIR
                    and write one row per file to BATCH.csv (wall, CPU, accepted theorems, kernel nanoseconds).
  --smt-batch DIR   and run every solver once on each batch script of DIR (the same instances, one process for
                    the file), with one row per solver and file in `--smt-batch-out`.
  --own-core        pin everything to the least busy physical core, with a core-scheduling cookie: no other task
                    then runs on the core's second hardware thread while a process is timed.
"""
import argparse
import csv
import ctypes
import json
import os
import resource
import subprocess
import threading
import time

STANDARD = {'propext', 'Classical.choice', 'Quot.sound'}


# ---------------------------------------------------------------------------------------------------------------
# A core of its own: the least busy physical core, with a core-scheduling cookie, so that no other task runs
# on its second hardware thread while a process is timed (as FloatLib's benchmarks/plots/timing.py does).

def cpu_times():
    """Per CPU, and for the machine under the key -1: user, nice, system, idle, iowait, irq, softirq, steal."""
    table = {}
    with open('/proc/stat', encoding='ascii') as handle:
        for line in handle:
            if line.startswith('cpu'):
                fields = line.split()
                key = -1 if fields[0] == 'cpu' else int(fields[0][3:])
                table[key] = [int(value) for value in fields[1:9]]
    return table


def shares(before, after):
    """(not niced, niced, idle) shares of a CPU between two samples."""
    delta = [b - a for a, b in zip(before, after)]
    total = sum(delta) or 1
    user, nice, system, idle, iowait, irq, softirq, steal = delta
    return (user + system + irq + softirq + steal) / total, nice / total, (idle + iowait) / total


def siblings(cpu):
    with open(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list', encoding='ascii') as handle:
        text = handle.read().strip()
    members = []
    for part in text.split(','):
        low, _, high = part.partition('-')
        members.extend(range(int(low), int(high or low) + 1))
    return members


def own_core(out, seconds=5.0):
    """Pin this process to the least busy core and give it a core-scheduling cookie. Returns the facts."""
    try:
        # 62 is PR_SCHED_CORE, 1 PR_SCHED_CORE_CREATE, and the scope 1 is the thread group.
        cookie = ctypes.CDLL(None, use_errno=True).prctl(62, 1, 0, 1, 0) == 0
    except (OSError, AttributeError):
        cookie = False
    allowed = os.sched_getaffinity(0)
    before = cpu_times()
    time.sleep(seconds)
    after = cpu_times()
    best = None
    for cpu in sorted(allowed):
        group = siblings(cpu)
        if cpu != group[0] or not all(member in allowed for member in group):
            continue
        work = sum(shares(before[member], after[member])[0] for member in group)
        niced = sum(shares(before[member], after[member])[1] for member in group)
        key = (round(work + niced, 2), work)
        if best is None or key < best[0]:
            best = (key, work, niced, cpu, group)
    _, work, niced, cpu, group = best
    os.sched_setaffinity(0, {cpu})
    machine = shares(before[-1], after[-1])
    facts = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'cpu': cpu, 'core_threads': group,
             'core_scheduling_cookie': cookie, 'core_not_niced_share_before': round(work, 4),
             'core_niced_share_before': round(niced, 4),
             'machine_shares_before': dict(zip(('not_niced', 'niced', 'idle'), (round(v, 4) for v in machine))),
             'loadavg_before': open('/proc/loadavg', encoding='ascii').read().split()[:3], 'nice': os.nice(0)}
    with open(out + '.launch.json', 'w', encoding='utf-8') as handle:
        json.dump(facts, handle, indent=1)

    def sample():
        with open(out + '.load.csv', 'w', encoding='utf-8') as stream:
            stream.write('utc,machine_not_niced,machine_niced,machine_idle,cpu_not_niced,cpu_niced,cpu_idle,'
                         'sibling_not_niced,sibling_niced,sibling_idle,load1\n')
            previous = cpu_times()
            while True:
                time.sleep(10)
                current = cpu_times()
                row = [time.strftime('%H:%M:%S', time.gmtime())]
                for key in (-1, cpu, group[-1]):
                    row += [f'{value:.4f}' for value in shares(previous[key], current[key])]
                row.append(open('/proc/loadavg', encoding='ascii').read().split()[0])
                stream.write(','.join(row) + '\n')
                stream.flush()
                previous = current

    threading.Thread(target=sample, daemon=True).start()
    return facts


def run(command, timeout):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic_ns()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               start_new_session=True)
    killed = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        killed = True
        os.killpg(process.pid, 9)
        output, _ = process.communicate()
    stop = time.monotonic_ns()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    return stop - start, int(cpu * 1e9), process.returncode, killed, output.decode('utf-8', errors='replace')


def accepted(output):
    """The theorems that Lean accepted, with their kernel nanoseconds, from the `QFFP|...` lines."""
    found = {}
    for line in output.splitlines():
        if line.startswith('QFFP|'):
            name, nanoseconds, axioms = line.split('|')[1:4]
            if nanoseconds != 'FAILED' and set(axioms.split(',')) - {''} <= STANDARD:
                found[name] = int(nanoseconds)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('list')
    parser.add_argument('out')
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--tool', action='append', required=True)
    parser.add_argument('--lean-single')
    parser.add_argument('--lean-batch')
    parser.add_argument('--batch-out')
    parser.add_argument('--smt-batch', help='a directory of solver batch scripts (make_smt_batches.py)')
    parser.add_argument('--smt-batch-out')
    parser.add_argument('--own-core', action='store_true',
                        help='pin to the least busy core with a core-scheduling cookie; facts in OUT.launch.json, '
                             'the load during the run in OUT.load.csv')
    arguments = parser.parse_args()
    if arguments.own_core:
        print(json.dumps(own_core(arguments.out)), flush=True)

    tools = []
    for item in arguments.tool:
        name, command = item.split('=', 1)
        tools.append((name, command.split()))
    lean_command = dict(tools).get('lean', ['lean', '-DmaxErrors=1000000'])
    with open(arguments.list, encoding='utf-8') as handle:
        instances = [line.strip() for line in handle if line.strip()]
    single_file, claims = {}, {}
    if arguments.lean_single:
        with open(os.path.join(arguments.lean_single, 'index.jsonl'), encoding='utf-8') as handle:
            for line in handle:
                record = json.loads(line)
                single_file[record['instance']] = os.path.join(arguments.lean_single, record['file'])
                claims[record['theorem']] = record['claims']
    batch_files = []
    if arguments.lean_batch:
        batch_files = sorted(os.path.join(arguments.lean_batch, name) for name in os.listdir(arguments.lean_batch)
                             if name.endswith('.lean'))

    smt_batches = []
    if arguments.smt_batch:
        smt_batches = sorted(os.path.join(arguments.smt_batch, name) for name in os.listdir(arguments.smt_batch)
                             if name.endswith('.smt2'))
    smt_handle = open(arguments.smt_batch_out, 'w', newline='', encoding='utf-8') if smt_batches else None
    smt_writer = csv.writer(smt_handle) if smt_handle else None
    if smt_writer:
        smt_writer.writerow(['tool', 'file', 'repetition', 'wall_ns', 'cpu_ns', 'exit_code', 'instances', 'answers',
                             'agree_with_status'])

    with open(arguments.out, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['tool', 'instance', 'repetition', 'answer', 'wall_ns', 'cpu_ns', 'exit_code', 'timed_out',
                         'kernel_ns'])
        batch_handle = open(arguments.batch_out, 'w', newline='', encoding='utf-8') if batch_files else None
        batch_writer = csv.writer(batch_handle) if batch_handle else None
        if batch_writer:
            batch_writer.writerow(['file', 'repetition', 'wall_ns', 'cpu_ns', 'exit_code', 'accepted', 'kernel_ns'])
        for repetition in range(arguments.reps):
            for position, instance in enumerate(instances):
                shift = (position + repetition) % len(tools)
                for name, command in tools[shift:] + tools[:shift]:
                    kernel = ''
                    if name == 'lean':
                        wall, cpu, code, killed, output = run(command + [single_file[instance]], arguments.timeout)
                        proved = accepted(output)
                        statuses = sorted({claims[theorem] for theorem in proved})
                        answer = statuses[0] if len(statuses) == 1 else 'rejected' if not statuses else 'both'
                        kernel = sum(proved.values())
                    else:
                        wall, cpu, code, killed, output = run(command + [os.path.join(arguments.root, instance)],
                                                              arguments.timeout)
                        answer = ''
                        for line in output.splitlines():
                            if line.strip() in ('sat', 'unsat', 'unknown'):
                                answer = line.strip()
                                break
                        if not answer:
                            answer = 'timeout' if killed else 'error: ' + output.strip().replace('\n', ' ')[:120]
                    writer.writerow([name, instance, repetition, answer, wall, cpu, code, int(killed), kernel])
            handle.flush()
            for path in batch_files:
                wall, cpu, code, killed, output = run(lean_command + [path], 3600)
                proved = accepted(output)
                batch_writer.writerow([os.path.basename(path), repetition, wall, cpu, code, len(proved),
                                       sum(proved.values())])
                with open(f'{path}.rep{repetition}.out', 'w', encoding='utf-8') as copy:
                    copy.write(output)
                batch_handle.flush()
            for position, path in enumerate(smt_batches):
                with open(os.path.splitext(path)[0] + '.expected', encoding='utf-8') as expected_handle:
                    expected = expected_handle.read().split()
                solvers = [tool for tool in tools if tool[0] != 'lean']
                shift = (position + repetition) % len(solvers)
                for name, command in solvers[shift:] + solvers[:shift]:
                    wall, cpu, code, killed, output = run(command + [path], 3600)
                    answers = [line.strip() for line in output.splitlines()
                               if line.strip() in ('sat', 'unsat', 'unknown')]
                    agree = sum(1 for answer, status in zip(answers, expected)
                                if status == 'unknown' or answer == status)
                    smt_writer.writerow([name, os.path.basename(path), repetition, wall, cpu, code, len(expected),
                                         len(answers), agree])
                smt_handle.flush()
        if batch_handle:
            batch_handle.close()
        if smt_handle:
            smt_handle.close()
    print(f'{len(instances)} instances, {len(tools)} tools, {arguments.reps} repetitions -> {arguments.out}')


if __name__ == '__main__':
    main()
