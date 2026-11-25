"""Medians of the kernel timing runs: the files behind the figures of `kernel-vs-coq/` and
`kernel-vs-native-decide/`.

    python3 scripts/collect.py RAW_TIMING_DIR...

One directory per run of `run_kernel_sections.py`. Lean's kernel times are the median per cell over
the runs, with the fastest and the slowest run; the Coq times are the recorded run
`benchmarks/results/figures/recorded/coq-flocq.csv` of the FloatLib checkout (`FLOATLIB_DIR`, by default
`.lake/packages/floatlib` of this week). Writes `kernel-coq.csv` and `run.csv` into `results/` of this
suite, and `kernel-native.csv` and `run.csv` into `../kernel-vs-native-decide/results/`. Every number is
copied from a log or a result file; nothing is measured or estimated here.
"""
import collections
import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
WEEK = SUITE.parent
FLOATLIB = Path(os.environ.get("FLOATLIB_DIR", WEEK / ".lake/packages/floatlib")).resolve()
COQ = SUITE / "results"
NATIVE = WEEK / "kernel-vs-native-decide/results"


def write(folder, name, header, rows):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(folder / name, len(rows), "rows")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def lean_version(text):
    return re.search(r"version (\d+\.\d+\.\d+)", text).group(1)


def main():
    raws = [Path(raw) for raw in sys.argv[1:]]
    lean = collections.defaultdict(list)
    bits_of = {}
    for raw in raws:
        for row in read(raw / "suite/matched/matched.csv"):
            assert float(row["lean_kernel_us"]) > 0, row
            lean[(row["format"], row["operation"])].append(float(row["lean_kernel_us"]))
            bits_of[(row["format"], row["operation"])] = int(row["bits"])
    recorded = read(FLOATLIB / "benchmarks/results/figures/recorded/coq-flocq.csv")
    rows = []
    for row in recorded:
        key = (row["format"], row["operation"])
        values = sorted(lean[key])
        assert len(values) == len(raws) and bits_of[key] == int(row["bits"]), key
        rows.append([row["operation"], row["bits"], f"{statistics.median(values):.3f}", row["flocq_vm_us"],
                     row["flocq_lazy_us"], row["values"], row["date"], f"{values[0]:.3f}", f"{values[-1]:.3f}",
                     len(values)])
    assert len(rows) == len(lean), (len(rows), len(lean))
    write(COQ, "kernel-coq.csv", ["operation", "bits", "lean_kernel_us", "flocq_vm_us", "flocq_lazy_us",
                             "coq_values", "coq_date", "lean_min_us", "lean_max_us", "lean_runs"], rows)
    times = collections.OrderedDict()
    for raw in raws:
        for line in (raw / "exact/native.log").read_text().splitlines():
            found = re.match(r"KCMD native/(.*)\|(kernel|native) ns=(\d+) errors=(true|false)$", line)
            if found:
                statement, method, nanos, errors = found.groups()
                times.setdefault(statement, {"kernel": [], "native": []})[method].append(
                    None if errors == "true" else int(nanos) / 1e6)
    out = []
    for statement, cell in times.items():
        assert len(cell["kernel"]) == len(raws) and all(cell["kernel"]), statement
        failed = [value is None for value in cell["native"]]
        assert all(failed) or not any(failed), statement
        kernel_values = sorted(cell["kernel"])
        native_values = [] if all(failed) else sorted(cell["native"])
        out.append([statement, f"{statistics.median(kernel_values):.3f}",
                    f"{statistics.median(native_values):.3f}" if native_values else "",
                    f"{kernel_values[0]:.3f}", f"{kernel_values[-1]:.3f}",
                    f"{native_values[0]:.3f}" if native_values else "",
                    f"{native_values[-1]:.3f}" if native_values else "", len(raws)])
    write(NATIVE, "kernel-native.csv", ["statement", "kernel_ms", "native_decide_ms", "kernel_min_ms", "kernel_max_ms",
                                "native_min_ms", "native_max_ms", "runs"], out)
    facts = [json.loads((raw / "facts.json").read_text()) for raw in raws]
    assert len({fact["lean"] for fact in facts}) == 1
    run = [["date", facts[0]["date"]], ["floatlib", "4.1.0"], ["lean", facts[0]["lean"]], ["runs", len(raws)],
           ["core_scheduling_cookie", all(fact["core_scheduling_cookie"] for fact in facts)],
           ["started_utc", " ".join(fact["started_utc"] for fact in facts)],
           ["finished_utc", " ".join(fact["finished_utc"] for fact in facts)],
           ["loadavg_before", "; ".join(fact["loadavg_before"] for fact in facts)],
           ["loadavg_after", "; ".join(fact["loadavg_after"] for fact in facts)],
           ["machine_idle_percent_before", " ".join(str(fact["machine_idle_percent_before"]) for fact in facts)],
           ["machine_idle_percent_after", " ".join(str(fact["machine_idle_percent_after"]) for fact in facts)],
           ["sections", " | ".join(fact["sections"] for fact in facts)]]
    for folder in (COQ, NATIVE):
        write(folder, "run.csv", ["key", "value"], run)



if __name__ == "__main__":
    main()
