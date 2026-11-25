"""Medians and spread of the comparison run, and the checks that make its series comparable.

    python3 -I summarize_comparison.py RUN_DIR OUT_DIR

Reads the rows that the benchmark executables printed (`RUN_DIR/raw/trial-NN.csv` for FloatLib's own
series, `RUN_DIR/public/trial-NN.csv` for the two series that `make_harness.py` adds) and writes

  OUT_DIR/compiled-trials.csv   one row per trial and cell: nanoseconds per operation
  OUT_DIR/compiled.csv          one row per cell: median, minimum, maximum, quartiles, MAD, trials
  OUT_DIR/compiled-ratios.csv   per cell, the ratio within each trial of FloatLib's public operation to
                                every other series: median, smallest and largest over the trials
  OUT_DIR/compiled-checks.txt   the agreement of every FloatLib cell with MPFR (and with the host
                                FPU and SoftFloat at 32 and 64 bits), the backends, the trial order

A time is `totalNanos / iterations` of one row, both printed by the executable (Lean's
`IO.monoNanosNow`, C's `clock_gettime`). Nothing is timed here.
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

run, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
OPERATIONS = ["add", "sub", "mul", "div", "sqrt", "fma"]


def series_of(row):
    """The series of a row: which implementation, and for FloatLib which path and observation."""
    implementation, klass = row["implementation"], row["executionClass"]
    if implementation == "ExecFloat":
        if klass == "proved-software-word-token":
            return "FloatLib public"
        if klass == "proved-kernel-word-token":
            return "FloatLib word kernel"
        if int(row["totalBits"]) <= 64:
            return "FloatLib word kernel, harness observation"
        return "FloatLib public"          # the harness's own FloatLib series above 64 bits: `ExecFloat.add`, ...
    return {"MPFR": "MPFR", "Native C": "native FPU", "Berkeley SoftFloat": "SoftFloat"}[implementation]


trials = defaultdict(list)          # (series, bits, operation) -> [(trial, ns per operation)]
meta = {}                           # (series, bits, operation) -> row of trial 1
agreement = defaultdict(dict)       # (bits, operation) -> series -> set of (count, sink, trace)
files = sorted(run.glob("raw/trial-*.csv")) + sorted(run.glob("public/trial-*.csv"))
for path in files:
    trial = int(path.stem.split("-")[1])
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (series_of(row), int(row["totalBits"]), row["operation"])
            trials[key].append((trial, int(row["totalNanos"]) / int(row["iterations"])))
            meta.setdefault(key, row)
            agreement[key[1:]].setdefault(key[0], set()).add(
                (int(row["agreementIterations"]), int(row["agreementSink"]) % 2 ** 64,
                 int(row["agreementFixtureTraceDigest"]) % 2 ** 64))

with (out / "compiled-trials.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["series", "bits", "operation", "trial", "nanoseconds", "iterations"])
    for key in sorted(trials, key=lambda k: (k[0], k[1], OPERATIONS.index(k[2]))):
        for trial, value in sorted(trials[key]):
            writer.writerow([*key, trial, f"{value:.4f}", meta[key]["iterations"]])

with (out / "compiled.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["series", "bits", "operation", "trials", "median_ns", "min_ns", "max_ns", "q1_ns",
                     "q3_ns", "mad_ns", "spread_percent", "iterations", "format", "backend"])
    for key in sorted(trials, key=lambda k: (k[0], k[1], OPERATIONS.index(k[2]))):
        values = sorted(value for _, value in trials[key])
        median = statistics.median(values)
        quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [median] * 3
        mad = statistics.median(abs(value - median) for value in values)
        writer.writerow([*key, len(values), f"{median:.3f}", f"{values[0]:.3f}", f"{values[-1]:.3f}",
                         f"{quartiles[0]:.3f}", f"{quartiles[2]:.3f}", f"{mad:.3f}",
                         f"{100 * (values[-1] - values[0]) / median:.1f}", meta[key]["iterations"],
                         meta[key]["format"], meta[key]["backend"]])

# Ratios within a trial: FloatLib's public operation over each other series of the same cell.
with (out / "compiled-ratios.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["numerator", "denominator", "bits", "operation", "trials", "median_ratio", "min_ratio",
                     "max_ratio"])
    for key in sorted(trials, key=lambda k: (k[0], k[1], OPERATIONS.index(k[2]))):
        if key[0] == "FloatLib public":
            continue
        ours = dict(trials.get(("FloatLib public", key[1], key[2]), []))
        ratios = sorted(ours[trial] / value for trial, value in trials[key] if trial in ours)
        if ratios:
            writer.writerow(["FloatLib public", key[0], key[1], key[2], len(ratios),
                             f"{statistics.median(ratios):.4f}", f"{ratios[0]:.4f}", f"{ratios[-1]:.4f}"])

lines = []
bad = 0
cells = 0
for (bits, operation), by_series in sorted(agreement.items(), key=lambda item: (item[0][0],
                                                                                 OPERATIONS.index(item[0][1]))):
    reference = by_series.get("MPFR")
    for name, values in sorted(by_series.items()):
        cells += 1
        if len(values) != 1 or reference is None or values != reference:
            bad += 1
            lines.append(f"DISAGREE {bits} {operation} {name}: {sorted(values)} against MPFR {reference}")
lines.insert(0, f"agreement prefix (256 steps: dependency sink and fixture trace) equal to MPFR's in "
                f"{cells - bad} of {cells} series cells, over {len(files)} trial files")
counts = defaultdict(set)
for key, values in trials.items():
    counts[key[0]].add(len(values))
for name in sorted(counts):
    cells_of = sorted({key[1] for key in trials if key[0] == name})
    lines.append(f"{name}: widths {cells_of}, trials per cell {sorted(counts[name])}")
lines.append("backends (FloatLib rows; the kernel named by each row, per series and width):")
kinds = defaultdict(set)
for key in meta:
    if key[0].startswith("FloatLib"):
        kinds[key[:2]].add(" ".join(meta[key]["backend"].split(" ")[-2:]))
for (name, bits), names in sorted(kinds.items()):
    lines.append(f"  {name} | {bits} bits | {', '.join(sorted(names))}")
(out / "compiled-checks.txt").write_text("\n".join(lines) + "\n")
print(lines[0])
print(f"{len(trials)} cells, {sum(len(v) for v in trials.values())} rows; disagreements: {bad}")
sys.exit(1 if bad else 0)
