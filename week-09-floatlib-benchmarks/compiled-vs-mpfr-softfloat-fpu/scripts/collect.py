"""The published files of the comparison run: the four series of the figures, and the facts of the run.

    python3 scripts/collect.py RUN_DIR SUMMARY_DIR [RESULTS_DIR]

RUN_DIR is the output directory of `run_comparison.py` (with `RUN_DIR-launch.json` and
`RUN_DIR-load.csv` beside it) and SUMMARY_DIR the output of `summarize_comparison.py RUN_DIR SUMMARY_DIR`.
Writes into RESULTS_DIR (by default `results/` of this suite) `compiled.csv`, `compiled-trials.csv`,
`compiled-ratios.csv` and `compiled-checks.txt` with the series that the figures draw (FloatLib's
public operations, MPFR, SoftFloat and the host FPU), and `compiled-run.csv` with the facts of the run
that the captions quote. Every number is copied; nothing is measured or estimated here.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
SERIES = ("FloatLib public", "MPFR", "SoftFloat", "native FPU")


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
    run, summary = Path(sys.argv[1]), Path(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else SUITE / "results"
    for name, column in (("compiled.csv", "series"), ("compiled-trials.csv", "series")):
        rows = read(summary / name)
        write(out, name, list(rows[0]), [list(row.values()) for row in rows if row[column] in SERIES])
    rows = read(summary / "compiled-ratios.csv")
    write(out, "compiled-ratios.csv", list(rows[0]),
          [list(row.values()) for row in rows if row["numerator"] == SERIES[0] and row["denominator"] in SERIES])
    checks = (summary / "compiled-checks.txt").read_text().splitlines()
    (out / "compiled-checks.txt").write_text(
        "\n".join(line for line in checks if "word kernel" not in line) + "\n")
    metadata = dict(line.split("=", 1) for line in (run / "metadata.txt").read_text().splitlines() if "=" in line)
    launch = json.loads(Path(f"{run}-launch.json").read_text())
    load = read(f"{run}-load.csv")
    column = lambda name: [float(row[name]) for row in load]
    facts = [
        ["date", launch["started_utc"][:10]], ["floatlib", "fc2fad1f"], ["started_utc", launch["started_utc"]],
        ["finished_utc", launch["finished_utc"]], ["seconds", launch["seconds"]], ["trials", metadata["runs"]],
        ["lean", lean_version(metadata["leanVersion"])], ["mpfr", metadata["mpfr"]],
        ["softfloat_commit", metadata["softfloatCommit"]], ["compiler", metadata["compiler"]],
        ["cflags", metadata["cflags"]], ["mathlib_commit", metadata["mathlibCommit"]],
        ["cpu", launch["cpu"]], ["core_threads", " ".join(map(str, launch["core_threads"]))],
        ["core_scheduling_cookie", launch["core_scheduling_cookie"]],
        ["target_nanos", metadata["targetMeasuredNanos"]], ["widths", metadata["widths"]],
        ["loadavg_before", " ".join(launch["loadavg_before"])], ["loadavg_after", " ".join(launch["loadavg_after"])],
        ["load_samples", len(load)],
        ["machine_idle_median", f"{statistics.median(column('machine_idle')):.3f}"],
        ["machine_idle_min", f"{min(column('machine_idle')):.3f}"],
        ["machine_idle_max", f"{max(column('machine_idle')):.3f}"],
        ["machine_niced_median", f"{statistics.median(column('machine_niced')):.3f}"],
        ["machine_not_niced_median", f"{statistics.median(column('machine_not_niced')):.3f}"],
        ["cpu_not_niced_median", f"{statistics.median(column('cpu_not_niced')):.3f}"],
        ["cpu_niced_max", f"{max(column('cpu_niced')):.3f}"],
        ["cpu_niced_mean", f"{statistics.mean(column('cpu_niced')):.4f}"],
        ["sibling_busy_max", f"{max(1 - value for value in column('sibling_idle')):.3f}"],
        ["sibling_busy_mean", f"{statistics.mean(1 - value for value in column('sibling_idle')):.4f}"],
        ["load1_min", min(column("load1"))], ["load1_median", statistics.median(column("load1"))],
        ["load1_max", max(column("load1"))],
    ]
    write(out, "compiled-run.csv", ["key", "value"], facts)



if __name__ == "__main__":
    main()
