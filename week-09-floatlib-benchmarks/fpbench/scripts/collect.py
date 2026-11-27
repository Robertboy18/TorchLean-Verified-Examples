"""The counts of FloatLib's recorded suite runs, as the figures of `fpbench/` and `metitarski-flyspeck/` use them.

    python3 scripts/collect.py fpbench|inequalities [RESULTS_DIR]

Reads `benchmarks/results/suites/counts.csv` and `provenance.json` of a FloatLib checkout
(`FLOATLIB_DIR`, by default the copy that Lake keeps for this week, `.lake/packages/floatlib`) and
writes `counts.csv` and `run.csv` into RESULTS_DIR (by default `results/` of the suite). Every number
is copied; nothing is measured or estimated here. To make the counts again, run FloatLib's own driver,
`bash benchmarks/suites/run.sh fpbench inequalities`, in that checkout and then
`python3 benchmarks/suites/summarize.py RESULTS` (see `benchmarks/suites/README.md` there).
"""
import collections
import csv
import json
import os
import re
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
WEEK = SUITE.parent
FLOATLIB = Path(os.environ.get("FLOATLIB_DIR", WEEK / ".lake/packages/floatlib")).resolve()
DEFAULT = {"fpbench": WEEK / "fpbench/results", "inequalities": WEEK / "metitarski-flyspeck/results"}
TABLE = [
    ("fpbench-forward-conditional", "fpbench", "forward, conditional theorem", "forward, conditional theorem"),
    ("fpbench-closed-conditional", "fpbench", "closed, conditional theorem", "closed, conditional theorem"),
    ("fpbench-backward-conditional", "fpbench", "backward, conditional theorem", "backward, conditional theorem"),
    ("fpbench-forward-numerical", "fpbench", "forward, numerical bound", "forward, numerical bound"),
    ("fpbench-closed-numerical", "fpbench", "closed, numerical bound", "closed, numerical bound"),
    ("metitarski-interval", "inequalities", "metitarski, interval", "MetiTarski, interval alone"),
    ("metitarski-constraint", "inequalities", "metitarski, constraint", "MetiTarski, with the constraint engine"),
    ("flyspeck-interval", "inequalities", "flyspeck, interval", "Flyspeck, interval alone"),
    ("flyspeck-constraint", "inequalities", "flyspeck, constraint", "Flyspeck, with the constraint engine"),
]
KNOWN = {"proved", "translation_refusal", "refused", "timeout", "not_attempted", "out_of_memory",
         "proved within 20 s of process time"}


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
    which = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT[which]
    counts = collections.defaultdict(dict)
    for row in read(FLOATLIB / "benchmarks/results/suites/counts.csv"):
        counts[(row["suite"], row["group"])][row["outcome"]] = int(row["count"])
    rows = []
    for key, suite, group, label in TABLE:
        if suite != which:
            continue
        cell = counts[(suite, group)]
        assert set(cell) <= KNOWN, cell
        proved = cell["proved"]
        skipped = cell.get("translation_refusal", 0)
        failed = sum(value for outcome, value in cell.items()
                     if outcome in ("refused", "timeout", "not_attempted", "out_of_memory"))
        rows.append([key, suite, label, proved, failed, skipped, proved + failed + skipped])
    if which == "fpbench":
        assert all(row[-1] == 272 for row in rows), rows
    else:
        assert [row[-1] for row in rows] == [948, 948, 164, 164], rows
    write(out, "counts.csv", ["key", "suite", "label", "proved", "not_proved", "not_translated", "total"], rows)
    provenance = json.loads((FLOATLIB / "benchmarks/results/suites/provenance.json").read_text())[which]
    name = "FPBench" if which == "fpbench" else "inequalities"
    write(out, "run.csv", ["key", "value"],
          [["date", provenance["first_utc"][:10]], ["floatlib", "fc2fad1f"],
           ["lean", lean_version(provenance["lean_version"])],
           ["measured_utc", f"{name} {provenance['first_utc']} to {provenance['last_utc']}"]])


if __name__ == "__main__":
    main()
