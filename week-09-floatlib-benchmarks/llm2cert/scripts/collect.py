"""The numbers of the LLM2Cert comparison: the two runs of `interval` next to LLM2Cert's own records.

    python3 scripts/collect.py DEFAULTS_DIR RAISED_DIR LEAN_RECORDS.json [RESULTS_DIR]

DEFAULTS_DIR and RAISED_DIR are the output directories of `scripts/run.sh` (the files of
`llm2cert.py gen --defaults` and of `llm2cert.py gen`, with `results.tsv`, `index.json` and `run.txt`);
LEAN_RECORDS.json is `results/rq1/data/lean_records.json` of LLM2Cert's repository after its
`scripts/unpack_records.py`. Writes `llm2cert.csv`, `llm2cert-run.csv`, `llm2cert-kinds.csv` and the two
per-obligation files into RESULTS_DIR (by default `results/` of this suite). Every number is copied
from a log or a record; nothing is measured or estimated here.
"""
import collections
import csv
import json
import re
import statistics
import sys
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]


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
    defaults, raised, records = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else SUITE / "results"
    theirs = {}
    for record in json.loads(records.read_text()):
        for condition in record["conditions"]:
            if not condition.get("active"):
                continue
            passed = bool((condition.get("formal_result") or {}).get("passed"))
            old = theirs.get(condition["condition_id"])
            if old is None or (passed and not old[0]):
                theirs[condition["condition_id"]] = (
                    passed, (condition.get("timing") or {}).get("wall_seconds"), condition.get("total_tokens") or 0)

    def one(directory):
        directory = Path(directory)
        index = json.loads((directory / "index.json").read_text())
        ours = {}
        for line in (directory / "results.tsv").read_text().splitlines():
            base, code, seconds, ok = line.split("\t")
            ours[base] = (ok == "yes", float(seconds), int(code))
        assert len(ours) == len(index) == 291, (len(ours), len(index))
        certificates = collections.defaultdict(list)
        for entry in index:
            certificates[entry["cert"]].append(ours[entry["base"]][0])
        closed = [entry for entry in index if ours[entry["base"]][0]]
        times = [ours[entry["base"]][1] for entry in closed]
        both = sum(theirs.get(entry["condition"], (False,))[0] for entry in closed)
        quick = {entry["base"] for entry in closed if ours[entry["base"]][1] <= 300}
        quick_certificates = collections.defaultdict(list)
        for entry in index:
            quick_certificates[entry["cert"]].append(entry["base"] in quick)
        return {"closed_300": len(quick),
                "certificates_300": sum(all(v) for v in quick_certificates.values()),
                "closed": len(closed), "certificates": sum(all(v) for v in certificates.values()),
                "certificate_total": len(certificates), "seconds": sum(value[1] for value in ours.values()),
                "median_s": statistics.median(times), "max_s": max(times),
                "p90_s": statistics.quantiles(times, n=10, method="inclusive")[8],
                "timeouts": sum(value[2] == 124 for value in ours.values()),
                "both": both, "only_ours": len(closed) - both,
                "open": sorted(entry["key"] for entry in index if not ours[entry["base"]][0]),
                "set": {entry["key"] for entry in closed}, "index": index, "ours": ours}

    def hours(seconds):
        return f"{seconds / 60:.1f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"

    a, b = one(defaults), one(raised)
    index = a["index"]
    their_closed = sum(theirs.get(entry["condition"], (False,))[0] for entry in index)
    their_certificates = collections.defaultdict(list)
    for entry in index:
        their_certificates[entry["cert"]].append(theirs.get(entry["condition"], (False,))[0])
    their_complete = sum(all(v) for v in their_certificates.values())
    their_seconds = sum(theirs[entry["condition"]][1] or 0 for entry in index
                        if theirs.get(entry["condition"], (False,))[0])
    their_tokens = sum(theirs[entry["condition"]][2] for entry in index if entry["condition"] in theirs)
    rows = [
        ["mode-a-defaults", "FloatLib", "interval alone, Lean's default limits", a["closed"], a["certificates"],
         f"{a['seconds'] / 3600:.4f}", hours(a["seconds"]), 0, f"{a['median_s']:.1f}", "yes",
         "measured here, FloatLib 4.1.0"],
        ["mode-a-raised", "FloatLib", "interval alone, raised limits", b["closed"], b["certificates"],
         f"{b['seconds'] / 3600:.4f}", hours(b["seconds"]), 0, f"{b['median_s']:.1f}", "yes",
         "measured here, FloatLib 4.1.0"],
        ["llm2cert-all-lean", "LLM2Cert", "LLM2Cert All-Lean, a model writes the proofs", their_closed,
         their_complete, f"{their_seconds / 3600:.4f}", hours(their_seconds), their_tokens, "", "yes",
         "their lean_records.json (script count)"],
    ]
    write(out, "llm2cert.csv", ["key", "tool", "label", "closed", "certificates", "hours", "time_label", "tokens",
                           "median_s", "drawn", "source"], rows)
    same = a["set"] == b["set"]
    overlap = (f"Against LLM2Cert All-Lean ({their_closed}): with the default limits {a['both']} obligations are "
               f"closed by both and {a['only_ours']} only by `interval`; with the raised limits {b['both']} and "
               f"{b['only_ours']}.")
    facts = [["overlap", overlap], ["same_closed_set", same],
             ["only_raised", " ".join(sorted(b["set"] - a["set"]))],
             ["only_defaults", " ".join(sorted(a["set"] - b["set"]))],
             ["open_defaults", " ".join(a["open"])], ["open_raised", " ".join(b["open"])],
             ["timeouts_defaults", a["timeouts"]], ["timeouts_raised", b["timeouts"]],
             ["closed_300_defaults", a["closed_300"]], ["certificates_300_defaults", a["certificates_300"]],
             ["closed_300_raised", b["closed_300"]], ["certificates_300_raised", b["certificates_300"]],
             ["max_s_defaults", f"{a['max_s']:.1f}"], ["max_s_raised", f"{b['max_s']:.1f}"],
             ["p90_s_defaults", f"{a['p90_s']:.1f}"], ["p90_s_raised", f"{b['p90_s']:.1f}"],
             ["lean_minutes_defaults", f"{a['seconds'] / 60:.1f}"], ["lean_minutes_raised", f"{b['seconds'] / 60:.1f}"],
             ["their_closed", their_closed], ["their_certificates", their_complete],
             ["their_hours_on_closed", f"{their_seconds / 3600:.2f}"], ["their_tokens", their_tokens]]
    # What `scripts/run.sh` wrote about each run (`run.txt` next to the results).
    for name, directory in (("defaults", defaults), ("raised", raised)):
        text = (Path(directory) / "run.txt").read_text()
        get = lambda key: re.search(rf"^{key} (.*)$", text, re.M).group(1)
        facts += [[f"start_{name}", get("start")], [f"end_{name}", get("end")],
                  [f"wall_seconds_{name}", get("wall_seconds")],
                  [f"loadavg_start_{name}", " ".join(get("loadavg").split()[:3])],
                  [f"loadavg_end_{name}", " ".join(get("loadavg_end").split()[:3])],
                  [f"idle_percent_start_{name}", get("idle")], [f"idle_percent_end_{name}", get("idle_end")]]
        if name == "defaults":
            facts += [["date", get("start")[:10]], ["lean", lean_version(get("lean"))],
                      ["floatlib", "4.1.0"], ["workers", get("workers")]]
    write(out, "llm2cert-run.csv", ["key", "value"], facts)
    kinds = collections.defaultdict(lambda: [0, 0, 0, 0])
    for entry in index:
        cell = kinds[entry["kind"]]
        cell[0] += 1
        cell[1] += theirs.get(entry["condition"], (False,))[0]
        cell[2] += a["ours"][entry["base"]][0]
        cell[3] += b["ours"][entry["base"]][0]
    write(out, "llm2cert-kinds.csv", ["kind", "count", "llm2cert_all_lean", "interval_default_limits",
                                 "interval_raised_limits"],
          [[kind, *cell] for kind, cell in sorted(kinds.items(), key=lambda item: -item[1][0])])

    for name, directory in (("default-limits", defaults), ("raised-limits", raised)):
        lines = (Path(directory) / "results.tsv").read_text().splitlines()
        (out / f"obligations-{name}.tsv").write_text(
            "obligation\tlean_exit_code\twall_seconds\tclosed\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
