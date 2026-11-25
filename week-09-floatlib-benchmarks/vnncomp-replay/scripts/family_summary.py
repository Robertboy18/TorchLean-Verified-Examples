"""Per-family summaries of the replay, as the essay and the figures use them.

    python3 scripts/family_summary.py ANALYSIS_DIR [RESULTS_DIR]

`ANALYSIS_DIR` holds what `analyze.py` writes for every family: `summary.csv` (one row per family and
verifier), `instances-FAMILY.csv`, `refuted-FAMILY.csv` and `files-FAMILY.csv`. This script adds the rows
up by family and writes two small files into `RESULTS_DIR` (default: `results/` of this suite):

  families.csv   one row per family: the instances, the counterexample files by verdict, and the "unsat"
                 results that an exact counterexample of the same instance refutes, with what the
                 competition's two scorings did with them
  margins.csv    one row per replayed counterexample file: its family, its verdict, the rounding error
                 of the binary32 output and the margin of the property in exact arithmetic

Nothing is evaluated here: every number is a count or a copy of a field of the analysis.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

TITLE = {"acasxu_2023": "ACAS Xu", "soundnessbench": "SoundnessBench", "safenlp_2024": "SafeNLP",
         "sat_relu": "SAT ReLU", "collins_rul_cnn_2022": "Collins RUL CNN", "cora_2024": "CORA benchmark",
         "malbeware": "Malbeware", "relusplitter": "ReLUSplitter", "tllverifybench_2023": "TLL Verify Bench",
         "test": "test", "cersyve": "Cersyve", "dist_shift_2023": "Dist Shift",
         "linearizenn_2024": "LinearizeNN", "lsnc_relu": "LSNC ReLU"}
CLASSES = ["exact", "projected", "rounded_box", "rounding_dependent", "rejected"]
PENALTY = "-150"          # the competition's score of an incorrect result
POINTS = "10"             # its score of a correct one


def read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(text: str) -> float:
    return float("inf") if text == "inf" else float(text)


def main() -> None:
    analysis = Path(sys.argv[1])
    results = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "results"
    results.mkdir(parents=True, exist_ok=True)
    summary = read(analysis / "summary.csv")
    families = []
    for row in summary:
        if row["family"] not in families:
            families.append(row["family"])
    rows = []
    for family in families:
        mine = [row for row in summary if row["family"] == family]
        total = lambda column: sum(int(row[column] or 0) for row in mine)       # noqa: E731
        instances = read(analysis / f"instances-{family}.csv")
        witnessed = [row for row in instances
                     if int(row["exact_witnesses"]) + int(row["projected_witnesses"]) > 0]
        rounded_only = [row for row in instances if row not in witnessed and int(row["rounded_box_witnesses"]) > 0]
        refuted_file = analysis / f"refuted-{family}.csv"
        refuted = [row for row in read(refuted_file) if row["refuted"] == "1"] if refuted_file.exists() else []
        assert len(refuted) == total("unsat_refuted"), family
        rows.append([
            family, TITLE.get(family, family), len(instances), sum(1 for row in instances if row["sat_by"]),
            len(witnessed), len(rounded_only), total("files"), total("points"),
            *[total(column) for column in CLASSES], total("files_not_replayed"), total("unsat"), len(refuted),
            total("unsat_refuted_kernel"),
            sum(row["competition_score_small"] == PENALTY for row in refuted),
            sum(row["competition_score_zero"] == PENALTY for row in refuted),
            sum(row["competition_score_zero"] == POINTS for row in refuted),
            sum(row["competition_score_small"] == "" for row in refuted)])
    header = ["family", "title", "instances", "instances_with_a_sat_result",
              "instances_with_an_exact_counterexample", "instances_with_a_rounded_box_counterexample_only",
              "files_replayed", "points_replayed", "as_reported", "moved_into_the_box", "rounded_box",
              "depends_on_rounding", "rejected", "files_not_replayed", "unsat_results", "unsat_refuted",
              "unsat_refuted_with_a_kernel_witness", "refuted_and_penalised_at_tolerance_1e-4",
              "refuted_and_penalised_at_tolerance_0", "refuted_and_given_points_at_tolerance_0",
              "refuted_and_not_scored"]
    with (results / "families.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    margins = []
    for family in families:
        path = analysis / f"files-{family}.csv"
        if not path.exists():
            continue
        for row in read(path):
            if row["out_real_slack"] in ("NA", "inf") or row["max_abs_32_minus_real"] == "NA":
                margins.append([family, row["class"], "", ""])
                continue
            margins.append([family, row["class"], f"{number(row['max_abs_32_minus_real']):.6g}",
                            f"{abs(number(row['out_real_slack'])):.6g}"])
    with (results / "margins.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["family", "verdict", "rounding_error", "margin"])
        writer.writerows(margins)
    print(results / "families.csv", len(rows), "families;", results / "margins.csv", len(margins), "files")


if __name__ == "__main__":
    main()
