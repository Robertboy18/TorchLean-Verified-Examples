#!/usr/bin/env python3
"""harvest_examples.py LEANCERT_DIR OUT_JSON

Collect the statements that LeanCert's own examples and tests prove with one tactic call.

A declaration is taken when all of this holds (the same rule for every file):

* it is `theorem NAME : S := by T`, `lemma NAME : S := by T` or `example : S := by T` at column 0,
  with no binders before the colon;
* the proof `T` is one line and starts with a LeanCert tactic;
* it is not under `#guard_msgs` (those are expected failures), and `T` has no `fail_if_success`;
* `S` names nothing of LeanCert (`LeanCert`, `Expr.`, `IntervalRat`, `IntervalDyadic`, ...), so that
  the same statement can be given to another tactic.  Statements that still fail to elaborate in
  a fresh file (they use a definition of their file) are dropped later, by the runner's records.

Equal statements are kept once.  The goal records its sources, LeanCert's call as written, and
the `open` commands of its file without LeanCert's own namespaces (a statement that needs one
names something of LeanCert and is not taken), the same header for both tools.
"""
import json
import pathlib
import re
import sys

TACTICS = ["leancert?", "leancert", "certify_bound", "interval_decide", "interval_auto",
           "interval_bound_subdiv", "interval_bound_adaptive", "interval_bound", "bernstein_bound",
           "multivariate_bound", "opt_bound", "root_bound", "interval_roots",
           "interval_unique_root", "interval_minimize_mv", "interval_maximize_mv",
           "interval_minimize", "interval_maximize", "interval_argmin", "interval_argmax",
           "integral_exact", "finsum_bound", "interval_refute", "eventual_bound",
           "enclosure_bound"]
TRUSTED = {"leancert", "certify_bound", "interval_decide", "interval_auto",
           "interval_bound_subdiv", "bernstein_bound", "multivariate_bound", "opt_bound",
           "root_bound", "interval_roots", "interval_unique_root", "interval_minimize",
           "interval_maximize", "interval_minimize_mv", "interval_maximize_mv", "interval_argmin",
           "interval_argmax", "finsum_bound", "enclosure_bound"}
FOREIGN = re.compile(r"LeanCert|Expr\.|IntervalRat|IntervalDyadic|IntervalEnv|ADSupported|"
                     r"ExprSupported|FinBoxMem|SystemZero|QPoly|Dyadic\b|checkUpper|checkLower")
DECL = re.compile(r"^(?:private\s+|protected\s+)?(theorem|lemma|example)\b", re.M)


def shape_of(statement):
    body = statement
    while True:
        match = re.match(r"\s*∀[^,]*,", body)
        if not match:
            break
        body = body[match.end():]
    if re.match(r"\s*∀ \w+ : (Nat|ℕ)\b", statement):
        return "tail", False  # a bound for all large natural numbers: no bounded box
    if "⌈" in statement or "⌊" in statement:
        return "ceiling", False  # an inequality between natural numbers
    if "∫" in statement:
        return "integral", False
    if "∃" in statement:
        return "existence", False
    if "deriv" in statement:
        return "derivative", False
    # `interval` also takes combinations of comparisons by `∧`, `∨`, `¬`, `→` and `≠` (it refutes the
    # negation on boxes), so these two shapes are in its scope
    if "∧" in body or "∨" in body:
        return "conjunction", True
    if "≠" in body:
        return "disequality", True
    if re.search(r"≤|<|≥|>", body):
        return "bound", True
    if "=" in body:
        return "equality", False
    return "other", False


def main():
    root, out = pathlib.Path(sys.argv[1]), sys.argv[2]
    files = sorted(list((root / "LeanCert/Examples").rglob("*.lean")) +
                   list((root / "LeanCert/Test").rglob("*.lean")))
    seen = {}
    stats = {"declarations": 0, "binders": 0, "multi_tactic": 0, "other_tactic": 0,
             "guarded": 0, "foreign": 0, "taken": 0, "duplicates": 0}
    for path in files:
        text = path.read_text(encoding="utf-8")
        starts = [m.start() for m in DECL.finditer(text)]
        tops = [m.start() for m in re.finditer(r"^\S", text, re.M)]
        for start in starts:
            end = next((t for t in tops if t > start), len(text))
            block = text[start:end]
            stats["declarations"] += 1
            if ":= by" not in block:
                continue
            head, _, proof = block.partition(":= by")
            head = re.sub(r"^(?:private\s+|protected\s+)?(theorem|lemma|example)\b", "", head, 1)
            before, colon, statement = head.partition(":")
            if not colon or not re.fullmatch(r"\s*[A-Za-z_][A-Za-z0-9_.'₀-₉]*\s*|\s*", before):
                stats["binders"] += 1
                continue
            statement = " ".join(statement.split())
            lines = [line.strip() for line in proof.split("\n")]
            lines = [line for line in lines if line and not line.startswith("--")]
            if len(lines) != 1:
                stats["multi_tactic"] += 1
                continue
            call = re.sub(r"\s*--.*$", "", lines[0])
            name = next((t for t in TACTICS if call == t or call.startswith(t + " ")), None)
            if name is None or "fail_if_success" in call or "<;>" in call or ";" in call:
                stats["other_tactic"] += 1
                continue
            preceding = text[:start].rstrip().split("\n")[-3:]
            if any("#guard_msgs" in line for line in preceding):
                stats["guarded"] += 1
                continue
            if FOREIGN.search(statement):
                stats["foreign"] += 1
                continue
            relative = str(path.relative_to(root))
            line_number = text.count("\n", 0, start) + 1
            # the `open` commands of the file that precede the declaration
            opens = [line.strip() for line in text[:start].split("\n")
                     if re.match(r"open (?!.* in$)", line)]
            lc_header = "".join(line + "\n" for line in opens)
            fl_opens = []
            for line in opens:
                words = [w for w in line.split()[1:] if not w.startswith("LeanCert") and w != "scoped"]
                if words:
                    fl_opens.append("open " + " ".join(words))
            fl_header = "".join(line + "\n" for line in fl_opens)
            # the one Mathlib constant used in a statement that `import FloatLib` does not bring
            if "eulerMascheroniConstant" in statement:
                fl_header = "import Mathlib.NumberTheory.Harmonic.EulerMascheroni\n" + fl_header
            if statement in seen:
                seen[statement]["cases"].append(f"{relative}:{line_number}")
                stats["duplicates"] += 1
                continue
            base = "leancert" if name == "leancert?" else name
            native = call.replace("leancert?", "leancert", 1) if name == "leancert?" else call
            if re.search(r"\(trust := \w+\)", native):
                kernel = re.sub(r"\(trust := \w+\)", "(trust := kernel)", native)
            elif base in TRUSTED:
                kernel = native + " (trust := kernel)"
            else:
                kernel = native
            shape, in_scope = shape_of(statement)
            stats["taken"] += 1
            seen[statement] = {
                "id": f"examples.{len(seen):03d}", "suite": "examples",
                "cases": [f"{relative}:{line_number}"], "statement": statement,
                "in_scope": in_scope, "shape": shape, "lc_tactic": base, "lc_only": False,
                "note": "", "lc_header": fl_header, "fl_header": fl_header,
                "lc_native_call": native, "lc_kernel_call": kernel,
            }
    goals = list(seen.values())
    json.dump(goals, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(stats)
    shapes = {}
    for goal in goals:
        shapes[goal["shape"]] = shapes.get(goal["shape"], 0) + 1
    print("shapes", shapes)
    tactics = {}
    for goal in goals:
        tactics[goal["lc_tactic"]] = tactics.get(goal["lc_tactic"], 0) + 1
    print("tactics", tactics)


if __name__ == "__main__":
    main()
