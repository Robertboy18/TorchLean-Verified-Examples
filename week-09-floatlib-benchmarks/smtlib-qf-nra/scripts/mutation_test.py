"""Does the differential test of `crosscheck.py` notice a wrong translation?  It is run on altered Lean goals.

usage: python3 -I scripts/mutation_test.py <QF_NRA root> <candidates.jsonl> <dir with lean/> <scratch dir>

For five selected benchmarks, the goal file is altered in one place after the conjuncts of the Boolean variables
(an `∧` for the first `∨`, the first `¬` dropped, a `<` for the first `≤`, an `∨` for the first `∧` between
conjuncts, the second conjunct dropped), written to the scratch directory, and checked against the SMT-LIB
source like a real translation.  Every altered goal must be reported, as a mismatch or as a goal body that is
no longer the conjunction of the expected conjuncts.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crosscheck as X

root, candidates, base, scratch = sys.argv[1:5]
PATHS = ["hycomp/ball_count_2d_slope.01.seq_lazy_lemmas_global_0.smt2",
         "hycomp/simple_ballistics_reach.01.seq_lazy_linear_enc_global_0.smt2",
         "hycomp/etcs_braking_2.01.redlog_global_0.smt2",
         "hong/hong_5.smt2",
         "20200911-Pine/1599121980156583000.smt2"]
KINDS = ["or->and", "drop-not", "le->lt", "and->or", "drop-conjunct"]


def mutate(text: str, kind: str) -> str:
    start = text.index("theorem goal")
    head, body = text[:start], text[start:]
    cut = body.rfind("= 0 ∨")                      # the last conjunct `b = 0 ∨ b = 1` of a Boolean variable
    cut = body.index("\n", cut) if cut >= 0 else 0
    kept, rest = body[:cut], body[cut:]
    if kind == "or->and":
        i = rest.index(" ∨ ")
        rest = rest[:i] + " ∧ " + rest[i + 3:]
    elif kind == "drop-not":
        i = rest.index("(¬ ")
        rest = rest[:i] + "(" + rest[i + 3:]
    elif kind == "le->lt":
        i = rest.index(" ≤ ")
        rest = rest[:i] + " < " + rest[i + 3:]
    elif kind == "and->or":
        i = rest.index(" ∧\n")
        rest = rest[:i] + " ∨\n" + rest[i + 3:]
    elif kind == "drop-conjunct":
        i = rest.index(" ∧\n")
        j = rest.index(" ∧\n", i + 3)
        rest = rest[:i] + rest[j:]
    return head + kept + rest


rows = {}
for line in open(candidates):
    r = json.loads(line)
    if r["outcome"] == "translated":
        rows[r["path"]] = r
os.makedirs(os.path.join(scratch, "lean"), exist_ok=True)
found = total = 0
for path in PATHS:
    row = rows[path]
    text = open(os.path.join(base, row["lean_dir"], row["lean"]), encoding="utf-8").read()
    clean = X.check((root, row, base, 12, "half"))
    print(f"{path}: unaltered -> {clean['mismatch'] or clean.get('error') or 'agrees'}")
    for kind in KINDS:
        try:
            altered = mutate(text, kind)
        except ValueError:
            continue                                 # the goal has no such place
        with open(os.path.join(scratch, "lean", row["lean"]), "w", encoding="utf-8") as f:
            f.write(altered)
        out = X.check((root, dict(row, lean_dir="lean"), scratch, 12, "half"))
        total += 1
        noticed = bool(out["mismatch"]) or "error" in out
        found += noticed
        print(f"   {kind}: {(out['mismatch'][:1] or [out.get('error')])[0] if noticed else 'NOT NOTICED'}")
    os.remove(os.path.join(scratch, "lean", row["lean"]))
print(f"{found} of {total} altered goals noticed")
sys.exit(0 if found == total else 1)
