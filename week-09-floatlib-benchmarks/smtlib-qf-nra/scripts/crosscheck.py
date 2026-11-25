"""Differential test of the translation: SMT-LIB source versus the emitted Lean text, at random rational points.

usage: python3 -I crosscheck.py <QF_NRA root> <candidates.jsonl> <dir with lean/ and lean-unbounded/> <out.jsonl>
           [--points N] [--workers N] [--only REGEX] [--bool-literal half|eq1]

For every translated candidate and every sampled point, two independent evaluations must agree:

* `Reference`: an interpreter of the SMT-LIB commands written for this test (it shares only the S-expression
  reader with the translator): `let` by environments, `define-fun` by evaluation, the operators by their SMT-LIB
  definition, exact arithmetic on `Fraction`.
* `LeanText`: a parser and evaluator of the subset of Lean that the translator writes (the statement of the goal
  file as it is on disk), with Lean's precedences.

Compared at each point: (1) the truth value of the conjunction of the assertions against the body under `∃`,
(2) the set of triples (left value, relation, right value) of all real comparisons, which checks the arithmetic of
both sides of every comparison even where the Boolean structure hides it, (3) the truth value of every single
assertion against the translator's rendering of that assertion, and (4) the truth value of every top-level
conjunct: the source's assertions are opened through `let`, `define-fun` and nested `and` into their conjuncts,
the goal body of the Lean file is opened along the balanced tree of `∧` that joins them, and the two lists must
agree entry by entry (after the conjuncts `b = 0 ∨ b = 1` of the Boolean variables, which must hold).  Check (4)
is what tests the Boolean structure of a benchmark that is one large assertion and false at almost every point.
Points where the source divides by zero are skipped; the set comparison (2) is skipped for benchmarks with a
real-valued `ite`, whose comparisons are split.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from fractions import Fraction
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smt2lean as S

sys.setrecursionlimit(1_000_000)


class DivZero(Exception):
    pass


# ------------------------------------------------------------------------------------------------ reference

class Reference:
    """Evaluate the assertions of an SMT-LIB script at a point, straight from the S-expressions."""

    def __init__(self, text: str) -> None:
        self.reals, self.bools, self.defs, self.asserts = [], [], {}, []
        self.real_ite = False
        for cmd in S.parse_sexprs(text):
            head = cmd[0]
            if head == "declare-fun":
                (self.reals if cmd[3] == "Real" else self.bools).append(cmd[1])
            elif head == "declare-const":
                (self.reals if cmd[2] == "Real" else self.bools).append(cmd[1])
            elif head == "define-fun":
                self.defs[cmd[1]] = ([p[0] for p in cmd[2]], cmd[4])
            elif head == "assert":
                self.asserts.append(cmd[1])

    def run(self, point: dict):
        """(truth of each assertion, set of comparison triples) at the point; raises DivZero."""
        self.point, self.atoms = point, set()
        truths, self.parts = [], []
        for a in self.asserts:
            parts = self.conjuncts(a, {}, {})       # the assertion holds when all its conjuncts hold
            truths.append(all(parts))
            self.parts += parts
        return truths, self.atoms

    def conjuncts(self, s, env, struct):
        """The truth values of the top-level conjuncts of the Boolean term `s`, in order.  `struct` maps a
        name bound by `let` or by a parameter to its defining term with the scope of that term, so that a
        conjunction behind a name is opened too."""
        if isinstance(s, str):
            if s in struct:
                term, env2, struct2 = struct[s]
                return self.conjuncts(term, env2, struct2)
            if s not in env and s not in self.point and s in self.defs and not self.defs[s][0]:
                return self.conjuncts(self.defs[s][1], {}, {})
            return [self.ev(s, env)]
        head = s[0]
        if head == "let":
            new_env, new_struct = dict(env), dict(struct)
            for name, value in s[1]:
                new_env[name] = self.ev(value, env)
                new_struct[name] = (value, env, struct)
            return self.conjuncts(s[2], new_env, new_struct)
        if head == "!":
            return self.conjuncts(s[1], env, struct)
        if head == "and":
            if len(s) == 1:
                return [True]
            return [truth for a in s[1:] for truth in self.conjuncts(a, env, struct)]
        if isinstance(head, str) and head in self.defs and head not in ("and",):
            params, body = self.defs[head]
            if params:
                return self.conjuncts(body, {p: self.ev(a, env) for p, a in zip(params, s[1:])},
                                      {p: (a, env, struct) for p, a in zip(params, s[1:])})
        if head in ("<", "<=", ">", ">=", "=") and len(s) > 3:          # a chain is a conjunction of its links
            return [self.ev([head, a, b], env) for a, b in zip(s[1:], s[2:])]
        if head == "distinct" and len(s) > 3:
            return [self.ev(["distinct", a, b], env) for i, a in enumerate(s[1:]) for b in s[i + 2:]]
        return [self.ev(s, env)]

    def ev(self, s, env):
        if isinstance(s, str):
            if s in env:
                return env[s]
            if s in self.point:
                return self.point[s]
            if s in self.defs:
                return self.ev(self.defs[s][1], {})
            if re.fullmatch(r"[0-9]+(\.[0-9]+)?", s):
                return Fraction(s)
            if s in ("true", "false"):
                return s == "true"
            raise KeyError(s)
        head = s[0]
        if head == "let":
            new = dict(env)
            for name, value in s[1]:
                new[name] = self.ev(value, env)
            return self.ev(s[2], new)
        if head == "!":
            return self.ev(s[1], env)
        if head in self.defs:
            params, body = self.defs[head]
            return self.ev(body, {p: self.ev(a, env) for p, a in zip(params, s[1:])})
        if head == "ite":
            cond, a, b = (self.ev(x, env) for x in s[1:])     # both branches, so that their comparisons count
            if not isinstance(a, bool):
                self.real_ite = True
            return a if cond else b
        v = [self.ev(a, env) for a in s[1:]]
        if head == "+":
            return sum(v, Fraction(0))
        if head == "*":
            out = Fraction(1)
            for x in v:
                out *= x
            return out
        if head == "-":
            if len(v) == 1:
                return -v[0]
            out = v[0]
            for x in v[1:]:
                out -= x
            return out
        if head == "/":
            out = v[0]
            for x in v[1:]:
                if x == 0:
                    raise DivZero
                out /= x
            return out
        if head in ("<", "<=", ">", ">="):
            ok = True
            for a, b in zip(v, v[1:]):
                self.atoms.add((a, head, b))
                ok &= {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[head]
            return ok
        if head == "=":
            if isinstance(v[0], bool):
                return all(a == b for a, b in zip(v, v[1:]))
            for a, b in zip(v, v[1:]):
                self.atoms.add((a, "=", b))
            return all(a == b for a, b in zip(v, v[1:]))
        if head == "distinct":
            if isinstance(v[0], bool):
                return len(set(v)) == len(v)
            for i, a in enumerate(v):
                for b in v[i + 1:]:
                    self.atoms.add((a, "!=", b))
            return len(set(v)) == len(v)
        if head == "not":
            return not v[0]
        if head == "and":
            return all(v)
        if head == "or":
            return any(v)
        if head == "=>":
            out = v[-1]
            for a in reversed(v[:-1]):
                out = (not a) or out
            return out
        if head == "xor":
            out = v[0]
            for a in v[1:]:
                out = out != a
            return out
        raise KeyError(head)


# ------------------------------------------------------------------------------------------------ Lean text

_LEAN_TOKEN = re.compile(r"\s+|([0-9]+)|([A-Za-z_][A-Za-z0-9_]*)|(.)", re.S)
_BIN = {"↔": (20, "n"), "→": (25, "r"), "∨": (30, "r"), "∧": (35, "r"),
        "=": (50, "n"), "≠": (50, "n"), "<": (50, "n"), "≤": (50, "n"), ">": (50, "n"), "≥": (50, "n"),
        "+": (65, "l"), "-": (65, "l"), "*": (70, "l"), "/": (70, "l")}
_REL = {"=": "=", "≠": "!=", "<": "<", "≤": "<=", ">": ">", "≥": ">="}


class LeanText:
    """Parse a term of the subset of Lean that the translator writes; evaluate it at a point."""

    def __init__(self, text: str, bool_start=None) -> None:
        # bool_start: in a goal over one vector `x : Fin n → ℝ`, the index of the first Boolean coordinate;
        # `x i` then reads as the variable x<i> (i < bool_start) or b<i - bool_start>.
        self.bool_start = bool_start
        self.toks = []
        for m in _LEAN_TOKEN.finditer(text):
            if m.group(1):
                self.toks.append(("num", Fraction(int(m.group(1)))))
            elif m.group(2):
                self.toks.append(("id", m.group(2)))
            elif m.group(3):
                self.toks.append(("sym", m.group(3)))
        self.pos = 0
        self.tree = self.expr(0)
        if self.pos != len(self.toks):
            raise SyntaxError(f"trailing tokens at {self.pos}: {self.toks[self.pos:self.pos + 5]}")

    def peek(self):
        return self.toks[self.pos] if self.pos < len(self.toks) else ("eof", None)

    def expect(self, sym):
        if self.peek() != ("sym", sym):
            raise SyntaxError(f"expected {sym} at token {self.pos}, found {self.peek()}")
        self.pos += 1

    def expr(self, min_prec):
        lhs = self.prefix()
        while True:
            kind, tok = self.peek()
            if kind != "sym" or tok not in _BIN or _BIN[tok][0] < min_prec:
                return lhs
            prec, assoc = _BIN[tok]
            self.pos += 1
            rhs = self.expr(prec if assoc == "r" else prec + 1)
            lhs = (tok, lhs, rhs)

    def prefix(self):
        kind, tok = self.peek()
        self.pos += 1
        if kind == "num":
            return ("num", tok)
        if kind == "id":
            if tok in ("True", "False"):
                return ("const", tok == "True")
            if tok == "x" and self.bool_start is not None and self.peek()[0] == "num":
                index = int(self.peek()[1])
                self.pos += 1
                return ("id", f"x{index}" if index < self.bool_start else f"b{index - self.bool_start}")
            return ("id", tok)
        if (kind, tok) == ("sym", "("):
            inner = self.expr(0)
            if self.peek() == ("sym", ":"):
                self.pos += 1
                self.expect("ℝ")
            self.expect(")")
            return inner
        if (kind, tok) == ("sym", "¬"):
            return ("¬", self.expr(40))
        if (kind, tok) == ("sym", "-"):
            return ("neg", self.expr(75))
        raise SyntaxError(f"unexpected token {tok!r} at {self.pos - 1}")

    def run(self, point: dict):
        self.point, self.atoms = point, set()
        return self.ev(self.tree), self.atoms

    def ev(self, n):
        op = n[0]
        if op in ("num", "const"):
            return n[1]
        if op == "id":
            return self.point[n[1]]
        if op == "neg":
            return -self.ev(n[1])
        if op == "¬":
            return not self.ev(n[1])
        a, b = self.ev(n[1]), self.ev(n[2])
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b if b != 0 else Fraction(0)          # Lean: x / 0 = 0
        if op in _REL:
            rel = _REL[op]
            boolean = any(side[0] == "id" and side[1].startswith("b") for side in (n[1], n[2]))
            if not boolean:
                self.atoms.add((a, rel, b))
            return {"=": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[rel]
        if op == "∧":
            return a and b
        if op == "∨":
            return a or b
        if op == "→":
            return (not a) or b
        if op == "↔":
            return a == b
        raise KeyError(op)


def leaves(tree, count: int) -> list:
    """The `count` operands that `Emitter.balanced` joined by `∧` into `tree`, in order."""
    if count == 1:
        return [tree]
    if tree[0] != "∧":
        raise SyntaxError(f"the goal body is not a balanced conjunction of {count} conjuncts")
    mid = (count + 1) // 2
    return leaves(tree[1], mid) + leaves(tree[2], count - mid)


def statement(lean_text: str) -> str:
    """The body under the binder of `theorem goal : ¬ ∃ (...), body := by` (or `¬ (body)`)."""
    start = lean_text.index("theorem goal : ¬") + len("theorem goal : ¬")
    end = lean_text.rindex(":= by")
    body = lean_text[start:end]
    if body.lstrip().startswith("∃ x : Fin"):
        body = body[body.index("→ ℝ,") + 4:]
    elif body.lstrip().startswith("∃"):
        body = body[body.index("),") + 2:]
    return body


VALUES = [Fraction(v) for v in ("-3", "-2", "-1", "-1/2", "0", "1/3", "1/2", "1", "2", "5/2")]


def check(job):
    root, row, base, points, literal = job
    out = {"path": row["path"], "points": 0, "skipped": 0, "mismatch": []}
    try:
        text = open(os.path.join(root, row["path"]), encoding="utf-8", errors="replace").read()
        ref = Reference(text)
        problem = S.Problem().load(text)
        reals, bools = problem.used()
        vector = bool(row.get("vector"))
        emitter = S.Emitter(problem, reals, bools, bool_literal=literal, vector=vector)
        names = S.Emitter(problem, reals, bools)             # x<i> and b<i>, the names of the points
        start = len(reals) if vector else None
        lean_file = open(os.path.join(base, row["lean_dir"], row["lean"]), encoding="utf-8").read()
        whole = LeanText(statement(lean_file), start)
        singles = [LeanText(emitter.prop(a), start) for a in problem.asserts]
        rng = random.Random(row["path"])
        for _ in range(points):
            rv = {name: rng.choice(VALUES) for name in ref.reals}
            bv = {name: rng.random() < 0.5 for name in ref.bools}
            lean_point = {names.rname[i]: rv[problem.real_names[i]] for i in reals}
            lean_point.update({names.bname[i]: Fraction(int(bv[problem.bool_names[i]])) for i in bools})
            try:
                truths, atoms = ref.run({**rv, **bv})
            except DivZero:
                out["skipped"] += 1
                continue
            out["points"] += 1
            truth, lean_atoms = whole.run(lean_point)
            if truth != all(truths):
                out["mismatch"].append("truth of the goal body")
            expected = [True] * len(bools) + ref.parts
            if expected:
                try:
                    parts = [whole.ev(leaf) for leaf in leaves(whole.tree, len(expected))]
                except SyntaxError as error:
                    parts = None
                    out["mismatch"].append(str(error))
                if parts is not None and parts != expected:
                    where = next(i for i, (a, b) in enumerate(zip(parts, expected)) if a != b)
                    out["mismatch"].append(f"truth of conjunct {where} of {len(expected)}")
                out["conjuncts"] = len(expected)
            if not ref.real_ite and atoms != lean_atoms:
                out["mismatch"].append(f"comparison values: {len(atoms)} in the source, {len(lean_atoms)} in Lean")
            for i, (single, expected) in enumerate(zip(singles, truths)):
                if single.run(lean_point)[0] != expected:
                    out["mismatch"].append(f"truth of assertion {i}")
            if out["mismatch"]:
                break
        out["assertions"] = len(ref.asserts)
        out["real_ite"] = ref.real_ite
    except Exception as error:                      # a failure of the test itself is reported, never ignored
        out["error"] = f"{type(error).__name__}: {error}"[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("candidates"); ap.add_argument("base"); ap.add_argument("out")
    ap.add_argument("--points", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="")
    ap.add_argument("--bool-literal", default="half")
    ap.add_argument("--skip", action="append", default=[],
                    help="a result file of an earlier run: its benchmarks are not checked again")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.candidates)]
    rows = [r for r in rows if r["outcome"] == "translated" and (not a.only or re.search(a.only, r["path"]))]
    skip = {json.loads(line)["path"] for name in a.skip for line in open(name) if line.strip()}
    rows = [r for r in rows if r["path"] not in skip]
    rows.sort(key=lambda r: r["lean_bytes"])
    jobs = [(a.root, r, a.base, a.points, a.bool_literal) for r in rows]
    bad = errors = 0
    with Pool(min(a.workers, 8), maxtasksperchild=100) as pool, open(a.out, "w") as out:
        for i, res in enumerate(pool.imap_unordered(check, jobs, chunksize=4)):
            out.write(json.dumps(res) + "\n")
            bad += bool(res["mismatch"])
            errors += "error" in res
            out.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} mismatches {bad} errors {errors}", flush=True)
    print(f"{len(jobs)} benchmarks checked: {bad} with a mismatch, {errors} test errors", flush=True)


if __name__ == "__main__":
    main()
