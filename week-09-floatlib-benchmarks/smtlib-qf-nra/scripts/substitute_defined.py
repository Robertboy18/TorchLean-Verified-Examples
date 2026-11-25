"""A diagnostic variant of a goal: variables that a top-level equation defines are substituted away.

usage: python3 -I substitute_defined.py <QF_NRA root> <out dir> <benchmark path> ...

For each benchmark, while a top-level conjunct is `v = e` or `e = v` with `v` a real variable that does not occur
in `e`, every other occurrence of `v` is replaced by `e` and the conjunct is dropped:
`∃ v, v = e ∧ φ v` is equivalent to `φ e`.  The goal is then written like any other goal (same emitter, same
Boolean literal, same header) to <out dir>/lean.  This is not the translation of the benchmark run: it is used to see
whether a goal that `interval` refuses is proved once the defined variables are gone.

Each variant is tested against the original translation at random rational points of the variables that
remain: with the removed variables set to the values of their definitions, every conjunct that is kept must have
the same truth value before and after the substitution.
"""
from __future__ import annotations

import os
import random
import re
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smt2lean as S
import crosscheck as X
from analyze import HEADER


def occurs(p: S.Problem, node, var) -> bool:
    stack, seen = [node], set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if cur is var:
            return True
        stack.extend(p.kids(cur))
    return False


def replace(p: S.Problem, node, var, value, memo: dict):
    """`node` with every occurrence of the variable node `var` replaced by `value`."""
    got = memo.get(id(node))
    if got is not None:
        return got
    kind = node[0]
    if node is var:
        out = value
    elif kind in ("add", "mul", "sub", "and", "or"):
        out = p.mk(kind, tuple(replace(p, k, var, value, memo) for k in node[1]))
    elif kind in ("neg", "not"):
        out = p.mk(kind, replace(p, node[1], var, value, memo))
    elif kind in ("div", "imp", "iff", "xor"):
        out = p.mk(kind, replace(p, node[1], var, value, memo), replace(p, node[2], var, value, memo))
    elif kind in ("ite", "bite"):
        out = p.mk(kind, *(replace(p, k, var, value, memo) for k in node[1:]))
    elif kind == "cmp":
        out = p.mk("cmp", node[1], replace(p, node[2], var, value, memo), replace(p, node[3], var, value, memo))
    else:
        out = node
    memo[id(node)] = out
    return out


def substitute(p: S.Problem):
    """The conjuncts after the substitutions, the index of each among the original conjuncts, and the list of
    (variable index, defining node) with every definition written in the variables that remain."""
    flat = S.Emitter(p, [], [])
    conjuncts = [k for a in p.asserts for k in flat.flat(a, "and")]
    origin = list(range(len(conjuncts)))
    removed = []
    while True:
        found = None
        for i, c in enumerate(conjuncts):
            if c[0] == "cmp" and c[1] == "=":
                for var, value in ((c[2], c[3]), (c[3], c[2])):
                    if var[0] == "var" and not occurs(p, value, var):
                        found = (i, var, value)
                        break
            if found:
                break
        if not found:
            return conjuncts, origin, removed
        i, var, value = found
        memo: dict = {}
        conjuncts = [replace(p, c, var, value, memo) for j, c in enumerate(conjuncts) if j != i]
        origin = [o for j, o in enumerate(origin) if j != i]
        removed = [(v, replace(p, e, var, value, memo)) for v, e in removed] + [(var[1], value)]


def evaluate_real(node, point: dict) -> Fraction:
    kind = node[0]
    if kind == "num":
        return node[1]
    if kind == "var":
        return point[node[1]]
    if kind == "add":
        return sum((evaluate_real(k, point) for k in node[1]), Fraction(0))
    if kind == "mul":
        out = Fraction(1)
        for k in node[1]:
            out *= evaluate_real(k, point)
        return out
    if kind == "sub":
        out = evaluate_real(node[1][0], point)
        for k in node[1][1:]:
            out -= evaluate_real(k, point)
        return out
    if kind == "neg":
        return -evaluate_real(node[1], point)
    if kind == "div":
        den = evaluate_real(node[2], point)
        return evaluate_real(node[1], point) / den if den != 0 else Fraction(0)
    raise S.Unsupported(f"real node {kind}")


def main():
    root, out = sys.argv[1:3]
    os.makedirs(os.path.join(out, "lean"), exist_ok=True)
    names = []
    for rel in sys.argv[3:]:
        text = open(os.path.join(root, rel), encoding="utf-8", errors="replace").read()
        p = S.Problem().load(text)
        reals0, bools0 = p.used()
        before = S.Emitter(p, reals0, bools0, bool_literal="half")
        old_conjuncts = [k for a in p.asserts for k in before.flat(a, "and")]
        conjuncts, origin, removed = substitute(p)
        gone = {v for v, _ in removed}
        q = S.Problem()
        q.intern, q.real_names, q.bool_names = p.intern, p.real_names, p.bool_names
        q.asserts = conjuncts
        reals, bools = q.used()
        emitter = S.Emitter(q, reals, bools, bool_literal="half")
        ident = re.sub(r"[^A-Za-z0-9_.-]", "_", rel[:-5].replace("/", "__"))
        comment = (f"SMT-LIB 2025 QF_NRA/{rel} (status {p.status}), diagnostic variant: "
                   + ", ".join(p.real_names[v] for v in sorted(gone)) + " substituted by their defining equations. "
                   "Variables: " + ", ".join(f"{emitter.rname[v]} = {p.real_names[v]}" for v in reals)
                   + "".join(f", {emitter.bname[v]} = {p.bool_names[v]} (Bool as 0/1)" for v in bools) + ".")
        with open(os.path.join(out, "lean", ident + ".lean"), "w", encoding="utf-8") as f:
            f.write(HEADER.format(comment=comment.replace("-/", "- /"), goal=emitter.goal()))
        names.append(ident + ".lean")
        # Test at points of the variables that remain, with each removed variable set to the value of its
        # definition: a removed equation holds, and a kept conjunct has the same truth value before and after.
        old_text = [X.LeanText(before.prop(c)) for c in old_conjuncts]
        new_text = [X.LeanText(emitter.prop(c)) for c in conjuncts]
        dropped = [i for i in range(len(old_conjuncts)) if i not in set(origin)]
        rng = random.Random(rel)
        bad = points = 0
        for _ in range(24):
            value = {v: rng.choice(X.VALUES) for v in reals0 if v not in gone}
            full = dict(value)
            for v, e in removed:
                full[v] = evaluate_real(e, value)
            flags = {b: rng.random() < 0.5 for b in bools0}
            old_point = {before.rname[v]: full[v] for v in reals0}
            old_point.update({before.bname[b]: Fraction(int(flags[b])) for b in bools0})
            new_point = {emitter.rname[v]: value[v] for v in reals}
            new_point.update({emitter.bname[b]: Fraction(int(flags[b])) for b in bools})
            olds = [text.run(old_point)[0] for text in old_text]
            news = [text.run(new_point)[0] for text in new_text]
            points += 1
            if not all(olds[i] for i in dropped) or [olds[i] for i in origin] != news:
                bad += 1
        print(f"{rel}: {len(reals0)} -> {len(reals)} real variables, {len(old_conjuncts)} -> {len(conjuncts)} "
              f"conjuncts, removed {[p.real_names[v] for v, _ in removed]}; {points} test points, "
              f"{bad} that disagree")
    with open(os.path.join(out, "names.list"), "w") as f:
        f.write("\n".join(names) + "\n")


def _vars(p, node):
    out, stack, seen = set(), [node], set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if cur[0] == "var":
            out.add(cur[1])
        stack.extend(p.kids(cur))
    return out


if __name__ == "__main__":
    main()
