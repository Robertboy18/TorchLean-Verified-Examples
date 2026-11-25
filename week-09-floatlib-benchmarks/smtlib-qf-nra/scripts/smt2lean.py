"""SMT-LIB 2 QF_NRA benchmark -> term DAG -> (a) boundedness analysis, (b) Lean goal `¬ ∃ x, constraints`.

The benchmark files are data: they are parsed as S-expressions and never executed.

Translation (one rule per SMT-LIB construct, the same for every instance):

* Real variables that occur in the assertions become `x<i> : ℝ` bound by one `∃`.
* Bool variables that occur become `b<i> : ℝ` with the conjunct `(b<i> = 0 ∨ b<i> = 1)`; the literal `b`
  is `1 / 2 < b<i>` (option "half", the final rule) or `b<i> = 1` (option "eq1", the first rule tried).
  (`interval` reads only real comparisons as constraints, so a `Prop` or `Bool` atom would be dropped; the
  0/1 encoding is equivalent: ∃ b : Bool, φ b ⟺ ∃ r : ℝ, (r = 0 ∨ r = 1) ∧ φ (1 / 2 < r).)
* `let` and nullary/parametric `define-fun` are substituted (the term DAG is expanded to a tree in the Lean
  text); an instance whose expansion exceeds the node cap is reported as not translated.
* `+ - * /`, numerals and decimals map to the same operations on ℝ with exact rational literals.  Terms are
  kept as written: `(* x x x)` stays a product of three factors, nothing is expanded, factored or reordered.
* The operands of an n-ary `+`, `*`, `and`, `or` (nested `and`/`or` flattened) and the assertions themselves
  are joined as balanced binary trees: the same value or proposition at depth log n, since Lean's default
  recursion depth (512) does not elaborate a chain of several hundred operands nested to one side.
* `< <= > >= = distinct` map to `< ≤ > ≥ = ≠` (chains and n-ary `=`/`distinct` to conjunctions).
* `and or not => xor`, Boolean `=` and Boolean `ite` map to `∧ ∨ ¬ → ¬(· ↔ ·) ↔` and `(c → a) ∧ (¬c → b)`.
* A real-valued `ite` inside a comparison is lifted to the comparison: `(c ∧ atom[a]) ∨ (¬c ∧ atom[b])`.
* SMT-LIB leaves `x / 0` unspecified (any value), Lean defines it as 0: an unsat benchmark is unsat for every
  interpretation of division by zero, so the Lean statement follows from the SMT-LIB one.
"""
from __future__ import annotations

import math
import re
import sys
from fractions import Fraction

sys.setrecursionlimit(1_000_000)


class Unsupported(Exception):
    """The benchmark uses something the translator does not handle; the reason is the message."""


# ----------------------------------------------------------------------------------------------- S-expressions

_TOKEN = re.compile(r'''\s+|;[^\n]*|(\()|(\))|(\|[^|]*\|)|("(?:[^"]|"")*")|([^\s()|";]+)''')


def parse_sexprs(text: str) -> list:
    """All top-level S-expressions of `text` as nested lists of strings (iterative, no recursion)."""
    stack: list[list] = [[]]
    pos, n = 0, len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if m is None:
            raise Unsupported(f"lexical error at offset {pos}")
        pos = m.end()
        if m.group(1):
            stack.append([])
        elif m.group(2):
            if len(stack) == 1:
                raise Unsupported("unbalanced parenthesis")
            done = stack.pop()
            stack[-1].append(done)
        elif m.group(3):
            stack[-1].append(m.group(3))  # quoted symbol, kept with its bars
        elif m.group(4):
            stack[-1].append(m.group(4))
        elif m.group(5):
            stack[-1].append(m.group(5))
    if len(stack) != 1:
        raise Unsupported("unbalanced parenthesis at end of file")
    return stack[0]


# ------------------------------------------------------------------------------------------------- term DAG
# A node is a tuple.  Real nodes:  ("num", Fraction) ("var", i) ("add", kids) ("mul", kids) ("neg", a)
#   ("sub", kids) ("div", a, b) ("ite", c, a, b).
# Bool nodes: ("true",) ("false",) ("bvar", i) ("not", a) ("and", kids) ("or", kids) ("imp", a, b)
#   ("iff", a, b) ("xor", a, b) ("bite", c, a, b) ("cmp", op, a, b)  with op in < <= > >= = != .
# Nodes are interned so that shared subterms (from `let`) are one object, and sizes are computed on the DAG.

REAL_KINDS = {"num", "var", "add", "mul", "neg", "sub", "div", "ite"}

_NUMERAL = re.compile(r"^[0-9]+$")
_DECIMAL = re.compile(r"^[0-9]+\.[0-9]+$")


class Problem:
    def __init__(self) -> None:
        self.intern: dict = {}
        self.real_names: list[str] = []
        self.bool_names: list[str] = []
        self.globals: dict = {}          # symbol -> node (declared constants and nullary define-funs)
        self.macros: dict = {}           # symbol -> (params, body sexpr) for define-fun with parameters
        self.asserts: list = []
        self.status = "none"
        self.logic = ""
        self.tree_size: dict = {}

    # ---- construction
    def mk(self, *node):
        return self.intern.setdefault(node, node)

    def is_real(self, node) -> bool:
        return node[0] in REAL_KINDS

    def declare(self, name: str, sort) -> None:
        if name in self.globals or name in self.macros:
            raise Unsupported(f"symbol declared twice: {name}")
        if sort == "Real":
            self.globals[name] = self.mk("var", len(self.real_names))
            self.real_names.append(name)
        elif sort == "Bool":
            self.globals[name] = self.mk("bvar", len(self.bool_names))
            self.bool_names.append(name)
        else:
            raise Unsupported(f"sort {sort!r} of {name}")

    def load(self, text: str) -> "Problem":
        for cmd in parse_sexprs(text):
            if not isinstance(cmd, list) or not cmd:
                raise Unsupported("top-level atom")
            head = cmd[0]
            if head == "set-info":
                if len(cmd) >= 3 and cmd[1] == ":status":
                    self.status = cmd[2]
            elif head == "set-logic":
                self.logic = cmd[1]
            elif head in ("set-option", "check-sat", "exit", "get-model", "get-info", "get-value"):
                pass
            elif head == "declare-fun":
                if cmd[2] != []:
                    raise Unsupported("uninterpreted function with arguments")
                self.declare(cmd[1], cmd[3])
            elif head == "declare-const":
                self.declare(cmd[1], cmd[2])
            elif head == "define-fun":
                name, params, _sort, body = cmd[1], cmd[2], cmd[3], cmd[4]
                if name in self.globals or name in self.macros:
                    raise Unsupported(f"symbol defined twice: {name}")
                if params:
                    self.macros[name] = ([p[0] for p in params], body)
                else:
                    self.globals[name] = self.term(body, {})
            elif head == "assert":
                node = self.term(cmd[1], {})
                if self.is_real(node):
                    raise Unsupported("assertion of a real term")
                self.asserts.append(node)
            else:
                raise Unsupported(f"command {head}")
        return self

    # ---- terms
    def term(self, s, env: dict):
        if isinstance(s, str):
            if s in env:
                return env[s]
            if s in self.globals:
                return self.globals[s]
            if _NUMERAL.match(s):
                return self.mk("num", Fraction(int(s)))
            if _DECIMAL.match(s):
                return self.mk("num", Fraction(s))
            if s == "true":
                return self.mk("true")
            if s == "false":
                return self.mk("false")
            raise Unsupported(f"unknown symbol {s}")
        if not s:
            raise Unsupported("empty application")
        head = s[0]
        if isinstance(head, list):
            raise Unsupported("application of a compound head")
        if head == "let":
            new = dict(env)
            for name, value in s[1]:
                new[name] = self.term(value, env)       # parallel let: values see the outer scope
            return self.term(s[2], new)
        if head == "!":                                  # annotation
            return self.term(s[1], env)
        if head in self.macros:
            params, body = self.macros[head]
            if len(params) != len(s) - 1:
                raise Unsupported(f"arity of {head}")
            return self.term(body, {p: self.term(a, env) for p, a in zip(params, s[1:])})
        args = [self.term(a, env) for a in s[1:]]
        real = [self.is_real(a) for a in args]
        mk = self.mk
        if head in ("+", "*"):
            if not all(real) or not args:
                raise Unsupported(f"{head} on non-real arguments")
            return args[0] if len(args) == 1 else mk("add" if head == "+" else "mul", tuple(args))
        if head == "-":
            if not all(real) or not args:
                raise Unsupported("- on non-real arguments")
            if len(args) == 1:
                a = args[0]
                return mk("num", -a[1]) if a[0] == "num" else mk("neg", a)
            return mk("sub", tuple(args))
        if head == "/":
            if not all(real) or len(args) < 2:
                raise Unsupported("/ on non-real arguments")
            out = args[0]
            for b in args[1:]:
                if out[0] == "num" and b[0] == "num" and b[1] != 0:
                    out = mk("num", out[1] / b[1])       # a rational literal such as (/ 1 3)
                else:
                    out = mk("div", out, b)
            return out
        if head in ("<", "<=", ">", ">="):
            if not all(real) or len(args) < 2:
                raise Unsupported(f"{head} on non-real arguments")
            parts = [mk("cmp", head, a, b) for a, b in zip(args, args[1:])]
            return parts[0] if len(parts) == 1 else mk("and", tuple(parts))
        if head == "=":
            if len(args) < 2:
                raise Unsupported("= with one argument")
            if all(real):
                parts = [mk("cmp", "=", a, b) for a, b in zip(args, args[1:])]
            elif not any(real):
                parts = [mk("iff", a, b) for a, b in zip(args, args[1:])]
            else:
                raise Unsupported("= between a real and a Boolean")
            return parts[0] if len(parts) == 1 else mk("and", tuple(parts))
        if head == "distinct":
            if not all(real):
                if not any(real) and len(args) == 2:
                    return mk("xor", args[0], args[1])
                raise Unsupported("distinct on Booleans")
            parts = [mk("cmp", "!=", a, b) for i, a in enumerate(args) for b in args[i + 1:]]
            return parts[0] if len(parts) == 1 else mk("and", tuple(parts))
        if head == "ite":
            if len(args) != 3 or real[0]:
                raise Unsupported("ite shape")
            if real[1] and real[2]:
                return mk("ite", *args)
            if not real[1] and not real[2]:
                return mk("bite", *args)
            raise Unsupported("ite branches of different sorts")
        if any(real):
            raise Unsupported(f"{head} on real arguments")
        if head == "not":
            if len(args) != 1:
                raise Unsupported("not arity")
            return mk("not", args[0])
        if head in ("and", "or"):
            if not args:
                return mk("true" if head == "and" else "false")
            return args[0] if len(args) == 1 else mk(head, tuple(args))
        if head == "=>":
            if len(args) < 2:
                raise Unsupported("=> arity")
            out = args[-1]
            for a in reversed(args[:-1]):
                out = mk("imp", a, out)
            return out
        if head == "xor":
            if len(args) < 2:
                raise Unsupported("xor arity")
            out = args[0]
            for a in args[1:]:
                out = mk("xor", out, a)
            return out
        raise Unsupported(f"operator {head}")

    # ---- traversal helpers
    @staticmethod
    def kids(node):
        kind = node[0]
        if kind in ("add", "mul", "sub", "and", "or"):
            return node[1]
        if kind in ("neg", "not"):
            return (node[1],)
        if kind in ("div", "imp", "iff", "xor"):
            return (node[1], node[2])
        if kind in ("ite", "bite"):
            return (node[1], node[2], node[3])
        if kind == "cmp":
            return (node[2], node[3])
        return ()

    def size(self, node) -> int:
        """Number of nodes of the tree expansion of `node` (shared subterms counted at each use)."""
        memo = self.tree_size
        got = memo.get(id(node))
        if got is not None:
            return got
        stack = [(node, False)]
        while stack:
            cur, ready = stack.pop()
            if id(cur) in memo:
                continue
            if ready:
                memo[id(cur)] = 1 + sum(memo[id(k)] for k in self.kids(cur))
            else:
                stack.append((cur, True))
                stack.extend((k, False) for k in self.kids(cur) if id(k) not in memo)
        return memo[id(node)]

    def used(self):
        """Indices of the real and Boolean variables that occur in the assertions."""
        seen, reals, bools = set(), set(), set()
        stack = list(self.asserts)
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            if cur[0] == "var":
                reals.add(cur[1])
            elif cur[0] == "bvar":
                bools.add(cur[1])
            else:
                stack.extend(self.kids(cur))
        return sorted(reals), sorted(bools)

    def features(self) -> dict:
        seen, count = set(), {}
        stack = list(self.asserts)
        while stack:
            cur = stack.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            count[cur[0]] = count.get(cur[0], 0) + 1
            if cur[0] == "div" and cur[2][0] != "num":
                count["div_nonconst"] = count.get("div_nonconst", 0) + 1
            stack.extend(self.kids(cur))
        return count


# ------------------------------------------------------------------------------------------ boundedness analysis
# "Bounded by their constraints": interval constraint propagation (forward and backward on the monomials of each
# comparison) over the conjunction of the assertions, through `and` (sequential) and `or` (hull of the branches),
# starting from (-inf, +inf) for every real variable, reaches a finite interval for every real variable that
# occurs.  Floats suffice: only finiteness is decided, and nothing computed here enters a Lean proof.

INF = math.inf
MAX_TERMS = 20000


class TooLarge(Exception):
    pass


def poly_add(p, q, sign=1):
    out = dict(p)
    for m, c in q.items():
        v = out.get(m, 0) + sign * c
        if v:
            out[m] = v
        else:
            out.pop(m, None)
    if len(out) > MAX_TERMS:
        raise TooLarge
    return out


def poly_mul(p, q):
    if len(p) * len(q) > 4 * MAX_TERMS:
        raise TooLarge
    out: dict = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            if not m1:
                m = m2
            elif not m2:
                m = m1
            else:
                d = dict(m1)
                for v, e in m2:
                    d[v] = d.get(v, 0) + e
                m = tuple(sorted(d.items()))
            v = out.get(m, 0) + c1 * c2
            if v:
                out[m] = v
            else:
                out.pop(m, None)
    if len(out) > MAX_TERMS:
        raise TooLarge
    return out


def to_poly(node, memo):
    """Polynomial {monomial: Fraction} of a real term; monomial = sorted tuple of (var, exponent).
    Raises TooLarge when the term is not a polynomial of moderate size (division by a term, ite)."""
    got = memo.get(id(node))
    if got is not None:
        return got
    kind = node[0]
    if kind == "num":
        out = {(): node[1]} if node[1] else {}
    elif kind == "var":
        out = {((node[1], 1),): Fraction(1)}
    elif kind == "add":
        out = {}
        for k in node[1]:
            out = poly_add(out, to_poly(k, memo))
    elif kind == "sub":
        out = to_poly(node[1][0], memo)
        for k in node[1][1:]:
            out = poly_add(out, to_poly(k, memo), -1)
    elif kind == "neg":
        out = {m: -c for m, c in to_poly(node[1], memo).items()}
    elif kind == "mul":
        out = {(): Fraction(1)}
        for k in node[1]:
            out = poly_mul(out, to_poly(k, memo))
    elif kind == "div":
        den = to_poly(node[2], memo)
        if list(den.keys()) != [()]:
            raise TooLarge
        out = {m: c / den[()] for m, c in to_poly(node[1], memo).items()}
    else:
        raise TooLarge
    memo[id(node)] = out
    return out


def _mul(a, b):
    if a == 0 or b == 0:
        return 0.0
    return a * b


def imul(x, y):
    c = (_mul(x[0], y[0]), _mul(x[0], y[1]), _mul(x[1], y[0]), _mul(x[1], y[1]))
    return (min(c), max(c))


def ipow(x, e):
    lo, hi = x
    if e % 2:
        return (lo ** e if lo != -INF else -INF, hi ** e if hi != INF else INF)
    if lo >= 0:
        return (lo ** e, hi ** e if hi != INF else INF)
    if hi <= 0:
        return (hi ** e, lo ** e if lo != -INF else INF)
    top = max(-lo, hi)
    return (0.0, top ** e if top != INF else INF)


def safe(f, *a):
    try:
        return f(*a)
    except OverflowError:
        return None


def mono_range(mono, box):
    out = (1.0, 1.0)
    for v, e in mono:
        try:
            out = imul(out, ipow(box[v], e))
        except OverflowError:
            return (-INF, INF)
    return out


def widen_hi(x):
    return x + abs(x) * 1e-9 + 1e-12 if x != INF else INF


def widen_lo(x):
    return x - abs(x) * 1e-9 - 1e-12 if x != -INF else -INF


class Bottom(Exception):
    """The box is empty: the branch is infeasible."""


def tighten(box, v, lo, hi):
    """Intersect the interval of variable v with [lo, hi]; returns True when an endpoint moved."""
    cl, ch = box[v]
    nl, nh = max(cl, lo), min(ch, hi)
    if nl > nh + 1e-9 * max(1.0, abs(nl), abs(nh)):
        raise Bottom
    if nl > nh:
        nl = nh = (nl + nh) / 2
    if nl == cl and nh == ch:
        return False
    box[v] = (nl, nh)
    # report progress only when an endpoint became finite or moved noticeably
    def moved(old, new):
        if math.isinf(old):
            return not math.isinf(new)
        return abs(new - old) > 1e-6 * max(1.0, abs(old))
    return moved(cl, nl) or moved(ch, nh)


def iroot(x, e):
    """The real e-th root for odd e, monotone, with infinities kept."""
    if math.isinf(x):
        return x
    return -((-x) ** (1.0 / e)) if x < 0 else x ** (1.0 / e)


def contract_mono(mono, lo, hi, box):
    """Use `mono ∈ [lo, hi]` to tighten the variables of the monomial."""
    changed = False
    for i, (v, e) in enumerate(mono):
        rest = mono[:i] + mono[i + 1:]
        if rest:
            rl, rh = mono_range(rest, box)
            if not (rl > 0 or rh < 0):
                continue                                  # the other factors may vanish: no bound
            inv = (0.0 if math.isinf(rh) else 1.0 / rh, 0.0 if math.isinf(rl) else 1.0 / rl)
            tl, th = imul((lo, hi), inv)                  # v ** e = mono / rest
        else:
            tl, th = lo, hi
        if e % 2:
            nl, nh = iroot(tl, e), iroot(th, e)
        else:
            if th < -1e-12:
                raise Bottom
            if th == INF:
                continue
            nh = max(th, 0.0) ** (1.0 / e)
            nl = -nh
        if tighten(box, v, widen_lo(nl), widen_hi(nh)):
            changed = True
    return changed


def contract_atom(op, poly, box):
    """Tighten the box with `poly op 0` for op in < <= = (strictness is ignored: closed intervals)."""
    if op == "!=":
        return False
    try:
        return _contract_atom(op, poly, box)
    except OverflowError:
        return False


def _contract_atom(op, poly, box):
    terms = list(poly.items())
    ranges = []
    for mono, c in terms:
        c = float(c)
        if not mono:
            ranges.append((c, c))
            continue
        r = mono_range(mono, box)
        ranges.append((_mul(c, r[0]), _mul(c, r[1])) if c > 0 else (_mul(c, r[1]), _mul(c, r[0])))
    changed = False
    sides = ("le", "ge") if op == "=" else ("le",)
    for side in sides:
        # le:  sum of terms <= 0, so term_k <= -sum_{i != k} lo_i
        los = [r[0] for r in ranges] if side == "le" else [-r[1] for r in ranges]
        infinite = [i for i, x in enumerate(los) if x == -INF]
        if len(infinite) > 1:
            continue
        total = sum(x for x in los if x != -INF)
        if not infinite and total > 1e-9 * max(1.0, sum(abs(x) for x in los)):
            raise Bottom
        for k, (mono, c) in enumerate(terms):
            if not mono:
                continue
            if infinite and infinite[0] != k:
                continue
            bound = -(total - (los[k] if not infinite else 0.0))      # (±term_k) <= bound
            c = float(c)
            sc = c if side == "le" else -c
            # sc * mono <= bound
            if sc > 0:
                lo, hi = -INF, bound / sc
            else:
                lo, hi = bound / sc, INF
            mr = mono_range(mono, box)
            lo, hi = max(lo, mr[0]), min(hi, mr[1])
            if lo > hi + 1e-9 * max(1.0, abs(lo), abs(hi)):
                raise Bottom
            if contract_mono(mono, lo, hi, box):
                changed = True
    return changed


class Overlay:
    """A box that reads through to its parent and keeps its own changes: the branch of a disjunction."""
    __slots__ = ("parent", "own")

    def __init__(self, parent) -> None:
        self.parent = parent
        self.own: dict = {}

    def __getitem__(self, v):
        o = self
        while type(o) is Overlay:
            got = o.own.get(v)
            if got is not None:
                return got
            o = o.parent
        return o[v]

    def __setitem__(self, v, value) -> None:
        self.own[v] = value


class Bounder:
    def __init__(self, problem: Problem, reals) -> None:
        self.p = problem
        self.reals = reals
        self.poly_memo: dict = {}
        self.atom_poly: dict = {}
        self.skipped_atoms = 0

    def atom(self, node):
        key = id(node)
        if key not in self.atom_poly:
            try:
                lhs, rhs = (node[3], node[2]) if node[1] in (">", ">=") else (node[2], node[3])
                poly = poly_add(to_poly(lhs, self.poly_memo), to_poly(rhs, self.poly_memo), -1)
            except TooLarge:
                poly = None
                self.skipped_atoms += 1
            self.atom_poly[key] = poly
        return self.atom_poly[key]

    def refine(self, node, positive, box):
        """Tighten `box` in place with `node` (negated when not positive); returns True on progress.
        Raises Bottom when the constraint is infeasible on the box."""
        kind = node[0]
        if kind == "cmp":
            op = {">": "<", ">=": "<="}.get(node[1], node[1])      # the polynomial is lhs - rhs after the swap
            poly = self.atom(node)
            if poly is None:
                return False
            if not positive:
                if op == "=":
                    return False                      # a ≠ b gives no bound
                if op == "!=":
                    op = "="
                else:
                    poly = {m: -c for m, c in poly.items()}   # ¬(p < 0) is -p <= 0
                    op = "<="
            return contract_atom(op, poly, box)
        if kind == "not":
            return self.refine(node[1], not positive, box)
        if kind == "imp":                               # a → b  ≡  ¬a ∨ b
            if positive:
                return self.hull([(node[1], False), (node[2], True)], box)
            return self.refine(node[1], True, box) | self.refine(node[2], False, box)
        if (kind == "and") == positive and kind in ("and", "or"):
            changed = False
            for k in node[1]:
                if self.refine(k, positive, box):
                    changed = True
            return changed
        if kind in ("and", "or"):
            return self.hull([(k, positive) for k in node[1]], box)
        if kind == "true" and not positive or kind == "false" and positive:
            raise Bottom
        return False                                    # Boolean atoms, iff, xor, bite: no bound

    def hull(self, branches, box):
        results = []
        for node, positive in branches:
            trial = Overlay(box)
            try:
                self.refine(node, positive, trial)
            except Bottom:
                continue
            results.append(trial)
        if not results:
            raise Bottom
        # a variable that some branch did not touch keeps its interval in the hull
        common = set(results[0].own)
        for r in results[1:]:
            common &= r.own.keys()
        changed = False
        for v in common:
            lo = min(r.own[v][0] for r in results)
            hi = max(r.own[v][1] for r in results)
            if tighten(box, v, lo, hi):
                changed = True
        return changed

    def run(self, max_rounds=60):
        """Returns (verdict, box) with verdict in bounded / unbounded / infeasible-by-propagation."""
        box = {v: (-INF, INF) for v in self.reals}
        try:
            for _ in range(max_rounds):
                changed = False
                for a in self.p.asserts:
                    if self.refine(a, True, box):
                        changed = True
                if not changed:
                    break
        except Bottom:
            return "bounded (propagation finds no point)", box
        if all(not math.isinf(lo) and not math.isinf(hi) for lo, hi in box.values()):
            return "bounded", box
        return "unbounded", box


# ----------------------------------------------------------------------------------------------- Lean emission

class Emitter:
    def __init__(self, problem: Problem, reals, bools, ite_cases=64, bool_literal="eq1", vector=False) -> None:
        # bool_literal: how the literal `b` of a Boolean variable reads on its 0/1 real: "eq1" is `b = 1`,
        # "half" is `1 / 2 < b`.  Both are equivalent under the conjunct `b = 0 ∨ b = 1`.
        # vector: bind one `x : Fin n → ℝ` and write the variables `x 0`, `x 1`, ... (the reals first, then the
        # Booleans) instead of n names under one `∃`, whose n nested binders need recursion depth of order n.
        self.bool_literal = bool_literal
        self.vector = vector
        self.p = problem
        if vector:
            self.rname = {v: f"x {i}" for i, v in enumerate(reals)}
            self.bname = {v: f"x {len(reals) + i}" for i, v in enumerate(bools)}
        else:
            self.rname = {v: f"x{i}" for i, v in enumerate(reals)}
            self.bname = {v: f"b{i}" for i, v in enumerate(bools)}
        self.ite_cases = ite_cases
        self.has_ite: dict = {}

    @staticmethod
    def num(q: Fraction) -> str:
        if q.denominator == 1:
            return str(q.numerator) if q >= 0 else f"(-{-q.numerator})"
        body = f"{abs(q.numerator)} / {q.denominator}"
        return f"({body})" if q > 0 else f"(-({body}))"

    @staticmethod
    def balanced(parts, op, sep=" ") -> str:
        """The operands joined by an associative operator as a balanced binary tree: the same value or
        proposition as the chain, at depth log n (Lean's default recursion depth is 512, and a chain of n
        operands nested to one side has depth n)."""
        if len(parts) == 1:
            return parts[0]
        mid = (len(parts) + 1) // 2
        left, right = Emitter.balanced(parts[:mid], op, sep), Emitter.balanced(parts[mid:], op, sep)
        return f"({left}{op.rstrip()}{sep}{right})"

    def flat(self, node, kind):
        """The operands of nested `and` (or `or`) nodes, in order: and(and(a, b), c) is and(a, b, c)."""
        out, stack = [], [node]
        while stack:
            cur = stack.pop()
            if cur[0] == kind:
                stack.extend(reversed(cur[1]))
            else:
                out.append(cur)
        return out

    def contains_ite(self, node) -> bool:
        got = self.has_ite.get(id(node))
        if got is None:
            got = node[0] == "ite" or any(self.contains_ite(k) for k in self.p.kids(node)
                                          ) if node[0] in REAL_KINDS else False
            self.has_ite[id(node)] = got
        return got

    def real(self, node) -> str:
        kind = node[0]
        if kind == "num":
            return self.num(node[1])
        if kind == "var":
            return self.rname[node[1]]
        if kind == "add":
            return self.balanced([self.real(k) for k in node[1]], " + ")
        if kind == "mul":
            return self.balanced([self.real(k) for k in node[1]], " * ")
        if kind == "sub":
            return "(" + " - ".join(self.real(k) for k in node[1]) + ")"
        if kind == "neg":
            return f"(-{self.real(node[1])})"
        if kind == "div":
            return f"({self.real(node[1])} / {self.real(node[2])})"
        raise Unsupported("real ite outside a comparison")

    def cases(self, node):
        """All ite-free versions of a real term with the conditions that select them:
        list of (conditions, term) where conditions is a tuple of (bool node, polarity)."""
        if not self.contains_ite(node):
            return [((), node)]
        kind, mk = node[0], self.p.mk
        if kind == "ite":
            out = [(((node[1], True),) + c, t) for c, t in self.cases(node[2])]
            out += [(((node[1], False),) + c, t) for c, t in self.cases(node[3])]
        else:
            kids = self.p.kids(node)
            out = [((), ())]
            for k in kids:
                out = [(c1 + c2, ts + (t,)) for c1, ts in out for c2, t in self.cases(k)]
                if len(out) > self.ite_cases:
                    raise Unsupported(f"more than {self.ite_cases} cases from real-valued ite in one comparison")
            if kind in ("add", "mul", "sub"):
                out = [(c, mk(kind, ts)) for c, ts in out]
            else:
                out = [(c, mk(kind, *ts)) for c, ts in out]
        if len(out) > self.ite_cases:
            raise Unsupported(f"more than {self.ite_cases} cases from real-valued ite in one comparison")
        return out

    def cmp(self, op, a, b) -> str:
        sym = {"<": "<", "<=": "≤", ">": ">", ">=": "≥", "=": "=", "!=": "≠"}[op]
        return f"(({self.real(a)} : ℝ) {sym} {self.real(b)})"

    def prop(self, node) -> str:
        kind = node[0]
        if kind == "true":
            return "True"
        if kind == "false":
            return "False"
        if kind == "bvar":
            if self.bool_literal == "half":
                return f"(((1 / 2) : ℝ) < {self.bname[node[1]]})"
            return f"({self.bname[node[1]]} = 1)"
        if kind == "not":
            return f"(¬ {self.prop(node[1])})"
        if kind == "and":
            return self.balanced([self.prop(k) for k in self.flat(node, "and")], " ∧ ")
        if kind == "or":
            return self.balanced([self.prop(k) for k in self.flat(node, "or")], " ∨ ")
        if kind == "imp":
            return f"({self.prop(node[1])} → {self.prop(node[2])})"
        if kind == "iff":
            return f"({self.prop(node[1])} ↔ {self.prop(node[2])})"
        if kind == "xor":
            return f"(¬ ({self.prop(node[1])} ↔ {self.prop(node[2])}))"
        if kind == "bite":
            c = self.prop(node[1])
            return f"(({c} → {self.prop(node[2])}) ∧ ((¬ {c}) → {self.prop(node[3])}))"
        if kind == "cmp":
            _, op, a, b = node
            if not (self.contains_ite(a) or self.contains_ite(b)):
                return self.cmp(op, a, b)
            parts = []
            for ca, ta in self.cases(a):
                for cb, tb in self.cases(b):
                    conds = [self.prop(c) if pos else f"(¬ {self.prop(c)})" for c, pos in ca + cb]
                    parts.append("(" + " ∧ ".join(conds + [self.cmp(op, ta, tb)]) + ")")
            if len(parts) > self.ite_cases:
                raise Unsupported(f"more than {self.ite_cases} cases from real-valued ite in one comparison")
            return "(" + " ∨ ".join(parts) + ")"
        raise Unsupported(f"node {kind}")

    def goal(self) -> str:
        binders = [self.rname[v] for v in sorted(self.rname)] + [self.bname[v] for v in sorted(self.bname)]
        conjuncts = [f"({self.bname[v]} = 0 ∨ {self.bname[v]} = 1)" for v in sorted(self.bname)]
        for a in self.p.asserts:
            conjuncts += [self.prop(k) for k in self.flat(a, "and")]
        body = self.balanced(conjuncts, " ∧ ", sep="\n    ") if conjuncts else "True"
        if not binders:
            return f"¬ (\n    {body})"
        if self.vector:
            return f"¬ ∃ x : Fin {len(binders)} → ℝ,\n    {body}"
        return f"¬ ∃ ({' '.join(binders)} : ℝ),\n    {body}"
