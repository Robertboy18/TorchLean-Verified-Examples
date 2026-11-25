"""Reader for the FPTaylor inputs of the NumFuzz artifact (`examples/FPTaylor/**/*.txt`).

It builds the same expression nodes as `nfz.py`. Only the syntax these files use is read:
`Variables` (float64 or real, with an interval), `Definitions` and `Expressions`, each either
`name = expr;` or `name rndNN_up= expr;`, with `+ * /`, `sqrt`, `rndNN_up(...)` and literals.

`name rnd= expr` applies the rounding to every operation, constant and variable of `expr`
(FPTaylor's REFERENCE). Rounding a float64 variable to binary64 changes nothing; rounding a real
variable, or a float64 variable to binary32, is a rounding.
"""
from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path


class FptError(Exception):
    pass


TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)
  | (?P<id>[A-Za-z_][A-Za-z_0-9]*)
  | (?P<sym>[-+*/()\[\],;={}])
""", re.X)


def lex(text):
    tokens, pos = [], 0
    while pos < len(text):
        match = TOKEN.match(text, pos)
        if not match:
            raise FptError(f"illegal character {text[pos]!r}")
        pos = match.end()
        if match.lastgroup != "ws":
            tokens.append((match.lastgroup, match.group()))
    tokens.append(("eof", ""))
    return tokens


class Reader:
    def __init__(self, tokens, node):
        self.t, self.i, self.node = tokens, 0, node
        self.vars = {}        # name -> (kind, lo, hi)
        self.defs = {}        # name -> node
        self.outputs = []     # (name, node)

    def peek(self):
        return self.t[self.i]

    def at(self, text):
        return self.t[self.i][1] == text

    def take(self, text=None):
        kind, value = self.t[self.i]
        if text is not None and value != text:
            raise FptError(f"expected {text!r}, found {value!r}")
        self.i += 1
        return value

    def number(self):
        sign = -1 if self.at("-") else 1
        if self.at("-") or self.at("+"):
            self.take()
        kind, value = self.t[self.i]
        if kind != "num":
            raise FptError(f"expected a number, found {value!r}")
        self.i += 1
        return sign * Fraction(value)

    def rounding(self, name):
        match = re.fullmatch(r"rnd(16|32|64)_up", name)
        return int(match.group(1)) if match else None

    def read(self):
        braces = 0
        while self.peek()[0] != "eof":
            if self.at("{"):
                self.take()
                braces += 1
            elif self.at("}"):
                self.take()
                braces -= 1
            elif self.at("Variables"):
                self.take()
                while self.peek()[1] in ("float64", "float32", "real"):
                    kind = self.take()
                    name = self.take()
                    self.take("in")
                    self.take("[")
                    lo = self.number()
                    self.take(",")
                    hi = self.number()
                    self.take("]")
                    self.take(";")
                    self.vars[name] = (kind, lo, hi)
            elif self.at("Definitions") or self.at("Expressions"):
                section = self.take()
                while self.peek()[0] == "id" and self.peek()[1] not in (
                        "Variables", "Definitions", "Expressions", "Constraints"):
                    name = self.take()
                    bits = None
                    if self.peek()[0] == "id":
                        bits = self.rounding(self.peek()[1])
                        if bits is None:
                            raise FptError(f"unsupported assignment operator {self.peek()[1]}=")
                        self.take()
                    self.take("=")
                    value = self.expr(bits)
                    self.take(";")
                    if section == "Definitions":
                        self.defs[name] = value
                    else:
                        self.outputs.append((name, value))
            else:
                raise FptError(f"unexpected {self.peek()[1]!r}")
        if braces:
            raise FptError("unbalanced braces")

    # expr := term (('+'|'-') term)*; term := factor (('*'|'/') factor)*
    def wrap(self, bits, value):
        return value if bits is None else self.node("rnd", bits, value)

    def expr(self, bits):
        left = self.term(bits)
        while self.at("+") or self.at("-"):
            op = self.take()
            if op == "-":
                raise FptError("subtraction")
            left = self.wrap(bits, self.node("add", left, self.term(bits)))
        return left

    def term(self, bits):
        left = self.factor(bits)
        while self.at("*") or self.at("/"):
            op = self.take()
            right = self.factor(bits)
            left = self.wrap(bits, self.node("mul" if op == "*" else "div", left, right))
        return left

    def factor(self, bits):
        kind, value = self.peek()
        if value == "(":
            self.take()
            inner = self.expr(bits)
            self.take(")")
            return inner
        if kind == "num":
            self.take()
            literal = self.node("const", str(Fraction(value)))
            return self.wrap(bits, literal)
        if kind == "id":
            self.take()
            if self.at("("):
                self.take("(")
                called = self.rounding(value)
                if called is not None:
                    # An explicit operator rounds its argument once; the argument is read unrounded.
                    inner = self.expr(None)
                    self.take(")")
                    return self.node("rnd", called, inner)
                if value == "sqrt":
                    inner = self.expr(bits)
                    self.take(")")
                    return self.wrap(bits, self.node("sqrt", inner))
                raise FptError(f"unsupported function {value}")
            if value in self.defs:
                return self.defs[value]
            if value in self.vars:
                variable = self.node("var", value)
                if bits is None:
                    return variable
                if self.vars[value][0] == "float64" and bits == 64:
                    return variable
                if self.vars[value][0] == "float32" and bits in (32, 64):
                    return variable
                return self.node("rnd", bits, variable)
            raise FptError(f"unknown identifier {value}")
        raise FptError(f"unexpected {value!r}")


def load(path: Path, node):
    """Return (variables, outputs): variables maps a name to (kind, lo, hi) and outputs is a list of
    (name, node). `node` is the hash-consing constructor shared with the caller."""
    text = path.read_text()
    if not text.strip():
        raise FptError("empty file")
    reader = Reader(lex(text), node)
    reader.read()
    if not reader.outputs:
        raise FptError("no expression")
    return reader.vars, reader.outputs
