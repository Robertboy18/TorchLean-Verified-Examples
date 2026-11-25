"""Reader for NumFuzz `.fz` programs: includes, parser and a symbolic evaluator.

The evaluator runs a program on symbolic inputs and returns the real expression it computes,
with the rounding operators where the program applies them. Nodes are hash-consed tuples:

  ('var', name)               an input (a positive real; `num` in NumFuzz)
  ('const', 'p/q')            a literal
  ('add'|'mul'|'div', a, b)   an exact real operation
  ('sqrt', a)
  ('rnd', bits, a)            rounding toward +infinity to binary16/32/64
  ('if', cond, a, b)          cond is ('gt', a, b), ('eq', a, b) or ('if', cond, c1, c2)

Types are parsed only to build the symbolic inputs. Nothing here checks NumFuzz's typing rules.
"""
from __future__ import annotations

import re
import sys
from fractions import Fraction
from pathlib import Path


class FzError(Exception):
    """A source the reader cannot parse or evaluate, with the reason."""


# ---------------------------------------------------------------- preprocessing

def strip_comments(text: str) -> str:
    out, i, depth = [], 0, 0
    while i < len(text):
        if text.startswith("/*", i):
            depth += 1
            i += 2
        elif depth and text.startswith("*/", i):
            depth -= 1
            i += 2
        elif depth:
            if text[i] == "\n":
                out.append("\n")
            i += 1
        elif text.startswith("//", i):
            while i < len(text) and text[i] != "\n":
                i += 1
        else:
            out.append(text[i])
            i += 1
    if depth:
        raise FzError("comment not terminated")
    return "".join(out)


INCLUDE = re.compile(r'^\s*#include\s+"([^"]+)"\s*$')


def preprocess(path: Path, root: Path, stack=(), seen=None) -> str:
    """Textual inclusion relative to the including file, as the C preprocessor does.

    A file already included is not included again (the artifact's files have no guards, and a
    second copy only redefines the same functions). Includes may not leave `root`.
    """
    seen = set() if seen is None else seen
    path = path.resolve()
    if root.resolve() not in path.parents:
        raise FzError(f"include outside the examples directory: {path}")
    if path in stack:
        raise FzError(f"include cycle at {path.name}")
    if not path.exists():
        raise FzError(f"included file does not exist: {path.relative_to(root.resolve())}")
    seen.add(path)
    lines = []
    for line in strip_comments(path.read_text()).splitlines():
        match = INCLUDE.match(line)
        if match:
            target = (path.parent / match.group(1)).resolve()
            if target in seen:
                continue
            lines.append(preprocess(target, root, stack + (path,), seen))
        else:
            lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------- lexer

TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>\d+\.\d+(?:e-?\d+)?|\d+)
  | (?P<id>[A-Za-z_][A-Za-z_0-9']*)
  | (?P<sym>-o|=>|\(\||\|\)|[(){}\[\]<>,;:=!+|&])
""", re.X)

KEYWORDS = {"function", "fun", "let", "if", "then", "else", "case", "of", "inl", "inr", "ret",
            "rnd", "rnd16", "rnd32", "rnd64", "add", "mul", "div", "sqrt", "gt", "eq",
            "pi1", "pi2", "num", "bool", "string", "inf", "M"}


def lex(text: str):
    tokens, pos = [], 0
    while pos < len(text):
        match = TOKEN.match(text, pos)
        if not match:
            line = text.count("\n", 0, pos) + 1
            raise FzError(f"illegal character {text[pos]!r} at line {line} of the preprocessed text")
        pos = match.end()
        kind = match.lastgroup
        if kind == "ws":
            continue
        tokens.append((kind, match.group()))
    tokens.append(("eof", ""))
    return tokens


# ---------------------------------------------------------------- parser

class Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    def peek(self, k=0):
        return self.t[min(self.i + k, len(self.t) - 1)]

    def at(self, text, k=0):
        return self.peek(k)[1] == text and self.peek(k)[0] != "num"

    def take(self, text=None):
        kind, value = self.t[self.i]
        if text is not None and value != text:
            raise FzError(f"expected {text!r}, found {value!r} (token {self.i})")
        self.i += 1
        return value

    def ident(self):
        kind, value = self.t[self.i]
        if kind != "id" or value in KEYWORDS:
            raise FzError(f"expected an identifier, found {value!r} (token {self.i})")
        self.i += 1
        return value

    # ---- types
    def sens(self):
        if self.at("{"):
            self.take("{")
            value = self.sens()
            self.take("}")
            return value
        return self.take()

    def atype(self):
        if self.at("("):
            self.take("(")
            if self.at(")"):
                self.take(")")
                return ("unit",)
            parts = [self.type()]
            while self.at(","):
                self.take(",")
                parts.append(self.type())
            self.take(")")
            result = parts[-1]
            for part in reversed(parts[:-1]):
                result = ("tensor", part, result)
            return result
        if self.at("<"):
            self.take("<")
            left = self.type()
            self.take(",")
            right = self.type()
            self.take(">")
            return ("tensor", left, right)
        if self.at("(|"):
            self.take("(|")
            left = self.type()
            self.take(",")
            right = self.type()
            self.take("|)")
            return ("tensor", left, right)
        if self.at("!"):
            self.take("!")
            self.take("[")
            self.sens()
            self.take("]")
            return self.type()                      # a scaled type holds the same values
        if self.at("M"):
            self.take("M")
            self.take("[")
            grade = self.sens()
            self.take("]")
            return ("monad", grade, self.type())
        if self.at("num"):
            self.take()
            return ("num",)
        if self.at("bool"):
            self.take()
            return ("bool",)
        raise FzError(f"unsupported type at {self.peek()[1]!r} (token {self.i})")

    def type(self):
        left = self.atype()
        if self.at("-o"):
            self.take("-o")
            return ("fun", left, self.type())
        if self.at("+") or self.at("&"):
            self.take()
            return ("sum", left, self.type())
        return left

    # ---- terms
    def term(self):
        if self.at("function"):
            self.take()
            name = self.ident()
            params = []
            while self.at("("):
                self.take("(")
                param = self.ident()
                self.take(":")
                params.append((param, self.type()))
                self.take(")")
            if self.at(":"):
                self.take(":")
                self.type()
            self.take("{")
            body = self.term()
            self.take("}")
            rest = None if (self.at("}") or self.peek()[0] == "eof") else self.term()
            return ("function", name, params, body, rest)
        if self.at("let"):
            self.take()
            if self.at("("):
                self.take("(")
                first = self.ident()
                self.take(",")
                second = self.ident()
                self.take(")")
                self.take("=")
                value = self.term()
                self.take(";")
                return ("letpair", first, second, value, self.term())
            if self.at("["):
                self.take("[")
                name = self.ident()
                self.take("]")
                self.take("=")
                value = self.term()
                self.take(";")
                return ("let", name, value, self.term())
            name = self.ident()
            self.take("=")
            value = self.term()
            self.take(";")
            return ("let", name, value, self.term())
        if self.at("if"):
            self.take()
            cond = self.val()
            self.take("then")
            self.take("{")
            yes = self.term()
            self.take("}")
            self.take("else")
            self.take("{")
            no = self.term()
            self.take("}")
            return ("if", cond, yes, no)
        if self.peek()[0] == "id" and self.peek()[1] not in KEYWORDS and (
                self.at("=", 1) or self.at(":", 1)):
            name = self.ident()
            if self.at(":"):
                self.take(":")
                self.type()
            self.take("=")
            value = self.term()
            self.take(";")
            return ("let", name, value, self.term())
        return self.application()

    def starts_val(self):
        kind, value = self.peek()
        if kind == "num":
            return True
        if kind == "id":
            return value not in KEYWORDS or value in ("fun", "inl", "inr", "ret", "rnd", "rnd16",
                                                       "rnd32", "rnd64")
        return value in ("(", "<", "[", "(|")

    def application(self):
        kind, value = self.peek()
        if kind == "id" and value in ("add", "mul", "div", "sqrt", "gt", "eq", "pi1", "pi2"):
            self.take()
            head = ("op", value, self.val())
        else:
            head = self.val()
        while self.starts_val():
            head = ("app", head, self.val())
        return head

    def val(self):
        kind, value = self.peek()
        if kind == "num":
            self.take()
            return ("lit", value)
        if value == "(":
            self.take("(")
            if self.at(")"):
                self.take(")")
                return ("unitv",)
            parts = [self.term()]
            while self.at(","):
                self.take(",")
                parts.append(self.term())
            self.take(")")
            result = parts[-1]
            for part in reversed(parts[:-1]):
                result = ("pair", part, result)
            return result
        if value in ("<", "(|"):
            close = ">" if value == "<" else "|)"
            self.take()
            left = self.term()
            self.take(",")
            right = self.term()
            self.take(close)
            return ("pair", left, right)
        if value == "[":
            self.take("[")
            inner = self.val()
            self.sens()
            self.take("]")
            return inner
        if kind == "id" and value in ("rnd", "rnd16", "rnd32", "rnd64"):
            self.take()
            return ("rnd", value, self.val())
        if kind == "id" and value == "ret":
            self.take()
            return self.val()
        if kind == "id" and value in ("inl", "inr"):
            self.take()
            return (value, self.val())
        if kind == "id" and value == "fun":
            self.take()
            self.take("(")
            param = self.ident()
            self.take(":")
            self.type()
            self.take(")")
            self.take("{")
            body = self.term()
            self.take("}")
            return ("fun", param, body)
        if kind == "id" and value not in KEYWORDS:
            self.take()
            return ("ref", value)
        raise FzError(f"unexpected {value!r} (token {self.i})")


def parse(text: str):
    parser = Parser(lex(text))
    tree = parser.term()
    if parser.peek()[0] != "eof":
        raise FzError(f"trailing text at {parser.peek()[1]!r} (token {parser.i})")
    return tree


# ---------------------------------------------------------------- evaluation

class Scope:
    """Bindings of one body. A scope that a closure captured is frozen: later bindings go to a
    child, so the closure keeps the bindings of its creation."""
    __slots__ = ("vars", "parent", "frozen")

    def __init__(self, vars=None, parent=None):
        self.vars = vars or {}
        self.parent = parent
        self.frozen = False

    def get(self, name):
        scope = self
        while scope is not None:
            if name in scope.vars:
                return scope.vars[name]
            scope = scope.parent
        raise FzError(f"unbound identifier {name}")

    def bind(self, name, value):
        if self.frozen:
            return Scope({name: value}, self)
        self.vars[name] = value
        return self


class Closure:
    __slots__ = ("param", "body", "env")

    def __init__(self, param, body, env):
        env.frozen = True
        self.param, self.body, self.env = param, body, env


class Function:
    """A top-level `function`: all its parameters, so that a call can be checked against them."""
    __slots__ = ("name", "params", "body", "env", "args")

    def __init__(self, name, params, body, env, args=()):
        self.name, self.params, self.body, self.env, self.args = name, params, body, env, args


class Evaluator:
    def __init__(self):
        self.table = {}
        self.notes = []
        self.rnd_default = False
        self.rounds = 0          # rounding operators evaluated
        self.operations = 0      # exact +, *, /, sqrt evaluated (the paper's "Ops")

    def node(self, *parts):
        return self.table.setdefault(parts, parts)

    @staticmethod
    def kind(value):
        if isinstance(value, (Closure, Function)):
            return "fun"
        if isinstance(value, tuple) and value and value[0] == "pair":
            return "pair"
        return "num"

    @staticmethod
    def type_kind(ty):
        while ty[0] == "monad":
            ty = ty[2]
        return {"fun": "fun", "tensor": "pair"}.get(ty[0], "num")

    def call(self, function, arg):
        if isinstance(function, Closure):
            return self.eval(function.body, Scope({function.param: arg}, function.env))
        if isinstance(function, Function):
            args = function.args + (arg,)
            if len(args) < len(function.params):
                return Function(function.name, function.params, function.body, function.env, args)
            wanted = [self.type_kind(ty) for _, ty in function.params]
            have = [self.kind(value) for value in args]
            if wanted != have and sorted(wanted) == sorted(have) and len(set(wanted)) == len(wanted):
                # The artifact's Horner50.fz and poly50.fz pass (function, tuple) to a fold that
                # declares (tuple, function). Bind each argument to the parameter of its kind.
                args = tuple(args[have.index(kind)] for kind in wanted)
                note = (f"call of {function.name} passes its arguments in another order than "
                        f"{function.name} declares; bound by kind")
                if note not in self.notes:
                    self.notes.append(note)
            scope = Scope({name: value for (name, _), value in zip(function.params, args)},
                          function.env)
            return self.eval(function.body, scope)
        raise FzError("application of a value that is not a function")

    def eval(self, term, env):
        # Sequencing forms are iterated, so that a body of thousands of bindings does not recurse.
        while True:
            tag = term[0]
            if tag == "let":
                env = env.bind(term[1], self.eval(term[2], env))
                term = term[3]
                continue
            if tag == "letpair":
                value = self.eval(term[3], env)
                if not (isinstance(value, tuple) and value[0] == "pair"):
                    raise FzError("tensor elimination of a value that is not a pair")
                env = env.bind(term[1], value[1])
                env = env.bind(term[2], value[2])
                term = term[4]
                continue
            if tag == "function":
                env.frozen = True
                function = Function(term[1], term[2], term[3], env)
                env = env.bind(term[1], function)
                if term[4] is None:
                    return function
                term = term[4]
                continue
            break
        if tag == "ref":
            return env.get(term[1])
        if tag == "lit":
            return self.node("const", str(Fraction(term[1])))
        if tag == "unitv":
            return ("unit",)
        if tag == "pair":
            return ("pair", self.eval(term[1], env), self.eval(term[2], env))
        if tag == "fun":
            return Closure(term[1], term[2], env)
        if tag == "app":
            return self.call(self.eval(term[1], env), self.eval(term[2], env))
        if tag == "rnd":
            if term[1] == "rnd":
                self.rnd_default = True
            bits = {"rnd": 64, "rnd16": 16, "rnd32": 32, "rnd64": 64}[term[1]]
            value = self.eval(term[2], env)
            self.numeric(value)
            self.rounds += 1
            return self.node("rnd", bits, value)
        if tag in ("inl", "inr"):
            return (tag, self.eval(term[1], env))
        if tag == "op":
            value = self.eval(term[2], env)
            name = term[1]
            if name == "sqrt":
                self.numeric(value)
                self.operations += 1
                return self.node("sqrt", value)
            if name in ("pi1", "pi2"):
                if not (isinstance(value, tuple) and value[0] == "pair"):
                    raise FzError("projection of a value that is not a pair")
                return value[1 if name == "pi1" else 2]
            if not (isinstance(value, tuple) and value[0] == "pair"):
                raise FzError(f"{name} applied to a value that is not a pair")
            self.numeric(value[1])
            self.numeric(value[2])
            if name in ("add", "mul", "div"):
                self.operations += 1
            return self.node(name, value[1], value[2])
        if tag == "if":
            cond = self.eval(term[1], env)
            if isinstance(cond, tuple) and cond[0] in ("inl", "inr"):
                return self.eval(term[2] if cond[0] == "inl" else term[3], env)
            if not (isinstance(cond, tuple) and cond[0] in ("gt", "eq", "if")):
                raise FzError("conditional on a value that is not a comparison")
            return self.node("if", cond, self.eval(term[2], env), self.eval(term[3], env))
        raise FzError(f"unsupported term {tag}")

    def numeric(self, value):
        if not (isinstance(value, tuple) and value[0] in
                ("var", "const", "add", "mul", "div", "sqrt", "rnd", "if")):
            raise FzError("arithmetic on a value that is not a number")

    # ---- symbolic inputs
    def fresh(self, ty, prefix, inputs):
        tag = ty[0]
        if tag == "num":
            name = f"{prefix}"
            inputs.append((name, "num"))
            return self.node("var", name)
        if tag == "monad":
            if ty[2][0] != "num":
                raise FzError("input of a monadic type that is not a number")
            name = f"{prefix}"
            inputs.append((name, "rounded:" + str(ty[1])))
            bits = {"eps64_up": 64, "eps32_up": 32, "eps16_up": 16}.get(ty[1])
            if bits is None:
                raise FzError(f"input with error grade {ty[1]}: no rounding operator to model it")
            return self.node("rnd", bits, self.node("var", name))
        if tag == "tensor":
            parts, cursor = [], ty
            while cursor[0] == "tensor":
                parts.append(cursor[1])
                cursor = cursor[2]
            parts.append(cursor)
            values = [self.fresh(part, f"{prefix}_{index}", inputs)
                      for index, part in enumerate(parts)]
            result = values[-1]
            for value in reversed(values[:-1]):
                result = ("pair", value, result)
            return result
        if tag == "fun":
            raise FzError("higher-order parameter: a library function, not a program")
        if tag == "bool":
            raise FzError("Boolean parameter: no numerical input")
        if tag == "unit":
            return ("unit",)
        raise FzError(f"input of unsupported type {tag}")


def clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "p", name)


def load(path: Path, root: Path):
    """Evaluate the program of `path` on symbolic inputs.

    Returns (name, inputs, outputs, notes, evaluator): `inputs` is a list of (name, kind),
    `outputs` a list of expression nodes (one for a scalar program, the entries in order for a
    tuple), and the evaluator holds the counts of the operations it ran.
    """
    evaluator = Evaluator()
    tree = parse(preprocess(path, root))
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    value = evaluator.eval(tree, Scope())
    if not isinstance(value, Function):
        raise FzError("the program's value is not a top-level function")
    inputs, result = [], value
    name = value.name
    used = set()
    while isinstance(result, Function):
        function = result
        for param, ty in function.params[len(function.args):]:
            base = clean(param)
            while base in used:
                base += "_"
            used.add(base)
            result = evaluator.call(result, evaluator.fresh(ty, base, inputs))
    if isinstance(result, Closure):
        raise FzError("the program returns a function")
    outputs = []

    def flatten(item):
        if isinstance(item, tuple) and item and item[0] == "pair":
            flatten(item[1])
            flatten(item[2])
        else:
            evaluator.numeric(item)
            outputs.append(item)

    flatten(result)
    if evaluator.rnd_default:
        evaluator.notes.append("uses the unsuffixed `rnd`, read as rnd64 (the artifact's README: "
                               "\"the precision is fixed as binary64\")")
    return name, inputs, outputs, evaluator.notes, evaluator
