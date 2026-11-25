"""From expression nodes (see `nfz.py`) to the two Lean translations.

Route A, "IEEE operations": an FPCore program for FloatLib's own FPBench pipeline
(`benchmarks/fpbench/translate.py`), with stored binary64 inputs and one IEEE operation per
rounding, in the attribute `toPositive`.

Route B, "rounded reals": the model that NumFuzz, FPTaylor and Gappa analyze. Inputs are real
numbers and each rounding operator is `Model.roundAt format t .up` applied to a real expression.
"""
from __future__ import annotations

from fractions import Fraction


class Refusal(Exception):
    """The program has no translation on this route, with the reason."""


def walk(outputs):
    """All nodes reachable from `outputs`, children before parents, each once."""
    order, seen, stack = [], set(), [(node, False) for node in reversed(outputs)]
    while stack:
        node, done = stack.pop()
        if done:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.append((node, True))
        for child in reversed(node[1:]):
            if isinstance(child, tuple):
                stack.append((child, False))
    return order


def counts(outputs):
    result = {}
    for node in walk(outputs):
        result[node[0]] = result.get(node[0], 0) + 1
    return result


def uses(outputs):
    """How many parents read each node (an output counts as one read)."""
    table = {}
    for node in outputs:
        table[id(node)] = table.get(id(node), 0) + 1
    for node in walk(outputs):
        for child in node[1:]:
            if isinstance(child, tuple):
                table[id(child)] = table.get(id(child), 0) + 1
    return table


# ---------------------------------------------------------------- route A: FPCore

def fpcore(name, inputs, output, box, default_bits=64):
    """FPCore text of one output, or a Refusal.

    Each `('rnd', bits, exact operation)` is one IEEE operation; `rnd (a * b + c)` is a fused
    multiply-add. A rounding of a real input, or an exact operation whose result is not rounded,
    has no IEEE operation on stored values.
    """
    for input_name, kind in inputs:
        if kind != "num":
            raise Refusal("an input carries a rounding error (a real input that the program "
                          "rounds); FPCore arguments are stored values")
    names, bindings = {}, []

    def annotate(bits, text):
        if bits == default_bits:
            return text
        return f"(! :precision binary{bits} {text})"

    def operand(node):
        tag = node[0]
        if tag == "var":
            return node[1]
        if tag == "const":
            return node[1]
        if tag in ("rnd", "if"):
            return value(node)
        raise Refusal(f"an exact {tag} is an operand of another operation without a rounding "
                      "between them; no single IEEE operation computes it")

    def condition(node):
        if node[0] in ("gt", "eq"):
            symbol = ">" if node[0] == "gt" else "=="
            return f"({symbol} {operand(node[1])} {operand(node[2])})"
        if node[0] == "if" and node[3] is node[1]:
            # `if c then d else c` is `c and d`; the artifact's `band` is two of these.
            return f"(and {condition(node[1])} {condition(node[2])})"
        raise Refusal("unsupported condition")

    def value(node):
        if id(node) in names:
            return names[id(node)]
        tag = node[0]
        if tag == "if":
            text_cond = condition(node[1])
            text = f"(if {text_cond} {operand(node[2])} {operand(node[3])})"
        elif tag == "rnd":
            bits, inner = node[1], node[2]
            kind = inner[0]
            if kind == "var":
                raise Refusal("rounding of an input")
            if kind == "const":
                text = annotate(bits, f"(cast {inner[1]})") if bits != default_bits else inner[1]
            elif kind == "rnd":
                text = annotate(bits, f"(cast {value(inner)})")
            elif kind == "add":
                left, right = inner[1], inner[2]
                if left[0] == "mul":
                    text = annotate(bits, f"(fma {operand(left[1])} {operand(left[2])} "
                                          f"{operand(right)})")
                elif right[0] == "mul":
                    text = annotate(bits, f"(fma {operand(right[1])} {operand(right[2])} "
                                          f"{operand(left)})")
                else:
                    text = annotate(bits, f"(+ {operand(left)} {operand(right)})")
            elif kind in ("mul", "div"):
                symbol = "*" if kind == "mul" else "/"
                text = annotate(bits, f"({symbol} {operand(inner[1])} {operand(inner[2])})")
            elif kind == "sqrt":
                text = annotate(bits, f"(sqrt {operand(inner[1])})")
            else:
                raise Refusal(f"rounding of {kind}")
        else:
            raise Refusal(f"the result is an exact {tag} that the program does not round")
        fresh = f"t{len(bindings) + 1}"
        bindings.append(f"({fresh} {text})")
        names[id(node)] = fresh
        return fresh

    # Children first, so that `value` never recurses deeper than one operation.
    for node in walk([output]):
        if node[0] == "rnd" or (node[0] == "if" and not _is_condition(node)):
            value(node)
    result = operand(output)
    pre = " ".join(f"(<= {lo} {input_name} {hi})" for (input_name, _), (lo, hi) in zip(inputs, box))
    arguments = " ".join(input_name for input_name, _ in inputs)
    body = f"(let* ({' '.join(bindings)}) {result})" if bindings else result
    return (f"(FPCore ({arguments}) :name \"{name}\" :precision binary{default_bits} "
            f":round toPositive :pre (and {pre}) {body})")


def _is_condition(node):
    """Whether an `if` node is a Boolean (its branches are comparisons), not a number."""
    return node[2][0] in ("gt", "eq") or (node[2][0] == "if" and _is_condition(node[2]))


# ---------------------------------------------------------------- route B: rounded reals

def real_literal(text):
    value = Fraction(text)
    if value.denominator == 1:
        return f"({value.numerator} : ℝ)"
    return f"({value.numerator} / {value.denominator} : ℝ)"


def lean_real(output, rounded=True, share=24):
    """`let` bindings and a final Lean term for `output` over real inputs.

    With `rounded = False` the rounding operators are dropped: the exact real expression. A node
    that several parents read is bound by `let`, and so is every `share`-th level of a deep
    expression, so that no nesting depth grows with the program.
    """
    table = uses([output])
    rendered, bindings = {}, []
    for node in walk([output]):
        tag = node[0]
        if tag == "var":
            rendered[id(node)] = (node[1], 0)
            continue
        if tag == "const":
            rendered[id(node)] = (real_literal(node[1]), 0)
            continue
        parts = [rendered[id(child)] for child in node[1:] if isinstance(child, tuple)]
        level = max((d for _, d in parts), default=0) + 1
        if tag == "rnd":
            if rounded:
                result = (f"(Model.roundAt FloatFormat.binary{node[1]} {parts[0][0]} "
                          f"Model.IEEERoundingMode.up)")
            else:
                rendered[id(node)] = parts[0]
                continue
        elif tag in ("add", "mul", "div"):
            symbol = {"add": "+", "mul": "*", "div": "/"}[tag]
            result = f"({parts[0][0]} {symbol} {parts[1][0]})"
        elif tag == "sqrt":
            result = f"(Real.sqrt {parts[0][0]})"
        elif tag in ("gt", "eq"):
            symbol = ">" if tag == "gt" else "="
            result = f"({parts[0][0]} {symbol} {parts[1][0]})"
        elif tag == "if":
            if _is_condition(node):
                # `if c then d else c` is `c ∧ d`; the artifact's `band` is two of these.
                if node[3] is not node[1]:
                    raise Refusal("unsupported Boolean combination")
                result = f"({parts[0][0]} ∧ {parts[1][0]})"
            else:
                result = f"(if {parts[0][0]} then {parts[1][0]} else {parts[2][0]})"
        else:
            raise Refusal(f"unsupported node {tag}")
        boolean = tag in ("gt", "eq") or (tag == "if" and _is_condition(node))
        if not boolean and (table.get(id(node), 0) > 1 or level >= share) and node is not output:
            fresh = f"t{len(bindings) + 1}"
            bindings.append(f"let {fresh} : ℝ := {result}")
            rendered[id(node)] = (fresh, 0)
        else:
            rendered[id(node)] = (result, level)
    return bindings, rendered[id(output)][0]


def lean_real_term(output, rounded=True, share=24):
    bindings, final = lean_real(output, rounded, share)
    if not bindings:
        return final
    return "(" + "; ".join(bindings) + "; " + final + ")"
