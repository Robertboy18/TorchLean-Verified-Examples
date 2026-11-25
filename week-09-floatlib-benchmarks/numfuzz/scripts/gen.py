"""Inventory of the NumFuzz artifact's programs and the plan of Lean cases.

Usage: python3 -I scripts/gen.py <numfuzz examples dir> <FloatLib checkout> <work dir>

Writes `inventory.json` (one row per `.fz` file: class, inputs, operation counts, notes, the
FPCore text and the Lean terms), `fpcore/*.fpcore`, and `plan.json` (the Lean cases).
The published numbers are transcribed from arXiv:2405.04612v2, Tables 3, 4 and 5.
"""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import emit
import fpt
import nfz

# name: (ops, NumFuzz, FPTaylor, Gappa, t NumFuzz, t FPTaylor, t Gappa), relative error bounds
TABLE3 = {
    "hypot": (4, "5.55e-16", "5.17e-16", "4.46e-16", "0.002", "3.55", "0.069"),
    "x_by_xy": (3, "4.44e-16", "fail", "2.22e-16", "0.002", "", "0.034"),
    "one_by_sqrtxx": (3, "5.55e-16", "5.09e-13", "3.33e-16", "0.002", "3.34", "0.047"),
    "sqrt_add": (5, "9.99e-16", "6.66e-16", "5.54e-16", "0.003", "3.28", "0.092"),
    "test02_sum8": (8, "1.55e-15", "9.32e-14", "1.55e-15", "0.002", "14.61", "0.244"),
    "nonlin1": (2, "4.44e-16", "4.49e-16", "2.22e-16", "0.003", "3.24", "0.040"),
    "test05_nonlin1": (2, "4.44e-16", "4.46e-16", "2.22e-16", "0.008", "3.27", "0.042"),
    "verhulst": (4, "8.88e-16", "7.38e-16", "4.44e-16", "0.002", "3.25", "0.069"),
    "predatorPrey": (7, "1.55e-15", "4.21e-11", "8.88e-16", "0.002", "3.28", "0.114"),
    "test06_sums4_sum1": (4, "6.66e-16", "6.71e-16", "6.66e-16", "0.003", "3.84", "0.069"),
    "test06_sums4_sum2": (4, "6.66e-16", "1.78e-14", "4.44e-16", "0.002", "11.02", "0.055"),
    "i4": (4, "4.44e-16", "4.50e-16", "4.44e-16", "0.002", "3.30", "0.055"),
    "Horner2": (4, "4.44e-16", "6.49e-11", "4.44e-16", "0.002", "11.72", "0.052"),
    "Horner2_with_er": (4, "1.55e-15", "1.61e-10", "1.11e-15", "0.002", "19.56", "0.119"),
    "Horner5": (10, "1.11e-15", "1.62e-01", "1.11e-15", "0.003", "22.08", "0.209"),
    "Horner10": (20, "2.22e-15", "1.14e+13", "2.22e-15", "0.003", "40.68", "0.650"),
    "Horner20": (40, "4.44e-15", "2.53e+43", "4.44e-15", "0.003", "109.42", "2.246"),
}
# name: (ops, NumFuzz bound, literature bound, t NumFuzz, t Satire)
TABLE4 = {
    "large_benchmarks/Horner50": (100, "1.11e-14", "1.11e-14", "9e-03", "5"),
    "large_benchmarks/mat_mul_4": (112, "1.55e-15", "8.88e-16", "3e-03", ""),
    "large_benchmarks/Horner75": (150, "1.66e-14", "1.66e-14", "2e-02", ""),
    "large_benchmarks/Horner100": (200, "2.22e-14", "2.22e-14", "4e-02", ""),
    "large_benchmarks/serial_sum": (1023, "2.27e-13", "2.27e-13", "5", "5407"),
    "large_benchmarks/poly50": (1325, "2.94e-13", "", "2.12", "3"),
    "large_benchmarks/mat_mul_16": (7936, "6.88e-15", "3.55e-15", "4e-02", ""),
    "large_benchmarks/mat_mul_64": (520192, "2.82e-14", "1.42e-14", "10", "65"),
    "large_benchmarks/mat_mul_128": (4177920, "5.66e-14", "2.84e-14", "1080", "763"),
}
# name: (NumFuzz bound, t NumFuzz in ms)
TABLE5 = {
    "conditionals/pythagorean_sum": ("8.88e-16", "2"),
    "conditionals/hammarling_distance": ("1.11e-15", "2"),
    "conditionals/squareRoot3": ("4.44e-16", "2"),
    "conditionals/squareRoot3Invalid": ("4.44e-16", "2"),
}


def signature(node, memo, order):
    """A structural key of an expression up to the names of its inputs: inputs are numbered in
    the order a fixed traversal first meets them."""
    key = id(node)
    if key in memo:
        return memo[key]
    tag = node[0]
    if tag == "var":
        result = ("v", order.setdefault(node[1], len(order)))
    elif tag == "const":
        result = ("c", node[1])
    else:
        result = (tag,) + tuple(signature(child, memo, order) if isinstance(child, tuple) else child
                                for child in node[1:])
    result = hash(result)
    memo[key] = result
    return result


def same_up_to_inputs(outputs):
    first = None
    for output in outputs:
        sys.setrecursionlimit(100000)
        key = signature(output, {}, {})
        if first is None:
            first = key
        elif key != first:
            return False
    return True


def commutative_key(node, memo):
    """A key of an expression up to the order of the operands of + and * and up to input names
    (inputs are told apart only by how often they occur). It only screens for differences."""
    key = id(node)
    if key in memo:
        return memo[key]
    tag = node[0]
    if tag == "var":
        result = "v"
    elif tag == "const":
        result = "c" + str(Fraction(node[1]))
    elif tag == "rnd" and node[2][0] == "const" and dyadic_exact(node[2][1], node[1]):
        result = commutative_key(node[2], memo)
    else:
        parts = [commutative_key(child, memo) if isinstance(child, tuple) else str(child)
                 for child in node[1:]]
        if tag in ("add", "mul"):
            parts.sort()
        result = tag + "(" + ",".join(parts) + ")"
    memo[key] = result
    return result


UNIT = {16: Fraction(1, 2 ** 10), 32: Fraction(1, 2 ** 23), 64: Fraction(1, 2 ** 52)}


def apriori(output):
    """The worst-case relative error of the program over all positive inputs when no result
    underflows, by the textbook rules: a rounding toward +infinity multiplies by at most 1 + u
    (u = 2^(1-p)), a product adds the relative errors of its factors, a quotient divides by one
    minus that of its divisor, a square root halves, and a sum of positive terms keeps the
    larger one. It is an estimate by this script, used only to propose bounds that Lean checks."""
    table = {}
    for node in emit.walk([output]):
        tag = node[0]
        if tag in ("var", "const"):
            value = Fraction(0)
        elif tag == "rnd":
            unit = UNIT[node[1]]
            value = (1 + unit) * table[id(node[2])] + unit
        elif tag == "add":
            value = max(table[id(node[1])], table[id(node[2])])
        elif tag == "mul":
            a, b = table[id(node[1])], table[id(node[2])]
            value = a + b + a * b
        elif tag == "div":
            a, b = table[id(node[1])], table[id(node[2])]
            value = (a + b) / (1 - b)
        elif tag == "sqrt":
            a = table[id(node[1])]
            value = a / 2 * (1 + a)
        elif tag == "if":
            if emit._is_condition(node):
                value = Fraction(0)
            else:
                value = max(table[id(node[2])], table[id(node[3])])
        else:
            value = Fraction(0)
        table[id(node)] = value
    return table[id(output)]


def exact_range(output, box):
    """An enclosure [lo, hi] of the exact value of the program on the box, by interval arithmetic
    on positive intervals (the rounding operators dropped). Only `lo` is used, as a proposal for
    the lower bound of the exact value that Lean then checks."""
    import math
    table = {}
    for node in emit.walk([output]):
        tag = node[0]
        if tag == "var":
            value = box[node[1]]
        elif tag == "const":
            value = (Fraction(node[1]), Fraction(node[1]))
        elif tag == "rnd":
            value = table[id(node[2])]
        elif tag in ("add", "mul", "div"):
            (a, b), (c, d) = table[id(node[1])], table[id(node[2])]
            if tag == "add":
                value = (a + c, b + d)
            elif tag == "mul":
                value = (a * c, b * d)
            else:
                value = (a / d, b / c) if c > 0 else (Fraction(0), Fraction(0))
        elif tag == "sqrt":
            a, b = table[id(node[1])]
            value = (Fraction(math.sqrt(float(a))) * Fraction(999999, 1000000),
                     Fraction(math.sqrt(float(b))) * Fraction(1000001, 1000000))
        elif tag == "if" and not emit._is_condition(node):
            (a, b), (c, d) = table[id(node[2])], table[id(node[3])]
            value = (min(a, c), max(b, d))
        else:
            value = (Fraction(0), Fraction(0))
        table[id(node)] = value
    return table[id(output)]


def dyadic_exact(text, bits):
    """Whether a literal is a binary floating-point number of `bits` (precision only)."""
    value = Fraction(text)
    if value == 0:
        return True
    if value.denominator & (value.denominator - 1):
        return False
    numerator = abs(value.numerator)
    while numerator % 2 == 0:
        numerator //= 2
    return numerator.bit_length() <= {16: 11, 32: 24, 64: 53}[bits]


def classify(rel):
    stem = rel[:-3]
    if stem in TABLE3:
        return "table3"
    if stem in TABLE4:
        return "table4"
    if stem in TABLE5:
        return "table5"
    if stem.startswith("mixed_precision/"):
        return "mixed"
    return "other"


def lean_name(text):
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def main():
    examples, floatlib, work = (Path(arg).resolve() for arg in sys.argv[1:4])
    root = examples / "NumFuzz"
    sys.path.insert(0, str(floatlib / "benchmarks/fpbench"))
    sys.path.insert(0, str(floatlib / "benchmarks/suites"))
    from translate import cores, translate      # FloatLib's FPBench translator
    from numeric import numeric_case             # and its numeric closure
    from sexpr import Refusal as SuiteRefusal
    (work / "fpcore").mkdir(parents=True, exist_ok=True)
    inventory, plan = [], []
    for path in sorted(root.rglob("*.fz")):
        rel = str(path.relative_to(root))
        stem = rel[:-3]
        row = {"file": rel, "class": classify(rel), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        inventory.append(row)
        try:
            name, inputs, outputs, notes, evaluator = nfz.load(path, root)
        except nfz.FzError as error:
            row.update(status="not_a_program", reason=str(error))
            continue
        counts = emit.counts(outputs)
        row.update(status="read", program=name, inputs=len(inputs),
                   input_kinds=sorted({kind for _, kind in inputs}),
                   outputs=len(outputs), notes=notes,
                   operations=evaluator.operations, roundings=evaluator.rounds,
                   distinct_roundings=counts.get("rnd", 0),
                   has_div=counts.get("div", 0) > 0, has_sqrt=counts.get("sqrt", 0) > 0,
                   has_if=counts.get("if", 0) > 0,
                   formats=sorted({node[1] for node in emit.walk(outputs) if node[0] == "rnd"}))
        if not evaluator.rounds:
            row.update(status="no_rounding", reason="the program rounds nothing")
            continue
        if len(outputs) > 1:
            if not same_up_to_inputs(outputs):
                row.update(status="outputs_differ", reason="outputs are not one program up to inputs")
                continue
            row["notes"] = notes + [f"{len(outputs)} outputs, each the first one at other inputs "
                                    "(checked by the script); one theorem covers every output"]
        output = outputs[0]
        # Inputs of the one output that is translated.
        used = {node[1] for node in emit.walk([output]) if node[0] == "var"}
        inputs = [(n, kind) for n, kind in inputs if n in used]
        row["theorem_inputs"] = len(inputs)
        estimate = apriori(output)
        row["apriori_script"] = float(estimate)
        row["apriori_script_rational"] = (str(estimate)
                                          if estimate.denominator.bit_length() < 1200 else "")
        # The interval the paper gives FPTaylor and Gappa, or the FPTaylor file's own.
        box = {n: (Fraction(1, 10), Fraction(1000)) for n, _ in inputs}
        fptaylor = examples / "FPTaylor" / (stem + ".txt")
        if fptaylor.exists():
            row["fptaylor_file"] = str(fptaylor.relative_to(examples))
            try:
                node = evaluator.node
                variables, expressions = fpt.load(fptaylor, node)
                theirs = expressions[-1][1]
                mine = commutative_key(output, {})
                other = commutative_key(theirs, {})
                row["fptaylor_same_program"] = mine == other
                if mine != other:
                    row["fptaylor_expression"] = emit.lean_real_term(theirs, share=10 ** 9)
                ranges = sorted({(str(lo), str(hi)) for _, lo, hi in variables.values()})
                row["fptaylor_ranges"] = ranges
                row["fptaylor_kinds"] = sorted({kind for kind, _, _ in variables.values()})
                if len(ranges) == 1:
                    lo, hi = Fraction(ranges[0][0]), Fraction(ranges[0][1])
                    box = {n: (lo, hi) for n, _ in inputs}
            except fpt.FptError as error:
                row["fptaylor_error"] = str(error)
        gappa = examples / "Gappa" / (stem + ".g")
        if gappa.exists():
            row["gappa_file"] = str(gappa.relative_to(examples))
        row["box"] = sorted({(str(lo), str(hi)) for lo, hi in box.values()})
        declaration = "nf_" + lean_name(stem)
        # ---- route B: rounded reals
        binders = " ".join(f"({n} : ℝ)" for n, _ in inputs)
        positive = " ".join(f"(h_{n} : 0 < {n})" for n, _ in inputs)
        boxed = " ".join(f"(hl_{n} : ({box[n][0].numerator} / {box[n][0].denominator} : ℝ) ≤ {n}) "
                         f"(hu_{n} : {n} ≤ ({box[n][1].numerator} / {box[n][1].denominator} : ℝ))"
                         for n, _ in inputs)
        try:
            term = emit.lean_real_term(output)
            exact = emit.lean_real_term(output, rounded=False)
            row["lean_rounded"] = term if len(term) < 4000 else term[:4000] + " ..."
            row["lean_exact"] = exact if len(exact) < 4000 else exact[:4000] + " ..."
            for variant, hypotheses in (("pos", positive), ("box", boxed)):
                plan.append({
                    "id": f"B:{variant}:forward:{stem}", "declaration": f"{declaration}_{variant}",
                    "commands": [f"roundoff mixed theorem {declaration}_{variant} := "
                                 f"fun {binders} {hypotheses} => {term}"],
                    "program": stem, "route": "B", "variant": variant, "method": "forward",
                    "binders": binders, "hypotheses": hypotheses, "term": term, "exact": exact,
                    "inputs": [n for n, _ in inputs],
                    "apriori": float(estimate),
                    "exact_lower": float(exact_range(output, box)[0]),
                    "box": {n: [str(box[n][0]), str(box[n][1])] for n, _ in inputs}})
        except emit.Refusal as error:
            row["route_b_refusal"] = str(error)
        # ---- route A: FloatLib's FPBench pipeline on an FPCore of the program
        # A program that rounds to one format is an FPCore of that precision (its arguments are
        # then values of that format); a program of several formats is a binary64 FPCore with
        # precision annotations.
        default_bits = row["formats"][0] if len(row["formats"]) == 1 else 64
        row["fpcore_precision"] = f"binary{default_bits}"
        for attribute in ("up", "rn"):
            try:
                text = emit.fpcore(name, inputs, output, [(str(box[n][0]), str(box[n][1]))
                                                          for n, _ in inputs], default_bits)
            except emit.Refusal as error:
                row["route_a_refusal"] = "translation to FPCore: " + str(error)
                break
            if attribute == "rn":
                text = text.replace(" :round toPositive", "")
            file = work / "fpcore" / f"{lean_name(stem)}.{attribute}.fpcore"
            file.write_text(text + "\n")
            if attribute == "up":
                row["fpcore"] = text if len(text) < 4000 else text[:4000] + " ..."
            try:
                index, core_name, args, attrs, body = next(iter(cores(file)))
            except (SuiteRefusal, ValueError, RecursionError) as error:
                row[f"route_a_{attribute}_refusal"] = "FloatLib's FPCore reader: " + str(error)
                continue
            for method in ("forward", "closed", "backward"):
                ident = f"A:{attribute}:{method}:{stem}"
                try:
                    translated = translate(args, attrs, body, f"binary{default_bits}", method, ident)
                except SuiteRefusal as error:
                    row[f"route_a_{attribute}_refusal"] = "FloatLib's translate.py: " + str(error)
                    break
                except RecursionError:
                    row[f"route_a_{attribute}_refusal"] = "FloatLib's translate.py: Python recursion limit"
                    break
                case = {"id": ident, "declaration": translated["declaration"],
                        "commands": translated["commands"], "program": stem, "route": "A",
                        "variant": attribute, "method": method}
                plan.append(case)
                if method == "backward":
                    continue
                try:
                    closure = numeric_case(args, attrs, body, f"binary{default_bits}", method,
                                           translated)
                except SuiteRefusal as error:
                    row[f"route_a_{attribute}_{method}_numeric_refusal"] = str(error)
                    continue
                except RecursionError:
                    row[f"route_a_{attribute}_{method}_numeric_refusal"] = "Python recursion limit"
                    continue
                plan.append({"id": ident + ":numeric", "declaration": closure["declaration"],
                             "commands": closure["commands"], "program": stem, "route": "A",
                             "variant": attribute, "method": method + ":numeric",
                             "proposed_bound": closure["proposed_bound"]})
    (work / "inventory.json").write_text(json.dumps(inventory, indent=1, ensure_ascii=False))
    (work / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    by_status = {}
    for row in inventory:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    print(f"{len(inventory)} files: {by_status}; {len(plan)} cases")
    for row in inventory:
        flags = [key for key in row if "refusal" in key or key == "fptaylor_error"]
        print(f"{row['file']:48s} {row['class']:7s} {row['status']:14s} "
              f"ops={row.get('operations', '')} rnd={row.get('roundings', '')} "
              f"same_fpt={row.get('fptaylor_same_program', '')} "
              + "; ".join(f"{key}: {row[key]}" for key in flags)[:230]
              + (" | " + row.get("reason", "") if row["status"] != "read" else ""))


if __name__ == "__main__":
    main()
