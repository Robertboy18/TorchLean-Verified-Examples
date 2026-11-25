"""The programs of the IEEE route in the direct spelling of a directed operation: the plan.

Usage: python3 -I scripts/direct.py <numfuzz examples dir> WORK

FloatLib's FPBench translator writes an operation of a directed attribute through the descriptor
model, `ExecFloat.Binary.ofModel (Model.add (toModel x) (toModel y) .up)`. The configured type has
the same operation with a mode argument, `ExecFloat.Binary.add x y (mode := .up)` (and `mul`,
`fma`), and for that spelling `roundoff closed` and `roundoff (backward := true)` have their closed
and perturbation forms. For every program whose forward theorem the IEEE route proves in
`toPositive` (read from `WORK/run`), this writes the three `roundoff` commands on the direct
spelling into `WORK/plan-direct.json`: one `let` per operation, in the order of the program, as the
translator does. A program of 30 operations or more is also written with the limits lifted (the
same prefix as `lifted.py`), by size and not by outcome. Programs with a division, a square root or
a literal are not in the set: `roundoff` has no rule for them in a directed attribute.
"""
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import closure
import emit
import gen
import nfz

PREFIX = "set_option maxHeartbeats 0 in set_option maxRecDepth 20000 in "
COMMANDS = (("forward", "roundoff mixed theorem"), ("closed", "roundoff closed theorem"),
            ("backward", "roundoff (backward := true) theorem"))


def term(output, bits):
    """`let` bindings and the final name of `output` over values of `Binary<bits>`."""
    names, lines = {}, []

    def operand(node):
        if node[0] == "var":
            return node[1]
        if node[0] == "rnd":
            return names[id(node)]
        raise emit.Refusal(f"operand {node[0]}")

    for node in emit.walk([output]):
        if node[0] != "rnd":
            if node[0] in ("var", "add", "mul"):
                continue
            raise emit.Refusal(f"node {node[0]}")
        if node[1] != bits:
            raise emit.Refusal("several formats")
        inner = node[2]
        if inner[0] == "add":
            left, right = inner[1], inner[2]
            if left[0] == "mul":
                text = f"ExecFloat.Binary.fma {operand(left[1])} {operand(left[2])} {operand(right)} (mode := .up)"
            elif right[0] == "mul":
                text = f"ExecFloat.Binary.fma {operand(right[1])} {operand(right[2])} {operand(left)} (mode := .up)"
            else:
                text = f"ExecFloat.Binary.add {operand(left)} {operand(right)} (mode := .up)"
        elif inner[0] == "mul":
            text = f"ExecFloat.Binary.mul {operand(inner[1])} {operand(inner[2])} (mode := .up)"
        else:
            raise emit.Refusal(f"rounding of {inner[0]}")
        fresh = f"v{len(lines) + 1}"
        lines.append(f"let {fresh} : Binary{bits} := {text}")
        names[id(node)] = fresh
    return "(" + "; ".join(lines) + "; " + operand(output) + ")"


def main():
    examples, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    root = examples / "NumFuzz"
    results = closure.load_results(work)
    inventory = {row["file"][:-3]: row for row in json.loads((work / "inventory.json").read_text())}
    cases, skipped = [], []
    for stem, row in inventory.items():
        proved = any(results.get(f"A:up:forward:{stem}{suffix}", {}).get("status") == "proved"
                     for suffix in ("", ":lifted"))
        if row["status"] != "read" or not proved:
            continue
        name, inputs, outputs, notes, evaluator = nfz.load(root / row["file"], root)
        output = outputs[0]
        bits = row["formats"][0]
        try:
            body = term(output, bits)
        except emit.Refusal as error:
            skipped.append((stem, str(error)))
            continue
        used = {node[1] for node in emit.walk([output]) if node[0] == "var"}
        binders = " ".join(f"({n} : Binary{bits})" for n, _ in inputs if n in used)
        function = f"fun {binders} => ExecFloat.Binary.toReal {body}"
        variants = [("", "")]
        if row["operations"] >= 30:
            variants.append((":lifted", PREFIX))
        for suffix, prefix in variants:
            for method, command in COMMANDS:
                declaration = f"nf_d_{gen.lean_name(stem)}_{method}"
                case = {"id": f"D:up:{method}:{stem}{suffix}", "declaration": declaration,
                        "commands": [f"{prefix}{command} {declaration} := {function}"],
                        "program": stem, "route": "D", "variant": "up", "method": method}
                if suffix:
                    case["lifted"] = True
                cases.append(case)
    (work / "plan-direct.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(f"{len(cases)} cases of {len({case['program'] for case in cases})} programs -> "
          f"{work / 'plan-direct.json'}; skipped {skipped}")


if __name__ == "__main__":
    main()
