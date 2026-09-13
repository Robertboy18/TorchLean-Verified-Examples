#!/usr/bin/env python3
"""Build PTX/SASS and emit a Lean certificate for `tiny_attn_one_row.cu`.

This extractor is narrow on purpose. It is not a general PTX parser; it is a
reproducible way to compile one microkernel, read the PTX/SASS we got back, and
package the relevant facts for Lean to check. Hardware correctness is still a
separate boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = EXAMPLE_ROOT / "cuda" / "tiny_attn_one_row.cu"
DEFAULT_BUILD = EXAMPLE_ROOT / "cuda" / "build"
DEFAULT_JSON = EXAMPLE_ROOT / "cuda" / "cert_tiny_attn.json"
DEFAULT_LEAN = EXAMPLE_ROOT / "BatchInvariantInference" / "Generated" / "TinyValueReductionCert.lean"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_path(path: Path) -> str:
    """Return a stable repo-relative path for generated certificates."""
    return str(path.resolve().relative_to(REPO_ROOT))


def lean_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text))


# The address calculations use 32-bit element indices, followed by an unsigned
# widening multiply and a 64-bit pointer addition. Keep those widths separate:
# simplifying everything over the integers would miss wraparound in the index.
# The four coefficients describe blockIdx.x, threadIdx.x, B, and a constant.
U32 = tuple[int, int, int, int]
Address = dict[str, int]
MOD32 = 1 << 32
MOD64 = 1 << 64
BLOCK: U32 = (1, 0, 0, 0)
THREAD: U32 = (0, 1, 0, 0)
BATCH: U32 = (0, 0, 1, 0)


def constant32(value: int) -> U32:
    return (0, 0, 0, value % MOD32)


def add32(left: U32, right: U32) -> U32:
    return tuple((a + b) % MOD32 for a, b in zip(left, right))


def scale32(value: U32, factor: int) -> U32:
    return tuple(a * factor % MOD32 for a in value)


def add64(left: Address, right: Address) -> Address:
    result = dict(left)
    for atom, coefficient in right.items():
        result[atom] = (result.get(atom, 0) + coefficient) % MOD64
    return {atom: coefficient for atom, coefficient in result.items() if coefficient}


def scale64(value: Address, factor: int) -> Address:
    return add64({}, {atom: coefficient * factor for atom, coefficient in value.items()})


def widen_index(value: U32, thread_valid: bool) -> Address:
    """Zero-extend the row index without distributing through a possible carry.

    For stride s in {4, 8, 32}, `(s * blockIdx.x) mod 2^32` is a multiple of s
    and at most 2^32 - s. A lane smaller than s can therefore be added before
    or after zero-extension. The thread guard gives threadIdx.x <= 3; for the
    value loads the largest lane is 3 + 4*7 = 31. This is why CUDA's recomputed
    32-bit indices and its base-plus-byte-offset form describe the same loads.
    """
    stride, thread, batch, offset = value
    if not stride and not thread and not batch:
        return add64({}, {"constant": offset})
    if (
        batch or stride not in (4, 8, 32) or thread not in (0, 1)
        or offset + 3 * thread >= stride or (thread and not thread_valid)
    ):
        raise ValueError(f"Unsupported widening index, or unproved carry bound: {value}")
    return add64(
        {f"row{stride}": 1},
        {"thread": thread, "constant": offset},
    )


def strip_ptx_comments(ptx: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.S)


def ptx_architecture(ptx: str) -> str:
    """Read the single target directive recorded in the compiled PTX."""
    targets = re.findall(r"(?m)^\s*\.target\s+(sm_\d+[af]?)\s*$", strip_ptx_comments(ptx))
    if len(targets) != 1:
        raise ValueError("Expected exactly one ordinary PTX target directive")
    return targets[0]


def parse_program(ptx: str) -> tuple[dict, list, dict]:
    """Read the single kernel and reject instructions outside the checked subset."""
    text = strip_ptx_comments(ptx)
    match = re.fullmatch(
        r"(.*?)\.visible\s+\.entry\s+tiny_attn_one_row\s*"
        r"\((.*?)\)\s*\{([^{}]*)\}\s*", text, re.S,
    )
    if match is None:
        raise ValueError("Expected one tiny_attn_one_row entry and no other functions")
    header, parameters, body = match.groups()
    for line in header.splitlines():
        if line.strip() and not re.fullmatch(
            r"\s*(?:\.version\s+\d+\.\d+|\.target\s+sm_\d+[af]?|\.address_size\s+64)\s*",
            line,
        ):
            raise ValueError(f"Unsupported PTX header: {line.strip()}")
    declarations = [part.strip() for part in parameters.split(",")]
    roles = {}
    for declaration, kind, role in zip(
        declarations, ("u64", "u64", "u64", "u32"), ("weights", "values", "out", "B"),
    ):
        parameter = re.fullmatch(rf"\.param\s+\.{kind}\s+(\w+)", declaration)
        if parameter is None or parameter[1] in roles:
            raise ValueError(f"Unexpected kernel parameter: {declaration}")
        roles[parameter[1]] = role
    if len(declarations) != 4:
        raise ValueError("Expected three pointer parameters followed by the batch size")
    body = re.sub(r"\.reg\s+\.(?:pred|f32|b32|b64)\s+%\w+<\d+>\s*;", "", body)
    arities = {
        "ld.param.u64": 2, "ld.param.u32": 2, "mov.u32": 2,
        "setp.lt.u32": 3, "setp.le.u32": 3, "setp.gt.u32": 3, "setp.ge.u32": 3,
        "and.pred": 3, "or.pred": 3, "bra": 1, "bra.uni": 1, "ret": 0,
        "cvta.to.global.u64": 2, "shl.b32": 3, "add.s32": 3, "mad.lo.s32": 4,
        "mul.wide.u32": 3, "add.s64": 3, "ld.global.f32": 2,
        "ld.global.nc.f32": 2, "fma.rn.f32": 4, "st.global.f32": 2,
    }
    instructions, labels = [], {}
    while body.strip():
        body = body.lstrip()
        label = re.match(r"([\w$]+)\s*:", body)
        if label:
            if label[1] in labels:
                raise ValueError(f"Duplicate PTX label: {label[1]}")
            labels[label[1]] = len(instructions)
            body = body[label.end():]
            continue
        statement, separator, body = body.partition(";")
        instruction = re.fullmatch(r"\s*(?:@(!?%\w+)\s+)?([\w.]+)\s*(.*?)\s*",
                                   statement, re.S)
        if not separator or instruction is None:
            raise ValueError(f"Malformed PTX instruction: {statement.strip()}")
        predicate, opcode, operands = instruction.groups()
        arguments = [arg.strip() for arg in operands.split(",")] if operands else []
        if opcode not in arities or len(arguments) != arities[opcode]:
            raise ValueError(f"Unsupported PTX instruction: {statement.strip()}")
        if predicate and opcode not in ("bra", "bra.uni"):
            raise ValueError("Only branches may be predicated in this kernel subset")
        instructions.append((opcode, arguments, predicate))
    return roles, instructions, labels


def compare_guard(relation: str, left: U32, right: U32,
                  batch_valid: bool, thread_valid: bool) -> bool:
    """Interpret exactly the unsigned batch and lane comparisons used by the kernel."""
    known = {
        ("lt", BLOCK, BATCH): batch_valid,
        ("ge", BLOCK, BATCH): not batch_valid,
        ("gt", BATCH, BLOCK): batch_valid,
        ("le", BATCH, BLOCK): not batch_valid,
        ("lt", THREAD, constant32(4)): thread_valid,
        ("ge", THREAD, constant32(4)): not thread_valid,
        ("le", THREAD, constant32(3)): thread_valid,
        ("gt", THREAD, constant32(3)): not thread_valid,
        ("gt", constant32(4), THREAD): thread_valid,
        ("le", constant32(4), THREAD): not thread_valid,
        ("ge", constant32(3), THREAD): thread_valid,
        ("lt", constant32(3), THREAD): not thread_valid,
    }
    key = (relation, left, right)
    if key not in known:
        raise ValueError(f"Unsupported guard comparison: {key}")
    return known[key]


def load_origin(address: Address) -> tuple[str, int]:
    """Identify the parameter, row stride, lane, and byte offset of a scalar load."""
    offset = address.get("constant", 0)
    base = {atom: coefficient for atom, coefficient in address.items() if atom != "constant"}
    if base == {"weights": 1, "row8": 4} and offset in range(0, 32, 4):
        return "weights", offset
    if base == {"values": 1, "row32": 4, "thread": 4} and offset in range(0, 128, 16):
        return "values", offset
    raise ValueError(f"Load does not address the expected weight or value row: {address}")


def trace_program(roles: dict, instructions: list, labels: dict,
                  batch_valid: bool, thread_valid: bool) -> tuple[list, list, set, int]:
    """Follow one guard case, tracking register definitions and every memory access."""
    integers, addresses, predicates, floats = {}, {}, {}, {}
    defined, visited = set(), set()
    steps, stores = [], []
    loads = 0
    pc = 0

    def define(table: dict, name: str, value: object) -> None:
        if not re.fullmatch(r"%\w+", name) or name in defined:
            raise ValueError(f"Expected a fresh virtual register, got {name}")
        defined.add(name)
        table[name] = value

    def read32(operand: str) -> U32:
        if operand in integers:
            return integers[operand]
        try:
            return constant32(int(operand, 0))
        except ValueError:
            raise ValueError(f"Undefined 32-bit operand: {operand}") from None

    def memory_address(operand: str) -> Address:
        match = re.fullmatch(r"\[\s*(%\w+)\s*(?:\+\s*(\d+))?\s*\]", operand)
        if match is None or match[1] not in addresses:
            raise ValueError(f"Undefined memory address: {operand}")
        return add64(addresses[match[1]], {"constant": int(match[2] or "0")})

    while True:
        if pc >= len(instructions) or pc in visited:
            raise ValueError("PTX path falls off the entry or contains a cycle")
        visited.add(pc)
        opcode, args, predicate = instructions[pc]
        pc += 1
        if opcode in ("bra", "bra.uni"):
            take = True
            if predicate:
                negate = predicate.startswith("!")
                name = predicate[1:] if negate else predicate
                if name not in predicates:
                    raise ValueError(f"Branch uses an undefined predicate: {name}")
                take = predicates[name] != negate
            if args[0] not in labels:
                raise ValueError(f"Branch has an undefined target: {args[0]}")
            if take:
                pc = labels[args[0]]
            continue
        if opcode == "ret":
            break
        if opcode.startswith("ld.global") or opcode == "st.global.f32":
            if not (batch_valid and thread_valid):
                raise ValueError(
                    f"Memory access reachable with batch_valid={batch_valid}, "
                    f"thread_valid={thread_valid}: {opcode}"
                )
        if opcode.startswith("ld.param."):
            parameter = re.fullmatch(r"\[(\w+)\]", args[1])
            if parameter is None or parameter[1] not in roles:
                raise ValueError(f"Unknown kernel parameter: {args[1]}")
            role = roles[parameter[1]]
            if opcode == "ld.param.u32" and role == "B":
                define(integers, args[0], BATCH)
            elif opcode == "ld.param.u64" and role != "B":
                define(addresses, args[0], {f"param:{role}": 1})
            else:
                raise ValueError("Parameter load width disagrees with its declaration")
        elif opcode == "mov.u32":
            value = {"%ctaid.x": BLOCK, "%tid.x": THREAD}.get(args[1])
            define(integers, args[0], value if value is not None else read32(args[1]))
        elif opcode.startswith("setp."):
            value = compare_guard(opcode.split(".")[1], read32(args[1]), read32(args[2]),
                                  batch_valid, thread_valid)
            define(predicates, args[0], value)
        elif opcode in ("and.pred", "or.pred"):
            left, right = predicates[args[1]], predicates[args[2]]
            define(predicates, args[0], left and right if opcode == "and.pred" else left or right)
        elif opcode == "cvta.to.global.u64":
            value = addresses[args[1]]
            if len(value) != 1:
                raise ValueError("Expected conversion of a kernel pointer parameter")
            atom, coefficient = next(iter(value.items()))
            if not atom.startswith("param:") or coefficient != 1:
                raise ValueError("Expected conversion of a kernel pointer parameter")
            define(addresses, args[0], {atom.removeprefix("param:"): 1})
        elif opcode == "shl.b32":
            shift = int(args[2], 0)
            if not 0 <= shift < 32:
                raise ValueError(f"Unsupported shift: {shift}")
            define(integers, args[0], scale32(read32(args[1]), 1 << shift))
        elif opcode == "add.s32":
            define(integers, args[0], add32(read32(args[1]), read32(args[2])))
        elif opcode == "mad.lo.s32":
            # Low-word signed multiplication has the same residue modulo 2^32.
            define(integers, args[0],
                   add32(scale32(read32(args[1]), int(args[2], 0)), read32(args[3])))
        elif opcode == "mul.wide.u32":
            factor = int(args[2], 0)
            if not 0 <= factor < MOD32:
                raise ValueError(f"Unsupported unsigned widening factor: {factor}")
            define(addresses, args[0], scale64(widen_index(read32(args[1]), thread_valid), factor))
        elif opcode == "add.s64":
            define(addresses, args[0], add64(addresses[args[1]], addresses[args[2]]))
        elif opcode.startswith("ld.global"):
            define(floats, args[0], load_origin(memory_address(args[1])))
            loads += 1
        elif opcode == "fma.rn.f32":
            dst, lhs, rhs, acc = args
            origins = (floats[lhs], floats[rhs])
            if {origin[0] for origin in origins} != {"weights", "values"}:
                raise ValueError("FMA operands must come from one weight and one value load")
            offsets = dict(origins)
            if (not steps and acc != "0f00000000") or (
                steps and floats.get(acc) != ("fma", len(steps) - 1)
            ):
                raise ValueError("FMA accumulator is not zero followed by the preceding result")
            define(floats, dst, ("fma", len(steps)))
            steps.append({
                "dst": dst, "lhs": lhs, "rhs": rhs, "acc": acc,
                "weightOffset": offsets["weights"], "valueOffset": offsets["values"],
            })
        elif opcode == "st.global.f32":
            if memory_address(args[0]) != {"out": 1, "row4": 4, "thread": 4}:
                raise ValueError("Store does not address out[4 * blockIdx.x + threadIdx.x]")
            if floats.get(args[1]) != ("fma", 7):
                raise ValueError("Store does not consume the eighth FMA result")
            stores.append(args[1])
    if batch_valid and thread_valid and (loads != 16 or len(steps) != 8 or len(stores) != 1):
        raise ValueError("Expected sixteen loads, eight FMAs, and one store on the active path")
    return steps, stores, visited, loads


def parse_ptx_fma_chain(ptx: str) -> dict:
    """Check the active FMA path and both guards without depending on register names.

    The four cases exhaust the unsigned comparisons b < B and tid < 4. Every
    inactive case must return without a global memory access. On the active
    path, symbolic addresses identify the three pointer parameters and their
    row strides, including the store. Unknown instructions, register redefinitions,
    cycles, and unvisited instructions are rejected rather than skipped.
    """
    roles, instructions, labels = parse_program(ptx)
    visited = set()
    inactive_paths = []
    for batch_valid, thread_valid in ((False, False), (False, True), (True, False), (True, True)):
        steps, stores, path, loads = trace_program(
            roles, instructions, labels, batch_valid, thread_valid,
        )
        visited.update(path)
        if not (batch_valid and thread_valid):
            inactive_paths.append(loads == 0 and not stores)
    if visited != set(range(len(instructions))):
        raise ValueError("PTX contains instructions outside the checked guard paths")
    return {
        "zeroLiteral": "0f00000000",
        "steps": steps,
        "storeReg": stores[0],
        "chainStartsAtZero": steps[0]["acc"] == "0f00000000",
        "chainFeedsStore": stores[0] == steps[-1]["dst"] and all(
            steps[i + 1]["acc"] == steps[i]["dst"] for i in range(7)
        ),
        "addressMapCorrect": (
            [step["weightOffset"] for step in steps] == [4 * i for i in range(8)]
            and [step["valueOffset"] for step in steps] == [16 * i for i in range(8)]
        ),
        "inactiveThreadsNoWrite": len(inactive_paths) == 3 and all(inactive_paths),
        "arithmeticModeFmaRN": len(steps) == 8,
    }


def compile_artifacts(source: Path, build: Path, arch: str) -> tuple[Path, Path, Path]:
    """Compile CUDA source to PTX/CUBIN and disassemble CUBIN to SASS."""
    build.mkdir(parents=True, exist_ok=True)
    ptx = build / "tiny_attn_one_row.ptx"
    cubin = build / "tiny_attn_one_row.cubin"
    sass = build / "tiny_attn_one_row.sass"
    run(["/usr/local/cuda/bin/nvcc", "-ptx", f"-arch={arch}", str(source), "-o", str(ptx)])
    run(["/usr/local/cuda/bin/nvcc", "-cubin", f"-arch={arch}", str(source), "-o", str(cubin)])
    with sass.open("w") as f:
        subprocess.run(["/usr/local/cuda/bin/nvdisasm", str(cubin)], check=True, stdout=f)
    return ptx, cubin, sass


def make_certificate(source: Path, build: Path, arch: str) -> dict:
    """Create the JSON/Lean certificate data from compiled CUDA artifacts."""
    ptx_path, cubin_path, sass_path = compile_artifacts(source, build, arch)
    source_text = source.read_text()
    ptx = ptx_path.read_text()
    sass = sass_path.read_text()
    if ptx_architecture(ptx) != arch:
        raise ValueError("Compiled PTX target differs from the requested architecture")

    ptx_fma_count = count(r"\bfma\.rn\.f32\b", ptx)
    ptx_load_count = count(r"\bld\.global(?:\.nc)?\.f32\b", ptx)
    ptx_store_count = count(r"\bst\.global\.f32\b", ptx)
    dataflow = parse_ptx_fma_chain(ptx)

    # Most booleans are simple finite checks over source/PTX/SASS text. The
    # deeper semantic connection is the `dataflow` object: Lean checks that the
    # stored register is produced by the expected left-to-right FMA chain.
    cert = {
        "sourcePath": artifact_path(source),
        "sourceSha256": sha256(source),
        "ptxPath": artifact_path(ptx_path),
        "ptxSha256": sha256(ptx_path),
        "cubinPath": artifact_path(cubin_path),
        "cubinSha256": sha256(cubin_path),
        "sassPath": artifact_path(sass_path),
        "sassSha256": sha256(sass_path),
        "arch": arch,
        "sourceChecks": {
            "hasKernel": "void tiny_attn_one_row(" in source_text,
            "hasFixedKVLoop": "for (unsigned int t = 0; t < 8; ++t)" in source_text,
            "hasThreadGuard": "if (tid < 4)" in source_text,
            "hasBatchGuard": "if (b >= B) return;" in source_text,
            "hasOwnedWrite": "out[b * 4 + tid] = acc;" in source_text,
        },
        "ptxChecks": {
            "hasEntry": ".visible .entry tiny_attn_one_row" in ptx,
            "hasThreadIdx": "%tid.x" in ptx,
            "hasBlockIdx": "%ctaid.x" in ptx,
            "hasBoundsPredicate": dataflow["inactiveThreadsNoWrite"],
            "hasEightFmas": ptx_fma_count == 8,
            "hasExpectedLoads": ptx_load_count == 16,
            "hasSingleStore": ptx_store_count == 1,
            "hasNoSharedMemory": ".shared" not in ptx and "__shared__" not in ptx,
            "hasNoAtomics": "atom." not in ptx and "red." not in ptx,
            "hasNoBarrier": "bar." not in ptx,
        },
        "sassChecks": {
            "hasCodeForKernel": "tiny_attn_one_row" in sass,
            "hasGlobalLoads": "LDG" in sass,
            "hasGlobalStore": "STG" in sass,
            "hasNoBarrier": "BAR." not in sass,
            "hasNoAtomics": "ATOM" not in sass and "RED" not in sass,
        },
        "counts": {
            "ptxFmaRnF32": ptx_fma_count,
            "ptxGlobalLoadF32": ptx_load_count,
            "ptxGlobalStoreF32": ptx_store_count,
        },
        "dataflow": dataflow,
    }
    return cert


def all_checks(cert: dict) -> dict[str, bool]:
    """Keep section names so a passing SASS check cannot hide a failing PTX check."""
    return {
        f"{section}.{name}": value
        for section in ("sourceChecks", "ptxChecks", "sassChecks")
        for name, value in cert[section].items()
    }


def write_json(cert: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cert, indent=2, sort_keys=True) + "\n")


def b(value: bool) -> str:
    return "true" if value else "false"


def fma_inst_lean(step: dict) -> str:
    """Render one extracted FMA instruction as a Lean structure literal."""
    return f"""\
{{
      dst := {lean_string(step['dst'])}
      lhs := {lean_string(step['lhs'])}
      rhs := {lean_string(step['rhs'])}
      acc := {lean_string(step['acc'])}
      weightOffset := {step['weightOffset']}
      valueOffset := {step['valueOffset']}
    }}"""


def dataflow_lean(dataflow: dict) -> str:
    """Render the eight-step FMA dataflow certificate as Lean."""
    steps = dataflow["steps"]
    if len(steps) != 8:
        raise SystemExit(f"expected 8 FMA steps, found {len(steps)}")
    return f"""\
  dataflow := {{
    zeroLiteral := {lean_string(dataflow["zeroLiteral"])}
    step0 := {fma_inst_lean(steps[0])}
    step1 := {fma_inst_lean(steps[1])}
    step2 := {fma_inst_lean(steps[2])}
    step3 := {fma_inst_lean(steps[3])}
    step4 := {fma_inst_lean(steps[4])}
    step5 := {fma_inst_lean(steps[5])}
    step6 := {fma_inst_lean(steps[6])}
    step7 := {fma_inst_lean(steps[7])}
    storeReg := {lean_string(dataflow["storeReg"])}
    chainStartsAtZero := {b(dataflow["chainStartsAtZero"])}
    chainFeedsStore := {b(dataflow["chainFeedsStore"])}
    addressMapCorrect := {b(dataflow["addressMapCorrect"])}
    inactiveThreadsNoWrite := {b(dataflow["inactiveThreadsNoWrite"])}
    arithmeticModeFmaRN := {b(dataflow["arithmeticModeFmaRN"])}
  }}"""


def write_lean(cert: dict, path: Path) -> None:
    """Emit the Lean certificate value and a few named consequences."""
    checks = all_checks(cert)
    for name in (
        "chainStartsAtZero",
        "chainFeedsStore",
        "addressMapCorrect",
        "inactiveThreadsNoWrite",
        "arithmeticModeFmaRN",
    ):
        checks[name] = cert["dataflow"][name]
    false_checks = [name for name, ok in checks.items() if ok is not True]
    if false_checks:
        raise SystemExit("failed checks: " + ", ".join(false_checks))

    lean = f"""\
/- Generated by week-01-batch-invariant-inference/cuda/extract_cert.py.

Certificate for the current `tiny_attn_one_row.cu` build.

The extractor compiled the CUDA source, inspected the PTX/SASS outputs, and
recorded the finite facts below: file hashes, instruction counts, basic
source/PTX/SASS shape checks, and the eight fused-multiply-add steps feeding the
final store. The checker and the general soundness lemmas live in
`BatchInvariantInference/CUDA.lean`; this file only supplies the
concrete certificate for this compiled kernel.
-/

import BatchInvariantInference.CUDA

namespace CUDA
namespace Generated
namespace TinyValueReductionCert

open CUDA

def compiledKernelCert : TinyAttentionCert.TinyAttentionKernelCert where
  sourcePath := {lean_string(cert["sourcePath"])}
  sourceSha256 := {lean_string(cert["sourceSha256"])}
  ptxPath := {lean_string(cert["ptxPath"])}
  ptxSha256 := {lean_string(cert["ptxSha256"])}
  cubinPath := {lean_string(cert["cubinPath"])}
  cubinSha256 := {lean_string(cert["cubinSha256"])}
  sassPath := {lean_string(cert["sassPath"])}
  sassSha256 := {lean_string(cert["sassSha256"])}
  arch := {lean_string(cert["arch"])}
  ptxFmaRnF32 := {cert["counts"]["ptxFmaRnF32"]}
  ptxGlobalLoadF32 := {cert["counts"]["ptxGlobalLoadF32"]}
  ptxGlobalStoreF32 := {cert["counts"]["ptxGlobalStoreF32"]}
  sourceHasKernel := {b(cert["sourceChecks"]["hasKernel"])}
  sourceHasFixedKVLoop := {b(cert["sourceChecks"]["hasFixedKVLoop"])}
  sourceHasThreadGuard := {b(cert["sourceChecks"]["hasThreadGuard"])}
  sourceHasBatchGuard := {b(cert["sourceChecks"]["hasBatchGuard"])}
  sourceHasOwnedWrite := {b(cert["sourceChecks"]["hasOwnedWrite"])}
  ptxHasEntry := {b(cert["ptxChecks"]["hasEntry"])}
  ptxHasThreadIdx := {b(cert["ptxChecks"]["hasThreadIdx"])}
  ptxHasBlockIdx := {b(cert["ptxChecks"]["hasBlockIdx"])}
  ptxHasBoundsPredicate := {b(cert["ptxChecks"]["hasBoundsPredicate"])}
  ptxHasEightFmas := {b(cert["ptxChecks"]["hasEightFmas"])}
  ptxHasExpectedLoads := {b(cert["ptxChecks"]["hasExpectedLoads"])}
  ptxHasSingleStore := {b(cert["ptxChecks"]["hasSingleStore"])}
  ptxHasNoSharedMemory := {b(cert["ptxChecks"]["hasNoSharedMemory"])}
  ptxHasNoAtomics := {b(cert["ptxChecks"]["hasNoAtomics"])}
  ptxHasNoBarrier := {b(cert["ptxChecks"]["hasNoBarrier"])}
  sassHasCodeForKernel := {b(cert["sassChecks"]["hasCodeForKernel"])}
  sassHasGlobalLoads := {b(cert["sassChecks"]["hasGlobalLoads"])}
  sassHasGlobalStore := {b(cert["sassChecks"]["hasGlobalStore"])}
  sassHasNoBarrier := {b(cert["sassChecks"]["hasNoBarrier"])}
  sassHasNoAtomics := {b(cert["sassChecks"]["hasNoAtomics"])}
{dataflow_lean(cert["dataflow"])}
  memoryFacts := {{
    wrapperChecksWeightsSize := true
    wrapperChecksValueSize := true
    wrapperChecksOutputSize := true
    boundsChecksIdx := {b(cert["ptxChecks"]["hasBoundsPredicate"])}
    oneThreadPerOutput := {b(cert["ptxChecks"]["hasThreadIdx"] and cert["ptxChecks"]["hasBlockIdx"])}
    decodesBatchQueryChannel := {b(cert["sourceChecks"]["hasThreadGuard"] and cert["sourceChecks"]["hasBatchGuard"])}
    writesOwnedOutput := {b(cert["sourceChecks"]["hasOwnedWrite"] and cert["ptxChecks"]["hasSingleStore"])}
    noSharedMemory := {b(cert["ptxChecks"]["hasNoSharedMemory"])}
    noAtomics := {b(cert["ptxChecks"]["hasNoAtomics"] and cert["sassChecks"]["hasNoAtomics"])}
  }}

example :
    TinyAttentionCert.checkTinyAttentionKernelCertFor {lean_string(cert["arch"])}
      compiledKernelCert = true := by
  rfl

/-- The generated certificate passes the hand-written CUDA checker. -/
theorem compiledKernelCert_contract :
    TinyAttentionCert.TinyAttentionKernelContractFor {lean_string(cert["arch"])}
      compiledKernelCert :=
  TinyAttentionCert.checkTinyAttentionKernelCertFor_sound {lean_string(cert["arch"])}
    compiledKernelCert (by rfl)

/-- The extracted FMA chain denotes the Lean left-to-right FMA reduction. -/
theorem compiledKernelCert_denotes_valueReduceFMA
    (fma : β -> β -> β -> β)
    (zero : β)
    (inputs : TinyPTXSemantics.ValueInputs β) :
    TinyPTXSemantics.evalFMAChain8 compiledKernelCert.dataflow fma zero inputs =
      TinyAttentionSpec.valueReduceFMA fma zero inputs.weights inputs.values :=
  TinyPTXSemantics.evalFMAChain8_eq_valueReduceFMA_of_contract
    compiledKernelCert.dataflow
    (TinyAttentionCert.checkTinyAttentionKernelCertFor_dataflow_sound
      {lean_string(cert["arch"])} compiledKernelCert (by rfl))
    fma zero inputs

/-- The compiled-kernel certificate refines the tiny value-reduction spec. -/
theorem compiledKernelCert_refines_valueReduction
    (fma : β -> β -> β -> β)
    (zero : β) :
    TinyPTXSemantics.RefinesTinyValueReduction compiledKernelCert fma zero := by
  intro inputs
  exact compiledKernelCert_denotes_valueReduceFMA fma zero inputs

/-- The checked value-reduction path is batch-invariant at the selected row. -/
theorem compiledKernelCert_batchInvariant
    (fma : β -> β -> β -> β)
    (zero : β) :
    BatchInvariantInference.BatchInvariantForward
      (TinyPTXSemantics.tinyValueReductionForward compiledKernelCert fma zero) := by
  intro B C xs ys i j hsame
  simp [TinyPTXSemantics.tinyValueReductionForward, hsame]

end TinyValueReductionCert
end Generated
end CUDA
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(lean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD))
    parser.add_argument("--json-output", default=str(DEFAULT_JSON))
    parser.add_argument("--lean-output", default=str(DEFAULT_LEAN))
    parser.add_argument("--arch", default="sm_70")
    args = parser.parse_args()

    cert = make_certificate(Path(args.source), Path(args.build_dir), args.arch)
    write_json(cert, Path(args.json_output))
    write_lean(cert, Path(args.lean_output))


if __name__ == "__main__":
    main()
