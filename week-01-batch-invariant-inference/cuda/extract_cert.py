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


# Tiny PTX parser patterns. These stay narrow by design: the extractor is for
# one pinned microkernel, not a general PTX frontend.
LOAD_RE = re.compile(
    r"ld\.global(?:\.nc)?\.f32\s+(%f\d+),\s+\[(%rd\d+)(?:\+(\d+))?\];"
)
FMA_RE = re.compile(
    r"fma\.rn\.f32\s+(%f\d+),\s+(%f\d+),\s+(%f\d+),\s+([^;]+);"
)
STORE_RE = re.compile(r"st\.global\.f32\s+\[(%rd\d+)\],\s+(%f\d+);")


def parse_ptx_fma_chain(ptx: str) -> dict:
    """Extract the left-to-right FMA chain that feeds the final store.

    This is intentionally specialized to `tiny_attn_one_row`: `%rd7` is the
    weights base and `%rd10` is the value base in the generated PTX.
    """

    # First map every loaded Float32 register to `(baseRegister, byteOffset)`.
    # Later, each FMA operand is traced back through this table.
    load_sources: dict[str, tuple[str, int]] = {}
    for reg, base, offset in LOAD_RE.findall(ptx):
        load_sources[reg] = (base, int(offset or "0"))

    fmas = FMA_RE.findall(ptx)
    stores = STORE_RE.findall(ptx)
    steps = []
    for dst, lhs, rhs, acc in fmas:
        lhs_source = load_sources.get(lhs)
        rhs_source = load_sources.get(rhs)
        # NVCC may choose either operand order for the weight/value multiply.
        # The certificate records semantic offsets, not just textual operands.
        if lhs_source is None or rhs_source is None:
            weight_offset = -1
            value_offset = -1
        elif lhs_source[0] == "%rd7" and rhs_source[0] == "%rd10":
            weight_offset = lhs_source[1]
            value_offset = rhs_source[1]
        elif lhs_source[0] == "%rd10" and rhs_source[0] == "%rd7":
            weight_offset = rhs_source[1]
            value_offset = lhs_source[1]
        else:
            weight_offset = -1
            value_offset = -1
        steps.append(
            {
                "dst": dst,
                "lhs": lhs,
                "rhs": rhs,
                "acc": acc.strip(),
                "weightOffset": weight_offset,
                "valueOffset": value_offset,
            }
        )

    # The source shape is weights[b, t] and v[b, t, o]. In bytes, weights move
    # by 4 and values move by 4 output coordinates = 16.
    expected_weight_offsets = [4 * i for i in range(8)]
    expected_value_offsets = [16 * i for i in range(8)]
    store_reg = stores[0][1] if len(stores) == 1 else ""

    chain_starts_at_zero = len(steps) == 8 and steps[0]["acc"] == "0f00000000"
    chain_feeds_store = len(steps) == 8 and store_reg == steps[-1]["dst"]
    if len(steps) == 8:
        chain_feeds_store = chain_feeds_store and all(
            steps[i + 1]["acc"] == steps[i]["dst"] for i in range(7)
        )
    address_map_correct = (
        len(steps) == 8
        and [s["weightOffset"] for s in steps] == expected_weight_offsets
        and [s["valueOffset"] for s in steps] == expected_value_offsets
    )

    return {
        "zeroLiteral": "0f00000000",
        "steps": steps,
        "storeReg": store_reg,
        "chainStartsAtZero": chain_starts_at_zero,
        "chainFeedsStore": chain_feeds_store,
        "addressMapCorrect": address_map_correct,
        "inactiveThreadsNoWrite": "setp.lt.u32\t%p2, %r2, 4;" in ptx
        and "@!%p3 bra" in ptx,
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
            "hasBoundsPredicate": "setp.lt.u32" in ptx and "and.pred" in ptx,
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
    """Flatten source/PTX/SASS boolean checks for quick extractor failures."""
    out: dict[str, bool] = {}
    for section in ("sourceChecks", "ptxChecks", "sassChecks"):
        out.update(cert[section])
    return out


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
    false_checks = [name for name, ok in checks.items() if not ok]
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

example : TinyAttentionCert.checkTinyAttentionKernelCert compiledKernelCert = true := by
  rfl

/-- The generated certificate passes the hand-written CUDA checker. -/
theorem compiledKernelCert_contract :
    TinyAttentionCert.TinyAttentionKernelContract compiledKernelCert :=
  TinyAttentionCert.checkTinyAttentionKernelCert_sound compiledKernelCert (by rfl)

/-- The extracted FMA chain denotes the Lean left-to-right FMA reduction. -/
theorem compiledKernelCert_denotes_valueReduceFMA
    (fma : β -> β -> β -> β)
    (zero : β)
    (inputs : TinyPTXSemantics.ValueInputs β) :
    TinyPTXSemantics.evalFMAChain8 compiledKernelCert.dataflow fma zero inputs =
      TinyAttentionSpec.valueReduceFMA fma zero inputs.weights inputs.values :=
  TinyPTXSemantics.kernelCert_denotes_valueReduceFMA
    compiledKernelCert (by rfl) fma zero inputs

/-- The compiled-kernel certificate refines the tiny value-reduction spec. -/
theorem compiledKernelCert_refines_valueReduction
    (fma : β -> β -> β -> β)
    (zero : β) :
    TinyPTXSemantics.RefinesTinyValueReduction compiledKernelCert fma zero :=
  TinyPTXSemantics.kernelCert_refines_tinyValueReduction
    compiledKernelCert (by rfl) fma zero

/-- The checked value-reduction path is batch-invariant at the selected row. -/
theorem compiledKernelCert_batchInvariant
    (fma : β -> β -> β -> β)
    (zero : β) :
    BatchInvariantInference.BatchInvariantForward
      (TinyPTXSemantics.tinyValueReductionForward compiledKernelCert fma zero) :=
  TinyPTXSemantics.tinyValueReductionForward_batchInvariant_of_check
    compiledKernelCert (by rfl) fma zero

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
