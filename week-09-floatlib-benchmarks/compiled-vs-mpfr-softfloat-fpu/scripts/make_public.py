"""Generate the public-operation benchmark from FloatLib's format comparison.

    python3 -I make_public.py FLOATLIB_CHECKOUT BUILD_DIR OUT_DIR [--low-word-128]

FloatLib's `benchmarks/lean/FloatLibBenchmarks/Public/FormatComparison.lean` times, at 32 and 64
bits, the fixed-word kernels by name (`Configured.Backend.wordAdd`, ...), and at every other width the
public operation (`ExecFloat.add`, ...) on the configured type. This script writes a copy of that file,
`PublicOperations.lean`, with the same loops, inputs, sink and CSV row, in which

  * the matrix has the generic (public-operation) row macro at binary32 (`layout 8 23`, which is the
    type `Binary32`), binary64 (`layout 11 52`, `Binary64`) and binary128 (the last only to print
    which kernel the default policy selects there), execution class `proved-software`;
  * two more row macros, copies of FloatLib's two, take the function that observes a result as
    a parameter: `binary_public_word_format` (the public operations, class
    `proved-software-word-token`) and `binary_kernel_word_format` (the fixed-word kernels by name,
    class `proved-kernel-word-token`). Their rows at binary32 and binary64 observe a result with
    `BinarySupport.wordResultBits`, which computes the fingerprint of `BinarySupport.resultBits` on
    machine words with the format's constants as literals, as `benchmark_protocol.h` does for the C
    adapters; `resultBits` computes the same value with `Nat` and `Int` arithmetic on the format
    descriptor in every iteration. The agreement prefix of every row is checked against MPFR;
  * the scope `ExecFloat.Backend.PlanningThroughput` is not opened, so the operations are planned with
    the default (balanced) policy, as a call in a user's file is;
  * the namespace is `FloatLibBenchmarks.Public.PublicOperations` (the module is `PublicOperations`).

With `--low-word-128` the executable has one more row, binary128 with the public operations and the
observation `BinarySupport.lowWordBits` (the low word of the encoding, which is what `resultBits`
returns above 64 bits, without its test of the format's width), class `proved-software-word-token`.
It is the check of how much of the harness's own FloatLib series at 128 bits is its observation.

Nothing else changes. It also writes the module setup file for `lean` (FloatLib's, with the
module name changed) and the linker response file (FloatLib's, with the object of
`FormatComparison` replaced by the object of `PublicOperations`). Nothing is written in the checkout.
"""
import json
import sys
from pathlib import Path

checkout, build, out = (Path(argument).resolve() for argument in sys.argv[1:4])
low_word_128 = "--low-word-128" in sys.argv[4:]
source = (checkout / "benchmarks/lean/FloatLibBenchmarks/Public/FormatComparison.lean").read_text()

scope = "open scoped FloatLib.Floats.ExecFloat.Backend.PlanningThroughput\n"
assert source.count(scope) == 1
source = source.replace(scope, "")

old_namespace = "FloatLibBenchmarks.Public.FormatComparison"
assert source.count(f"namespace {old_namespace}\n") == 1 and source.count(f"end {old_namespace}\n") == 1
source = source.replace(f"namespace {old_namespace}\n", "namespace FloatLibBenchmarks.Public.PublicOperations\n")
source = source.replace(f"end {old_namespace}\n", "end FloatLibBenchmarks.Public.PublicOperations\n")
assert source.count(f"{old_namespace}.run") == 1
source = source.replace(f"{old_namespace}.run", "FloatLibBenchmarks.Public.PublicOperations.run")

# The observation on machine words, after `resultBits`.
anchor = "end BinarySupport\n"
assert source.count(anchor) == 1
word_token = """/--
The value of `resultBits` for a format of at most 64 bits, computed on machine words: the widths and
the bias are literals and the exponent is kept modulo `2 ^ 64`, which is what `UInt64.ofInt` returns.
-/
@[inline] def wordResultBits
    {F : Type} [EncodedFormat F]
    [carrier : BenchmarkCarrier F FloatFormat
      FloatLib.Floats.Formats.BinaryInterchange.Model]
    (fracWidth expWidth bias : UInt64) (value : ExecFloat F) : UInt64 :=
  let bits := carrier.lowBits value
  let fractionMask := (1 <<< fracWidth) - 1
  let exponentMask := (1 <<< expWidth) - 1
  let fraction := bits &&& fractionMask
  let exponentField := (bits >>> fracWidth) &&& exponentMask
  let sign := if ((bits >>> (fracWidth + expWidth)) &&& 1) != 0 then signMix else 0
  if exponentField == exponentMask then
    if fraction == 0 then infinityTag ^^^ sign else nanTag
  else if exponentField == 0 then
    if fraction == 0 then
      zeroTag ^^^ sign
    else
      let shift := fracWidth - UInt64.ofNat fraction.toNat.log2
      (fraction <<< shift) ^^^ ((2 - bias - shift) * exponentMix) ^^^ sign
  else
    ((1 <<< fracWidth) ||| fraction) ^^^ ((exponentField - bias + 1) * exponentMix) ^^^ sign

/-- `wordResultBits` with the constants of binary32. -/
@[inline] def wordResultBits32
    {F : Type} [EncodedFormat F]
    [BenchmarkCarrier F FloatFormat FloatLib.Floats.Formats.BinaryInterchange.Model]
    (value : ExecFloat F) : UInt64 :=
  wordResultBits 23 8 127 value

/-- `wordResultBits` with the constants of binary64. -/
@[inline] def wordResultBits64
    {F : Type} [EncodedFormat F]
    [BenchmarkCarrier F FloatFormat FloatLib.Floats.Formats.BinaryInterchange.Model]
    (value : ExecFloat F) : UInt64 :=
  wordResultBits 52 11 1023 value

/-- The value of `resultBits` above 64 bits: the low word of the encoding. -/
@[inline] def lowWordBits
    {F : Type} [EncodedFormat F]
    [carrier : BenchmarkCarrier F FloatFormat FloatLib.Floats.Formats.BinaryInterchange.Model]
    (value : ExecFloat F) : UInt64 :=
  carrier.lowBits value

"""
source = source.replace(anchor, word_token + anchor)

# The three execution classes of this executable.
check = 'unless executionClass = "proved-software" do'
assert source.count(check) == 1
source = source.replace(check, 'unless executionClass ∈ ["proved-software", "proved-software-word-token",\n'
                               '        "proved-kernel-word-token"] do')


def with_token(block, keyword, new_keyword, execution_class):
    """A copy of one row macro whose observation is a parameter and whose rows have another class."""
    head = f'elab "{keyword} " stem:ident " with " runner:ident " named "\n' \
           '    formatName:term " at " width:num " layout " exponentBits:num fractionBits:num :\n'
    assert block.count(head) == 1, keyword
    block = block.replace(head, f'elab "{new_keyword} " stem:ident " with " runner:ident " named "\n'
                                '    formatName:term " at " width:num " layout " exponentBits:num fractionBits:num\n'
                                '    " observed_by " observer:ident :\n')
    assert block.count("observing BinarySupport.resultBits") == 6
    block = block.replace("observing BinarySupport.resultBits", "observing $observer")
    assert block.count('"proved-software"') == 6
    return block.replace('"proved-software"', f'"{execution_class}"')


public_start = source.index('open Lean Elab Command in\nelab "binary_comparison_format "')
kernel_comment = source.index("/-\nGenerate binary workloads that exercise the proved fixed-word kernels by name")
kernel_start = source.index('open Lean Elab Command in\nelab "binary_word_comparison_format "')
matrix = source.index("/-!\n## Compiled comparison matrix")
public_block, kernel_block = source[public_start:kernel_comment], source[kernel_start:matrix]
source = (source[:matrix]
          + with_token(public_block, "binary_comparison_format", "binary_public_word_format",
                       "proved-software-word-token")
          + with_token(kernel_block, "binary_word_comparison_format", "binary_kernel_word_format",
                       "proved-kernel-word-token")
          + source[matrix:])

start = source.index("posit_comparison_format posit2 with runPosit2 at 2")
stop = source.index("def run : IO Unit := do")
rows = """binary_comparison_format binary32Public with runBinary32Public
  named "binary32" at 32 layout 8 23
binary_comparison_format binary64Public with runBinary64Public
  named "binary64" at 64 layout 11 52
binary_comparison_format binary128Public with runBinary128Public
  named "binary128" at 128 layout 15 112
binary_public_word_format binary32PublicWord with runBinary32PublicWord
  named "binary32" at 32 layout 8 23 observed_by BinarySupport.wordResultBits32
binary_public_word_format binary64PublicWord with runBinary64PublicWord
  named "binary64" at 64 layout 11 52 observed_by BinarySupport.wordResultBits64
binary_kernel_word_format binary32KernelWord with runBinary32KernelWord
  named "binary32" at 32 layout 8 23 observed_by BinarySupport.wordResultBits32
binary_kernel_word_format binary64KernelWord with runBinary64KernelWord
  named "binary64" at 64 layout 11 52 observed_by BinarySupport.wordResultBits64

/-- Ordered rows emitted by the public-call executable. -/
private def benchmarkRows : List (Filters → IO Unit) :=
  [ runBinary32Public
  , runBinary64Public
  , runBinary128Public
  , runBinary32PublicWord
  , runBinary64PublicWord
  , runBinary32KernelWord
  , runBinary64KernelWord
  ]

"""
if low_word_128:
    rows = rows.replace("\n/-- Ordered rows emitted", """binary_public_word_format binary128PublicWord with runBinary128PublicWord
  named "binary128" at 128 layout 15 112 observed_by BinarySupport.lowWordBits

/-- Ordered rows emitted""").replace("  , runBinary64KernelWord\n", "  , runBinary64KernelWord\n  , runBinary128PublicWord\n")
    assert rows.count("runBinary128PublicWord") == 2
source = source[:start] + rows + source[stop:]
out.mkdir(parents=True, exist_ok=True)
(out / "PublicOperations.lean").write_text(source)

setup = json.loads((build / "ir/FloatLibBenchmarks/Public/FormatComparison.setup.json").read_text())
assert setup["name"] == old_namespace
setup["name"] = "PublicOperations"
(out / "PublicOperations.setup.json").write_text(json.dumps(setup))

response = (build / "bin/execFloatFormatComparison.rsp").read_text().split("\n")
target = f'"{build}/ir/FloatLibBenchmarks/Public/FormatComparison.c.o.export"'
assert response.count(target) == 1
response[response.index(target)] = f'"{out}/PublicOperations.c.o.export"'
(out / "publicOperations.rsp").write_text("\n".join(response))
print(out / "PublicOperations.lean")
