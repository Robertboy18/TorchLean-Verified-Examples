import Lake
open Lake DSL

/-!
The benchmarks of this week run against FloatLib. This Lake project is separate from the
repository's root project, which follows TorchLean on another toolchain: `lake build` here fetches
FloatLib's main branch, with the Mathlib that FloatLib pins, and `lake env lean FILE` then checks
a generated benchmark file against it.
-/

package floatlibBenchmarks where
  leanOptions := #[⟨`autoImplicit, false⟩, ⟨`relaxedAutoImplicit, false⟩]

require floatlib from git "https://github.com/lean-dojo/FloatLib" @ "main"

-- A first check that the library is there: one calculation proved in Lean's kernel.
@[default_target]
lean_lib FloatLibBenchmarks

-- The compiled replay of `vnncomp-replay/`: binary32 and exact evaluation of a network at a point.
lean_exe replay where
  srcDir := "vnncomp-replay/scripts"
  root := `Replay
