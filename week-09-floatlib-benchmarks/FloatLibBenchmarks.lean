import FloatLib

/-!
# FloatLib's benchmarks: a first check

`lake build` in this directory builds FloatLib and this file. Each benchmark suite then runs its
generated files with `lake env lean` from its own `scripts/` directory.
-/

open FloatLib.Floats

/-- The calculation from FloatLib's README: Lean's kernel runs the binary32 addition itself. -/
example : (1.5 : Binary32) + 2.25 = 3.75 := by float_decide
