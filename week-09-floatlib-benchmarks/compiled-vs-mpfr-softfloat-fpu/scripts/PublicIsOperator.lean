import FloatLib

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

/-! The operators of `Binary32` and `Binary64` are the functions that the benchmark rows call. -/

example (x y : Binary32) : x + y = ExecFloat.add x y := rfl
example (x y : Binary32) : x - y = ExecFloat.sub x y := rfl
example (x y : Binary32) : x * y = ExecFloat.mul x y := rfl
example (x y : Binary32) : x / y = ExecFloat.div x y := rfl
example (x y : Binary64) : x + y = ExecFloat.add x y := rfl
example (x y : Binary64) : x - y = ExecFloat.sub x y := rfl
example (x y : Binary64) : x * y = ExecFloat.mul x y := rfl
example (x y : Binary64) : x / y = ExecFloat.div x y := rfl
example : Binary32 = ExecFloat.Binary (exponentBits := 8) (fractionBits := 23) := rfl
example : Binary64 = ExecFloat.Binary (exponentBits := 11) (fractionBits := 52) := rfl
#check (ExecFloat.sqrt : Binary32 → Binary32)
#check (ExecFloat.fma : Binary64 → Binary64 → Binary64 → Binary64)
