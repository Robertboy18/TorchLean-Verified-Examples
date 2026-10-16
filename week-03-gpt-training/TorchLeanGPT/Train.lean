/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Run

/-!
# Training executable

Root-level `main` for `train_torchlean_gpt`. The implementation lives in `TorchLeanGPT.Run`,
which the generation executables import as well.
-/

@[expose] public section

def main (args : List String) : IO UInt32 :=
  TorchLeanGPT.Run.main args
