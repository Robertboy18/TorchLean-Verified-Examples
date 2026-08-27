/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Run

/-!
# Training executable

This thin module gives Lake a conventional root-level `main` while keeping the training and
generation implementation importable by the standalone interactive runner.
-/

@[expose] public section

def main (args : List String) : IO UInt32 :=
  TorchLeanGPT.Run.main args
