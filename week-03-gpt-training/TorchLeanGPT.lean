/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Model
public import TorchLeanGPT.CausalTraining
public import TorchLeanGPT.DialogueRecords
public import TorchLeanGPT.CachedDecode.Correctness
public import TorchLeanGPT.CachedDecode.Layered
public import TorchLeanGPT.CachedDecode.Runtime
public import TorchLeanGPT.GenerationCorrectness
public import TorchLeanGPT.TrainingCorrectness

/-!
# Week 3: GPT-style training

`TorchLeanGPT.Model` defines the causal Transformer used by the experiment.
`TorchLeanGPT.CausalTraining` proves next-token target alignment and causal isolation in the forward
and backward attention equations. `TorchLeanGPT.DialogueRecords` proves that selected instruction-
tuning targets remain inside their bounded dialogue windows and excludes padded rows.
`TorchLeanGPT.CachedDecode.Correctness` proves that incremental
key-value caching returns the same causal outputs as full-prefix recomputation, while
`TorchLeanGPT.CachedDecode.Layered` lifts that result to a complete decoder stack with one cache per
layer and proves that every layer stores one key/value row per consumed state.
`TorchLeanGPT.GenerationCorrectness` turns certified logit-error bounds into equality of complete
greedy continuations. `TorchLeanGPT.TrainingCorrectness` proves the indexed state-splitting law
needed by exact checkpoint resumption. Training and text generation are separate executables.
-/
