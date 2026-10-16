/-
Copyright (c) 2026 Robert Joseph George
Released under the MIT license.
-/

module

public import TorchLeanGPT.Model
public import TorchLeanGPT.CausalTraining
public import TorchLeanGPT.DialogueRecords
public import TorchLeanGPT.Batching
public import TorchLeanGPT.Validation
public import TorchLeanGPT.CachedDecode.Correctness
public import TorchLeanGPT.CachedDecode.Layered
public import TorchLeanGPT.CachedDecode.Runtime
public import TorchLeanGPT.GenerationCorrectness
public import TorchLeanGPT.TrainingCorrectness
public import TorchLeanGPT.Run

/-!
# Week 3: GPT-style training

`TorchLeanGPT.Model` defines the causal Transformer used by the experiment. `TorchLeanGPT.Run`
builds the training batches and proves, for the two batch builders the trainer calls, that every
target is the next corpus token and that the assistant-only loss mask follows the target.
`TorchLeanGPT.CausalTraining` proves causal isolation in the forward and backward attention
equations and the zero-weight-row property of the weighted objective. `TorchLeanGPT.DialogueRecords`
proves that selected instruction-tuning targets stay inside their bounded dialogue windows and that
padded rows are never selected, while `TorchLeanGPT.Batching` and `TorchLeanGPT.Validation` prove
the loader's row-packing and shard-scanning loops.

`TorchLeanGPT.CachedDecode.Correctness` proves that incremental key-value caching returns the same
causal outputs as full-prefix recomputation. `TorchLeanGPT.CachedDecode.Layered` lifts that result
to a decoder stack with one cache per layer, each holding one key/value row per consumed state.
`TorchLeanGPT.GenerationCorrectness` turns certified logit-error bounds into equality of complete
greedy continuations, and `TorchLeanGPT.TrainingCorrectness` proves the indexed state-splitting law
needed by exact checkpoint resumption. Training and text generation are separate executables.
-/
