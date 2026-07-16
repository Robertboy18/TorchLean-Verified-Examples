import Lake
open Lake DSL
open Lean

/-- Forward native-backend options to the TorchLean dependency. -/
private def torchLeanOptions : NameMap String :=
  let opts : NameMap String := {}
  let opts := match get_config? cuda with
    | some value => opts.insert `cuda value
    | none => opts
  let opts := match get_config? cuda_home with
    | some value => opts.insert `cuda_home value
    | none => opts
  let opts := match get_config? libtorch with
    | some value => opts.insert `libtorch value
    | none => opts
  match get_config? libtorch_home with
  | some value => opts.insert `libtorch_home value
  | none => opts

/-- CUDA libraries needed when this downstream package links a TorchLean executable. -/
private def nativeLinkArgs : Array String :=
  let cudaEnabled := match get_config? cuda with
    | some value => value == "true" || value == "1"
    | none => false
  if cudaEnabled then
    let cudaHome := (get_config? cuda_home).getD "/usr/local/cuda"
    #[
      "-L", s!"{cudaHome}/lib64", "-lcudart", "-lcublas", "-lcufft",
      "-Wl,-rpath," ++ s!"{cudaHome}/lib64"
    ]
  else
    #[]

/-!
Shared Lake package for the TorchLean verified examples repository.

The examples live in week folders, but they intentionally share one Lake
project and one TorchLean dependency. Each week we can add modules under this same
package instead of creating a separate Lake project.
-/

package TorchLeanVerifiedExamples where
  version := v!"0.1.0"
  description := "Weekly TorchLean examples with checked Lean developments."
  moreLinkArgs := nativeLinkArgs

@[default_target]
lean_lib BatchInvariantInference where
  srcDir := "week-01-batch-invariant-inference"
  roots := #[
    `BatchInvariantInference.Core,
    `BatchInvariantInference.CUDA,
    `BatchInvariantInference.Generated
  ]
  defaultFacets := #[LeanLib.staticFacet]

@[default_target]
lean_lib VerifiableTransformers where
  srcDir := "week-02-verifiable-transformer-checkpoint"
  roots := #[
    `VerifiableTransformers,
    `VerifiableTransformers.Replay.UpstreamFloatReplay
  ]
  defaultFacets := #[LeanLib.staticFacet]

lean_exe verify_upstream_forward where
  srcDir := "week-02-verifiable-transformer-checkpoint"
  root := `VerifiableTransformers.Replay.UpstreamFloatReplay

lean_exe train_torchlean_small_gpt where
  srcDir := "week-02-verifiable-transformer-checkpoint"
  root := `VerifiableTransformers.TorchLean.TrainSmallGPT

require TorchLean from git
  "https://github.com/lean-dojo/TorchLean.git" @ "main" with torchLeanOptions
