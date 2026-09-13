import Lake
open Lake DSL
open Lean
open System

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
  let opts := match get_config? libtorch_home with
    | some value => opts.insert `libtorch_home value
    | none => opts
  match get_config? torchleanBuildDir with
  | some value => opts.insert `torchleanBuildDir value
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
  else if Platform.isWindows || Platform.isOSX then
    #[]
  else
    #["-lm", "-lstdc++"]

/-- Whether the Week 3 cached decoder should compile its CUDA implementation or host stub. -/
private def cudaEnabled : Bool :=
  match get_config? cuda with
  | some value => value == "true" || value == "1"
  | none => false

/-- Build the example-local key/value-cache runtime for the active device configuration. -/
private def buildCachedDecoder (pkg : Package) : FetchM (Job FilePath) := do
  let lean ← getLeanInstall
  let some torchLean ← findPackageByName? `TorchLean
    | error "The cached decoder requires the TorchLean package."
  let includeArgs := #[
    "-I", lean.includeDir.toString,
    "-I", (torchLean.dir / "csrc/cuda/common").toString
  ]
  let libFile := pkg.buildDir / nameToStaticLib "torchlean_gpt_cached_decode"
  if cudaEnabled then
    let cudaHome := (get_config? cuda_home).getD "/usr/local/cuda"
    let source ← inputFile
      (pkg.dir /
        "week-03-gpt-training/csrc/cached_decode/torchlean_gpt_cached_decode.cu") false
    let objectFile := pkg.buildDir / "torchlean_gpt_cached_decode.o"
    let object ← buildO objectFile source
      (includeArgs ++ #[
        "-I", s!"{cudaHome}/include",
        "-c", "--std=c++17", "-O3", "-Xcompiler", "-fPIC"
      ]) #[] "nvcc"
    buildStaticLib libFile #[object]
  else
    let source ← inputFile
      (pkg.dir /
        "week-03-gpt-training/csrc/cached_decode/torchlean_gpt_cached_decode_stub.c") false
    let objectFile := pkg.buildDir / "torchlean_gpt_cached_decode_stub.o"
    let object ← buildO objectFile source
      (includeArgs ++ #["-O2", "-fPIC"]) #[] "cc"
    buildStaticLib libFile #[object]

/-!
Shared Lake package for the TorchLean verified examples repository.

The examples live in week folders, but they intentionally share one Lake
project and one TorchLean dependency. Each week we can add modules under this same
package instead of creating a separate Lake project.
-/

package TorchLeanVerifiedExamples where
  buildDir := FilePath.mk ((get_config? verifiedExamplesBuildDir).getD ".lake/build")
  version := v!"0.1.0"
  description := "Weekly TorchLean examples with checked Lean developments."
  moreLinkArgs := nativeLinkArgs

/-- Week 3's stateful key/value cache; this is not part of the TorchLean core runtime. -/
extern_lib torchlean_gpt_cached_decode (pkg) :=
  buildCachedDecoder pkg

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

@[default_target]
lean_lib TorchLeanGPT where
  srcDir := "week-03-gpt-training"
  roots := #[`TorchLeanGPT]
  defaultFacets := #[LeanLib.staticFacet]

@[default_target]
lean_lib KimiK3 where
  srcDir := "week-04-kimi-k3-specification"
  roots := #[`KimiK3]
  defaultFacets := #[LeanLib.staticFacet]

lean_exe train_torchlean_gpt where
  srcDir := "week-03-gpt-training"
  root := `TorchLeanGPT.Train

lean_exe generate_torchlean_gpt where
  srcDir := "week-03-gpt-training"
  root := `TorchLeanGPT.Generate

lean_exe generate_torchlean_gpt_cached where
  srcDir := "week-03-gpt-training"
  root := `TorchLeanGPT.CachedGenerate

lean_exe check_torchlean_gpt_cache where
  srcDir := "week-03-gpt-training"
  root := `TorchLeanGPT.CachedDecode.Check

lean_exe benchmark_torchlean_gpt_cache where
  srcDir := "week-03-gpt-training"
  root := `TorchLeanGPT.CachedDecode.Benchmark

require TorchLean from git
  "https://github.com/lean-dojo/TorchLean.git" @
    "4ec1f62bf8308e2dc7f4d73e64205e66270ccfd1" with torchLeanOptions

require LeanProfiler from git
  "https://github.com/Robertboy18/LeanProfiler.git" @
    "1008fc1dd7c416d0d225383f75ee5e581ad55712"
