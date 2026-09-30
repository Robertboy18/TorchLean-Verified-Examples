import Lake
open Lake DSL
open Lean
open System

/-- Forward native-backend options to the TorchLean dependency. -/
private def torchLeanOptions : NameMap String :=
  let opts : NameMap String := {}
  let opts := match get_config? torchleanBuildDir with
    | some value => opts.insert `torchleanBuildDir value
    | none => opts
  let opts := match get_config? cuda with
    | some value => opts.insert `cuda value
    | none => opts
  let opts := match get_config? cuda_home with
    | some value => opts.insert `cuda_home value
    | none => opts
  let opts := match get_config? libtorch_home with
    | some value => opts.insert `libtorch_home value
    | none => opts
  opts

/-- LibTorch's SDK dependencies are carried by its shared libraries. -/
private def nativeLinkArgs : Array String :=
  if Platform.isWindows || Platform.isOSX then
    #[]
  else
    #["-lm"]

/-- Whether the cached decoder can use TorchLean's LibTorch storage. -/
private def cudaEnabled : Bool :=
  match get_config? cuda with
  | some value => value == "true" || value == "1"
  | none => false

/-- Build the example-local key/value-cache runtime for the active device configuration. -/
private def buildCachedDecoder (pkg : Package) : FetchM (Job FilePath) := do
  let lean ← getLeanInstall
  let some torchLean ← findPackageByName? `TorchLean
    | error "the cached decoder requires the TorchLean dependency"
  let includeArgs := #["-I", lean.includeDir.toString]
  let libFile := pkg.buildDir / nameToStaticLib "torchlean_gpt_cached_decode"
  if cudaEnabled then
    let backend ← torchLean.fetchTargetJob `torchlean_libtorch
    backend.mapM fun _ => do
      let sourceDir ← IO.FS.realPath (pkg.dir / "week-03-gpt-training/csrc/cached_decode")
      let buildDir := pkg.buildDir / "cached-decode"
      IO.FS.createDirAll buildDir
      let buildDir ← IO.FS.realPath buildDir
      let torchSource ← IO.FS.realPath torchLean.dir
      let backendFile ← IO.FS.realPath
        (torchLean.buildDir / "libtorch" / nameToSharedLib "torchlean_libtorch")
      let mut resolveArgs := #[(torchSource / "scripts/libtorch_build.py").toString,
        "--resolve-home"]
      if let some home := get_config? libtorch_home then
        resolveArgs := resolveArgs.push s!"--libtorch-home={home}"
      let home ← captureProc { cmd := "python3", args := resolveArgs }
      let cmake := (← IO.getEnv "TORCHLEAN_CMAKE").getD "cmake"
      let fallbackCompiler ← IO.getEnv "CXX"
      let compiler := ((← IO.getEnv "TORCHLEAN_CXX").orElse
        (fun _ => fallbackCompiler)).getD "c++"
      let mut args := #["-S", sourceDir.toString, "-B", buildDir.toString,
        s!"-DTORCHLEAN_LIBTORCH_HOME={home.trimAscii.toString}",
        s!"-DTORCHLEAN_LEAN_INCLUDE={lean.includeDir}",
        s!"-DTORCHLEAN_SOURCE={torchSource}",
        s!"-DTORCHLEAN_BACKEND={backendFile}",
        s!"-DTORCHLEAN_OUTPUT_DIR={buildDir}", "-DCMAKE_BUILD_TYPE=Release",
        s!"-DCMAKE_CXX_COMPILER={compiler}"]
      if let some cudaHome := get_config? cuda_home then
        args := args.push s!"-DCUDAToolkit_ROOT={cudaHome}"
      if let some extra := ← IO.getEnv "TORCHLEAN_LIBTORCH_CMAKE_ARGS" then
        let parsed := Json.parse extra >>= Json.getArr? >>= fun xs => xs.mapM Json.getStr?
        match parsed with
        | .ok flags =>
          unless flags.all (·.startsWith "-D") do
            error "TORCHLEAN_LIBTORCH_CMAKE_ARGS must contain only CMake -D options"
          args := args ++ flags
        | .error message => error message
      proc { cmd := cmake, args := args }
      proc { cmd := cmake, args := #["--build", buildDir.toString, "--parallel", "2"] }
      let output := buildDir / nameToSharedLib "torchlean_gpt_cached_decode"
      addTrace (.ofHash (← computeFileHash output) output.toString)
      return output
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

lean_exe check_torchlean_gpt_data where
  srcDir := "week-03-gpt-training"
  root := `TorchLeanGPT.DataCheck

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
  "https://github.com/lean-dojo/TorchLean.git" @ "main" with torchLeanOptions

require velvet from git "https://github.com/verse-lab/velvet.git" @ "main"

require LeanProfiler from git
  "https://github.com/lean-dojo/LeanProfiler.git" @
    "271b0b4cfa7de8c29b92ae34cf7ca8c79ac989a2"
