import Lake
open Lake DSL

/-!
Shared Lake package for the TorchLean verified examples repository.

The examples live in week folders, but they intentionally share one Lake
project and one TorchLean dependency. Each week we can add modules under this same
package instead of creating a separate Lake project.
-/

package TorchLeanVerifiedExamples where
  version := v!"0.1.0"
  description := "Weekly TorchLean examples with checked Lean developments."

lean_lib BatchInvariantInference where
  srcDir := "week-01-batch-invariant-inference"
  roots := #[
    `BatchInvariantInference.Core,
    `BatchInvariantInference.CUDA,
    `BatchInvariantInference.Generated
  ]
  defaultFacets := #[LeanLib.staticFacet]

require TorchLean from git
  "https://github.com/lean-dojo/TorchLean.git" @ "main"
