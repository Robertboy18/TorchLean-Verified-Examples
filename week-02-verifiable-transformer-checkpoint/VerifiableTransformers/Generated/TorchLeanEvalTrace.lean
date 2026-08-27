/-
Generated TorchLean CUDA checkpoint summary and eval trace.

Lean checks two facts here: the exported parameter summary has the expected
shape/count metadata, and the finite eval rows satisfy the projected
quote/bracket property.

Sources: artifacts/torchlean-cuda-small-gpt.parambits.json and artifacts/torchlean-cuda-small-gpt.eval.json.
-/

import VerifiableTransformers.Certificate.FiniteEval

set_option maxRecDepth 4096

namespace VerifiableTransformers.Generated.TorchLeanEvalTrace

open VerifiableTransformers.Certificate.FiniteEval

def checkpointPath : String :=
  "artifacts/torchlean-cuda-small-gpt.parambits.json"

def checkpointSha256 : String :=
  "94a86fd6a0e875e1ec54d9c7bcd55ade995950ef7a5f4e6809a4c59763649442"

def formatTag : String :=
  "torchlean_paramlist_bits_v1"

def modelTag : String :=
  "torchlean-causal-softmax-gpt-v1"

/-- Parameter order recorded by the archived native TorchLean run. -/
def paramTensorShapes : List (List Nat) :=
  [[32, 16], [6, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16], [16], [64, 16], [64], [16, 64], [16], [16], [16], [16, 16], [16, 16], [16, 16], [16, 16], [16], [16], [64, 16], [64], [16, 64], [16], [16], [16], [16], [16], [32, 16], [32]]

def paramValueCounts : List Nat :=
  [512, 96, 256, 256, 256, 256, 16, 16, 1024, 64, 1024, 16, 16, 16, 256, 256, 256, 256, 16, 16, 1024, 64, 1024, 16, 16, 16, 16, 16, 512, 32]

def first16WeightBits : List Nat :=
  [4580822060778913792, 13786571689515024384, 4580322692984471552, 4575273028530733056, 4576124839730872320, 4577722385565745152, 13804188507733753856, 13797190456062246912, 13799140694947790848, 4581322400309706752, 4579280821428420608, 4568802726334431232, 13795012310105849856, 4580434968798298112, 13802389934880849920, 4567282599432552448]

def totalValueCount : Nat :=
  paramValueCounts.foldl (fun acc n => acc + n) 0

def checkpointSummaryOk : Bool :=
  formatTag == "torchlean_paramlist_bits_v1" &&
  modelTag == "torchlean-causal-softmax-gpt-v1" &&
  paramTensorShapes == [[32, 16], [6, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16], [16], [64, 16], [64], [16, 64], [16], [16], [16], [16, 16], [16, 16], [16, 16], [16, 16], [16], [16], [64, 16], [64], [16, 64], [16], [16], [16], [16], [16], [32, 16], [32]] &&
  paramValueCounts.length == 30 &&
  totalValueCount == 7616 &&
  first16WeightBits.length == 16

theorem checkpointSummary_ok :
    checkpointSummaryOk = true := by
  rfl

def evalRows : List CandidateEval :=
[
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 2974195, alternateScoreMicros := 250352 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 3670591, alternateScoreMicros := 573505 },
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 2972749, alternateScoreMicros := 266189 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 3616520, alternateScoreMicros := 474240 },
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 3009252, alternateScoreMicros := 178288 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 3619198, alternateScoreMicros := 577301 },
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 2916232, alternateScoreMicros := 205574 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 3682099, alternateScoreMicros := 521334 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 2967488, alternateScoreMicros := 286236 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 3673242, alternateScoreMicros := 542971 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 2977937, alternateScoreMicros := 271971 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 3620272, alternateScoreMicros := 446609 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 3016662, alternateScoreMicros := 191692 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 3624776, alternateScoreMicros := 549777 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 2915245, alternateScoreMicros := 237006 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 3687531, alternateScoreMicros := 493852 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 2971449, alternateScoreMicros := 320426 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 3675000, alternateScoreMicros := 539545 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 2986076, alternateScoreMicros := 340835 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 3624105, alternateScoreMicros := 442046 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 3020505, alternateScoreMicros := 244745 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 3628376, alternateScoreMicros := 539799 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 2903743, alternateScoreMicros := 301347 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 3687512, alternateScoreMicros := 483659 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 2950993, alternateScoreMicros := 294277 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 3686419, alternateScoreMicros := 538891 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 2958963, alternateScoreMicros := 286500 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 3637468, alternateScoreMicros := 437086 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 3000501, alternateScoreMicros := 199335 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 3638106, alternateScoreMicros := 545334 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 2895344, alternateScoreMicros := 246833 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 3698486, alternateScoreMicros := 489297 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 2966467, alternateScoreMicros := 305464 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 3673395, alternateScoreMicros := 536139 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 2974474, alternateScoreMicros := 281785 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 3621323, alternateScoreMicros := 440659 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 3013499, alternateScoreMicros := 212338 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 3625826, alternateScoreMicros := 544926 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 2914726, alternateScoreMicros := 244381 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 3687893, alternateScoreMicros := 490945 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 2956819, alternateScoreMicros := 343289 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 3674531, alternateScoreMicros := 505526 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 2974118, alternateScoreMicros := 295859 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 3623462, alternateScoreMicros := 412902 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 3016146, alternateScoreMicros := 230979 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 3630103, alternateScoreMicros := 517330 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 2910068, alternateScoreMicros := 278343 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 3691974, alternateScoreMicros := 463525 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 2961020, alternateScoreMicros := 374140 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 3676481, alternateScoreMicros := 504233 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 2983116, alternateScoreMicros := 357209 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 3627365, alternateScoreMicros := 410512 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 3020064, alternateScoreMicros := 279043 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 3633420, alternateScoreMicros := 509321 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 2898942, alternateScoreMicros := 337403 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 3691720, alternateScoreMicros := 454945 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 2941361, alternateScoreMicros := 352334 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 3687878, alternateScoreMicros := 500829 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 2956767, alternateScoreMicros := 309309 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 3640393, alternateScoreMicros := 402827 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 3001900, alternateScoreMicros := 238786 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 3643325, alternateScoreMicros := 512544 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 2891774, alternateScoreMicros := 288601 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 3703156, alternateScoreMicros := 458792 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 2976946, alternateScoreMicros := 170921 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 3676061, alternateScoreMicros := 532883 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 2991737, alternateScoreMicros := 195312 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 3626811, alternateScoreMicros := 437074 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 3013104, alternateScoreMicros := 119347 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 3629981, alternateScoreMicros := 535139 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 2895262, alternateScoreMicros := 157255 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 3688715, alternateScoreMicros := 480542 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 2968604, alternateScoreMicros := 204328 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 3677348, alternateScoreMicros := 503644 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 2992507, alternateScoreMicros := 198273 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 3629034, alternateScoreMicros := 411089 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 3016423, alternateScoreMicros := 130330 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 3634007, alternateScoreMicros := 509168 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 2891518, alternateScoreMicros := 184807 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 3692633, alternateScoreMicros := 454693 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 2966896, alternateScoreMicros := 240254 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 3677942, alternateScoreMicros := 500375 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 2991671, alternateScoreMicros := 276976 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 3631520, alternateScoreMicros := 406310 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 3011912, alternateScoreMicros := 189310 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 3636272, alternateScoreMicros := 499669 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 2876736, alternateScoreMicros := 247843 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 3691588, alternateScoreMicros := 445398 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 2953540, alternateScoreMicros := 214769 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 3690614, alternateScoreMicros := 500078 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 2976113, alternateScoreMicros := 214962 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 3645552, alternateScoreMicros := 402178 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 3002640, alternateScoreMicros := 139274 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 3647035, alternateScoreMicros := 504806 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 2873555, alternateScoreMicros := 197537 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 3703657, alternateScoreMicros := 450354 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 2959293, alternateScoreMicros := 239395 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 3689066, alternateScoreMicros := 539273 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 2964001, alternateScoreMicros := 225106 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 3642560, alternateScoreMicros := 438965 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 3003786, alternateScoreMicros := 157826 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 3641561, alternateScoreMicros := 546999 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 2899438, alternateScoreMicros := 207651 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 3700771, alternateScoreMicros := 489898 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 2950608, alternateScoreMicros := 278144 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 3690902, alternateScoreMicros := 508959 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 2964818, alternateScoreMicros := 239015 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 3645028, alternateScoreMicros := 411215 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 3007707, alternateScoreMicros := 176105 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 3646116, alternateScoreMicros := 519486 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 2896238, alternateScoreMicros := 241535 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 3705340, alternateScoreMicros := 462714 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 2953879, alternateScoreMicros := 312252 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 3692616, alternateScoreMicros := 507888 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 2972818, alternateScoreMicros := 308058 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 3648410, alternateScoreMicros := 409463 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 3010705, alternateScoreMicros := 231603 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 3649336, alternateScoreMicros := 511570 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 2883805, alternateScoreMicros := 307823 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 3704977, alternateScoreMicros := 454427 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 2931195, alternateScoreMicros := 288437 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 3703013, alternateScoreMicros := 503314 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 2941327, alternateScoreMicros := 256477 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 3661369, alternateScoreMicros := 399846 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 2988981, alternateScoreMicros := 187436 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 3659091, alternateScoreMicros := 513669 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 2874043, alternateScoreMicros := 254410 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 3715799, alternateScoreMicros := 457211 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 3623334, alternateScoreMicros := 666990 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 2816869, alternateScoreMicros := 762086 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 3625225, alternateScoreMicros := 959411 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 2875514, alternateScoreMicros := 888595 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 3698029, alternateScoreMicros := 627419 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 2800564, alternateScoreMicros := 908414 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 3668450, alternateScoreMicros := 731419 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 2803378, alternateScoreMicros := 821314 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 3629801, alternateScoreMicros := 637994 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 2791528, alternateScoreMicros := 830569 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 3642737, alternateScoreMicros := 918333 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 2860765, alternateScoreMicros := 955979 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 3709872, alternateScoreMicros := 588376 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 2782749, alternateScoreMicros := 975857 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 3684947, alternateScoreMicros := 705044 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 2782390, alternateScoreMicros := 883406 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 3616790, alternateScoreMicros := 665800 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 2799048, alternateScoreMicros := 850596 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 3627924, alternateScoreMicros := 935076 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 2862500, alternateScoreMicros := 970367 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 3702067, alternateScoreMicros := 629918 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 2788477, alternateScoreMicros := 988302 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 3661541, alternateScoreMicros := 722968 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 2785767, alternateScoreMicros := 897517 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 3641093, alternateScoreMicros := 615081 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 2797618, alternateScoreMicros := 790459 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 3656263, alternateScoreMicros := 892271 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 2866193, alternateScoreMicros := 939721 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 3721355, alternateScoreMicros := 580467 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 2790964, alternateScoreMicros := 949185 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 3689035, alternateScoreMicros := 678417 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 2789393, alternateScoreMicros := 851420 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 3629373, alternateScoreMicros := 611139 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 2795948, alternateScoreMicros := 820738 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 3638335, alternateScoreMicros := 927885 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 2870813, alternateScoreMicros := 927615 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 3709034, alternateScoreMicros := 580411 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 2788751, alternateScoreMicros := 964755 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 3682938, alternateScoreMicros := 676376 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 2786718, alternateScoreMicros := 873554 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 3631389, alternateScoreMicros := 585146 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 2767810, alternateScoreMicros := 889962 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 3650268, alternateScoreMicros := 885341 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 2852441, alternateScoreMicros := 997652 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 3716878, alternateScoreMicros := 542256 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 2768128, alternateScoreMicros := 1033230 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 3695242, alternateScoreMicros := 651845 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 2763488, alternateScoreMicros := 935878 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 3620215, alternateScoreMicros := 612392 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 2776745, alternateScoreMicros := 906999 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 3637633, alternateScoreMicros := 904020 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 2854971, alternateScoreMicros := 1008201 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 3710517, alternateScoreMicros := 584773 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 2774905, alternateScoreMicros := 1042108 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 3672722, alternateScoreMicros := 670714 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 2767875, alternateScoreMicros := 947101 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 3643816, alternateScoreMicros := 563366 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 2775342, alternateScoreMicros := 851372 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 3665775, alternateScoreMicros := 863155 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 2860587, alternateScoreMicros := 980302 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 3729733, alternateScoreMicros := 536391 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 2777987, alternateScoreMicros := 1007157 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 3700906, alternateScoreMicros := 627073 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 2771845, alternateScoreMicros := 904874 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 3623275, alternateScoreMicros := 631160 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 2782391, alternateScoreMicros := 935992 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 3634108, alternateScoreMicros := 912928 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 2838068, alternateScoreMicros := 1049260 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 3704411, alternateScoreMicros := 599365 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 2760866, alternateScoreMicros := 1078754 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 3669512, alternateScoreMicros := 693351 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 2764481, alternateScoreMicros := 976124 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 3626673, alternateScoreMicros := 603455 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 2754434, alternateScoreMicros := 1002145 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 3647194, alternateScoreMicros := 872899 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 2819278, alternateScoreMicros := 1115229 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 3713131, alternateScoreMicros := 562375 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 2740052, alternateScoreMicros := 1144128 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 3682041, alternateScoreMicros := 668426 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 2741274, alternateScoreMicros := 1035882 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 3613058, alternateScoreMicros := 630581 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 2761345, alternateScoreMicros := 1017583 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 3631791, alternateScoreMicros := 888444 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 2820683, alternateScoreMicros := 1127298 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 3703954, alternateScoreMicros := 601578 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 2745457, alternateScoreMicros := 1152726 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 3658617, alternateScoreMicros := 686256 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 2744594, alternateScoreMicros := 1047563 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 3637343, alternateScoreMicros := 576857 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 2762252, alternateScoreMicros := 969350 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 3661639, alternateScoreMicros := 844268 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 2826501, alternateScoreMicros := 1104275 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 3724576, alternateScoreMicros := 550099 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 2749254, alternateScoreMicros := 1123160 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 3687028, alternateScoreMicros := 638963 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 2749085, alternateScoreMicros := 1009638 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 3643238, alternateScoreMicros := 644703 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 2827320, alternateScoreMicros := 810152 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 3646617, alternateScoreMicros := 951826 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 2881679, alternateScoreMicros := 931759 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 3720818, alternateScoreMicros := 612901 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 2815598, alternateScoreMicros := 954457 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 3687359, alternateScoreMicros := 709317 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 2815048, alternateScoreMicros := 862241 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 3646331, alternateScoreMicros := 616523 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 2801291, alternateScoreMicros := 882556 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 3660981, alternateScoreMicros := 908347 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 2865621, alternateScoreMicros := 1004080 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 3730084, alternateScoreMicros := 572928 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 2797118, alternateScoreMicros := 1024984 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 3701190, alternateScoreMicros := 682737 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 2793741, alternateScoreMicros := 926760 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 3634705, alternateScoreMicros := 645013 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 2809082, alternateScoreMicros := 901103 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 3647737, alternateScoreMicros := 927705 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 2868023, alternateScoreMicros := 1015486 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 3722995, alternateScoreMicros := 616391 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 2802450, alternateScoreMicros := 1035786 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 3678692, alternateScoreMicros := 702729 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 2797080, alternateScoreMicros := 940022 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 3655083, alternateScoreMicros := 589952 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 2805807, alternateScoreMicros := 842386 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 3672827, alternateScoreMicros := 879385 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 2868766, alternateScoreMicros := 988411 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 3740073, alternateScoreMicros := 563978 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 2804376, alternateScoreMicros := 997338 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 3703807, alternateScoreMicros := 654730 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 2800103, alternateScoreMicros := 894106 }
]

def evalCertificate : EvalCertificate where
  sourceWeightsSha256 := "94a86fd6a0e875e1ec54d9c7bcd55ade995950ef7a5f4e6809a4c59763649442"
  checkpointPath := checkpointPath
  scoreScale := 1000000
  rows := evalRows

theorem evalTrace_ok :
    checkEvalCertificateWithSha
      checkpointSha256
      evalCertificate = true := by
  rfl

end VerifiableTransformers.Generated.TorchLeanEvalTrace
