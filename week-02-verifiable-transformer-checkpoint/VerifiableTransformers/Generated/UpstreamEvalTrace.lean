/-
Generated finite-domain evaluation certificate for:

  checkpoint: artifacts/upstream-small-gpt/checkpoint-final
  weights:    artifacts/upstream-small-gpt/smt_weights.json

Scores are final-position candidate logits from the trained checkpoint, scaled
by 1000000 and rounded to integers.  Lean checks the task-domain shape and a
strictly positive projected candidate margin for every row.
-/

import VerifiableTransformers.Certificate.FiniteEval
import VerifiableTransformers.Certificate.Circuit
import VerifiableTransformers.Generated.TorchLeanEvalTrace

set_option maxRecDepth 4096

namespace VerifiableTransformers.Generated.UpstreamEvalTrace

open VerifiableTransformers.Certificate.FiniteEval
open VerifiableTransformers.Certificate.Circuit
open VerifiableTransformers.Certificate.Properties

def evalRows : List CandidateEval :=
[
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 2983574, alternateScoreMicros := 2527483 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 2394625, alternateScoreMicros := 2185041 },
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 2983011, alternateScoreMicros := 2527657 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 2412668, alternateScoreMicros := 2206408 },
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 2983613, alternateScoreMicros := 2527455 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 2397793, alternateScoreMicros := 2185801 },
  { task := .quoteClose, input := [1, 2, 5, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 2982810, alternateScoreMicros := 2527476 },
  { task := .quoteClose, input := [1, 2, 5, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 2372600, alternateScoreMicros := 2154401 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 2396029, alternateScoreMicros := 2186940 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 2414247, alternateScoreMicros := 2208945 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 2399972, alternateScoreMicros := 2188730 },
  { task := .quoteClose, input := [1, 2, 5, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 5, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 2376072, alternateScoreMicros := 2158593 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 2983680, alternateScoreMicros := 2527409 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 2410166, alternateScoreMicros := 2207862 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 2983158, alternateScoreMicros := 2527588 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 2421738, alternateScoreMicros := 2221770 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 2983720, alternateScoreMicros := 2527381 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 2412166, alternateScoreMicros := 2207165 },
  { task := .quoteClose, input := [1, 2, 5, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 2982934, alternateScoreMicros := 2527460 },
  { task := .quoteClose, input := [1, 2, 5, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 2388660, alternateScoreMicros := 2175018 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 2438217, alternateScoreMicros := 2254628 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 2438982, alternateScoreMicros := 2251138 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 2435740, alternateScoreMicros := 2247437 },
  { task := .quoteClose, input := [1, 2, 5, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 2983650, alternateScoreMicros := 2527368 },
  { task := .quoteClose, input := [1, 2, 5, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 2419308, alternateScoreMicros := 2220745 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 2983595, alternateScoreMicros := 2527468 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 2392081, alternateScoreMicros := 2181436 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 2983046, alternateScoreMicros := 2527645 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 2412668, alternateScoreMicros := 2206408 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 2983635, alternateScoreMicros := 2527441 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 2400710, alternateScoreMicros := 2189889 },
  { task := .quoteClose, input := [1, 2, 6, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 2982842, alternateScoreMicros := 2527472 },
  { task := .quoteClose, input := [1, 2, 6, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 2372600, alternateScoreMicros := 2154401 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 2393479, alternateScoreMicros := 2183326 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 2414247, alternateScoreMicros := 2208945 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 2402864, alternateScoreMicros := 2192784 },
  { task := .quoteClose, input := [1, 2, 6, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 6, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 2376072, alternateScoreMicros := 2158593 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 2983707, alternateScoreMicros := 2527390 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 2407883, alternateScoreMicros := 2204395 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 2983204, alternateScoreMicros := 2527571 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 2421738, alternateScoreMicros := 2221770 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 2983746, alternateScoreMicros := 2527362 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 2413599, alternateScoreMicros := 2209625 },
  { task := .quoteClose, input := [1, 2, 6, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 2982976, alternateScoreMicros := 2527455 },
  { task := .quoteClose, input := [1, 2, 6, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 2388660, alternateScoreMicros := 2175018 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 2436152, alternateScoreMicros := 2251102 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 2438982, alternateScoreMicros := 2251138 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 2436198, alternateScoreMicros := 2248232 },
  { task := .quoteClose, input := [1, 2, 6, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 2983727, alternateScoreMicros := 2527358 },
  { task := .quoteClose, input := [1, 2, 6, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 2420911, alternateScoreMicros := 2223502 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 2983587, alternateScoreMicros := 2527474 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 2393340, alternateScoreMicros := 2183220 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 2983033, alternateScoreMicros := 2527649 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 2412668, alternateScoreMicros := 2206408 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 2983626, alternateScoreMicros := 2527446 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 2399334, alternateScoreMicros := 2187961 },
  { task := .quoteClose, input := [1, 2, 7, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 2982830, alternateScoreMicros := 2527474 },
  { task := .quoteClose, input := [1, 2, 7, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 2372600, alternateScoreMicros := 2154401 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 2394741, alternateScoreMicros := 2185115 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 2414247, alternateScoreMicros := 2208945 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 2401501, alternateScoreMicros := 2190872 },
  { task := .quoteClose, input := [1, 2, 7, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 7, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 2376072, alternateScoreMicros := 2158593 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 2983699, alternateScoreMicros := 2527395 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 2409011, alternateScoreMicros := 2206108 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 2983190, alternateScoreMicros := 2527576 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 2421738, alternateScoreMicros := 2221770 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 2983739, alternateScoreMicros := 2527367 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 2412931, alternateScoreMicros := 2208478 },
  { task := .quoteClose, input := [1, 2, 7, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 2982963, alternateScoreMicros := 2527456 },
  { task := .quoteClose, input := [1, 2, 7, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 2388660, alternateScoreMicros := 2175018 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 2437170, alternateScoreMicros := 2252840 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 2438982, alternateScoreMicros := 2251138 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 2436010, alternateScoreMicros := 2247904 },
  { task := .quoteClose, input := [1, 2, 7, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 2983716, alternateScoreMicros := 2527359 },
  { task := .quoteClose, input := [1, 2, 7, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 2420175, alternateScoreMicros := 2222235 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 5], target := 9, alternate := 10, targetScoreMicros := 2983587, alternateScoreMicros := 2527474 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 5], target := 10, alternate := 9, targetScoreMicros := 2392042, alternateScoreMicros := 2181381 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 6], target := 9, alternate := 10, targetScoreMicros := 2983034, alternateScoreMicros := 2527649 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 6], target := 10, alternate := 9, targetScoreMicros := 2412668, alternateScoreMicros := 2206408 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 7], target := 9, alternate := 10, targetScoreMicros := 2983627, alternateScoreMicros := 2527446 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 7], target := 10, alternate := 9, targetScoreMicros := 2400583, alternateScoreMicros := 2189711 },
  { task := .quoteClose, input := [1, 2, 8, 9, 5, 8], target := 9, alternate := 10, targetScoreMicros := 2982831, alternateScoreMicros := 2527473 },
  { task := .quoteClose, input := [1, 2, 8, 10, 5, 8], target := 10, alternate := 9, targetScoreMicros := 2372600, alternateScoreMicros := 2154401 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 5], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 5], target := 10, alternate := 9, targetScoreMicros := 2393441, alternateScoreMicros := 2183273 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 6], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 6], target := 10, alternate := 9, targetScoreMicros := 2414247, alternateScoreMicros := 2208945 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 7], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 7], target := 10, alternate := 9, targetScoreMicros := 2402738, alternateScoreMicros := 2192607 },
  { task := .quoteClose, input := [1, 2, 8, 9, 6, 8], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 8, 10, 6, 8], target := 10, alternate := 9, targetScoreMicros := 2376072, alternateScoreMicros := 2158593 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 5], target := 9, alternate := 10, targetScoreMicros := 2983699, alternateScoreMicros := 2527395 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 5], target := 10, alternate := 9, targetScoreMicros := 2407857, alternateScoreMicros := 2204355 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 6], target := 9, alternate := 10, targetScoreMicros := 2983191, alternateScoreMicros := 2527576 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 6], target := 10, alternate := 9, targetScoreMicros := 2421738, alternateScoreMicros := 2221770 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 7], target := 9, alternate := 10, targetScoreMicros := 2983739, alternateScoreMicros := 2527367 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 7], target := 10, alternate := 9, targetScoreMicros := 2413538, alternateScoreMicros := 2209521 },
  { task := .quoteClose, input := [1, 2, 8, 9, 7, 8], target := 9, alternate := 10, targetScoreMicros := 2982964, alternateScoreMicros := 2527456 },
  { task := .quoteClose, input := [1, 2, 8, 10, 7, 8], target := 10, alternate := 9, targetScoreMicros := 2388660, alternateScoreMicros := 2175018 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 5], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 5], target := 10, alternate := 9, targetScoreMicros := 2436150, alternateScoreMicros := 2251097 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 6], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 6], target := 10, alternate := 9, targetScoreMicros := 2438982, alternateScoreMicros := 2251138 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 7], target := 9, alternate := 10, targetScoreMicros := 2983758, alternateScoreMicros := 2527354 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 7], target := 10, alternate := 9, targetScoreMicros := 2436184, alternateScoreMicros := 2248207 },
  { task := .quoteClose, input := [1, 2, 8, 9, 8, 8], target := 9, alternate := 10, targetScoreMicros := 2983716, alternateScoreMicros := 2527359 },
  { task := .quoteClose, input := [1, 2, 8, 10, 8, 8], target := 10, alternate := 9, targetScoreMicros := 2420845, alternateScoreMicros := 2223388 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 2817411, alternateScoreMicros := 2794872 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 2817406, alternateScoreMicros := 2795176 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 2816123, alternateScoreMicros := 2793899 },
  { task := .bracketType, input := [1, 3, 5, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 2814053, alternateScoreMicros := 2790938 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 2820657, alternateScoreMicros := 2800063 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 2820614, alternateScoreMicros := 2800307 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 2819358, alternateScoreMicros := 2799071 },
  { task := .bracketType, input := [1, 3, 5, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 2817230, alternateScoreMicros := 2796020 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 2817635, alternateScoreMicros := 2795232 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 2817627, alternateScoreMicros := 2795532 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 2816346, alternateScoreMicros := 2794258 },
  { task := .bracketType, input := [1, 3, 5, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 2814273, alternateScoreMicros := 2791291 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 2821646, alternateScoreMicros := 2801645 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 2821602, alternateScoreMicros := 2801888 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 2820343, alternateScoreMicros := 2800648 },
  { task := .bracketType, input := [1, 3, 5, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 5, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 2818195, alternateScoreMicros := 2797562 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 2817411, alternateScoreMicros := 2794872 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 2817406, alternateScoreMicros := 2795176 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 2816123, alternateScoreMicros := 2793899 },
  { task := .bracketType, input := [1, 3, 6, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 2814053, alternateScoreMicros := 2790938 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 2820657, alternateScoreMicros := 2800063 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 2820614, alternateScoreMicros := 2800307 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 2819358, alternateScoreMicros := 2799071 },
  { task := .bracketType, input := [1, 3, 6, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 2817230, alternateScoreMicros := 2796020 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 2817635, alternateScoreMicros := 2795232 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 2817627, alternateScoreMicros := 2795532 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 2816346, alternateScoreMicros := 2794258 },
  { task := .bracketType, input := [1, 3, 6, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 2814273, alternateScoreMicros := 2791291 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 2821646, alternateScoreMicros := 2801645 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 2821602, alternateScoreMicros := 2801888 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 2820343, alternateScoreMicros := 2800648 },
  { task := .bracketType, input := [1, 3, 6, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 6, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 2818195, alternateScoreMicros := 2797562 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 2817411, alternateScoreMicros := 2794872 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 2817406, alternateScoreMicros := 2795176 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 2816123, alternateScoreMicros := 2793899 },
  { task := .bracketType, input := [1, 3, 7, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 2814053, alternateScoreMicros := 2790938 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 2820657, alternateScoreMicros := 2800063 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 2820614, alternateScoreMicros := 2800307 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 2819358, alternateScoreMicros := 2799071 },
  { task := .bracketType, input := [1, 3, 7, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 2817230, alternateScoreMicros := 2796020 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 2817635, alternateScoreMicros := 2795232 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 2817627, alternateScoreMicros := 2795532 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 2816346, alternateScoreMicros := 2794258 },
  { task := .bracketType, input := [1, 3, 7, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 2814273, alternateScoreMicros := 2791291 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 2821646, alternateScoreMicros := 2801645 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 2821602, alternateScoreMicros := 2801888 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 2820343, alternateScoreMicros := 2800648 },
  { task := .bracketType, input := [1, 3, 7, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 7, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 2818195, alternateScoreMicros := 2797562 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 5], target := 14, alternate := 13, targetScoreMicros := 2817411, alternateScoreMicros := 2794872 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 6], target := 14, alternate := 13, targetScoreMicros := 2817406, alternateScoreMicros := 2795176 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 7], target := 14, alternate := 13, targetScoreMicros := 2816123, alternateScoreMicros := 2793899 },
  { task := .bracketType, input := [1, 3, 8, 11, 5, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 5, 8], target := 14, alternate := 13, targetScoreMicros := 2814053, alternateScoreMicros := 2790938 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 5], target := 14, alternate := 13, targetScoreMicros := 2820657, alternateScoreMicros := 2800063 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 6], target := 14, alternate := 13, targetScoreMicros := 2820614, alternateScoreMicros := 2800307 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 7], target := 14, alternate := 13, targetScoreMicros := 2819358, alternateScoreMicros := 2799071 },
  { task := .bracketType, input := [1, 3, 8, 11, 6, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 6, 8], target := 14, alternate := 13, targetScoreMicros := 2817230, alternateScoreMicros := 2796020 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 5], target := 14, alternate := 13, targetScoreMicros := 2817635, alternateScoreMicros := 2795232 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 6], target := 14, alternate := 13, targetScoreMicros := 2817627, alternateScoreMicros := 2795532 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 7], target := 14, alternate := 13, targetScoreMicros := 2816346, alternateScoreMicros := 2794258 },
  { task := .bracketType, input := [1, 3, 8, 11, 7, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 7, 8], target := 14, alternate := 13, targetScoreMicros := 2814273, alternateScoreMicros := 2791291 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 5], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 5], target := 14, alternate := 13, targetScoreMicros := 2821646, alternateScoreMicros := 2801645 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 6], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 6], target := 14, alternate := 13, targetScoreMicros := 2821602, alternateScoreMicros := 2801888 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 7], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 7], target := 14, alternate := 13, targetScoreMicros := 2820343, alternateScoreMicros := 2800648 },
  { task := .bracketType, input := [1, 3, 8, 11, 8, 8], target := 13, alternate := 14, targetScoreMicros := 2976590, alternateScoreMicros := 2890881 },
  { task := .bracketType, input := [1, 3, 8, 12, 8, 8], target := 14, alternate := 13, targetScoreMicros := 2818195, alternateScoreMicros := 2797562 }
]

def evalCertificate : EvalCertificate where
  sourceWeightsSha256 := "a647e7e07512596b70b2b41740ab53e1c1fa94e1fce5d13eceb8d86caec61e88"
  checkpointPath := "artifacts/upstream-small-gpt/checkpoint-final"
  scoreScale := 1000000
  rows := evalRows

/- Neel's trace passes the finite-domain checker. -/
theorem evalTrace_ok :
    checkEvalCertificate evalCertificate = true := by
  rfl


def quoteCircuit : CircuitSummary where
  task := .quoteClose
  nLayers := 2
  edges := quoteEdges
  ablation := "zero"
  metric := "candidate_kl"
  minAgreementMicros := 1000000
  thresholdMicros := 50000

def bracketCircuit : CircuitSummary where
  task := .bracketType
  nLayers := 2
  edges := bracketEdges
  ablation := "zero"
  metric := "candidate_kl"
  minAgreementMicros := 1000000
  thresholdMicros := 5000

def quoteVerification : VerificationSummary where
  task := .quoteClose
  numInputs := 128
  candidateTokens := [9, 10]
  circuit := quoteCircuit
  pytorchValidationPassed := true
  pytorchExamplesChecked := 128
  pytorchFailures := 0
  functionalStatusVerified := true
  functionalVerifiedCount := 128
  functionalTotalSequences := 128
  functionalTimeouts := 0
  functionalErrors := 0
  functionalCounterexamples := 0
  edgeNecessityStatusVerified := true
  edgeTotal := 3
  edgeNecessary := 3
  edgeUnnecessary := 0
  edgeUnresolved := 0
  edgeTimeouts := 0
  edgeErrors := 0
  robustnessStatusVerified := true
  robustnessVerifiedCount := 128
  robustnessTimeouts := 0
  robustnessErrors := 0
  robustnessEpsilonMicros := 10000
  robustnessViolations := 0
  robustnessDecisionViolations := 0
  robustnessBranchUnstable := 0

def bracketVerification : VerificationSummary where
  task := .bracketType
  numInputs := 128
  candidateTokens := [13, 14]
  circuit := bracketCircuit
  pytorchValidationPassed := true
  pytorchExamplesChecked := 128
  pytorchFailures := 0
  functionalStatusVerified := true
  functionalVerifiedCount := 128
  functionalTotalSequences := 128
  functionalTimeouts := 0
  functionalErrors := 0
  functionalCounterexamples := 0
  edgeNecessityStatusVerified := true
  edgeTotal := 6
  edgeNecessary := 6
  edgeUnnecessary := 0
  edgeUnresolved := 0
  edgeTimeouts := 0
  edgeErrors := 0
  robustnessStatusVerified := true
  robustnessVerifiedCount := 128
  robustnessTimeouts := 0
  robustnessErrors := 0
  robustnessEpsilonMicros := 10000
  robustnessViolations := 0
  robustnessDecisionViolations := 0
  robustnessBranchUnstable := 0

/- Both saved circuit summaries from Neel's run pass the Lean-side checker. -/
theorem neelCircuitSummaries_ok :
    checkVerificationSummary quoteVerification = true ∧
    checkVerificationSummary bracketVerification = true := by
  exact ⟨rfl, rfl⟩

/- Both finite traces satisfy the projected properties used in the checkpoint claim. -/
theorem generatedTraceProperties_ok :
    TracePropertiesAccepted
      evalCertificate.rows ∧
    TracePropertiesAccepted
      VerifiableTransformers.Generated.TorchLeanEvalTrace.evalCertificate.rows := by
  exact ⟨by rfl, by rfl⟩

end VerifiableTransformers.Generated.UpstreamEvalTrace
