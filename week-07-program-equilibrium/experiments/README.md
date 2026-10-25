# Inkling: proposing policies with checked feedback

The [article](../) explains the experiment
and results. This directory contains the evaluated programs, their exact populations and
certificates, and an offline replay. It does not contain API credentials, private sampler
checkpoints, or raw prompts, sampled tokens and log probabilities. The export supports
rechecking outcomes, not reproducing the optimizer trajectory from scratch.

## Replay

From the repository root, summarize the exported results with Python's standard library:

```bash
python3 week-07-program-equilibrium/experiments/replay.py
```

Build the Lean modules, then recheck the original success and two transfer failures:

```bash
lake build ProgramEquilibrium
lake build check_program_policy
python3 week-07-program-equilibrium/experiments/replay.py --lean example
```

To recompute every exported outcome certificate:

```bash
python3 week-07-program-equilibrium/experiments/replay.py --lean all
```

This uses one Lean process, without model calls. It compares every returned verdict and
table with the export and also checks rejection of a deliberately corrupted table.
Without `--lean`, replay verifies data bindings and recomputes scores from the recorded
tables, but does not independently establish that those tables are the modal outcomes.

`inkling.json` contains observations and pools of schemas and populations. Two certificate
files contain 2,115 distinct requests and expected verdicts, deduplicated across studies
and controls. Reconstructed requests must match their original canonical JSON hashes.
These hashes detect inconsistent edits; they are not signatures proving model provenance.

## The two training studies

Both studies used trainable `thinkingmachines/Inkling` with rank-16 LoRA. The unrestricted
arm proposed complete formulas; the shielded arm proposed a guard that the host conjoined
with `□ me`. Each proposal was conditioned on one population, with its opponent schemas
visible. Formulas could have at most 48 nodes and depth eight. Populations contained
three to ten agents. The fixed JSON checker limits certificates to world 64; exceeding a
resource limit is inconclusive, not evidence of safety.

Each arm had two runs with 16 update opportunities each and eight on-policy samples per
opportunity. Adam used learning rate 1e−5, betas (0.9, 0.95), epsilon 1e−8, zero weight decay
and gradient clipping at one. Training used the importance-sampling loss with the native
sampled tokens and log probabilities. Invalid or exploitable proposals received reward −1;
safe but unhelpful proposals received zero; admitted proposals received their fraction of
mutually cooperating external opponents. Equal-reward groups produced no update.

There was one request per proposal with a distinct recorded seed, temperature and top-p
one, thinking effort 0.7, and a 4,096-token cap including reasoning. No full token chain
repeated within a sampling group; this does not establish statistical independence.

The first completed study used seeds 81 and 97, with 62 updates and two zero-signal skips.
It continued an interrupted run after two protocol corrections: singleton sampling replaced
multi-sample requests that returned repeated chains, and the stopping-witness limit rose
from eight to 64. Completed trajectories and existing responses were retained and rechecked;
the unfinished trainer was recreated. Bit-identical initial backend state was not verified.
This amended comparison is exploratory.

The fresh replication used seeds 113 and 131 and the corrected protocol throughout, with
64 updates. Its plan was fixed before inspecting the first study's completed comparison.
Each study had 512 training proposals and 512 evaluation proposals. Both reused the same
two sixteen-case training halves and the same 32 test populations: 16 same-distribution
and 16 deeper cases. Four seeds do not mean four distinct curricula.

Each arm's initial and final evaluation had 128 attempts, equally weighting cases and runs.
Rejected and invalid attempts count as zero cooperation. Reported 95% intervals resample
whole cases and are conditional on these runs and this structured case set. Replay checks
the point estimates; the exported intervals are recorded analyses, not recomputed by it.

## Transfer and delayed reciprocity

The transfer analysis takes every admitted final attempt, unchanged, into all eight reserved
validation populations. Repeated programs share computation, while retaining their original
attempt weights. The first study used 416 distinct program/population checks and the
replication 440; these counts are not globally deduplicated. Every new population needs its
own certificate. The first study has two exploitable transfers of one unrestricted policy;
both are rejected. There are no unresolved checks, and that failure does not recur in the
replication. This is a secondary, exploratory analysis of the admitted cohorts.

The final search expands the guard to `□ (¬ □^k ⊥ → me) ∧ guard`, with cutoffs zero through
four. Controls include FairBot, PrudentBot, four delayed-reciprocity policies and four
WaitFairBot policies with the paper's extra outer consistency condition. Base Inkling and
the saved unrestricted seed-113 adapter each proposed 24 policies over six rounds, with
four proposals per source per round and the same pooled development feedback. No weights
were updated. Forty-one proposals passed syntax validation.

There were 16 new development populations and 48 held-out populations, all drawn from the
same opponent catalog. The primary development-only selection retained a handwritten
control on ties and was frozen before held-out evaluation. The secondary model-only rule
was also declared before held-out outcomes; its selected trees were reconstructed from
development records after the primary evaluation. This is not an independent replication.
The primary run used 576 distinct program/population checks, and the actual model-only
trees added 96. All passed certificate checks with no exploitation, though admission could
still fail for lack of self-cooperation or an external partner.

An earlier feedback-only search also failed to beat its handwritten fallback. It is not
pooled into these results. An interrupted delayed-search diagnostic at effort 0.9 exhausted
an 8,192-token cap on all 16 returned responses before producing final text, with eight
more requests already dispatched when it stopped. It had no held-out evaluation. The
completed search used effort 0.7 and a 16,384-token cap on the unchanged case grid; all
48 responses ended normally. The truncated diagnostic is excluded from the score table,
not treated as unsafe-policy evidence.

## What is proved

`Certificate.lean` proves that complete nested-box agreement certifies permanent outcomes
and that admission implies non-exploitation in the supplied population. `Guards.lean` proves
eventual non-exploitation for the strict and delayed classes against arbitrary opponents
in any compatible modal universe. `Inkling.lean` proves the two best model-only trees
equivalent to the selected control, and proves the control's non-exploitation and
self-cooperation.

These general results are kernel-checked. `--lean` executes the proved Boolean checker
using Lean's compiler and runtime; it does not create a separate kernel proof of each
concrete verdict. The JSON flag `universal_guard` recognizes only the exact strict prefix,
not all semantically safe policies or the delayed family. Modal worlds are not game rounds
or arithmetic proof budgets. None of these results verifies the neural model itself.
