# Training a GPT-2-style model in Lean

Okay, let's train a GPT in Lean :) I wanted to take TorchLean beyond small examples and get the
model, data loader, autograd, optimizer, and checkpoint code all working together, so for this
experiment I trained all 124,412,160 stored parameters of a GPT-2-style model on one A100.

I borrowed the recipe from [minGPT](https://github.com/karpathy/minGPT) and [nanoGPT](https://github.com/karpathy/nanoGPT):
a causal transformer, next-token prediction, AdamW, and a
warmup-and-cosine learning-rate schedule. From [Radford and colleagues' GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
I took the main dimensions and the tied token embeddings, but the model has its own initialization
and weights, it was trained on [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), and our current version also uses
bias-free query, key, and value projections. So this is a GPT-2-style architecture trained from
scratch (not OpenAI's released GPT-2 checkpoint).

The model learned! Validation loss fell during pre-training, and more training brought it down
further. I also tried turning the model into an assistant, with much less success: the
instruction-tuning losses went down, but it still got elementary questions wrong. We'll look at
both, then compare 50 matched updates against PyTorch, and then go through the proofs for data
alignment, resuming a run, and cached generation.

Karpathy describes [minGPT](https://github.com/karpathy/minGPT) as trying to be:

> small, clean, interpretable and educational

I want to keep that quality in my Lean version too. You should be able to follow a batch from
its token IDs all the way to the loss and the update, and then open the proof of some specific
property of that path. The big run tests whether those pieces work together at scale, and I put in the
small calculations below so you can see what each of them is actually doing.

<div id="the-gpt-model-and-training-update"></div>

## The model and one training step

Our large preset has 12 transformer blocks, hidden width 768, 12 attention heads, and a
50,257-token vocabulary, so each head works with 64 coordinates (768 divides evenly among the 12
heads). Each feed-forward part expands to four times the hidden width and then projects back.
None of these dimensions depend on how many sequences you put in one batch.

Start with one sequence of token IDs. Embedding lookup turns each ID into a width-768 vector and
adds a learned position vector to it. Each block normalizes those vectors, computes causal
attention, and adds the result to the residual stream. Then a second normalization and a
feed-forward network produce another residual update. After the last block and one final
normalization, the language-model head gives one score for every vocabulary item at every
position.

By residual stream I mean the hidden sequence carried from block to block. A block adds an
update to it rather than replacing it outright. Attention mixes information between token
positions that are allowed to see each other, while the feed-forward network transforms each
position's vector on its own. Before either one, normalization rescales the vector using its
mean and variance (plus a learned scale and shift). That pattern repeats all the way up the
stack, with different learned weights in every block.

Input embedding and output projection share one parameter table. Its rows are the input vectors,
and its transpose maps the final hidden vector to vocabulary scores, so training can change the
same parameters through either use. Keeping a single shared table in the model definition means
the optimizer and the checkpoint never treat the two copies as independent weights.

For a next-token target <span data-tex="y_t">yₜ</span>, the cross-entropy loss at position
<span data-tex="t">t</span> is the negative log probability assigned to that token:

<div class="math-display" data-tex="\ell_t=-\log\frac{\exp(z_{t,y_t})}{\sum_v\exp(z_{t,v})}">Next-token loss penalizes low probability for the observed target.</div>

Here <span data-tex="z_{t,v}">zₜ,ᵥ</span> is the raw score (the logit) for vocabulary item
<span data-tex="v">v</span>, and raising the target's score relative to its competitors lowers
the loss. A correct token ranked first can still incur loss if its probability is small. It goes
the other way too. The average loss can improve while particular questions stay wrong, and you'll
see exactly that with the instruction-tuned model.

Take the gradient of this real-valued loss with respect to one vocabulary score and you get
$p_{t,v}-\mathbf{1}_{v=y_t}$, where $p$ is the softmax probability (the indicator is one only for
the target). So if the target has probability 0.6, its score derivative is −0.4, and a competing
token with probability 0.1 has derivative 0.1. If these scores were parameters we could adjust
independently, gradient descent would raise the target score and lower that competing score.

In the model, they aren't independent parameters. They come out of the shared embeddings and the
transformer blocks, and backward follows those dependencies to find derivatives for the
parameters themselves, so changing one weight can move many positions and many scores at once.
That's why a sensible local loss derivative doesn't guarantee that every question improves after
an optimizer step. (And the formula is exact real arithmetic, while the training run computes
rounded versions of it.)

An update starts by choosing token windows and their targets. From there the model computes
logits, the objective reduces them to a scalar loss, and our automatic differentiation (autograd)
tape propagates
derivatives back to the parameters. Lean controls the training loop and the backward traversal.
On GPU, the operation adapters call LibTorch's numerical primitives and local gradient operations
(with LibTorch's own autograd engine disabled).

[AdamW](https://arxiv.org/abs/1711.05101) (from Loshchilov and Hutter) combines that gradient
with two running statistics for each parameter:
a first moment, which averages recent gradients, and a second moment, which averages their
squares. Both start at zero, and bias correction accounts for that. The learning rate sets the
size of the update, and decoupled weight decay shrinks the parameters separately (pre-training
used a decay of 0.1, while the instruction-tuning runs and the PyTorch comparison below used 0).
Those running statistics are why a training checkpoint needs more than the current weights.

During warmup the learning rate rises, and after that it decreases on a cosine schedule whose
input is the optimizer-step index, so if you restart that index after loading a checkpoint, you
replay the early schedule on parameters that are already somewhere else. We'll come back to the
proof about global indices after looking at the completed runs.

## Pre-training on FineWeb-Edu

My first run used a 1,024-token context and batch size 8. It processed 1,000,005,632 scheduled
tokens over 122,071 updates, which took about 46 hours and 35 minutes on one A100. Validation
loss fell from 10.963 at initialization to a best recorded value of 3.308 at step 107,000, and
then climbed back to 3.488 by the final step.

I continued from the checkpoint saved at step 100,000. Why not from step 107,000? The lowest
evaluation fell between saves, so step 100,000 was the best one on disk (validation loss 3.333).
That distinction matters when you count the training history. At step 100,000 the checkpoint had seen 819,200,000
tokens, and adding 1,500,008,448 continuation tokens gives a total of 2,319,208,448 for the
parameters along this history.

For the continuation I used a new FineWeb-Edu shard, batch size 12, and a lower learning rate.
Its best recorded validation loss was 3.135083 at continuation step 122,000. But the final
evaluation, 71 updates later, was 3.273021, so the loss went back up at the very end (at the
minimum learning rate and on the same validation windows). Once again the best measurement fell
between checkpoint boundaries, and the final parameter file holds the later state.

All of those measurements come from the archived strict-FP32 runs. Today's runtime uses LibTorch
and keeps Lean `Float`'s binary64 precision, optimizer and cache state included. So the old loss
curves and timings describe the experiments I completed, and a new run with the current adapter
has different precision (and may round differently and run at a different speed). You'll find
the configurations in the [development notes](development.html), but the metric files and
checkpoints of these runs aren't in the repository.

A token budget here counts scheduled training positions. With batch size 8 and context length
1,024, one update schedules 8,192 positions, and multiplying by 122,071 gives the first run's
1,000,005,632 tokens. Since the runner picks a whole number of updates, it can overshoot a
requested budget by less than one batch. It counts positions processed, so it says nothing about
whether every token came from a different document or appeared exactly once.

Checkpoint history is just as concrete. Updates after step 100,000 in the first run never reached
the continuation, because the continuation was initialized from that earlier checkpoint, so
if you added the complete first-run total, you'd be counting updates the continued parameters
never received. I use the saved step's token count instead, which is why the total along this
parameter history is about 2.32 billion (and not 2.50 billion).

Validation evaluates held-out windows without updating the model, but a recorded loss is still a
measurement on those chosen windows. (In the instruction-tuning runs below, validation uses a
fixed deterministic sample of eight batches, not every record in the shard.) Checkpoint saving
and evaluation also happen at different intervals. So a best observed loss can mark a useful
point on the curve and still leave no parameter file for that exact step.

<div id="instruction-tuning-and-the-answers-we-got"></div>

## Lower loss, unreliable answers

Next I trained the model on a `User:`/`Assistant:` transcript format, in two rounds. The first
round scheduled about ten million token rows from [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk),
and the second started from that round's best saved checkpoint and scheduled about fifty million
rows from a [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) mixture. Loss
was charged only to assistant targets (the prompt stayed available as context).

Within each round the sampled validation loss fell, from 2.673017 to 2.135422 in the first and
from 2.818384 to 2.216872 in the second. Don't compare the two final values, though. Since
the validation shards differ, you'd be mixing a model change with a dataset change.

But lower validation loss did not give us reliable answers to simple prompts.
The recorded outputs are in the [prompt suite](results/instruction-prompt-suite.json), all
greedy (top-k 1) with the small repeat penalty every Week 3 generator applies (1.05 per earlier
occurrence among the last 64 tokens, prompt included).
The checkpoint named Paris as the capital of France and correctly summarized a sentence about
freezing water, but then it got `7 + 5` wrong, gave an incorrect explanation of a blue sky, refused a
harmless rewrite, and produced malformed Python. It did recognize the positive sentiment of a film
review, but that didn't make it obey the requested one-word answer format.

Could a transcript mismatch explain the wrong answers? To rule that out, I checked an actual
validation record containing the question about Bartolomeo Cristofori. Its prefix matched the
generator's transcript and its target was `Piano`, and the model still answered incorrectly, so
at least for this case the answer is no. The objective had improved. My pretraining and
instruction-tuning recipe still didn't produce a dependable assistant at this model size.

In training, a supplied answer gives the model all of its preceding tokens as context while it
predicts the next one, but in free generation it has to continue from its own predictions, and one
poor choice can lead it to a prefix unlike anything in the validation examples. So the loss and
the prompt suite inspect different behavior, and that's before we even consider that a numeric
answer or a valid Python function may require several correct tokens in a row.

I don't know yet which changes would fix each failed answer, because this experiment doesn't
isolate their causes. What I do know is that the curves improved and the model still got `7 + 5`
wrong. So... I'd keep the actual answers beside the loss plot, rather than treat either one as
the whole evaluation.

<div id="next-token-targets-and-assistant-only-loss"></div>

## Getting the targets and loss mask right

Each training row predicts one token ahead, so a window's targets are just its input tokens
shifted by one:

<figure class="token-figure">
  <div><span class="row-label">Input</span><span>The</span><span>cat</span><span>sat</span><span>on</span></div>
  <div><span class="row-label">Target</span><span>cat</span><span>sat</span><span>on</span><span>the</span></div>
  <figcaption>Each column is one next-token prediction, and the target mask has to shift along with the
    target row. (I wrote words in place of the tokenizer IDs the model actually sees.)</figcaption>
</figure>

That shift gets easy to mess up once you restrict the loss to assistant responses. If row
<span data-tex="t">t</span> predicts corpus token <span data-tex="t+1">t + 1</span>, the mask for
that row has to say whether token <span data-tex="t+1">t + 1</span> belongs to the answer, and
using the mask attached to the input token instead would train a different set of predictions.
I proved the next-token and mask equalities (entry by entry) for the two batch builders the
trainer calls, in [Run.lean](TorchLeanGPT/Run.lean).

Written in exact arithmetic, the weighted objective has the form

<div class="math-display" data-tex="L=\sum_t w_t\ell_t,\qquad w_t\ge 0,\qquad \sum_t w_t=1">Row weights average the active next-token losses.</div>

For assistant-only training, prompt and padding targets get zero weight, and the active
assistant targets share the normalized weight. Does that mean the prompt has no effect on
learning? No. Attention reads its representations when predicting the answer, and gradients can flow
through those uses. So the mask picks which predictions we ask the model to get right, and it has
nothing to say about which preceding tokens the model can see.

Our loader gives every active target in the batch the same weight. If one response has two
active targets and another has six, each target receives 1/8, so the responses contribute one
quarter and three quarters of the loss respectively. That's a token average, which is different
from averaging equally over responses. In code, the loader counts the enabled rows, forms a
floating-point reciprocal (so there's rounding in there), and converts those weights to the
model's scalar type. Padding gets zero weight no matter how many padding rows it took to form
the tensor.

Suppose an input row ends at the final prompt token and its target is the first answer token:
that prediction should count toward the answer loss, but if we used the input token's prompt
mask we would discard it. At the other end, an answer input whose target lies beyond the answer
interval shouldn't pick up weight just because the input belonged to the answer. Shift the mask
with the target and both boundaries come out right.

An earlier instruction-tuning run had a concrete packing bug. It sampled arbitrary windows from a
concatenated dialogue tape, which let a window begin inside an answer or cross into another
conversation, so I replaced that sampler with bounded dialogue records and reran both stages (the
losses above come from the repaired runs). Each record describes a token window and an
assistant-target interval inside it, and long answers can use several records with overlapping
context. And the loader checks that target intervals cover active targets once and
stay within their own windows.

In [DialogueRecords.lean](TorchLeanGPT/DialogueRecords.lean), the Lean results connect the
executable target selector to that interval definition and exclude padded rows after a record
ends. In [Batching.lean](TorchLeanGPT/Batching.lean), I used Lean's `vcgen` tactic to prove
that the packing loop appends each input, target, and mask bit in row order. Both proofs cover
the pure index and packing functions, which leaves byte decoding and device execution to their
own checks.

A record does two jobs. Its window provides the context available to the model, and its target
interval says which predictions belong to this training example. A long answer can need several
windows because one context can't hold all of it, and then overlapping context lets a later
window see earlier answer tokens while disjoint target intervals keep us from charging loss twice
for the same answer token. The loader validates those intervals before it assembles the batch.

My packing proof follows the assembled contents, not only their lengths. Starting from
arbitrary initial arrays, appending rows preserves the input, successor target, and mask bit at
each appended position. A shape check alone would happily accept a batch with two target rows
swapped. With the content invariant, those swapped values would have to equal the intended row
functions, and that's the stronger property we need here.

Inside the model, causal attention handles the corresponding constraint: during training, an
input position must not read the future target. For the hard-mask specification, our Lean
theorem gives every strict-future attention coordinate zero forward weight and zero score
derivative, for arbitrary scores and incoming weight gradients. It's a local statement about the
attention equations (implementing them is the numerical backend's job).

In the attention figure, select “sat.” Its position may read “The,” “cat,” and itself,
but not “on.” A training batch can compute all four positions at once because this mask enforces
their different allowed contexts, so parallel computation doesn't mean the prediction at an
earlier position receives the later target. And the real-valued hard-mask theorem also rules out
a backward score contribution at those forbidden coordinates.

<div id="matching-50-updates-against-pytorch"></div>

## Fifty matched updates with PyTorch

I wanted to check my training calculation against PyTorch too. So I
reconstructed the Week 3 model in PyTorch from the same TorchLean checkpoint and ran both
trainers on the same 50 batches, with matching dialogue records, assistant-row weights,
learning-rate schedule, and AdamW settings. Both runs used strict FP32 on the same A100 with TF32
disabled, and PyTorch used scaled-dot-product attention (SDPA) and fused AdamW.
[TensorFloat-32 (TF32)](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#tensorfloat-32-tf32-on-nvidia-ampere-and-later-devices)
is an NVIDIA matrix-computation mode with less input precision than full FP32, and
I disabled it so those arithmetic choices wouldn't get mixed into this comparison.

<figure class="data-chart" id="matched-training-chart">
  <picture>
    <source media="(max-width: 600px)" srcset="results/matched-runtime-50-step/loss-and-throughput-mobile.svg">
    <img src="results/matched-runtime-50-step/loss-and-throughput.svg" alt="Loss and throughput of TorchLean and PyTorch over fifty matched training updates">
  </picture>
  <figcaption>Both FP32 runs start from the same checkpoint and use the same 50 batches.
    Their loss curves nearly overlap (the markers are validation checks), and the bars below compare
    median update throughput on the same A100. These are the saved FP32 results (if you rerun the
    comparison script now, it uses binary64).</figcaption>
</figure>

A matched comparison needs agreement before the first update, on the parameter ordering, the
shared embedding table, normalization constants, and sampled data, and then also on the state
carried between updates (especially AdamW's moments and step count). If you matched only the
model dimensions, you'd be left with many possible reasons for the curves to differ.

Precision has to be specified too. The archived comparison used FP32 on both paths and disabled
TF32, while the current scripts match the newer binary64 runtime. Week 3 also retains the
original checkpoints' different LayerNorm epsilon settings on CPU and CUDA, so a cross-device
comparison must account for that choice. And the archived graph measures the configuration
recorded with it, so it shouldn't be relabeled as a result for the new adapter.

How close did the two trainers get? The largest absolute training-loss difference was about
0.00001193. Final validation losses
were 2.431321 in TorchLean and 2.431322932 in PyTorch. This differential test exercises the
checkpoint layout, forward pass, target selection, loss, backward pass, and optimizer update all
together, so curves this close give me evidence that those pieces agreed on the measured run,
though they don't establish agreement for every batch or explain the checkpoint's remaining
errors.

PyTorch also ran faster. Its median update throughput was about 19,207 scheduled tokens per
second, compared with 5,596 for TorchLean, a ratio of 3.43 on my setup.
Both timings include the full training update, but PyTorch used SDPA and fused AdamW, so
several implementation choices changed together. I don't know yet where the speed difference comes
from. To work that out, I'd need a backend trace (to identify its selected attention kernel) and
timings of the individual operations.
These values are all in the [comparison record](results/matched-runtime-50-step/comparison.json).

<div id="saving-and-resuming-the-whole-training-state"></div>

## Resuming with the optimizer state intact

What does a long run need to save so it can resume? More than a parameter file. AdamW carries first and second moments, the
learning-rate schedule and sampled data windows depend on the step number, and a stochastic
continuation also needs its random-operation state. Reload the weights but restart those other
fields, and you get a different sequence of updates.

In [TrainingCorrectness.lean](TorchLeanGPT/TrainingCorrectness.lean) I gave the indexed loop a
small pure semantics, in which an update receives both the state and the global step number.
Split the run after <span data-tex="m">m</span> steps, restore that exact state, and continue for
<span data-tex="n">n</span> steps with the same global indices:

<div class="math-display" data-tex="\operatorname{run}(s,m+n,x_0)=\operatorname{run}\bigl(s+m,n,\operatorname{run}(s,m,x_0)\bigr)">A split run equals an uninterrupted run when its state and global step indices agree.</div>

`resume_eq_uninterrupted` proves this for any step-indexed update function, which tells me what
the checkpoint path must preserve. That pure theorem doesn't cover the serializer or the
restoration of device buffers, so for those I ran a separate small CUDA resume check (before the
LibTorch migration) that compared the final parameter and optimizer files byte for byte after a split
run with dropout, and both matched the uninterrupted run.

Underneath, the definition is short. This excerpt from `TrainingCorrectness.lean` leaves
the state type and update function general:

<pre data-lean-block="training-steps"></pre>

With zero steps it returns the state unchanged. Otherwise it applies `update` at
`start` and runs the remaining steps beginning at `start + 1`. Induction over the number
of steps in the first part then proves the split-run equation: peel off one update, and apply
the induction hypothesis to what's left (the remaining first part and the second part).

I left `State` general on purpose, because the equation concerns how we compose updates. In a
training run, though, it must represent everything that influences those updates: proving this
equation for a state containing only weights wouldn't justify forgetting the optimizer moments,
and in the same way any random stream that changes the updates has to be included in the
restored state or otherwise fixed by the update function's inputs.

Let's make the step index visible with a small update that adds `step + 1` to a natural-number
state. Four updates give `1 + 2 + 3 + 4`, and if you resume after the first two you get the same
answer, provided the suffix starts at step two.

<pre data-lean-example="resume"></pre>

Optimizer memory can change the very next update even when the restored weights are identical,
and a one-dimensional momentum example makes that visible. Take the rule

<div class="math-display" data-tex="m_{t+1}=0.9m_t+g_{t+1},\qquad w_{t+1}=w_t-0.1m_{t+1}.">A momentum update uses both the weight and its retained gradient history.</div>

At $w_t=1$, $m_t=2$, and next gradient three, the new momentum is 4.8 and the weight is
0.52. Reload the same weight but reset momentum to zero, and the next weight is 0.7 instead.
Both checkpoints begin with the same model output, yet their training continuations differ.
I picked this toy rule only to illustrate state restoration. The GPT run uses AdamW, whose two moment
buffers and bias-correction index create their own restoration requirements.

Our resume theorem works with the complete state, so it never has to expand every AdamW
coordinate to prove how adjacent update sequences compose. What it gives the serializer is a
precise obligation: the loaded state must equal the state reached at the split point, and the
continuation must use the same global indices. The earlier byte-for-byte experiment checks one
serializer and runtime configuration, while the general equation explains why restoring those
values preserves the continuation for any indexed update function.

<div id="cached-generation-and-its-correctness-proof"></div>

## Reusing the KV cache

A key/value (KV) cache stores earlier attention projections. Without one, the full-prefix
generator reevaluates the complete padded prefix after each token. Most of that
work repeats, because earlier tokens have already produced their attention keys and values, so
the cached generator retains those rows and adds only the projections for the newly consumed
token.

Our cache proof states exactly what must be stored. A represented cache contains the key and
value projections of the consumed history, in order, and one incremental step appends the new
rows, returns the reference output, and preserves that invariant. Induction over a suffix gives
`CachedDecode.run_correct`, and the layered result carries it through the complete causal stack.
(Both paths use the same abstract attention function in these results.)

At one layer, the new token has a query and contributes a new key and value. Its query reads all
the retained keys, and the resulting weights combine the retained values. Append the new
key/value pair in the correct order and the next step gets the same history that the full-prefix
calculation would reconstruct. We cache the projections rather than the final vocabulary logits
because those are the values future attention needs.

A multi-layer cache contains separate projections for each layer. A key from the first
block can't stand in for a key from the twelfth, since those blocks have different parameters
and hidden inputs. So the proof carries the representation relation through the stack, and each
incremental block sees the cached values for its own reference calculation.

Why is it valid to keep the earlier entries? Because of the causal condition: an earlier hidden
position cannot depend on a token that has not arrived yet. In deterministic inference (fixed
parameters, same positional convention), processing a new suffix does not require recomputing
those
earlier projections. That argument applies to the represented history. Dropping old tokens or
changing their position indices needs its own cache relation, and no, a cache doesn't make
the model's 1,024-token context unlimited.

Different numerical kernels can still produce different logits from that logical computation.
The [generation theorem](TorchLeanGPT/GenerationCorrectness.lean) handles this with a margin
condition: if the cached logits are within ε of the reference logits and the reference winner
has margin greater than 2ε, both choose the same next token, and applying that condition at each
prefix reached by the reference continuation proves equality of the whole greedy suffix. Both
generators subtract the same repeat penalty before choosing, so the argument applies to those
penalized scores too.
For now, the runtime checker reports observed errors and margins. It doesn't construct the Lean
certificates required by that theorem, and I haven't built that part yet.

We only need certificates along the reference continuation we reach. After certifying one
choice, both generators append the same token and receive the same next history, and we can
then use the certificate for that history, so you never have to enumerate all vocabulary
branches of an enormous generation tree, although obtaining the numerical bound at each
reached prefix is still a separate task.

This connects the cache result to the [first post's margin argument](../week-01-batch-invariant-inference/#when-score-changes-preserve-the-next-token). The cache proof checks
what the saved keys and values mean, and a numerical error bound plus a winning margin then
connect possibly different implementations of that attention calculation to the same greedy
tokens. You need both, because neither a well-shaped cache nor one agreeing output establishes
the complete suffix on its own.

In three archived warm runs, both generation paths emitted the same eleven token IDs before
ending, and cached prefill and generation took about 168–172 milliseconds, versus 9.43–9.74 seconds
for the full-prefix path. That's a roughly 56-fold speedup! It belongs to this checkpoint,
prompt, short continuation, and A100 setup, though. I reran the benchmark on the current LibTorch adapter, which
computes in binary64. My trained checkpoint isn't public, so I used a GPT-2-small model with one
update: 16 tokens took 383–416 milliseconds with the cache and 25.5–26.6 seconds without it (about
64 to 67 times faster), with the same token IDs on both paths.

## Conclusion

The assistant still needs work. Lower validation loss did not make its answers dependable,
and I want to keep measuring that behavior as we improve the training recipe. I think it's useful
to have the training curves and the actual answers together, especially when they tell
different stories.

I'd also like to rerun the matched comparison with the current LibTorch adapter. The
archived FP32 run was 3.43 times slower than PyTorch. At least that gives me a specific
performance problem to work on. We can retain the target-alignment, resume, and cache proofs
while changing the implementation, and then check that the new run still meets their
assumptions.

## References

- Alec Radford and colleagues. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), 2019. GPT-2's architecture and next-token objective.
- Andrej Karpathy. [minGPT](https://github.com/karpathy/minGPT) and [nanoGPT](https://github.com/karpathy/nanoGPT). The training implementations I used as a starting point.
- Ilya Loshchilov and Frank Hutter. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), 2019. AdamW.
- Hugging Face. [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), and [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2). Dataset cards for the pre-training and instruction-tuning data.
- Lean. [Weakest-precondition semantics](https://github.com/leanprover/lean4/blob/v4.35.0/src/Std/WP.lean) and [`vcgen`](https://github.com/leanprover/lean4/blob/v4.35.0/src/Init/Tactics.lean#L2426). The program-verification tools used for the batch-packing proof.
- PyTorch. [Numerical accuracy](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html). The TF32 arithmetic choice disabled in the archived matched comparison.
