# Programs that read each other's code

On October 21, Jacob Tsimerman gave a talk at SLMath called
[Mathematical AI Safety: What is it, why do it, how is it looking?](https://www.slmath.org/seminars/29593).
He is the scientific director of [MAISI](https://maisi.org/), the institute whose research
directions I wrote about in the [last post](../week-06-heuristic-estimators/). Near the end he
described one direction he finds promising. Two programs play a prisoner's dilemma, and each
one can read the other's source code before it moves. Each program also has a budget of
computation, and the question is what happens when programs with different budgets meet.

The programs in his example come from a paper by Andrew Critch, Michael Dennis and Stuart
Russell, [Cooperative and uncooperative institution designs](https://arxiv.org/abs/2208.07006).
One of them, CUPOD(k), cooperates unless it finds a proof of at most k characters that its
opponent will defect against it. Its mirror image, DUPOC(k), defects unless it finds a proof of
at most k characters that its opponent will cooperate. These programs decide by proving things
about each other, so I wanted to see how much of this I could check in Lean.

The paper leaves several bounded matchups as open problems. I'll start with an idealized
version, where an agent asks what is provable at all instead of running a search with a budget.
Barasz, Christiano, Fallenstein, Herreshoff, LaVictoire and Yudkowsky
[worked out](https://arxiv.org/abs/1401.5577) how to compute the outcomes of such agents with
the logic of provability. In this post I build that computation in Lean, prove the outcomes of
the classic matchups at every world, and then add the idealized versions of the talk's two
agents.

I also asked Thinking Machines' Inkling to propose agents of its own, using Lean's checked
outcomes as training feedback. The proposed policies didn't beat FairBot, but one passed its
original population and failed when the opponents changed. I'll come back to that experiment
after the classic matchups, then finish with a smaller, executable bounded-search prototype.

## The prisoner's dilemma with source code

In a prisoner's dilemma each player chooses to cooperate (C) or defect (D). Mutual cooperation
is better for both than mutual defection, but whatever the other player does, defecting pays a
little more. Two players who reason only about payoffs therefore defect, and both end up worse
off than they could have been.

Reading the opponent's source code changes this. The simplest program that uses it is
CliqueBot, which cooperates exactly when its opponent's code is identical to its own. Two copies
of CliqueBot cooperate, and nobody can exploit CliqueBot, because a defecting program is never
an identical copy. But CliqueBot also defects against a program that does exactly what it does
and differs by a single space. I'd want cooperation that survives that.

The fix is to cooperate based on what the opponent will do, rather than on what its code looks
like. The trouble is that what the opponent will do may depend on what I do, so a program that
simulates its opponent can run forever. Reasoning with proofs avoids that recursion, although
a proof search with no budget can still run forever when there is no proof to find.

## FairBot and Löb's theorem

FairBot cooperates with its opponent when it can prove that the opponent cooperates with it.
For these classic agents, "prove" means prove in Peano arithmetic, assumed sound.
Against DefectBot, which always defects, there is no such proof, so FairBot defects. Against
CooperateBot, which always cooperates, the proof is immediate, so FairBot cooperates. Against
itself, FairBot cooperates if it can prove that FairBot cooperates with FairBot. At first both
mutual cooperation and mutual defection look possible.

Löb's theorem settles it. It says that if arithmetic proves "if $P$ is provable, then $P$", then
arithmetic proves $P$. Take $P$ to be "FairBot cooperates with FairBot". FairBot's own definition
says that a proof of $P$ makes $P$ true, so arithmetic proves $P$, and both copies cooperate.
Barasz and his coauthors prove this as their Theorem 3.1, after an argument by Vladimir Slepnev.

To compute outcomes like this, they write each agent as a formula of provability logic, where
$\Box\varphi$ reads "it is provable that $\varphi$". FairBot against an opponent $b$ is
$\Box\,[b \text{ cooperates with FairBot}]$. Every mention of the opponent's behavior sits under
a box, though an agent can combine its provability tests with the usual connectives, negation
included. These formulas have a semantics that a computer can evaluate, and that is what I build
next.

## Evaluating provability on a chain of worlds

The semantics uses a chain of worlds $0, 1, 2, \dots$, where world $n$ sees exactly the worlds
below it. A box holds at world $n$ when its contents hold at every world below $n$. At world $0$
there is nothing below, so every box holds there, $\Box\bot$ included: world $0$ is a world where
everything is provable. It is a device of the semantics, not a model of arithmetic, and the
worlds are not rounds of play or proof budgets. What carries over to arithmetic is the value a
formula settles on at high worlds, and the last section says when.

<pre data-lean-block="program-modal"></pre>

On this chain, Löb's theorem becomes a statement I can prove directly: $\Box(\Box\varphi \to
\varphi) \to \Box\varphi$ holds at every world, for every formula and every valuation of its
atoms. The proof is an induction down the chain. The same argument works on any frame whose
"sees" relation is transitive and well founded, and transitivity matters. On three worlds where
2 sees 1 and 1 sees 0, but 2 does not see 0, Löb's axiom fails.

<pre data-lean-block="program-loeb"></pre>

## Agents and their outcome table

I write each agent as a schema over two kinds of atoms: "my opponent cooperates with me", and
"my opponent cooperates with agent $c$" for some fixed agent $c$. The same schema is used against
every opponent. An agent therefore cannot look at who it is playing, only at what it can prove
about how the opponent plays. That excludes CliqueBot, which is the point.

Every atom sits under a box, so an agent's decision at world $n$ only reads worlds below $n$.
That lets me compute the whole table of outcomes world by world: `play n a b` says whether $a$
cooperates with $b$ at world $n$.

<pre data-lean-block="program-agents"></pre>

The recursion isn't just one way to fill in the table. The table satisfies every agent's
defining equation at every world, and any other valuation that does so is equal to it, so the
agents' definitions leave no choices on the chain.

<pre data-lean-block="program-unique"></pre>

Each matchup also settles. If the contents of a box first fail at world $m$, the box is false at
world $m + 1$ and stays false, and if they never fail, the box is true everywhere. A modalized
formula is built from finitely many boxes, so it settles too, under any valuation. Applied to the
table, every matchup gets its own cutoff, after which it never changes, and so it has exactly one
eventual outcome.

<pre data-lean-block="program-stabilize"></pre>

<pre data-lean-block="program-table-stabilize"></pre>

## The table of outcomes

Here are the agents. CooperateBot and DefectBot ignore the opponent. FairBot is the formula
above. `twin` is FairBot written differently, $\Box(\text{me} \lor \bot)$. PrudentBot adds a
second requirement, and `cupod` is the idealized CUPOD. I come back to both below.

<pre data-lean-block="program-bots"></pre>

Lean can print the moves. Each line is one matchup, and its four letters are the first agent's
moves at worlds $0$, $1$, $2$ and $3$.

<pre data-lean-example="programTable"></pre>

At world $0$, FairBot, its twin and PrudentBot cooperate with everyone, even DefectBot, because
every box holds there. CUPOD, whose formula negates a box, defects against everyone at world
$0$. From world $1$ on, FairBot defects against DefectBot. PrudentBot cooperates with
CooperateBot at worlds $0$ and $1$ and defects from world $2$ on. After that none of these
sequences changes.

I don't have to trust the printout for small worlds only. Each matchup here is proved at every
world.

<pre data-lean-block="program-fair"></pre>

FairBot's safety property holds against every agent in any population, not just the agents in
my table. If an agent defects against FairBot at some world, FairBot defects against it at every
higher world, so nobody exploits FairBot in the limit. The exact condition goes further: FairBot
eventually cooperates with $b$ exactly when $b$ cooperates with FairBot at every world.

<pre data-lean-block="program-unexploitable"></pre>

## Cooperation without identical code

FairBot and `twin` have different formulas. CliqueBot would refuse to cooperate across that
difference, but FairBot and its twin cooperate with each other at every world, for the same
reason FairBot cooperates with itself. Each one's cooperation is provable from the other's, and
Löb's theorem turns that into a proof that both cooperate. This is the robustness I was after:
the agents only need to prove things about each other's behavior, not recognize each other's
text.

## PrudentBot and CooperateBot

FairBot has one weakness. It cooperates with CooperateBot, which would also cooperate with
anyone, so FairBot leaves something on the table. PrudentBot fixes this. Besides a proof that
its opponent cooperates with it, it wants a proof that its opponent defects against DefectBot,
assuming arithmetic is consistent:
$\Box\,\text{me} \land \Box(\lnot\Box\bot \to \lnot[\text{opponent cooperates with
DefectBot}])$.

CooperateBot fails the second test, so PrudentBot defects against it from world $2$ on. The
assumption $\lnot\Box\bot$ explains the delay. It is false at world $0$, so the second test
first bites at world $1$, and a box first looks at world $1$ from world $2$. FairBot and
PrudentBot pass each other's tests, so they cooperate with each other at every world, and
PrudentBot cooperates with itself.

<pre data-lean-block="program-prudent"></pre>

## The talk's agents, idealized

Back to the talk. If I replace DUPOC(k)'s bounded search by ordinary provability, "defect unless
it is provable that you cooperate" is FairBot. The same change to CUPOD(k) gives
$\lnot\Box\lnot\,\text{me}$: cooperate unless it is provable that the opponent defects. I add
that agent to the table.

<pre data-lean-block="program-cupod"></pre>

Critch, Dennis and Russell prove that, under their assumptions on the proof system, for every
sufficiently large $k$ two copies of CUPOD(k) defect against each other (their Theorem 3.4) and
two copies of DUPOC(k) cooperate (Theorem 3.7). The idealized agents do the same at every
world. CUPOD against CUPOD is a fixed point of $p \leftrightarrow \lnot\Box\lnot p$, and Löb's
theorem makes $\lnot p$ provable. Both results surprise people: the paper notes that most
computer science graduate students, asked about two DUPOCs, guess that they defect.

The mixed pairing is their Open Problem 3: does DUPOC(k) against CUPOD(k) end in (D, C) for
every sufficiently large $k$, with DUPOC defecting and CUPOD cooperating? They conjecture that it
does. For the idealized agents it does. FairBot cooperates with CUPOD only at world $0$, and
CUPOD cooperates with FairBot from world $1$ on. That doesn't settle their question, though.
Their search ranges over proofs of at most $k$ characters, which bounds the length of a proof
rather than the running time, and whether proofs of the right lengths exist is exactly what is
open. Whether a bounded PrudentBot exists at all is their Open Problem 9.

The table also has no proof-length parameter, so it cannot reproduce what Tsimerman was after,
agents with different budgets meeting each other.

<pre data-lean-block="program-outcomes"></pre>

## Cooperation is also collusion

Tsimerman ended on a point about all of this. Cooperation sounds good, but cooperation between
agents is the same thing as collusion, and with many agents in the wild we might not want them
to cooperate too much. He suggested we might want gatekeeping, and he presented this line of
work as one promising direction among several.

Nothing here addresses that question directly. The cooperation in my table assumes that each
agent can read the other's code, and it says nothing about how cheaply real systems could
coordinate, or how to stop them. FairBot and its twin have different formulas and still reach
mutual cooperation by reasoning about each other's behavior.

## Asking Inkling to write the policies

Could a model find a better policy if it could try formulas and receive checked feedback?
I connected Inkling to the evaluator above. The model returns a small constructor tree:
boxes, Boolean connectives and the allowed references to other agents. It does not return
Lean code for us to execute. A host program searches for a stopping certificate, and a fixed
Lean program checks it before the proposed policy can enter the game.
In the training studies, each proposal is conditioned on the current population, with the
opponents' schemas visible.

There is a subtlety in that stopping test. Seeing the same outcome table at two consecutive
worlds isn't enough: a box buried inside a formula might still change later. The certificate
checks agreement of every nested box between a world and its successor, as well as the
claimed outcome table. I prove that this complete box state keeps agreeing, so the table
never changes again.

<pre data-lean-block="program-certificate"></pre>

A valid certificate needn't describe a policy we'd want. DefectBot has perfectly definite
outcomes! Admission additionally requires no unilateral cooperation against a defector,
self-cooperation, and mutual cooperation with at least one other member of the population.
`accepted_safe` connects that executable criterion to eventual non-exploitation. Its
quantifier ranges over the supplied population; it doesn't include every possible
opponent.

<pre data-lean-block="program-admission"></pre>

I compared two ways of proposing policies. In the unrestricted arm, Inkling writes the whole
formula. In the shielded arm, it writes an extra condition $g$, and the host constructs
$\Box\,\text{me} \land g$. That prefix already has a non-exploitation theorem, for any extra
condition and any opponent in a compatible modal universe. Training can change how useful the
policy is, but it isn't responsible for this guarantee.

<pre data-lean-block="program-guard"></pre>

## What changed with training?

The two completed studies used 2,048 proposals altogether, half for training and half for
before-and-after evaluation. There were 126 actual optimizer updates. Each row below pools
two runs, with 128 evaluation attempts before training and 128 afterward.

Cooperation is the fraction of external opponents with whom an admitted proposal mutually
cooperates. An invalid or rejected attempt scores zero. Admission is reported separately so
we can see whether a model produces useful policies more often, rather than only inspect
its successful attempts.

| Study | Policy class | Admission, before → after | Cooperation, before → after |
| --- | --- | ---: | ---: |
| First, with protocol amendments | Unrestricted | 43.75% → 50.00% | 13.73% → 17.59% |
| First, with protocol amendments | Shielded | 76.56% → 68.75% | 27.06% → 24.30% |
| Fresh replication | Unrestricted | 35.94% → 47.66% | 13.67% → 18.02% |
| Fresh replication | Shielded | 74.22% → 71.09% | 26.07% → 25.29% |

The unrestricted arm improved modestly in the replication. Its cooperation change was
+4.35 percentage points, with a case-resampled 95% interval of [+0.61, +8.20]. It produced
acceptable policies more often; cooperation among admitted proposals stayed roughly flat.
Neither shielded comparison showed a clear learning gain. These intervals describe these
runs and populations, not a general result about training: both studies reuse the same
32 test populations and two training halves.

FairBot scored 36.27% on those populations, above every trained arm. No admitted final
proposal beat FairBot on its original population. The unrestricted arms also proposed
39 and 46 exploitable policies in their final evaluations. Those were rejected and never deployed.
The shielded arms proposed none, as their fixed wrapper theorem predicts. I wouldn't call
that learned alignment.

## A certificate that doesn't transfer

One unrestricted proposal from the first study passed its original population. Written in
the experiment's notation, it was

$$
\Box\bigl(\lnot\Box\,\text{me}
  \;\to\; \Box[\text{opponent cooperates with DefectBot}]\bigr).
$$

I kept that policy unchanged and tried it in eight reserved populations. Two of them exposed
it: the policy cooperates with an opponent that defects against it. Lean checked the new
outcome tables and rejected admission in both cases.

The original theorem was still right. It concerned a particular population, and these were
different populations. This is the example I'd use to explain why testing more opponents
and proving an arbitrary-opponent theorem are different tasks. The guarded class has the
latter theorem; an unrestricted policy doesn't acquire it by passing a finite test.

Counting repeated proposals with their original attempt weights, 510 of 512 unrestricted
transfers passed in the first study, and all 488 passed in the replication. The failure did
not recur in the second study. Both guarded cohorts passed all their transfers. The
[offline replay](../week-07-program-equilibrium/experiments/) includes the exact
three certificates for the original success and the two failures, as well as the full
evaluated set.

## Giving reciprocity more room

FairBot demands reciprocity at every earlier world, including the early worlds that don't
model sound arithmetic. I tried a larger family that only demands it above a finite cutoff:

$$
\Box(\lnot\Box^k\bot \to \text{me}) \land g.
$$

On the chain, $\Box^k\bot$ holds exactly below world $k$. This policy therefore ignores
reciprocity below $k$, while still eventually defecting against any eventual defector.
The theorem holds for any finite $k$ and extra condition $g$, in finite or infinite
compatible populations. It doesn't promise non-exploitation at every early world, and
$k$ is still a modal index, not a number of game rounds or a proof-search budget.

<pre data-lean-block="program-delayed-guard"></pre>

This is related to the paper's
[WaitFairBot family](https://arxiv.org/html/1401.5577#S5), whose formula also includes an
outer $\lnot\Box^k\bot$ conjunct. I included those policies among the handwritten controls;
the delayed-cooperation idea isn't a new mechanism discovered by Inkling.

For the follow-up, I looked for one fixed policy to use across populations. Base Inkling and
one saved trained adapter each proposed 24 policies over six feedback rounds. They saw the
same pooled development feedback. There were no further
weight updates. Forty-one of the 48 responses passed syntax validation. Selection used
16 new development populations, with ties retaining a handwritten control, before evaluating
48 held-out populations. The combinations were new, but the opponent catalog was the same.

| Policy | Development cooperation | Held-out cooperation |
| --- | ---: | ---: |
| FairBot | 42.75% | 45.71% |
| Selected handwritten control | 46.76% | 45.21% |
| Best base-model proposal | 46.76% | 45.21% |
| Best trained-model proposal | 46.76% | 45.21% |

The primary selection retained the control for both sources. I also checked the actual best
model proposals separately. Their constructor trees differ, but the base model duplicated a
conjunct and the trained model added a tautology. Lean proves both equivalent to the supplied
one-world control for every valuation, not just on these test populations.

<pre data-lean-block="program-inkling"></pre>

So the development improvement didn't carry over: the selected policy was 0.50 percentage
points below FairBot on held-out cases, with interval [−3.58, +2.52]. This didn't find a new
strategy or establish an advantage from training. It did leave us with an experiment where
we can inspect the proposed programs, reject the ones that fail the criterion, and prove why
some apparently different answers are the same policy.

The [experiment notes and replay commands](../week-07-program-equilibrium/experiments/)
give the training settings, protocol amendments, selection rules and exclusions. The export contains programs and certificates. It leaves out credentials, private checkpoints
and raw API traces.

## What Lean checks here

Lean checks these properties of the chain semantics:

- Löb's axiom at every world of the chain, and on every transitive, well-founded frame, together
  with the counterexample that shows transitivity is needed.
- The outcome table: it satisfies every agent's defining equation at every world, it is the
  only valuation that does, and every matchup settles to one eventual outcome.
- The matchups in this post at every world, and FairBot's safety property against every agent in
  any population that contains it.
- The stopping certificate and admission criterion for an exact finite population, the strict
  and delayed guard theorems for arbitrary compatible populations, and the semantic equalities
  of the selected Inkling proposals.

The general theorems are kernel-checked. Individual experimental certificates run the proved
Boolean checker through Lean's runtime; they don't each produce a new kernel proof object.
Those runs also trust the compiler and runtime. None of these statements verifies Inkling's
neural weights or establishes real-world alignment.

The step from the chain to arithmetic is cited, not proved, and it needs more than my
definitions. Barasz and his coauthors consider agents given by arithmetic formulas whose
defining equations Peano arithmetic proves, ordered by rank so that an agent only refers to
simpler ones. The classic bots above fit that: PrudentBot refers to DefectBot, and the others
refer to no fixed agent. I haven't established that rank condition for the generated
populations; their results here remain about the chain. For the ranked agents, Theorem 4.7
gives each matchup a representative formula
without atoms, using the arithmetic soundness of provability logic and its fixed point theorem
(their Theorems 4.1 to 4.3, after Lindström). Because the chain validates provability
logic, those representatives satisfy the agents' equations there, so by uniqueness their values
are my table. For the matchups here they
come out as $\top$, $\bot$, $\Box\bot$, $\Box^2\bot$ or $\lnot\Box\bot$: PrudentBot against
CooperateBot is $\Box^2\bot$, and the two sides of FairBot against CUPOD are $\Box\bot$ and
$\lnot\Box\bot$. If arithmetic is sound, $\Box\bot$ and $\Box^2\bot$ are false, which is exactly
their value at high worlds. I haven't formalized this arithmetic step or Peano arithmetic's
proof predicate.

## A bounded self-play prototype

The modal index isn't a proof-search budget. I wanted a small example where we could change
an actual budget, run the agents, and inspect the evidence behind a move. The new interpreter
does that with two cooperation-certificate rules.

The first rule recognizes a program that always cooperates. The second recognizes two
searchers, checks their program descriptions and budgets, and verifies that each can afford
the reciprocal certificate its own search will look for. It doesn't call either agent while
checking the certificate. Both searches contain that certificate, so their actual executions
return cooperation. Lean proves this argument from the interpreter's definitions; there is
no assumed reflection axiom.

Each agent tries at most two candidates. The budget counts the UTF-8 bytes in a certificate,
including the names and punctuation. A searcher cooperates only if it finds an affordable,
valid certificate that its opponent cooperates with it. Here is the same program with three
different budgets:

<pre data-lean-example="boundedSelfPlay"></pre>

At 15 bytes it defects against itself. At 16 bytes it finds `R("self","self")`, a 16-byte
certificate, and cooperates. It still cooperates at 64. The general theorem gives the exact
threshold for every name and budget, and works for either search order. Another theorem says
the searcher cannot cooperate unilaterally against any registered opponent, including an
arbitrary total Lean function. A function can genuinely cooperate without either rule proving
it, so this search is incomplete.

<pre data-lean-block="program-bounded"></pre>

To run self-play, the two search orders and the failure cases together:

```bash
lake exe play_bounded_agents 16
```

This is a first bounded experiment, **not a solution to Open Problem 4**. The certificates refer to two specialized rules that Lean proves sound, and none of them is
an arithmetic proof. Their byte counts
don't include expanded Lean proof terms. Names resolve through a fixed program registry;
they aren't source-code fingerprints. The two different search orders cooperate, but this
doesn't establish cooperation between arbitrary implementations of DUPOC.

The full target is still
[Critch, Dennis and Russell's parametric bounded Löb theorem](https://arxiv.org/abs/2208.07006),
Lemma 3.6. Under their proof-system and growth assumptions, a uniform proof of

$$
\forall k > k_1,\quad \Box_{f(k)}p(k) \to p(k)
$$

leads to a proof of $\forall k > k_2,\ p(k)$ for some $k_2$. Here $\Box_{f(k)}$ means an actual
arithmetic proof with a length bound. Connecting that theorem to the executable search needs a
real proof encoding and checked length bounds. Reinterpreting modal worlds won't do it. The
[next post](../week-08-bounded-lob/) proves that theorem in Lean from the properties it needs,
builds a proof checker that has all of them, and runs two copies of DUPOC that find a proof and
cooperate. For now, this prototype is a terminating example with explicit budgets and proved
operational guarantees to build on :)
