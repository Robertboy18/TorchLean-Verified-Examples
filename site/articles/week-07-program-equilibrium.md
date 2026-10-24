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

The paper leaves several bounded matchups as open problems. What I can do is study an idealized
version, where an agent asks what is provable at all instead of running a search with a budget.
Barasz, Christiano, Fallenstein, Herreshoff, LaVictoire and Yudkowsky
[worked out](https://arxiv.org/abs/1401.5577) how to compute the outcomes of such agents with
the logic of provability. In this post I build that computation in Lean, prove the outcomes of
the classic matchups at every world, and then add the idealized versions of the talk's two
agents.

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
Here and below, "prove" means prove in Peano arithmetic, and I assume arithmetic is sound.
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

## What Lean checks here

Lean checks these properties of the chain semantics:

- Löb's axiom at every world of the chain, and on every transitive, well-founded frame, together
  with the counterexample that shows transitivity is needed.
- The outcome table: it satisfies every agent's defining equation at every world, it is the
  only valuation that does, and every matchup settles to one eventual outcome.
- The matchups in this post at every world, and FairBot's safety property against every agent in
  any population that contains it.

The step from the chain to arithmetic is cited, not proved, and it needs more than my
definitions. Barasz and his coauthors consider agents given by arithmetic formulas whose
defining equations Peano arithmetic proves, ordered by rank so that an agent only refers to
simpler ones. My schemas fit that: PrudentBot refers to DefectBot, and the others refer to no
fixed agent. For such agents their Theorem 4.7 gives each matchup a representative formula
without atoms, using the arithmetic soundness of provability logic and its fixed point theorem
(their Theorems 4.1 to 4.3, after Lindström). Because the chain validates provability
logic, those representatives satisfy the agents' equations there, so by uniqueness their values
are my table. For the matchups here they
come out as $\top$, $\bot$, $\Box\bot$, $\Box^2\bot$ or $\lnot\Box\bot$: PrudentBot against
CooperateBot is $\Box^2\bot$, and the two sides of FairBot against CUPOD are $\Box\bot$ and
$\lnot\Box\bot$. If arithmetic is sound, $\Box\bot$ and $\Box^2\bot$ are false, which is exactly
their value at high worlds. I haven't formalized this step, Peano arithmetic's proof predicate,
or bounded proof search.

The next thing I'd like to try is the bounded version. Critch, Dennis and Russell prove their
results with a parametric bounded Löb theorem, and stating it in Lean, next to a proof search
with a real budget, would let us test the talk's agents directly. Their Open Problem 4 asks for
an implementation of DUPOC with heuristic proof search in a proof assistant, one whose self-play
halts with cooperation on an ordinary computer. We'll keep adding examples as this goes :)
