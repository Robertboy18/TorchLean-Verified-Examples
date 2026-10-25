# Löb's theorem with a budget

[Last week's](../week-07-program-equilibrium/) modal agents could prove anything that is
provable, which no computer can do, and the bounded prototype at the end of that page only
searched a few fixed kinds of certificate. Jacob Tsimerman's question at the end of his SLMath
talk was about real bounded agents: programs that only accept proofs up to a fixed length. Those
are the only agents we will ever actually deploy, so they are the ones I care about.

The agent at the center of that question is DUPOC(k), from Critch, Dennis and Russell. It
searches for a proof, of at most k characters, that its opponent will cooperate with it. If it
finds one, it cooperates; otherwise it defects. Their Theorem 3.7 says that two copies of
DUPOC(k) cooperate once k is large enough, and their Open Problem 4 asks whether anyone can make
that happen on an ordinary computer, with a real proof search that halts.

This week I got it to happen, in a proof system built for these agents whose bounded Löb
properties are all proved. Two copies of DUPOC(17880), each reading the other's source and
searching for a proof about it, find a proof of 3,007 characters, check it, and cooperate. The
match takes about two milliseconds on one core. Everything around that run is checked in Lean:
the proof system, the search, the theorem that guarantees the proof exists, and what the
programs do with it. Along the way the same machinery answered another of the paper's open
problems and turned up a mistake in one of its examples.

## The theorem behind bounded cooperation

Löb's theorem says that if a theory like Peano arithmetic proves "if $P$ is provable then $P$",
it proves $P$. With a budget, the statement has to track lengths. Write $\Box_n \varphi$ for
"there is a proof of $\varphi$ with at most $n$ characters". Andrew Critch's parametric bounded
Löb theorem says, in the form the 2022 paper uses: if the theory proves

$$
\forall k,\quad \Box_{f(k)}\, p(k) \to p(k),
$$

and the budget $f(k)$ grows faster than $\log k$, then for some $K$ it proves
$\forall k > K,\ p(k)$. For DUPOC, $p(k)$ is "DUPOC(k) cooperates with itself", and $f(k) = k$.

Its proof is a careful version of the classic one. You build a sentence $\psi(k)$ that says "if
I have a proof of at most $g(k)$ characters, then $p(k)$". Proving $\psi$ costs some fixed
number of characters, and using that proof inside the theory costs about $\log k$ more, because
writing the number $k$ in binary takes that long. Everything else fits into the rest of the
budget. What surprised me is how little room the argument needs: once $f(k)$ beats any constant
times $\log k$, the budget always wins in the end.

Critch proves this for any proof system with a short list of properties, and that is how I
formalized it first. Each property is a field of a structure, so the theorem holds for every
system that has them:

<pre data-lean-block="bounded-system"></pre>

These are the properties a well-designed proof system can be expected to have, though each
depends on how proofs are written: combining two proofs with modus ponens costs a constant number
of extra characters; a proof of a universal statement gives short proofs of its instances; a
theory can check one of its own proofs with a proof only a constant factor longer (the linear
case of Critch's theorem, the one the paper uses); and a diagonal lemma lets a formula talk about
its own provability. Two of them only need to hold from some $k$ on, which turned out to matter.
With those in hand, the theorem is fully proved, with $g(k) = \lfloor b(k) / 2(e+1) \rfloor$ for
the budget $b$, where $e$ is the expansion constant.

<pre data-lean-block="bounded-pblt"></pre>

## One quantifier in the hypothesis

Working through the proof in Lean made me read the growth condition very literally. In the 2022
paper it is written as "there is some constant $c > 0$ with $f(k) > c \log k$ eventually". The
theorem they cite uses it for *every* constant: the length of the proof built from the diagonal
lemma depends on the formula that lemma returns, so it isn't known in advance, and the argument
absorbs it by letting the budget beat every multiple of $\log k$. Critch's own paper states the
condition this way, and so do I. The paper's applications use $f(k) = k$ or $\lfloor k/2
\rfloor$, which satisfy the stronger condition, so their results stand; only the stated
hypothesis of their Lemma 3.6 needs the stronger form.

<pre data-lean-block="bounded-growth"></pre>

## A proof system you can search

A theorem about bounded proofs is only as good as the proof system it talks about, so the next
step was to build one and prove that it has every property above. Its formulas talk about
agents directly. $\Box_t\varphi$ says that $\varphi$ has a proof with at most $t(k)$ characters,
where $t$ is a bound like $k$, $2k$ or $\lfloor k/6 \rfloor + \log k$. $\mathrm{coop}(a, b)$ says
that the agent with source $a$ cooperates with the agent with source $b$. A source is itself a
formula, written with two placeholders, the opponent and me, and running an agent means reading
its formula with the two real sources filled in. DUPOC's source is
$\Box_k\,\mathrm{coop}(\text{opponent}, \text{me})$.

Proofs are strings over 35 characters, with numbers in binary. A proof of a statement is the
statement followed by a certificate, and its length is the number of characters of both. The
checker only reads syntax: the axioms of classical logic, modus ponens, the bounds of the
quantifier, and one rule for each bounded property, with the side condition that makes it
sound. Lean proves that every sentence the checker accepts is true:

<pre data-lean-block="bounded-sound"></pre>

The checker satisfies every field of the structure, with explicit constants, so the bounded Löb
theorem applies to it:

<pre data-lean-block="bounded-calculus"></pre>

Building it changed two of the properties. Inner necessitation, $\Box_x \varphi \to \Box_{2x}
\Box_x \varphi$, needs room in the outer budget for the text of $x$ itself, and a bound can be
small in value but long in text, so as I first stated it, it was false. It holds from some $k$
on: past $2^{|x|}$, where the binary numeral for $k$ that every proof contains already pays for
the text, or past the point where $x(k)$ exceeds the length of its own text by two. Monotonicity
of the bound needs the checker to decide inequalities like $\lfloor k/6\rfloor + \log k + 4000
\le k$ for all $k$ past a point, which it can do whenever the gap between the two sides grows
with $k$, and every inequality the proof uses has such a gap. Both properties now hold from some
$k$ on, and the bounded Löb theorem goes through with the guards carried along.

## DUPOC as a program

An agent's search for a proof of a sentence $\sigma$ with at most $t$ characters has three
stages. First it checks whether $\sigma$ fails at one of the first six worlds of the chain from
last week. Forget the budgets and every rule of the checker is valid at every world, so a
sentence that fails at one world has no proof of any length, and the search can answer no right
away. Then it builds a few candidate proofs from the two sources and asks the checker about
them. Then it checks every certificate with at most $t - |\sigma|$ characters. Lean proves that
the three stages together decide exactly whether a proof of at most $t$ characters exists, so
the first two only make it faster:

<pre data-lean-block="bounded-search"></pre>

The candidates are the paper's own suggestion. Critch, Dennis and Russell point out that if the
search tried the proof of Theorem 3.7 first, both agents would finish at once. Here the agent
builds that proof itself: it runs a certificate builder that follows the proof of the bounded
Löb theorem step by step on the two sources it is given, and the checker decides whether the
result is a proof. The candidate sits in front of the complete search, never instead of it,
which is the lesson from last week's version of this page. In a toy checker with two proofs of
size zero, a search that only tries a template defects while a valid, affordable proof exists:

<pre data-lean-example="boundedCoverage"></pre>

## Two copies of DUPOC

The proof the two DUPOCs build is a certificate of 2,962 characters for the sentence
"for every $k > 17879$, DUPOC(k) cooperates with DUPOC(k)". At a budget $k$ it becomes a proof
of $2993 + \lfloor \log_2 k\rfloor$ characters: the certificate, 17 more for instantiating it
at the guard, 11 for the formula and 3 for the sentence around it, plus the numeral for $k$. At
$k = 17880$ the proof has 3,007 characters, and the whole match, both agents reading both sources
and building and checking their proofs, takes about two milliseconds. The certificate is only
valid past 17879; below that some other affordable proof might exist, and nothing here says what
the smallest cooperating budget is.

Lean proves this about the program that runs, threshold included. The kernel builds the
certificate and checks it, and from there, for every $k > 17879$, DUPOC's search accepts it
without ever reaching the complete search, and DUPOC cooperates with itself:

<pre data-lean-block="bounded-dupoc-self"></pre>

Here is the program itself, run a few times while this page was built:

<pre data-lean-example="boundedRun"></pre>

CUPOD, which cooperates unless it finds a proof that its opponent defects, gets the same
treatment: against itself it builds a 4,457-character certificate of its own defection, valid
past $k = 26963$, and defects (their Theorem 3.4). That threshold comes from the run; only the
DUPOC threshold has its own theorem.

## Programs that are written differently

Open Problem 4 has a second half: how much can the two implementations differ while cooperation
still halts? Critch's 2019 paper answers it for $G$-fair agents, which provably cooperate as soon
as they have a short enough proof that the opponent cooperates, each with its own parameter. I
proved the version for agents that share the parameter $k$, in a form general enough for the
programs I wanted to try. Each agent provably cooperates once it has a proof, within its own
budget, of something the system proves from mutual cooperation. If some budget $u(k)$, growing
faster than $\log k$, leaves room under both, meaning $u(k) + \log k + c$ is eventually below
both agents' budgets for every constant $c$, then the two cooperate for every large $k$.

<pre data-lean-block="bounded-fair"></pre>

For DUPOCs with linear budgets this means: for every fixed $p, q \ge 1$ and $r, s$, DUPOC with
budget $pk + r$ and DUPOC with budget $qk + s$ cooperate for all large $k$. DUPOC against CIMCIC,
which cooperates if it can prove that its cooperation implies its opponent's, is another instance
(their Theorem 5.2). CIMCIC against itself needs no bounded Löb argument at all: in self-play its
condition reads "if I cooperate, then I cooperate", which has a proof of under two hundred
characters, so two CIMCICs already cooperate at a budget of 400.

The agents build these proofs the same way, with a builder that follows the argument. I played
nine agents against each other at $k = 500{,}000$: CooperateBot, DefectBot, DUPOC, CUPOD, CIMCIC,
DIMCID, DUPOC with budget $2k$, DUPOC with budget $k + 100$, and a DUPOC whose source is
$\Box_k(\top \to \mathrm{coop}(\text{opponent}, \text{me}))$. All 81 ordered matches were decided
by the first two stages of the search, together in 0.7 seconds and none in more than 16
milliseconds (timing only the two runs of each match), and the five agents that look for proofs of
cooperation cooperated with each other in every pairing. For the four pairs of DUPOC with a
differently written agent, the certificates have 24,000 to 32,000 characters and are valid from
budgets between about 268,000 and 362,000 on. Those are budgets at which these particular
certificates work, not budgets cooperation needs. The builder also only tries a few ratios between
the two budgets, so a lopsided pair like $128k$ against $k$ is covered by the theorem but not by
the fast path.

## DIMCID defects

DIMCID ("defect if my cooperation implies defection") searches for a proof that if it
cooperates, its opponent defects, and defects if it finds one. Critch, Dennis and Russell leave
the match against DUPOC open (their Open Problem 6). The answer is mutual defection, and the
argument is short. A proof that DIMCID defects is, a few lines later, a proof of DIMCID's
condition, since "I cooperate" is then false and anything follows from it. So a short proof of
DIMCID's defection makes DIMCID defect, which is exactly the self-fulfilling shape the bounded
Löb theorem turns into a proof. DIMCID defects for every large $k$, and DUPOC, which would need a
proof that DIMCID cooperates, finds none, because there is no true one to find. The argument
works in any sound system with the properties above that proves the agents' code and the
propositional fact $\neg A \to (A \to B)$:

<pre data-lean-block="bounded-dimcid"></pre>

The same argument says something about an easier question. Their Example 5.3 asks what DIMCID
does against CooperateBot, and the footnote answers that both cooperate. But against CooperateBot
DIMCID's condition is just "I defect", so the self-fulfilling argument applies again and DIMCID
defects for every large $k$. I think the footnote is a slip. In this system the theorem holds
against every fixed opponent, with a threshold that depends on the opponent, and against DUPOC,
DUPOC defects at every budget:

<pre data-lean-block="bounded-dimcid-run"></pre>

A [Lean engine for open-source game theory](https://github.com/ColombanD/open-source-game-theory),
written for a master's thesis, covers a lot of this ground with a different proof system, a
classical proof search and its own way of counting proof length. It proves the footnote's
$(C, C)$ for DIMCID and CooperateBot, and $(C, D)$ for DIMCID and DUPOC, at every budget. Its
rule for contradictions needs two closed proofs, so it can't derive $A \to B$ from $\neg A$
under a hypothesis, and that step is exactly the one these proofs need.

## What this system can decide

Two of the paper's open problems concern matches whose outcome depends on whether bounded
searches succeed: DUPOC against CUPOD (their Open Problem 3, conjectured $(D, C)$) and CUPOD
against CIMCIC (Open Problem 5). In this system both are settled at every budget, by the worlds.
Each agent's search asks for a sentence that is false at the first world of the chain, so
neither finds a proof of any length: DUPOC defects and CUPOD cooperates, and CUPOD cooperates
while CIMCIC defects.

<pre data-lean-block="bounded-worlds"></pre>

This is an answer about this proof system and not about arithmetic, and the reason is worth
saying plainly. The checker can't verify even trivial finite facts about itself. $\neg \Box_0
\top$, "there is no proof of $\top$ with zero characters", is true, but it is false at the first
world, so the checker has no proof of it. An arithmetic like Peano's proves facts like that by
checking, and it could prove that CUPOD's search runs out, if it really does, by simulating it,
when the simulation is short enough. So this development doesn't settle the two problems for
arithmetic. The thesis engine proves the analogue of Open Problem 3 in its own calculus, by a
transposition of C and D that preserves budgets exactly.

## What is still open

The proof system here is built for agents. It has the bounded properties as rules rather than
deriving them in arithmetic, its diagonal sentences are a primitive, and its programs are the
agents of this game rather than arbitrary code. That is enough for the deterministic results
discussed here and for real runs, though not for the paper's probabilistic agents. An arithmetic
version still needs Gödel numbering, a programming language, proof predicates that count
characters, diagonalization, and the derivability properties with explicit lengths, of which
bounded inner necessitation is the central one. And a certificate that is valid only past a
threshold says nothing about whether shorter proofs exist below it.

## Why I care about this

It's easy to prove things about agents that can't exist. The modal agents of last week are
beautiful, and they settle questions that bounded agents leave open, but the agents that will
read each other's code in the real world will have budgets, deadlines and heuristics. The space
between a theorem about idealized reasoners and a program that halts on a laptop is exactly where
I think formal methods can do the most for AI safety. This week that space got smaller: two
programs that read each other's source, with nothing but a proof checker and a budget, found
their way to cooperation, and Lean checked every step that says why. Lean also caught a
quantifier I would have read past, a property I had stated wrong, and an answer in a footnote.

That's what I'd like to keep working on: the same run inside arithmetic, and agents that are
written more differently than these. We'll keep adding to this as it goes :)
