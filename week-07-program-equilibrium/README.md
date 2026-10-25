# Programs that read each other's code

In the [article](../site/articles/week-07-program-equilibrium.md) I follow a direction from Jacob
Tsimerman's MAISI talk: programs that play a prisoner's dilemma after reading each other's
source, and decide by proving things about each other.

The development computes the outcomes of the modal agents of Barasz et al. (2014) on the chain
of worlds of provability logic:

- `Modal` defines the formulas and their evaluation on the chain, and proves Löb's axiom there
  and on every transitive, well-founded frame (with a counterexample when transitivity fails).
  It also proves that every modalized formula settles to one value along the chain.
- `Agents` defines agents as schemas that cannot look at who they are playing, computes the
  outcome table world by world, and proves that the table is the unique solution of the agents'
  equations and that every matchup has one eventual outcome. FairBot cannot be exploited, and it
  eventually cooperates with an agent exactly when that agent cooperates with it at every world.
- `Bots` defines CooperateBot, DefectBot, FairBot, a differently written FairBot, PrudentBot and
  an idealized CUPOD, proves the matchups the article discusses at every world, and collects
  their eventual outcomes.
- `Certificate` checks full nested-box stability and the claimed table, and proves that
  admitted policies cannot be exploited in the supplied finite population.
- `Guards` proves eventual non-exploitation for strict and delayed reciprocity with arbitrary
  additional guards, against every opponent in a compatible modal universe.
- `Inkling` contains the actual best model-only proposals from the delayed search. Both are
  proved equivalent to the supplied control; its self-cooperation is proved too.
- `Check` is the fixed JSON-lines entry point used by the offline certificate replay.
- `Bounded` is a separate, terminating certificate-search interpreter. It proves soundness,
  non-exploitation and an exact self-play byte threshold for its two certificate rules.
  `Bounded.Run` runs the example with a chosen budget.

The step from the chain to the agents' behavior in Peano arithmetic is cited from Barasz et
al., not proved; the article lists what is checked and what is cited. Everything here is plain
Lean and Mathlib and builds with `lake build ProgramEquilibrium`.

## Inkling experiment

We used checked outcomes to train Inkling to propose policies, then tried development-guided
search in a larger proved policy class. The two training studies used 2,048 proposals and
126 optimizer updates. Neither trained arm beat handwritten FairBot. One unrestricted policy
passed its original population and became exploitable in two transfer populations; the new
certificates correctly rejected it. The delayed search recovered Boolean rewrites of a
supplied control rather than a new strategy.

The [experiment notes](experiments/) describe the settings, exclusions and exact scope.
To replay the three transfer certificates after building:

```bash
lake build check_program_policy
python3 week-07-program-equilibrium/experiments/replay.py --lean example
```

## Bounded self-play prototype

```bash
lake exe play_bounded_agents 15
lake exe play_bounded_agents 16
```

At 15 bytes, self-play returns `(D, D)`. At 16 bytes, it returns `(C, C)`, using the
16-byte certificate `R("self","self")`. Two different search orders also cooperate at
16 bytes. The demo includes a defector, a constant cooperator, and an opaque function
whose cooperation these rules cannot prove.

The budget counts UTF-8 certificate bytes, not compiler time, expanded Lean proof terms or
arbitrary arithmetic proofs. Each search tries at most two candidates. The reciprocal rule
inspects the registered program descriptions and budgets, without running either searcher;
its soundness theorem connects accepted certificates to actual program outputs. Registry
names are references, not source-code hashes.

This is not the general DUPOC implementation requested in
[Critch, Dennis and Russell's Open Problem 4](https://arxiv.org/abs/2208.07006).
The parametric bounded Löb theorem and the arithmetic proof encoding with its length bounds
remain to be formalized and connected to a richer proof search. The prototype uses neither
an assumed reflection axiom nor a modal-world cutoff as its budget.
