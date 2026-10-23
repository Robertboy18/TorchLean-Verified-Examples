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

The step from the chain to the agents' behavior in Peano arithmetic is cited from Barasz et
al., not proved; the article lists what is checked and what is cited. Everything here is plain
Lean and Mathlib and builds with `lake build ProgramEquilibrium`.
