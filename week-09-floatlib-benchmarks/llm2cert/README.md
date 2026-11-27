# LLM2Cert's proof obligations with `interval` alone

[LLM2Cert](https://github.com/FormalCert-AI/llm2cert) ([paper](https://arxiv.org/abs/2610.30524)) synthesizes
Lyapunov functions and barrier certificates with a language model and sends their proof obligations to SMT solvers
or to Lean. In its All-Lean setting a model also writes the Lean proofs. Its benchmark DynaCertBench has 291 Lean
obligations for 81 certificates, and each one says that a set of real constraints has no point. That is the kind of
goal FloatLib's `interval` tactic refutes, so I gave it every obligation as the one-line proof `by interval`. No
model is involved anywhere.

## Result

![Three panels of three horizontal bars each, two blue bars for interval with Lean's default limits and with raised limits and one grey bar for LLM2Cert. In the first two panels the three bars are of similar length, each ending in its count; in the third the two blue bars are a small fraction of the grey bar, and each bar ends in its time.](figures/01-llm2cert-obligations.svg)

**Figure 1.** The 291 proof obligations of LLM2Cert's benchmark DynaCertBench, each given to FloatLib's interval tactic as the one-line proof `by interval` with no model, next to LLM2Cert's own All-Lean result, in which a language model writes the proofs. Panel (a) counts the obligations that Lean accepts with the three standard axioms only, panel (b) the 81 certificates all of whose obligations are closed, and panel (c) is the time Lean ran, in hours on a linear axis: for interval the sum over all 291 files, for LLM2Cert the time on the obligations it closed, to which its 1.94 million model tokens are added. interval was run on FloatLib on 2026-11-20 with Lean 4.36.0 on an Intel Xeon Platinum 8275CL (96 threads, shared with other jobs); LLM2Cert's numbers are those of its published records.

| DynaCertBench, 291 obligations | closed | certificates (of 81) | model tokens | time Lean ran |
| --- | ---: | ---: | ---: | --- |
| LLM2Cert All-Lean, from its published records | 274 | 64 | 1,942,231 | 13.6 h on the obligations it closed |
| `interval` alone, Lean's default limits | 281 | 71 | 0 | 43.8 min over all 291 files, median 3.8 s per closed obligation |
| `interval` alone, raised limits | 282 | 72 | 0 | 49.5 min over all 291 files, median 3.9 s per closed obligation |

A certificate counts when every one of its obligations is closed. By kind of obligation:

| kind of obligation | count | LLM2Cert | `interval`, default limits | `interval`, raised limits |
| --- | ---: | ---: | ---: | ---: |
| `unsafe_separation` | 52 | 50 | 51 | 51 |
| `initial_containment` | 45 | 45 | 45 | 45 |
| `lie_strict` | 42 | 35 | 39 | 40 |
| `equilibrium_fixed` | 29 | 29 | 29 | 29 |
| `v_at_equilibrium` | 29 | 29 | 29 | 29 |
| `positive_definite` | 29 | 29 | 29 | 29 |
| `lie_negative` | 25 | 19 | 24 | 24 |
| `control_admissible` | 8 | 8 | 8 | 8 |
| `discrete_inductive` | 8 | 8 | 8 | 8 |
| `sublevel_in_neighborhood` | 7 | 7 | 7 | 7 |
| `radial_unbounded` | 3 | 2 | 0 | 0 |
| `quadratic_lower` | 3 | 3 | 3 | 3 |
| `quadratic_upper` | 3 | 3 | 3 | 3 |
| `exponential_decay` | 3 | 3 | 1 | 1 |
| `certified_set_in_X` | 3 | 3 | 3 | 3 |
| `lie_nonpositive` | 1 | 0 | 1 | 1 |
| `sublevel_contains_initial` | 1 | 1 | 1 | 1 |
| all | 291 | 274 | 281 | 282 |

- **Where the two differ.** With the default limits 266 obligations are closed by both, 15 only by `interval` and 8
  only by LLM2Cert. With the raised limits it is 267, 15 and 7. The 15 are Lie-derivative conditions (`lie_negative`,
  `lie_strict`, `lie_nonpositive`) and two `unsafe_separation`. Two obligations are closed by neither.
- **What `interval` leaves open (9 with both limits).** `DCB-EXT-P1-001|05-radial_unbounded`, `DCB-LIT-P1-003|04-lie_negative`, `DCB-LIT-P1-021|05-radial_unbounded`, `DCB-LIT-P3-002|00-unsafe_separation`, `DCB-LIT-P4-002|03-lie_strict`, `DCB-OURS-P1-001|06-exponential_decay`, `DCB-OURS-P1-002|06-exponential_decay`, `DCB-OURS-P2-001|00-radial_unbounded`, `DCB-OURS-P4-004|03-lie_strict`. LLM2Cert
  closed all of them but the first two. The three `radial_unbounded` statements say that a function is radially
  unbounded, which is no set of constraints, so `interval` has nothing to refute there. One more,
  `DCB-OURS-P4-003|03-lie_strict`, closes only with the raised limits (190 s).
- **Time.** The quickest files take 2.5 s, nearly all of it the import. The 90th percentile is
  7.9 s and 9.0 s per closed obligation. Three closed obligations take more than
  a minute with the default limits (65 s, 203 s, 489 s) and four with the raised ones. No file ran into the 600
  seconds. With a limit of 300 seconds the counts would be 280 and
  70 with the default limits, 281 and
  71 with the raised ones.

Please read the time column with care. LLM2Cert's 13.6 hours are its own Lean wall time on its own machines, on
the obligations it closed, and its pipeline does more than close obligations: it finds the certificates in the
first place. `interval` only discharges what LLM2Cert already synthesized. What the comparison shows is that these
obligations need no model once the certificate is there.

## What was run

| | |
| --- | --- |
| FloatLib | `main` at [`fc2fad1f`](https://github.com/lean-dojo/FloatLib/commit/fc2fad1f9c7de970f0a10a44c58cdf2909a867b4), Lean 4.36.0, Mathlib v4.36.0 |
| LLM2Cert | <https://github.com/FormalCert-AI/llm2cert> at `37b6a0484fc8b532bb29588b56f5993193c42fcc`, MIT. Its numbers are a script count over `results/rq1/data/lean_records.json`: 274 of 291 closed, 64 of 81 certificates, 13.61 hours of Lean wall time on the obligations it closed, 1,942,231 tokens over all of them |
| Statements | [`inputs/obligations.json`](inputs/obligations.json): the 291 locked Lean statements of the RQ1 All-Lean records, one per obligation, with LLM2Cert's license next to them |
| Date and machine | 2026-11-20, an Intel Xeon Platinum 8275CL (2 sockets, 96 hardware threads, 1.1 TB), Amazon Linux 2023, shared with other jobs. The run with Lean's limits went from 2026-11-20T20:51:58Z to 2026-11-20T21:02:33Z (48.58 % idle at the start) and the run with raised limits from 2026-11-20T21:02:38Z to 2026-11-20T21:13:31Z. The largest Lean process took 7.5 GB |

**Method.** [`scripts/llm2cert.py`](scripts/llm2cert.py) writes one Lean file per obligation: LLM2Cert's
definitions and statement verbatim, and the proof `by interval`. `gen --defaults` leaves Lean's own
`maxHeartbeats` (200,000) and `maxRecDepth`, and plain `gen` raises them to 4,000,000 heartbeats and a recursion
depth of 100,000. Then `run` checks every file with `lean`, six processes at a time and 600 seconds per obligation,
and `table` compares with LLM2Cert's records. An obligation counts as closed only when Lean accepts the file without
an error and `#print axioms` shows exactly `propext`, `Classical.choice` and `Quot.sound`.

**From Lean:** whether each file is accepted, and its axioms. **From scripts:** the generated files, the wall time of
each Lean process, the counts, the sums and the comparison with LLM2Cert's records.

## Files

- [`results/llm2cert.csv`](results/llm2cert.csv), [`results/llm2cert-kinds.csv`](results/llm2cert-kinds.csv) and
  [`results/llm2cert-run.csv`](results/llm2cert-run.csv): the numbers above.
  [`results/obligations-default-limits.tsv`](results/obligations-default-limits.tsv) and
  [`results/obligations-raised-limits.tsv`](results/obligations-raised-limits.tsv) have one row per obligation.
- [`figures/`](figures/CAPTIONS.md): the figure as PNG and SVG, with its caption and notes.
- [`scripts/`](scripts/llm2cert.py): `fetch.sh` (LLM2Cert's repository and records), `llm2cert.py`, `run.sh`,
  `collect.py`, `plot.py`.

## Run it again

```bash
lake build                                       # in the week's directory: FloatLib
bash llm2cert/scripts/fetch.sh                   # LLM2Cert's records, pinned
bash llm2cert/scripts/run.sh defaults raised     # about 45 minutes of Lean each
python3 llm2cert/scripts/collect.py llm2cert/run/defaults llm2cert/run/raised \
    llm2cert/run/dl/llm2cert/results/rq1/data/lean_records.json
python3 llm2cert/scripts/plot.py
```
