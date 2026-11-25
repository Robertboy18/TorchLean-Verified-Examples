# Third-party inputs of week 9

Nothing in this directory is a copy of a benchmark, a network or another tool, with one exception that its license
allows: the 291 statements of LLM2Cert (MIT) in `llm2cert/inputs/`. Everything else is fetched by a
`scripts/fetch.sh` at a pinned commit or release, or comes with FloatLib's own checkout. What the directory ships
of other projects is their published numbers, with the source named, and our own results on their inputs.

Our own code and text here are under the repository's [MIT license](../LICENSE). `plots/figstyle.py` is FloatLib's
figure style (MIT). `numfuzz/scripts/ProofsNF.lean` extends FloatLib's `benchmarks/suites/Proofs.lean` (MIT).

| input | used in | version | license | how it is obtained | in this repository |
| --- | --- | --- | --- | --- | --- |
| [FloatLib](https://github.com/lean-dojo/FloatLib) | every suite | 4.1.0, tag `v4.1.0` | MIT | `lake update` in this directory (`lakefile.lean`) | no (`plots/figstyle.py` and the runner in `numfuzz/scripts/ProofsNF.lean` are copies of two of its files) |
| [Lean](https://github.com/leanprover/lean4) and [Mathlib](https://github.com/leanprover-community/mathlib4) | every suite | Lean v4.36.0, Mathlib v4.36.0 | Apache-2.0 | `lean-toolchain`, and FloatLib's manifest | no |
| [LLM2Cert](https://github.com/FormalCert-AI/llm2cert) and its benchmark DynaCertBench | `llm2cert/` | commit `37b6a0484fc8b532bb29588b56f5993193c42fcc` | MIT (c) 2026 The LLM2Cert Authors | `llm2cert/scripts/fetch.sh` | **yes**: `llm2cert/inputs/obligations.json`, the 291 locked Lean statements of its RQ1 All-Lean records, with its license in `llm2cert/inputs/LICENSE-LLM2Cert`. Its result numbers are quoted from its records |
| [SMT-LIB](https://smt-lib.org/) release 2025, logics QF_FP and QF_NRA | `smtlib-qf-fp/`, `smtlib-qf-nra/` | release 2025.08.04, [Zenodo record 16740866](https://zenodo.org/records/16740866) | CC BY 4.0 | `scripts/fetch.sh` of each suite (`QF_FP.tar.zst`, `QF_NRA.tar.zst`, checked by MD5) | no. The license allows it with attribution, and the benchmarks are 3.3 GB. Benchmark names appear in the result files |
| [SMT-COMP](https://smt-comp.github.io/) results of 2025 and 2018 | `smtlib-qf-fp/`, `smtlib-qf-nra/` | `data/` of [SMT-COMP/smt-comp.github.io](https://github.com/SMT-COMP/smt-comp.github.io) at `cd15cbedd4ef312d615448100797a80e922cf575` | not determined (the result files carry no license statement), so nothing is copied | `scripts/fetch.sh` of each suite | no. The competition's answer per benchmark (unsat, sat or none) is quoted in `smtlib-qf-nra/results/per-instance.csv`, and totals per solver in the other result files, with the source named |
| [Z3](https://github.com/Z3Prover/z3) | `smtlib-qf-fp/` (5.4.1), `smtlib-qf-nra/` (5.3.3) | the binaries of the PyPI wheels `z3-solver` 5.4.1.0 and 5.3.3.0 | MIT | `scripts/fetch.sh` | no |
| [cvc5](https://github.com/cvc5/cvc5) | `smtlib-qf-fp/`, `smtlib-qf-nra/` | 1.4.2, the static release binary | BSD-3-Clause | `scripts/fetch.sh` | no |
| [Yices 2](https://github.com/SRI-CSL/yices2) | `smtlib-qf-nra/` | 2.8.0, the static release binary | GPL-3.0 | `scripts/fetch.sh` | no |
| [VNN-COMP 2025 benchmarks](https://github.com/VNN-COMP/vnncomp2025_benchmarks) | `vnncomp-replay/` | commit `8b7b811b`, 14 families | varies by family, and not checked family by family, so nothing is copied | `vnncomp-replay/scripts/fetch.sh` (sparse checkout) | no. No network, property or counterexample is copied |
| [VNN-COMP 2025 results](https://github.com/VNN-COMP/vnncomp2025_results) | `vnncomp-replay/` | commit `ea89fbc2` | not determined, so nothing is copied | `vnncomp-replay/scripts/fetch.sh` | no. Only our own verdicts, added up by benchmark family |
| [onnxruntime](https://github.com/microsoft/onnxruntime), [numpy](https://numpy.org/), [onnx](https://github.com/onnx/onnx) | `vnncomp-replay/` (cross-checks) | 1.32.0, 2.5.0, 1.22.0 | MIT, BSD-3-Clause, Apache-2.0 | `pip` | no |
| [NumFuzz](https://github.com/ak-2485/NumFuzz) artifact | `numfuzz/` | `main` at `80c54f35348c60b8a1e95f1881adda8d2e9c1dfb` | GPL-3.0 | `numfuzz/scripts/fetch.sh` | no. Its programs are read as data, and none of them, and none of its code, is copied. `numfuzz/scripts/gen.py` holds the numbers of Tables 3, 4 and 5 of the paper (arXiv:2405.04612v2), as published |
| [LeanCert](https://github.com/alerad/leancert) | `leancert/` | commit `44b323a0d6af00211d5013f6e021432db032087c` | Apache-2.0 | `leancert/scripts/fetch.sh`, which also applies the port | a patch only: `leancert/scripts/leancert-lean-4.36.diff` (our port to Lean v4.36.0, with the few context lines of LeanCert that a patch needs). Statements of its benchmarks and tests appear by name in the result files |
| [FPBench](https://github.com/FPBench/FPBench) | `fpbench/` | commit `8ac4ff6cf0c3be4f9669a562ef2bbbbc3b15182d` | MIT | FloatLib's suite driver (`benchmarks/suites/fetch.py`, pinned with checksums) | no |
| [MetiTarski](https://bitbucket.org/lcpaulson/metitarski-git/) problem set | `metitarski-flyspeck/` | commit `5a63f3ecb8c8edf721f0051431db3e0f689d5c17` | MIT | FloatLib's suite driver | no |
| [Flyspeck](https://github.com/flyspeck/flyspeck) | `metitarski-flyspeck/` | commit `32948714cdf06d700af2c993c9a92b7b551d7af5` | MIT | FloatLib's suite driver | no |
| [GNU MPFR](https://www.mpfr.org/) | `compiled-vs-mpfr-softfloat-fpu/` | 4.1.0, patch level 13, the system's library | LGPL-3.0-or-later | the system's package | no |
| [Berkeley SoftFloat](https://github.com/ucb-bar/berkeley-softfloat-3) | `compiled-vs-mpfr-softfloat-fpu/` | 3e at `a0c6494cdc11865811dec815d5c0049fba9d82a8` | BSD-3-Clause | FloatLib's comparison harness clones it at that revision | no |
| [Coq](https://coq.inria.fr/) and [Flocq](https://flocq.gitlabpages.inria.fr/) | `kernel-vs-coq/` | Coq 8.20.1, Flocq 4.2.2 | LGPL-2.1, LGPL-3.0-or-later | a container, by FloatLib's `benchmarks/suite/run.sh matched` | no. The recorded Coq times are in `kernel-vs-coq/results/kernel-coq.csv` |
