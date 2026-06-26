# `docs/` — planning documents for the redesign

Start here before touching code on branch `redesign-from-scratch`.

| File | Purpose |
|---|---|
| `01_concepts.md` | Formal definitions. The ground truth for every object we implement: ADMG, worlds, counterfactual variables (with nested subscripts), canonical SCM state space, queries, observation model, LP formulation, Prop. 3.1 quotient, Prop. 4.2 cardinality. |
| `02_ambiguities.md` | What is unclear in the 1086 paper, Zhang-Tian-Bareinboim 2022, and Yang-Bareinboim 2025 — plus our resolution for each. Pinned to avoid re-debating in the code review. |
| `03_modules.md` | Python package architecture (`canonred/`), per-module API sketches, test strategy, implementation order. |

If any definition in `01` contradicts what the code does, fix the code; if any ambiguity in `02` is under-specified, update the doc first and then the code. `03` is the planning map — update when we learn something during implementation.

Branch: `redesign-from-scratch`. Legacy files on this branch are untouched for now; plan is to get `canonred/` to parity on the NDE L₂.₅ example first, then evaluate pruning.
