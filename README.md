# canonred: Canonical Domain Reduction for Partial Counterfactual Identification

Reference implementation for the UAI 2026 paper *Canonical Domain Reduction for
Partial Counterfactual Identification*.

Repository: <https://github.com/snu-causality-lab/efficient-canonical-bounding>

`canonred` builds finite canonical-domain linear programs for partial
counterfactual identification and reduces them by the constraint-aware
sigma-signature quotient. The reduced LP merges canonical states that are
indistinguishable to both the query objective and the information constraints,
so it returns the same sharp bounds as the lifted LP while often being much
smaller.

## What Is Included

- `canonred/`: package code for ADMGs, counterfactual queries, canonical state
  enumeration, sigma-signature quotients, LP construction, and LP solving.
- `tests/`: unit and exactness tests. Slow full-enumeration checks are marked
  with `pytest.mark.slow`.
- `experiments/`: paper experiment drivers and reproducibility scripts.
- `experiments/*/results_stable/` and `experiments/pobs_safe_quotient/`: compact CSV
  shards needed to reproduce the reported table numbers.

The large per-draw posterior CSVs used to draw Figures 3, 5, and 6 are not
committed. They can be regenerated with the stable drivers.

## Install

Use Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`requirements.txt` mirrors the runtime dependency floors in `pyproject.toml` for
users who prefer a requirements file. The recommended development install is
still `pip install -e ".[dev]"`.

For the exact package versions recorded in the committed stable timing
metadata, use Python 3.11.x and:

```bash
pip install -e ".[dev]" -r requirements-repro.txt
```

Those pins are intended for reproducing the paper timing tables. For normal
library use, the dependency floors in `pyproject.toml` are less restrictive.

## Quick Check

```bash
python -m pytest
python experiments/validate.py
python experiments/aggregate_stable.py --expect-seeds 20
```

Expected local sanity result:

- `pytest`: fast suite passes with the slow tests deselected.
- `validate.py`: checks the committed stable shard grids, solver provenance,
  and `STABLE_SUMMARY.csv` consistency, then exits nonzero on any failure.
- `aggregate_stable.py`: prints `[validation] OK` and rewrites
  `experiments/paper_posterior_sweep/results_stable/STABLE_SUMMARY.csv`.

To run the heavier exactness checks:

```bash
python -m pytest -m slow
```

## Basic Usage

Graph constructors live in `canonred.examples`. Full paper examples are built in
the experiment drivers, for example
`experiments/paper_posterior_sweep/run.py:build_nde_l25`.

```python
import numpy as np

from canonred.lp import CachedSharpLP, build_lp_lifted, build_lp_reduced
from canonred.signature import build_signature_quotient

# Given a graph G, counterfactual query Q, and constraints cs:
lifted = build_lp_lifted(G, Q, cs)
quotient = build_signature_quotient(G, Q, cs)
reduced = build_lp_reduced(quotient)

print(lifted.dim, reduced.dim)

p = lifted.M @ (np.ones(lifted.dim) / lifted.dim)
lb_full, ub_full, _ = CachedSharpLP(lifted).solve(p)
lb_red, ub_red, _ = CachedSharpLP(reduced).solve(p)

assert abs(lb_full - lb_red) < 1e-7
assert abs(ub_full - ub_red) < 1e-7
```

## Reproducing Reported Tables

The paper table numbers are derived from committed 20-seed stable shards, not
hand-edited summaries:

```bash
python experiments/aggregate_stable.py --expect-seeds 20
```

This reproduces:

| Paper item | Reported quantity |
| --- | --- |
| Table 3, solver time | 8-state vs 128-state speedup: `9.42x [9.38, 9.48]` |
| Table 4, wall time | end-to-end wall-clock speedup: `1.27x` |
| Table 5, stress scaling | `10.1x / 58.6x / 300.6x` |
| Table 7, named graphs | PNS `128 -> 12`: `6.89x`; ternary NDE `4608 -> 48`: `118.6x` |

`experiments/validate.py` is the stricter release gate. It checks the raw shard
schemas, seed grids, expected dimensions/fixtures, ECOS-only solver provenance,
single-host timing per table, and equality with the committed
`STABLE_SUMMARY.csv`.

### Timing Protocol

The reported speedups are steady-state solver timings: ECOS
`solver_stats.solve_time` from cached CVXPY `Parameter` LPs, warm-started,
single-threaded, median over batches and 20 seeds on a fixed node. This isolates
LP solve cost from one-time Python/CVXPY construction overhead. Table 4 reports
the more modest end-to-end wall-clock speedup for the small running example.

## Regenerating Shards

For a local smoke run, use one seed and allow non-SLURM execution:

```bash
python experiments/paper_stress_scaling/run_stable.py \
  --seed 42 --allow-busy --n-repeats 25 --out-dir experiments/paper_stress_scaling/results_stable

python experiments/paper_posterior_sweep/run_stable.py \
  --seed 42 --allow-busy --S 10000 --burn 5000 --n-repeats 25 \
  --out-dir experiments/paper_posterior_sweep/results_stable

python experiments/named_graphs/run_stable.py \
  --seed 42 --allow-busy --n-repeats 25 --out-dir experiments/named_graphs/results_stable

python experiments/pobs_safe_quotient/make_pobs_data.py
```

The paper timing tables used seeds 42 through 61 as Slurm arrays. The scripts
`experiments/run_stable_slurm.sh` and `experiments/run_named_slurm.sh` are
templates for that protocol. They use the checkout containing the script and the
current `python` by default; set `REPO_ROOT` or `PYTHON_BIN` when submitting from
a different layout or environment. Add any cluster-specific scheduler directives
at submission time. The historical per-directory `sbatch.sh` files are retained only as
historical single-run launchers and are not the source of the stable table numbers.

## Regenerating Figures

Figures 3, 5, and 6 require per-draw posterior CSVs. Those files are intentionally
excluded from the repository to keep the release small. Generate them first with
`experiments/paper_posterior_sweep/run_stable.py`, then run:

```bash
python -m canonred.figviz \
  --in-dir experiments/paper_posterior_sweep/results_stable \
  --out-dir figures
```

Figure 8 data are regenerated by:

```bash
python experiments/pobs_safe_quotient/make_pobs_data.py
```

Visualization style may be adjusted without changing the reported numerical
results.

## Repository Layout

```text
canonred/            package source
docs/                implementation notes and design notes
experiments/         reproducibility drivers, compact result shards, validators
scripts/             small standalone examples
tests/               pytest suite
```

## Citation

```bibtex
@inproceedings{canonred2026,
  title     = {Canonical Domain Reduction for Partial Counterfactual Identification},
  booktitle = {Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2026}
}
```
