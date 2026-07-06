# Experiments

This directory contains two kinds of experiment material:

- Stable paper-release drivers and committed compact result shards.
- Additional reproducibility drivers and compact result shards.

The stable paper numbers reported in the README come from the `run_stable.py`
drivers and the committed `results_stable/` shards. The older `run.py` and
per-directory `sbatch.sh` files are useful for development history and
single-seed checks, but they are not the source of the final stable timing
tables.

## Stable Artifacts

| Output | Driver | Committed compact shards |
| --- | --- | --- |
| Table 3, solver speedup | `paper_posterior_sweep/run_stable.py` | `paper_posterior_sweep/results_stable/table3_solve_time_seed*.csv` |
| Table 4, wall speedup | `paper_posterior_sweep/run_stable.py` | `paper_posterior_sweep/results_stable/table4_wall_time_seed*.csv` |
| Fig. 7 agreement summary | `paper_posterior_sweep/run_stable.py` | `paper_posterior_sweep/results_stable/fig7_agreement_per_n_seed*.csv` |
| Table 5 stress scaling | `paper_stress_scaling/run_stable.py` | `paper_stress_scaling/results_stable/table5_stress_scaling_seed*.csv` |
| Table 7 named fixtures | `named_graphs/run_stable.py` | `named_graphs/results_stable/named_fixtures_stable_seed*.csv` |
| Fig. 8 safe-vs-query-only discrepancy | `pobs_safe_quotient/make_pobs_data.py` | `pobs_safe_quotient/pobs_discrepancy.csv` |

The large per-draw posterior shards for Figures 3, 5, and 6 are not committed.
Regenerate them with `paper_posterior_sweep/run_stable.py` before running
`python -m canonred.figviz`.

## Verification

From the repository root:

```bash
python experiments/validate.py
python experiments/aggregate_stable.py --expect-seeds 20
```

`validate.py` is the release integrity gate. It checks the expected 20-seed
grids, required columns, dimensions/fixtures, ECOS-only solver provenance,
single-host timing per table, and agreement between recomputed summaries and
the committed `STABLE_SUMMARY.csv`.

`aggregate_stable.py` is the human-readable summarizer. It recomputes medians,
IQRs, ranges, and seed counts from the raw shards.

## Local Smoke Regeneration

Use one seed locally to make sure the drivers run in your environment:

```bash
python experiments/paper_stress_scaling/run_stable.py \
  --seed 42 --allow-busy --n-repeats 25 --out-dir experiments/paper_stress_scaling/results_stable

python experiments/paper_posterior_sweep/run_stable.py \
  --seed 42 --allow-busy --S 10000 --burn 5000 --n-repeats 25 \
  --out-dir experiments/paper_posterior_sweep/results_stable

python experiments/named_graphs/run_stable.py \
  --seed 42 --allow-busy --n-repeats 25 --out-dir experiments/named_graphs/results_stable
```

These commands overwrite the seed-42 shards in place. Use a separate `--out-dir`
if you want to keep the committed shards untouched while testing.

## Full Stable Timing Regeneration

The committed timing tables were generated with seeds 42 through 61 under Slurm,
single-threaded, exclusive-node timing, strict ECOS, and `--n-repeats 25`.

Templates:

```bash
sbatch experiments/run_stable_slurm.sh
sbatch experiments/run_named_slurm.sh
```

Before submitting on a new cluster, configure the scripts for your environment:

- Set `PYTHON_BIN` if the active `python` is not the intended environment.
- Set `REPO_ROOT` if submitting from a path outside this checkout.
- Override Slurm output paths or add node constraints at submission time.
- Prefer exclusive-node timing for final table regeneration.

Do not use the legacy per-directory `sbatch.sh` files to regenerate the stable
paper numbers. They are retained as historical single-run launchers for older
additional experiments.

## Additional Experiment Material

The following directories contain additional experiments and earlier development
outputs:

- `random_admg_sweep/`
- `scalability_demo/`
- `lazy_phase1/`
- `prop42_tightness/`
- `balke_pearl_iv/`

Some CSVs in those directories are compact summaries; some scripts may still
need local scheduler options. Treat them as provenance unless a paper section
explicitly points to them.
