"""Shared helpers for experiment runners.

Each experiment is a self-contained directory under `experiments/` with:
  - `run.py`      entry script (writes CSVs in cwd)
  - `sbatch.sh`   slurm launcher
  - `notes.md`    purpose + parameters + known caveats

Slurm convention for the release scripts:
  - submit from an installed checkout, or set PYTHON_BIN to the intended Python
  - CPU-only jobs (no --gres=gpu)
  - use exclusive nodes for stable timing tables when possible
"""

from __future__ import annotations

import csv
import json
import time
from contextlib import contextmanager
from pathlib import Path


def dump_csv(rows: list[dict], path: str | Path) -> None:
    """Write rows (list of dicts) to `path` as CSV. Union of keys as header."""
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def dump_meta(meta: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(meta, f, indent=2, default=str)


@contextmanager
def timed(label: str = ""):
    """Context manager: print elapsed wall-clock on exit."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    if label:
        print(f"[{label}] wall-clock: {dt:.2f}s", flush=True)
    else:
        print(f"[elapsed] {dt:.2f}s", flush=True)


def closed_form_nde_l25(p_w1: float, p_y00: float, p_y01: float) -> tuple[float, float]:
    """Paper §6 eq. closed-form bounds for Q = P(Y_{0, W_1} = 1) under L_{2.5} marginals."""
    L = max(0.0, p_y00 - p_w1) + max(0.0, p_y01 - (1.0 - p_w1))
    U = min(p_y00, 1.0 - p_w1) + min(p_y01, p_w1)
    return L, U


# ======================================================================
# Reproducible timing infrastructure
# ----------------------------------------------------------------------
# NOTE on thread pinning: the env vars below MUST be set *before* numpy is
# imported, so they belong at the very top of each run.py (above `import
# numpy`). This module only *records/asserts* them (it is imported after numpy).
# ======================================================================

THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ACCELERATE_MAXIMUM_THREADS",
)

# Pinned ECOS tolerances/iters for reproducible solver work across runs.
ECOS_OPTS = {"feastol": 1e-8, "abstol": 1e-8, "reltol": 1e-8, "max_iters": 100}


def pin_threads_block() -> str:
    """Return the snippet that must be pasted at the TOP of run.py (pre-numpy)."""
    return (
        "import os as _os\n"
        "for _k in " + repr(THREAD_ENV_KEYS) + ":\n"
        "    _os.environ.setdefault(_k, '1')\n"
        "_os.environ.setdefault('PYTHONHASHSEED', '0')\n"
    )


def preflight_cpu_check(*, max_load_per_core: float = 0.4, abort: bool = True,
                        require_slurm: bool = False, top_n: int = 8) -> dict:
    """Peek at machine load before timing; warn/abort if the CPU is contended.

    Catches the two failure modes the lab cares about: (a) the box is heavily
    used, (b) someone is bypassing SLURM. Uses load average per core, the SLURM
    allocation env, and (if psutil is present) the top CPU consumers.
    """
    import os
    rep: dict = {}
    ncpu = os.cpu_count() or 1
    try:
        la1, la5, la15 = os.getloadavg()
    except (OSError, AttributeError):
        la1 = la5 = la15 = float("nan")
    rep.update(n_cpu=ncpu, loadavg=(la1, la5, la15), load_per_core_1m=la1 / ncpu)
    rep["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    rep["slurm_cpus"] = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    rep["top_procs"] = "psutil-unavailable"
    try:
        import psutil
        psutil.cpu_percent(interval=None)
        time.sleep(0.4)
        procs = [
            (round(p.info.get("cpu_percent") or 0.0, 1), p.info.get("username"),
             p.info.get("name"), p.info.get("pid"))
            for p in psutil.process_iter(["pid", "username", "name", "cpu_percent"])
            if (p.info.get("cpu_percent") or 0.0) > 5.0
        ]
        procs.sort(reverse=True)
        rep["top_procs"] = procs[:top_n]
        rep["total_cpu_percent"] = psutil.cpu_percent(interval=None)
    except Exception:
        pass

    busy = (rep["load_per_core_1m"] == rep["load_per_core_1m"]  # not NaN
            and rep["load_per_core_1m"] > max_load_per_core)
    rep["busy"] = bool(busy)
    print(f"[preflight] host_cpus={ncpu} load(1/5/15)={la1:.2f}/{la5:.2f}/{la15:.2f} "
          f"per-core_1m={rep['load_per_core_1m']:.3f} slurm_job={rep['slurm_job_id']}",
          flush=True)
    if isinstance(rep["top_procs"], list) and rep["top_procs"]:
        print("[preflight] top CPU procs(>5%): "
              + ", ".join(f"{c}%/{u}/{nm}" for c, u, nm, _ in rep["top_procs"]), flush=True)
    if require_slurm and rep["slurm_job_id"] is None:
        raise RuntimeError("[preflight] not inside a SLURM allocation (require_slurm=True). "
                           "Submit via sbatch with --exclusive.")
    if rep["slurm_job_id"] is None:
        print("[preflight] WARNING: SLURM_JOB_ID unset — running outside the scheduler "
              "(shared login node?). Timing may be unreliable.", flush=True)
    if busy:
        msg = (f"[preflight] MACHINE BUSY: per-core 1m load {rep['load_per_core_1m']:.3f} "
               f"> {max_load_per_core}. Timing will be unreliable.")
        if abort:
            raise RuntimeError(msg + " (pass abort=False to override)")
        print(msg + " (continuing; abort=False)", flush=True)
    return rep


def _cpu_detail() -> dict:
    """Detailed CPU info (model, clocks, cache sizes) — timing depends on these.

    platform.processor() returns only 'x86_64' on Linux, so parse lscpu (and
    fall back to /proc/cpuinfo) to record the model name and L1/L2/L3 cache.
    """
    import subprocess
    want = {
        "Model name": "model_name", "Vendor ID": "vendor", "CPU family": "family",
        "CPU(s)": "cpus", "Thread(s) per core": "threads_per_core",
        "Core(s) per socket": "cores_per_socket", "Socket(s)": "sockets",
        "NUMA node(s)": "numa_nodes", "CPU max MHz": "cpu_max_mhz",
        "CPU min MHz": "cpu_min_mhz", "L1d cache": "l1d", "L1i cache": "l1i",
        "L2 cache": "l2", "L3 cache": "l3",
    }
    d: dict = {}
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                key = want.get(k.strip())
                if key:
                    d[key] = v.strip()
    except Exception:
        pass
    if "model_name" not in d:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        d["model_name"] = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    return d


def provenance(extra: dict | None = None) -> dict:
    """Full environment snapshot for meta.json so a timing run is reproducible."""
    import os
    import platform
    prov: dict = {
        "host": platform.node(), "os": platform.platform(), "arch": platform.machine(),
        "cpu": platform.processor(), "n_cpu": os.cpu_count(),
        "python": platform.python_version(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "threads": {k: os.environ.get(k) for k in THREAD_ENV_KEYS},
        "slurm": {k: os.environ.get(k) for k in
                  ("SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_JOB_NODELIST")},
    }
    for mod in ("numpy", "scipy", "cvxpy", "ecos", "scs"):
        try:
            prov[f"{mod}_version"] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            prov[f"{mod}_version"] = None
    try:
        import numpy as np
        cfg = np.show_config(mode="dicts")
        prov["numpy_blas"] = (cfg.get("Build Dependencies", {})
                              .get("blas", {}).get("name", "?"))
    except Exception:
        prov["numpy_blas"] = "?"
    prov["cpu_detail"] = _cpu_detail()
    if extra:
        prov.update(extra)
    return prov


def lp_fingerprint(coef) -> dict:
    """SHA256 (truncated) of the LP coefficient arrays — proves the same problem ran."""
    import hashlib
    import numpy as np

    def h(a) -> str:
        return hashlib.sha256(
            np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:16]

    return {"c_sha256": h(coef.c), "M_sha256": h(coef.M),
            "dim": int(coef.dim), "n_constraints": int(coef.n_constraints)}


def time_solve_stable(cached_lp, p_or_pool, *, n_warmup: int = 5, n_batch: int = 200,
                      n_repeats: int = 15, rotate_pool: bool = True) -> dict:
    """Amortized per-solve timing for repeated solves of the SAME cached LP.

    Times N consecutive solves under one clock and divides by N (amortizing the
    fixed per-call solver setup that dominates microsecond-scale LPs), repeated
    over R batches for a robust median + IQR. `cached_lp.solve(p)` does one
    (min,max) pair, so every reported time is per (min,max) pair.

    Returns medians/IQRs of (per-solve wall seconds, mean ECOS solver_time),
    plus raw per-batch arrays for paired-ratio speedup.
    """
    import numpy as np
    pool = np.asarray(p_or_pool, dtype=np.float64)
    if pool.ndim == 1:
        pool = pool[None, :]
    npool = pool.shape[0]
    solvers_seen: set = set()

    def one(j: int):
        _, _, d = cached_lp.solve(pool[j % npool])
        solvers_seen.add(d.get("solver_used"))
        return d["wall_time"], d["solver_time"]

    for j in range(n_warmup):
        one(j)

    bw = np.empty(n_repeats)
    bs = np.empty(n_repeats)
    for r in range(n_repeats):
        ss = 0.0
        offset = r if rotate_pool else 0
        t0 = time.perf_counter_ns()
        for k in range(n_batch):
            _, s = one(k + offset)
            ss += s
        bw[r] = ((time.perf_counter_ns() - t0) * 1e-9) / n_batch
        bs[r] = ss / n_batch

    def agg(a):
        lo, mid, hi = (float(x) for x in np.quantile(a, [0.25, 0.5, 0.75]))
        return mid, lo, hi

    wm, wlo, whi = agg(bw)
    sm, slo, shi = agg(bs)
    return {
        "wall_med_s": wm, "wall_iqr_lo_s": wlo, "wall_iqr_hi_s": whi,
        "solver_med_s": sm, "solver_iqr_lo_s": slo, "solver_iqr_hi_s": shi,
        "n_warmup": n_warmup, "n_batch": n_batch, "n_repeats": n_repeats,
        "pool_size": npool,
        "solvers_used": sorted(s for s in solvers_seen if s),
        "_raw_solver": bs, "_raw_wall": bw,
    }


def paired_speedup(res_slow: dict, res_fast: dict, key: str = "solver") -> dict:
    """Robust speedup with IQR from paired per-batch ratios (slow/fast)."""
    import numpy as np
    num = np.asarray(res_slow[f"_raw_{key}"])
    den = np.asarray(res_fast[f"_raw_{key}"])
    ratios = num / np.maximum(den, 1e-12)
    lo, mid, hi = (float(x) for x in np.quantile(ratios, [0.25, 0.5, 0.75]))
    return {"speedup": mid, "speedup_iqr_lo": lo, "speedup_iqr_hi": hi}
