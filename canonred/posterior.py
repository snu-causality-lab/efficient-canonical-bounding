"""Posterior sampling on phi (reduced simplex) and lifts to psi (Omega_full).

See docs/01_concepts.md §5.3 and paper App. `app:posterior` / Fig 1.

Scope:
  - `dirichlet_posterior(counts, alpha, S, seed)`: closed-form Dirichlet
    posterior for Regime II (direct multinomial query-state observations).
  - `binomial_marginal_gibbs(quotient, observations, ...)`: allocation-
    augmented Gibbs for Regime I (binomial-per-binary-marginal observations).
  - `lift_phi_uniform(phi, quotient)`: psi[i] = phi[class(i)] / class_size(class(i)).
  - `lift_phi_dirichlet(phi, quotient, alpha, rng)`: same aggregation, but
    within-fiber mass distributed by Dirichlet(alpha / |fiber|).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from canonred.signature import SignatureQuotient


# ----------------------------------------------------------------------
# Closed-form Dirichlet posterior (Regime II)
# ----------------------------------------------------------------------


def dirichlet_posterior(
    counts: np.ndarray,
    *,
    alpha: float = 1.0,
    S: int = 5000,
    seed: int | None = None,
) -> np.ndarray:
    """Sample S draws from Dir(alpha + counts).

    Parameters
    ----------
    counts : np.ndarray of non-negative ints, shape (reduced_dim,)
        Observed counts per reduced-class index.
    alpha : float
        Symmetric prior concentration.
    S : int
        Number of posterior samples.
    seed : int or None
        RNG seed (for determinism).

    Returns
    -------
    np.ndarray of shape (S, reduced_dim)
        One row per draw; each row is on the simplex.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 1:
        raise ValueError(f"counts must be 1-D; got shape {counts.shape}")
    if (counts < 0).any():
        raise ValueError("counts must be non-negative")
    rng = np.random.default_rng(seed)
    post_conc = alpha + counts.astype(np.float64)
    return rng.dirichlet(post_conc, size=S)


# ----------------------------------------------------------------------
# Lifts: phi (reduced) -> psi (Omega_full)
# ----------------------------------------------------------------------


def _check_phi_shape(phi: np.ndarray, q: SignatureQuotient) -> None:
    if phi.shape != (q.reduced_dim,):
        raise ValueError(
            f"phi has shape {phi.shape}; expected ({q.reduced_dim},) matching quotient.reduced_dim"
        )


def lift_phi_uniform(phi: np.ndarray, quotient: SignatureQuotient) -> np.ndarray:
    """Deterministic uniform-within-fiber lift.

        psi[i] = phi[t] / class_sizes[t]       where t = state_class_assignment[i]

    Guarantees `Pi @ psi == phi` exactly (aggregation preserved).
    """
    _check_phi_shape(phi, quotient)
    assignment = np.asarray(quotient.state_class_assignment, dtype=np.int64)
    sizes = np.asarray(quotient.class_sizes, dtype=np.float64)
    # per-state mass = phi[class(i)] / size[class(i)]
    psi = phi[assignment] / sizes[assignment]
    return psi


def lift_phi_dirichlet(
    phi: np.ndarray,
    quotient: SignatureQuotient,
    *,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random lift: within each fiber, draw Dirichlet(alpha / |fiber|).

    For fiber F_t (states in class t):

        frac ~ Dirichlet(alpha / |F_t|, ..., alpha / |F_t|)   length |F_t|
        psi[i] = phi[t] * frac[idx of i within F_t]

    This matches the prior consistency used in the earlier NDE code: psi drawn
    from `Dir(alpha_0, ..., alpha_0)` with `alpha_0 = alpha / fiber` aggregates
    to phi ~ `Dir(alpha, ..., alpha)` (sum of Dirichlets).
    """
    _check_phi_shape(phi, quotient)
    if rng is None:
        rng = np.random.default_rng()
    assignment = np.asarray(quotient.state_class_assignment, dtype=np.int64)
    # Group state indices by class.
    classes: list[list[int]] = [[] for _ in range(quotient.reduced_dim)]
    for i, t in enumerate(assignment):
        classes[int(t)].append(i)

    psi = np.zeros(quotient.lifted_dim, dtype=np.float64)
    for t, members in enumerate(classes):
        k = len(members)
        if k == 0:
            continue
        if phi[t] <= 0.0:
            continue
        conc = np.full(k, alpha / k)
        frac = rng.dirichlet(conc)
        for j, i in enumerate(members):
            psi[i] = phi[t] * frac[j]
    return psi


# ----------------------------------------------------------------------
# Regime I: allocation-augmented Gibbs for binary-indicator marginals
# ----------------------------------------------------------------------


def binomial_marginal_gibbs(
    quotient: SignatureQuotient,
    *,
    observations: Sequence[tuple[int, int]],
    alpha: float = 1.0,
    S: int = 5000,
    burn: int = 1000,
    thin: int = 1,
    seed: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Allocation-augmented Gibbs for phi on the reduced simplex under
    binary-indicator binomial marginals.

    Parameters
    ----------
    quotient : SignatureQuotient
        Its `constraint_matrix` rows must be 0/1 (binary indicators).
    observations : sequence of (K_i, n_i)
        One pair per constraint row: K_i successes out of n_i Bernoulli trials,
        where each success is drawn from a distribution that marginalises to
        `sum_{t : M_tilde[i, t]=1} phi[t]`.
    alpha : float
        Symmetric Dirichlet prior on phi.
    S, burn, thin, seed : standard MCMC knobs.
    eps : numerical floor for mass guards.

    Returns
    -------
    np.ndarray of shape (S, reduced_dim), one posterior draw per row.

    Notes
    -----
    This generalises the earlier 8-state NDE sampler (`posterior_gibbs_8state`)
    to any (quotient, binary-indicator constraints). Each Gibbs iteration:

      1. For each constraint i, split its K_i successes across classes
         where M_tilde[i, :] = 1 (probability proportional to current phi_t),
         and its (n_i - K_i) failures across classes where M_tilde[i, :] = 0.
      2. Accumulate allocations into pseudocounts N_t.
      3. Draw phi ~ Dir(alpha + N).
    """
    M = np.asarray(quotient.constraint_matrix, dtype=np.int64)
    if M.shape[0] != len(observations):
        raise ValueError(
            f"observations length {len(observations)} != constraint rows {M.shape[0]}"
        )
    if not ((M == 0) | (M == 1)).all():
        raise ValueError("binomial_marginal_gibbs requires binary (0/1) constraint rows")

    reduced_dim = quotient.reduced_dim
    n_constraints = M.shape[0]
    success_idx = [np.where(M[i] == 1)[0] for i in range(n_constraints)]
    failure_idx = [np.where(M[i] == 0)[0] for i in range(n_constraints)]

    obs = [(int(k), int(n)) for k, n in observations]
    for i, (k, n) in enumerate(obs):
        if not (0 <= k <= n):
            raise ValueError(f"observation {i} invalid: K={k}, n={n}")

    rng = np.random.default_rng(seed)
    phi = rng.dirichlet(np.full(reduced_dim, alpha))

    out = np.empty((S, reduced_dim), dtype=np.float64)
    store = 0
    for it in range(S * thin + burn):
        pseudo = np.zeros(reduced_dim, dtype=np.int64)
        for i in range(n_constraints):
            k, n = obs[i]
            if k > 0:
                w = phi[success_idx[i]]
                probs = w / max(w.sum(), eps)
                alloc = rng.multinomial(k, probs)
                pseudo[success_idx[i]] += alloc
            fails = n - k
            if fails > 0:
                w = phi[failure_idx[i]]
                probs = w / max(w.sum(), eps)
                alloc = rng.multinomial(fails, probs)
                pseudo[failure_idx[i]] += alloc
        phi = rng.dirichlet(alpha + pseudo.astype(np.float64))

        if it >= burn and ((it - burn) % thin == 0):
            out[store] = phi
            store += 1
            if store == S:
                break

    return out[:store]


__all__ = [
    "binomial_marginal_gibbs",
    "dirichlet_posterior",
    "lift_phi_dirichlet",
    "lift_phi_uniform",
]
