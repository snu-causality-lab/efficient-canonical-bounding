# Module architecture — `canonred/`

Fresh package, clean split. Every function in `canonred/` points back to a specific concept in `01_concepts.md` or resolves a point in `02_ambiguities.md`.

---

## 0. Top-level layout

```
efficient_canonical_bounding/
├── docs/
│   ├── 01_concepts.md
│   ├── 02_ambiguities.md
│   └── 03_modules.md            ← this file
├── canonred/                    ← new library (replaces legacy flat scripts)
│   ├── __init__.py
│   ├── types.py
│   ├── graph.py
│   ├── worlds.py
│   ├── counterfactual.py
│   ├── expansion.py
│   ├── canonical.py
│   ├── observation.py
│   ├── context_set.py
│   ├── signature.py
│   ├── lp.py
│   ├── posterior.py
│   ├── examples.py
│   └── viz.py
├── tests/
│   ├── test_graph.py
│   ├── test_worlds.py
│   ├── test_counterfactual.py
│   ├── test_expansion.py
│   ├── test_canonical.py
│   ├── test_context_set.py
│   ├── test_signature.py
│   ├── test_lp.py
│   ├── test_posterior.py
│   └── test_end_to_end_nde.py
├── scripts/
│   ├── reproduce_nde.py
│   ├── cardinality_sanity.py
│   ├── random_admg_sweep.py
│   ├── infeasible_feasible_demo.py
│   └── tightness_examples.py
├── pyproject.toml
├── requirements.txt
└── README.md                     (rewritten for redesign)
```

Legacy files (`allinone_*.py`, `main.py`, `posterior_gibbs_8state.py`, `rebuttal_experiments/`) stay untouched on this branch — we do not delete during scaffolding. After `canonred/` reaches parity on the NDE example, we can evaluate whether to prune the legacy tree.

---

## 1. `canonred/types.py`

```python
from typing import TypeAlias, NewType

VarName: TypeAlias = str
Value: TypeAlias = int        # non-negative, < domain cardinality
Label: TypeAlias = str        # world label
EdgeKey: TypeAlias = tuple[VarName, VarName]  # (parent, child) directed edge
```

**Responsibilities**: type aliases only. No logic.

---

## 2. `canonred/graph.py`

```python
@dataclass(frozen=True)
class ADMG:
    nodes: tuple[VarName, ...]
    cardinality: Mapping[VarName, int]
    directed: frozenset[EdgeKey]       # (p, v) with p, v in nodes
    bidirected: frozenset[frozenset[VarName]]  # set of 2-element sets

    def __post_init__(self): ...       # validate DAG-ness, cardinality ≥ 2

    def parents(self, v: VarName) -> tuple[VarName, ...]: ...
    def children(self, v: VarName) -> tuple[VarName, ...]: ...
    def ancestors(self, S: Iterable[VarName], inclusive: bool = True) -> frozenset[VarName]: ...
    def descendants(self, S, inclusive=True) -> frozenset[VarName]: ...
    def topo_order(self) -> tuple[VarName, ...]: ...

    def c_components(self) -> tuple[frozenset[VarName], ...]: ...
    def c_cliques(self) -> tuple[frozenset[VarName], ...]: ...   # Def A.2
    def c_component_of(self, v: VarName) -> frozenset[VarName]: ...

    def induce(self, S: Iterable[VarName]) -> "ADMG": ...
    def mutilate_incoming(self, X: Iterable[VarName]) -> "ADMG": ...  # G_{\overline X}
    def mutilate_outgoing(self, X) -> "ADMG": ...                     # G_{\underline X}

    def is_dag(self) -> bool: ...
    def has_directed_path(self, src, dst) -> bool: ...
```

**Responsibilities**: immutable ADMG with standard graph operations. No counterfactual semantics here — this is pure graph theory.

**Tests** (`tests/test_graph.py`):

- NDE graph: c-components are `{X,W}` and `{Y}`; c-cliques are identical.
- Non-IV graph: one c-component `{X,Y,Z}`, two c-cliques `{X,Y}` and `{Z,Y}`.
- IV graph: two c-components `{Z}` and `{X,Y}`, same c-cliques.
- Ancestors, descendants, topo order round-trip.
- `mutilate_incoming([X])` on NDE removes edges $X\to Y, X\to W$ and the $X \leftrightarrow W$ bidirected edge (per paper §2 convention).

---

## 3. `canonred/worlds.py`

```python
@dataclass(frozen=True)
class World:
    label: Label                          # symbolic name
    do: frozenset[tuple[VarName, Value]]  # intervention map

    @property
    def intervened(self) -> frozenset[VarName]: ...
    def value_of(self, v: VarName) -> Optional[Value]: ...
    def compatible_with(self, other: "World") -> bool: ...  # same do[X] where both intervene on X

    @staticmethod
    def observational() -> "World": ...                # label="obs", do=frozenset()
    @staticmethod
    def from_dict(label: Label, mapping: Mapping[VarName, Value]) -> "World": ...
```

**Responsibilities**: intervention specification + compatibility checks. No graph-dependence.

---

## 4. `canonred/counterfactual.py`

```python
@dataclass(frozen=True)
class CFVar:
    """Counterfactual variable: base node V with edge-label map σ on An(V)."""
    base: VarName
    edge_labels: frozenset[tuple[EdgeKey, Label]]   # edges in An(V) and their world labels

    # Constructors:
    @staticmethod
    def standard(base: VarName, world: World, G: ADMG) -> "CFVar":
        """V under a single world w applied globally; σ assigns label=w.label to every edge in An(V)."""

    @staticmethod
    def observational(base: VarName, G: ADMG) -> "CFVar": ...

    @staticmethod
    def nested(base: VarName, edge_worlds: Mapping[EdgeKey, World], G: ADMG) -> "CFVar":
        """For NDE: edge_worlds = {(X,Y): w_x, (W,Y): w_x, (X,W): w_xp}."""


@dataclass(frozen=True)
class CFQuery:
    """Joint event Q = P(∧ CFVar_i = v_i*)."""
    events: tuple[tuple[CFVar, Value], ...]   # (cfvar, target_value)
    name: str = "Q"

    @property
    def worlds(self) -> frozenset[World]: ...       # all worlds appearing in any σ
    @property
    def intervened_any(self) -> frozenset[VarName]: ...
    @property
    def labels(self) -> frozenset[Label]: ...       # Λ(Q)
```

**Responsibilities**: encode CFVars with edge-label maps. `CFVar.standard` and `.nested` are the two most common constructors.

**Examples stored in `examples.py`**:

- `nde_query()` → `CFVar.nested(Y, {(X,Y): W_x0, (W,Y): W_x0, (X,W): W_x1})` for $Y_{0, W_1}$.
- `pns_query(x, xp)` → two CFVars at base $Y$ with labels.
- `iv_effect_query()` → `CFVar.standard(Y, W_x1)` for $P(Y_{x=1})$.

---

## 5. `canonred/expansion.py`

```python
@dataclass(frozen=True)
class SplitNode:
    """A copy of an original node in the edge-split graph G*."""
    base: VarName              # original node
    label: Optional[Label]     # world under which this copy lives; None = unsplit
    fixed_value: Optional[Value]  # set if this copy is intervened

    @property
    def is_fixed(self) -> bool: ...


def edge_split_graph(G: ADMG, Q: CFQuery) -> tuple[ADMG, Mapping[SplitNode, ...]]:
    """Return (G*, mapping) where G* is a DAG over SplitNodes.

    Every fixed intervention is realized as a SplitNode with fixed_value set.
    Every non-fixed node V shared across worlds corresponds to ONE SplitNode per
    world that feeds into Q (collapsed by label inheritance; see 01_concepts §3.5).

    Raises ConflictError if Q is not even L_{2.5}-realizable.
    """


def enumerate_contexts(G: ADMG, Q: CFQuery) -> Mapping[VarName, frozenset[tuple[tuple[Value, ...], Label]]]:
    """Return {V: set of distinct labeled parent-value tuples at which f_V is invoked}.

    This is S_V(Q) from 01_concepts §10.1. Cardinality gives m_V(Q).

    Uses edge_split_graph internally: iterate all non-fixed SplitNodes of base V,
    read off their parent-values in G*, dedup.
    """


def recognize_layer(G: ADMG, Q: CFQuery) -> Literal["L1", "L2", "L2.25", "L2.5", "L3"]:
    """Classify Q per PCH / YB25 (only the finest-fit label)."""
```

**Responsibilities**: the bridge from symbolic query to enumerable contexts. Critical correctness module — lots of tests here.

---

## 6. `canonred/canonical.py`

```python
@dataclass(frozen=True)
class ResponseFn:
    """f_V: Omega_pa(V) -> Omega_V, encoded as a tuple over a fixed ordering of pa-value tuples."""
    var: VarName
    arity_domains: tuple[int, ...]   # cardinalities of pa(V) in canonical order
    codomain: int                    # |Omega_V|
    table: tuple[Value, ...]         # length = prod(arity_domains); entry = f_V(pa_tuple)

    def __call__(self, *pa_values: Value) -> Value: ...


@dataclass(frozen=True)
class CanonicalState:
    """u = (f_V) for all V in the ADMG."""
    funcs: Mapping[VarName, ResponseFn]


def enumerate_states(G: ADMG) -> Iterator[CanonicalState]:
    """Iterate over |Omega_full| = prod_V d_V^(prod_p d_p) canonical states.

    Heavy — use only when explicit enumeration is needed. For a 7-node graph at binary
    cardinality this can blow up; guard with a size check.
    """


def state_cardinality(G: ADMG) -> int:
    """Return |Omega_full| without enumerating (for feasibility checks)."""


def eval_cfvar(u: CanonicalState, cfvar: CFVar, G: ADMG) -> Value:
    """Compute u(CFVar) via edge_split_graph + topological propagation."""


def eval_query_indicator(u: CanonicalState, Q: CFQuery, G: ADMG) -> int:
    """Return 1 if all (cfvar_i == target_i), else 0. This is c_u."""
```

**Responsibilities**: response-function objects + the *correct* evaluator that respects edge-labels. This is the concrete realization of `01_concepts §6`.

---

## 7. `canonred/observation.py`

```python
@dataclass(frozen=True)
class MomentConstraint:
    """One linear functional on psi that the data identifies."""
    name: str                          # e.g., "P(W_1)" or "P_obs(X=0,W=0,Y=1)"
    cfvars_and_targets: tuple[tuple[CFVar, Value], ...]  # conjunction of CFVar=v events
    # Semantic: row entry = indicator that all events hold at state u.


def pobs_constraints(G: ADMG) -> tuple[MomentConstraint, ...]:
    """One constraint per observable cell v in prod_V Omega_V."""


def l25_marginal_constraints(targets: Iterable[CFVar]) -> tuple[MomentConstraint, ...]:
    """Each input CFVar contributes one marginal moment.

    For NDE L_{2.5}: targets = [W_1, Y_{0,0}, Y_{0,1}], and we create three
    constraints each encoding E_psi[CFVar=1].
    """
```

**Responsibilities**: build observation matrices from domain-specific regimes.

---

## 8. `canonred/context_set.py`

```python
def m_V(G: ADMG, Q: CFQuery, V: VarName) -> int:
    """Return |S_V(Q)|, i.e. m_V(Q). Uses enumerate_contexts(G, Q)."""


def context_multiplicities(G: ADMG, Q: CFQuery) -> Mapping[VarName, int]: ...


def prop42_cardinality(G: ADMG, Q: CFQuery) -> int:
    """d_U(Q) = prod over c-components C of G.induce(An(Q)) of
       prod over V in C of d_V ** m_V(Q).

    Note: c-components are of G itself restricted to ancestors of Q, NOT the
    post-do subgraph. See ambiguity §A.4.
    """


def pch_cardinality(G: ADMG, layer: Literal["L1", "L2", "L3"]) -> int:
    """Textbook L1/L2/L3 cardinalities (no query specialisation)."""
```

**Responsibilities**: the structural counting rules. Checked against NDE expected values (128, 8, 64) in tests.

---

## 9. `canonred/signature.py`

```python
@dataclass(frozen=True)
class SignatureQuotient:
    """The coarsest safe quotient pi: Omega_full -> Omega_red (Prop 3.1)."""
    classes: tuple[tuple[int, ...], ...]   # for each class, indices of representative states
    objective: np.ndarray                   # tilde c
    constraint_matrix: np.ndarray            # tilde M


def build_signature_quotient(
    G: ADMG, Q: CFQuery, constraints: Iterable[MomentConstraint]
) -> SignatureQuotient:
    """Enumerate Omega_full, compute sigma(u) = (c_u, M_{.,u}), hash-bucket."""
```

**Responsibilities**: Alg. 1 Phase 3 in one function. Returns everything needed to build the reduced LP.

---

## 10. `canonred/lp.py`

```python
@dataclass
class LPCoefficients:
    c: np.ndarray          # objective, length dim
    M: np.ndarray          # constraints, (m, dim)
    # NB: observational normalization sum(psi)=1 is separate; row names in `row_labels`
    row_labels: tuple[str, ...]


def build_lp_lifted(G: ADMG, Q: CFQuery, constraints: Iterable[MomentConstraint]) -> LPCoefficients:
    """Full Omega_full LP."""


def build_lp_reduced(quotient: SignatureQuotient) -> LPCoefficients:
    """Reduced LP via signature quotient."""


def solve_sharp_bounds(
    coef: LPCoefficients,
    p: np.ndarray,
    solver: str = "ECOS",
    backup: str = "SCS",
) -> tuple[float, float, dict]:
    """Return (lb, ub, diagnostics) where diagnostics has wall_time, solver_time, status."""


class CachedSharpLP:
    """Reusable LP with cp.Parameter for sweep across p values (mirrors legacy API)."""
    def __init__(self, coef: LPCoefficients, solver="ECOS", backup="SCS"): ...
    def warmup(self, p): ...
    def solve(self, p) -> tuple[float, float, dict]: ...
```

**Responsibilities**: build LP coefficients + solve. Keep CVXPY usage isolated here.

---

## 11. `canonred/posterior.py`

```python
def gibbs_posterior_phi(
    # Regime I: observe K_A ~ Bin(n_A, p_A), K_B ~ Bin(n_B, p_B), K_C ~ Bin(n_C, p_C)
    # General version: moment-indicator data, not just 3 marginals.
    moment_counts: Sequence[tuple[MomentConstraint, int, int]],   # (constraint, k, n)
    G: ADMG, Q: CFQuery, red_omega: SignatureQuotient,
    alpha: float = 1.0, S: int = 5000, burn: int = 1000, thin: int = 1, seed: int = 0,
) -> np.ndarray:
    """Allocation-augmented Gibbs sampler for phi on reduced simplex.

    Generalises the legacy 8-state Gibbs to arbitrary G/Q/reduced-domain.
    """


def multinomial_posterior_phi(
    counts: Mapping[int, int],      # counts per reduced class
    alpha: float = 1.0, S: int = 5000, seed: int = 0,
) -> np.ndarray: ...


def lift_phi_to_psi(
    phi: np.ndarray, quotient: SignatureQuotient, alpha: float = 1.0, rng=None
) -> np.ndarray:
    """Random Dirichlet lift within each fiber (legacy's lift_phi_to_psi128_random
    generalised)."""


def lift_phi_to_psi_uniform(phi: np.ndarray, quotient: SignatureQuotient) -> np.ndarray: ...
```

**Responsibilities**: posterior sampling. Generalises the current binomial-3-arm 8-state code to arbitrary `(G, Q, constraints)` without hardcoding fiber size 16.

---

## 12. `canonred/examples.py`

```python
# Concrete (graph, query, regime) bundles with hand-verified expected numbers.

def nde_graph() -> ADMG: ...                              # X->W->Y, X->Y, X<->W
def nde_query_l25() -> CFQuery: ...                        # P(Y_{0, W_1}=1)
def nde_constraints_l25() -> tuple[MomentConstraint, ...]: ...
def nde_constraints_pobs() -> tuple[MomentConstraint, ...]: ...

def pns_graph() -> ADMG: ...                               # X->M->Y, X<->Y, M<->Y
def pns_query(x=0, xp=1) -> CFQuery: ...                   # P(Y_x=1, Y_{x'}=0)

def iv_graph() -> ADMG: ...                                # Z->X->Y, X<->Y
def iv_effect_query() -> CFQuery: ...                       # P(Y_{x=1}=1)

def non_iv_graph() -> ADMG: ...                            # Z->X->Y, X<->Y, Z<->Y
def frontdoor_graph() -> ADMG: ...                         # X->W->Y, X<->Y
def bow_graph() -> ADMG: ...                               # X->Y, X<->Y

def ternary_nde_graph() -> ADMG: ...                        # Z->X->M->Y, X<->Y, M ternary
def ternary_nde_query() -> CFQuery: ...                     # P(Y_{1, M_0}=1)


@dataclass(frozen=True)
class Example:
    name: str
    graph: ADMG
    query: CFQuery
    constraints: tuple[MomentConstraint, ...]
    expected: Mapping[str, int]     # {'lifted_dim', 'reduced_dim', 'd_U_prop42', ...}

ALL_EXAMPLES: tuple[Example, ...] = (
    Example("NDE-L2.5", nde_graph(), nde_query_l25(), nde_constraints_l25(),
            expected={'lifted_dim': 128, 'reduced_dim': 8, 'd_U_prop42': 8}),
    Example("NDE-Pobs", nde_graph(), nde_query_l25(), nde_constraints_pobs(),
            expected={'lifted_dim': 128, 'reduced_dim': 64}),
    Example("PNS-query-only", pns_graph(), pns_query(), (),
            expected={'lifted_dim': 128, 'reduced_dim': 64}),
    Example("PNS-Pobs", pns_graph(), pns_query(), pns_pobs_constraints(),
            expected={'lifted_dim': 128, 'reduced_dim': 32}),
    Example("TernaryNDE-query-only", ternary_nde_graph(), ternary_nde_query(), (),
            expected={'lifted_dim': 4608, 'reduced_dim': 24}),
    Example("TernaryNDE-Pobs", ternary_nde_graph(), ternary_nde_query(), ternary_pobs(),
            expected={'lifted_dim': 4608, 'reduced_dim': 576}),
)
```

**Responsibilities**: canonical test fixtures with paper-reported numbers (from `04-cardinality_in_any_layers.tex`, `05-bounding_posterior.tex`, `natural_bound_app.tex`). Tests iterate `ALL_EXAMPLES` and verify expected[*] against the actual code output.

---

## 13. `canonred/viz.py`

```python
def plot_posterior_histograms(
    Q8: np.ndarray, Q128: np.ndarray,
    plugin_bounds, truth_bounds, out_path,
    title: str = "", bins: int = 40,
): ...

def plot_bounds_vs_n(results: Sequence[PerNResult], out_path): ...

def plot_agreement(results, out_path): ...
```

**Responsibilities**: thin wrappers around matplotlib; keep numerical code in other modules.

---

## 14. Scripts (entry points)

### `scripts/reproduce_nde.py`

Reproduces 1086 Fig 1 + Table 1 on the NDE L_{2.5} regime. Sanity check: 128↔8 agreement at solver tolerance.

### `scripts/cardinality_sanity.py`

For each example in `examples.ALL_EXAMPLES`, compute $|\Omega_{\mathrm{full}}|$, $|\Omega_{\mathrm{red}}|$ (via signature quotient), and Prop. 4.2's $d_U$ — compare to `expected`. Fail loudly on mismatch.

### `scripts/random_admg_sweep.py`

Sweeps random semi-Markovian ADMGs at `|V| ∈ {4, 5, 6}` with L_{2.5} queries, measures reduced_dim / lifted_dim / solve_time ratios. (Replaces the old exp01 sweep but on the corrected code.)

### `scripts/infeasible_feasible_demo.py`

L_3 queries at `|V| ∈ {6, 7, 8}`: lifted LP times out / OOMs, reduced LP completes. (Replaces old exp02.)

### `scripts/tightness_examples.py`

Hand-crafted tight/slack Prop. 4.2 examples. (Replaces old exp03.)

---

## 15. Testing strategy

### 15.1 Unit tests per module

Each module has a corresponding `tests/test_*.py`. Graph and worlds are heavily tested at the combinatorial level; counterfactual/expansion need careful tests on small graphs.

### 15.2 Integration tests on named examples

`tests/test_end_to_end_nde.py`:

- Builds NDE graph + L_{2.5} query + marginal constraints.
- Enumerates $\Omega_{\mathrm{full}}$ = 128 states (assert).
- Computes signature quotient, asserts dim = 8.
- Solves lifted and reduced LPs on a grid of $p$ values (uniform samples), asserts bounds agree within $10^{-6}$.
- Computes $m_V(Q)$ and Prop 4.2 bound; asserts $(m_X, m_W, m_Y) = (0, 1, 2)$, Prop. 4.2 = 8.

Equivalent tests for PNS, IV, ternary NDE — all with expected numbers from paper.

### 15.3 Randomised property tests

For random small ADMGs (seeded), check:

- `signature_quotient.reduced_dim <= lifted_dim`.
- Lifted LP bounds ≥ reduced LP bounds in the sense of identical values (not strict inequality) on a feasible `p`.
- Prop. 4.2's $d_U$ ≥ `reduced_dim` (structural bound dominates LP quotient).

---

## 16. Dependency and build

`pyproject.toml`:

```toml
[project]
name = "canonred"
version = "0.0.1"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "cvxpy>=1.5",
  "ecos>=2.0",
  "scs>=3.2",
  "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "mypy", "ruff"]
```

Keep dependencies minimal. No pydantic; dataclasses suffice.

---

## 17. Implementation order

Concrete sequencing for the rewrite:

1. **`types.py` + `graph.py`**: graph primitives. Verify c-components / c-cliques / topo / mutilate / induce on NDE, IV, Non-IV, bow, frontdoor, PNS-graph, ternary-NDE.
2. **`worlds.py`**: trivial but foundational.
3. **`counterfactual.py`**: CFVar (standard, observational, nested). No evaluation yet — pure data.
4. **`expansion.py`**: `edge_split_graph` is the hardest function in the library. Write it carefully with thorough tests:
   - NDE $Y_{0, W_1}$: 5 nodes in G* (X^{(0)}, X^{(1)}, W, W_split copies, Y) — verify parents/labels.
   - PNS $Y_x \wedge Y_{x'}$: X^{(x)}, X^{(x')}, M^{(x)}, M^{(x')}, Y^{(x)}, Y^{(x')} — 6 split nodes.
   - IV $Y_{x=1}$: Z, X^{(1)}, Y — just one split.
   - `enumerate_contexts` on each → verify against hand-computed $m_V$.
5. **`canonical.py`**: enumerate, eval_cfvar, eval_query_indicator. Sanity check on NDE that each state evaluates consistently between reduced 8-state and lifted 128-state.
6. **`observation.py`**: build moment constraints. Test P_obs on NDE produces 8 rows matching 1086 Eq. (6).
7. **`signature.py`**: build quotient. Sanity: NDE L_{2.5} → 8 classes; NDE P_obs → 64 classes.
8. **`lp.py`**: build & solve. Cached version mirrors legacy `CachedSharpLP`.
9. **`context_set.py`**: $m_V$ + Prop. 4.2. Test on all named examples against paper numbers.
10. **`posterior.py`**: generalised Gibbs + Dirichlet.
11. **`examples.py`**: populate; run `scripts/cardinality_sanity.py` — should pass.
12. **Scripts**: reproduce NDE paper numbers; then exp01/02/03 replacements.
13. (Optional) viz + docs.

Between steps, run `pytest`. Keep each commit small.

---

## 18. Non-goals for now

- No plotting of the "realizable NDE" Fig 2b — that's a rendering concern.
- No symbolic bound derivation (Frechet closed-form) — LP only.
- No BP94's IV-specific parametric canonical SCM — go direct via response functions.
- No autobounds / moment polynomial formulation (ZTB22 Eq. 10–11) — LP on linear functionals is enough for 1086's scope.
- No extension to non-categorical / continuous variables.
