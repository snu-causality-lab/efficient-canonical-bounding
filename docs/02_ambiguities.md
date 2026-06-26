# Ambiguity notes — 1086 paper + supporting references

Collected while pinning down the formalism in `01_concepts.md`. Each item: (a) what's unclear, (b) why, (c) how we resolve it in code. Author-confirmed answers on the five main items are at the top.

---

## Summary of author confirmations (Yesong, 2026-04-25)

| Item | Status | Disposition |
|---|---|---|
| A (PNS lifted dim) | ✅ resolved | PNS TikZ was missing an `X → Y` edge (copy-paste miss from NDE). With the edge in, lifted dim = 128 matches the paper. `pns_graph()` includes the edge. |
| B (Case 1 "64") | ✅ resolved | The 64 is an illustrative **structural** safe reduction via a specific π_obs, not claimed to be coarsest. Camera-ready will reword §6 Case 1 to say so explicitly. Code exposes both: `signature_quotient` → 16 (coarsest), and the paper's π_obs construction can be a separate helper. |
| C (Pa vs pa) | ✅ resolved | `PA_V` means **strict directed parents** (no self). §2's `Pa(V) = pa(V) ∪ {V}` is a leftover that was not meant to stay; camera-ready fix. Code already uses strict parents exclusively. |
| D (Prop 4.2 `C(U)`) | ✅ resolved | `C(U)` is the c-component of the **original** ADMG, restricted to `An(Q)`. The "delete incoming edges into intervened nodes" step only fixes which f_V get invoked (i.e., the ancestor subgraph construction and `m_V(Q)` enumeration) — it does **not** change the c-component partition. `context_set.prop42_cardinality` already implements this. |
| E (conflict-free / realizability) | ✅ resolved | All paper examples are $\mathcal{L}_{2.5}$. Def. 4.1's "conflict-free" hook was dropped by accident in the final edit pass; camera-ready will reinstate the definition. For this code: keep current label-consistency check; L_{2.25}/L_{2.5} distinction is not required for any of the paper's running examples. |

---

## A. 1086 paper itself

### A.1 `Pa(V)` vs. `pa(V)` — ✅ resolved 2026-04-25

§2 defines $\mathrm{Pa}(V) = \mathrm{pa}(V) \cup \{V\}$ — upper-case *includes* the node itself, lower-case doesn't. But §4.1 and Prop. 4.2 use $|\Omega_V|^{|\Omega_{\mathrm{PA}_V}|}$ as the response-function cardinality, which should be $d_V^{|\Omega_{\mathrm{pa}(V)}|}$ (parents only, *not* including $V$). Same in Table `tab:constraint_aware_summary`.

**Author (2026-04-25)**: `PA_V` is strict directed parents. The inclusive §2 definition is a leftover from an earlier draft and should be deleted in the camera-ready.

**Code**: already uses strict parents via `graph.parents(V)` exclusively.

### A.2 "Conflict-free" (Def. 4.1) — ✅ resolved 2026-04-25

Def. 4.1 requires the query to be "realizable" and "conflict-free" but neither term is defined in the main text. Def. A.6 (preliminaries appendix, from YB25) gives the formal conditions (i)–(ii) for $\mathcal{L}_{2.25}$ and $\mathcal{L}_{2.5}$ but the main-text reader without appendix access is stuck.

**Author (2026-04-25)**: all paper examples are $\mathcal{L}_{2.5}$. A main-text definition was in an earlier draft but got dropped in the final edit pass; camera-ready will restore it.

**Code**: only the $\mathcal{L}_{2.5}$ layer is exercised. Current check is label-consistency (no two `World` objects with same label but different `do`-maps). A richer L_{2.25}/L_{2.5}-distinguishing checker can be added later but is not required for any paper example.

### A.3 $\Lambda(\mathcal{Q})$ and $\lambda_\mathcal{Q}(p)$ construction

Def. 4.1 introduces these as notation but gives no algorithm. We infer:

- $\Lambda(\mathcal{Q})$ = the finite set of world labels appearing anywhere in $\sigma$ of any CFVar in $\mathcal{Q}$. E.g., for $\mathcal{Q} = P(Y_{0, W_1})$, $\Lambda = \{\text{"0"}, \text{"1"}\}$; for $\mathcal{Q} = P(Y_x, Y_{x'})$ PNS, $\Lambda = \{x, x'\}$.
- $\lambda_\mathcal{Q}(p)$ = the world label attached to node $p$ on the path from the CFVar root down to $p$. **Not** a well-defined single value for a node that appears at multiple tree positions — it's a set/multi-valued function. Code: when we enumerate $S_V(\mathcal{Q})$, each tuple entry is a pair `(parent_value, edge_label)` from the edge-split graph.

### A.4 Prop. 4.2: which $\mathbf{C}(U)$? — ✅ resolved 2026-04-25

The proposition says "for each canonical exogenous $U$ indexing a (maximal) c-component in this subgraph, let $\mathbf{C}(U)$ denote the corresponding c-component." The "this subgraph" refers to the ancestor subgraph *after deleting incoming edges into intervened nodes*. But the NDE example immediately after uses the *original* graph's c-components ($\{X,W\}$ and $\{Y\}$) to multiply $d_{U_1} \cdot d_{U_2}$.

**Author (2026-04-25)**: (a) — $\mathbf{C}(U)$ uses the **original** ADMG's bidirected c-components, restricted to $\mathrm{An}(\mathcal{Q})$. The "delete incoming edges into intervened nodes" step is purely to identify which $f_V$ are invoked (i.e., feeds the $m_V(Q)$ enumeration), not to re-partition c-components. Camera-ready will clarify the Prop. 4.2 wording.

**Code**: `context_set.prop42_cardinality(G, Q)` iterates c-components of `G.induce(An(Q))` — already correct.

### A.5 Case 1 "64" wording — ✅ resolved 2026-04-25

§6 Case 1 defines a specific $\pi_{\mathrm{obs}}$ keeping 6 of 7 canonical bits and reports $|\Omega_{\mathrm{red}}| = 64$. The coarsest-safe signature quotient (Prop. 3.1) on NDE + full P_obs is actually 16 (2 query-indicator values × 8 observation cells).

**Author (2026-04-25)**: the paper's 64 is an illustrative **structural** safe reduction, not claimed to be coarsest. Camera-ready will reword §6 Case 1 to frame it as "a safe reduction" rather than "the reduction".

**Code**: `signature.build_signature_quotient` returns the coarsest-safe 16. A user-chosen structural π_obs quotient can be added as a separate helper (pending demand).

### A.6 Structural expansion algorithm

The paper uses "structural expansion" informally. The explicit procedure (based on Eq. (6) in ZTB22):

```
def expand_cfvar(cfvar, G_star):
    # G_star is the edge-split graph.
    # Each non-fixed node V^(i) in G_star corresponds to one f_V invocation.
    # Its argument tuple is determined recursively by its parents in G_star.
    for node in topo_order(G_star):
        if is_fixed_split(node):
            continue
        base_V = original_of(node)
        parent_values_symbolic = [eval_symbolic(p, G_star) for p in parents(node)]
        record_S_V(base_V, parent_values_symbolic)
```

Code: `expansion.enumerate_contexts(G, Q)` returns `{V: list[tuple[(parent_name, parent_value, edge_label)]]}` — the exact set of labeled parent-world contexts visited. $S_V(\mathcal{Q})$ is this, deduplicated.

### A.7 Lemma 4.3 (Lem A.1) "disjoint" condition

Says $m_V(\mathcal{Q}) = \sum_T |S_V(T)|$ iff the per-term $S_V(T)$ are **pairwise disjoint**. The paper doesn't specify how to *check* disjointness — it's obvious once you enumerate $S_V$, but in code this is literally just a set-union implementation.

**Resolution**: `context_set.m_V(G, Q)` returns `(|union|, sum_of_sizes, disjoint_flag)`.

### A.8 PNS graph TikZ — ✅ resolved 2026-04-25

Paper `natural_bound_app.tex`'s PNS figure omits the `X → Y` directed edge that the associated table implicitly assumes (table reports Y indexed by $(X, M)$, giving lifted dim = 128). Our strict reading of the TikZ gave only 32.

**Author (2026-04-25)**: `X → Y` was dropped during a copy-paste from the NDE template. The intended graph is $X \to M \to Y, X \to Y, X \leftrightarrow Y, M \leftrightarrow Y$. TikZ fix coming in camera-ready.

**Code**: `examples.pns_graph()` now includes the `X → Y` edge; lifted dim = 128 matches the paper.

### A.9 Canonical SCM per-c-component vs per-c-clique

Prelims Def. A.5 defines `Explicit(G)` as adding one exogenous $U$ **per c-clique** (not per c-component). But §2 Def. 2.2 (canonical SCM) has $|\Omega_U| \leq \prod_{V \in \mathbf{C}(U)} |\Omega_{\mathrm{PA}_V} \mapsto \Omega_V|$ — using c-component, not c-clique. ZTB22 Def. 2.2 also uses c-component. These are consistent only when every c-component is also a c-clique; in general (e.g., Non-IV graph, appendix §3), they differ.

**Resolution**: For our code, the LP simplex only needs $\Omega_{\mathrm{full}}$ (product of per-node response-function spaces). The distinction c-component vs c-clique only affects how the *prior* over $\psi$ factorizes (product over c-components, or finer over c-cliques). Since partial-ID doesn't depend on a prior, we ignore this subtlety and enumerate $\Omega_{\mathrm{full}}$ per-node.

---

## B. Zhang-Tian-Bareinboim (ZTB22)

### B.1 $\mu$-notation in Eq. (10)–(11)

ZTB22 Eq. (10): $P(x_{z'}, y_{x'}) = \sum_{u_1, u_2} \mu_x^{(z', u_2)} \mu_y^{(x', u_1, u_2)} \theta_{u_1} \theta_{u_2}$.

At first glance $\mu_y$ appears to have three superscripts. It is really $\mu_V^{(\mathrm{pa}_V, u_V)}$ where for $Y$ in the Non-IV graph, $u_V = (u_1, u_2)$ because $Y$ belongs to a c-component spanning both latent confounders. So $\mu_y^{(x', u_1, u_2)}$ means "the value of $f_Y$ when $\mathrm{pa}_Y = x'$ and exogenous $(u_1, u_2)$".

**Resolution**: In code, a response function is `f_V: tuple[pa_value, ...] -> V_value` — a pure function that doesn't carry a $u$ argument. The "joint" behavior across c-components is represented by the joint distribution on $\psi$, not by a single response function. So we don't need $\mu$ notation at all; we just need $f_V \in \mathcal{F}_V$.

### B.2 "Canonical" overloading

ZTB22 uses "canonical" for two things:

1. Discretized-to-finite exogenous domain (the BP94 result for IV).
2. Response-function encoding (the general case per Thm 2.4).

These are equivalent once you fix the encoding, but the prose flips between them.

**Resolution**: We only implement (2). `canonical.enumerate_states(G)` returns the product of $\mathcal{F}_V$ — no reference to exogenous domains.

### B.3 $\Omega_{\mathrm{PA}_V} \mapsto \Omega_V$ notation

ZTB22 + 1086 both write $|\Omega_{\mathrm{PA}_V} \mapsto \Omega_V|$ to mean "number of functions from $\Omega_{\mathrm{PA}_V}$ to $\Omega_V$", i.e., $|\Omega_V|^{|\Omega_{\mathrm{PA}_V}|}$.

**Resolution**: Code uses the explicit formula `d_V ** prod(d_p for p in parents(V))` — no custom $\mapsto$ operator.

### B.4 Exogenous as "u" vs per-variable "u_V"

ZTB22 Def. 2.3 writes $v \leftarrow f_V(\mathrm{pa}_V, u_V)$ — one exogenous per V — but then Thm 2.4 says c-components share exogenous. The response-function representation *eliminates* per-node exogenous: each node has its own response function $f_V$, and the joint distribution on response functions absorbs any correlation across c-components.

**Resolution**: Same as B.1 — drop $u$ entirely from the response-function layer.

---

## C. Yang-Bareinboim (YB25)

### C.1 Def. 11 vs Def. 12 formal distinction

Both define realizable distributions, both via Eq. (37) / (38), both with conditions (i)–(ii). The key differences:

- **Def. 11 ($\mathcal{L}_{2.25}$)** is indexed by an **interventional value set** $\mathbf{x} \in \mathrm{Val}(\mathbf{X})$: fixed specific values.
- **Def. 12 ($\mathcal{L}_{2.5}$)** is indexed by an **interventional variable set** $\mathbf{X}$: specific values can differ across counterfactuals.

Condition (ii) difference:

- **$\mathcal{L}_{2.25}$**: "for any $v_i \in \mathbf{x}$ and all $V_j \in \mathbf{Y}$, if $V_i \in \mathrm{An}(V_j)$ in $\mathcal{M}_{\mathbf{x} \setminus V_j}$, then $v_i \in \mathbf{x}_j$." → consistency at the variable's own X-value.
- **$\mathcal{L}_{2.5}$**: "for any $V_i$ and any $B \in \mathbf{X} \cap \mathrm{Pa}(V_i)$, for all $V_j \in \mathbf{Y}$: if $V_i \notin \mathbf{X}_j$ and $V_i \in \mathrm{An}(V_j)$ in $\mathcal{M}_{\mathbf{x}_j}$, then $\mathbf{x}_i \cap B = \mathbf{x}_j \cap B$." → consistency at the **parent-of-$V_i$** level, not at $V_i$ itself.

**Resolution**: Code's `CFQuery.layer` field with explicit checks. The test for $\mathcal{L}_{2.5}$ is weaker, and `edge_split_graph(Q)` constructs the split graph; if that construction succeeds without a well-definedness error, the query is at least $\mathcal{L}_{2.5}$-realizable.

### C.2 CTF-RAND (Def. 8)

YB25 introduces this as an experimental primitive: intervene on $X$'s *perception* by a subset of children only. This is what makes $\mathcal{L}_{2.5}$ physically meaningful. **We do not need to implement CTF-RAND** — it's a physical-experiment concept; we only need the queries it generates, which are captured by the edge-label map.

---

## D. Balke-Pearl (BP94)

### D.1 Response function ≡ our $\mathcal{F}_V$

BP94's "response function variable" is exactly our $f_V \in \mathcal{F}_V$. BP94 only considered the IV graph; the generalization to arbitrary ADMG is ZTB22 Thm 2.4.

**Resolution**: No code impact.

### D.2 BP94's 16-state domain for IV

BP94 discretized $U_2$ (the X↔Y confounder in IV graph) to 16 values. In our framework, the response-function enumeration gives $|\mathcal{F}_Z| \cdot |\mathcal{F}_X| \cdot |\mathcal{F}_Y| = 2 \cdot 4 \cdot 16 = 128$ states (if Z has no parent, X has parent Z, Y has parent X). BP94's 16 is the contribution of the $\{X, Y\}$ c-component for *$\mathcal{L}_1$* observational — consistent with our Prop layer1 formula.

**Resolution**: Sanity test in `tests/test_cardinality.py` verifies both.

---

## E. Practical ambiguities in existing code (for comparison)

Not strictly about the paper, but worth flagging so the new code doesn't repeat these:

### E.1 `count_parent_world_contexts` in the old `common_lib.py`

It keys contexts by `tuple(sorted(ctx.items()))` without the world index — so it merges across worlds. For NDE this happens to be correct (two worlds have the same intervenor $X$ with different values, and the tuple encoding captures the value), but in general this drops label information and undercounts.

### E.2 `prop42_upper_bound` in the old `common_lib.py`

Restricts to `ancestors(outcomes) ∩ c_component(outcomes)` — neither matches paper's actual $\mathbf{C}(U)$ selection.

**Resolution in new code**: `context_set.m_V(G, Q)` uses the edge-split graph and never merges labels. `context_set.prop42_cardinality(G, Q)` iterates over c-components of `G.induce(An(Q))`, which is the correct set.
