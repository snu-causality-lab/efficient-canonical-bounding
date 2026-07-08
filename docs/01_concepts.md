# Concept Specification — Canonical Reduction for Partial CF Identification

Target paper: "Canonical Domain Reduction for Partial Counterfactual Identification" (UAI 2026 paper). Supporting references: Zhang-Tian-Bareinboim 2022 (ZTB22), Balke-Pearl 1994 (BP94), Yang-Bareinboim 2025 (YB25), Tian-Pearl 2002.

This doc fixes the precise objects and semantics we implement. Every code module in `canonred/` refers back here.

---

## 1. Causal diagram — ADMG

An **acyclic directed mixed graph** (ADMG) is a triple

$$
\mathcal{G} = (\mathbf{V},\ \mathbf{E}_\rightarrow,\ \mathbf{E}_\leftrightarrow)
$$

where:

- $\mathbf{V} = \{V_1, \ldots, V_n\}$ is a finite set of **observed** variables. Each $V \in \mathbf{V}$ carries a finite domain $\Omega_V$ with $d_V := |\Omega_V| \geq 2$.
- $\mathbf{E}_\rightarrow \subseteq \mathbf{V} \times \mathbf{V}$: directed edges. The subgraph $(\mathbf{V}, \mathbf{E}_\rightarrow)$ is a DAG.
- $\mathbf{E}_\leftrightarrow$: bidirected edges, i.e., a set of unordered pairs $\{V_i, V_j\}$ with $i \neq j$. Semantically, each bidirected edge encodes an unobserved common cause.

**Notation**. We use lower-case $\mathrm{pa}(V) = \{p : (p \to V) \in \mathbf{E}_\rightarrow\}$ for the (strict) parent set. The paper sometimes writes $\mathrm{Pa}(V) = \mathrm{pa}(V) \cup \{V\}$ — in code we stick to strict parents and flag this in docstrings. $\mathrm{ch}(V)$, $\mathrm{an}(V)$, $\mathrm{de}(V)$ are children, ancestors (excl. $V$), descendants (excl. $V$). $\mathrm{An}(\mathbf{Y}) := \mathbf{Y} \cup \bigcup_{V\in \mathbf{Y}} \mathrm{an}(V)$ is the **inclusive** ancestor set.

**C-component**. A c-component $\mathbf{C} \subseteq \mathbf{V}$ is a maximal connected component of $(\mathbf{V}, \mathbf{E}_\leftrightarrow)$'s transitive closure. Equivalence classes $\{\mathbf{C}_1, \ldots, \mathbf{C}_k\}$ partition $\mathbf{V}$.

**C-clique** (appendix Def. A.2): a c-component such that every pair is joined by *a direct bidirected edge* (i.e., the bidirected subgraph induced on $\mathbf{K}$ is complete). Every c-clique is a c-component, but not vice versa. Example (Non-IV graph): $\mathbf{V}=\{X,Y,Z\}$ is one c-component; $\{X,Y\}$ and $\{Z,Y\}$ are c-cliques.

**Subgraphs and operations**:

- $\mathcal{G}[\mathbf{S}]$: vertex-induced subgraph on $\mathbf{S} \subseteq \mathbf{V}$.
- $\mathcal{G}_{\overline{\mathbf{X}}}$: **post-do graph** — delete all directed edges *into* every $X \in \mathbf{X}$, and all bidirected edges incident on $\mathbf{X}$. (This is the graph after a do-intervention on $\mathbf{X}$.)
- $\mathcal{G}_{\underline{\mathbf{X}}}$: delete edges *out of* $\mathbf{X}$ (rare — used in twin-network constructions).

**Topological order**: any linear extension compatible with $\mathbf{E}_\rightarrow$. For a semi-Markovian ADMG this is always a permutation of $\mathbf{V}$.

---

## 2. Worlds and interventions

A **world** is a do-intervention specification — a partial function

$$
w: \mathbf{V} \rightharpoonup \bigsqcup_V \Omega_V
$$

with $\mathrm{dom}(w) =: \mathbf{X}_w \subseteq \mathbf{V}$ and, for each $X \in \mathbf{X}_w$, $w(X) \in \Omega_X$. We write this as $w = \mathrm{do}(X_1=x_1, \ldots, X_k=x_k)$.

- The **observational world** is $w_\emptyset = \emptyset$ (no intervention).
- Two worlds $w, w'$ are **compatible** on a variable $X \in \mathbf{X}_w \cap \mathbf{X}_{w'}$ iff $w(X) = w'(X)$.

A **world label** is a short symbolic ID (string) attached to a world (e.g., `"obs"`, `"x0"`, `"x1"`, `"xp"`). Two worlds with the same intervention map but different labels are semantically *distinct* when they appear as labels on different edges of a counterfactual query (this is exactly what makes $\mathcal{L}_{2.5}$ strictly richer than $\mathcal{L}_{2.25}$, per YB25 Def. 11–12).

---

## 3. Counterfactual variables (CFVars)

We adopt the **edge-labeled intervention** representation, which is equivalent to the nested-subscript notation $Y_{x, W_{x'}}$ but is algorithmically cleaner.

### 3.1 Edge-labeled intervention

Given the ancestor subgraph $\mathcal{G}[\mathrm{An}(V)]$, a **CFVar** $\underline{V}$ at base $V$ is specified by an **edge-label map**

$$
\sigma_V : \mathbf{E}_\rightarrow \cap \big(\mathrm{An}(V) \times \mathrm{An}(V)\big) \to \mathcal{L}(\mathcal{Q})
$$

where $\mathcal{L}(\mathcal{Q})$ is the finite set of world labels used by the query $\mathcal{Q}$. Semantically, $\sigma_V(p \to c)$ says "when evaluating node $c$'s parent input $p$, we are in world $\sigma_V(p \to c)$".

### 3.2 Recursive evaluation

For a canonical SCM $\mathcal{M}$ with response functions $\{f_V\}$ and exogenous $u$, the value of $\underline{V}$ under $u$ is:

$$
\underline{V}(u; \sigma_V) = f_V\left(\bigl( \mathrm{value}(p \to V)(u) \bigr)_{p \in \mathrm{pa}(V)},\ u_V\right)
$$

where

$$
\mathrm{value}(p \to V)(u)
=
\begin{cases}
w(p) & \text{if } p \in \mathbf{X}_w \text{ and } w := \sigma_V(p \to V), \\
\underline{p}(u; \sigma_V|_p) & \text{otherwise}.
\end{cases}
$$

The recursion $\sigma_V|_p$ is defined by **label inheritance**: all incoming edges of $p$ carry the label $\sigma_V(p \to V)$, unless a stronger constraint is imposed by the query (e.g., $\mathcal{L}_{2.5}$ allows $\sigma_V$ to assign distinct labels to two children of a single $X$; when we recurse into $p$'s ancestors, we inherit $p$'s edge-label from the edge we arrived on).

### 3.3 Common cases

- **Standard potential outcome** $Y_x$ (equivalently, $\mathrm{CFVar}(Y)$ in world $\mathrm{do}(X=x)$): $\sigma_Y$ assigns label $x$ to every edge in the ancestor subgraph.
- **Observational** $Y$: $\sigma_Y$ assigns label `"obs"` (empty do) to every edge.
- **Nested NDE-style** $Y_{x, W_{x'}}$ on graph $X \to W \to Y$, $X \to Y$, $X \leftrightarrow W$: assign label $x$ to edges $X \to Y$ and $W \to Y$; assign label $x'$ to edge $X \to W$. By label inheritance, when we recurse into $W$'s parents, $X$'s incoming edges get label $x'$, so $W$'s value is $W_{x'} = f_W(x', u_W)$.
- **PNS** $Y_x \wedge Y_{x'}$: two CFVars, both on the same base $Y$, with $\sigma$ from labels $x$ and $x'$ respectively.

### 3.4 Realizable (conflict-free) queries

Following YB25:

- A query is **$\mathcal{L}_{2.25}$-realizable** iff every intervened variable $X$ carries the **same** label on all of its outgoing edges in $\sigma$ (across all CFVars in $\mathcal{Q}$). Equivalently, in the edge-split representation of §3.5 below, $X$ is duplicated into at most one copy *per query*.
- A query is **$\mathcal{L}_{2.5}$-realizable** iff the children-level consistency condition (YB25 Def. 12 (ii)) holds: for any child $V$ of an intervened $X$, if two CFVars $V_*$ and $V_*'$ disagree on the value assigned to $X$ on the edge $(X \to V)$, then their sub-world-labelings below $V$ must still be consistent on $X$'s non-$V$ ancestors. The cleanest operational check: duplicating $X$ per-outgoing-edge label yields a well-defined DAG with no contradictions (see §3.5).

### 3.5 Edge-split graph $\mathcal{G}^\star$

For any $\mathcal{L}_{2.5}$-realizable query $\mathcal{Q}$:

- Collect the distinct labels $\mathcal{L}(\mathcal{Q})$ appearing in $\sigma$.
- For each intervened node $X$, for each distinct label $w \in \mathcal{L}(\mathcal{Q})$ that intervenes on $X$, create a split copy $X^{(w)}$ with value fixed to $w(X)$.
- Redirect each edge $(X \to V)$ with label $\sigma(X \to V) = w$ to $(X^{(w)} \to V)$.

The resulting graph $\mathcal{G}^\star$ has no free $X$-variable (all are split-fixed). The query now lives on a pure response-function space and is a *plain* do-intervention on $\mathcal{G}^\star$. This is exactly the "realizable NDE graph" in paper Fig. 2b.

**Conflict-freeness** = the edge-split operation produces a well-defined labeling (no edge is simultaneously assigned two incompatible labels).

---

## 4. Queries

A **query** is a set of counterfactual equality events:

$$
\mathcal{Q} = P\big( \underline{V}_1 = v_1^*,\ \underline{V}_2 = v_2^*,\ \ldots,\ \underline{V}_k = v_k^* \big)
$$

where each $\underline{V}_i = (V_i, \sigma_i)$ is a CFVar and $v_i^* \in \Omega_{V_i}$ is a target value.

For partial ID we typically have $\mathcal{Q}$ = a single joint probability. The paper NDE running example is $\mathcal{Q} = P(\underline{Y} = 1)$ with $\underline{Y} = Y_{0, W_1}$.

---

## 5. Canonical SCM and response-function space

Following BP94, ZTB22 Def. 2.3, and paper Def. 2.2:

### 5.1 Per-node response functions

A node $V$ with strict parents $\mathrm{pa}(V)$ has a **response function space**

$$
\mathcal{F}_V := \Omega_V^{\Omega_{\mathrm{pa}(V)}} \cong \{f : \Omega_{\mathrm{pa}(V)} \to \Omega_V\}.
$$

Cardinality: $|\mathcal{F}_V| = d_V^{\prod_{p \in \mathrm{pa}(V)} d_p}$. For root $V$ (empty parents), $|\mathcal{F}_V| = d_V$.

### 5.2 Canonical state space

The **full lifted canonical state space** of $\mathcal{G}$ is

$$
\Omega_{\mathrm{full}} := \prod_{V \in \mathbf{V}} \mathcal{F}_V.
$$

Each state $u = (f_V)_{V \in \mathbf{V}}$ is a tuple of response functions. $|\Omega_{\mathrm{full}}| = \prod_V d_V^{\prod_p d_p}$ is the **$\mathcal{L}_3$ cardinality** (paper Sec. 4.1).

### 5.3 Canonical distribution $\psi$

A **canonical distribution** $\psi \in \Delta(\Omega_{\mathrm{full}})$ assigns probability mass to each response-function tuple.

Relationship to c-components: in the canonical SCM with one exogenous per c-clique (appendix Def. A.5), $\psi$ is a product of per-c-component distributions if response-functions across c-components are independent. For partial-ID LPs we do not rely on this product factorization — $\psi$ is just a probability vector on $\Omega_{\mathrm{full}}$.

### 5.4 L₁, L₂, L₃ sufficient cardinalities

Layer-by-layer sufficient exogenous cardinalities (paper §2 / Appendix A.7–A.10):

| Layer | Formula | Interpretation |
|---|---|---|
| $\mathcal{L}_1$ (obs) | $d_U = \prod_{V \in \mathbf{C}(U) \cup \mathrm{pa}(\mathbf{C}(U))} d_V$ | product over the c-component variables and their directed-parent domains |
| $\mathcal{L}_2$ (single intv) | $d_U = \prod_{V \in \mathbf{C}(U)} d_V \cdot \prod_p d_p$ | product over input-output pairs |
| $\mathcal{L}_3$ (CF) | $d_U = \prod_{V \in \mathbf{C}(U)} d_V^{\prod_p d_p}$ | product over *functions* |

For our code, only $\mathcal{L}_3$ is exercised by the LP (since LP bounds for partial ID require the response-function space).

---

## 6. Evaluation of a CFVar under a canonical state

Given a state $u = (f_V)$ and a CFVar $\underline{V} = (V, \sigma)$, the value $\underline{V}(u)$ is computed by recursive structural expansion. Operationally, we preprocess $\mathcal{Q}$ into its edge-split graph $\mathcal{G}^\star$ (§3.5), which is an ordinary DAG with all interventions expressed as "fixed-value" split copies. Then each original $V \in \mathbf{V}$ maps to one canonical state entry $f_V$ (the same $f_V$ is used for all split copies of $V$ that are not themselves fixed).

Pseudocode (see `canonred/canonical.py`):

```python
def eval_cfvar(u, cfvar, G_star):
    """Return u(V, σ)."""
    val = {}
    for node in topo_order(G_star):
        if is_fixed(node):                 # split copy X^{(w)} with value w(X)
            val[node] = fixed_value(node)
        else:
            base = original_node_of(node)  # f_V shared across non-fixed copies
            pa_vals = tuple(val[p] for p in parents(node, G_star))
            val[node] = u[base][pa_vals]
    return val[cfvar.base_node_in_G_star]
```

The key correctness property: for the original NDE query $Y_{0, W_1}$ on the binary NDE graph, this returns exactly the 8-state reduced evaluation used in the paper: $Y_{0, W_1}(u) = (1 - a)b + ac$ with $a = W_1(u) = f_W(1, u_W)$, $b = Y_{0,0}(u) = f_Y(0, 0, u_Y)$, $c = Y_{0,1}(u) = f_Y(0, 1, u_Y)$.

---

## 7. Observation model and constraints

For a given query $\mathcal{Q}$ we pair it with an **information set** $\mathcal{I}$ — the set of linear functionals of $\psi$ that the data identify. Common cases:

- **$P_{\mathrm{obs}}$ regime** ($\mathcal{I} = P_{\mathrm{obs}}(\mathbf{V})$). For each observable cell $\mathbf{v} \in \Omega_{\mathbf{V}}$, the row
$$M_{\mathbf{v}, u} = \mathds{1}\{V(u) = v_V \ \forall V \in \mathbf{V}\}$$
where $V(u) := \mathrm{eval}\_\mathrm{cfvar}(u, \mathrm{CFVar}(V, \sigma_\emptyset))$ is the observational evaluation.
- **$\mathcal{L}_{2.5}$ marginal regime** (the paper running example). $\mathcal{I} = \{P(W_1), P(Y_{00}), P(Y_{01})\}$. Each marginal is the expectation over $\psi$ of the CFVar indicator.
- **Mixed / multi-world experimental data**: combinations of observational and interventional moments (e.g., ZTB22 polynomial-program formulation).

Each element of $\mathcal{I}$ gives one linear equality $M_i \psi = p_i$ (plus simplex $\mathbf{1}^\top \psi = 1$).

---

## 8. LP formulation

For a linear-in-$\psi$ query $\mathcal{Q}$ (indicator functional $c$), the sharp bounds are

$$
\underline{\mathcal{Q}}(p) = \min_{\psi \in \Delta(\Omega_{\mathrm{full}})} c^\top \psi \ \ \text{s.t.} \ M\psi = p,
\quad
\overline{\mathcal{Q}}(p) = \max_{\psi \in \Delta(\Omega_{\mathrm{full}})} c^\top \psi \ \ \text{s.t.} \ M\psi = p.
$$

Both are standard LPs. We solve with CVXPY (ECOS primary, SCS backup) and cache the problem with `cp.Parameter` to make repeated $p$-sweeps fast.

---

## 9. Constraint-aware reduction (Prop. 3.1)

Given a surjection $\pi: \Omega_{\mathrm{full}} \to \Omega_{\mathrm{red}}$ with pushforward matrix $\Pi \in \{0,1\}^{|\Omega_{\mathrm{red}}| \times |\Omega_{\mathrm{full}}|}$, $\Pi_{t,u} = \mathds{1}\{\pi(u) = t\}$, define $\phi := \Pi \psi$. The reduction is **safe** iff

$$
\exists\ \tilde c, \tilde M\ :\ c = \Pi^\top \tilde c\ \land\ M = \tilde M \Pi.
$$

Under this condition, the LPs on $(\Omega_{\mathrm{full}}, c, M)$ and on $(\Omega_{\mathrm{red}}, \tilde c, \tilde M)$ have identical optimal values (paper Prop. 3.1, proof in Appendix E).

### 9.1 Coarsest safe quotient

The **signature** of $u \in \Omega_{\mathrm{full}}$ is

$$
\sigma(u) := (c_u, M_{\cdot, u}) \in \mathbb{R}^{1+m}.
$$

Declare $u \sim u' \iff \sigma(u) = \sigma(u')$. The equivalence classes of $\sim$ define the coarsest safe quotient. Implementing it is one pass over $\Omega_{\mathrm{full}}$ plus a hash lookup on $\sigma$.

### 9.2 Reduced LP coefficients

For each class $t = [u]$, pick any representative and set

$$
\tilde c_t := c_u,\qquad \tilde M_{i, t} := M_{i, u}.
$$

These are well-defined because $\sigma$ is constant inside a class. The reduced LP:

$$
\min_{\phi \in \Delta(\Omega_{\mathrm{red}})} \tilde c^\top \phi\ \ \text{s.t.}\ \tilde M \phi = p.
$$

---

## 10. Context multiplicity and Prop. 4.2

### 10.1 Context set $S_V(\mathcal{Q})$

For a realizable mixed-world query $\mathcal{Q}$ and a node $V$ not entirely intervened out, the **parent-world context set** is the collection of **distinct labeled parent tuples** at which $f_V$ is invoked in the structural expansion of $\mathcal{Q}$:

$$
S_V(\mathcal{Q}) := \bigcup_{\underline{V_j} \in \mathcal{Q}} \Big\{\ (p = a_p)_{\lambda_p}\ :\ \text{(edge-split parent tuple)} f_V \text{ is called with in computing } \underline{V_j}\ \Big\}.
$$

Equivalently, letting $\mathcal{G}^\star$ be the edge-split graph of $\mathcal{Q}$ and $V^{(1)}, \ldots, V^{(k)}$ the non-fixed copies of $V$ in $\mathcal{G}^\star$:

$$
S_V(\mathcal{Q}) = \bigcup_{i=1}^{k} \big\{\ \text{value assignment to } \mathrm{pa}(V^{(i)}) \text{ in } \mathcal{G}^\star\ \big\}.
$$

This is a **set** — duplicate entries are merged. Each distinct tuple must include its **label** (the world context the tuple belongs to) so that two $f_V$ invocations with identical parent values but under different world labels are counted distinctly in general (necessary for L₃-style queries on the same graph). For realizable $\mathcal{L}_{2.5}$ queries, labels are implicit in the edge-split $V^{(i)}$ index.

Multiplicity: $m_V(\mathcal{Q}) := |S_V(\mathcal{Q})|$.

### 10.2 Prop. 4.2 cardinality bound

For a query $\mathcal{Q}$ with ancestor-post-do subgraph $\mathcal{G}^{(Q)} = \mathcal{G}[\mathrm{An}(\mathcal{Q})]_{\overline{\mathbf{X}_\mathcal{Q}}}$ (where $\mathbf{X}_\mathcal{Q}$ is the union of all intervened nodes across $\mathcal{Q}$'s worlds), and $\mathbf{C}(U) \subseteq \mathrm{An}(\mathcal{Q})$ a c-component in the **original** $\mathcal{G}$ (not $\mathcal{G}^{(Q)}$) restricted to ancestors:

$$
d_U(\mathcal{Q}) = \prod_{V \in \mathbf{C}(U)} d_V^{m_V(\mathcal{Q})}.
$$

Total sufficient cardinality: $\prod_U d_U(\mathcal{Q})$.

(For fully intervened $V$, $m_V(\mathcal{Q}) = 0$ and its factor is $1$.)

**Running example verification** (NDE with $\mathcal{Q} = P(Y_{0, W_1})$):

- c-components of original graph (bidirected $X \leftrightarrow W$): $\mathbf{C}_1 = \{X, W\}$, $\mathbf{C}_2 = \{Y\}$.
- $m_X(\mathcal{Q}) = 0$ (X fully intervened in both worlds).
- $m_W(\mathcal{Q}) = 1$: $W$ is called once, with $\mathrm{pa}(W) = \{X\}$ under label $x' = 1$.
- $m_Y(\mathcal{Q}) = 2$: $Y$ is called with $(X=0, W=0)$ and $(X=0, W=1)$.
- $d_{U_1}(\mathcal{Q}) = 2^0 \cdot 2^1 = 2$; $d_{U_2}(\mathcal{Q}) = 2^2 = 4$.
- Total: $2 \times 4 = 8$ ✓.

### 10.3 Lemma 4.3 (union decomposition)

For each CFVar term $T \in \mathcal{Q}$ separately, let $S_V(T)$ be the contexts induced by $T$'s expansion. Then $S_V(\mathcal{Q}) = \bigcup_T S_V(T)$ and $m_V(\mathcal{Q}) \leq \sum_T |S_V(T)|$, with equality iff the $S_V(T)$ are pairwise disjoint.

---

## 11. Summary: what the code must produce

Per (graph, query, information-set) triple:

- $\Omega_{\mathrm{full}}$ enumeration (or its cardinality symbolically, for large graphs).
- $c \in \{0,1\}^{|\Omega_{\mathrm{full}}|}$ (query indicator) and $M$ (observation matrix).
- Signature $\sigma(u)$ and the quotient $\pi$, reduced $\tilde c, \tilde M$.
- Cardinalities: $|\Omega_{\mathrm{full}}|, |\Omega_{\mathrm{red}}|$, and Prop. 4.2's $d_U$.
- LP solves (min/max) on both $(c, M)$ and $(\tilde c, \tilde M)$ — must agree up to solver tolerance.
- Runtime comparison.
- For posterior experiments: sample $\psi$ or $\phi$; propagate query and bound; verify agreement.
