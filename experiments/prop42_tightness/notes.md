# prop42_tightness — Prop 4.2 tightness examples ( Q5.1)

**Purpose**: paper Q5.1: *"how tight is Prop. 4.2?"*

**Answer structure**:
- Prop 4.2 gives a **sufficient** cardinality `d_U(Q)` for a realizable query. It is not a coarsest-safe lower bound.
- The coarsest safe quotient (Prop 3.1's σ-signature classes) can be **strictly smaller** than `d_U` when the information set lets us further collapse states.

**Fixtures** (both on the bow graph `X→Y, X↔Y`, binary):

| Case | Query | d_full | d_U (Prop 4.2) | d_red (σ-quotient) |
|---|---|---:|---:|---:|
| Tight | `P(Y_{x=0}=1, Y_{x=1}=1)` (PNS-like) | 8 | 4 | ≤4 (observable marginals tell the rest) |
| Slack | `P(Y_{x=0}=1, Y_obs=1)` (mixed world) | 8 | 8 | <8 |

**Output**: `prop42_tightness.csv` with `case, d_full, d_U, d_red, m_V`.

**Argument in the appendix**: Prop 4.2 is not claimed to be the coarsest; it is a structural upper bound via the context multiplicity `m_V(Q)`. The two cases above make concrete when d_U matches d_full (structural tight) and when observational constraints allow a tighter quotient.

**Runtime**: trivial (<1 s).
