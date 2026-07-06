# named_graphs — named-graph comparison ()

**Purpose**: additional graph-variety checks beyond the NDE running example.

**Fixtures**:

| Name | Graph | Query | Lifted dim |
|---|---|---|---|
| `bow` | `X→Y, X↔Y` | `Y_{X=1}` | 8 |
| `iv` | `Z→X→Y, X↔Y` | `Y_{X=1}` | 32 |
| `non_iv` | `Z→X→Y, X↔Y, Z↔Y` | `Y_{X=1}` | 32 |
| `frontdoor` | `X→W→Y, X↔Y` | `Y_{X=1}` | 32 |
| `nde` | `X→W→Y, X→Y, X↔W` | `Y_{0, W_1}` (L_{2.5}) | 128 |
| `pns` | `X→M→Y, X→Y, X↔Y, M↔Y` | `Y_x ∧ Y_{x'}` cross-world | 128 |
| `ternary_nde` | `Z→X→M→Y, X→Y, X↔Y`; M ternary | `Y_{1, M_0}` | 4608 |

All queries paired with full `P_obs` constraints (`2^|V|` moments).

**Output**: `named_fixtures.csv` with columns `fixture, n_nodes, lifted_dim, reduced_dim, prop42_d_U, build_*, med_*_ms, speedup, max_abs_lb_diff, max_abs_ub_diff`.

**Expected**:
- Bounds agreement at solver tolerance (~1e-4 to 1e-6).
- Reduced_dim ≪ lifted_dim for most; speedup grows with lifted size.
- Ternary NDE (lifted=4608) gives large speedup.
