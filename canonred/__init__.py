"""Canonical domain reduction for partial counterfactual identification.

Implements the 1086 UAI 2026 paper "Canonical Domain Reduction for Partial
Counterfactual Identification" on a clean foundation. See ``docs/01_concepts.md``
for the formalism and ``docs/03_modules.md`` for per-module roles.

Public API (stable names):

- ``graph``: ADMG with directed and bidirected edges, c-components, mutilation.
- ``worlds``: World (do-intervention) spec.
- ``counterfactual``: CFVar (edge-labeled) and CFQuery (joint event).
- ``expansion``: edge-split graph + S_V(Q) enumeration.
- ``canonical``: response functions + canonical state evaluation.
- ``observation``: moment constraints (P_obs, L2.5 marginals, etc.).
- ``context_set``: Prop. 4.2 cardinality.
- ``signature``: coarsest safe quotient (Prop. 3.1).
- ``lp``: build and solve sharp-bound LPs.
- ``posterior``: Dirichlet / Gibbs samplers for phi.
- ``examples``: named (graph, query, constraints) fixtures with expected values.
"""

from canonred.types import EdgeKey, Label, Value, VarName  # noqa: F401

__all__ = ["EdgeKey", "Label", "Value", "VarName"]
__version__ = "0.0.1"
