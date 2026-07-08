"""Context multiplicity m_V(Q) and Prop. 4.2 cardinality.

See docs/01_concepts.md §10 and paper §4.1–4.2.

  m_V(Q) = |S_V(Q)| = # distinct labeled parent-world contexts at which
                       f_V is invoked in the structural expansion of Q.

Prop. 4.2 (realizable mixed-world queries):

  d_U(Q)  =  prod_{V in C(U)}  d_V ^ m_V(Q),

with the product over c-components C of the ancestor-subgraph G[An(Q)].
(See docs/01_concepts.md §8 for why this uses the original c-components
restricted to ancestors, not the post-do mutilated subgraph.)

PCH worst-case cardinalities (analytic, query-independent):

  L_1:  d_U = prod_{V in C(U) union Pa(C(U))} d_V
            [c-component variables and their directed parents]
  L_2:  d_U = prod_{V in C(U)} d_{Pa(V) × V}   = prod_{V in C(U)} d_V * prod_p d_p
  L_3:  d_U = prod_{V in C(U)} d_V ^ |Omega_{Pa(V)}|    (response-function count)
"""

from math import prod
from typing import Literal

from canonred.counterfactual import CFQuery
from canonred.expansion import enumerate_contexts
from canonred.graph import ADMG
from canonred.types import VarName


def m_V(G: ADMG, Q: CFQuery, V: VarName) -> int:
    """Context multiplicity |S_V(Q)|."""
    return len(enumerate_contexts(G, Q)[V])


def context_multiplicities(G: ADMG, Q: CFQuery) -> dict[VarName, int]:
    """Return {V: m_V(Q)} for every V in G.nodes."""
    S = enumerate_contexts(G, Q)
    return {v: len(S[v]) for v in G.nodes}


def prop42_cardinality(G: ADMG, Q: CFQuery) -> int:
    """Prop. 4.2 sufficient cardinality d_U(Q).

    d_U(Q) = prod over c-components C of (G restricted to An(Q)) of
             prod over V in C of d_V ^ m_V(Q).

    Nodes with m_V = 0 contribute d_V^0 = 1, so intervened / unused nodes
    vanish.
    """
    an = G.ancestors(Q.intervened_any | {cv.base for cv in Q.cfvars}, inclusive=True)
    G_sub = G.induce(an)
    comps = G_sub.c_components()
    m = context_multiplicities(G, Q)
    total = 1
    for C in comps:
        block = 1
        for v in C:
            block *= G.cardinality[v] ** m[v]
        total *= block
    return total


def pch_cardinality(G: ADMG, layer: Literal["L1", "L2", "L3"]) -> int:
    """Analytic PCH-layer canonical cardinality for the full graph.

    Computed per c-component of G and multiplied across components.
    """
    if layer not in ("L1", "L2", "L3"):
        raise ValueError(f"layer must be one of 'L1'/'L2'/'L3' (got {layer!r})")
    total = 1
    for C in G.c_components():
        total *= _layer_component_count(G, C, layer)
    return total


def _layer_component_count(
    G: ADMG, C: frozenset[VarName], layer: Literal["L1", "L2", "L3"]
) -> int:
    if layer == "L1":
        # Paper examples use the closed directed-parent domain C union pa(C).
        pa_of_C: set[VarName] = set()
        for v in C:
            pa_of_C.update(G.parents(v))
        domain = set(C) | pa_of_C
        return prod(G.cardinality[p] for p in domain)
    if layer == "L2":
        # d_V × prod_p d_p  for each V in C
        block = 1
        for v in C:
            pa_cards = [G.cardinality[p] for p in G.parents(v)]
            block *= G.cardinality[v] * prod(pa_cards) if pa_cards else G.cardinality[v]
        return block
    # L3: response-function count per V
    block = 1
    for v in C:
        pa_dom = prod(G.cardinality[p] for p in G.parents(v)) if G.parents(v) else 1
        block *= G.cardinality[v] ** pa_dom
    return block


__all__ = [
    "context_multiplicities",
    "m_V",
    "pch_cardinality",
    "prop42_cardinality",
]
