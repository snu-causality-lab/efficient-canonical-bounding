"""World / do-intervention specification.

See docs/01_concepts.md §2. A World is a labelled partial map from VarName to
Value — a do-intervention. Two worlds are compatible iff they agree on every
variable in the intersection of their intervened sets. Labels are part of the
identity (needed for L_{2.5} where two edges may carry the same do-map under
different labels).
"""

from collections.abc import Mapping
from dataclasses import dataclass

from canonred.types import Label, Value, VarName

_OBS_LABEL: Label = "obs"


def _validate_do_entry(var: VarName, val: object) -> Value:
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(
            f"World value for {var!r} must be a non-negative int (got {val!r} of type {type(val).__name__})"
        )
    if val < 0:
        raise ValueError(f"World value for {var!r} must be non-negative (got {val})")
    return val


@dataclass(frozen=True)
class World:
    """A do-intervention. Immutable, hashable.

    Attributes
    ----------
    label : Label
        Symbolic name. Part of identity; two worlds with identical `do` but
        different labels compare non-equal.
    do : frozenset of (VarName, Value)
        Intervention map, stored as a hashable set of (var, value) pairs.
    """

    label: Label
    do: frozenset[tuple[VarName, Value]]

    @staticmethod
    def observational() -> "World":
        """Return the observational world: no interventions, label='obs'."""
        return World(label=_OBS_LABEL, do=frozenset())

    @staticmethod
    def from_dict(label: Label, mapping: Mapping[VarName, Value]) -> "World":
        """Build a world from a {var: value} dict, validating each entry."""
        items: list[tuple[VarName, Value]] = []
        for var, val in mapping.items():
            checked = _validate_do_entry(var, val)
            items.append((var, checked))
        return World(label=label, do=frozenset(items))

    @property
    def intervened(self) -> frozenset[VarName]:
        """Set of variables fixed by this world."""
        return frozenset(var for var, _ in self.do)

    def value_of(self, var: VarName) -> Value | None:
        """Return the do-value assigned to `var`, or None if not intervened."""
        for v, x in self.do:
            if v == var:
                return x
        return None

    def compatible_with(self, other: "World") -> bool:
        """True iff self and other agree on every shared intervened variable."""
        shared = self.intervened & other.intervened
        for v in shared:
            if self.value_of(v) != other.value_of(v):
                return False
        return True


__all__ = ["World"]
