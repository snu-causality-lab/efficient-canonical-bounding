"""Unit tests for canonred.worlds (spec: docs/01_concepts.md §2).

A World is a do-intervention spec: a label + a partial map from VarName to Value.
Core responsibilities:
  - construction via `from_dict` and `observational`;
  - read-only access (`value_of`, `intervened`);
  - equality / hashability (frozen);
  - compatibility check between two worlds on their shared intervened set.
"""

import pytest

from canonred.worlds import World


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_observational_world_has_empty_do():
    """The observational world has no interventions."""
    w = World.observational()
    assert w.intervened == frozenset()
    assert len(w.do) == 0


def test_observational_world_label_is_obs():
    """Observational label convention is 'obs'."""
    w = World.observational()
    assert w.label == "obs"


def test_from_dict_single_intervention():
    """A single do(X=1) yields intervened={'X'} and value_of('X')==1."""
    w = World.from_dict("x1", {"X": 1})
    assert w.label == "x1"
    assert w.intervened == frozenset({"X"})
    assert w.value_of("X") == 1


def test_from_dict_multi_intervention():
    """do(X=0, Y=1) intervenes on both."""
    w = World.from_dict("xy", {"X": 0, "Y": 1})
    assert w.intervened == frozenset({"X", "Y"})
    assert w.value_of("X") == 0
    assert w.value_of("Y") == 1


def test_value_of_non_intervened_is_none():
    """value_of for a variable not in the do-set returns None."""
    w = World.from_dict("x0", {"X": 0})
    assert w.value_of("Y") is None


# ---------------------------------------------------------------------
# Immutability / hashability
# ---------------------------------------------------------------------


def test_world_is_frozen():
    """A World cannot be mutated after construction."""
    w = World.from_dict("x1", {"X": 1})
    with pytest.raises((AttributeError, TypeError)):
        w.label = "other"  # type: ignore[misc]


def test_worlds_with_same_do_and_label_are_equal():
    """Frozen dataclass equality: identical fields → equal."""
    a = World.from_dict("x1", {"X": 1})
    b = World.from_dict("x1", {"X": 1})
    assert a == b
    assert hash(a) == hash(b)


def test_worlds_with_different_label_not_equal():
    """Label is part of identity: same do, different label → distinct worlds.

    This is what makes L_{2.5} richer than L_{2.25}: two edges can carry the
    same value-intervention but different labels.
    """
    a = World.from_dict("x1", {"X": 1})
    b = World.from_dict("x1-alt", {"X": 1})
    assert a != b


def test_worlds_with_different_do_not_equal():
    a = World.from_dict("x1", {"X": 1})
    b = World.from_dict("x1", {"X": 0})
    assert a != b


def test_world_is_hashable_for_set_membership():
    """Must work as dict key and set member (frozen dataclass)."""
    a = World.from_dict("x1", {"X": 1})
    b = World.from_dict("x0", {"X": 0})
    s = {a, b, a}
    assert len(s) == 2


# ---------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------


def test_compatibility_disjoint_domains():
    """Worlds that intervene on disjoint variables are compatible."""
    a = World.from_dict("x1", {"X": 1})
    b = World.from_dict("y1", {"Y": 1})
    assert a.compatible_with(b)
    assert b.compatible_with(a)


def test_compatibility_observational_always_compatible():
    """The empty world is compatible with everyone."""
    obs = World.observational()
    w = World.from_dict("x1", {"X": 1})
    assert obs.compatible_with(w)
    assert w.compatible_with(obs)


def test_compatibility_matching_values():
    """Shared intervened variable with the same value → compatible."""
    a = World.from_dict("xy", {"X": 1, "Y": 0})
    b = World.from_dict("x", {"X": 1})
    assert a.compatible_with(b)


def test_compatibility_conflicting_values():
    """Shared intervened variable with different values → incompatible."""
    a = World.from_dict("x0", {"X": 0})
    b = World.from_dict("x1", {"X": 1})
    assert not a.compatible_with(b)


def test_self_compatible():
    """Any world is compatible with itself."""
    a = World.from_dict("xy", {"X": 1, "Y": 0})
    assert a.compatible_with(a)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------


def test_empty_label_is_allowed():
    """Empty string label is legal (caller responsibility)."""
    w = World.from_dict("", {})
    assert w.label == ""


def test_negative_value_rejected():
    """Values must be non-negative integers (0-indexed domain)."""
    with pytest.raises(ValueError):
        World.from_dict("bad", {"X": -1})


def test_non_int_value_rejected():
    """Values must be ints, not strings or floats."""
    with pytest.raises(TypeError):
        World.from_dict("bad", {"X": 1.5})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        World.from_dict("bad", {"X": "1"})  # type: ignore[dict-item]


# ---------------------------------------------------------------------
# intervened() is a pure accessor
# ---------------------------------------------------------------------


def test_intervened_frozenset_type():
    """The `intervened` property returns a frozenset for hashability."""
    w = World.from_dict("xy", {"X": 0, "Y": 1})
    assert isinstance(w.intervened, frozenset)
