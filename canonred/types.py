"""Type aliases used across the package.

See docs/01_concepts.md §1.
"""

from typing import TypeAlias

VarName: TypeAlias = str
"""Name of an observed variable."""

Value: TypeAlias = int
"""A value from a variable's finite domain: non-negative integer < cardinality."""

Label: TypeAlias = str
"""Symbolic name for a world (e.g., "obs", "x0", "x1")."""

EdgeKey: TypeAlias = tuple[VarName, VarName]
"""Directed edge, as an ordered pair (parent, child)."""
