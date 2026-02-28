from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    """Metadata for a single dimension of the behavioral vector.

    Each dimension returned by measure() has a corresponding Dimension
    that names it and describes what it captures. This metadata makes
    field metrics interpretable — "variance on scope_ratio: ratio of
    actions on relevant files" instead of "variance on dimension 5".
    """

    name: str
    """Short identifier used as a key in metrics summaries."""

    description: str
    """What this dimension measures — written for a human reading the summary."""
