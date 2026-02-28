from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .dimension import Dimension


class FieldMetrics:
    """Computes all field metrics from a point cloud and outcome labels.

    The cloud (points) describes *what trajectories did* — behavioral coordinates.
    The outcomes describe *how well they did* — labels painted on the cloud.
    These are kept separate: the cloud is the map, outcomes are the terrain.
    """

    def __init__(
        self,
        points: np.ndarray,
        outcomes: np.ndarray,
        dimensions: list[Dimension],
    ):
        """
        Args:
            points: (K, d) array — the point cloud.
            outcomes: (K,) array — y(tau) labels for each point.
            dimensions: list of d Dimension objects.
        """
        self._points = points
        self._outcomes = outcomes
        self._dimensions = dimensions

    @property
    def K(self) -> int:
        return self._points.shape[0]

    @property
    def d(self) -> int:
        return self._points.shape[1]

    # ── Section 4.1: Field Center ──────────────────────────────────────

    def center(self) -> np.ndarray:
        """mu_F — the centroid of the point cloud. Returns R^d vector."""
        return np.mean(self._points, axis=0)

    # ── Section 4.2: Field Width ───────────────────────────────────────

    def width(self) -> float:
        """W_F — trace of covariance matrix. Single scalar summary of spread."""
        return float(np.sum(self.variance()))

    def variance(self) -> np.ndarray:
        """Per-dimension variance. R^d vector — the diagnostic breakdown."""
        return np.var(self._points, axis=0)

    def covariance(self) -> np.ndarray:
        """Full (d, d) covariance matrix of the point cloud."""
        return np.cov(self._points, rowvar=False)

    # ── Section 4.3: Field Convergence ─────────────────────────────────

    def convergence(self) -> float:
        """C_F — E[y] / sigma[y]. How reliably do trajectories succeed?"""
        std = np.std(self._outcomes)
        if std == 0:
            return float("inf") if np.mean(self._outcomes) > 0 else 0.0
        return float(np.mean(self._outcomes) / std)

    # ── Section 4.4: Field Skew ────────────────────────────────────────

    def skew(self, cost_dim: str | int) -> float:
        """S_F — correlation between outcome and a cost dimension.

        Args:
            cost_dim: dimension name (str) or index (int).
        """
        idx = self._resolve_dim(cost_dim)
        cost = self._points[:, idx]
        if np.std(cost) == 0 or np.std(self._outcomes) == 0:
            return 0.0
        return float(np.corrcoef(self._outcomes, cost)[0, 1])

    # ── Section 4.5: Separation Vector ─────────────────────────────────

    def separation(self, threshold: float = 0.5) -> np.ndarray:
        """mu+ - mu- — the vector in behavioral space separating success
        from failure. Returns R^d vector.

        Args:
            threshold: outcome >= threshold is success.
        """
        success_mask = self._outcomes >= threshold
        failure_mask = ~success_mask

        if not np.any(success_mask) or not np.any(failure_mask):
            return np.zeros(self.d)

        mu_plus = np.mean(self._points[success_mask], axis=0)
        mu_minus = np.mean(self._points[failure_mask], axis=0)
        return mu_plus - mu_minus

    # ── Summary ────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Human-readable summary keyed by dimension names."""
        center = self.center()
        var = self.variance()
        sep = self.separation()

        per_dim = {}
        for i, dim in enumerate(self._dimensions):
            per_dim[dim.name] = {
                "description": dim.description,
                "mean": float(center[i]),
                "variance": float(var[i]),
                "separation": float(sep[i]),
            }

        return {
            "dimensions": per_dim,
            "width": self.width(),
            "convergence": self.convergence(),
            "K": self.K,
            "d": self.d,
        }

    # ── Internal ───────────────────────────────────────────────────────

    def _resolve_dim(self, dim: str | int) -> int:
        if isinstance(dim, int):
            return dim
        for i, d in enumerate(self._dimensions):
            if d.name == dim:
                return i
        raise ValueError(f"Unknown dimension: {dim!r}")
