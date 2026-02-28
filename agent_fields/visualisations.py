"""Vega-Lite specs for FieldMetrics and Field visualization.

Each function takes a FieldMetrics or Field and returns a plain dict — a
complete Vega-Lite spec ready to hand to any renderer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .field import Field
    from .metrics import FieldMetrics

_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"


def center_bar(m: FieldMetrics, *, width: int = 500, height: int = 250) -> dict:
    """Bar chart of per-dimension mean (the field center)."""
    center = m.center()
    dims = m._dimensions
    data = [
        {"dimension": dims[i].name, "mean": round(float(center[i]), 3)}
        for i in range(m.d)
    ]
    return {
        "$schema": _SCHEMA,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {"field": "dimension", "type": "nominal",
                   "sort": None, "axis": {"labelAngle": -45}},
            "y": {"field": "mean", "type": "quantitative",
                   "title": "Mean value"},
            "color": {"value": "#4c78a8"},
        },
        "width": width,
        "height": height,
    }


def variance_bar(m: FieldMetrics, *, width: int = 500, height: int = 250) -> dict:
    """Bar chart of per-dimension variance."""
    var = m.variance()
    dims = m._dimensions
    data = [
        {"dimension": dims[i].name, "variance": round(float(var[i]), 4)}
        for i in range(m.d)
    ]
    return {
        "$schema": _SCHEMA,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {"field": "dimension", "type": "nominal",
                   "sort": None, "axis": {"labelAngle": -45}},
            "y": {"field": "variance", "type": "quantitative",
                   "title": "Variance"},
            "color": {"value": "#f58518"},
        },
        "width": width,
        "height": height,
    }


def separation_bar(m: FieldMetrics, *, width: int = 500, height: int = 250) -> dict:
    """Bar chart of the separation vector (mu+ - mu-).

    Green bars = success trajectories averaged higher on that dimension.
    Red bars = success trajectories averaged lower.
    """
    sep = m.separation()
    dims = m._dimensions
    data = [
        {"dimension": dims[i].name, "separation": round(float(sep[i]), 4)}
        for i in range(m.d)
    ]
    return {
        "$schema": _SCHEMA,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {"field": "dimension", "type": "nominal",
                   "sort": None, "axis": {"labelAngle": -45}},
            "y": {"field": "separation", "type": "quantitative",
                   "title": "Separation (μ+ − μ−)"},
            "color": {
                "condition": {"test": "datum.separation >= 0",
                              "value": "#59a14f"},
                "value": "#e15759",
            },
        },
        "width": width,
        "height": height,
    }


# ── Comparative charts (two FieldMetrics side by side) ───────────────


def _grouped_bar(
    m_a: FieldMetrics,
    m_b: FieldMetrics,
    label_a: str,
    label_b: str,
    value_field: str,
    value_title: str,
    extract,
    *,
    width: int = 600,
    height: int = 250,
) -> dict:
    """Internal: grouped bar chart comparing a metric across two fields."""
    dims_a = m_a._dimensions
    vals_a = extract(m_a)
    vals_b = extract(m_b)
    data = []
    for i, dim in enumerate(dims_a):
        data.append({"dimension": dim.name, "prompt": label_a,
                     value_field: round(float(vals_a[i]), 4)})
        data.append({"dimension": dim.name, "prompt": label_b,
                     value_field: round(float(vals_b[i]), 4)})
    return {
        "$schema": _SCHEMA,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {"field": "dimension", "type": "nominal",
                   "sort": None, "axis": {"labelAngle": -45}},
            "y": {"field": value_field, "type": "quantitative",
                   "title": value_title},
            "color": {"field": "prompt", "type": "nominal"},
            "xOffset": {"field": "prompt", "type": "nominal"},
        },
        "width": width,
        "height": height,
    }


def compare_center_bar(
    m_a: FieldMetrics, m_b: FieldMetrics,
    label_a: str = "A", label_b: str = "B",
    **kw,
) -> dict:
    """Grouped bar chart comparing per-dimension means of two fields."""
    return _grouped_bar(m_a, m_b, label_a, label_b,
                        "mean", "Mean value",
                        lambda m: m.center(), **kw)


def compare_variance_bar(
    m_a: FieldMetrics, m_b: FieldMetrics,
    label_a: str = "A", label_b: str = "B",
    **kw,
) -> dict:
    """Grouped bar chart comparing per-dimension variance of two fields."""
    return _grouped_bar(m_a, m_b, label_a, label_b,
                        "variance", "Variance",
                        lambda m: m.variance(), **kw)


def skew_bar(m: FieldMetrics, *, width: int = 500, height: int = 250) -> dict:
    """Bar chart of skew — correlation between outcome and each dimension.

    Green = positive (expensive success). Red = negative (cheap success).
    """
    dims = m._dimensions
    data = [
        {"dimension": dims[i].name, "skew": round(float(m.skew(i)), 4)}
        for i in range(m.d)
    ]
    return {
        "$schema": _SCHEMA,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {"field": "dimension", "type": "nominal",
                   "sort": None, "axis": {"labelAngle": -45}},
            "y": {"field": "skew", "type": "quantitative",
                   "title": "Skew (corr with outcome)",
                   "scale": {"domain": [-1, 1]}},
            "color": {
                "condition": {"test": "datum.skew >= 0",
                              "value": "#59a14f"},
                "value": "#e15759",
            },
        },
        "width": width,
        "height": height,
    }


# ── Horizon charts ─────────────────────────────────────────────────


def horizon_width(field: Field, *, width: int = 500, height: int = 250) -> dict:
    """Bar chart: field width at each state along the progression."""
    data = [
        {"state": s, "width": round(field.horizon(s).metrics().width(), 4)}
        for s in field.states
    ]
    return {
        "$schema": _SCHEMA,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {"field": "state", "type": "ordinal", "sort": None,
                   "title": "State"},
            "y": {"field": "width", "type": "quantitative",
                   "title": "Horizon Width"},
            "color": {"value": "#4c78a8"},
        },
        "width": width,
        "height": height,
    }


def horizon_convergence(field: Field, *, width: int = 500, height: int = 250) -> dict:
    """Bar chart: convergence at each state along the progression."""
    data = []
    for s in field.states:
        h = field.horizon(s)
        if h.K < 2:
            continue
        c = h.metrics().convergence()
        # Cap infinite convergence for display
        if not isinstance(c, (int, float)) or c != c:  # NaN check
            continue
        display_c = min(float(c), 20.0)
        data.append({"state": s, "convergence": round(display_c, 4)})
    return {
        "$schema": _SCHEMA,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {"field": "state", "type": "ordinal", "sort": None,
                   "title": "State"},
            "y": {"field": "convergence", "type": "quantitative",
                   "title": "Convergence"},
            "color": {"value": "#59a14f"},
        },
        "width": width,
        "height": height,
    }


def horizon_drift(
    field: Field,
    *,
    threshold: float = 0.5,
    width: int = 500,
    height: int = 250,
) -> dict:
    """Bar chart: drift delta(s) at each state.

    delta(s) = W_H(s) - W_H+(s) — the gap between overall horizon width
    and success-only horizon width. Positive means failing trajectories
    have diverged from the success corridor at that state.
    """
    data = []
    for s in field.states:
        h = field.horizon(s)
        W_all = h.metrics().width()
        sr = h.success_region(threshold)
        W_success = sr.metrics().width() if sr.K >= 2 else W_all
        data.append({"state": s, "drift": round(W_all - W_success, 4)})
    return {
        "$schema": _SCHEMA,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {"field": "state", "type": "ordinal", "sort": None,
                   "title": "State"},
            "y": {"field": "drift", "type": "quantitative",
                   "title": "Drift δ(s)"},
            "color": {
                "condition": {"test": "datum.drift >= 0",
                              "value": "#e15759"},
                "value": "#59a14f",
            },
        },
        "width": width,
        "height": height,
    }
