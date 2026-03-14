"""Helpers for MultiFetchTaskField visualizations.

Loads all multi-fetch experiments, builds one universal field,
provides horizon heatmap and horizon drift chart builders.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STUDY_DIR)
sys.path.insert(0, VIZ_DIR)

from task_field import MultiFetchTaskField

BLOG_DIR = os.path.join(STUDY_DIR, "blog")
DATASET_PATH = os.path.join(BLOG_DIR, "dataset.json")

# Dracula palette — one color per experiment
EXPERIMENT_COLORS = {
    "progressive": "#ff5555",
    "rescue": "#ffb86c",
    "breadcrumb": "#50fa7b",
    "breadcrumb_poison": "#ff79c6",
    "breadcrumb_exec": "#bd93f9",
    "url_redirect": "#8be9fd",
    "url_redirect_funky": "#6272a4",
    "ipa_progressive": "#f1fa8c",
    "ipa_exfil": "#d4a017",
    "base64_breadcrumb": "#ff6e6e",
}


# ── Data Loading ─────────────────────────────────────────────


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def get_latest_multi_fetch(experiments: list[dict]) -> dict[str, dict]:
    """Get latest multi-fetch experiment per injection strategy.

    Returns dict: injection_name → experiment dict.
    """
    latest = {}
    for e in experiments:
        if e["K"] < 2 or e["environment"] != "multi_fetch":
            continue
        key = e["injection"]
        if key not in latest or e["created_at"] > latest[key]["created_at"]:
            latest[key] = e
    return latest


def build_universal_field(
    experiments: dict[str, dict],
    norms: dict[str, float] | None = None,
) -> tuple[MultiFetchTaskField, list[str]]:
    """Build one field from ALL multi-fetch experiments.

    Returns (field, labels) where labels[i] is the injection strategy
    name for trajectory i.
    """
    field = MultiFetchTaskField(norms=norms)
    labels = []

    for inj_name in sorted(experiments.keys()):
        exp = experiments[inj_name]
        for traj, outcome in zip(
            exp["join"]["trajectories"],
            exp["join"]["outcomes"],
        ):
            field.add(traj, outcome)
            labels.append(inj_name)

    return field, labels


# ── Chart 1: Horizon × Experiment Heatmap ────────────────────


def build_experiment_heatmap(
    field: MultiFetchTaskField,
    labels: list[str],
) -> go.Figure:
    """Heatmap: experiments (rows) × dimensions (columns), cell = center value.

    Shows the behavioral profile of each strategy at a glance.
    Normalized dimensions: 1.0 = expected baseline.
    """
    dim_names = [d.name for d in field.dimensions()]
    exp_names = sorted(set(labels))
    labels_arr = np.array(labels)

    # Build matrix: experiments × dimensions
    matrix = np.zeros((len(exp_names), len(dim_names)))

    for i, exp_name in enumerate(exp_names):
        mask = labels_arr == exp_name
        exp_field = field.subset(mask)
        if exp_field.K > 0:
            matrix[i] = exp_field.metrics().center()

    # Annotations with values
    annotations = []
    vmax = np.max(matrix)
    threshold = vmax * 0.5
    for i, exp_name in enumerate(exp_names):
        for j, dim in enumerate(dim_names):
            val = matrix[i, j]
            annotations.append(dict(
                x=dim, y=exp_name,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(
                    color="#282a36" if val > threshold else "#f8f8f2",
                    size=11,
                ),
            ))

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=dim_names,
        y=exp_names,
        colorscale=[
            [0.0, "#282a36"],
            [0.05, "#44475a"],
            [0.2, "#6272a4"],
            [0.4, "#bd93f9"],
            [0.7, "#ff79c6"],
            [1.0, "#ff5555"],
        ],
        colorbar=dict(
            title=dict(text="Center", side="right"),
            tickfont=dict(color="#f8f8f2"),
        ),
        hovertemplate="Experiment: %{y}<br>Dimension: %{x}<br>Center: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title={
            "text": "Field Center (μ<sub>F</sub>) — Average Behavior per Experiment<br>"
                    "<sup style='color:#6272a4'>"
                    "Each cell = mean across K=5 runs, normalized so 1.0 = expected baseline. | "
                    "num_fetches: ÷5 expected | total_tool_calls: ÷10 baseline | "
                    "explicit_refusal: 0/1 per run"
                    "</sup>",
            "font": {"size": 14},
        },
        annotations=annotations,
        height=max(400, len(exp_names) * 40 + 150),
        paper_bgcolor="#282a36",
        plot_bgcolor="#282a36",
        font={"color": "#f8f8f2"},
        xaxis={"title": "Dimension (φ)", "side": "bottom"},
        yaxis={"title": ""},
        margin={"l": 160},
    )

    return fig


# ── Chart 2: Horizon Drift Lines ─────────────────────────────


def build_horizon_drift(
    field: MultiFetchTaskField,
    labels: list[str],
    metric: str = "width",
) -> go.Figure:
    """Line chart: state progression (x) vs metric (y), one line per experiment.

    Shows how behavioral diversity evolves through the task for each strategy.
    Lines that diverge from the cluster reveal WHERE the injection takes effect.
    """
    state_order = ["on_track", "drifting", "deep_drift"]
    states = [s for s in state_order if s in field.states]

    exp_names = sorted(set(labels))
    labels_arr = np.array(labels)

    fig = go.Figure()

    for exp_name in exp_names:
        mask = labels_arr == exp_name
        exp_field = field.subset(mask)

        values = []
        for s in states:
            h = exp_field.horizon(s)
            if h.K > 1:
                m = h.metrics()
                values.append(m.width() if metric == "width" else float(np.linalg.norm(m.center())))
            else:
                values.append(0.0)

        color = EXPERIMENT_COLORS.get(exp_name, "#6272a4")
        fig.add_trace(go.Scatter(
            x=states,
            y=values,
            mode="lines+markers",
            name=exp_name,
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color),
            hovertemplate=f"{exp_name}<br>State: %{{x}}<br>{metric}: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        title={
            "text": f"Horizon Drift — {metric.replace('_', ' ').title()} across States<br>"
                    "<sup style='color:#6272a4'>How behavioral diversity evolves through the task</sup>",
            "font": {"size": 14},
        },
        xaxis={"title": "State (ψ) — task progression", "gridcolor": "#44475a"},
        yaxis={"title": metric.replace("_", " ").title(), "gridcolor": "#44475a"},
        height=500,
        paper_bgcolor="#282a36",
        plot_bgcolor="#282a36",
        font={"color": "#f8f8f2"},
        legend={"x": 0.02, "y": 0.98},
    )

    return fig


# ── Chart 3: Per-dimension center at each horizon ────────────


def build_horizon_dimension_bars(
    field: MultiFetchTaskField,
    labels: list[str],
    experiments: list[str] | None = None,
) -> go.Figure:
    """Grouped bar: center per dimension at each state, for selected experiments.

    Shows HOW the behavioral profile shifts across the task for each strategy.
    If experiments is None, shows all.
    """
    state_order = ["on_track", "drifting", "deep_drift"]
    states = [s for s in state_order if s in field.states]
    dim_names = [d.name for d in field.dimensions()]

    exp_names = sorted(set(labels))
    if experiments:
        exp_names = [e for e in exp_names if e in experiments]
    labels_arr = np.array(labels)

    fig = make_subplots(
        rows=1, cols=len(states),
        subplot_titles=[f"State: {s}" for s in states],
        shared_yaxes=True,
    )

    for exp_name in exp_names:
        mask = labels_arr == exp_name
        exp_field = field.subset(mask)
        color = EXPERIMENT_COLORS.get(exp_name, "#6272a4")

        for j, s in enumerate(states):
            h = exp_field.horizon(s)
            if h.K == 0:
                continue
            center = h.metrics().center()
            fig.add_trace(
                go.Bar(
                    name=exp_name,
                    x=dim_names,
                    y=center.tolist(),
                    marker_color=color,
                    showlegend=(j == 0),
                    legendgroup=exp_name,
                ),
                row=1, col=j + 1,
            )

    fig.update_layout(
        title={
            "text": "Behavioral Profile at Each State<br>"
                    "<sup style='color:#6272a4'>Normalized dimension centers — "
                    "1.0 = expected baseline behavior</sup>",
            "font": {"size": 14},
        },
        barmode="group",
        height=500,
        paper_bgcolor="#282a36",
        plot_bgcolor="#282a36",
        font={"color": "#f8f8f2"},
        legend={"x": 0.02, "y": -0.15, "orientation": "h"},
    )

    for i in range(len(states)):
        fig.update_xaxes(gridcolor="#44475a", row=1, col=i + 1)
        fig.update_yaxes(gridcolor="#44475a", row=1, col=i + 1)

    return fig


# ── Chart 4: Horizon Width Heatmap (experiment × state) ──────


def build_horizon_width_heatmap(
    field: MultiFetchTaskField,
    labels: list[str],
) -> go.Figure:
    """Heatmap: experiments (rows) × states (columns), cell = field width.

    Shows how behavioral diversity (spread) changes across task states
    for each injection strategy.

    NOTE: if all trajectories pass through all states, columns will be
    identical — that itself is a finding (the state boundaries need
    refinement, or behavior should be measured per-state window).
    """
    state_order = ["on_track", "drifting", "deep_drift"]
    states = [s for s in state_order if s in field.states]

    exp_names = sorted(set(labels))
    labels_arr = np.array(labels)

    matrix = np.zeros((len(exp_names), len(states)))

    for i, exp_name in enumerate(exp_names):
        mask = labels_arr == exp_name
        exp_field = field.subset(mask)
        for j, s in enumerate(states):
            h = exp_field.horizon(s)
            if h.K > 1:
                matrix[i, j] = h.metrics().width()

    # Annotations
    annotations = []
    vmax = np.max(matrix) if np.max(matrix) > 0 else 1.0
    threshold = vmax * 0.5
    for i, exp_name in enumerate(exp_names):
        for j, s in enumerate(states):
            val = matrix[i, j]
            annotations.append(dict(
                x=s, y=exp_name,
                text=f"{val:.2f}",
                showarrow=False,
                font=dict(
                    color="#282a36" if val > threshold else "#f8f8f2",
                    size=11,
                ),
            ))

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=states,
        y=exp_names,
        colorscale=[
            [0.0, "#282a36"],
            [0.05, "#44475a"],
            [0.2, "#6272a4"],
            [0.4, "#bd93f9"],
            [0.7, "#ff79c6"],
            [1.0, "#ff5555"],
        ],
        colorbar=dict(
            title=dict(text="Width", side="right"),
            tickfont=dict(color="#f8f8f2"),
        ),
        hovertemplate="Experiment: %{y}<br>State: %{x}<br>Width: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title={
            "text": "Field Width (W<sub>F</sub>) — Behavioral Diversity per State<br>"
                    "<sup style='color:#6272a4'>"
                    "Width = tr(Cov) = sum of per-dimension variances. "
                    "0 = every run identical. Higher = more behavioral spread."
                    "</sup>",
            "font": {"size": 14},
        },
        annotations=annotations,
        height=max(400, len(exp_names) * 40 + 150),
        paper_bgcolor="#282a36",
        plot_bgcolor="#282a36",
        font={"color": "#f8f8f2"},
        xaxis={"title": "State (ψ) — task progression", "side": "bottom"},
        yaxis={"title": ""},
        margin={"l": 160},
    )

    return fig
