"""Shared helpers for prototype visualization notebooks.

Loads dataset, builds BreadcrumbFamilyField from real data,
provides chart builders for horizon waterfall and intent Sankey.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import plotly.graph_objects as go

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STUDY_DIR)
sys.path.insert(0, VIZ_DIR)

from blog_field import BreadcrumbFamilyField

BLOG_DIR = os.path.join(STUDY_DIR, "blog")
DATASET_PATH = os.path.join(BLOG_DIR, "dataset.json")


# ── Data Loading ─────────────────────────────────────────────


def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


def get_breadcrumb_family(experiments: list[dict]) -> dict:
    """Get latest breadcrumb + breadcrumb_poison experiments.

    Returns dict with keys 'breadcrumb' and 'breadcrumb_poison',
    each containing the experiment dict.
    """
    result = {}
    for e in experiments:
        if e["K"] < 2:
            continue
        if e["environment"] != "multi_fetch":
            continue
        if e["injection"] not in ("breadcrumb", "breadcrumb_poison"):
            continue
        key = e["injection"]
        if key not in result or e["created_at"] > result[key]["created_at"]:
            result[key] = e
    return result


def build_family_field(
    bc_exp: dict,
    bp_exp: dict,
    poison_after: int = 15,
) -> tuple[BreadcrumbFamilyField, list[str]]:
    """Build a BreadcrumbFamilyField from both experiments.

    Returns (field, labels) where labels[i] is 'breadcrumb' or 'breadcrumb_poison'
    for each trajectory, in ingestion order.
    """
    field = BreadcrumbFamilyField(poison_after=poison_after)
    labels = []

    # Ingest breadcrumb trajectories (outcome = 1.0 since all held)
    for traj, outcome in zip(bc_exp["join"]["trajectories"], bc_exp["join"]["outcomes"]):
        field.add(traj, outcome)
        labels.append("breadcrumb")

    # Ingest breadcrumb_poison trajectories
    for traj, outcome in zip(bp_exp["join"]["trajectories"], bp_exp["join"]["outcomes"]):
        field.add(traj, outcome)
        labels.append("breadcrumb_poison")

    return field, labels


# ── Chart 1: Horizon Width Waterfall (ψ lens) ───────────────


def build_horizon_waterfall(field: BreadcrumbFamilyField, labels: list[str]) -> go.Figure:
    """Bar chart: field width at each horizon state, split by experiment.

    X-axis = state (horizon). Y-axis = width.
    Two bar groups: breadcrumb vs breadcrumb_poison.
    Shows WHERE in the trajectory's progression behavioral diversity appears.
    """
    # Sort states in logical order
    state_order = ["browsing", "poison_exposed", "leaked"]
    states = [s for s in state_order if s in field.states]

    # Split into sub-fields by experiment label
    bc_mask = np.array([l == "breadcrumb" for l in labels])
    bp_mask = np.array([l == "breadcrumb_poison" for l in labels])

    fig = go.Figure()

    for mask, name, color in [
        (bc_mask, "breadcrumb", "#50fa7b"),
        (bp_mask, "breadcrumb_poison", "#ff79c6"),
    ]:
        # Subset by experiment first, then query horizons on that sub-field
        exp_field = field.subset(mask)
        widths = []
        ks = []
        for s in states:
            h = exp_field.horizon(s)
            if h.K > 1:
                widths.append(float(h.metrics().width()))
            else:
                widths.append(0.0)
            ks.append(h.K)

        fig.add_trace(go.Bar(
            name=name,
            x=states,
            y=widths,
            marker_color=color,
            text=[f"K={k}, W={w:.1f}" for k, w in zip(ks, widths)],
            textposition="outside",
        ))

    fig.update_layout(
        title={
            "text": "Horizon Width Waterfall (ψ lens)<br>"
                    "<sup style='color:#6272a4'>Field width at each irreversible milestone</sup>",
            "font": {"size": 14},
        },
        barmode="group",
        xaxis_title="State (ψ) — trajectory milestone",
        yaxis_title="Field Width (behavioral diversity)",
        height=500,
        paper_bgcolor="#282a36",
        plot_bgcolor="#282a36",
        font={"color": "#f8f8f2"},
        xaxis={"gridcolor": "#44475a"},
        yaxis={"gridcolor": "#44475a"},
        legend={"x": 0.02, "y": 0.98},
    )

    return fig


# ── Chart 2: Horizon Center Comparison (ψ lens) ─────────────


def build_horizon_centers(field: BreadcrumbFamilyField, labels: list[str]) -> go.Figure:
    """Grouped bar: field center (mean per dimension) at each horizon, by experiment.

    Shows HOW the behavioral profile shifts at each milestone.
    """
    states = [s for s in ["browsing", "poison_exposed", "leaked"] if s in field.states]
    dim_names = [d.name for d in field.dimensions()]

    bc_mask = np.array([l == "breadcrumb" for l in labels])
    bp_mask = np.array([l == "breadcrumb_poison" for l in labels])

    fig = go.Figure()

    colors = {"breadcrumb": "#50fa7b", "breadcrumb_poison": "#ff79c6"}

    for mask, name in [(bc_mask, "breadcrumb"), (bp_mask, "breadcrumb_poison")]:
        exp_field = field.subset(mask)
        for s in states:
            h = exp_field.horizon(s)
            if h.K == 0:
                continue
            center = h.metrics().center()
            fig.add_trace(go.Bar(
                name=f"{name} @ {s}",
                x=dim_names,
                y=center.tolist(),
                marker_color=colors[name],
                opacity=0.4 + 0.3 * (states.index(s) / max(len(states) - 1, 1)),
            ))

    fig.update_layout(
        title={
            "text": "Horizon Centers (ψ lens)<br>"
                    "<sup style='color:#6272a4'>Mean behavioral vector at each milestone</sup>",
            "font": {"size": 14},
        },
        barmode="group",
        xaxis_title="Dimension",
        yaxis_title="Mean value",
        height=500,
        paper_bgcolor="#282a36",
        plot_bgcolor="#282a36",
        font={"color": "#f8f8f2"},
        xaxis={"gridcolor": "#44475a"},
        yaxis={"gridcolor": "#44475a"},
    )

    return fig


# ── Chart 3: Intent Sankey (ρ_π lens) ────────────────────────


def build_intent_sankey(field: BreadcrumbFamilyField, labels: list[str]) -> go.Figure:
    """Sankey diagram from program chains (ρ_π).

    Each unique intent is a node column. Flows show how many trajectories
    transition between intents. Colored by experiment.
    """
    from collections import Counter

    # Split by experiment, get programs from each sub-field
    bc_mask = np.array([l == "breadcrumb" for l in labels])
    bp_mask = np.array([l == "breadcrumb_poison" for l in labels])

    # Collect (program, experiment_label, count) triples using public API
    chains: list[tuple[tuple[str, ...], str, int]] = []
    for mask, name in [(bc_mask, "breadcrumb"), (bp_mask, "breadcrumb_poison")]:
        exp_field = field.subset(mask)
        for prog in exp_field.programs:
            fam = exp_field.program_family(prog)
            chains.append((prog, name, fam.K))

    # Build node list: (step_index, intent_label)
    node_ids = {}  # (step, intent) → id
    node_labels = []
    node_colors = []

    intent_colors = {
        "fetching": "#8be9fd",
        "lured": "#ff5555",
        "refusing": "#ffb86c",
        "summarizing": "#50fa7b",
    }

    def get_node(step: int, intent: str) -> int:
        key = (step, intent)
        if key not in node_ids:
            node_ids[key] = len(node_ids)
            node_labels.append(f"{intent} (t={step})")
            node_colors.append(intent_colors.get(intent, "#6272a4"))
        return node_ids[key]

    # Count transitions weighted by trajectory count
    link_counts: Counter = Counter()
    link_experiments: dict = {}  # (src, tgt) → Counter of experiment labels

    for prog, label, count in chains:
        for i in range(len(prog) - 1):
            src = get_node(i, prog[i])
            tgt = get_node(i + 1, prog[i + 1])
            link_counts[(src, tgt)] += count
            if (src, tgt) not in link_experiments:
                link_experiments[(src, tgt)] = Counter()
            link_experiments[(src, tgt)][label] += count

    # Build Sankey data
    sources = []
    targets = []
    values = []
    link_colors = []

    exp_colors = {
        "breadcrumb": "rgba(80, 250, 123, 0.5)",
        "breadcrumb_poison": "rgba(255, 121, 198, 0.5)",
    }

    for (src, tgt), count in link_counts.items():
        # Determine dominant experiment for coloring
        exp_counter = link_experiments[(src, tgt)]
        dominant = exp_counter.most_common(1)[0][0]

        sources.append(src)
        targets.append(tgt)
        values.append(count)
        link_colors.append(exp_colors.get(dominant, "rgba(98, 114, 164, 0.5)"))

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node={
            "label": node_labels,
            "color": node_colors,
            "pad": 20,
            "thickness": 20,
        },
        link={
            "source": sources,
            "target": targets,
            "value": values,
            "color": link_colors,
        },
    ))

    fig.update_layout(
        title={
            "text": "Intent Flow — Program Chains (ρ_π lens)<br>"
                    "<sup style='color:#6272a4'>How model intent transitions across program steps. "
                    "Green=breadcrumb, Pink=breadcrumb_poison</sup>",
            "font": {"size": 14},
        },
        height=600,
        paper_bgcolor="#282a36",
        font={"color": "#f8f8f2", "size": 11},
    )

    return fig


# ── Chart 4: Program Family Distribution (ρ_π lens) ──────────


def build_program_families(field: BreadcrumbFamilyField, labels: list[str]) -> go.Figure:
    """Horizontal bar: each distinct program chain, how many trajectories, colored by experiment.

    Shows the partitioning of trajectories by behavioral signature.
    """
    from collections import Counter

    bc_mask = np.array([l == "breadcrumb" for l in labels])
    bp_mask = np.array([l == "breadcrumb_poison" for l in labels])

    # Count per program per experiment using public API
    prog_counts: dict[tuple, Counter] = {}
    for mask, name in [(bc_mask, "breadcrumb"), (bp_mask, "breadcrumb_poison")]:
        exp_field = field.subset(mask)
        for prog in exp_field.programs:
            fam = exp_field.program_family(prog)
            if prog not in prog_counts:
                prog_counts[prog] = Counter()
            prog_counts[prog][name] = fam.K

    # Sort by total count descending
    sorted_progs = sorted(prog_counts.items(), key=lambda x: sum(x[1].values()), reverse=True)

    # Format program chain names (truncate long ones)
    def fmt_prog(prog: tuple) -> str:
        parts = [f"{p}" for p in prog]
        s = " → ".join(parts)
        return s if len(s) <= 80 else s[:77] + "..."

    prog_labels = [fmt_prog(p) for p, _ in sorted_progs]
    bc_counts = [c.get("breadcrumb", 0) for _, c in sorted_progs]
    bp_counts = [c.get("breadcrumb_poison", 0) for _, c in sorted_progs]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="breadcrumb",
        y=prog_labels,
        x=bc_counts,
        orientation="h",
        marker_color="#50fa7b",
    ))
    fig.add_trace(go.Bar(
        name="breadcrumb_poison",
        y=prog_labels,
        x=bp_counts,
        orientation="h",
        marker_color="#ff79c6",
    ))

    fig.update_layout(
        title={
            "text": "Program Families (ρ_π lens)<br>"
                    "<sup style='color:#6272a4'>Distinct behavioral signatures — "
                    "how trajectories partition by intent pattern</sup>",
            "font": {"size": 14},
        },
        barmode="stack",
        xaxis_title="Number of trajectories",
        yaxis_title="Program chain",
        height=max(400, len(sorted_progs) * 50 + 150),
        paper_bgcolor="#282a36",
        plot_bgcolor="#282a36",
        font={"color": "#f8f8f2", "size": 11},
        xaxis={"gridcolor": "#44475a"},
        yaxis={"gridcolor": "#44475a", "autorange": "reversed"},
        legend={"x": 0.7, "y": 0.98},
        margin={"l": 300},
    )

    return fig
