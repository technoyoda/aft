"""Generate all test chart JSONs from the real dataset for the viz test page.

Produces 5 chart variants to compare approaches:
1. parallel_coords.json — Parallel coordinates (old vs new sonnet, all dims)
2. scatter_2d.json — 2D scatter on continuous dims (tool_calls vs duration)
3. strip_plot.json — Strip/jitter per dimension, grouped by model×outcome
4. radar.json — Radar chart showing centroid per group
5. scatter3d_real.json — 3D scatter with real data (improved axes)

Plus the existing:
6. width_bar.json — Cross-strategy width bar chart (multi_fetch)

Usage:
    cd aft/studies/study-2
    python blog/viz/generate_all_test_charts.py
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, STUDY_DIR)

from defense_field import (
    PromptDefenseField,
    GeneralBehaviorField,
    MultiDefenseField,
    _extract_tool_calls,
    _extract_text_blocks,
    _has_explicit_refusal,
    _called_feedback,
    _attempted_file_read,
    _engaged_with_injection,
)
from trajectory_utils import semantic_sequence, rle_string

BLOG_DIR = os.path.join(STUDY_DIR, "blog")
DATA_DIR = "/Users/valaydave/Documents/valays-passion-projects/technoyoda.github.io/assets/data"


def load_dataset() -> list[dict]:
    with open(os.path.join(BLOG_DIR, "dataset.json")) as f:
        return json.load(f)


# ────────────────────────────────────────────────────────────
# Helpers: measure one trajectory
# ────────────────────────────────────────────────────────────


def measure_trajectory(traj: dict, flog: dict) -> dict:
    """Compute all behavioral dimensions for one trajectory."""
    msgs = traj["messages"]

    # Extract tool calls (list of dicts with 'name', 'input', etc.)
    tool_calls = _extract_tool_calls(traj)

    # Extract text blocks (list of strings from assistant messages)
    texts = _extract_text_blocks(msgs)

    # Count fetches
    n_fetches = 0
    for tc in tool_calls:
        if tc.get("name") == "Bash":
            cmd = (tc.get("input") or {}).get("command", "")
            if "/fetch?" in cmd:
                n_fetches += 1

    return {
        "honeypot_called": 1.0 if _called_feedback(tool_calls) else 0.0,
        "file_read_attempt": 1.0 if _attempted_file_read(tool_calls) else 0.0,
        "explicit_refusal": 1.0 if _has_explicit_refusal(texts) else 0.0,
        "injection_engaged": 1.0 if _engaged_with_injection(texts) else 0.0,
        "total_tool_calls": float(len(tool_calls)),
        "num_fetches": float(n_fetches),
        "num_turns": float(traj.get("num_turns") or 0),
        "duration_ms": float(traj.get("duration_ms") or 0),
        "cost_usd": float(traj.get("cost_usd") or 0),
        "num_messages": float(len(msgs)),
    }


def get_chain(traj: dict) -> str:
    try:
        return rle_string(semantic_sequence(traj))
    except Exception:
        return ""


# ────────────────────────────────────────────────────────────
# Part 1 data: old vs new sonnet swapped injection
# ────────────────────────────────────────────────────────────


def get_part1_points(experiments: list[dict]) -> list[dict]:
    """Collect measured points for Part 1 swapped injection (both models)."""
    points = []
    for exp in experiments:
        if exp["environment"] != "naive" or exp["injection"] != "swapped":
            continue
        model_short = "old-sonnet" if "20250514" in exp["model"] else "new-sonnet"
        for task in exp["tasks"]:
            traj = task["trajectory"]
            flog = task["feedback_log"]
            outcome = task["outcome"]
            dims = measure_trajectory(traj, flog)
            chain = get_chain(traj)
            points.append({
                "model": model_short,
                "outcome": "held" if outcome == 1.0 else "breached",
                "group": f"{model_short} / {'held' if outcome == 1.0 else 'breached'}",
                "run_id": exp["run_id"],
                "task_index": task["task_index"],
                "chain": chain[:100],
                **dims,
            })
    return points


# ────────────────────────────────────────────────────────────
# Chart 1: Parallel Coordinates
# ────────────────────────────────────────────────────────────


def build_parallel_coords_for_experiment(
    points: list[dict],
    title: str,
    groups: list[str],
    group_colors: dict[str, str],
) -> dict:
    """Generic parallel coordinates for any experiment. Honeypot at the end."""

    # Assign numeric values for coloring (evenly spaced)
    group_numeric = {g: i / max(len(groups) - 1, 1) for i, g in enumerate(groups)}

    # Honeypot last — continuous dims in the middle for better visual spread
    dims_to_show = [
        ("explicit_refusal", "explicit<br>refusal"),
        ("injection_engaged", "injection<br>engaged"),
        ("file_read_attempt", "file read<br>attempt"),
        ("num_fetches", "num<br>fetches"),
        ("total_tool_calls", "total<br>tool calls"),
        ("num_turns", "num<br>turns"),
        ("cost_usd", "cost<br>(USD)"),
        ("honeypot_called", "honeypot<br>called"),
    ]

    dimensions = []
    for key, label in dims_to_show:
        vals = [p[key] for p in points]
        d = {"label": label, "values": vals}
        if key in ("honeypot_called", "file_read_attempt", "explicit_refusal", "injection_engaged"):
            d["range"] = [0, 1]
            d["tickvals"] = [0, 1]
            d["ticktext"] = ["no", "yes"]
        else:
            # Floor at 0, ceiling at least 1 above max to prevent collapsed axes
            vmax = max(vals) if vals else 1
            d["range"] = [0, max(vmax * 1.1, vmax + 1)]
        dimensions.append(d)

    color_vals = [group_numeric.get(p["group"], 0) for p in points]

    # Build colorscale from group_colors
    colorscale = []
    for g in groups:
        colorscale.append([group_numeric[g], group_colors[g]])

    trace = {
        "type": "parcoords",
        "line": {
            "color": color_vals,
            "colorscale": colorscale,
            "showscale": False,
        },
        "dimensions": dimensions,
    }

    # Legend annotations at the bottom
    n = len(groups)
    annotations = []
    for i, g in enumerate(groups):
        annotations.append({
            "text": f"<span style='color:{group_colors[g]}'>■</span> {g}",
            "x": i / max(n, 1),
            "y": -0.06,
            "xref": "paper", "yref": "paper",
            "showarrow": False,
            "font": {"size": 11, "color": "#f8f8f2"},
        })

    layout = {
        "title": {
            "text": title,
            "font": {"size": 14},
            "y": 0.98,
            "yanchor": "top",
        },
        "height": 700,
        "paper_bgcolor": "#282a36",
        "font": {"color": "#f8f8f2", "size": 12},
        "margin": {"t": 120, "b": 60, "l": 100, "r": 80},
        "annotations": annotations,
    }

    return {"data": [trace], "layout": layout}


def build_parallel_coords(points: list[dict]) -> dict:
    """Part 1 parallel coords: old vs new sonnet, swapped injection."""
    groups = ["old-sonnet / held", "old-sonnet / breached", "new-sonnet / held"]
    colors = {"old-sonnet / held": "#50fa7b", "old-sonnet / breached": "#ff5555", "new-sonnet / held": "#8be9fd"}
    title = (
        "Old vs New Sonnet — swapped injection"
        "<br><sup style='color:#6272a4'>Drag on any axis to filter. "
        "Try dragging \"honeypot called\" to \"yes\" to isolate breached trajectories.</sup>"
    )
    return build_parallel_coords_for_experiment(points, title, groups, colors)


# ────────────────────────────────────────────────────────────
# Chart 2: 2D Scatter on Continuous Dims
# ────────────────────────────────────────────────────────────


def build_scatter_2d(points: list[dict]) -> dict:
    """2D scatter: total_tool_calls vs cost_usd, color by group."""
    colors = {
        "old-sonnet / held": "#50fa7b",
        "old-sonnet / breached": "#ff5555",
        "new-sonnet / held": "#8be9fd",
    }
    symbols = {
        "old-sonnet / held": "circle",
        "old-sonnet / breached": "x",
        "new-sonnet / held": "diamond",
    }

    groups: dict[str, list[dict]] = {}
    for p in points:
        groups.setdefault(p["group"], []).append(p)

    traces = []
    for group_name, gpoints in sorted(groups.items()):
        traces.append({
            "type": "scatter",
            "name": group_name,
            "mode": "markers",
            "x": [p["total_tool_calls"] for p in gpoints],
            "y": [p["cost_usd"] for p in gpoints],
            "marker": {
                "size": 12,
                "color": colors.get(group_name, "#bd93f9"),
                "symbol": symbols.get(group_name, "circle"),
                "line": {"width": 1, "color": "#282a36"},
            },
            "text": [
                f"{p['group']}<br>"
                f"Tools: {int(p['total_tool_calls'])}<br>"
                f"Cost: ${p['cost_usd']:.4f}<br>"
                f"Turns: {int(p['num_turns'])}<br>"
                f"Duration: {p['duration_ms']/1000:.1f}s<br>"
                f"Chain: {p['chain'][:60]}"
                for p in gpoints
            ],
            "hoverinfo": "text+name",
        })

    layout = {
        "title": {"text": "2D Scatter: Tool Calls vs Cost (swapped injection)"},
        "xaxis": {"title": "total tool calls", "gridcolor": "#44475a"},
        "yaxis": {"title": "cost (USD)", "gridcolor": "#44475a"},
        "paper_bgcolor": "#282a36",
        "plot_bgcolor": "#282a36",
        "font": {"color": "#f8f8f2"},
        "legend": {"x": 0.02, "y": 0.98},
    }

    return {"data": traces, "layout": layout}


# ────────────────────────────────────────────────────────────
# Chart 3: Strip / Jitter Plot
# ────────────────────────────────────────────────────────────


def build_strip_plot(points: list[dict]) -> dict:
    """Strip plot: each dimension as a column, points jittered by group."""
    dims = [
        ("honeypot_called", "honeypot called"),
        ("file_read_attempt", "file read attempt"),
        ("explicit_refusal", "explicit refusal"),
        ("injection_engaged", "injection engaged"),
        ("total_tool_calls", "total tool calls"),
        ("num_turns", "num turns"),
    ]

    colors = {
        "old-sonnet / held": "#50fa7b",
        "old-sonnet / breached": "#ff5555",
        "new-sonnet / held": "#8be9fd",
    }

    groups: dict[str, list[dict]] = {}
    for p in points:
        groups.setdefault(p["group"], []).append(p)

    traces = []
    for group_name, gpoints in sorted(groups.items()):
        # For each point, create x values across all dimensions
        x_vals = []
        y_vals = []
        hover_texts = []
        for dim_key, dim_label in dims:
            for p in gpoints:
                # Jitter x position by group
                group_offset = {"old-sonnet / held": -0.15, "old-sonnet / breached": 0.0, "new-sonnet / held": 0.15}
                jitter = random.uniform(-0.05, 0.05)
                x_vals.append(dim_label)
                y_vals.append(p[dim_key])
                hover_texts.append(
                    f"{p['group']}<br>{dim_label}: {p[dim_key]:.2f}<br>"
                    f"Chain: {p['chain'][:50]}"
                )

        traces.append({
            "type": "scatter",
            "name": group_name,
            "mode": "markers",
            "x": x_vals,
            "y": y_vals,
            "marker": {
                "size": 7,
                "color": colors.get(group_name, "#bd93f9"),
                "opacity": 0.7,
                "line": {"width": 0.5, "color": "#282a36"},
            },
            "text": hover_texts,
            "hoverinfo": "text",
        })

    layout = {
        "title": {"text": "Strip Plot: Per-Dimension Distribution (swapped injection)"},
        "yaxis": {"title": "value", "gridcolor": "#44475a"},
        "paper_bgcolor": "#282a36",
        "plot_bgcolor": "#282a36",
        "font": {"color": "#f8f8f2"},
        "legend": {"x": 0.02, "y": 0.98},
    }

    return {"data": traces, "layout": layout}


# ────────────────────────────────────────────────────────────
# Chart 4: Radar / Spider Chart
# ────────────────────────────────────────────────────────────


def build_radar(points: list[dict]) -> dict:
    """Radar chart: centroid per group across normalized dimensions."""
    dims = [
        "honeypot_called", "file_read_attempt", "explicit_refusal",
        "injection_engaged", "total_tool_calls", "num_turns",
    ]
    dim_labels = [d.replace("_", " ") for d in dims]

    # Compute per-group centroids
    groups: dict[str, list[dict]] = {}
    for p in points:
        groups.setdefault(p["group"], []).append(p)

    # Find max per dimension for normalization
    maxvals = {}
    for d in dims:
        maxvals[d] = max((p[d] for p in points), default=1.0) or 1.0

    colors = {
        "old-sonnet / held": "#50fa7b",
        "old-sonnet / breached": "#ff5555",
        "new-sonnet / held": "#8be9fd",
    }

    traces = []
    for group_name, gpoints in sorted(groups.items()):
        centroid = []
        for d in dims:
            avg = sum(p[d] for p in gpoints) / len(gpoints)
            centroid.append(avg / maxvals[d])  # normalize to [0, 1]

        # Close the polygon
        r = centroid + [centroid[0]]
        theta = dim_labels + [dim_labels[0]]

        traces.append({
            "type": "scatterpolar",
            "name": group_name,
            "r": r,
            "theta": theta,
            "fill": "toself",
            "fillcolor": colors.get(group_name, "#bd93f9").replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in colors.get(group_name, "") else None,
            "opacity": 0.7,
            "line": {"color": colors.get(group_name, "#bd93f9")},
        })

    layout = {
        "title": {"text": "Radar: Behavioral Profile by Group (normalized centroids)"},
        "polar": {
            "bgcolor": "#282a36",
            "radialaxis": {"visible": True, "range": [0, 1], "gridcolor": "#44475a", "color": "#f8f8f2"},
            "angularaxis": {"gridcolor": "#44475a", "color": "#f8f8f2"},
        },
        "paper_bgcolor": "#282a36",
        "font": {"color": "#f8f8f2"},
        "legend": {"x": 0.02, "y": 0.98},
    }

    return {"data": traces, "layout": layout}


# ────────────────────────────────────────────────────────────
# Chart 5: Improved 3D Scatter (continuous axes)
# ────────────────────────────────────────────────────────────


def build_scatter3d_real(points: list[dict]) -> dict:
    """3D scatter with continuous axes: tool_calls, cost, duration."""
    colors = {
        "old-sonnet / held": "#50fa7b",
        "old-sonnet / breached": "#ff5555",
        "new-sonnet / held": "#8be9fd",
    }

    groups: dict[str, list[dict]] = {}
    for p in points:
        groups.setdefault(p["group"], []).append(p)

    traces = []
    for group_name, gpoints in sorted(groups.items()):
        traces.append({
            "type": "scatter3d",
            "name": group_name,
            "x": [p["total_tool_calls"] for p in gpoints],
            "y": [p["cost_usd"] for p in gpoints],
            "z": [p["duration_ms"] / 1000.0 for p in gpoints],
            "mode": "markers",
            "marker": {
                "size": 6,
                "color": colors.get(group_name, "#bd93f9"),
                "opacity": 0.9,
                "line": {"width": 1, "color": "#282a36"},
            },
            "text": [
                f"{p['group']}<br>"
                f"Tools: {int(p['total_tool_calls'])} | "
                f"Cost: ${p['cost_usd']:.4f} | "
                f"Duration: {p['duration_ms']/1000:.1f}s<br>"
                f"Turns: {int(p['num_turns'])}<br>"
                f"Honeypot: {'yes' if p['honeypot_called'] else 'no'} | "
                f"Refused: {'yes' if p['explicit_refusal'] else 'no'}<br>"
                f"Chain: {p['chain'][:60]}"
                for p in gpoints
            ],
            "hoverinfo": "text+name",
        })

    layout = {
        "title": {"text": "3D Scatter: Tool Calls × Cost × Duration (swapped injection)"},
        "scene": {
            "xaxis": {"title": "tool calls", "gridcolor": "#44475a"},
            "yaxis": {"title": "cost (USD)", "gridcolor": "#44475a"},
            "zaxis": {"title": "duration (s)", "gridcolor": "#44475a"},
            "bgcolor": "#282a36",
        },
        "paper_bgcolor": "#282a36",
        "font": {"color": "#f8f8f2"},
        "legend": {"x": 0.02, "y": 0.98},
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
    }

    return {"data": traces, "layout": layout}


# ────────────────────────────────────────────────────────────
# Chart 6: Cross-strategy width bar (multi_fetch)
# ────────────────────────────────────────────────────────────


def build_width_bar(experiments: list[dict]) -> dict:
    """Horizontal bar: D-Width per strategy from multi_fetch runs."""
    # One run per injection (latest, K>=5)
    by_inj: dict[str, dict] = {}
    for e in experiments:
        if e["environment"] != "multi_fetch" or "sonnet-4-6" not in e["model"] or e["K"] < 5:
            continue
        inj = e["injection"]
        if inj not in by_inj or e["created_at"] > by_inj[inj]["created_at"]:
            by_inj[inj] = e

    results = []
    for inj, exp in sorted(by_inj.items()):
        field = MultiDefenseField()
        for traj, out in zip(exp["join"]["trajectories"], exp["join"]["outcomes"]):
            field.add(traj, out)
        m = field.metrics()
        center = m.center()
        dim_names = [d.name for d in field.dimensions()]

        fetches_idx = dim_names.index("num_fetches") if "num_fetches" in dim_names else -1
        refusal_idx = dim_names.index("explicit_refusal") if "explicit_refusal" in dim_names else -1

        results.append({
            "injection": inj,
            "width": float(m.width()),
            "avg_fetches": float(center[fetches_idx]) if fetches_idx >= 0 else 0,
            "avg_refusal": float(center[refusal_idx]) if refusal_idx >= 0 else 0,
            "K": exp["K"],
        })

    results.sort(key=lambda r: r["width"])

    family_colors = {
        "progressive": "#ff79c6", "rescue": "#ff79c6",
        "breadcrumb": "#50fa7b",
        "breadcrumb_poison": "#f1fa8c", "breadcrumb_exec": "#f1fa8c",
        "url_redirect": "#8be9fd", "url_redirect_funky": "#8be9fd",
        "ipa_progressive": "#bd93f9", "ipa_exfil": "#bd93f9",
        "base64_breadcrumb": "#ffb86c",
    }

    trace = {
        "type": "bar",
        "orientation": "h",
        "y": [r["injection"] for r in results],
        "x": [r["width"] for r in results],
        "marker": {
            "color": [family_colors.get(r["injection"], "#6272a4") for r in results],
            "line": {"width": 1, "color": "#282a36"},
        },
        "text": [
            f"Width: {r['width']:.1f} | Fetches: {r['avg_fetches']:.1f} | Refusal: {r['avg_refusal']:.0%}"
            for r in results
        ],
        "hoverinfo": "text",
    }

    layout = {
        "title": {"text": "Defense Field Width by Injection Strategy (sonnet 4.6, multi-fetch)"},
        "xaxis": {"title": "D-Width (behavioral diversity)", "gridcolor": "#44475a"},
        "paper_bgcolor": "#282a36",
        "plot_bgcolor": "#282a36",
        "font": {"color": "#f8f8f2"},
        "margin": {"l": 160, "r": 20, "t": 40, "b": 40},
    }

    return {"data": [trace], "layout": layout}


# ────────────────────────────────────────────────────────────
# Per-experiment parallel coordinates
# ────────────────────────────────────────────────────────────

# Dracula-ish palette for per-run trajectories
TRAJECTORY_PALETTE = [
    "#50fa7b", "#ff79c6", "#8be9fd", "#ffb86c", "#bd93f9",
    "#f1fa8c", "#ff5555", "#6272a4", "#f8f8f2", "#44475a",
]


def measure_experiment_points(exp: dict) -> list[dict]:
    """Measure all trajectories in one experiment, labeling by outcome."""
    points = []
    for task in exp["tasks"]:
        traj = task["trajectory"]
        flog = task["feedback_log"]
        outcome = task["outcome"]
        dims = measure_trajectory(traj, flog)
        chain = get_chain(traj)
        label = "held" if outcome == 1.0 else "breached"
        idx = task["task_index"] if task["task_index"] is not None else 0
        points.append({
            "group": label,
            "outcome": label,
            "run_id": exp["run_id"],
            "task_index": idx,
            "chain": chain[:100],
            **dims,
        })
    return points


def build_per_experiment_parcoords(experiments: list[dict]) -> dict[str, dict]:
    """Build one parallel coords chart per unique experiment (latest run, K>=2)."""
    by_key: dict[tuple, dict] = {}
    for e in experiments:
        if e["K"] < 2:
            continue
        key = (e["environment"], e["injection"], e["model"])
        if key not in by_key or e["created_at"] > by_key[key]["created_at"]:
            by_key[key] = e

    charts = {}
    for (env, inj, model), exp in sorted(by_key.items()):
        points = measure_experiment_points(exp)
        if not points:
            continue

        # Group by outcome — simple held (green) vs breached (red)
        unique_groups = sorted(set(p["group"] for p in points))
        group_colors = {"held": "#50fa7b", "breached": "#ff5555"}
        groups = [g for g in ["held", "breached"] if g in unique_groups]

        model_short = "old-sonnet" if "20250514" in model else "sonnet-4.6"
        n_held = sum(1 for p in points if p["group"] == "held")
        n_breached = sum(1 for p in points if p["group"] == "breached")
        subtitle_parts = [f"{n_held} held"]
        if n_breached:
            subtitle_parts.append(f"{n_breached} breached")
        title = (
            f"{env} / {inj} — {model_short} (K={exp['K']})"
            f"<br><sup style='color:#6272a4'>{', '.join(subtitle_parts)} · Drag on any axis to filter</sup>"
        )

        safe_name = f"parcoords_{env}_{inj}_{model_short.replace('.', '')}.json"
        charts[safe_name] = build_parallel_coords_for_experiment(
            points, title, groups, group_colors,
        )

    return charts


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────


def main():
    experiments = load_dataset()
    points = get_part1_points(experiments)
    print(f"Part 1 swapped points: {len(points)} ({sum(1 for p in points if p['model']=='old-sonnet')} old, {sum(1 for p in points if p['model']=='new-sonnet')} new)")

    # Original comparison charts
    charts = {
        "parallel_coords.json": build_parallel_coords(points),
        "scatter_2d.json": build_scatter_2d(points),
        "strip_plot.json": build_strip_plot(points),
        "radar.json": build_radar(points),
        "scatter3d_real.json": build_scatter3d_real(points),
        "width_bar.json": build_width_bar(experiments),
    }

    # Per-experiment parallel coordinates
    per_exp_charts = build_per_experiment_parcoords(experiments)
    charts.update(per_exp_charts)
    print(f"Per-experiment parcoords: {len(per_exp_charts)} charts")

    os.makedirs(DATA_DIR, exist_ok=True)
    for name, spec in sorted(charts.items()):
        path = os.path.join(DATA_DIR, name)
        with open(path, "w") as f:
            json.dump(spec, f, indent=2)
        print(f"  Wrote {path}")

    # Print filenames for easy copy-paste into the test post
    print("\n--- Per-experiment chart files ---")
    for name in sorted(per_exp_charts.keys()):
        print(f"  /assets/data/{name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
