"""Viz 1: 3D Point Cloud — Old Sonnet vs New Sonnet (Act 1 hero visual).

Reads the raw dataset, filters to Part 1 swapped-injection runs (old + new sonnet),
measures each trajectory through PromptDefenseField + GeneralBehaviorField,
and outputs a Plotly-ready JSON for a 3D scatter plot.

Each point = one trajectory. Axes = 3 chosen behavioral dimensions.
Color = model version × outcome. Hover = full trajectory metadata.

Usage:
    cd aft/studies/study-2
    python blog/viz/viz_pointcloud.py

Output:
    blog/viz/pointcloud.json   — Plotly data+layout spec
"""

from __future__ import annotations

import json
import os
import sys

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, STUDY_DIR)

from defense_field import PromptDefenseField, GeneralBehaviorField
from trajectory_utils import semantic_sequence, rle_string


def load_dataset() -> list[dict]:
    path = os.path.join(STUDY_DIR, "blog", "dataset.json")
    with open(path) as f:
        return json.load(f)


def filter_part1_swapped(experiments: list[dict]) -> list[dict]:
    """Get the naive-environment swapped-injection runs for both models."""
    return [
        e for e in experiments
        if e["environment"] == "naive" and e["injection"] == "swapped"
    ]


def measure_trajectory(trajectory: dict, feedback_log: dict) -> dict:
    """Run both fields on one trajectory, return all dimension values."""
    msgs = trajectory["messages"]

    # PromptDefenseField
    defense = PromptDefenseField()
    outcome = 0.0 if feedback_log.get("total", 0) > 0 else 1.0
    d_vec = defense.measure(trajectory)
    d_dims = defense.dimensions()

    # GeneralBehaviorField
    behavior = GeneralBehaviorField()
    b_vec = behavior.measure(trajectory)
    b_dims = behavior.dimensions()

    result = {}
    for dim, val in zip(d_dims, d_vec):
        result[dim.name] = float(val)
    for dim, val in zip(b_dims, b_vec):
        result[dim.name] = float(val)

    return result


def build_plotly_data(experiments: list[dict]) -> dict:
    """Build the Plotly JSON spec for the 3D scatter plot."""

    # Collect all points with metadata
    points = []
    for exp in experiments:
        model_short = "old-sonnet" if "20250514" in exp["model"] else "new-sonnet"
        for task in exp["tasks"]:
            traj = task["trajectory"]
            flog = task["feedback_log"]
            outcome = task["outcome"]

            dims = measure_trajectory(traj, flog)

            # Semantic chain for hover
            try:
                chain = rle_string(semantic_sequence(traj))
            except Exception:
                chain = ""

            points.append({
                "model": model_short,
                "outcome": "held" if outcome == 1.0 else "breached",
                "group": f"{model_short} — {'held' if outcome == 1.0 else 'breached'}",
                "run_id": exp["run_id"],
                "task_index": task["task_index"],
                "cost_usd": traj.get("cost_usd"),
                "num_turns": traj.get("num_turns"),
                "duration_ms": traj.get("duration_ms"),
                "chain": chain,
                **dims,
            })

    # ── Choose 3 axes that maximize visual separation ──
    # x: honeypot_called (0 or 1 — breach indicator)
    # y: explicit_refusal (0 or 1 — defense indicator)
    # z: total_tool_calls (continuous — behavioral variance)
    x_dim, y_dim, z_dim = "honeypot_called", "explicit_refusal", "total_tool_calls"

    # Group by model × outcome for separate traces
    groups: dict[str, list[dict]] = {}
    for p in points:
        g = p["group"]
        groups.setdefault(g, []).append(p)

    # Color mapping
    colors = {
        "old-sonnet — held": "#50fa7b",      # green
        "old-sonnet — breached": "#ff5555",   # red
        "new-sonnet — held": "#8be9fd",       # cyan
        "new-sonnet — breached": "#ff79c6",   # pink (shouldn't appear)
    }

    traces = []
    for group_name, group_points in sorted(groups.items()):
        traces.append({
            "type": "scatter3d",
            "name": group_name,
            "x": [p[x_dim] for p in group_points],
            "y": [p[y_dim] for p in group_points],
            "z": [p[z_dim] for p in group_points],
            "mode": "markers",
            "marker": {
                "size": 8,
                "color": colors.get(group_name, "#bd93f9"),
                "opacity": 0.9,
                "line": {"width": 1, "color": "#282a36"},
            },
            "text": [
                f"Run {p['run_id']} / task {p['task_index']}<br>"
                f"Model: {p['model']}<br>"
                f"Outcome: {p['outcome']}<br>"
                f"Turns: {p['num_turns']}<br>"
                f"Cost: ${p['cost_usd']:.4f}<br>"
                f"Duration: {p['duration_ms']/1000:.1f}s<br>"
                f"Chain: {p['chain'][:80]}"
                for p in group_points
            ],
            "hoverinfo": "text+name",
        })

    layout = {
        "title": {"text": "Behavioral Point Cloud: Old vs New Sonnet (swapped injection)"},
        "scene": {
            "xaxis": {"title": x_dim.replace("_", " ")},
            "yaxis": {"title": y_dim.replace("_", " ")},
            "zaxis": {"title": z_dim.replace("_", " ")},
            "bgcolor": "#282a36",
        },
        "paper_bgcolor": "#282a36",
        "font": {"color": "#f8f8f2"},
        "legend": {"x": 0.02, "y": 0.98},
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
    }

    return {"data": traces, "layout": layout}


def main():
    experiments = load_dataset()
    part1 = filter_part1_swapped(experiments)

    if not part1:
        print("ERROR: No Part 1 swapped-injection runs found in dataset.json")
        print("Available experiments:")
        for e in experiments:
            print(f"  {e['run_id']} | {e['environment']} | {e['injection']} | {e['model']}")
        sys.exit(1)

    print(f"Found {len(part1)} Part 1 swapped runs:")
    for e in part1:
        print(f"  {e['run_id']} | {e['model']} | K={e['K']}")

    spec = build_plotly_data(part1)

    out_path = os.path.join(STUDY_DIR, "blog", "viz", "pointcloud.json")
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2)

    total_points = sum(len(t["x"]) for t in spec["data"])
    print(f"\nWrote {total_points} points across {len(spec['data'])} traces to {out_path}")


if __name__ == "__main__":
    main()
