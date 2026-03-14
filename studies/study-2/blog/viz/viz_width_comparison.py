"""Viz 2: Cross-Strategy D-Width Bar Chart (Act 7 hero chart).

Reads the raw dataset, filters to multi_fetch sonnet-4.6 runs (one per strategy),
measures each through the appropriate field, computes width, and outputs
a Plotly-ready JSON for a horizontal bar chart sorted by D-Width.

Usage:
    cd aft/studies/study-2
    python blog/viz/viz_width_comparison.py

Output:
    blog/viz/width_comparison.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, STUDY_DIR)

from defense_field import MultiDefenseField


def load_dataset() -> list[dict]:
    path = os.path.join(STUDY_DIR, "blog", "dataset.json")
    with open(path) as f:
        return json.load(f)


def filter_multi_fetch_sonnet46(experiments: list[dict]) -> list[dict]:
    """Get multi_fetch runs on sonnet 4.6, one per injection strategy (latest)."""
    # Group by injection, keep latest run per injection
    by_injection: dict[str, dict] = {}
    for e in experiments:
        if e["environment"] != "multi_fetch":
            continue
        if "sonnet-4-6" not in e["model"]:
            continue
        if e["K"] < 5:
            continue  # skip test runs
        inj = e["injection"]
        if inj not in by_injection or e["created_at"] > by_injection[inj]["created_at"]:
            by_injection[inj] = e
    return list(by_injection.values())


def compute_defense_width(experiment: dict) -> dict:
    """Compute MultiDefenseField width for one experiment."""
    field = MultiDefenseField()
    trajectories = experiment["join"]["trajectories"]
    feedback_logs = experiment["join"]["feedback_logs"]
    outcomes = experiment["join"]["outcomes"]

    for traj, flog, out in zip(trajectories, feedback_logs, outcomes):
        field.add(traj, out)

    m = field.metrics()
    return {
        "injection": experiment["injection"],
        "width": float(m.width()),
        "center": [float(v) for v in m.center()],
        "K": experiment["K"],
        "held": experiment.get("held", sum(1 for o in outcomes if o == 1.0)),
        "breached": experiment.get("breached", sum(1 for o in outcomes if o != 1.0)),
        "avg_fetches": float(np.mean([
            float(c) for c, d in zip(m.center(), field.dimensions())
            if d.name == "num_fetches"
        ])) if any(d.name == "num_fetches" for d in field.dimensions()) else 0,
        "avg_refusal": float(np.mean([
            float(c) for c, d in zip(m.center(), field.dimensions())
            if d.name == "explicit_refusal"
        ])) if any(d.name == "explicit_refusal" for d in field.dimensions()) else 0,
    }


def build_plotly_data(results: list[dict]) -> dict:
    """Build Plotly JSON for horizontal bar chart."""
    # Sort by width ascending (so largest is at top in horizontal bar)
    results.sort(key=lambda r: r["width"])

    # Strategy family colors
    family_colors = {
        "progressive": "#ff79c6",
        "rescue": "#ff79c6",
        "breadcrumb": "#50fa7b",
        "breadcrumb_poison": "#f1fa8c",
        "breadcrumb_exec": "#f1fa8c",
        "url_redirect": "#8be9fd",
        "url_redirect_funky": "#8be9fd",
        "ipa_progressive": "#bd93f9",
        "ipa_exfil": "#bd93f9",
        "base64_breadcrumb": "#ffb86c",
    }

    traces = [{
        "type": "bar",
        "orientation": "h",
        "y": [r["injection"] for r in results],
        "x": [r["width"] for r in results],
        "marker": {
            "color": [family_colors.get(r["injection"], "#6272a4") for r in results],
            "line": {"width": 1, "color": "#282a36"},
        },
        "text": [
            f"Width: {r['width']:.1f} | "
            f"Fetches: {r['avg_fetches']:.1f} | "
            f"Refusal: {r['avg_refusal']:.0%} | "
            f"{r['held']}/{r['held']+r['breached']} held"
            for r in results
        ],
        "hoverinfo": "text",
    }]

    layout = {
        "title": {"text": "Defense Field Width by Injection Strategy (sonnet 4.6)"},
        "xaxis": {"title": "D-Width (behavioral diversity under attack)"},
        "yaxis": {"title": ""},
        "paper_bgcolor": "#282a36",
        "plot_bgcolor": "#282a36",
        "font": {"color": "#f8f8f2"},
        "margin": {"l": 160, "r": 20, "t": 40, "b": 40},
    }

    return {"data": traces, "layout": layout}


def main():
    experiments = load_dataset()
    multi_fetch = filter_multi_fetch_sonnet46(experiments)

    if not multi_fetch:
        print("ERROR: No multi_fetch sonnet-4.6 runs found in dataset.json")
        sys.exit(1)

    print(f"Computing field width for {len(multi_fetch)} strategies:")

    results = []
    for exp in sorted(multi_fetch, key=lambda e: e["injection"]):
        r = compute_defense_width(exp)
        results.append(r)
        print(f"  {r['injection']:20s} | width={r['width']:.3f} | K={r['K']}")

    spec = build_plotly_data(results)

    out_path = os.path.join(STUDY_DIR, "blog", "viz", "width_comparison.json")
    with open(out_path, "w") as f:
        json.dump(spec, f, indent=2)

    # Also dump the raw results table for the blog
    table_path = os.path.join(STUDY_DIR, "blog", "viz", "width_comparison_table.json")
    with open(table_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
