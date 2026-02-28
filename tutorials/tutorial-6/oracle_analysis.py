"""Tutorial 6: The oracle as a ruler.

Pulls trajectories from tutorial-1 (GelloFlow) via Client API.
Runs OracleField (LLM-scored) and CodeFixField (hand-crafted) on the same
trajectories. Compares metrics side by side.

Usage:
  python oracle_analysis.py [--run-id METAFLOW_RUN_ID]
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

import agent_fields as aft
from agent_fields import Field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tutorial-1"))
from field_def import CodeFixField

from oracle_field import OracleField


def _get_latest_run(flow_name: str) -> str:
    from metaflow import Flow
    flow = Flow(flow_name)
    for run in flow.runs():
        if run.successful:
            return run.id
    raise RuntimeError(f"No successful runs found for {flow_name}")


def _pull_trajectories(run_id: str):
    """Pull all trajectories and outcomes from GelloFlow."""
    from metaflow import Run
    run = Run(f"GelloFlow/{run_id}")

    trajectories = []
    outcomes = []

    for arm in ("run_vague", "run_concise"):
        step = run[arm]
        for task in step:
            trajectories.append(task.data.trajectory)
            outcomes.append(task.data.outcome)

    return trajectories, outcomes


def _print_comparison(name: str, field: Field):
    m = field.metrics()
    dims = field.dimensions()
    sep = m.separation()
    var = m.variance()

    def _fmt_conv(val):
        return "inf" if val == float("inf") else f"{val:.4f}"

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  K={field.K}  Width={m.width():.4f}  "
          f"Convergence={_fmt_conv(m.convergence())}")

    print(f"\n  {'Dimension':25s}  {'Variance':>10s}  {'Separation':>12s}  {'Skew':>8s}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 12}  {'-' * 8}")
    for i, dim in enumerate(dims):
        skew = m.skew(i)
        print(f"  {dim.name:25s}  {var[i]:10.4f}  {sep[i]:+12.4f}  {skew:+8.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="The oracle as a ruler: LLM-scored vs hand-crafted fields"
    )
    parser.add_argument("--run-id", default=None,
                        help="Metaflow run ID (default: latest GelloFlow)")
    args = parser.parse_args()

    run_id = args.run_id or _get_latest_run("GelloFlow")
    print(f"Using GelloFlow run: {run_id}")

    trajectories, outcomes = _pull_trajectories(run_id)
    print(f"Loaded {len(trajectories)} trajectories\n")

    # Build both fields
    hand_crafted = CodeFixField()
    oracle = OracleField()

    print("Scoring trajectories with OracleField (LLM calls, cached)...")
    for traj, outcome in zip(trajectories, outcomes):
        hand_crafted.add(traj, outcome)
        oracle.add(traj, outcome)

    _print_comparison("HAND-CRAFTED FIELD (CodeFixField)", hand_crafted)
    _print_comparison("ORACLE FIELD (LLM-scored)", oracle)


if __name__ == "__main__":
    main()
