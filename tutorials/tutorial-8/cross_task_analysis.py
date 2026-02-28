"""Tutorial 8: Comparing across tasks.

Pulls trajectories from tutorial-1 (GelloFlow, bug-fix task) and tutorial-3
(EnvShapeFlow, data-processing task) via Metaflow Client API.

Builds a TaskAgnosticField for each. Compares which dimensions are stable
(agent signature — same regardless of task) vs which shift (task-dependent
behavior).

Usage:
  python cross_task_analysis.py [--run-id-1 ID] [--run-id-3 ID]
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

from cross_task_field import TaskAgnosticField


def _get_latest_run(flow_name: str) -> str:
    from metaflow import Flow
    flow = Flow(flow_name)
    for run in flow.runs():
        if run.successful:
            return run.id
    raise RuntimeError(f"No successful runs found for {flow_name}")


def _pull_gello_trajectories(run_id: str):
    """Pull all trajectories from GelloFlow (tutorial-1)."""
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


def _pull_env_trajectories(run_id: str):
    """Pull all trajectories from EnvShapeFlow (tutorial-3)."""
    from metaflow import Run
    run = Run(f"EnvShapeFlow/{run_id}")

    trajectories = []
    outcomes = []
    for arm in ("run_clean", "run_poisoned"):
        step = run[arm]
        for task in step:
            trajectories.append(task.data.trajectory)
            outcomes.append(task.data.outcome)
    return trajectories, outcomes


def _build_field(trajectories, outcomes):
    field = TaskAgnosticField()
    for traj, outcome in zip(trajectories, outcomes):
        field.add(traj, outcome)
    return field


def main():
    parser = argparse.ArgumentParser(
        description="Cross-task comparison: agent signature vs task-dependent behavior"
    )
    parser.add_argument("--run-id-1", default=None,
                        help="GelloFlow run ID (default: latest)")
    parser.add_argument("--run-id-3", default=None,
                        help="EnvShapeFlow run ID (default: latest)")
    args = parser.parse_args()

    run_id_1 = args.run_id_1 or _get_latest_run("GelloFlow")
    run_id_3 = args.run_id_3 or _get_latest_run("EnvShapeFlow")

    print(f"Bug-fix task (GelloFlow):        run {run_id_1}")
    print(f"Data-processing task (EnvShapeFlow): run {run_id_3}")

    trajs_1, outcomes_1 = _pull_gello_trajectories(run_id_1)
    trajs_3, outcomes_3 = _pull_env_trajectories(run_id_3)

    field_bugfix = _build_field(trajs_1, outcomes_1)
    field_data = _build_field(trajs_3, outcomes_3)

    m1 = field_bugfix.metrics()
    m3 = field_data.metrics()
    dims = TaskAgnosticField().dimensions()

    center_1 = m1.center()
    center_3 = m3.center()
    var_1 = m1.variance()
    var_3 = m3.variance()

    # ── Print comparison ───────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("TASK-AGNOSTIC FIELD COMPARISON")
    print(f"{'=' * 72}")
    print(f"\n  {'':25s}  {'Bug-fix':>12s}  {'Data-proc':>12s}  {'Delta':>10s}  {'Signal':>10s}")
    print(f"  {'-' * 25}  {'-' * 12}  {'-' * 12}  {'-' * 10}  {'-' * 10}")

    for i, dim in enumerate(dims):
        delta = abs(center_1[i] - center_3[i])
        # Normalize delta by pooled std to determine significance
        pooled_std = np.sqrt((var_1[i] + var_3[i]) / 2) or 1.0
        effect = delta / pooled_std

        if effect < 0.5:
            signal = "STABLE"
        elif effect < 1.0:
            signal = "shifting"
        else:
            signal = "TASK-DEP"

        print(f"  {dim.name:25s}  {center_1[i]:12.3f}  {center_3[i]:12.3f}  "
              f"{delta:10.3f}  {signal:>10s}")

    # ── Width and convergence ──────────────────────────────────────────
    def _fmt_conv(val):
        return "inf" if val == float("inf") else f"{val:.4f}"

    print(f"\n  {'Scalar':20s}  {'Bug-fix':>12s}  {'Data-proc':>12s}")
    print(f"  {'-' * 20}  {'-' * 12}  {'-' * 12}")
    print(f"  {'Width':20s}  {m1.width():12.4f}  {m3.width():12.4f}")
    print(f"  {'Convergence':20s}  {_fmt_conv(m1.convergence()):>12s}  "
          f"{_fmt_conv(m3.convergence()):>12s}")
    print(f"  {'K':20s}  {field_bugfix.K:12d}  {field_data.K:12d}")

    # ── Interpretation ─────────────────────────────────────────────────
    stable_dims = []
    task_dep_dims = []
    for i, dim in enumerate(dims):
        delta = abs(center_1[i] - center_3[i])
        pooled_std = np.sqrt((var_1[i] + var_3[i]) / 2) or 1.0
        effect = delta / pooled_std
        if effect < 0.5:
            stable_dims.append(dim.name)
        elif effect >= 1.0:
            task_dep_dims.append(dim.name)

    print(f"\n{'=' * 72}")
    print("INTERPRETATION")
    print(f"{'=' * 72}")
    if stable_dims:
        print(f"\nStable dimensions (agent signature):")
        for d in stable_dims:
            print(f"  - {d}")
    if task_dep_dims:
        print(f"\nTask-dependent dimensions:")
        for d in task_dep_dims:
            print(f"  - {d}")

    print("\nStable dimensions are the agent's behavioral fingerprint.")
    print("Task-dependent dimensions respond to what the agent is asked to do.")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
