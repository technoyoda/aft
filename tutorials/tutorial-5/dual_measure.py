"""Tutorial 5: measure() is your hypothesis.

Pulls raw trajectories from tutorial-1 (GelloFlow) via Metaflow Client API.
Defines two Field subclasses that measure the SAME trajectories differently:

  - StructuralField: what the agent did (tool counts, scope, bugs addressed)
  - StrategyField: how the agent approached it (exploration ratio, commit speed,
    direction changes, verification effort)

Same data, different fields, different insights. The measure function is your
hypothesis about what matters. Change it and the field changes.

Usage:
  python dual_measure.py [--run-id METAFLOW_RUN_ID]
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

import agent_fields as aft
from agent_fields import Dimension, Field

# Import CodeFixField for reference dimensions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tutorial-1"))
from field_def import _extract_tool_calls


# ═══════════════════════════════════════════════════════════════════════
# Two Fields, same trajectory type
# ═══════════════════════════════════════════════════════════════════════


class StructuralField(Field[dict]):
    """What the agent did — counts and ratios of actions."""

    def dimensions(self) -> list[Dimension]:
        return [
            Dimension("tool_calls", "Total tool invocations"),
            Dimension("reads", "File read operations"),
            Dimension("edits", "File edit operations"),
            Dimension("scope_ratio", "Fraction of file ops on buggy.py"),
            Dimension("bugs_addressed", "Known bugs touched in edits"),
        ]

    def measure(self, trajectory: dict) -> np.ndarray:
        tool_calls = _extract_tool_calls(trajectory)
        reads = sum(1 for tc in tool_calls if tc["name"] == "Read")
        edits = sum(1 for tc in tool_calls if tc["name"] == "Edit")

        on_target = 0
        file_calls = 0
        for tc in tool_calls:
            inp = tc.get("input") or {}
            path = str(inp.get("file_path", "") or inp.get("path", ""))
            if path:
                file_calls += 1
                if "buggy" in path:
                    on_target += 1
        scope_ratio = on_target / file_calls if file_calls else 0.0

        bugs = 0
        for tc in tool_calls:
            if tc["name"] == "Edit":
                inp = tc.get("input") or {}
                text = str(inp.get("old_string", "")) + str(inp.get("new_string", ""))
                for bug in ["divide", "average", "first_element"]:
                    if bug in text:
                        bugs += 1
                        break

        return np.array([
            len(tool_calls), reads, edits, scope_ratio, bugs,
        ], dtype=float)


class StrategyField(Field[dict]):
    """How the agent approached it — behavioral strategy."""

    def dimensions(self) -> list[Dimension]:
        return [
            Dimension("exploration_ratio", "Reads / total tool calls"),
            Dimension("commit_speed", "Normalized position of first edit (0=early, 1=late)"),
            Dimension("direction_changes", "Read-to-edit switches (hesitation signal)"),
            Dimension("verification_effort", "Bash calls / total tool calls"),
            Dimension("message_count", "Total messages (conversation length)"),
        ]

    def measure(self, trajectory: dict) -> np.ndarray:
        tool_calls = _extract_tool_calls(trajectory)
        n = len(tool_calls) or 1

        reads = sum(1 for tc in tool_calls if tc["name"] == "Read")
        exploration_ratio = reads / n

        # Commit speed: position of first Edit
        first_edit_pos = n
        for i, tc in enumerate(tool_calls):
            if tc["name"] == "Edit":
                first_edit_pos = i
                break
        commit_speed = first_edit_pos / n

        # Direction changes: transitions between Read and Edit
        direction_changes = 0
        last_type = None
        for tc in tool_calls:
            curr_type = "read" if tc["name"] == "Read" else (
                "edit" if tc["name"] == "Edit" else None)
            if curr_type and last_type and curr_type != last_type:
                direction_changes += 1
            if curr_type:
                last_type = curr_type

        # Verification effort: Bash calls (running tests)
        bash_calls = sum(1 for tc in tool_calls if tc["name"] == "Bash")
        verification_effort = bash_calls / n

        return np.array([
            exploration_ratio,
            commit_speed,
            direction_changes,
            verification_effort,
            len(trajectory["messages"]),
        ], dtype=float)


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════


def _get_latest_run(flow_name: str) -> str:
    from metaflow import Flow
    flow = Flow(flow_name)
    for run in flow.runs():
        if run.successful:
            return run.id
    raise RuntimeError(f"No successful runs found for {flow_name}")


def _pull_both_arms(run_id: str):
    """Pull trajectories and outcomes from both arms of GelloFlow."""
    from metaflow import Run
    run = Run(f"GelloFlow/{run_id}")

    all_trajectories = []
    all_outcomes = []

    for arm in ("join_vague", "join_concise"):
        step = run[arm]
        task = next(iter(step))
        points = task.data.field_points
        outcomes = task.data.field_outcomes
        # We need the raw trajectories — pull from the run steps
        all_outcomes.extend(outcomes)

    # Pull raw trajectories from the foreach steps
    for arm in ("run_vague", "run_concise"):
        step = run[arm]
        for task in step:
            all_trajectories.append(task.data.trajectory)

    # Outcomes come from join steps (already ordered)
    return all_trajectories, all_outcomes


def _print_field_summary(name: str, field: Field, dims: list[Dimension]):
    m = field.metrics()
    sep = m.separation()
    var = m.variance()

    def _fmt_conv(val):
        return "inf" if val == float("inf") else f"{val:.4f}"

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  K={field.K}  Width={m.width():.4f}  "
          f"Convergence={_fmt_conv(m.convergence())}")

    print(f"\n  {'Dimension':25s}  {'Variance':>10s}  {'Separation':>12s}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 12}")
    for i, dim in enumerate(dims):
        print(f"  {dim.name:25s}  {var[i]:10.4f}  {sep[i]:+12.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="measure() is your hypothesis: two fields on same data"
    )
    parser.add_argument("--run-id", default=None,
                        help="Metaflow run ID (default: latest GelloFlow)")
    args = parser.parse_args()

    run_id = args.run_id or _get_latest_run("GelloFlow")
    print(f"Using GelloFlow run: {run_id}")

    trajectories, outcomes = _pull_both_arms(run_id)
    print(f"Loaded {len(trajectories)} trajectories\n")

    # Build both fields from the same data
    structural = StructuralField()
    strategy = StrategyField()

    for traj, outcome in zip(trajectories, outcomes):
        structural.add(traj, outcome)
        strategy.add(traj, outcome)

    _print_field_summary("STRUCTURAL FIELD (what the agent did)",
                         structural, structural.dimensions())
    _print_field_summary("STRATEGY FIELD (how the agent approached it)",
                         strategy, strategy.dimensions())

    # Final message
    print(f"\n{'=' * 60}")
    print("The same trajectories. Two different fields.")
    print("Each field highlights different signals from identical data.")
    print("measure() is the hypothesis. Change it and the answer changes.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
