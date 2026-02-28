"""Tutorial 4: Conditional questions — horizon separation vs global separation.

Uses Metaflow Client API to pull tutorial-3's poisoned-arm data. Reconstructs
the DataProcessingField with state(), then compares:
  - Global separation: which dimensions separate success from failure overall?
  - Conditional separation at "processing": among trajectories that reached
    the processing state, which dimensions separate success from failure?

The two vectors differ. The global view says "make more API calls." The
conditional view (given that the agent already fetched and is processing)
says "validate the data." Different question, different answer.

Usage:
  python conditional_analysis.py [--run-id METAFLOW_RUN_ID]
"""

from __future__ import annotations

import argparse
import sys
import os

import numpy as np

# Import DataProcessingField from tutorial-3
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tutorial-3"))
from data_field import DataProcessingField


def _get_latest_run(flow_name: str) -> str:
    """Get the latest successful run ID for a flow."""
    from metaflow import Flow
    flow = Flow(flow_name)
    for run in flow.runs():
        if run.successful:
            return run.id
    raise RuntimeError(f"No successful runs found for {flow_name}")


def _pull_poisoned_arm(run_id: str):
    """Pull trajectories and outcomes from the poisoned arm of EnvShapeFlow."""
    from metaflow import Run
    run = Run(f"EnvShapeFlow/{run_id}")

    # The join_poisoned step has the data
    join_step = run["join_poisoned"]
    task = next(iter(join_step))

    trajectories = task.data.trajectories
    outcomes = task.data.outcomes
    return trajectories, outcomes


def main():
    parser = argparse.ArgumentParser(
        description="Conditional questions: global vs horizon separation"
    )
    parser.add_argument("--run-id", default=None,
                        help="Metaflow run ID (default: latest EnvShapeFlow)")
    args = parser.parse_args()

    run_id = args.run_id or _get_latest_run("EnvShapeFlow")
    print(f"Using EnvShapeFlow run: {run_id}\n")

    trajectories, outcomes = _pull_poisoned_arm(run_id)

    # Build the field with state tracking
    field = DataProcessingField()
    for traj, outcome in zip(trajectories, outcomes):
        field.add(traj, outcome)

    m = field.metrics()
    dims = field.dimensions()

    # ── Global separation ──────────────────────────────────────────────
    sep_global = m.separation()

    print("=" * 60)
    print("GLOBAL SEPARATION (all trajectories)")
    print("=" * 60)
    print(f"K = {field.K}, success rate = "
          f"{float(np.sum(field.outcomes >= 0.5)):.0f}/{field.K}\n")

    for i, dim in enumerate(dims):
        bar = "+" * int(abs(sep_global[i]) * 5)
        sign = "+" if sep_global[i] >= 0 else "-"
        print(f"  {dim.name:20s}  {sep_global[i]:+.4f}  {sign}{bar}")

    # ── States observed ────────────────────────────────────────────────
    print(f"\nStates observed: {field.states}")
    for s in field.states:
        h = field.horizon(s)
        sr = h.success_region()
        fr = h.failure_region()
        print(f"  {s:15s}  K={h.K}  (success={sr.K}, failure={fr.K})")

    # ── Conditional separation at "processing" ─────────────────────────
    target_state = "processing"
    if target_state not in field.states:
        # Fall back to whatever intermediate state exists
        non_terminal = [s for s in field.states if s not in ("start", "validated")]
        target_state = non_terminal[0] if non_terminal else field.states[-1]

    h = field.horizon(target_state)

    if h.K < 2:
        print(f"\nOnly {h.K} trajectory(ies) reached '{target_state}' — "
              f"cannot compute conditional separation.")
        return

    sr = h.success_region()
    fr = h.failure_region()

    if sr.K == 0 or fr.K == 0:
        print(f"\nAll trajectories at '{target_state}' have the same outcome — "
              f"conditional separation is zero everywhere.")
        return

    hm = h.metrics()
    sep_conditional = hm.separation()

    print(f"\n{'=' * 60}")
    print(f"CONDITIONAL SEPARATION at '{target_state}'")
    print(f"  (among trajectories that reached this state)")
    print(f"{'=' * 60}")
    print(f"K = {h.K}, success rate = "
          f"{float(np.sum(h.outcomes >= 0.5)):.0f}/{h.K}\n")

    for i, dim in enumerate(dims):
        bar = "+" * int(abs(sep_conditional[i]) * 5)
        sign = "+" if sep_conditional[i] >= 0 else "-"
        print(f"  {dim.name:20s}  {sep_conditional[i]:+.4f}  {sign}{bar}")

    # ── Comparison ─────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("COMPARISON: which dimensions shift?")
    print(f"{'=' * 60}")

    delta = sep_conditional - sep_global
    print(f"\n  {'Dimension':20s}  {'Global':>10s}  {'Conditional':>12s}  {'Delta':>10s}")
    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 12}  {'-' * 10}")
    for i, dim in enumerate(dims):
        print(f"  {dim.name:20s}  {sep_global[i]:+10.4f}  "
              f"{sep_conditional[i]:+12.4f}  {delta[i]:+10.4f}")

    print("\nThe global question: 'What separates success from failure?'")
    print(f"The conditional question: 'Given the agent reached "
          f"'{target_state}', what separates success from failure?'")
    print("Different question, different answer.")


if __name__ == "__main__":
    main()
