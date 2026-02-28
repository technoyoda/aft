"""HorizonFlow: diagnose *where* trajectories diverge using horizon analysis.

DAG:
  start → setup → run_agent (foreach K) → join → analyze → end

Uses the vague prompt (which produces mixed outcomes — exactly what makes
horizon analysis interesting). The analyze step walks the horizon chain
to find the phase where trajectories diverge and what separates success
from failure at that phase.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import subprocess

import numpy as np
from metaflow import FlowSpec, Parameter, card, current, step
from metaflow import checkpoint
from metaflow.cards import Markdown, Table, VegaChart

import agent_fields as aft
from agent_fields import FieldMetrics
from agent_fields import visualisations as viz
from horizon_field import HorizonCodeFixField

# Static files that live next to this script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUGGY_FILE = os.path.join(_SCRIPT_DIR, "buggy.py")
VERIFY_FILE = os.path.join(_SCRIPT_DIR, "verify.py")

PROMPT = (
    "Read buggy.py in this directory. Find all bugs that would cause crashes "
    "or exceptions, and fix them. Then run the file to verify your fixes work."
)


# ═══════════════════════════════════════════════════════════════════════
# Agent runner
# ═══════════════════════════════════════════════════════════════════════


async def _run_agent(task_dir: str, prompt: str, model: str | None = None) -> dict:
    """Run the Claude agent once. Returns a plain dict trajectory."""
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        query,
    )

    messages = []

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Edit", "Glob", "Grep", "Bash"],
            permission_mode="bypassPermissions",
            cwd=task_dir,
            max_turns=15,
            model=model,
        ),
    ):
        messages.append(dataclasses.asdict(message))

    # The last message is always the ResultMessage
    result = messages[-1] if messages else {}

    return {
        "messages": messages,
        "success": result.get("subtype") == "success",
        "cost_usd": result.get("total_cost_usd"),
        "num_turns": result.get("num_turns"),
        "duration_ms": result.get("duration_ms"),
        "task_dir": task_dir,
    }


# ═══════════════════════════════════════════════════════════════════════
# The Metaflow flow
# ═══════════════════════════════════════════════════════════════════════


class HorizonFlow(FlowSpec):
    """Horizon analysis: run K agents, walk the horizon chain to diagnose."""

    K = Parameter("K", help="Number of parallel agent runs", default=10)
    model = Parameter("model", help="Model ID", default=None)

    @step
    def start(self):
        """Entry point."""
        self.next(self.setup)

    @step
    def setup(self):
        """Prepare foreach."""
        self.run_ids = list(range(self.K))
        self.next(self.run_agent, foreach="run_ids")

    @checkpoint
    @step
    def run_agent(self):
        """Run one agent with the vague prompt."""
        run_id = self.input
        task_dir = current.checkpoint.directory

        shutil.copy(BUGGY_FILE, os.path.join(task_dir, "buggy.py"))

        self.trajectory = asyncio.run(
            _run_agent(task_dir, PROMPT, model=self.model)
        )

        fixed_file = os.path.join(task_dir, "buggy.py")
        result = subprocess.run(
            ["python", VERIFY_FILE, fixed_file],
            capture_output=True, text=True, timeout=10,
        )
        self.verification_output = result.stdout.strip()
        self.outcome = 1.0 if result.returncode == 0 else 0.0

        current.checkpoint.save(name=f"run_{run_id}")

        print(f"Run {run_id}: "
              f"{len(self.trajectory['messages'])} messages, "
              f"verification={'pass' if self.outcome == 1.0 else 'fail'}")
        if result.returncode != 0:
            print(f"  {self.verification_output}")

        self.next(self.join)

    @step
    def join(self, inputs):
        """Build HorizonCodeFixField from all trajectories."""
        field = HorizonCodeFixField()
        for inp in inputs:
            field.add(inp.trajectory, inp.outcome)

        m = field.metrics()
        self.field_summary = m.summary()
        self.field_points = field.points.tolist()
        self.field_outcomes = field.outcomes.tolist()
        self.trajectories = [inp.trajectory for inp in inputs]
        self.outcomes = [inp.outcome for inp in inputs]

        print(f"\nK={field.K}, d={field.d}")
        print(f"  Width:       {m.width():.4f}")
        print(f"  Convergence: {m.convergence():.4f}")
        print(f"  Success:     {sum(self.outcomes):.0f}/{len(self.outcomes)}")
        print(f"  States:      {field.states}")

        self.next(self.analyze)

    @card(type="blank", id="horizon_report")
    @step
    def analyze(self):
        """Render the horizon analysis card."""
        # Reconstruct field from stored data
        field = HorizonCodeFixField()
        for traj, outcome in zip(self.trajectories, self.outcomes):
            field.add(traj, outcome)

        m = field.metrics()
        c = current.card["horizon_report"]
        model_label = self.model or "default"

        # ── Header ────────────────────────────────────────────────────
        c.append(Markdown(
            f"# Horizon Analysis Report\n"
            f"**Model:** `{model_label}` | **K:** {field.K}"
        ))

        # ── Overall field metrics ─────────────────────────────────────
        def _fmt_conv(val):
            return "∞" if val == float("inf") else f"{val:.4f}"

        c.append(Markdown("## Overall Field Metrics"))
        c.append(Table(
            headers=["Metric", "Value"],
            data=[
                ["Width", f"{m.width():.4f}"],
                ["Convergence", _fmt_conv(m.convergence())],
                ["Success rate",
                 f"{sum(self.outcomes):.0f}/{len(self.outcomes)}"],
            ],
        ))

        # ── States observed ───────────────────────────────────────────
        c.append(Markdown("## States Observed"))
        state_rows = []
        for s in field.states:
            h = field.horizon(s)
            hm = h.metrics() if h.K >= 2 else None
            state_rows.append([
                s,
                str(h.K),
                f"{hm.width():.4f}" if hm else "—",
                _fmt_conv(hm.convergence()) if hm else "—",
            ])
        c.append(Table(
            headers=["State", "K", "Width", "Convergence"],
            data=state_rows,
        ))

        # ── Horizon width chart ───────────────────────────────────────
        c.append(Markdown("## Horizon Width by State"))
        c.append(VegaChart(viz.horizon_width(field)))

        # ── Horizon convergence chart ─────────────────────────────────
        c.append(Markdown("## Horizon Convergence by State"))
        c.append(VegaChart(viz.horizon_convergence(field)))

        # ── Horizon drift chart ───────────────────────────────────────
        c.append(Markdown("## Horizon Drift by State"))
        c.append(Markdown(
            "*Drift δ(s) = W(H(s)) − W(H⁺(s)). Positive means failing "
            "trajectories have diverged from the success corridor at that state.*"
        ))
        c.append(VegaChart(viz.horizon_drift(field)))

        # ── Find critical state (highest drift) ──────────────────────
        best_drift = -float("inf")
        critical_state = None
        for s in field.states:
            h = field.horizon(s)
            if h.K < 2:
                continue
            W_all = h.metrics().width()
            sr = h.success_region()
            W_success = sr.metrics().width() if sr.K >= 2 else W_all
            drift = W_all - W_success
            if drift > best_drift:
                best_drift = drift
                critical_state = s

        if critical_state:
            h_crit = field.horizon(critical_state)
            if h_crit.K >= 2:
                hm_crit = h_crit.metrics()
                c.append(Markdown(
                    f"## Separation at Critical State: `{critical_state}`\n"
                    f"*This is where drift is highest (δ = {best_drift:.4f}) — "
                    f"failing trajectories diverge most from successes here.*"
                ))
                c.append(VegaChart(viz.separation_bar(hm_crit)))

        # ── Per-state separation ──────────────────────────────────────
        c.append(Markdown("## Per-State Separation"))
        for s in field.states:
            h = field.horizon(s)
            sr = h.success_region()
            fr = h.failure_region()
            if sr.K >= 1 and fr.K >= 1 and h.K >= 2:
                c.append(Markdown(f"### `{s}` (K={h.K})"))
                c.append(VegaChart(viz.separation_bar(h.metrics())))

        # ── Overall separation and skew ───────────────────────────────
        c.append(Markdown("## Overall Separation Vector"))
        c.append(VegaChart(viz.separation_bar(m)))

        c.append(Markdown("## Overall Skew"))
        c.append(VegaChart(viz.skew_bar(m)))

        self.next(self.end)

    @step
    def end(self):
        """Done."""
        print("Horizon analysis complete.")


if __name__ == "__main__":
    HorizonFlow()
