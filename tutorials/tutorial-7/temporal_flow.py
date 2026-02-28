"""TemporalFlow: compare the field across two model periods.

DAG:
  start
  ├── setup_a → run_a (foreach K) → join_a ──┐
  └── setup_b → run_b (foreach K) → join_b ──┤
                                         compare → end

Same task (buggy.py fix), same prompt. Two arms representing two "periods":
  - Period A: model_a (default: claude-sonnet-4-5-20250514)
  - Period B: model_b (default: same model with temperature param, or different model)

The question: did the field shift between periods? If the same task produces
a different behavioral distribution with a different model, the field has
drifted. The compare step shows which dimensions shifted.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import shutil
import subprocess
import sys

import numpy as np
from metaflow import FlowSpec, Parameter, card, current, step
from metaflow import checkpoint
from metaflow.cards import Markdown, Table, VegaChart

import agent_fields as aft
from agent_fields import FieldMetrics
from agent_fields import visualisations as viz

# Import CodeFixField from tutorial-1
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tutorial-1"))
from field_def import CodeFixField

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_TUTORIAL_1_DIR = os.path.join(_SCRIPT_DIR, "..", "tutorial-1")
BUGGY_FILE = os.path.join(_TUTORIAL_1_DIR, "buggy.py")
VERIFY_FILE = os.path.join(_TUTORIAL_1_DIR, "verify.py")

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


class TemporalFlow(FlowSpec):
    """Temporal comparison: run same task on two models, compare fields."""

    K = Parameter("K", help="Number of parallel agent runs per arm", default=5)
    model_a = Parameter("model_a", help="Model ID for period A",
                        default="claude-sonnet-4-5-20250514")
    model_b = Parameter("model_b", help="Model ID for period B",
                        default="claude-haiku-4-5-20251001")

    def _run_and_verify(self, model: str):
        """Shared logic for run_a and run_b steps."""
        run_id = self.input
        task_dir = current.checkpoint.directory

        shutil.copy(BUGGY_FILE, os.path.join(task_dir, "buggy.py"))

        self.trajectory = asyncio.run(
            _run_agent(task_dir, PROMPT, model=model)
        )

        fixed_file = os.path.join(task_dir, "buggy.py")
        result = subprocess.run(
            [sys.executable, VERIFY_FILE, fixed_file],
            capture_output=True, text=True, timeout=10,
        )
        self.verification_output = result.stdout.strip()
        self.outcome = 1.0 if result.returncode == 0 else 0.0

        current.checkpoint.save(name=f"run_{run_id}")

        print(f"Run {run_id} ({model}): "
              f"{len(self.trajectory['messages'])} messages, "
              f"verification={'pass' if self.outcome == 1.0 else 'fail'}")
        if result.returncode != 0:
            print(f"  {self.verification_output}")

    @staticmethod
    def _build_field(inputs):
        """Build field from foreach inputs."""
        field = CodeFixField()
        for inp in inputs:
            field.add(inp.trajectory, inp.outcome)
        return field, field.metrics()

    @step
    def start(self):
        """Branch into period A and period B arms."""
        self.next(self.setup_a, self.setup_b)

    # ── Period A arm ──────────────────────────────────────────────────

    @step
    def setup_a(self):
        """Prepare foreach for period A."""
        self.run_ids = list(range(self.K))
        self.next(self.run_a, foreach="run_ids")

    @checkpoint
    @step
    def run_a(self):
        """Run one agent with model A."""
        self._run_and_verify(self.model_a)
        self.next(self.join_a)

    @step
    def join_a(self, inputs):
        """Build field for period A."""
        field, m = self._build_field(inputs)
        self.period_label = "period_a"
        self.model_used = self.model_a
        self.field_summary = m.summary()
        self.field_points = field.points.tolist()
        self.field_outcomes = field.outcomes.tolist()
        self.outcomes = [inp.outcome for inp in inputs]

        print(f"\n[PERIOD A: {self.model_a}] K={field.K}, d={field.d}")
        print(f"  Width:       {m.width():.4f}")
        print(f"  Convergence: {m.convergence():.4f}")
        print(f"  Success:     {sum(self.outcomes):.0f}/{len(self.outcomes)}")

        self.next(self.compare)

    # ── Period B arm ──────────────────────────────────────────────────

    @step
    def setup_b(self):
        """Prepare foreach for period B."""
        self.run_ids = list(range(self.K))
        self.next(self.run_b, foreach="run_ids")

    @checkpoint
    @step
    def run_b(self):
        """Run one agent with model B."""
        self._run_and_verify(self.model_b)
        self.next(self.join_b)

    @step
    def join_b(self, inputs):
        """Build field for period B."""
        field, m = self._build_field(inputs)
        self.period_label = "period_b"
        self.model_used = self.model_b
        self.field_summary = m.summary()
        self.field_points = field.points.tolist()
        self.field_outcomes = field.outcomes.tolist()
        self.outcomes = [inp.outcome for inp in inputs]

        print(f"\n[PERIOD B: {self.model_b}] K={field.K}, d={field.d}")
        print(f"  Width:       {m.width():.4f}")
        print(f"  Convergence: {m.convergence():.4f}")
        print(f"  Success:     {sum(self.outcomes):.0f}/{len(self.outcomes)}")

        self.next(self.compare)

    # ── Comparative join ──────────────────────────────────────────────

    @card(type="blank", id="temporal_report")
    @step
    def compare(self, inputs):
        """Join both periods, build comparative visualizations."""
        arms = {}
        for inp in inputs:
            arms[inp.period_label] = inp
        period_a = arms["period_a"]
        period_b = arms["period_b"]

        dims = CodeFixField().dimensions()
        m_a = FieldMetrics(
            np.array(period_a.field_points),
            np.array(period_a.field_outcomes),
            dims,
        )
        m_b = FieldMetrics(
            np.array(period_b.field_points),
            np.array(period_b.field_outcomes),
            dims,
        )

        self.summary_a = period_a.field_summary
        self.summary_b = period_b.field_summary

        # ── Card ──────────────────────────────────────────────────────
        c = current.card["temporal_report"]
        K = len(period_a.outcomes)

        c.append(Markdown(
            f"# Temporal Comparison Report\n"
            f"**Period A:** `{period_a.model_used}` | "
            f"**Period B:** `{period_b.model_used}` | "
            f"**K:** {K} per period\n\n"
            f"Same task, same prompt. Different model. "
            f"Has the field shifted?"
        ))

        def _fmt_conv(val):
            return "∞" if val == float("inf") else f"{val:.4f}"

        c.append(Markdown("## Scalar Metrics"))
        c.append(Table(
            headers=["Metric", "Period A", "Period B"],
            data=[
                ["Width",
                 f"{m_a.width():.4f}",
                 f"{m_b.width():.4f}"],
                ["Convergence",
                 _fmt_conv(m_a.convergence()),
                 _fmt_conv(m_b.convergence())],
                ["Success rate",
                 f"{sum(period_a.outcomes):.0f}/{len(period_a.outcomes)}",
                 f"{sum(period_b.outcomes):.0f}/{len(period_b.outcomes)}"],
            ],
        ))

        label_a = f"A ({period_a.model_used})"
        label_b = f"B ({period_b.model_used})"

        c.append(Markdown("## Mean Behavior"))
        c.append(VegaChart(viz.compare_center_bar(m_a, m_b, label_a, label_b)))

        c.append(Markdown("## Variance"))
        c.append(VegaChart(viz.compare_variance_bar(m_a, m_b, label_a, label_b)))

        c.append(Markdown(f"## Separation Vector — Period A"))
        c.append(VegaChart(viz.separation_bar(m_a)))

        c.append(Markdown(f"## Separation Vector — Period B"))
        c.append(VegaChart(viz.separation_bar(m_b)))

        c.append(Markdown("## Skew — Period A"))
        c.append(VegaChart(viz.skew_bar(m_a)))

        c.append(Markdown("## Skew — Period B"))
        c.append(VegaChart(viz.skew_bar(m_b)))

        self.next(self.end)

    @step
    def end(self):
        """Done."""
        print("Temporal comparison complete.")


if __name__ == "__main__":
    TemporalFlow()
