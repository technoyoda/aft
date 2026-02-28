"""GelloFlow: prompt ablation — run K agents on two prompts, compare fields.

DAG:
  start
  ├── setup_vague → run_vague (foreach K) → join_vague ────────┐
  └── setup_concise → run_concise (foreach K) → join_concise ──┤
                                                            compare → end

The vague prompt says "fix bugs". The concise prompt spells out exactly
what to fix and how. Both use the same model, same K, same task.
The compare step renders a comparative card.
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
from field_def import CodeFixField

# Static files that live next to this script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUGGY_FILE = os.path.join(_SCRIPT_DIR, "buggy.py")
VERIFY_FILE = os.path.join(_SCRIPT_DIR, "verify.py")

PROMPT_VAGUE = (
    "Read buggy.py in this directory. Find all bugs that would cause crashes "
    "or exceptions, and fix them. Then run the file to verify your fixes work."
)

PROMPT_CONCISE = (
    "Read buggy.py in this directory. There are three functions with bugs:\n"
    "1. divide(a, b) — crashes on b=0. Fix it to return 0 when b is 0.\n"
    "2. average(numbers) — crashes on empty list. Fix it to return 0 for [].\n"
    "3. first_element(lst) — crashes on empty list. Fix it to return None for [].\n"
    "Keep the normal behavior intact. Then run the file to verify."
)


# ═══════════════════════════════════════════════════════════════════════
# Agent runner — captures everything as dicts
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


class GelloFlow(FlowSpec):
    """Prompt ablation: run K agents with a vague vs concise prompt, compare."""

    K = Parameter("K", help="Number of parallel agent runs per prompt", default=5)
    model = Parameter("model", help="Model ID", default=None)

    def _run_and_verify(self, prompt: str):
        """Shared logic for run_vague and run_concise steps."""
        run_id = self.input
        task_dir = current.checkpoint.directory

        shutil.copy(BUGGY_FILE, os.path.join(task_dir, "buggy.py"))

        self.trajectory = asyncio.run(
            _run_agent(task_dir, prompt, model=self.model)
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

    @staticmethod
    def _build_field(inputs):
        """Build field from foreach inputs, return (field, metrics)."""
        field = CodeFixField()
        for inp in inputs:
            field.add(inp.trajectory, inp.outcome)
        return field, field.metrics()

    @step
    def start(self):
        """Branch into vague and concise prompt arms."""
        self.next(self.setup_vague, self.setup_concise)

    # ── Vague arm ────────────────────────────────────────────────────

    @step
    def setup_vague(self):
        """Prepare foreach for the vague prompt."""
        self.run_ids = list(range(self.K))
        self.next(self.run_vague, foreach="run_ids")

    @checkpoint
    @step
    def run_vague(self):
        """Run one agent with the vague prompt."""
        self._run_and_verify(PROMPT_VAGUE)
        self.next(self.join_vague)

    @step
    def join_vague(self, inputs):
        """Build field for the vague prompt arm."""
        field, m = self._build_field(inputs)
        self.prompt_label = "vague"
        self.field_summary = m.summary()
        self.field_points = field.points.tolist()
        self.field_outcomes = field.outcomes.tolist()
        self.outcomes = [inp.outcome for inp in inputs]

        print(f"\n[VAGUE] K={field.K}, d={field.d}")
        print(f"  Width:       {m.width():.4f}")
        print(f"  Convergence: {m.convergence():.4f}")
        print(f"  Success:     {sum(self.outcomes):.0f}/{len(self.outcomes)}")

        self.next(self.compare)

    # ── Concise arm ──────────────────────────────────────────────────

    @step
    def setup_concise(self):
        """Prepare foreach for the concise prompt."""
        self.run_ids = list(range(self.K))
        self.next(self.run_concise, foreach="run_ids")

    @checkpoint
    @step
    def run_concise(self):
        """Run one agent with the concise prompt."""
        self._run_and_verify(PROMPT_CONCISE)
        self.next(self.join_concise)

    @step
    def join_concise(self, inputs):
        """Build field for the concise prompt arm."""
        field, m = self._build_field(inputs)
        self.prompt_label = "concise"
        self.field_summary = m.summary()
        self.field_points = field.points.tolist()
        self.field_outcomes = field.outcomes.tolist()
        self.outcomes = [inp.outcome for inp in inputs]

        print(f"\n[CONCISE] K={field.K}, d={field.d}")
        print(f"  Width:       {m.width():.4f}")
        print(f"  Convergence: {m.convergence():.4f}")
        print(f"  Success:     {sum(self.outcomes):.0f}/{len(self.outcomes)}")

        self.next(self.compare)

    # ── Comparative join ─────────────────────────────────────────────

    @card(type="blank", id="ablation_report")
    @step
    def compare(self, inputs):
        """Join both arms, build comparative visualizations."""
        arms = {}
        for inp in inputs:
            arms[inp.prompt_label] = inp
        vague = arms["vague"]
        concise = arms["concise"]

        # Reconstruct FieldMetrics from stored arrays
        dims = CodeFixField().dimensions()
        m_vague = FieldMetrics(
            np.array(vague.field_points),
            np.array(vague.field_outcomes),
            dims,
        )
        m_concise = FieldMetrics(
            np.array(concise.field_points),
            np.array(concise.field_outcomes),
            dims,
        )

        self.summary_vague = vague.field_summary
        self.summary_concise = concise.field_summary

        # ── Card ─────────────────────────────────────────────────────
        c = current.card["ablation_report"]
        model_label = self.model or "default"
        K = len(vague.outcomes)

        c.append(Markdown(
            f"# Prompt Ablation Report\n"
            f"**Model:** `{model_label}` | **K:** {K}"
        ))

        # Scalar comparison table
        def _fmt_conv(val):
            return "∞" if val == float("inf") else f"{val:.4f}"

        c.append(Markdown("## Scalar Metrics"))
        c.append(Table(
            headers=["Metric", "Vague Prompt", "Concise Prompt"],
            data=[
                ["Width",
                 f"{m_vague.width():.4f}",
                 f"{m_concise.width():.4f}"],
                ["Convergence",
                 _fmt_conv(m_vague.convergence()),
                 _fmt_conv(m_concise.convergence())],
                ["Success rate",
                 f"{sum(vague.outcomes):.0f}/{len(vague.outcomes)}",
                 f"{sum(concise.outcomes):.0f}/{len(concise.outcomes)}"],
            ],
        ))

        # Grouped mean comparison
        c.append(Markdown("## Mean Behavior (Vague vs Concise)"))
        c.append(VegaChart(viz.compare_center_bar(
            m_vague, m_concise, "Vague", "Concise"
        )))

        # Grouped variance comparison
        c.append(Markdown("## Variance (Vague vs Concise)"))
        c.append(VegaChart(viz.compare_variance_bar(
            m_vague, m_concise, "Vague", "Concise"
        )))

        # Per-arm separation vectors
        c.append(Markdown("## Separation Vector — Vague Prompt"))
        c.append(VegaChart(viz.separation_bar(m_vague)))

        c.append(Markdown("## Separation Vector — Concise Prompt"))
        c.append(VegaChart(viz.separation_bar(m_concise)))

        # Pooled field: all trajectories, real verification outcomes
        all_points = np.concatenate([
            np.array(vague.field_points),
            np.array(concise.field_points),
        ])
        all_outcomes = np.concatenate([
            np.array(vague.field_outcomes),
            np.array(concise.field_outcomes),
        ])
        m_pooled = FieldMetrics(all_points, all_outcomes, dims)

        c.append(Markdown(
            "## Pooled Separation Vector\n"
            "All trajectories from both prompts, labeled by actual "
            "verification outcome. Shows which behavioral dimensions "
            "predict success regardless of prompt."
        ))
        c.append(VegaChart(viz.separation_bar(m_pooled)))

        c.append(Markdown("### Pooled Scalar Metrics"))
        c.append(Table(
            headers=["Metric", "Value"],
            data=[
                ["K (total trajectories)", str(m_pooled.K)],
                ["Width", f"{m_pooled.width():.4f}"],
                ["Convergence", _fmt_conv(m_pooled.convergence())],
                ["Success rate",
                 f"{float(np.sum(all_outcomes >= 0.5)):.0f}/{len(all_outcomes)}"],
            ],
        ))

        # Skew tables — correlation between outcome and each dimension
        def _skew_section(m, label):
            rows = []
            for i, dim in enumerate(dims):
                s = m.skew(i)
                direction = "expensive success" if s > 0.1 else (
                    "cheap success" if s < -0.1 else "no signal")
                rows.append([dim.name, f"{s:+.4f}", direction])
            c.append(Markdown(f"## Skew — {label}"))
            c.append(VegaChart(viz.skew_bar(m)))
            c.append(Table(
                headers=["Dimension", "Skew", "Interpretation"],
                data=rows,
            ))

        _skew_section(m_vague, "Vague Prompt")
        _skew_section(m_concise, "Concise Prompt")
        _skew_section(m_pooled, "Pooled (Both Arms)")

        self.next(self.end)

    @step
    def end(self):
        """Done."""
        print("Ablation complete.")


if __name__ == "__main__":
    GelloFlow()
