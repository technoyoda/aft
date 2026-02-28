"""EnvShapeFlow: environment ablation — clean vs poisoned API data.

DAG:
  start
  ├── setup_clean → run_clean (foreach K) → join_clean ──────────┐
  └── setup_poisoned → run_poisoned (foreach K) → join_poisoned ─┤
                                                            compare → end

Same agent, same prompt, same task. The only difference is the data the API
serves. Clean data: all values correct. Poisoned data: one salary in cents
instead of dollars (5800000 instead of 58000). The field captures whether
the environment shapes agent behavior even when the agent doesn't know the
data has changed.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from metaflow import FlowSpec, Parameter, card, current, step
from metaflow import checkpoint
from metaflow.cards import Markdown, Table, VegaChart

import agent_fields as aft
from agent_fields import FieldMetrics
from agent_fields import visualisations as viz
from data_field import DataProcessingField

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATA = os.path.join(_SCRIPT_DIR, "api_data_clean.json")
POISONED_DATA = os.path.join(_SCRIPT_DIR, "api_data_poisoned.json")
VERIFY_SCRIPT = os.path.join(_SCRIPT_DIR, "verify_report.py")

PROMPT_TEMPLATE = (
    "There is a REST API at http://localhost:{port}. "
    "Endpoints: GET /records (employee data with name, department, salary, "
    "hours_worked), GET /metadata (summary info). "
    "Fetch the data, compute summary statistics (mean salary, headcount per "
    "department, total payroll), and write the results to report.txt."
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
            allowed_tools=["Read", "Edit", "Write", "Glob", "Grep", "Bash"],
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


def _start_server(data_file: str, port: int) -> subprocess.Popen:
    """Start the Flask API server as a subprocess."""
    proc = subprocess.Popen(
        [sys.executable, os.path.join(_SCRIPT_DIR, "api_server.py"),
         "--data", data_file, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to be ready
    for _ in range(30):
        time.sleep(0.5)
        try:
            import urllib.request
            urllib.request.urlopen(f"http://localhost:{port}/metadata", timeout=2)
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"Server exited early: {proc.stderr.read().decode()}"
                )
    raise RuntimeError("Server did not start within 15 seconds")


def _stop_server(proc: subprocess.Popen) -> None:
    """Stop the Flask server."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# ═══════════════════════════════════════════════════════════════════════
# The Metaflow flow
# ═══════════════════════════════════════════════════════════════════════


class EnvShapeFlow(FlowSpec):
    """Environment ablation: clean vs poisoned API data, compare fields."""

    K = Parameter("K", help="Number of parallel agent runs per arm", default=5)
    model = Parameter("model", help="Model ID", default=None)

    def _run_arm(self, data_file: str, variant: str):
        """Shared logic for clean and poisoned run steps."""
        run_id = self.input
        task_dir = current.checkpoint.directory

        port = random.randint(9000, 9999)
        server = _start_server(data_file, port)

        try:
            prompt = PROMPT_TEMPLATE.format(port=port)
            self.trajectory = asyncio.run(
                _run_agent(task_dir, prompt, model=self.model)
            )
        finally:
            _stop_server(server)

        # Verify report
        report_path = os.path.join(task_dir, "report.txt")
        result = subprocess.run(
            [sys.executable, VERIFY_SCRIPT, report_path, variant],
            capture_output=True, text=True, timeout=10,
        )
        self.verification_output = result.stdout.strip()
        self.outcome = 1.0 if result.returncode == 0 else 0.0

        current.checkpoint.save(name=f"run_{variant}_{run_id}")

        print(f"Run {variant}/{run_id}: "
              f"{len(self.trajectory['messages'])} messages, "
              f"verification={'pass' if self.outcome == 1.0 else 'fail'}")
        if result.returncode != 0:
            print(f"  {self.verification_output}")

    @staticmethod
    def _build_field(inputs):
        """Build field from foreach inputs."""
        field = DataProcessingField()
        for inp in inputs:
            field.add(inp.trajectory, inp.outcome)
        return field, field.metrics()

    @step
    def start(self):
        """Branch into clean and poisoned arms."""
        self.next(self.setup_clean, self.setup_poisoned)

    # ── Clean arm ─────────────────────────────────────────────────────

    @step
    def setup_clean(self):
        """Prepare foreach for the clean arm."""
        self.run_ids = list(range(self.K))
        self.next(self.run_clean, foreach="run_ids")

    @checkpoint
    @step
    def run_clean(self):
        """Run one agent with clean API data."""
        self._run_arm(CLEAN_DATA, "clean")
        self.next(self.join_clean)

    @step
    def join_clean(self, inputs):
        """Build field for the clean arm."""
        field, m = self._build_field(inputs)
        self.arm_label = "clean"
        self.field_summary = m.summary()
        self.field_points = field.points.tolist()
        self.field_outcomes = field.outcomes.tolist()
        self.outcomes = [inp.outcome for inp in inputs]
        self.trajectories = [inp.trajectory for inp in inputs]

        print(f"\n[CLEAN] K={field.K}, d={field.d}")
        print(f"  Width:       {m.width():.4f}")
        print(f"  Convergence: {m.convergence():.4f}")
        print(f"  Success:     {sum(self.outcomes):.0f}/{len(self.outcomes)}")

        self.next(self.compare)

    # ── Poisoned arm ──────────────────────────────────────────────────

    @step
    def setup_poisoned(self):
        """Prepare foreach for the poisoned arm."""
        self.run_ids = list(range(self.K))
        self.next(self.run_poisoned, foreach="run_ids")

    @checkpoint
    @step
    def run_poisoned(self):
        """Run one agent with poisoned API data."""
        self._run_arm(POISONED_DATA, "poisoned")
        self.next(self.join_poisoned)

    @step
    def join_poisoned(self, inputs):
        """Build field for the poisoned arm."""
        field, m = self._build_field(inputs)
        self.arm_label = "poisoned"
        self.field_summary = m.summary()
        self.field_points = field.points.tolist()
        self.field_outcomes = field.outcomes.tolist()
        self.outcomes = [inp.outcome for inp in inputs]
        self.trajectories = [inp.trajectory for inp in inputs]

        print(f"\n[POISONED] K={field.K}, d={field.d}")
        print(f"  Width:       {m.width():.4f}")
        print(f"  Convergence: {m.convergence():.4f}")
        print(f"  Success:     {sum(self.outcomes):.0f}/{len(self.outcomes)}")

        self.next(self.compare)

    # ── Comparative join ──────────────────────────────────────────────

    @card(type="blank", id="env_report")
    @step
    def compare(self, inputs):
        """Join both arms, build comparative visualizations."""
        arms = {}
        for inp in inputs:
            arms[inp.arm_label] = inp
        clean = arms["clean"]
        poisoned = arms["poisoned"]

        dims = DataProcessingField().dimensions()
        m_clean = FieldMetrics(
            np.array(clean.field_points),
            np.array(clean.field_outcomes),
            dims,
        )
        m_poisoned = FieldMetrics(
            np.array(poisoned.field_points),
            np.array(poisoned.field_outcomes),
            dims,
        )

        self.summary_clean = clean.field_summary
        self.summary_poisoned = poisoned.field_summary

        # ── Card ──────────────────────────────────────────────────────
        c = current.card["env_report"]
        model_label = self.model or "default"
        K = len(clean.outcomes)

        c.append(Markdown(
            f"# Environment Ablation Report\n"
            f"**Model:** `{model_label}` | **K:** {K} per arm\n\n"
            f"Same agent, same prompt, same task. Only the API data differs.\n"
            f"Clean: all values correct. Poisoned: one salary is 100x too high."
        ))

        # Scalar comparison table
        def _fmt_conv(val):
            return "∞" if val == float("inf") else f"{val:.4f}"

        c.append(Markdown("## Scalar Metrics"))
        c.append(Table(
            headers=["Metric", "Clean", "Poisoned"],
            data=[
                ["Width",
                 f"{m_clean.width():.4f}",
                 f"{m_poisoned.width():.4f}"],
                ["Convergence",
                 _fmt_conv(m_clean.convergence()),
                 _fmt_conv(m_poisoned.convergence())],
                ["Success rate",
                 f"{sum(clean.outcomes):.0f}/{len(clean.outcomes)}",
                 f"{sum(poisoned.outcomes):.0f}/{len(poisoned.outcomes)}"],
            ],
        ))

        # Grouped comparisons
        c.append(Markdown("## Mean Behavior (Clean vs Poisoned)"))
        c.append(VegaChart(viz.compare_center_bar(
            m_clean, m_poisoned, "Clean", "Poisoned"
        )))

        c.append(Markdown("## Variance (Clean vs Poisoned)"))
        c.append(VegaChart(viz.compare_variance_bar(
            m_clean, m_poisoned, "Clean", "Poisoned"
        )))

        # Per-arm separation
        c.append(Markdown("## Separation Vector — Clean"))
        c.append(VegaChart(viz.separation_bar(m_clean)))

        c.append(Markdown("## Separation Vector — Poisoned"))
        c.append(VegaChart(viz.separation_bar(m_poisoned)))

        # Skew
        c.append(Markdown("## Skew — Clean"))
        c.append(VegaChart(viz.skew_bar(m_clean)))

        c.append(Markdown("## Skew — Poisoned"))
        c.append(VegaChart(viz.skew_bar(m_poisoned)))

        self.next(self.end)

    @step
    def end(self):
        """Done."""
        print("Environment ablation complete.")


if __name__ == "__main__":
    EnvShapeFlow()
