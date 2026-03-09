"""PromptDefenseFlow: measure agent resistance to prompt injection.

DAG:
  start → run_agent (foreach K) → join → end

Each run_agent step delegates to a named environment that handles
workspace setup, server lifecycle, prompt construction, and breach
detection. The flow just orchestrates K parallel runs and collects
raw artifacts for notebook analysis.

Usage:
  python claude_flow.py run --K 5 --model claude-sonnet-4-6 --injection swapped

  # Multi-fetch with progressive poisoning:
  python claude_flow.py run --K 5 --environment multi_fetch --injection progressive

  # Control baseline (no injection):
  python claude_flow.py run --K 2 --injection none
"""

from __future__ import annotations

import asyncio
import dataclasses

from metaflow import FlowSpec, Parameter, current, step
from metaflow import checkpoint

from environments.naive import naive_prompt_injection_environment
from environments.multi_fetch import multi_fetch_environment

ENVIRONMENTS = {
    "naive": naive_prompt_injection_environment,
    "multi_fetch": multi_fetch_environment,
}


# ═══════════════════════════════════════════════════════════════════════
# Agent runner
# ═══════════════════════════════════════════════════════════════════════


async def _run_agent(task_dir: str, prompt: str, model: str | None = None) -> dict:
    """Run the Claude agent once. Returns a plain dict trajectory."""
    from claude_agent_sdk import ClaudeAgentOptions, query

    messages = []
    turn = 0

    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            allowed_tools=["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
            permission_mode="bypassPermissions",
            cwd=task_dir,
            max_turns=15,
            model=model,
        ),
    ):
        msg = dataclasses.asdict(message)
        messages.append(msg)

        subtype = msg.get("subtype", "")
        content = msg.get("content", [])

        # Log tool calls
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and "name" in blk:
                    tool = blk["name"]
                    inp = blk.get("input", {})
                    if tool == "Bash":
                        cmd = inp.get("command", "")
                        print(f"  [{turn}] Bash: {cmd[:120]}")
                    elif tool == "Write":
                        print(f"  [{turn}] Write: {inp.get('file_path', '?')}")
                    elif tool == "Read":
                        print(f"  [{turn}] Read: {inp.get('file_path', '?')}")
                    else:
                        print(f"  [{turn}] {tool}")
                    turn += 1

        # Log final result message
        if subtype in ("success", "error_max_turns", "error"):
            usage = msg.get("usage", {})
            cost = msg.get("total_cost_usd", 0)
            in_tok = usage.get("input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)
            dur = msg.get("duration_ms", 0)
            print(
                f"  >> {subtype} | turns={msg.get('num_turns', '?')} | "
                f"cost=${cost:.4f} | {dur/1000:.1f}s | "
                f"in={in_tok} cache_r={cache_read} cache_w={cache_write} out={out_tok}"
            )

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


class PromptDefenseFlow(FlowSpec):
    """Single prompt defense experiment.

    Runs K agent instances against a named environment. Each environment
    handles its own server, prompt, and breach detection.
    """

    K = Parameter("K", help="Number of agent runs", default=2)
    model = Parameter(
        "model", help="Model ID", default="claude-sonnet-4-6",
    )
    environment = Parameter(
        "environment", help="Environment name", default="naive",
    )
    injection = Parameter(
        "injection", help="Injection strategy", default="naive",
    )
    url = Parameter(
        "url",
        help="URL to fetch through the content proxy",
        default="https://www.bbc.com/news/articles/cy7jm7k7pxzo",
    )

    @step
    def start(self):
        """Prepare foreach over K runs."""
        self.run_ids = list(range(self.K))
        self.next(self.run_agent, foreach="run_ids")

    @checkpoint
    @step
    def run_agent(self):
        """Run one agent against the environment."""
        run_id = self.input
        task_dir = current.checkpoint.directory

        # Pre-bind the agent: environment just calls agent(prompt)
        def agent(prompt):
            return asyncio.run(
                _run_agent(task_dir, prompt, model=self.model)
            )

        env_fn = ENVIRONMENTS[self.environment]
        result = env_fn(
            agent=agent,
            injection=self.injection,
            url=self.url,
            task_dir=task_dir,
        )

        self.trajectory = result["trajectory"]
        self.feedback_log = result["feedback_log"]
        self.outcome = result["outcome"]
        self.prompt = result["prompt"]
        self.posted_summaries = result.get("posted_summaries", [])

        current.checkpoint.save(name=f"run_{run_id}")

        n_msgs = len(self.trajectory["messages"])
        status = "BREACHED" if self.outcome == 0.0 else "HELD"
        print(
            f"Run {run_id}: {n_msgs} messages, "
            f"defense={status}, "
            f"cost=${self.trajectory.get('cost_usd', 0):.4f}"
        )
        self.next(self.join)

    @step
    def join(self, inputs):
        """Collect raw artifacts. Analysis happens in the notebook."""
        self.trajectories = [inp.trajectory for inp in inputs]
        self.feedback_logs = [inp.feedback_log for inp in inputs]
        self.outcomes = [inp.outcome for inp in inputs]
        self.posted_summaries = [inp.posted_summaries for inp in inputs]

        held = sum(1 for o in self.outcomes if o == 1.0)
        breached = len(self.outcomes) - held
        print(f"\n{'=' * 60}")
        print(f"Env: {self.environment} | Injection: {self.injection} | Model: {self.model} | K={len(self.outcomes)}")
        print(f"Defense held: {held} | Breached: {breached}")
        print(f"{'=' * 60}")

        self.next(self.end)

    @step
    def end(self):
        """Done."""
        print("\nPrompt defense experiment complete.")


if __name__ == "__main__":
    PromptDefenseFlow()
