"""OracleField — a Field where measure() calls an LLM to score trajectories.

Instead of hand-crafted dimension extraction, an LLM reads each trajectory
and scores it on 4 semantic dimensions (0-5 scale):
  - reasoning_coherence: how logically structured was the agent's approach?
  - strategy_focus: did the agent maintain a clear strategy or wander?
  - error_recovery: how well did the agent handle unexpected results?
  - task_understanding: did the agent understand what was being asked?

Scores are cached to a JSON file to avoid redundant API calls.

The oracle state() function has the LLM read a trajectory prefix and return
a phase label. Also cached.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

import agent_fields as aft
from agent_fields import Dimension, Field

_CACHE_DIR = os.path.dirname(os.path.abspath(__file__))
_SCORE_CACHE_FILE = os.path.join(_CACHE_DIR, ".oracle_score_cache.json")
_STATE_CACHE_FILE = os.path.join(_CACHE_DIR, ".oracle_state_cache.json")

SCORING_DIMENSIONS = [
    ("reasoning_coherence",
     "How logically structured was the agent's approach? "
     "0 = random/chaotic, 5 = perfectly systematic"),
    ("strategy_focus",
     "Did the agent maintain a clear strategy or wander? "
     "0 = constant context-switching, 5 = laser-focused"),
    ("error_recovery",
     "How well did the agent handle unexpected results? "
     "0 = ignored errors or panicked, 5 = diagnosed and adapted cleanly"),
    ("task_understanding",
     "Did the agent understand what was being asked? "
     "0 = completely misunderstood, 5 = perfect grasp"),
]


def _trajectory_hash(trajectory: dict) -> str:
    """Stable hash for caching."""
    raw = json.dumps(trajectory, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(path: str, cache: dict) -> None:
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def _summarize_trajectory(trajectory: dict, max_chars: int = 4000) -> str:
    """Create a human-readable summary of the trajectory for the LLM."""
    lines = []
    for i, msg in enumerate(trajectory.get("messages", [])):
        content = msg.get("content")
        if isinstance(content, str):
            lines.append(f"[msg {i}] {content[:200]}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if "text" in block:
                        lines.append(f"[msg {i}] {block['text'][:200]}")
                    elif "name" in block:
                        inp = block.get("input") or {}
                        lines.append(
                            f"[msg {i}] Tool: {block['name']}("
                            f"{json.dumps(inp, default=str)[:150]})"
                        )
    summary = "\n".join(lines)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n... (truncated)"
    return summary


def _call_llm_for_scores(trajectory_summary: str) -> dict[str, float]:
    """Call Anthropic API to score a trajectory on the 4 dimensions."""
    import anthropic

    dim_descriptions = "\n".join(
        f"- {name}: {desc}" for name, desc in SCORING_DIMENSIONS
    )

    prompt = (
        f"You are evaluating an AI agent's trajectory on a coding task. "
        f"Score it on each dimension from 0 to 5 (integers only).\n\n"
        f"Dimensions:\n{dim_descriptions}\n\n"
        f"Trajectory:\n{trajectory_summary}\n\n"
        f"Respond with ONLY a JSON object mapping dimension names to scores. "
        f"Example: {{\"reasoning_coherence\": 3, \"strategy_focus\": 4, "
        f"\"error_recovery\": 2, \"task_understanding\": 5}}"
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()
    # Extract JSON from response
    match = re.search(r'\{[^}]+\}', text)
    if match:
        scores = json.loads(match.group())
    else:
        scores = json.loads(text)

    # Clamp to [0, 5]
    return {
        name: max(0, min(5, int(scores.get(name, 0))))
        for name, _ in SCORING_DIMENSIONS
    }


def _call_llm_for_state(trajectory_summary: str) -> str:
    """Call Anthropic API to classify the trajectory phase."""
    import anthropic

    prompt = (
        f"You are analyzing an AI agent's partial trajectory on a coding task. "
        f"What phase is the agent in? Choose exactly one:\n"
        f"- exploring: reading files, understanding the codebase\n"
        f"- diagnosing: identified the problem, reasoning about the fix\n"
        f"- implementing: actively editing code\n"
        f"- verifying: running tests or checking the fix\n\n"
        f"Trajectory:\n{trajectory_summary}\n\n"
        f"Respond with ONLY the phase label (one word)."
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip().lower()
    valid = {"exploring", "diagnosing", "implementing", "verifying"}
    return text if text in valid else "exploring"


class OracleField(Field[dict]):
    """Field where measure() uses an LLM to score trajectories.

    4-dimensional measure (all 0-5 scale):
      0: reasoning_coherence
      1: strategy_focus
      2: error_recovery
      3: task_understanding
    """

    def dimensions(self) -> list[Dimension]:
        return [
            Dimension(name, desc) for name, desc in SCORING_DIMENSIONS
        ]

    def measure(self, trajectory: dict) -> np.ndarray:
        key = _trajectory_hash(trajectory)
        cache = _load_cache(_SCORE_CACHE_FILE)

        if key in cache:
            scores = cache[key]
        else:
            summary = _summarize_trajectory(trajectory)
            scores = _call_llm_for_scores(summary)
            cache[key] = scores
            _save_cache(_SCORE_CACHE_FILE, cache)

        return np.array([
            scores.get(name, 0) for name, _ in SCORING_DIMENSIONS
        ], dtype=float)

    def trajectory_length(self, trajectory: dict) -> int:
        return len(trajectory["messages"])

    def state(self, trajectory: dict, t: int) -> str:
        messages = trajectory["messages"]
        prefix = {"messages": messages[: t + 1]}

        key = _trajectory_hash(prefix)
        cache = _load_cache(_STATE_CACHE_FILE)

        if key in cache:
            return cache[key]

        summary = _summarize_trajectory(prefix, max_chars=2000)
        state_label = _call_llm_for_state(summary)
        cache[key] = state_label
        _save_cache(_STATE_CACHE_FILE, cache)

        return state_label
