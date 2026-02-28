"""TaskAgnosticField — dimensions that apply to any agent task.

No task-specific knowledge. Measures only generic behavioral properties:
how many tools were used, the ratio of exploration to action, how quickly
the agent committed to editing, how many direction changes occurred, etc.

These dimensions should capture the agent's "signature" — behavioral
patterns that are stable across tasks vs those that shift with the task.
"""

from __future__ import annotations

import numpy as np

import agent_fields as aft
from agent_fields import Dimension, Field


def _extract_tool_calls(trajectory: dict) -> list[dict]:
    """Pull all ToolUseBlocks out of the message stream."""
    tool_calls = []
    for msg in trajectory["messages"]:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "name" in block:
                tool_calls.append(block)
    return tool_calls


class TaskAgnosticField(Field[dict]):
    """Task-agnostic behavioral field.

    6 dimensions that apply to any agent task:
      0: tool_calls         — total tool invocations
      1: exploration_ratio  — reads / total tool calls
      2: commit_speed       — normalized position of first write (0=early, 1=late)
      3: direction_changes  — read-to-write switches
      4: verification_effort — bash calls / total tool calls
      5: message_count      — total messages
    """

    def dimensions(self) -> list[Dimension]:
        return [
            Dimension("tool_calls", "Total tool invocations"),
            Dimension("exploration_ratio", "Reads / total tool calls"),
            Dimension("commit_speed", "Normalized position of first write (0=early, 1=late)"),
            Dimension("direction_changes", "Read-to-write switches"),
            Dimension("verification_effort", "Bash calls / total tool calls"),
            Dimension("message_count", "Total messages in trajectory"),
        ]

    def measure(self, trajectory: dict) -> np.ndarray:
        tool_calls = _extract_tool_calls(trajectory)
        n = len(tool_calls) or 1

        reads = sum(1 for tc in tool_calls if tc["name"] == "Read")
        exploration_ratio = reads / n

        # Commit speed: position of first Edit or Write
        first_write_pos = n
        for i, tc in enumerate(tool_calls):
            if tc["name"] in ("Edit", "Write"):
                first_write_pos = i
                break
        commit_speed = first_write_pos / n

        # Direction changes
        direction_changes = 0
        last_type = None
        for tc in tool_calls:
            curr = "read" if tc["name"] == "Read" else (
                "write" if tc["name"] in ("Edit", "Write") else None)
            if curr and last_type and curr != last_type:
                direction_changes += 1
            if curr:
                last_type = curr

        bash_calls = sum(1 for tc in tool_calls if tc["name"] == "Bash")
        verification_effort = bash_calls / n

        return np.array([
            len(tool_calls),
            exploration_ratio,
            commit_speed,
            direction_changes,
            verification_effort,
            len(trajectory["messages"]),
        ], dtype=float)
