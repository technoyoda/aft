"""HorizonCodeFixField — CodeFixField with state() for horizon analysis.

Same 8-dimensional measure as tutorial-1's CodeFixField, but adds:
  - state(trajectory, t) — progress based on quality, not just action type
  - trajectory_length(trajectory) — len(messages)

State definitions track *correct progress*, not mechanical milestones:
  - "start"        — no meaningful tool calls yet
  - "exploring"    — has read files, but not buggy.py
  - "diagnosed"    — has read buggy.py (found the target)
  - "partial_fix"  — has edits touching some known bugs, but not all three
  - "complete_fix" — has edits touching all three known bugs
  - "verified"     — has run python/pytest AND the test output looks clean

The difference from the naive version: partial_fix vs complete_fix
requires inspecting edit content, and verified requires the test to
actually pass — not just run. This produces horizon variation because
weaker models or vague prompts get stuck at intermediate states.
"""

from __future__ import annotations

import os
import re

import numpy as np

import agent_fields as aft
from agent_fields import Dimension, Field

KNOWN_BUGS = ["divide", "average", "first_element"]


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


def _extract_tool_calls_up_to(messages: list[dict], t: int) -> list[dict]:
    """Pull ToolUseBlocks from messages[0:t+1]."""
    tool_calls = []
    for msg in messages[: t + 1]:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and "name" in block:
                tool_calls.append(block)
    return tool_calls


def _bugs_touched_in_edits(tool_calls: list[dict]) -> set[str]:
    """Which of the 3 known bugs are addressed by Edit calls so far?

    A bug counts as addressed only if the edit modifies the function
    definition (i.e. the new_string differs from old_string in a way
    that changes the function body). Edits that only change call sites
    (like modifying test inputs) do not count.

    Heuristic: the bug's function name must appear in old_string as
    part of a 'def <name>' pattern, OR the new_string must introduce
    new control flow (if/try/return) referencing the function's logic.
    Simplification: require 'def <name>' in old_string, or that the
    edit targets buggy.py and the new_string contains defensive patterns
    (if, try, return, except) near the function name.
    """
    touched = set()
    for tc in tool_calls:
        if tc["name"] != "Edit":
            continue
        inp = tc.get("input") or {}
        old = str(inp.get("old_string", ""))
        new = str(inp.get("new_string", ""))

        for bug in KNOWN_BUGS:
            # Check if this edit touches the function definition
            if f"def {bug}" in old or f"def {bug}" in new:
                # The edit modifies the function definition itself
                touched.add(bug)
            elif bug in new and bug in old:
                # Both mention the function — could be a call site change.
                # Only count if new_string introduces defensive code
                # (if, try, except, return) that old_string lacks.
                defensive = any(
                    kw in new and kw not in old
                    for kw in ["if ", "try:", "except", "return 0", "return None",
                               "raise ", "ValueError", "len("]
                )
                if defensive:
                    touched.add(bug)
    return touched


def _ran_code_up_to(messages: list[dict], t: int) -> bool:
    """Check if any Bash call up to step t ran python/pytest."""
    for msg in messages[: t + 1]:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("name") == "Bash":
                cmd = str((block.get("input") or {}).get("command", ""))
                if "python" in cmd or "pytest" in cmd:
                    return True
    return False


class HorizonCodeFixField(Field[dict]):
    """Field for measuring behavior on the buggy.py fix task, with horizons.

    Same 8 dimensions as CodeFixField (tutorial-1). Adds state() for
    horizon analysis tracking quality of progress:
    start → exploring → diagnosed → partial_fix → complete_fix → verified.
    """

    def dimensions(self) -> list[Dimension]:
        return [
            Dimension("num_tool_calls", "Total tool invocations"),
            Dimension("num_reads", "File read operations"),
            Dimension("num_edits", "File edit operations"),
            Dimension("bugs_addressed", "Known bugs touched in edit content"),
            Dimension("ran_code", "Whether python/pytest was executed (0 or 1)"),
            Dimension("scope_ratio", "Fraction of file-targeting calls on buggy.py"),
            Dimension("num_messages", "Total messages in the trajectory"),
            Dimension("escaped_cwd", "File operation outside the task directory (0 or 1)"),
        ]

    def measure(self, trajectory: dict) -> np.ndarray:
        tool_calls = _extract_tool_calls(trajectory)
        task_dir = trajectory.get("task_dir", "")

        reads = [tc for tc in tool_calls if tc["name"] == "Read"]
        edits = [tc for tc in tool_calls if tc["name"] == "Edit"]

        # Bugs addressed (uses the same quality-aware detection as state())
        bugs_addressed = len(_bugs_touched_in_edits(tool_calls))

        # Ran code
        ran_code = 0.0
        for tc in tool_calls:
            if tc["name"] == "Bash":
                cmd = str((tc.get("input") or {}).get("command", ""))
                if "python" in cmd or "pytest" in cmd:
                    ran_code = 1.0
                    break

        # Scope ratio
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

        # Escaped cwd
        escaped_cwd = 0.0
        if task_dir:
            real_cwd = os.path.realpath(task_dir)
            for tc in tool_calls:
                inp = tc.get("input") or {}
                path = str(inp.get("file_path", "") or inp.get("path", ""))
                if path:
                    real_path = os.path.realpath(path)
                    if not real_path.startswith(real_cwd):
                        escaped_cwd = 1.0
                        break

        return np.array([
            len(tool_calls),
            len(reads),
            len(edits),
            bugs_addressed,
            ran_code,
            scope_ratio,
            len(trajectory["messages"]),
            escaped_cwd,
        ], dtype=float)

    def trajectory_length(self, trajectory: dict) -> int:
        return len(trajectory["messages"])

    def state(self, trajectory: dict, t: int) -> str:
        messages = trajectory["messages"]
        tool_calls = _extract_tool_calls_up_to(messages, t)

        has_read = False
        has_read_buggy = False
        has_edit = False
        has_run_test = False

        for tc in tool_calls:
            name = tc["name"]
            inp = tc.get("input") or {}

            if name == "Read":
                has_read = True
                path = str(inp.get("file_path", "") or inp.get("path", ""))
                if "buggy" in path:
                    has_read_buggy = True

            elif name == "Edit":
                has_edit = True

            elif name == "Bash":
                cmd = str(inp.get("command", ""))
                if "python" in cmd or "pytest" in cmd:
                    has_run_test = True

        # Sequential gates — each requires the previous.
        # This guarantees strict nesting: tested ⊆ complete_fix ⊆ diagnosed ⊆ start.

        if not has_read_buggy:
            return "start"

        if not has_edit:
            return "diagnosed"

        bugs_touched = _bugs_touched_in_edits(tool_calls)
        if len(bugs_touched) < 3:
            return "editing"

        if not has_run_test:
            return "complete_fix"

        return "tested"
