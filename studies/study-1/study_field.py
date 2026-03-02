"""CodeFixField with acting/introspecting intent for study-1.

Uses the same 8-dimensional measure and state chain as the tutorial
HorizonCodeFixField, plus intent() returning binary acting/introspecting
labels derived from the message structure.

The intent function reads the policy's rhythm: when it acts (tool calls)
vs when it thinks (reasoning tokens). This lets the Field machinery
(regimes, program families) operate on the act/think structure directly.
"""

from __future__ import annotations

import numpy as np

import agent_fields as aft

from trajectory_utils import is_assistant, ACTING, INTROSPECTING

# ── Known bugs in buggy.py (same as tutorial-2) ─────────────────────

KNOWN_BUGS = ["divide", "average", "first_element"]


# ── Helpers ──────────────────────────────────────────────────────────


def _extract_tool_calls(trajectory: dict) -> list[dict]:
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
    touched = set()
    for tc in tool_calls:
        if tc["name"] != "Edit":
            continue
        inp = tc.get("input") or {}
        old = str(inp.get("old_string", ""))
        new = str(inp.get("new_string", ""))

        for bug in KNOWN_BUGS:
            if f"def {bug}" in old or f"def {bug}" in new:
                touched.add(bug)
            elif bug in new and bug in old:
                defensive = any(
                    kw in new and kw not in old
                    for kw in ["if ", "try:", "except", "return 0", "return None",
                               "raise ", "ValueError", "len("]
                )
                if defensive:
                    touched.add(bug)
    return touched


def _msg_has_tool_calls(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and "name" in b for b in content)


def _msg_has_text(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    for b in content:
        if isinstance(b, dict) and "text" in b and "name" not in b:
            if b["text"].strip():
                return True
    return False


# ── The Field ────────────────────────────────────────────────────────


class IntentFluctuationField(aft.Field[dict]):
    """CodeFixField with three lenses for study-1 analysis.

    - measure() → 8-dim behavioral vector (phi)
    - state()   → task progress: start → diagnosed → editing → complete_fix → tested (psi)
    - intent()  → policy rhythm: acting vs introspecting (rho_pi)
    """

    def dimensions(self) -> list[aft.Dimension]:
        return [
            aft.Dimension("num_tool_calls", "Total tool invocations"),
            aft.Dimension("num_reads", "File read operations"),
            aft.Dimension("num_edits", "File edit operations"),
            aft.Dimension("bugs_addressed", "Known bugs touched in edit content"),
            aft.Dimension("scope_ratio", "Fraction of file-targeting calls on buggy.py"),
        ]

    def measure(self, trajectory: dict) -> np.ndarray:
        tool_calls = _extract_tool_calls(trajectory)

        reads = [tc for tc in tool_calls if tc["name"] == "Read"]
        edits = [tc for tc in tool_calls if tc["name"] == "Edit"]
        bugs_addressed = len(_bugs_touched_in_edits(tool_calls))

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

        return np.array([
            len(tool_calls),
            len(reads),
            len(edits),
            bugs_addressed,
            scope_ratio,
        ], dtype=float)

    def trajectory_length(self, trajectory: dict) -> int:
        return len(trajectory["messages"])

    def state(self, trajectory: dict, t: int) -> str:
        messages = trajectory["messages"]
        tool_calls = _extract_tool_calls_up_to(messages, t)

        has_read_buggy = False
        has_edit = False
        has_run_test = False

        for tc in tool_calls:
            name = tc["name"]
            inp = tc.get("input") or {}

            if name == "Read":
                path = str(inp.get("file_path", "") or inp.get("path", ""))
                if "buggy" in path:
                    has_read_buggy = True
            elif name == "Edit":
                has_edit = True
            elif name == "Bash":
                cmd = str(inp.get("command", ""))
                if "python" in cmd or "pytest" in cmd:
                    has_run_test = True

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

    def intent(self, trajectory: dict, t: int) -> str:
        """Acting vs introspecting at message index t.

        Only assistant messages carry intent:
        - acting: contains tool call(s)
        - introspecting: reasoning text, no tool calls

        Non-assistant messages (tool results, system envelopes) are
        transparent — they inherit the label of the most recent
        assistant message so they don't create spurious RLE boundaries.
        """
        messages = trajectory["messages"]
        msg = messages[t]

        if is_assistant(msg):
            if _msg_has_tool_calls(msg):
                return ACTING
            if _msg_has_text(msg):
                return INTROSPECTING
            return ACTING

        # Non-assistant: walk backwards to find the last assistant label
        for prev in range(t - 1, -1, -1):
            prev_msg = messages[prev]
            if is_assistant(prev_msg):
                if _msg_has_tool_calls(prev_msg):
                    return ACTING
                if _msg_has_text(prev_msg):
                    return INTROSPECTING
                return ACTING

        # No previous assistant — look forward to inherit from the first one
        for fwd in range(t + 1, len(messages)):
            fwd_msg = messages[fwd]
            if is_assistant(fwd_msg):
                if _msg_has_tool_calls(fwd_msg):
                    return ACTING
                if _msg_has_text(fwd_msg):
                    return INTROSPECTING
                return ACTING
        return ACTING
