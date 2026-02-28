"""DataProcessingField — Field subclass for the data processing experiment.

Trajectory type: plain dict from dataclasses.asdict() on claude_agent_sdk
messages. Same structure as tutorial-1's CodeFixField.

7-dimensional measure:
  0: api_calls       — number of HTTP requests to the API (curl/requests)
  1: data_validation — did the agent validate or cross-check data (0/1)
  2: num_reads       — file read operations
  3: num_writes      — file write/edit operations
  4: tool_calls      — total tool invocations
  5: scope_focus     — fraction of operations on report-related files vs total
  6: num_messages    — total messages

State function tracks progress through the data processing task:
  start → fetching → processing → reporting → validated
"""

from __future__ import annotations

import re

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


def _is_api_call(tc: dict) -> bool:
    """Check if a tool call is an HTTP request to the API."""
    if tc["name"] != "Bash":
        return False
    cmd = str((tc.get("input") or {}).get("command", ""))
    return any(kw in cmd for kw in ["curl", "wget", "requests.get", "http"])


def _has_validation_signal(tool_calls: list[dict], messages: list[dict]) -> bool:
    """Detect whether the agent performed data validation.

    Signals:
    - Multiple API calls (re-fetching to cross-check)
    - Bash calls with validation keywords (outlier, anomaly, sanity, validate)
    - Text blocks mentioning validation, checking, or outlier detection
    """
    api_call_count = sum(1 for tc in tool_calls if _is_api_call(tc))
    if api_call_count >= 3:
        return True

    validation_keywords = re.compile(
        r"(validat|outlier|anomal|sanity|cross.?check|suspicious|unusual|"
        r"incorrect|error in|data quality|seems wrong|too high|too low)",
        re.IGNORECASE,
    )

    for tc in tool_calls:
        if tc["name"] == "Bash":
            cmd = str((tc.get("input") or {}).get("command", ""))
            if validation_keywords.search(cmd):
                return True

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    if validation_keywords.search(block["text"]):
                        return True
        elif isinstance(content, str):
            if validation_keywords.search(content):
                return True

    return False


def _is_report_related(path: str) -> bool:
    """Check if a file path is related to the report output."""
    path_lower = path.lower()
    return any(kw in path_lower for kw in ["report", "summary", "output", "result"])


class DataProcessingField(Field[dict]):
    """Field for measuring behavior on the data processing task.

    7-dimensional measure:
      0: api_calls       — HTTP requests to the API
      1: data_validation — whether the agent validated/cross-checked data (0/1)
      2: num_reads       — file read operations
      3: num_writes      — file write/edit operations
      4: tool_calls      — total tool invocations
      5: scope_focus     — fraction of file ops on report-related files
      6: num_messages    — total messages
    """

    def dimensions(self) -> list[Dimension]:
        return [
            Dimension("api_calls", "HTTP requests to the API"),
            Dimension("data_validation", "Whether the agent validated data (0 or 1)"),
            Dimension("num_reads", "File read operations"),
            Dimension("num_writes", "File write/edit operations"),
            Dimension("tool_calls", "Total tool invocations"),
            Dimension("scope_focus", "Fraction of file ops on report-related files"),
            Dimension("num_messages", "Total messages in the trajectory"),
        ]

    def measure(self, trajectory: dict) -> np.ndarray:
        tool_calls = _extract_tool_calls(trajectory)
        messages = trajectory["messages"]

        api_calls = sum(1 for tc in tool_calls if _is_api_call(tc))

        data_validation = 1.0 if _has_validation_signal(
            tool_calls, messages
        ) else 0.0

        reads = sum(1 for tc in tool_calls if tc["name"] == "Read")
        writes = sum(
            1 for tc in tool_calls if tc["name"] in ("Edit", "Write")
        )

        # Scope focus: fraction of file-targeting ops on report-related files
        report_ops = 0
        file_ops = 0
        for tc in tool_calls:
            inp = tc.get("input") or {}
            path = str(inp.get("file_path", "") or inp.get("path", ""))
            if path:
                file_ops += 1
                if _is_report_related(path):
                    report_ops += 1
        scope_focus = report_ops / file_ops if file_ops else 0.0

        return np.array([
            api_calls,
            data_validation,
            reads,
            writes,
            len(tool_calls),
            scope_focus,
            len(messages),
        ], dtype=float)

    def trajectory_length(self, trajectory: dict) -> int:
        return len(trajectory["messages"])

    def state(self, trajectory: dict, t: int) -> str:
        messages = trajectory["messages"]
        tool_calls = _extract_tool_calls_up_to(messages, t)

        has_api_call = any(_is_api_call(tc) for tc in tool_calls)
        has_write = any(
            tc["name"] in ("Edit", "Write") for tc in tool_calls
        )
        has_report = any(
            _is_report_related(
                str((tc.get("input") or {}).get("file_path", "")
                    or (tc.get("input") or {}).get("path", ""))
            )
            for tc in tool_calls
            if tc["name"] in ("Edit", "Write")
        )
        has_recheck = False
        api_count = 0
        past_first_write = False
        for tc in tool_calls:
            if tc["name"] in ("Edit", "Write"):
                past_first_write = True
            if _is_api_call(tc):
                api_count += 1
                if past_first_write:
                    has_recheck = True

        if has_recheck or _has_validation_signal(tool_calls, messages[: t + 1]):
            return "validated"

        if has_report:
            return "reporting"

        if has_write:
            return "processing"

        if has_api_call:
            return "fetching"

        return "start"
