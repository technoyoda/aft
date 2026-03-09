"""Composable primitives for analyzing Claude agent trajectories.

Pure functions operating on the message list (``list[dict]``) produced
by ``claude_agent_sdk``.  No I/O, no printing, no disk assumptions.
The caller is responsible for loading trajectories however they wish
(Metaflow, JSON files, API responses, etc.) and passing the messages
in.

Message format (claude_agent_sdk):
    - Assistant turns have a ``model`` key.  Content blocks use
      ``text`` (reasoning) or ``name``+``input`` (tool call) directly
      — there is no ``type`` wrapper and no ``role`` key.
    - Tool-result turns have a ``tool_use_result`` key.
    - The first message is a system envelope (``subtype``+``data``).
    - The last message is a session summary (``duration_ms``, ``result``, …).

Usage::

    from trajectory_utils import semantic_sequence, rle, rle_string

    messages = ...  # however you load them
    seq = semantic_sequence(messages)
    print(rle_string(seq))
"""

from collections import Counter


# ── Constants ────────────────────────────────────────────────────────

MODEL_INTROSPECTION = "model_introspection"

# Intent labels (acting / introspecting)
ACTING = "acting"
INTROSPECTING = "introspecting"


# ── Message predicates ───────────────────────────────────────────────


def is_assistant(msg: dict) -> bool:
    """True if this message is an assistant (model) turn."""
    return "model" in msg


def is_tool_result(msg: dict) -> bool:
    """True if this message is a tool-result turn."""
    return "tool_use_result" in msg


# ── Sequence extraction ──────────────────────────────────────────────


def semantic_sequence(messages: list[dict]) -> list[str]:
    """Extract the full semantic action sequence from messages.

    Walks every assistant turn in temporal order.  For each turn:
    - If the turn contains text (reasoning tokens), emits
      ``"model_introspection"``.
    - For each tool call in the turn, emits the tool name
      (e.g. ``"Read"``, ``"Edit"``, ``"Bash"``).

    Text always precedes tools within a single assistant message,
    preserving the within-turn ordering.

    Returns a flat ``list[str]`` — the trajectory's semantic trace.
    """
    seq: list[str] = []
    for msg in messages:
        if not is_assistant(msg):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        has_text = False
        tools: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if "text" in block and "name" not in block:
                if block["text"].strip():
                    has_text = True
            elif "name" in block:
                tools.append(block["name"])

        if has_text:
            seq.append(MODEL_INTROSPECTION)
        seq.extend(tools)
    return seq


def tool_sequence(messages: list[dict]) -> list[str]:
    """Extract tool-only action sequence (no introspection labels)."""
    return [s for s in semantic_sequence(messages) if s != MODEL_INTROSPECTION]


def tool_counts(messages: list[dict]) -> Counter:
    """Count occurrences of each tool across all assistant turns."""
    return Counter(tool_sequence(messages))


# ── Intent (acting / introspecting) ──────────────────────────────────


def intent_sequence(messages: list[dict]) -> list[str]:
    """Map the semantic sequence to binary intent labels.

    Every element of the semantic sequence becomes either
    ``"acting"`` (tool call) or ``"introspecting"`` (model tokens).
    The result is the intent trace ρ_π for this trajectory.
    """
    return [
        INTROSPECTING if s == MODEL_INTROSPECTION else ACTING
        for s in semantic_sequence(messages)
    ]


def introspection_ratio(messages: list[dict]) -> float:
    """Fraction of the intent sequence that is introspecting.

    Returns 0.0–1.0.  Higher means the model spends more of its
    sequence on reasoning relative to tool use.
    """
    seq = intent_sequence(messages)
    if not seq:
        return 0.0
    return sum(1 for s in seq if s == INTROSPECTING) / len(seq)


def action_run_lengths(messages: list[dict]) -> list[int]:
    """Lengths of consecutive acting runs (no introspection between).

    Opus chains actions (run length > 1).  Haiku almost never does.
    """
    return [n for label, n in rle(intent_sequence(messages)) if label == ACTING]


def introspection_run_lengths(messages: list[dict]) -> list[int]:
    """Lengths of consecutive introspecting runs."""
    return [n for label, n in rle(intent_sequence(messages)) if label == INTROSPECTING]


# ── Run-length encoding ─────────────────────────────────────────────


def rle(seq: list[str]) -> list[tuple[str, int]]:
    """Run-length encode a sequence.

    Returns list of ``(label, count)`` tuples.  Consecutive identical
    labels are collapsed::

        >>> rle(["Read", "Read", "Edit", "Edit", "Edit", "Bash"])
        [("Read", 2), ("Edit", 3), ("Bash", 1)]
    """
    if not seq:
        return []
    runs: list[tuple[str, int]] = []
    for item in seq:
        if runs and runs[-1][0] == item:
            runs[-1] = (item, runs[-1][1] + 1)
        else:
            runs.append((item, 1))
    return runs


def rle_string(seq: list[str]) -> str:
    """Human-readable RLE string.

    ``["Read", "Edit", "Edit", "Bash"]`` → ``"Read → Edit×2 → Bash"``
    """
    return " → ".join(
        f"{name}×{n}" if n > 1 else name
        for name, n in rle(seq)
    )


# ── Annotated steps ─────────────────────────────────────────────────


def tool_arg_summary(block: dict) -> str:
    """One-line summary of a tool call's key argument."""
    name = block["name"]
    inp = block.get("input", {})
    if name == "Read":
        p = inp.get("file_path", "")
        return p.rsplit("/", 1)[-1] if "/" in p else p
    if name == "Edit":
        p = inp.get("file_path", "")
        return p.rsplit("/", 1)[-1] if "/" in p else p
    if name == "Bash":
        cmd = inp.get("command", "")
        return truncate(cmd, 40)
    if name == "Glob":
        return inp.get("pattern", "")
    if name == "Grep":
        return inp.get("pattern", "")
    if name == "TodoWrite":
        return "todo"
    return ""


def annotated_steps(messages: list[dict]) -> list[dict]:
    """Extract steps merging text-before-tool into unified actions.

    The SDK often sends reasoning text and the tool call as separate
    messages.  We merge: if a text-only message is immediately
    followed by a tool-only message, they become one step.

    Returns a list of step dicts::

        {
            "text": str,         # assistant reasoning text
            "tools": list[str],  # tool names in this step
            "args": list[str],   # one-line arg summary per tool
        }
    """
    raw = []
    for msg in messages:
        if not is_assistant(msg):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue

        text_parts: list[str] = []
        tools: list[str] = []
        args: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if "text" in block and "name" not in block:
                text_parts.append(block["text"])
            elif "name" in block:
                tools.append(block["name"])
                args.append(tool_arg_summary(block))

        if tools or text_parts:
            raw.append({
                "text": " ".join(text_parts).strip(),
                "tools": tools,
                "args": args,
            })

    # Merge: text-only followed by tool-only → single step
    steps = []
    i = 0
    while i < len(raw):
        cur = raw[i]
        if cur["text"] and not cur["tools"] and i + 1 < len(raw):
            nxt = raw[i + 1]
            if nxt["tools"] and not nxt["text"]:
                steps.append({
                    "text": cur["text"],
                    "tools": nxt["tools"],
                    "args": nxt["args"],
                })
                i += 2
                continue
        steps.append(cur)
        i += 1
    return steps


# ── Helpers ──────────────────────────────────────────────────────────


def truncate(s: str, n: int) -> str:
    """Truncate string to n chars, add ellipsis if truncated."""
    s = s.strip().replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s
