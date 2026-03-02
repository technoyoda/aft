"""Raw statistical summary of extracted trajectory data.

Loads trajectory JSON files from disk and prints per-model and
per-trajectory breakdowns including the semantic sequence (tool calls
interleaved with model_introspection labels).

Usage:
    python raw_stats.py
"""

import json
from collections import Counter
from pathlib import Path

from trajectory_utils import (
    MODEL_INTROSPECTION,
    annotated_steps,
    rle_string,
    semantic_sequence,
    tool_sequence,
    truncate,
)

DATA_DIR = Path(__file__).parent / "data"


# ── Loading ──────────────────────────────────────────────────────────


def load_model(label: str) -> list[dict]:
    """Load all trajectories for a model from disk."""
    model_dir = DATA_DIR / label
    manifest = json.loads((model_dir / "manifest.json").read_text())
    trajectories = []
    for i in range(manifest["K"]):
        path = model_dir / f"trajectory_{i:02d}.json"
        trajectories.append(json.loads(path.read_text()))
    return trajectories


# ── Stats (operates on our on-disk trajectory format) ────────────────


def trajectory_stats(traj: dict) -> dict:
    """Compute summary stats for a single on-disk trajectory dict."""
    msgs = traj["messages"]
    sem_seq = semantic_sequence(msgs)
    t_seq = [s for s in sem_seq if s != MODEL_INTROSPECTION]

    return {
        "index": traj["index"],
        "outcome": traj["outcome"],
        "success": traj["outcome"] == 1.0,
        "num_messages": traj["num_messages"],
        "duration_ms": traj.get("duration_ms", 0),
        "tool_seq": t_seq,
        "semantic_seq": sem_seq,
        "tool_rle": rle_string(t_seq),
        "semantic_rle": rle_string(sem_seq),
        "tool_count": len(t_seq),
        "tool_dist": Counter(t_seq),
    }


def model_stats(trajectories: list[dict]) -> dict:
    """Compute aggregate stats across all trajectories for a model."""
    per_traj = [trajectory_stats(t) for t in trajectories]
    K = len(per_traj)

    agg_tools = Counter()
    for ts in per_traj:
        agg_tools += ts["tool_dist"]

    return {
        "K": K,
        "success_count": sum(1 for ts in per_traj if ts["success"]),
        "success_rate": sum(1 for ts in per_traj if ts["success"]) / K if K else 0,
        "per_trajectory": per_traj,
        "tool_dist": agg_tools,
        "msg_counts": [ts["num_messages"] for ts in per_traj],
        "tool_counts": [ts["tool_count"] for ts in per_traj],
        "durations": [ts["duration_ms"] for ts in per_traj],
    }


# ── Printers ─────────────────────────────────────────────────────────


def print_model_summary(label: str, stats: dict) -> None:
    """Print aggregate stats header for one model."""
    K = stats["K"]
    sc = stats["success_count"]

    print(f"\n{'='*60}")
    print(f"  {label.upper()}  (K={K}, success={sc}/{K})")
    print(f"{'='*60}")

    mc = stats["msg_counts"]
    tc = stats["tool_counts"]
    dur = stats["durations"]

    print(f"\n  Messages:   min={min(mc)}  max={max(mc)}  "
          f"mean={sum(mc)/K:.1f}")
    print(f"  Tool calls: min={min(tc)}  max={max(tc)}  "
          f"mean={sum(tc)/K:.1f}")
    if any(dur):
        print(f"  Duration:   min={min(dur)/1000:.1f}s  "
              f"max={max(dur)/1000:.1f}s  "
              f"mean={sum(dur)/K/1000:.1f}s")

    print(f"\n  Tool distribution (total across all trajectories):")
    for tool, count in stats["tool_dist"].most_common():
        print(f"    {tool:>12}: {count}")


def print_tool_sequences(stats: dict) -> None:
    """Print compact per-trajectory tool sequence table."""
    print(f"\n  Per-trajectory tool sequences:")
    print(f"  {'#':>3}  {'outcome':>7}  {'msgs':>4}  {'tools':>5}  tool sequence")
    print(f"  {'-'*3}  {'-'*7}  {'-'*4}  {'-'*5}  {'-'*40}")

    for ts in stats["per_trajectory"]:
        outcome = "PASS" if ts["success"] else "FAIL"
        print(f"  {ts['index']:>3}  {outcome:>7}  {ts['num_messages']:>4}  "
              f"{ts['tool_count']:>5}  {ts['tool_rle']}")


def print_semantic_sequences(stats: dict) -> None:
    """Print compact per-trajectory semantic sequence table."""
    print(f"\n  Per-trajectory semantic sequences (MI = model_introspection):")
    print(f"  {'#':>3}  {'outcome':>7}  semantic sequence")
    print(f"  {'-'*3}  {'-'*7}  {'-'*60}")

    for ts in stats["per_trajectory"]:
        outcome = "PASS" if ts["success"] else "FAIL"
        short_rle = ts["semantic_rle"].replace("model_introspection", "MI")
        print(f"  {ts['index']:>3}  {outcome:>7}  {short_rle}")


def print_annotated_trajectories(trajectories: list[dict]) -> None:
    """Print annotated per-trajectory detail with model text."""
    print(f"\n  Annotated trajectories:")
    for t in trajectories:
        outcome = "PASS" if t["outcome"] == 1.0 else "FAIL"
        print(f"\n  ── trajectory {t['index']:02d} ({outcome}) ──")
        steps = annotated_steps(t["messages"])
        for si, step in enumerate(steps):
            text_preview = truncate(step["text"], 70) if step["text"] else ""
            tool_strs = []
            for tname, targ in zip(step["tools"], step["args"]):
                tool_strs.append(f"{tname}({targ})" if targ else tname)
            tools_line = ", ".join(tool_strs)

            if text_preview:
                print(f"    step {si}: [{text_preview}]")
                print(f"            → {tools_line}")
            else:
                print(f"    step {si}: → {tools_line}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    for label in ["haiku", "sonnet", "opus"]:
        if not (DATA_DIR / label).exists():
            continue
        trajectories = load_model(label)
        stats = model_stats(trajectories)

        print_model_summary(label, stats)
        print_tool_sequences(stats)
        print_semantic_sequences(stats)
        print_annotated_trajectories(trajectories)


if __name__ == "__main__":
    main()
