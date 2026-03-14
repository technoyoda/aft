"""Extract all PromptDefenseFlow runs into a single JSON dataset.

Walks every successful run, extracts ALL parameters, raw trajectories,
raw feedback logs, raw outcomes, raw posted summaries — everything stored
in Metaflow artifacts. No analysis, no condensation, no pre-computed fields.
Downstream viz scripts consume this raw dataset and compute what they need.

Usage:
    cd aft/studies/study-2
    METAFLOW_DEFAULT_DATASTORE=local METAFLOW_DEFAULT_METADATA=local \
        python blog/extract_dataset.py

Output:
    blog/dataset.json
"""

from __future__ import annotations

import json
import os
import sys

STUDY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STUDY_DIR)

from metaflow import Flow, namespace


def extract_all_runs() -> list[dict]:
    """Extract all successful PromptDefenseFlow runs with full raw data."""
    namespace(None)
    flow = Flow("PromptDefenseFlow")

    experiments = []

    for run in flow:
        if not run.successful:
            continue

        run_id = run.id

        # ── Parameters from start task ──
        start = run["start"].task
        params = {
            "run_id": run_id,
            "created_at": str(run.created_at),
            "injection": getattr(start.data, "injection", "unknown"),
            "model": getattr(start.data, "model", "unknown"),
            "environment": getattr(start.data, "environment", "naive"),
            "K": getattr(start.data, "K", 0),
            "url": getattr(start.data, "url", ""),
        }

        # ── Per-task raw artifacts from run_agent step ──
        # Each task in the foreach has: trajectory, feedback_log, outcome,
        # prompt, posted_summaries. We collect per-task to preserve task metadata.
        run_agent_step = run["run_agent"]
        per_task_records = []
        for task in run_agent_step:
            task_record = {
                "task_id": task.id,
                "task_index": task.index,
                "task_successful": task.successful,
                # Raw artifacts — everything the flow stored
                "trajectory": task.data.trajectory,
                "feedback_log": task.data.feedback_log,
                "outcome": task.data.outcome,
                "prompt": getattr(task.data, "prompt", ""),
                "posted_summaries": getattr(task.data, "posted_summaries", []),
            }
            per_task_records.append(task_record)

        # Sort by task index so ordering is deterministic
        per_task_records.sort(key=lambda t: t["task_index"] or 0)

        # ── Also grab the join-level aggregated arrays ──
        # (redundant with per-task, but useful for quick access)
        join = run["join"].task
        join_data = {
            "trajectories": getattr(join.data, "trajectories", []),
            "feedback_logs": getattr(join.data, "feedback_logs", []),
            "outcomes": getattr(join.data, "outcomes", []),
            "posted_summaries": getattr(join.data, "posted_summaries", []),
        }

        # Fall back to per-task data for any missing join-level arrays
        if per_task_records:
            if not join_data["trajectories"]:
                join_data["trajectories"] = [t["trajectory"] for t in per_task_records]
            if not join_data["feedback_logs"]:
                join_data["feedback_logs"] = [t["feedback_log"] for t in per_task_records]
            if not join_data["outcomes"]:
                join_data["outcomes"] = [t["outcome"] for t in per_task_records]
            if not join_data["posted_summaries"]:
                join_data["posted_summaries"] = [t.get("posted_summaries", []) for t in per_task_records]

        # ── Assemble experiment record ──
        experiment = {
            **params,
            "tasks": per_task_records,
            "join": join_data,
        }
        experiments.append(experiment)

        held = sum(1 for o in join_data["outcomes"] if o == 1.0)
        total = len(join_data["outcomes"])
        print(
            f"  {run_id} | {params['environment']:12s} | "
            f"{params['injection']:20s} | "
            f"{params['model']:30s} | K={params['K']} | "
            f"{held}/{total} held"
        )

    # Sort chronologically (oldest first)
    experiments.sort(key=lambda e: e["created_at"])

    return experiments


def main():
    print("Extracting all PromptDefenseFlow runs...\n")
    experiments = extract_all_runs()

    out_path = os.path.join(STUDY_DIR, "blog", "dataset.json")
    with open(out_path, "w") as f:
        json.dump(experiments, f, indent=2, default=str)

    total_tasks = sum(len(e["tasks"]) for e in experiments)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(
        f"\nWrote {len(experiments)} experiments "
        f"({total_tasks} trajectories) to {out_path} "
        f"({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
