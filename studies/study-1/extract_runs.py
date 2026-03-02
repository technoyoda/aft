"""Extract trajectory data from HorizonFlow runs to disk.

Run tutorial-2's HorizonFlow (tutorials/tutorial-2/) for each model
(Haiku, Sonnet, Opus) with K=20, then fill in the run IDs below.

Usage:
    1. Run HorizonFlow three times (once per model) with K=20
    2. Fill in RUNS dict with the resulting pathspecs
    3. python extract_runs.py
"""

import json
import os

from metaflow import Run

# ── Run IDs ──────────────────────────────────────────────────────────
# Fill these in after running tutorial-2's HorizonFlow for each model.
RUNS = {
    "haiku": "HorizonFlow/<your-haiku-run-id>",
    "sonnet": "HorizonFlow/<your-sonnet-run-id>",
    "opus": "HorizonFlow/<your-opus-run-id>",
}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def extract_run(label: str, pathspec: str) -> None:
    """Pull trajectories + outcomes from a HorizonFlow run and save to disk."""
    run = Run(pathspec)
    join_step = run["join"].task

    trajectories = join_step.data.trajectories
    outcomes = join_step.data.outcomes

    run_dir = os.path.join(OUT_DIR, label)
    os.makedirs(run_dir, exist_ok=True)

    # Save each trajectory as its own JSON file
    for i, (traj, outcome) in enumerate(zip(trajectories, outcomes)):
        entry = {
            "index": i,
            "outcome": outcome,
            "success": traj.get("success", outcome == 1.0),
            "num_turns": traj.get("num_turns"),
            "cost_usd": traj.get("cost_usd"),
            "duration_ms": traj.get("duration_ms"),
            "num_messages": len(traj.get("messages", [])),
            "messages": traj["messages"],
        }
        path = os.path.join(run_dir, f"trajectory_{i:02d}.json")
        with open(path, "w") as f:
            json.dump(entry, f, indent=2, default=str)

    # Save a summary manifest
    manifest = {
        "pathspec": pathspec,
        "label": label,
        "K": len(trajectories),
        "success_count": sum(1 for o in outcomes if o == 1.0),
        "outcomes": outcomes,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  {label}: {len(trajectories)} trajectories → {run_dir}/")
    print(f"    success: {manifest['success_count']}/{manifest['K']}")


def main():
    print(f"Extracting to {OUT_DIR}/\n")
    for label, pathspec in RUNS.items():
        extract_run(label, pathspec)
    print("\nDone.")


if __name__ == "__main__":
    main()
