"""Run multiple prompt defense experiments concurrently via Metaflow Runner.

Usage:
  python run_experiments.py                    # default matrix
  python run_experiments.py --K 5             # override K for all runs
  python run_experiments.py --models sonnet   # just sonnet
  python run_experiments.py --injections swapped naive  # subset of strategies

Launches one PromptDefenseFlow per (injection, model) combination using
Metaflow's Runner API with async_run() for concurrency.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FLOW_FILE = os.path.join(_SCRIPT_DIR, "claude_flow.py")

# ── Model shorthand ────────────────────────────────────────────────

MODEL_MAP = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6-20250500",
    "opus": "claude-opus-4-6-20250500",
}

# ── Default experiment matrix ──────────────────────────────────────

DEFAULT_INJECTIONS = ["none", "naive", "authoritative", "contextual", "repeated", "swapped"]
DEFAULT_MODELS = ["sonnet"]
DEFAULT_K = 5
DEFAULT_MAX_WORKERS = 2


def _resolve_model(name: str) -> str:
    return MODEL_MAP.get(name, name)


async def run_matrix(
    injections: list[str],
    models: list[str],
    K: int,
    max_workers: int,
    max_concurrent: int,
) -> list[dict]:
    """Launch all (injection x model) combinations concurrently."""
    from metaflow import Runner

    env = {
        "METAFLOW_DEFAULT_DATASTORE": "local",
        "METAFLOW_DEFAULT_METADATA": "local",
    }

    combos = [
        (inj, model) for inj in injections for model in models
    ]
    total = len(combos)
    print(f"Launching {total} experiments: {len(injections)} injections x {len(models)} models, K={K}")
    print(f"Max concurrent flows: {max_concurrent}")
    print()

    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(idx: int, injection: str, model_name: str) -> dict:
        model_id = _resolve_model(model_name)
        label = f"[{idx+1}/{total}] {injection} / {model_name}"

        async with semaphore:
            print(f"{label}: starting...")
            t0 = time.time()

            async with Runner(
                FLOW_FILE,
                env=env,
                cwd=_SCRIPT_DIR,
                show_output=False,
            ) as runner:
                executing = await runner.async_run(
                    K=K,
                    injection=injection,
                    model=model_id,
                    max_workers=max_workers,
                )
                # Wait for it to finish
                run = executing.run
                elapsed = time.time() - t0

            status = "finished" if run.finished else "failed"
            print(f"{label}: {status} in {elapsed:.0f}s (run_id={run.id})")

            return {
                "injection": injection,
                "model": model_name,
                "model_id": model_id,
                "run_id": run.id,
                "finished": run.finished,
                "elapsed_s": elapsed,
            }

    tasks = [
        run_one(i, inj, model)
        for i, (inj, model) in enumerate(combos)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"{'Injection':<16} {'Model':<10} {'Status':<10} {'Time':>8}  {'Run ID'}")
    print(f"{'-' * 70}")
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        else:
            status = "ok" if r["finished"] else "FAIL"
            print(
                f"{r['injection']:<16} {r['model']:<10} {status:<10} "
                f"{r['elapsed_s']:>6.0f}s  {r['run_id']}"
            )
    print(f"{'=' * 70}")

    return [r for r in results if not isinstance(r, Exception)]


def main():
    parser = argparse.ArgumentParser(description="Run prompt defense experiments concurrently")
    parser.add_argument(
        "--injections", nargs="+", default=DEFAULT_INJECTIONS,
        help=f"Injection strategies (default: {DEFAULT_INJECTIONS})",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"Model shortnames or full IDs (default: {DEFAULT_MODELS})",
    )
    parser.add_argument("--K", type=int, default=DEFAULT_K, help=f"Runs per experiment (default: {DEFAULT_K})")
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
        help=f"max-workers passed to each flow (default: {DEFAULT_MAX_WORKERS})",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=3,
        help="Max flows running at once (default: 3)",
    )
    args = parser.parse_args()

    asyncio.run(run_matrix(
        injections=args.injections,
        models=args.models,
        K=args.K,
        max_workers=args.max_workers,
        max_concurrent=args.max_concurrent,
    ))


if __name__ == "__main__":
    main()
