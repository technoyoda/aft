# Tutorial 4: Conditional Questions

**The idea:** You can ask conditional questions. "What separates success from failure?" is different from "What separates success from failure *among trajectories that reached the processing phase*?"

This tutorial is a script, not a flow. It pulls data from a completed tutorial-3 run (EnvShapeFlow) via the Metaflow Client API and asks two different questions of the same data.

## The framing problem

Global separation tells you what matters *on average*. Conditional separation tells you what matters *at a specific point*. An agent that never fetches the API data fails for a different reason than one that fetches, processes, but produces the wrong report. Horizons let you ask: "Among agents that got past fetching, what distinguishes success from failure?" That question has a different answer than the global one.

## Prerequisites

A completed tutorial-3 run:

```bash
cd ../tutorial-3
python data_flow.py run
```

## Running

```bash
cd examples/tutorial-4

# Uses latest EnvShapeFlow run
python conditional_analysis.py

# Or specify a run ID
python conditional_analysis.py --run-id 12345
```

## What to look for

The script prints two separation vectors side by side:

- **Global separation:** Which dimensions separate success from failure across all poisoned-arm trajectories.
- **Conditional separation:** Which dimensions separate success from failure *among trajectories that reached "processing"*.

The delta column shows where the two vectors diverge. Dimensions that matter globally may not matter conditionally, and vice versa. The global view might say "make more API calls." The conditional view (given the agent already fetched data) might say "validate the data." Different question, different answer.

## Files

| File | What it does |
|------|-------------|
| `conditional_analysis.py` | Pulls tutorial-3 data, compares global vs horizon separation |
