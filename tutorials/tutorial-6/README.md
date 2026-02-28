# Tutorial 6: The Oracle as a Ruler

**The idea:** `measure()` can use an LLM to score semantic behavioral properties that can't be computed with string matching or counters.

Some dimensions -- "was the reasoning coherent?", "did the agent maintain a clear strategy?" -- require judgment to evaluate. An LLM can provide that judgment. You decide what to measure (the dimensions). The oracle decides the value (the score).

## The framing problem

When do you need a model to compute dimensions vs when is feature extraction enough? "Number of tool calls" is structural -- a counter works. "Did the agent's strategy stay coherent after encountering an error?" is semantic -- you need a model. The oracle computes behavioral dimensions, not outcome labels. Outcomes go in `y(tau)`, not in `phi(tau)`.

## Prerequisites

A completed tutorial-1 run and an `ANTHROPIC_API_KEY` set:

```bash
cd ../tutorial-1
python agent_field_flow.py run

export ANTHROPIC_API_KEY=sk-ant-...
```

## Running

```bash
cd examples/tutorial-6

# Uses latest GelloFlow run; scores are cached to .oracle_score_cache.json
python oracle_analysis.py

# Or specify a run ID
python oracle_analysis.py --run-id 12345
```

First run calls the Anthropic API (using claude-haiku-4-5-20251001) to score each trajectory. Subsequent runs use the cache.

## What to look for

The script prints two fields side by side:

1. **Hand-crafted (CodeFixField):** Structural dimensions -- tool counts, scope, bugs addressed.
2. **Oracle (OracleField):** Semantic dimensions -- reasoning coherence, strategy focus, error recovery, task understanding. Scored by an LLM reading each trajectory.

Compare:
- Do the semantic dimensions reveal something the structural ones miss?
- Do different separation patterns emerge? That means the oracle is surfacing behavioral signals that counters can't capture.
- Where does skew differ? The oracle may identify dimensions where success is cheap (low effort correlates with good outcomes) that the structural field misses entirely.

## Files

| File | What it does |
|------|-------------|
| `oracle_analysis.py` | Builds CodeFixField and OracleField on tutorial-1 data, compares metrics |
| `oracle_field.py` | `OracleField` -- LLM-scored 4 dimensions with caching, oracle `state()` |
