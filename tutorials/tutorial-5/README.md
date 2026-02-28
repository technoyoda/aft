# Tutorial 5: measure() is Your Hypothesis

**The idea:** Same data, different `measure()`, different field. The measurement function encodes your hypothesis about what matters. Change it and the answer changes.

This tutorial defines two Field subclasses that measure the same trajectories from tutorial-1 differently:

- **StructuralField:** What the agent *did* -- tool counts, scope ratio, bugs addressed. Counts and ratios.
- **StrategyField:** How the agent *approached* it -- exploration ratio, commit speed, direction changes, verification effort. Behavioral strategy.

Both fields are built from identical trajectory data. They produce different widths, different convergence values, different separation vectors. One may show a clear signal where the other shows noise.

## The framing problem

A field with no separation signal doesn't mean the agent is unpredictable. It might mean you're measuring the wrong things. If your `measure()` captures dimensions that don't vary with outcome, separation is flat. If you switch to dimensions that track strategic choices, separation might light up. The field told you your hypothesis was wrong -- not by failing, but by showing no signal.

## Prerequisites

A completed tutorial-1 run:

```bash
cd ../tutorial-1
python agent_field_flow.py run
```

## Running

```bash
cd examples/tutorial-5

# Uses latest GelloFlow run
python dual_measure.py

# Or specify a run ID
python dual_measure.py --run-id 12345
```

## What to look for

The script prints variance and separation per dimension for both fields side by side. Compare:

- **Which field has higher convergence?** That field's dimensions are better predictors of outcome.
- **Which dimensions have large separation?** Those are the behavioral properties that actually distinguish success from failure.
- **Does one field have no separation signal at all?** Its hypothesis is wrong for this task -- those dimensions don't predict success.

## Files

| File | What it does |
|------|-------------|
| `dual_measure.py` | Defines StructuralField and StrategyField, builds both from tutorial-1 data |
