# Tutorial 2: Horizon Analysis

**The idea:** Every execution collapses into a chain of labels. The distribution has temporal structure.

An agent fixes the same buggy.py from tutorial 1, but now we track *where in the process* things go wrong. The `state()` function labels each point in the trajectory with a progress phase:

```
start → diagnosed → editing → complete_fix → tested
```

The field at each state is a sub-field with its own width, convergence, and separation. Walking the horizon chain reveals which phase is the bottleneck -- where failing trajectories diverge from successful ones.

## The framing problem

Pass/fail tells you *that* something went wrong. Horizons tell you *where*. If all trajectories reach "diagnosed" but only some reach "complete_fix," the problem is in the fixing phase, not the diagnosis phase. The separation vector at that horizon tells you *what* distinguishes the agents that got through from those that didn't.

## Running

```bash
cd examples/tutorial-2
pip install -r requirements.txt

# Default: K=10 runs with vague prompt (mixed outcomes = interesting horizons)
python horizon_flow.py run

# Fewer runs for a quick test
python horizon_flow.py run --K 5

# Specific model
python horizon_flow.py run --model claude-haiku-4-5-20251001
```

The horizon analysis card (width/convergence/drift charts per state, per-state separation vectors) is viewable in the Metaflow UI.

## What to look for

- **Horizon width by state:** Does the field narrow as trajectories progress? If width increases at a late state, agents are diverging after an initial consensus.
- **Horizon drift by state:** `drift(s) = W(H(s)) - W(H+(s))`. Positive drift means failing trajectories have diverged from the success corridor at that state. The state with highest drift is where you should intervene.
- **Per-state separation:** The separation vector at the critical state tells you which behavioral dimensions matter *at that phase*. This is often different from the global separation vector.

## Querying past runs

```python
from metaflow import Flow, Run
from agent_fields import FieldMetrics
from horizon_field import HorizonCodeFixField
import numpy as np

run = Flow('HorizonFlow').latest_run
task = next(iter(run['analyze']))

# Reconstruct field with state tracking
field = HorizonCodeFixField()
for traj, outcome in zip(task.data.trajectories, task.data.outcomes):
    field.add(traj, outcome)

for s in field.states:
    h = field.horizon(s)
    if h.K >= 2:
        hm = h.metrics()
        print(f"{s:15s}  K={h.K}  width={hm.width():.4f}")
```

## Files

| File | What it does |
|------|-------------|
| `horizon_flow.py` | Metaflow flow -- single-arm with horizon analysis card |
| `horizon_field.py` | `HorizonCodeFixField` -- 8 dimensions + quality-aware `state()` |
| `buggy.py` | The buggy file agents fix (same as tutorial 1) |
| `verify.py` | Verification script (same as tutorial 1) |
