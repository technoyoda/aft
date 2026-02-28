# Tutorial 7: The Field Over Time

**The idea:** Fields are snapshots. Compare snapshots across periods to detect behavioral drift.

Same task (buggy.py fix from tutorial 1), same prompt. Two arms representing two "periods" -- different models by default. The question: did the field shift? If the same task produces a different behavioral distribution with a different model, the field has drifted. The compare step shows which dimensions shifted and whether the change is a regression or just a different strategy.

## The framing problem

Pass rate staying the same doesn't mean behavior stayed the same. A model update might succeed in a completely different way -- fewer reads, more edits, different scope focus. The field captures that. When you see a width increase across periods, behavioral consistency dropped. When you see the center shift, the average strategy changed. Whether that matters depends on which dimensions shifted and whether you care about them.

## Running

```bash
cd examples/tutorial-7
pip install -r requirements.txt

# Default: period A = claude-sonnet-4-5-20250514, period B = claude-haiku-4-5-20251001
python temporal_flow.py run

# Custom models
python temporal_flow.py run \
  --model_a claude-sonnet-4-5-20250514 \
  --model_b claude-haiku-4-5-20251001

# Fewer runs for a quick test
python temporal_flow.py run --K 2

# Same model, different K to check stability
python temporal_flow.py run \
  --model_a claude-sonnet-4-5-20250514 \
  --model_b claude-sonnet-4-5-20250514 --K 10
```

The comparative card is viewable in the Metaflow UI.

## What to look for

- **Width comparison:** Did one period produce more behavioral diversity? Higher width means less consistency.
- **Center shift:** Did the average strategy change? The `compare_center_bar` chart shows this per dimension.
- **Convergence:** Did reliability change? A model update that drops convergence is a behavioral regression even if pass rate holds.
- **Separation vectors per period:** Do different behavioral properties predict success in each period? If so, the *nature* of the problem changed, not just the difficulty.

## Querying past runs

```python
from metaflow import Flow
from agent_fields import FieldMetrics
import numpy as np
import sys, os
sys.path.insert(0, os.path.join('..', 'tutorial-1'))
from field_def import CodeFixField

run = Flow('TemporalFlow').latest_run
dims = CodeFixField().dimensions()

for step_name in ['join_a', 'join_b']:
    task = next(iter(run[step_name]))
    d = task.data
    m = FieldMetrics(np.array(d.field_points), np.array(d.field_outcomes), dims)
    print(f"{d.period_label} ({d.model_used}): "
          f"width={m.width():.4f}, convergence={m.convergence():.4f}")
```

## Files

| File | What it does |
|------|-------------|
| `temporal_flow.py` | Metaflow flow -- two-arm temporal comparison, reuses tutorial-1's CodeFixField |

Also depends on `../tutorial-1/field_def.py`, `../tutorial-1/buggy.py`, `../tutorial-1/verify.py`.
