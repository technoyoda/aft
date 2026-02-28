# TODO: Tutorial 3: Environment Shapes the Field

**The idea:** The environment shapes the field. Same agent, same prompt, different data -- different behavioral distribution.

An agent fetches employee records from a REST API, computes summary statistics (mean salary, headcount per department, total payroll), and writes the results to `report.txt`. The API is a Flask server we control.

Two arms:
- **Clean:** All employee data is correct.
- **Poisoned:** One employee's salary is `5800000` instead of `58000` (cents vs dollars). Subtle but detectable.

The question: does the agent behave differently when the environment contains bad data, even though the prompt is identical? The field captures that difference.

## The framing problem

When you change the environment but keep everything else constant, any behavioral shift is caused by the environment. The dimensions must capture behaviors that might change in response to bad data: did the agent validate the data? Did it make more API calls? Did it spend more time processing vs reporting? The measure function is a hypothesis about where the environment exerts influence.

## Running

```bash
cd examples/tutorial-3
pip install -r requirements.txt

# Test the Flask server standalone
python api_server.py --data api_data_clean.json --port 5050
# In another terminal: curl http://localhost:5050/records

# Run the experiment (K=5 per arm, 10 total)
python data_flow.py run

# Fewer runs for a quick test
python data_flow.py run --K 2

# Specific model
python data_flow.py run --model claude-haiku-4-5-20251001
```

The comparative card is viewable in the Metaflow UI.

## What to look for

- **data_validation dimension:** In the clean arm, the agent probably doesn't bother validating. In the poisoned arm, some trajectories may detect the outlier and validate. If the separation vector on `data_validation` is large in the poisoned arm, validation predicts success.
- **Width comparison:** The poisoned arm should have higher width -- the bad data introduces behavioral diversity as some agents catch the error and others don't.
- **api_calls:** More API calls in the poisoned arm may indicate re-fetching or cross-checking.

## Querying past runs

```python
from metaflow import Flow
from agent_fields import FieldMetrics
from data_field import DataProcessingField
import numpy as np

run = Flow('EnvShapeFlow').latest_run
dims = DataProcessingField().dimensions()

for step_name in ['join_clean', 'join_poisoned']:
    task = next(iter(run[step_name]))
    d = task.data
    m = FieldMetrics(np.array(d.field_points), np.array(d.field_outcomes), dims)
    print(f"{d.arm_label}: width={m.width():.4f}, "
          f"convergence={m.convergence():.4f}, "
          f"success={sum(d.outcomes):.0f}/{len(d.outcomes)}")
```

## Files

| File | What it does |
|------|-------------|
| `data_flow.py` | Metaflow flow -- two-arm ablation (clean vs poisoned) with comparative card |
| `data_field.py` | `DataProcessingField` -- 7 dimensions + `state()` tracking data processing phases |
| `api_server.py` | Flask API server -- serves employee records from a JSON file |
| `api_data_clean.json` | 10 employee records, all correct |
| `api_data_poisoned.json` | Same records, one salary 100x too high |
| `verify_report.py` | Checks agent's report.txt against expected values |


## TODO: [NEED TO MAKE BETTER ADVERSARIAL EXAMPLE; BOILER PLATE SETUP. I NEED TO THINK MORE]