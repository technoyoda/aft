# Tutorial 1: Prompt Ablation

An agent is given a buggy Python file with three functions — `divide`, `average`, `first_element` — and asked to fix them. A verification script checks that edge cases return sensible defaults and normal behavior is preserved.

We run the same agent K=5 times with two different prompts:

- **Vague:** "Find all bugs that would cause crashes or exceptions, and fix them."
- **Concise:** Spells out each bug, what input triggers it, and exactly what the fix should return.

Both arms run in parallel using Metaflow. Each builds its own field. A final step pools all trajectories and compares.

## What happened

The concise prompt got 5/5 pass. The vague prompt got 2/5. The 3 failures all chose to raise `ValueError` — a reasonable interpretation of "fix" that doesn't match what the verification script expects. The prompt was ambiguous about what "fix" means, and the agent explored that ambiguity.

## Reading the metrics

### Width

Width is the [trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)) of the [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix) of the point cloud. The covariance matrix is a (d, d) matrix where entry (i, j) is the covariance between dimensions i and j. The trace — the sum of diagonal entries — gives you the total variance across all dimensions in one scalar. It's a standard measure of multivariate dispersion.

Each trajectory becomes a point in d-dimensional space (here, d=8). The covariance matrix captures how those points co-vary — including off-diagonal terms that tell you when dimensions move together. Width discards the off-diagonal structure and keeps only the total spread. For the full picture, `FieldMetric.covariance()` gives you the (d, d) matrix; `FieldMetric.variance()` gives you just the diagonal (per-dimension variances); width is `sum(FieldMetric.variance())`.

- **Concise: 0.00** — every trajectory produced the exact same behavioral vector. Same number of reads, same edits, same messages, same everything. Five runs, five identical points. The cloud is a single point in 8-dimensional space — zero volume, zero width.
- **Vague: 4.88** — the five trajectories landed in different places. Some used more tool calls, some addressed different bugs, some took more messages. The cloud has volume.

A field that collapses to a single point means the prompt left no room for behavioral variation — there was only one path the model could take. A field with large width means the agent explored multiple strategies, which could mean the task is genuinely complex (many valid approaches) or the prompt is ambiguous (many interpretations). Width alone doesn't tell you which — you need convergence and separation to distinguish useful exploration from confusion.

The per-dimension variance (available via `m.variance()`) tells you *where* the spread is. If most of the width comes from one dimension, that's where trajectories diverge. If it's spread evenly, the agent is varying across the board.

### Convergence

Convergence is the mean outcome divided by its standard deviation — a signal-to-noise ratio on success. A high mean is not enough; if outcomes swing between pass and fail, the mean is misleading. Convergence penalizes that variance. It asks: how much of the outcome is signal (consistent success) versus noise (luck)?

- **Concise: ∞** — all outcomes identical (5/5 pass), zero variance. The ratio blows up. Convergence is only informative when there's variance in outcomes.
- **Vague: 0.82** — some pass, some fail. The mean outcome is decent but unreliable.

High convergence means the field reliably produces good outcomes — you can trust the agent on this task. Low convergence means the agent is a coin flip — same prompt, same task, different result each time. Zero means consistent failure. When you change a prompt or swap a model, convergence is the first thing to check: did reliability improve, or did you just get lucky on one run?

### Separation vector

The separation vector is μ+ − μ− : the mean of successful trajectories minus the mean of failed trajectories, computed per dimension. It tells you where successes and failures land differently in behavioral space.

A positive value on a dimension means successful trajectories averaged higher there. A negative value means they averaged lower. A value near zero means that dimension doesn't distinguish the two groups.

This is a diagnostic tool. When you see a large separation on a dimension, it points you to the behavioral property that matters most for outcome. If you're comparing two prompts, two models, or two environments, the separation vector tells you *what changed* in behavioral terms — not just that one is better, but along which axis.

When all trajectories have the same outcome (all pass or all fail), separation is zero everywhere — there's no contrast to measure. The metric only speaks when the field contains both successes and failures.

In this ablation, the **pooled** separation vector is the most useful — all 10 trajectories from both prompts, labeled by actual verification outcome. It factors out the prompt and shows which behavioral properties predict success across the entire experiment.

### Skew

Skew is the Pearson correlation between outcome and a single behavioral dimension. It answers: "is success cheap or expensive on this axis?"

- **Negative skew**: lower values on that dimension correlate with success. Success is cheap — the agent didn't need to do much along this axis. If you see negative skew on a cost dimension like `num_tool_calls`, the failing agents weren't under-resourced — they were lost. Giving them more turns wouldn't help.
- **Positive skew**: higher values correlate with success. Success is expensive — the task demands thoroughness on that axis, and agents that cut corners failed. This tells you to increase the budget or simplify the task.
- **Zero skew**: no relationship. That dimension doesn't predict outcome.

Skew is a per-dimension scalar, so you can check it against any dimension you consider a "cost" — tool calls, messages, time, edits. It tells you whether the bottleneck is resources or strategy, which determines whether the right intervention is more budget or a better prompt.

## Running

We use metaflow on top of the claude code agent so that we get cheap parallelization and easy access to storage of any state on a per-experiment basis. 
<!-- TODO : Add more about metaflow over here. Make it shine in the background while we do fascinating stuff around it -->


```bash
cd examples/tutorial-1
pip install -r requirements.txt

# Default model, K=5
python agent_field_flow.py run

# Specific model
python agent_field_flow.py run --model claude-haiku-4-5-20251001

# More agents
python agent_field_flow.py run --K 10
```

The comparative card (Vega charts + tables) is viewable in the Metaflow UI at the URL printed when the run completes.

## Querying past runs

```python
from metaflow import Flow
from agent_fields import FieldMetrics
from field_def import CodeFixField
import numpy as np

run = Flow('GelloFlow').latest_run
for step_name in ['join_vague', 'join_concise']:
    for task in run[step_name]:
        d = task.data
        m = FieldMetrics(
            np.array(d.field_points),
            np.array(d.field_outcomes),
            CodeFixField().dimensions(),
        )
        print(f"{d.prompt_label}: width={m.width():.2f}, "
              f"convergence={m.convergence():.2f}")
```

## Files

| File | What it does |
|------|-------------|
| `agent_field_flow.py` | The Metaflow flow — two-arm ablation DAG with comparative card |
| `field_def.py` | `CodeFixField` — 8-dimensional measure on raw dict trajectories |
| `buggy.py` | The static buggy file agents are given |
| `verify.py` | Verification script — tests edge cases and normal behavior |
