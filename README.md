# agent_fields

> [blog on why this exists](https://technoyoda.github.io/agent-science.html)

A Python toolkit for studying AI agent behavior without looking inside the model.

> Disclaimer : this is a hobby project. I have a full time job and I am doing this because I find this very interesting. I would love constructive human feedback, otherwise everybody has access to the same policy. 

> This project comes with an [AI notice](./docs/AI_NOTICE.md) BUT its [core libarary](./agent_fields/) only has 400 lines of API surface area (where even most of that code is just docstrings) for other humans and AI. All the things around the core source have been crafted with human love (wall-time >>>> api-time). Most files in [docs](./docs/). 

## Motivation

[Agents are not thinking — they are searching.](https://technoyoda.github.io/agent-search.html). Mordern day AI agents are are searching toward a reward signal, and your environment bounds that search. Pre-training establishes what's reachable — the statistical distribution of language the model can produce. Reinforcement learning determines the navigation strategy — which trajectories through that space the policy favors. The prompt is not an instruction — it's the universe you create for the model, persistently narrowing the space of reachable behaviors from context-start onward.

The original formulation defines three primitives: the **environment** (the real-world state like repo, tools, permissions), the **context window** (everything the model has seen : system prompt, conversation history, tool outputs), and the **field** (the space of reachable behaviors conditioned on the context window plus the trained policy). The field is not static. It shifts every time a token enters the context window. During an inference rollout, the policy produces an action, the environment returns feedback that enters the context, the new tokens reshape the field, and the process repeats. A precise prompt narrows the field. Noise warps it. Permissions bound it.

This per-timestep field is a theoretical object — it would require enumerating every possible future trajectory from every possible context state, which is infinite. **agent_fields (aft)** approximates it empirically. Run an agent K times on the same task, record what it did each time, measure each completed trajectory, and the resulting point cloud is an empirical approximation ; it is not the field at any single timestep, but the distribution of behaviors the field produced across all runs.

**aft** does not prevent bad agent behavior. It is not a guardrail, filter, or constrain. It is an empirical framework for measuring what trajectories actually do. You run trajectories, build a field, and the metrics tell you what's happening: where trajectories succeed, where they fail, what behavioral patterns predict each, and how those patterns shift when you change the environment or the prompt. The interventions come from you — **aft** gives you the data to know where to intervene.

## Related works 
- AI-tasked research [over here](./docs/related.md)

## Math

The formal derivation behind this toolkit — including the definition of the field, the trajectory embedding contract, field metrics, ablation decomposition, the field horizon, and the intent/regime/program family formalization — is in [**math.md**](./docs/math.md).

## Core Idea

1. You run an agent K times on the same task
2. For each run, you record the trajectory (what the agent did) and the outcome (did it work)
3. You define a `measure()` function that extracts a behavioral vector from each trajectory
4. **agent_fields (aft)** builds a point cloud from those vectors and computes metrics on it

The metrics tell you things like:
- **How consistent is the agent?** (field width)
- **How reliably does it succeed?** (convergence)
- **What behavioral dimensions predict success vs failure?** (separation vector)
- **Where does behavior vary most?** (per-dimension variance)

Change the prompt, compare two fields. Change the environment, compare two fields. Swap models, compare two fields. The field is the unit of comparison.

## Install

```bash
pip install git+https://github.com/technoyoda/aft.git
```

## Usage

### 1. Subclass `Field` and implement `measure()`

```python
from agent_fields as aft
import numpy as np

class MyTaskField(aft.Field):
    def dimensions(self):
        return [
            aft.Dimension("files_touched", "Number of distinct files accessed"),
            aft.Dimension("backtracks", "Re-visits to previously seen files"),
            aft.Dimension("tool_calls", "Total number of tool invocations"),
            aft.Dimension("scope_ratio", "Fraction of touched files that are task-relevant"),
        ]

    def measure(self, trajectory):
        # trajectory is YOUR data — whatever shape you have
        files = len(set(s["file"] for s in trajectory if "file" in s))
        backtracks = sum(1 for i, s in enumerate(trajectory)
                         if s.get("file") in [t.get("file") for t in trajectory[:i]])
        tool_calls = len(trajectory)
        relevant = {"main.py", "config.yaml"}
        touched = set(s.get("file") for s in trajectory if "file" in s)
        scope = len(touched & relevant) / len(touched) if touched else 0

        return np.array([files, backtracks, tool_calls, scope])
```

`measure()` and `dimensions()` are what you implement. `measure()` determines which behavioral properties become dimensions of the point cloud — only those properties exist for metrics to operate on. Any behavior not captured by a dimension is absent from the cloud and invisible to every metric downstream. `dimensions()` describes each dimension — name and what it measures. Every dimension should capture a behavioral property — *what the agent did*, not how well it did. Outcome quality goes in the label.

### 2. Feed it trajectories

```python
field = MyTaskField()

for run in my_collected_runs:
    field.add(run["trajectory"], outcome=run["score"])
```

Each call to `add()` measures the trajectory and places it in the point cloud.

### 3. Read the metrics

```python
m = field.metrics()

m.center()       # R^d vector — average behavior
m.width()        # scalar — total behavioral spread
m.variance()     # R^d vector — per-dimension variance
m.convergence()  # scalar — E[outcome] / std[outcome]
m.separation()   # R^d vector — what separates success from failure
m.summary()      # dict — everything above, keyed by dimension name
```

### 4. Compare configurations

```python
# Same task, different prompts
field_a = MyTaskField()  # prompt A
field_b = MyTaskField()  # prompt B

for run in runs_with_prompt_a:
    field_a.add(run["trajectory"], outcome=run["score"])

for run in runs_with_prompt_b:
    field_b.add(run["trajectory"], outcome=run["score"])

ma, mb = field_a.metrics(), field_b.metrics()

print(f"Width:  {ma.width():.3f} vs {mb.width():.3f}")
print(f"Convergence: {ma.convergence():.3f} vs {mb.convergence():.3f}")
```

Width went down? Trajectories are more consistent. Convergence went up? More runs succeed. The separation vector shifted? Different behaviors are driving success now.

## API

### `Dimension`

| Field | Description |
|-------|-------------|
| `name: str` | Short identifier used as key in summaries |
| `description: str` | What this dimension measures |

### `Field` (abstract)

| Method | You implement | Description |
|--------|:---:|-------------|
| `measure(trajectory) -> np.ndarray` | yes | Map a trajectory to a behavioral vector |
| `dimensions() -> list[Dimension]` | yes | Metadata for each dimension of the vector |
| `add(trajectory, outcome)` | | Measure + store one trajectory |
| `ingest(trajectories, outcomes)` | | Bulk add |
| `metrics() -> FieldMetrics` | | Compute all metrics on the current cloud |
| `points -> np.ndarray` | | The (K, d) point cloud |
| `outcomes -> np.ndarray` | | The (K,) outcome labels |
| `state(trajectory, t) -> str` | optional | Discrete task-progress label at step t (monotonic) |
| `intent(trajectory, t) -> str` | optional | Policy's operational character at step t (non-monotonic) |
| `trajectory_length(trajectory) -> int` | when state/intent overridden | Number of steps in the trajectory |
| `horizon(state)` | | Sub-field of trajectories through a state |
| `horizon_at(t)` | | Sub-field of trajectories alive at step t |
| `states -> list[str]` | | All observed state labels |
| `regime(pattern)` | | Sub-field by intent pattern presence (overlapping) |
| `program_family(prefix)` | | Sub-field by program prefix (partitioning) |
| `intents -> list[str]` | | All observed intent labels |
| `programs -> list[tuple]` | | All distinct program strings |
| `success_region(threshold)` | | Subset: only successful trajectories |
| `failure_region(threshold)` | | Subset: only failed trajectories |
| `Field.from_arrays(points, outcomes, dimensions)` | | Construct from pre-computed arrays |

### `FieldMetrics`

| Method | Returns | Description |
|--------|---------|-------------|
| `center()` | R^d vector | Centroid of the point cloud |
| `width()` | scalar | Trace of covariance (total spread) |
| `variance()` | R^d vector | Per-dimension variance |
| `covariance()` | (d, d) matrix | Full covariance |
| `convergence()` | scalar | E[y] / std[y] |
| `skew(cost_dim)` | scalar | Correlation between outcome and a cost dimension |
| `separation(threshold)` | R^d vector | Centroid of successes minus centroid of failures |
| `summary()` | dict | All of the above, keyed by dimension name |

## Metrics

For a detailed guide to every object in AFT — `Dimension`, `Field`, and `FieldMetrics` — what they compute, how to interpret them, and when to use them — see [**API.md**](./docs/API.md).
