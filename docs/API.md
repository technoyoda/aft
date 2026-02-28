# API Reference

This document covers the three core objects in AFT — `Dimension`, `Field`, and `FieldMetrics` — what they compute, how to interpret them, and when to use them.

For the formal derivation behind these objects, see [**math.md**](./math.md).

---

## `Dimension`

A named axis of the behavioral space. Each dimension returned by `measure()` has a corresponding `Dimension` that makes metrics interpretable — "variance on `scope_ratio`" instead of "variance on dimension 5."

```python
from agent_fields import Dimension

Dimension("num_tool_calls", "Total number of tool invocations")
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Short identifier used as key in summaries |
| `description` | `str` | What this dimension measures — written for a human reading the summary |

---

## `Field`

The central object in AFT. A `Field` is a point cloud of behavioral vectors derived from agent trajectories, with outcome labels painted on each point as a separate quantity. The cloud describes *what trajectories did*. The labels describe *how well they did*. This separation is what lets you ask *where in behavioral space do good outcomes cluster?* without conflating behavioral diversity with outcome noise.

Formally, the field is the empirical approximation of the distribution over trajectories $\mathcal{F}(E, c_0) \coloneqq P_{\mathcal{M}}(\tau \mid E, c_0)$ induced by a configuration — see [math.md §2](./math.md#2-the-empirical-field-as-a-distribution). We never have the analytical form. We have samples.

### Subclassing

You subclass `Field` and implement two methods:

```python
import agent_fields as aft
import numpy as np

class MyField(aft.Field):
    def dimensions(self):
        return [
            aft.Dimension("tool_calls", "Total tool invocations"),
            aft.Dimension("edits", "Number of file edits"),
        ]

    def measure(self, trajectory):
        # trajectory is YOUR data — any shape
        tool_calls = len([s for s in trajectory if s["kind"] == "tool_call"])
        edits = len([s for s in trajectory if s.get("tool") == "Edit"])
        return np.array([tool_calls, edits])
```

### `measure(trajectory) → np.ndarray`

The measurement function $\varphi: \mathcal{T} \to \mathbb{R}^d$ — see [math.md §3.1](./math.md#31-definition). Maps a trajectory to a fixed-dimensional behavioral vector. Each dimension captures *what the agent did*, not how well it did. Outcome quality belongs in the `outcome` label, not here.

$\varphi$ determines the field. Different measurement functions produce different clouds from the same trajectory data. Same cloud machinery, entirely different interpretation — see [math.md §3.3](./math.md#33-the-choice-of-varphi-determines-the-field).

### `dimensions() → list[Dimension]`

Returns metadata for each component of the vector that `measure()` returns. One `Dimension` per axis. The list order must match the vector indices.

### `add(trajectory, outcome) → np.ndarray`

Measures one trajectory and adds the resulting point to the cloud. Returns the measured vector.

```python
field = MyField()
vec = field.add(run["trajectory"], outcome=run["score"])
```

### `ingest(trajectories, outcomes)`

Bulk `add()`. Takes a list of trajectories and a matching list of outcomes.

### `points → np.ndarray`

The $(K, d)$ point cloud — all measured behavioral vectors stacked. This is the empirical field $\hat{\mathcal{F}}$ from [math.md §3.2](./math.md#32-the-point-cloud):

$$\hat{\mathcal{F}}(E, c_0) \approx \lbrace\varphi(\tau_1),\; \varphi(\tau_2),\; \dots,\; \varphi(\tau_K)\rbrace \subset \mathbb{R}^d$$

### `outcomes → np.ndarray`

The $(K,)$ outcome labels $y(\tau)$ — one per trajectory. These are painted on the cloud, not coordinates of it.

### `K` — number of trajectories

How many points in the cloud. Your sample size.

### `d` — dimensionality

Number of behavioral dimensions, determined by `dimensions()`.

### `metrics() → FieldMetrics`

Compute all field metrics on the current cloud. Returns a `FieldMetrics` object.

### `subset(mask) → Field`

Returns a new `Field` containing only the points matching a boolean mask.

### `success_region(threshold=0.5) → Field`

The subset where $y(\tau) \geq \theta$. This is $\mathcal{S}$ from [math.md §4.5](./math.md#45-derived-success-and-failure-regions).

### `failure_region(threshold=0.5) → Field`

The subset where $y(\tau) < \theta$. This is $\mathcal{R}$ from [math.md §4.5](./math.md#45-derived-success-and-failure-regions).

### `state(trajectory, t) → str`

The state function $\psi$ — reduces the trajectory prefix up to step $t$ into a discrete label representing semantic progress toward the goal. See [math.md §6.1](./math.md#61-the-state-function).

Optional. The default returns a constant (all trajectories share one state). Override to define meaningful task phases and enable horizon analysis.

The state progression should be linear: `"start"` → `"oriented"` → `"fixed"` → `"verified"`. Behavioral variation within a phase is captured by `measure()`, not by subdividing state.

```python
def state(self, trajectory, t):
    steps = trajectory.steps[:t + 1]
    has_read = any(s.tool_name == "Read" for s in steps)
    has_edited = any(s.tool_name == "Edit" for s in steps)
    if has_edited: return "fixed"
    if has_read: return "oriented"
    return "start"
```

### `trajectory_length(trajectory) → int`

Returns the number of steps in the trajectory. Required when `state()` is overridden — the framework calls `state(trajectory, t)` for each `t` in `range(trajectory_length(trajectory))`.

```python
def trajectory_length(self, trajectory):
    return len(trajectory.steps)
```

### `horizon(state) → Field`

The field horizon at one or more states — the sub-field of trajectories that passed through the given state(s). See [math.md §6.2](./math.md#62-the-field-horizon).

**Single state:**

$$\mathcal{H}(s) = \lbrace\varphi(\tau_k) : \exists\, t \text{ s.t. } \psi(\tau_k, t) = s\rbrace$$

```python
h = field.horizon("diagnosed")
m = h.metrics()
print(f"Width from diagnosed: {m.width():.3f}")
print(f"Convergence from diagnosed: {m.convergence():.3f}")
```

**Multiple states** — pass a list of states to treat multiple fine-grained labels as one semantic phase. See [math.md §6.3](./math.md#63-horizon-over-multiple-states).

$$\mathcal{H}(G) = \lbrace\varphi(\tau_k) : \exists\, t,\; \exists\, s \in G \text{ s.t. } \psi(\tau_k, t) = s\rbrace$$

```python
h = field.horizon(["debugging:core_files", "debugging:tests", "debugging:logs"])
```

This is a query-time operation — `state()` stays unchanged. Define states at full resolution, select at analysis time.

Returns a `Field` with full metrics. Same pattern as `success_region()` / `failure_region()`.

### `horizon_at(t) → Field`

The field horizon at step $t$ — the sub-field of trajectories that were still alive (had at least $t+1$ steps) at step $t$. This is the step-index proxy for state — useful when you don't have a `state()` override, or when you want to see the field at a specific point in raw execution time.

```python
for t in range(10):
    h = field.horizon_at(t)
    if h.K >= 2:
        print(f"step {t}: width={h.metrics().width():.3f}")
```

### `states → list[str]`

All states observed across trajectories, in first-seen order.

### `Field.from_arrays(points, outcomes, dimensions) → Field`

Construct a `Field` directly from pre-computed numpy arrays. Use this when you already have embedded vectors and don't need the `measure()` pipeline.

```python
from agent_fields import Field
import numpy as np

field = Field.from_arrays(
    points=np.array([[3, 2, 1], [5, 4, 2]]),
    outcomes=np.array([1.0, 0.0]),
    dimensions=[...],
)
```

---

## [`FieldMetrics`](./METRICS.md)
