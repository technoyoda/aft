# AFT — Agent Field Theory

This file is written for AI agents. If you are an agent and you have been asked to use this library, or to define a new Field, read this entire file first.

## What this library is

AFT is a measurement framework for AI agent behavior. It does not control, guardrail, or constrain agents. It provides an empirical way to measure what agents do across repeated runs of the same task, so that humans can understand behavioral patterns and make informed decisions about how to improve prompts, environments, and tooling.

## The theory behind it

### Agents are searching, not thinking

The foundational insight (from https://technoyoda.github.io/agent-search.html) is that AI agents do not deliberate. They navigate probabilistic trajectories toward reward signals shaped during training. The prompt constrains which trajectories are reachable. The environment reshapes the space of possible behaviors at every step.

### Three primitives

The original formulation defines three things:

1. **Environment** — the real-world state: repo on disk, tools available, network, permissions. It changes whether or not the agent observes it.
2. **Context window** — everything the model has seen so far: system prompt, conversation history, tool outputs, accumulated tokens. This is the agent's entire world.
3. **Field** — the space of reachable behaviors conditioned on the context window plus the trained policy. This is the theoretical object we are approximating.

### The field shifts at every timestep

The field is not static. Every time a token enters the context window, the field changes. During an inference rollout:

1. The policy produces an action shaped by training and the current context
2. The environment returns feedback that enters the context window
3. The new tokens reshape the field
4. The system prompt persists, continuously constraining the field
5. The process repeats

Each cycle, the context grows and the field shifts. A precise prompt narrows the field. Noise warps it. Permissions bound it from outside. A bad observation at step 2 stays in the context window for every subsequent step, warping the field each time. There is no built-in reasoning module — any token that enters the context window can reshape which behaviors the field makes likely.

### The trained policy is opaque

The trained policy is a black box. You did not design the reward function, you do not know its full specification, and you can only observe the behavior it produces. This is why AFT works entirely from the outside — it reconstructs the field from observed trajectories without ever looking inside the model.

### From theory to approximation

The per-timestep field is a theoretical object. It would require enumerating every possible future trajectory from every possible context state, which is infinite. AFT approximates it empirically: run an agent K times on the same task, record what it did each time, measure each completed trajectory, and the resulting point cloud is an empirical approximation — not the field at any single timestep, but the distribution of behaviors the field produced across all runs.

## The core objects

AFT has three objects: `Field`, `Dimension`, and `FieldMetrics`.

### Dimension

`Dimension` is a frozen dataclass with two fields:

- `name: str` — short identifier used as a key in metrics summaries (e.g., `"scope_ratio"`)
- `description: str` — what this dimension measures, written for a human reading the summary (e.g., `"Fraction of actions on relevant files vs total"`)

### Field

`Field` is the central object. It is an abstract class. Users subclass it and implement two required methods, plus one optional:

- `measure(trajectory) -> np.ndarray` — takes a trajectory (any data type the user has) and returns a fixed-dimensional numpy vector. This is the measurement function. It determines which behavioral properties become dimensions of the point cloud — only those properties exist for metrics to operate on. Any behavior not captured by a dimension is absent from the cloud and invisible to every metric downstream. Each dimension should capture a behavioral property — what the agent *did*, not how well it did.
- `dimensions() -> list[Dimension]` — returns metadata for each dimension. Each Dimension has a name and description that make metrics interpretable.
- `state(trajectory, t) -> str` (optional) — reduces the trajectory prefix up to step t into a discrete label representing semantic progress. The default returns a constant. Override to enable horizon analysis — see "The state function and field horizons" below.
- `trajectory_length(trajectory) -> int` (required when `state()` is overridden) — returns the number of steps in the trajectory. The framework calls `state(trajectory, t)` for each `t` in `range(trajectory_length(trajectory))`. Override this alongside `state()` to tell the framework how many steps your trajectory type has.

Then the user feeds trajectories into the field:

```python
field = MyField()
field.add(trajectory, outcome=1.0)   # measure + store one trajectory
field.add(trajectory2, outcome=0.0)  # another one
field.ingest(many_trajectories, outcomes)  # bulk add
```

The field builds a point cloud internally. Each trajectory becomes a point in R^d (where d is the number of dimensions returned by `measure()`). Outcomes (success/failure scores) are stored as labels on each point — NOT as coordinates of the cloud. This separation is critical.

The field exposes:

- `field.points` — the (K, d) numpy array, the point cloud
- `field.outcomes` — the (K,) numpy array, outcome labels
- `field.K` — number of trajectories
- `field.d` — dimensionality
- `field.metrics()` — compute all metrics (returns a FieldMetrics)
- `field.success_region(threshold)` — the region of the field where trajectories succeeded
- `field.failure_region(threshold)` — the region of the field where trajectories failed
- `field.subset(mask)` — subset by boolean mask
- `field.horizon(state)` — the sub-field of trajectories that passed through a given state (see "The state function and field horizons" below)
- `field.horizon_at(t)` — the sub-field of trajectories that were still alive at step t
- `field.states` — all states observed across trajectories, in first-seen order
- `Field.from_arrays(points, outcomes, dimensions)` — construct from pre-computed arrays

### FieldMetrics

`FieldMetrics` is returned by `field.metrics()`. It computes properties of the point cloud:

- `center() -> np.ndarray` — R^d vector. The centroid of the cloud. The "average behavior."
- `width() -> float` — scalar. Trace of the covariance matrix. Total behavioral spread. High = trajectories vary a lot across runs. Low = trajectories are consistent.
- `variance() -> np.ndarray` — R^d vector. Per-dimension variance. This is the diagnostic breakdown — it tells you WHICH dimensions vary most.
- `covariance() -> np.ndarray` — (d, d) matrix. Full covariance.
- `convergence() -> float` — scalar. E[outcome] / std[outcome]. How reliably trajectories succeed. High = most runs succeed. Low = scattered outcomes.
- `skew(cost_dim) -> float` — scalar. Correlation between outcome and a cost dimension (by name or index). Detects if the agent is taking shortcuts.
- `separation(threshold) -> np.ndarray` — R^d vector. Centroid of successful trajectories minus centroid of failed ones. THIS IS THE MOST ACTIONABLE METRIC. The dimensions with the largest absolute values are the behavioral factors that most strongly predict success vs failure.
- `summary() -> dict` — everything above in a readable dict keyed by dimension name, with descriptions included.

## How to design measure() — the most important decision

The choice of `measure()` determines everything. It determines which behavioral properties become dimensions of the point cloud, and therefore which properties the metrics can detect, compare, and reason about. Any behavior not captured by a dimension is absent from the cloud and invisible to every metric. Different `measure()` functions produce different fields from the same trajectory data.

### Design it backwards

Do NOT start by asking "what features can I extract from a trajectory?" Start by asking:

1. **What decisions do I need to make?** (e.g., "is my prompt good enough?", "does the agent complete all required steps?", "is it getting distracted?")
2. **What metric answers each decision?** (e.g., convergence, step coverage in the center vector, scope ratio variance)
3. **What must measure() capture for that metric to be meaningful?** (e.g., binary step coverage flags, scope containment ratio, backtrack count)

Every dimension in measure()'s output should exist because a specific question demands it.

### What to include

- **Step coverage** — did the agent do the required things? Binary flags (0/1) for each required step.
- **Step ordering** — when in the trajectory did each step happen? Normalized position (0.0 = start, 1.0 = end).
- **Scope containment** — how focused was the agent? Ratio of actions on relevant files vs total actions.
- **Behavioral signals** — backtracking (re-reads, re-edits), verification (did it check its work), tool usage patterns.

### What to EXCLUDE

- **Outcome / quality score** — this goes in the `outcome` parameter of `add()`, NOT in the measure vector. If you put outcome inside measure(), the field width will conflate behavioral diversity with outcome noise, and the separation vector becomes meaningless.
- **Token counts or generation length** — usually not behaviorally meaningful.
- **Specific code content** — measure() captures what the agent *did*, not what it *wrote*. Code correctness belongs in the outcome label.

### measure() is task-specific

A measure() designed for "fix bugs in a Python file" is useless for "deploy a Kubernetes service." This is by design. The formalism (Field, FieldMetrics) is general. The measurement is always specific to the task. When you are asked to create a new Field subclass, study the task first, identify the required steps and behavioral properties that matter, then design measure() to capture exactly those.

## Comparing fields

The primary use of AFT is comparison. You construct two fields under different conditions and compare their metrics:

- **Different prompts, same environment** — which prompt produces more consistent behavior? Higher convergence? Better scope containment?
- **Different environments, same prompt** — does adding a file to the environment help? Does removing permissions narrow the field?
- **Different models, same everything else** — which model is more reliable on this task?

To compare: instantiate two Field objects (same subclass, same measure()), feed each one trajectories from its condition, then compare `metrics()` side by side.

## The state function and field horizons

### The state function

`state(trajectory, t) -> str` is an optional method on Field. It reduces the trajectory prefix up to step t into a discrete label representing semantic progress toward the goal. It answers "where in the task is the agent?" — not "how is it behaving?" (that's what `measure()` does).

The default implementation returns a constant (`"_"`), meaning all trajectories share one state. Override it when you want to slice the field by task phase.

**What `state()` is projecting from:**

Every task has logical structure — phases, dependencies, decision points. "Fix the bugs" implies read, diagnose, fix, verify. That structure might be complex: the agent could approach diagnosis through logs, through tests, through reading the code directly. And that structure is dynamic — every token that enters the context window changes what the agent will do next, effectively reshaping the graph of reachable states at every step.

`state()` projects your understanding of that structure onto the linear trace. The trace already happened — it is a chain. You label positions on that chain using the phases that matter for your analysis. The linearity is a compression choice, not a simplification. You are discarding the complexity of the underlying graph in exchange for a stable labeling that lets you slice the behavioral distribution by phase.

Think of `state()` as an invariant. The underlying graph shifted with every token during execution. The state labels remain stable — "diagnosed" means the agent reached the diagnosis phase, regardless of how it got there. That stability is what makes horizons useful.

**Design principles:**

- **State is discrete.** It returns a string label, not a vector. It captures coarse progress: "started", "oriented", "fixed", "verified".
- **Transitions are linear.** The state progression should be a single chain: start → phase_1 → phase_2 → ... → done. Do NOT design branching state graphs. Behavioral variation within a phase is captured by `measure()`, not by subdividing state.
- **State is a reduce function.** It aggregates over the accumulated context (the trajectory prefix up to step t) to produce a summary label. Think of it as folding over the steps seen so far.
- **Division of labor.** `state()` captures *where* in the task. `measure()` captures *how* the agent behaves. Together they give a complete view — state provides the temporal slice, measurement provides the behavioral detail within that slice.

```python
def state(self, trajectory, t):
    steps = trajectory.steps[:t + 1]
    has_read = any(s.tool_name == "Read" for s in steps)
    has_edited = any(s.tool_name == "Edit" for s in steps)
    has_verified = any(
        s.tool_name == "Bash" and "python" in str((s.tool_input or {}).get("command", ""))
        for s in steps
    )
    if has_verified: return "verified"
    if has_edited: return "fixed"
    if has_read: return "oriented"
    return "start"

def trajectory_length(self, trajectory):
    return len(trajectory.steps)
```

### The field horizon

The field horizon at state s is the sub-field containing only trajectories that passed through state s at some point during execution. It is itself a field — all metrics (center, width, convergence, separation, skew) work on it directly.

```python
h = field.horizon("oriented")
m = h.metrics()
print(f"Width from oriented: {m.width():.3f}")
print(f"Convergence from oriented: {m.convergence():.3f}")
```

Because state transitions are linear, horizons nest naturally:

```
horizon("start") ⊇ horizon("oriented") ⊇ horizon("fixed") ⊇ horizon("verified")
```

Every trajectory that reached "verified" also passed through "fixed", "oriented", and "start". Width typically narrows as you move deeper — trajectories that got further are more behaviorally consistent. When width does NOT narrow at a state, that state is a divergence point — the agent's behavior is scattering there.

**`horizon_at(t)`** is the step-index proxy — the sub-field of trajectories that were still alive (had at least t+1 steps) at step t. Useful when you don't have a `state()` override, or when you want to see the field at a specific point in raw execution time.

```python
for t in range(10):
    h = field.horizon_at(t)
    if h.K >= 2:
        print(f"step {t}: width={h.metrics().width():.3f}")
```

### Horizon over multiple states

Each trajectory unrolls its own state machine. Different trajectories can produce different state labels — one might return `"debugging:core_files"`, another `"debugging:tests"`. These are different labels but they may represent the same semantic phase.

Pass a list of states to `horizon()`:

```python
# All trajectories that were in any "debugging" sub-state
h = field.horizon(["debugging:core_files", "debugging:tests", "debugging:logs"])
m = h.metrics()
```

This is a query-time operation. `state()` stays as-is — define it at full resolution, select at analysis time. Write `state()` once with all the detail you want, then query at whatever level of abstraction the question demands.

Compare broad against fine-grained:

```python
# Broad: all debugging trajectories
h_all = field.horizon(["debugging:core_files", "debugging:tests"])

# Narrow: just the ones debugging via tests
h_tests = field.horizon("debugging:tests")

# Did debugging-via-tests behave differently?
print(f"All debugging width: {h_all.metrics().width():.3f}")
print(f"Tests-only width:    {h_tests.metrics().width():.3f}")
```

### When to implement state()

Implement `state()` when you want to answer questions like:

- "At what point in the task do trajectories start diverging?"
- "Is the agent consistently reaching the verification phase?"
- "Does convergence improve or degrade as the agent progresses?"

Do NOT implement `state()` just because you can. If your analysis only needs the full field metrics, the default (single state) is sufficient. State adds a slicing language on top of the existing measurement — it is optional because `measure()` already captures the superset of behavioral information.

## Example: CodeFixField

Here is a complete Field subclass for a bug-fixing task. Study this pattern when creating new ones.

```python
import agent_fields as aft
import numpy as np

KNOWN_BUGS = ["divide", "average", "first_element"]

class CodeFixField(aft.Field):
    def dimensions(self):
        return [
            aft.Dimension("num_tool_calls", "Total number of tool invocations"),
            aft.Dimension("num_reads", "Number of file read operations"),
            aft.Dimension("num_edits", "Number of file edit operations"),
            aft.Dimension("bugs_addressed", "How many of the 3 known bugs were touched in edits"),
            aft.Dimension("ran_code", "Whether the code was executed to verify fixes (0 or 1)"),
            aft.Dimension("scope_ratio", "Fraction of tool calls targeting the buggy file vs total"),
            aft.Dimension("trajectory_length", "Total number of steps in the trajectory"),
        ]

    def measure(self, trajectory):
        steps = trajectory.steps
        tool_calls = [s for s in steps if s.kind == "tool_call"]
        reads = [s for s in tool_calls if s.tool_name == "Read"]
        edits = [s for s in tool_calls if s.tool_name == "Edit"]

        bugs_addressed = 0
        for edit in edits:
            inp = edit.tool_input or {}
            old = str(inp.get("old_string", "")) + str(inp.get("new_string", ""))
            for bug_fn in KNOWN_BUGS:
                if bug_fn in old:
                    bugs_addressed += 1
                    break

        ran_code = 0.0
        for tc in tool_calls:
            if tc.tool_name == "Bash":
                cmd = str((tc.tool_input or {}).get("command", ""))
                if "python" in cmd or "pytest" in cmd:
                    ran_code = 1.0
                    break

        on_target = 0
        for tc in tool_calls:
            inp = tc.tool_input or {}
            path = str(inp.get("file_path", "") or inp.get("path", ""))
            if "buggy" in path:
                on_target += 1
        scope_ratio = on_target / len(tool_calls) if tool_calls else 0.0

        return np.array([
            len(tool_calls),
            len(reads),
            len(edits),
            bugs_addressed,
            ran_code,
            scope_ratio,
            len(steps),
        ], dtype=float)
```

Notice:
- Every dimension has a name, a description, and a reason to exist
- Outcome is NOT in the vector
- The measurement is specific to this task (it knows about KNOWN_BUGS, it checks for "buggy" in file paths)
- It captures behavior (what the agent did) not quality (whether the code is correct)

## Reading metrics — what to look for

When you compute `field.metrics()`:

1. **Check convergence first.** If it's low, trajectories are unreliable on this task. Everything else is secondary.
2. **Check width.** If it's high, trajectories are varying significantly across runs. Look at per-dimension variance to find where.
3. **Check the separation vector.** The dimensions with the largest absolute values tell you what behavioral factors predict success. This is where interventions should focus.
4. **Check center.** This is the behavioral fingerprint — what does the average run look like? Are required steps being covered?
5. **Check per-dimension variance.** High variance on a specific dimension means that dimension is unreliable. If that dimension also has high separation, it's both important and inconsistent — that's the highest priority intervention target.
