# Proposal: Field Horizon

> **Status:** Active discussion
> **Scope:** Defining what the field horizon is, what it means, and how it composes with the existing field machinery.

---

## 1. The Idea

The field we build today is a cloud of terminal states — $K$ trajectories, each measured by $\varphi$ at completion, outcomes painted on top. `FieldMetrics` operates on this cloud. It answers: "across all runs, what did the agent do and how well did it work?"

But this is a view from the end. We want to look from the middle: **given that the agent has reached a certain state, what does its future look like?**

The **field horizon** at a state $s$ is the sub-field containing only trajectories that passed through $s$. It's a field — same object, same cloud machinery, same metrics — but scoped to a particular moment in the agent's journey.

---

## 2. What Is State?

The agent is a black box. We can't read its internal state. But we can observe what it has done — the trajectory so far. State is what we extract from the prefix.

Consider our CodeFixField dimensions applied to a prefix:

| Dimension | Complete trajectory | Prefix at step 5 |
|-----------|-------------------|-------------------|
| `num_tool_calls` | 8 | 3 |
| `num_reads` | 3 | 1 |
| `num_edits` | 2 | 1 |
| `bugs_addressed` | 3 | 1 |
| `ran_code` | 1 | 0 |
| `scope_ratio` | 0.75 | 0.80 |

The complete-trajectory measurement is the terminal state. The prefix measurement is the intermediate state. **$\varphi$ has always been a state function — we just only evaluated it at the end.**

But the question is whether $\varphi$ (designed for terminal measurement) is the right lens for intermediate state. The answer: not necessarily. "Number of bugs addressed" works both as terminal measurement ("how thorough was the agent?") and as intermediate state ("how much progress has the agent made?"). But a dimension like "did the agent run the code to verify?" is meaningful at the end but trivially zero at step 3 — it hasn't gotten there yet. That zero doesn't describe state; it describes absence of state.

This motivates separating the two functions.

---

## 3. Two Functions on the Field

The field already has $\varphi$ — the measurement function. We introduce $\psi$ — the **state function**.

$$\varphi: \mathcal{T} \to \mathbb{R}^d \qquad \text{(measurement — what did the agent do?)}$$

$$\psi: \mathcal{T} \times \mathbb{N} \to \mathcal{S} \qquad \text{(state — where is the agent now?)}$$

$\varphi$ takes a complete trajectory and returns a behavioral vector. This is what builds the cloud. It exists today.

$\psi$ takes a trajectory and a position $t$, and returns a state label $s$. The state space $\mathcal{S}$ is defined by the user — it could be discrete labels, a vector, or a categorical partition. The field doesn't prescribe the form; it requires the interface: trajectory + position in, state out.

$\psi$ and $\varphi$ are related but distinct. They might share dimensions (both care about `bugs_addressed`) but they serve different roles:

- $\varphi$ is the lens on the **cloud** — it determines the coordinate system of the field.
- $\psi$ is the lens on the **path** — it determines how trajectories are grouped for horizon construction.

---

## 4. The Field Horizon

### 4.1 Definition

Given a state $s \in \mathcal{S}$, the field horizon at $s$ is:

$$\mathcal{H}(s) = \lbrace\varphi(\tau_k) : \exists\, t \text{ s.t. } \psi(\tau_k, t) = s\rbrace$$

In words: the cloud of terminal measurements, but only for trajectories that passed through state $s$ at some point.

$\mathcal{H}(s)$ is a sub-field. It has all the same structure as the parent field — a point cloud in $\mathbb{R}^d$, outcome labels, dimensions. `FieldMetrics` applies directly. No new metric objects needed.

### 4.2 What the Horizon Tells You

The metrics of $\mathcal{H}(s)$ answer: **given that the agent reached state $s$, what happens next?**

$$W_{\mathcal{H}(s)} \quad \text{— how diverse are the futures from state } s \text{?}$$

$$C_{\mathcal{H}(s)} \quad \text{— how reliably do trajectories through } s \text{ succeed?}$$

$$\mu_{\mathcal{H}(s)}^{+} - \mu_{\mathcal{H}(s)}^{-} \quad \text{— among trajectories through } s \text{, what separates success from failure?}$$

A narrow horizon with high convergence at state $s$ means: if the agent reaches $s$, it's on track. The future is constrained and it's constrained toward success.

A wide horizon with low convergence at state $s$ means: reaching $s$ is not enough. From $s$, the agent can go many directions and most of them fail. The state is a fork — what happens *after* $s$ is what matters.

### 4.3 Comparing Horizons

$$\Delta W = W_{\mathcal{H}(s_1)} - W_{\mathcal{H}(s_2)}$$

"Trajectories that read the file first ($s_1$) have a narrower horizon than trajectories that started editing immediately ($s_2$)."

$$\Delta C = C_{\mathcal{H}(s_1)} - C_{\mathcal{H}(s_2)}$$

"Trajectories that pass through $s_1$ are more reliably successful than those through $s_2$."

These are actionable statements about strategy. They tell you which intermediate states lead to good outcomes — which is the same as telling you what the agent should do next.

### 4.4 The Drift Detector (Revised)

Previously defined as $\delta(t) = W_t - \tilde{W}_t$ (step-indexed). Now conditioned on state:

$$\delta(s) = W_{\mathcal{H}(s)} - W_{\mathcal{H}^+(s)}$$

where $\mathcal{H}^+(s)$ is the horizon restricted to successful trajectories.

"At state $s$, the overall horizon is wider than the success-only horizon." The extra width comes from trajectories that passed through $s$ but later failed — they diverged from the success corridor after $s$.

This is more meaningful than the step-indexed version because states that "look the same" (per $\psi$) are genuinely comparable. You're not comparing step 5 of a fast trajectory with step 5 of a slow one.

---

## 5. Composability

The key property: **the field horizon at state $s$ is itself a field.** This means:

1. You can compute any `FieldMetrics` on a horizon — center, width, variance, convergence, skew, separation. No new metric objects.

2. You can take a horizon and compute *its* horizon. If the agent passed through state $s_1$ and later through state $s_2$, the horizon of the horizon is the sub-field of trajectories that passed through both. This is nested conditioning:

$$\mathcal{H}(s_2 \mid s_1) = \lbrace\varphi(\tau_k) : \exists\, t_1 < t_2 \text{ s.t. } \psi(\tau_k, t_1) = s_1 \text{ and } \psi(\tau_k, t_2) = s_2\rbrace$$

"Given that the agent read the file first ($s_1$) and then found the bug ($s_2$), what's the distribution of outcomes?" This is the horizon of a horizon — composability through nesting.

3. The existing `success_region()` and `failure_region()` are horizons too — they're just conditioned on the terminal outcome rather than an intermediate state. The pattern is the same: filter the cloud by a predicate, get a sub-field.

4. The parent field is the trivial horizon — $\mathcal{H}(\text{start})$ where every trajectory passes through the start state. The hierarchy is:

$$\mathcal{F} \supseteq \mathcal{H}(s_1) \supseteq \mathcal{H}(s_2 \mid s_1) \supseteq \cdots$$

Each level of nesting is a more specific conditioning. The field narrows as you condition on more states, until eventually you've conditioned on so much of the trajectory that the horizon contains a single point.

---

## 6. The Role of $t$

Step index $t$ is not gone — it's reframed. $t$ is a proxy for state when you don't have a meaningful $\psi$:

$$\psi_{\text{step}}(\tau, t) = t$$

This is the simplest possible state function: the state is just "how many steps have happened." It's crude — it groups trajectories by position, not by what they've done — but it's zero-configuration and works when trajectories are structurally similar.

Better state functions use the trajectory content:

$$\psi_{\text{milestone}}(\tau, t) = \text{last milestone reached by step } t$$

$$\psi_{\text{prefix}}(\tau, t) = \varphi(\tau^{\text{pre}}_t)$$

The prefix-measurement state function ($\psi = \varphi$ applied to the prefix) is interesting because it means state and measurement share the same space. The state at step $t$ is a point in $\mathbb{R}^d$, and the terminal measurement is also a point in $\mathbb{R}^d$. The trajectory traces a path from the origin to its terminal point, and the horizon at any point along that path is a sub-field of the terminal cloud.

But as discussed in Section 2, $\varphi$ may not be the right lens for intermediate state. The option to define $\psi$ separately from $\varphi$ is the escape hatch.

---

## 7. Decisions

### 7.1 State is discrete

$\psi(\tau, t)$ returns a discrete label — a string, an enum, a hashable value. Grouping trajectories by state is exact equality, not proximity. No kernels, no clustering, no radius.

This is a practical choice. The theoretical state space of an agent is vast — the full context window is the true state. We are not trying to represent that. We are giving the user a way to say "at this point in the trajectory, the agent is in situation X" where X is a human-legible label they define. The labels are coarse projections of the true state, chosen to be operationally useful.

Examples of what $\psi$ might return:

- `"read_file"` — the agent has read the target file
- `"first_edit"` — the agent has made its first edit
- `"verified"` — the agent has run the code
- `"exploring"` — the agent is still reading files without editing

The coarseness is a feature. A discrete label like `"first_edit"` groups all trajectories that reached their first edit — regardless of which edit, which step, how long it took. The horizon $\mathcal{H}(\text{"first\_edit"})$ answers: "among all trajectories that made a first edit, what does the distribution of final outcomes look like?" That's a useful question precisely because it's coarse.

### 7.3 State as a reduce function

$\psi$ is a **reduce** over the context. At any point $t$, the agent's context window contains every observation and action so far — a growing, high-dimensional object. $\psi$ aggregates all of that into a single discrete label that answers one question: **how far along the semantic execution toward the goal has the agent reached?**

This is lossy by design. The full context is the true state; the label is a coarse projection of it. But the projection is chosen to capture *progress* — not what the agent has done (that's $\varphi$), but where in the logical path toward the goal the agent is.

### 7.4 Linear transitions

The state sequence is linear:

$$s_0 \to s_1 \to s_2 \to \cdots \to s_n$$

"Not started" → "oriented" → "diagnosed" → "fixed" → "verified"

The task has a logical progression — a semantic ordering of what needs to happen to reach the goal. Different agents traverse this progression at different speeds, with different behaviors at each phase, but the progression itself is a line. States are monotonic — once you've diagnosed, you don't go back to not having diagnosed.

This is a deliberate constraint. The agent's *behavior* may be complex — it can read files, backtrack, try things, abandon approaches. That complexity is captured by $\varphi$, not by $\psi$. The state stays simple. The measurement stays rich.

### 7.5 The division of labor

$\psi$ and $\varphi$ together give the full picture:

| | $\psi$ (state) | $\varphi$ (measurement) |
|---|---|---|
| **Answers** | *Where* in the task is the agent? | *How* is the agent behaving? |
| **Output** | Discrete label | Vector in $\mathbb{R}^d$ |
| **Dynamics** | Linear progression toward goal | High-dimensional, varies freely |
| **Captures** | Semantic progress | Strategy, cost, variation |
| **Example** | `"diagnosed"` | `[tool_calls=6, reads=3, edits=0, scope=0.8]` |

The state says "the agent is in the diagnosis phase." The measurement says "within that phase, it's read 3 files, made 0 edits, and 80% of its actions target the right files." Together: *where* and *how*.

Sub-state nuance — "within diagnosis, did the agent read the right file or the wrong file?" — is captured by measurement dimensions, not by subdividing the state. This keeps $\psi$ coarse and linear, and $\varphi$ high-dimensional and detailed.

### 7.6 Nested horizons revisited

Because states are linear, the horizons are naturally nested:

$$\mathcal{H}(s_n) \subseteq \mathcal{H}(s_{n-1}) \subseteq \cdots \subseteq \mathcal{H}(s_0) = \mathcal{F}$$

Every trajectory that reached "verified" also passed through "diagnosed" and "oriented." The parent field is $\mathcal{H}(s_0)$ — the horizon at the start state, which contains all trajectories.

This nesting means you can track how the field narrows as the agent progresses:

- $W_{\mathcal{H}(\text{"oriented"})}$ — how diverse are futures from orientation?
- $W_{\mathcal{H}(\text{"diagnosed"})}$ — how diverse are futures from diagnosis?
- $W_{\mathcal{H}(\text{"fixed"})}$ — how diverse are futures after the fix?

If width drops sharply from "oriented" to "diagnosed," the diagnosis phase is where trajectories commit to a strategy. If width stays high through "fixed," the fix itself is where variation lives — agents fix different bugs, or fix them differently.

### 7.7 The emergent graph

The user doesn't need to predefine the full state graph. They define a labeling function — $\psi(\tau, t)$ returns a discrete label at each point. The graph emerges from the observed trajectories:

```python
def state(self, trajectory, t):
    steps = trajectory.steps[:t+1]
    has_read = any(s.tool_name == "Read" for s in steps)
    has_edited = any(s.tool_name == "Edit" for s in steps)
    has_run = any(s.tool_name == "Bash" for s in steps)

    if has_run: return "verified"
    if has_edited: return "fixed"
    if has_read: return "oriented"
    return "start"
```

The user writes a simple reduce function. The field applies it to every trajectory at every step. The transitions — which states lead to which — materialize from the data. If 4 out of 5 trajectories go `start → oriented → fixed → verified` and 1 goes `start → oriented → verified` (skipping the edit), that's visible in the horizon structure without the user having to anticipate it.

### 7.8 On naming

We're calling it "state" for now. It might not be the right word — "state" in RL means the full Markov state, which this is not. What $\psi$ returns is closer to a **phase** — a coarse landmark in the trajectory's progress toward the goal. But "state" is understood and unambiguous enough to build with. We can rename later if something better emerges.

---

## 8. Relationship to the Transfer Function

[EXTENSIONS.md](./EXTENSIONS.md) defines $\Psi: (E, c_0) \to (W_{\mathcal{F}}, C_{\mathcal{F}}, \mu_{\mathcal{F}})$ — mapping a configuration to field properties. One configuration in, one set of properties out. This is a static map. It doesn't know anything about what happens *during* execution.

With horizons, the transfer function gains a new input:

$$\Psi: (E, c_0, s) \to (W_{\mathcal{H}(s)}, C_{\mathcal{H}(s)}, \mu_{\mathcal{H}(s)})$$

Because states are linear, this isn't just "one more input." For a fixed $(E, c_0)$, it produces a **sequence** of predictions along the state progression:

$$\Psi(E, c_0, s_0),\; \Psi(E, c_0, s_1),\; \dots,\; \Psi(E, c_0, s_n)$$

This is the transfer function unrolled over state. It tells you how field properties evolve as the agent progresses through the task. Width should decrease (futures narrow). Convergence should increase (outcomes become more predictable). When they don't — when width increases at some state or convergence drops — that state is where things go wrong.

### 8.1 State-conditioned ablation

The ablation decomposition (math.md §5) becomes state-dependent. Today:

$$\Delta W = W_{\mathcal{F}}(E^+_{\text{tests}}) - W_{\mathcal{F}}(E^-_{\text{tests}})$$

"Does adding tests narrow the field overall?" With horizons:

$$\Delta W(s) = W_{\mathcal{H}(s)}(E^+_{\text{tests}}) - W_{\mathcal{H}(s)}(E^-_{\text{tests}})$$

"Does adding tests narrow the field *for agents in state $s$*?"

A factor might have no effect on the overall field but a large effect at a specific state. Tests might not change overall width — but at the `"fixed"` state, they might dramatically narrow the horizon, because agents that have fixed the bug and then run tests get immediate pass/fail feedback, while agents without tests wander after their fix. That insight is invisible in the global $\Psi$ but visible in the state-conditioned $\Psi$.

The relationship is: **the horizon gives the transfer function temporal resolution.** The global transfer function is a summary. The state-conditioned transfer function is the full story.

---

## 9. Implementation

### 9.1 The `Field` contract

`measure()` and `dimensions()` remain the two abstract methods — they define the field. `state()` is a first-class method with a default, not abstract. The measurement captures the superset of information; state adds an optional slicing language on top.

```python
class Field(ABC, Generic[T]):

    # ── Abstract: the user MUST implement these ──────────────

    @abstractmethod
    def measure(self, trajectory: T) -> np.ndarray:
        """phi — map a trajectory to a behavioral vector. What the agent did."""
        ...

    @abstractmethod
    def dimensions(self) -> list[Dimension]:
        """Metadata for each dimension of the behavioral vector."""
        ...

    # ── Optional: the user CAN override for horizon analysis ─

    def state(self, trajectory: T, t: int) -> str:
        """psi — reduce the prefix up to step t into a discrete state label.

        This answers "where in the task is the agent?" It is a reduce
        function over the context accumulated so far — a lossy compression
        of the full prefix into a progress marker.

        The state progression should be linear: start → phase_1 → ... → done.
        Behavioral variation within a phase is captured by measure(), not by
        subdividing state.

        The default returns a constant — all trajectories share one state,
        which means horizon() returns the full field. Override to define
        meaningful task phases.

        Args:
            trajectory: the full trajectory.
            t: the step index (0-based).

        Returns:
            A discrete label (any hashable, typically a string).
        """
        return "_"

```

Fields that don't override `state()` work exactly as they do today. `horizon()` is available but returns the full cloud (since every trajectory shares the same default state). When the user adds a `state()` override, horizon analysis activates — no other changes needed.

### 9.2 State sequence computed at ingestion

The state sequence for each trajectory is computed once, at `add()` time, and stored alongside the point and outcome. The trajectory is an iterator — the field walks it, calls `state()` at each step, and records the result. No need for a `trajectory_length()` method.

```python
class Field(ABC, Generic[T]):

    def __init__(self):
        self._points: list[np.ndarray] = []
        self._outcomes: list[float] = []
        self._raw: list[T] = []
        self._state_sequences: list[list[str]] = []   # NEW

    def add(self, trajectory: T, outcome: float) -> np.ndarray:
        vector = self.measure(trajectory)
        self._points.append(vector)
        self._outcomes.append(outcome)
        self._raw.append(trajectory)

        # Walk the trajectory once, record the state at each step
        seq = []
        for t, _ in enumerate(trajectory):
            seq.append(self.state(trajectory, t))
        self._state_sequences.append(seq)

        return vector
```

The trajectory just needs to be iterable — `for t, _ in enumerate(trajectory)` gives us the step indices. The user's type (`list`, a dataclass with a `steps` list, any iterable) works naturally. No length function, no sizing contract.

### 9.3 `horizon()` on `Field`

The field yields its own horizons. Same pattern as `success_region()` and `failure_region()` — filter the cloud by a predicate, return a sub-field.

```python
class Field(ABC, Generic[T]):

    # ... (existing: add, ingest, points, outcomes, metrics,
    #      subset, success_region, failure_region)

    def horizon(self, state: str) -> _MaterializedField:
        """The field horizon at state s.

        Returns the sub-field containing only trajectories that passed
        through state s at some point. The sub-field has the same
        dimensions and the same terminal measurements — it's a filtered
        view of the parent cloud.
        """
        mask = self._trajectories_through(state)
        return _MaterializedField(
            self.points[mask],
            self.outcomes[mask],
            self.dimensions(),
        )

    @property
    def states(self) -> list[str]:
        """All states observed across trajectories, in first-seen order."""
        seen = {}
        for seq in self._state_sequences:
            for s in seq:
                if s not in seen:
                    seen[s] = len(seen)
        return list(seen.keys())

    def _trajectories_through(self, state: str) -> np.ndarray:
        """Boolean mask: which trajectories passed through this state."""
        mask = np.zeros(self.K, dtype=bool)
        for k, seq in enumerate(self._state_sequences):
            if state in seq:
                mask[k] = True
        return mask
```

### 9.3 Example: `CodeFixField`

Two required methods (`measure`, `dimensions`), one optional (`state`).

```python
class CodeFixField(Field[CapturedTrajectory]):

    def dimensions(self) -> list[Dimension]:
        return [
            Dimension("num_tool_calls", "Total tool invocations"),
            Dimension("num_reads", "File read operations"),
            Dimension("num_edits", "File edit operations"),
            Dimension("bugs_addressed", "Known bugs touched in edits"),
            Dimension("ran_code", "Whether code was executed to verify"),
            Dimension("scope_ratio", "Fraction of tool calls on target file"),
            Dimension("trajectory_length", "Total steps"),
            Dimension("escaped_cwd", "Whether agent left working directory"),
        ]

    def measure(self, trajectory: CapturedTrajectory) -> np.ndarray:
        # ... (unchanged — the 8-dimensional behavioral vector)
        ...

    def state(self, trajectory: CapturedTrajectory, t: int) -> str:
        steps = trajectory.steps[:t + 1]
        tool_calls = [s for s in steps if s.kind == "tool_call"]

        has_read = any(s.tool_name == "Read" for s in tool_calls)
        has_edited = any(s.tool_name == "Edit" for s in tool_calls)
        has_run = any(
            s.tool_name == "Bash"
            and "python" in str((s.tool_input or {}).get("command", ""))
            for s in tool_calls
        )

        if has_run: return "verified"
        if has_edited: return "fixed"
        if has_read: return "oriented"
        return "start"
```

### 9.4 Usage

```python
field = CodeFixField(agent_cwd=workdir)
for run in runs:
    field.add(run.trajectory, run.outcome)

# End-to-end (existing)
m = field.metrics()
print(f"Overall width: {m.width():.3f}")
print(f"Overall convergence: {m.convergence():.3f}")

# Horizons (new — on the field itself)
print(f"\nStates observed: {field.states}")

for s in field.states:
    h = field.horizon(s)
    m_s = h.metrics()
    print(f"\n  {s}: {h.K}/{field.K} trajectories")
    print(f"    width={m_s.width():.3f}")
    print(f"    convergence={m_s.convergence():.3f}")

# Horizons compose with existing methods
diagnosed = field.horizon("diagnosed")
diagnosed_success = diagnosed.success_region()
diagnosed_failure = diagnosed.failure_region()

# Separation within the "diagnosed" horizon —
# what distinguishes success from failure among agents
# that have reached diagnosis?
sep = diagnosed.metrics().separation()
```

### 9.5 File changes

| File | What |
|------|------|
| `agent_fields/field.py` | Add `state()`, `horizon()`, `states`, `_trajectories_through()`, store `_state_sequences` in `add()` |
| `agent_fields/__init__.py` | No new exports needed — `horizon()` is on `Field` |
| `agent_fields/visualisations.py` | `horizon_width(field)`, `horizon_drift(field)` charts |

No new module. The horizon lives on the field.

### 9.6 Visualization

Two Vega-Lite charts, keyed by state label:

```python
def horizon_width(field: Field, *, width: int = 500, height: int = 250) -> dict:
    """Bar chart: field width at each state along the progression."""
    data = [
        {"state": s, "width": round(field.horizon(s).metrics().width(), 4)}
        for s in field.states
    ]
    return {
        "$schema": _SCHEMA,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {"field": "state", "type": "ordinal", "sort": None,
                   "title": "State"},
            "y": {"field": "width", "type": "quantitative",
                   "title": "Horizon Width"},
        },
        "width": width,
        "height": height,
    }

def horizon_drift(field: Field, *, threshold: float = 0.5,
                   width: int = 500, height: int = 250) -> dict:
    """Bar chart: drift delta(s) at each state."""
    data = []
    for s in field.states:
        h = field.horizon(s)
        W_all = h.metrics().width()
        sr = h.success_region(threshold)
        W_success = sr.metrics().width() if sr.K >= 2 else W_all
        data.append({"state": s, "drift": round(W_all - W_success, 4)})
    return {
        "$schema": _SCHEMA,
        "data": {"values": data},
        "mark": "bar",
        "encoding": {
            "x": {"field": "state", "type": "ordinal", "sort": None,
                   "title": "State"},
            "y": {"field": "drift", "type": "quantitative",
                   "title": "Drift δ(s)"},
            "color": {
                "condition": {"test": "datum.drift >= 0", "value": "#e15759"},
                "value": "#59a14f",
            },
        },
        "width": width,
        "height": height,
    }
```
