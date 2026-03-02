# Proposal: Intent and the Policy Lens

> **Status:** Active discussion
> **Scope:** Defining the third user-defined function on the field — what it is, what it serves, and the two analytical children it produces (regimes and program families).

---

## 1. The Idea

The field today has two user-defined functions. $\varphi$ collapses a trajectory into a behavioral vector — it serves the field. $\psi$ labels a trajectory prefix with a task-progress marker — it serves the trajectory. Together they answer *what did the agent do?* and *where in the task is it?*

Neither answers: **how is the policy operating?**

Two trajectories can reach the same task state (`"editing"`) with identical behavioral measurements — same tool counts, same scope ratio — yet one is in confident execution while the other is recovering from a failed attempt. The task progress is identical. The behavioral fingerprint is identical. The policy's operational character is different. And that difference predicts what happens next.

The **intent function** reads this character from the trajectory. It is a third user-defined function on the field, and it serves the policy.

---

## 2. Three Functions, Three Masters

| | $\varphi$ (measure) | $\psi$ (state) | $\rho_\pi$ (intent) |
|---|---|---|---|
| **Serves** | The field | The trajectory | The policy |
| **Answers** | *What* did the agent do? | *Where* in the task? | *How* is the policy operating? |
| **Input** | Complete trajectory | Trajectory prefix at $t$ | Trajectory prefix at $t$ |
| **Output** | $\mathbb{R}^d$ | Discrete label (monotonic) | Discrete label (non-monotonic) |
| **Encodes** | "These behavioral properties matter" | "These milestones define progress" | "These are the meaningful operational characters of the policy" |
| **Produces** | The cloud | State sequences $\to$ horizons | Intent sequences $\to$ regimes + program families |

$\varphi$ is a lens on the **cloud** — it determines the coordinate system of the field.
$\psi$ is a lens on the **path** — it determines how trajectories are grouped by task progress.
$\rho_\pi$ is a lens on the **policy** — it determines how trajectories are grouped by operational character.

---

## 3. What Is Intent?

The policy $\pi_\theta$ is invisible. We didn't train it, we can't inspect it. But at each step, the policy's behavior has a qualitative character — broad and exploratory, narrow and decisive, reactive and corrective. The intent function is our best inference of this character from the trajectory.

$$\rho_\pi: \mathcal{T} \times \mathbb{N} \to \mathcal{I}$$

where $\mathcal{I}$ is a finite alphabet of intent labels defined by the user.

Like $\psi$, the intent function takes a trajectory and a step index. Like $\psi$, it returns a discrete label. The structural difference: **intent is non-monotonic.** The policy can explore, then execute, then explore again when it hits a wall. Recurrence is intrinsic to how policies operate — the same operational character can arise at multiple points in a trajectory.

This non-monotonicity is what gives intent its analytical richness. State sequences are monotonic — thin objects, all subsequences of the same ordered chain. Intent sequences are strings over an alphabet — rich objects with variable length, variable content, and recurrence patterns. This richness produces two analytical children: regimes (presence queries, overlapping) and program families (lineage queries, partitioning).

### 3.1 Intent arises from the generative process

The policy produces structured behavior because it was trained on structured data. Pre-training exposed the model to computational patterns: read before write, diagnose before fix, verify after change. RL reinforced transitions that lead to reward. The environment triggers shifts — error messages change the policy's character, successful verification focuses it.

Intent is not imposed by the analyst. It is the analyst's hypothesis about structure the generative process already produces.

| Force | What it produces in intent |
|---|---|
| **Pre-training** | The repertoire of intents — what operational characters the model can exhibit |
| **RL** | Transition dynamics — which intent sequences are favored |
| **Environment** | Triggers — feedback that shifts intent (errors $\to$ recovery, success $\to$ verification) |
| **Prompt** | Initial bias — detailed prompts may start in execution; vague prompts in exploration |

### 3.2 Intent is not state

$\psi$ and $\rho_\pi$ both label the trajectory at each step. They are orthogonal:

- **$\psi$ reads the task's signal in the trajectory.** Where has the task gotten to? What milestones have been achieved? This is a property of the data — the trajectory *has* states.

- **$\rho_\pi$ reads the policy's signal in the trajectory.** What is the policy attempting? How is it operating? This is an inference about the generator — the trajectory *reveals* intent.

A trajectory can be in state `"editing"` (task has progressed to the editing phase) while the policy's intent is `"recovering"` (the agent is reacting to a failed attempt). State says where. Intent says how.

---

## 4. The Intent Sequence and the Program String

At ingestion, the field computes the intent at each step, producing the **intent sequence**:

$$[\rho_\pi(\tau_k, 0),\; \rho_\pi(\tau_k, 1),\; \ldots,\; \rho_\pi(\tau_k, T_k - 1)]$$

Collapsing consecutive identical labels (run-length encoding) yields the **program string**:

$$\pi(\tau_k) = \text{RLE}([\rho_\pi(\tau_k, 0), \ldots, \rho_\pi(\tau_k, T_k - 1)])$$

The program string discards dwell time and preserves transitions. It is the policy's architectural skeleton — the sequence of operational characters the policy moved through.

Example: raw sequence `[exploring, exploring, executing, executing, recovering, exploring, executing, verifying]` produces program string `[exploring, executing, recovering, exploring, executing, verifying]`.

The program string is a string over the alphabet $\mathcal{I}$. Different trajectories produce different-length strings. This is the object that captures the *computational architecture* of the trajectory — something $\varphi$ (which erases temporal order) and $\psi$ (which tracks a monotonic chain) cannot see.

### 4.1 The Program Trie

The $K$ program strings are stored in a **prefix tree (trie)**. Each path from root to leaf is one program string. Each node stores the set of trajectory indices that pass through it.

```
root (K=8)
├── exploring (K=6)
│   ├── executing (K=5)
│   │   ├── verifying                  ← [τ1, τ4, τ7]   clean path
│   │   └── recovering (K=2)
│   │       └── executing
│   │           └── verifying           ← [τ2, τ5]       one recovery loop
│   └── exploring
│       └── executing
│           └── verifying               ← [τ3]           re-explored
└── orienting (K=2)
    └── exploring
        └── executing
            └── verifying               ← [τ6, τ8]       oriented first
```

The trie makes program families a **hierarchy**, not a flat partition. At any depth $d$, the nodes at that level define a grouping of trajectories by the first $d$ intent transitions they share:

- **Depth 0**: one family (all trajectories). The full field.
- **Depth 1**: families by first intent. "Started exploring" vs "started orienting."
- **Depth 2**: finer. "Started exploring then executing" vs "started exploring then re-exploring."
- **Full depth**: exact program match. Finest possible grouping.

No distance thresholds. No parameters. The tree structure itself determines the natural groupings.

### 4.2 Why a trie

The trie solves three problems:

**Oscillation.** When agents oscillate between intents, program strings become long and unique. Exact match produces $K$ families of size 1 — useless. The trie handles this by sharing prefixes: trajectories that started the same way but diverged later are in the same family at coarse depth and different families at fine depth. You choose the resolution that gives statistical power.

**Lineage queries.** The trie answers questions about shared ancestry — questions that pattern presence (regimes) cannot:

- **Prefix match**: traverse to a node, take its subtree $\to$ sub-field. "All trajectories that started with `[exploring, executing]`."
- **Divergence analysis**: at a node, examine the children. "After `[exploring, executing]`, 60% went to `verifying` and 40% went to `recovering`. The first group has 90% success rate; the second has 30%." The branching point is where the policy's choice of program determines the outcome.

Pattern presence — "does `[executing, recovering, executing]` appear anywhere in the program?" — is a regime query, not a trie query. The trie handles lineage. Regime handles presence. Clean separation.

**Structural diversity.** The trie's shape IS the analysis. A deep, narrow trie means most trajectories ran similar programs — low structural diversity. A broad, shallow trie means programs diverged early — high structural diversity. The branching factor at each depth tells you where in the program the policy's choices become consequential.

---

## 5. Where Intent Fits in the Theory

The root thesis is that agents are searching, not thinking. The trajectory is the search. The field is the distribution of searches. Run $K$ trajectories from the same configuration and you get $K$ samples of how the policy navigated from initial context to terminal state.

The empirical framework already provides two lenses on each search:

- $\varphi$ sees the **outcome** of the search. The behavioral vector collapses a trajectory into what happened — how much effort, how many tools, what scope. This is the search measured after it terminates. Temporal order is erased.

- $\psi$ sees the **progress** of the search. The state sequence tracks task milestones — what the search achieved, in what order. This is the search measured by its product. Temporal order is preserved but only as monotonic advancement.

Neither sees the **strategy** of the search — the sequence of operational characters the policy moved through while navigating. The policy $\pi_\theta$ is opaque. We did not design the reward function. We cannot inspect the weights. But the trajectory reveals how the policy operated at each step: broadly when exploring, narrowly when executing, reactively when recovering. This is not the outcome of the search and not the progress of the search. It is the structure of the search itself.

$\rho_\pi$ reads this structure. The intent sequence is the search strategy at full resolution. The program string is its skeleton — transitions without dwell time. The program trie is the family tree of all observed search strategies.

### 5.1 The trajectory as three objects

The trajectory $\tau$ is one object. The three functions read it as three different things:

| Function | Reads the trajectory as | Object produced |
|---|---|---|
| $\varphi$ | A point in behavioral space | The cloud — what happened |
| $\psi$ | A temporal progression | State sequence — where it got to |
| $\rho_\pi$ | A program execution | Program string — how the search was structured |

The trajectory in its rawest form IS the policy's program fully unrolled — every token, every tool call, every observation. $\varphi$ collapses this into a vector. $\psi$ reads milestones from it. $\rho_\pi$ reads the computational architecture from it.

Any form of comparative analysis that requires a continuous variable is served by $\varphi$. Any form of temporal reasoning about task progress is served by $\psi$. Any form of semantic reasoning about the policy's operational structure — what program the policy was executing, what strategies it employed, where it shifted — is served by $\rho_\pi$.

### 5.2 Why the field needs this

The field without intent can answer: *what did the agent do?* ($\varphi$) and *where in the task did it get?* ($\psi$). It cannot answer: *how did the policy operate?*

This gap has concrete consequences:

**Drift is a black box.** The root theory identifies drift — the field warping as context accumulates — as the central failure mode for long-horizon agents. The framework can detect drift (width increases, convergence drops) but cannot explain it. Intent provides the vocabulary: the policy shifted from executing to recovering, the recovery loop appeared, the program diverged from the pattern that succeeds. The trie's branching points are the points where searches diverge — where the policy's choice of next intent determines the outcome.

**Identical clouds hide different strategies.** Two sub-fields can have the same width, the same convergence, the same centroid — yet one is dominated by clean programs `[explore, execute, verify]` and the other by thrashing programs `[explore, execute, recover, explore, execute, recover, ...]`. Without intent, these are indistinguishable. With intent, the variance decomposition separates them: the first has low structural diversity (one dominant program), the second has high structural diversity (many programs, all noisy).

**Environment effects have no mechanism.** The root theory says "clean your repo" and "strong tests produce clear signal." Intent provides the mechanism: a clean environment produces focused program strings (short, convergent). A noisy environment produces scattered program strings (long, oscillatory). The trie's shape IS the measurement of how well the environment bounds the search.

### 5.3 Completing the vocabulary

The root theory defines the field as shaped by three forces: pre-training (what is reachable), RL (how the policy navigates), and the environment (what bounds the search). The empirical framework now has one lens per force:

| Force | What it shapes | Lens |
|---|---|---|
| Pre-training | What behaviors are reachable | $\varphi$ — the cloud's extent and dimensions |
| Environment | What bounds the search | $\psi$ — milestones, horizons, drift detection |
| RL / the policy | How the search navigates | $\rho_\pi$ — programs, regimes, families |

$\varphi$ makes the space of reachable behaviors visible. $\psi$ makes the temporal structure of the task visible. $\rho_\pi$ makes the policy's contribution visible. The three functions together exhaust what can be read from a trajectory without inspecting the model.

---

## 6. Two Analytical Children

State produced one analytical child: the horizon. Intent produces two.

### 6.1 The Regime

A **regime** is the sub-field of trajectories whose program string contains a given pattern. The pattern can be a single intent label or a sequential motif:

$$\mathcal{R}(p) = \{\varphi(\tau_k) : p \sqsubseteq \pi(\tau_k)\}$$

where $p \sqsubseteq s$ means $p$ appears as a contiguous subsequence of $s$.

A single label is a length-1 pattern. So `regime("exploring")` and `regime(("executing", "recovering", "executing"))` are the same operation at different resolutions. The word scales: "under the regime of exploring" and "under the regime of [executing, recovering, executing]" both read as "under this behavioral pattern."

**Regimes overlap.** A trajectory whose program is `[exploring, executing, recovering, executing, verifying]` belongs to `regime("exploring")`, `regime("recovering")`, `regime(("executing", "recovering"))`, and `regime(("executing", "recovering", "executing"))` simultaneously. This is structurally different from horizons, which nest ($\mathcal{H}(\text{verified}) \subseteq \mathcal{H}(\text{editing}) \subseteq \mathcal{F}$).

The analytical motion is **comparison**: the exploring regime has width $X$, the recovery-loop regime has width $Y$. Trajectories where the pattern `[executing, recovering, executing]` appeared have lower convergence than trajectories where it didn't. The regime is a lens for asking: *what does the field look like when the policy exhibits this behavioral pattern?*

### 6.2 The Program Family

A **program family** is a sub-field of trajectories whose program strings share a common prefix to depth $d$ in the program trie. At any node in the trie, the trajectories passing through that node form a family:

$$G_{\text{node}} = \{\tau_k : \pi(\tau_k) \text{ passes through node}\}$$

$$\hat{\mathcal{F}}_{G} = \{\varphi(\tau_k) : \tau_k \in G\}$$

At full depth (leaf nodes), this is exact program match — trajectories that ran the identical program. At shallower depths, it groups trajectories that share the same opening sequence of intents but may diverge later. The trie defines families at every granularity without parameters.

**Program families partition** at any chosen depth. Every trajectory belongs to exactly one family at depth $d$. No overlap.

The analytical motion has two parts:

**Variance decomposition.** The field has width $W$. Each family has width $W_j$. If $W_j \ll W$ for all families, the field's variation is **structural** — different programs produce different behaviors. If some $W_j \approx W$, the variation is **parametric** — same program, different execution. The intervention is different: structural variation $\to$ steer toward the right program. Parametric variation $\to$ reduce execution noise.

**Divergence analysis.** At any node in the trie, examine the children. "After `[exploring, executing]`, 60% of trajectories went to `verifying` and 40% went to `recovering`. The first group has 90% success rate; the second has 30%." The branching point is where the policy's choice of program determines the outcome. This is a question about *lineage* — where in the family tree do outcomes diverge — and only the trie can answer it.

### 6.3 Why two children from one function

State produces one child because state sequences are monotonic — thin, all shaped the same. The only interesting query is "condition on reaching this milestone." Horizons consume all of the analytical value.

Intent produces two children because intent sequences are non-monotonic with recurrence — rich strings with variable length and content. There are exactly two fundamental ways to query a string:

- **Presence:** does this pattern appear anywhere? $\to$ **regime** (overlapping sub-fields)
- **Lineage:** what prefix does this string share with others? $\to$ **program family** (partitioning sub-fields)

These exhaust the geometry. Presence produces overlap — a trajectory can match many patterns. Lineage produces partition — a trajectory has exactly one position at each depth in the trie. There is no third kind.

Both are sub-fields. Both compose with existing metrics. Both arise naturally from the non-monotonic, recurrent nature of intent.

---

## 7. Composability

### With horizons

Regime $\times$ horizon:

$$\mathcal{H}(s) \cap \mathcal{R}(m)$$

Trajectories that reached state $s$ AND where intent $m$ appeared. This is the composition that reveals: *at this task milestone, what was the policy doing?*

"At state `editing`, trajectories in the recovery regime have width 28 while those in the execution regime have width 5." The task state is the same. The policy character is different. The outcome diverges.

Family $\times$ horizon:

$$\mathcal{H}(s) \cap \hat{\mathcal{F}}_{G_j}$$

Among trajectories running program $j$, how many reached state $s$ and what's their behavioral profile?

### With regions

$$\hat{\mathcal{F}}^+(G_j) = \hat{\mathcal{F}}_{G_j} \cap \text{success\_region}$$

Which program families have high success rates? Which have low? This answers: *which computational architectures lead to success?*

### With metrics

All existing metrics (width, convergence, separation, skew) apply to regimes and program families without modification. No new metric objects.

### With drift

Intent-aware drift at state $s$:

$$\Delta \mathcal{I}(s) = P(m \mid s, y=1) - P(m \mid s, y=0)$$

Not just "failures are wider at state $s$," but "failures at state $s$ are in recovery intent while successes are in execution intent." The drift has a *cause* now.

---

## 8. Decisions

### 8.1 Intent is discrete

Same argument as state. Discrete labels enable exact grouping, trivial filtering, and clean sub-field construction. The burden of choosing good labels sits with the implementor — that is their hypothesis about the policy's operational character.

### 8.2 Intent is non-monotonic

This is a structural constraint, not a suggestion. States only advance because task progress only advances. Intent has no such constraint — the policy can return to a mode it was in before. Enforcing monotonicity would destroy the recurrence that makes program strings analytically rich.

### 8.3 The program string uses run-length encoding

The raw intent sequence has one label per step. The program string collapses consecutive identical labels. This discards dwell time and preserves transitions. The program string is the natural unit for comparing computational architectures.

### 8.4 Program families use a trie

Program strings are stored in a prefix tree. Families are defined by subtrees at a chosen depth — no distance thresholds, no clustering parameters. The trie provides:

- Hierarchical grouping at every granularity (depth 1 is coarse, full depth is exact match)
- Prefix queries and divergence analysis as tree operations
- A natural representation of the field's structural diversity (the shape of the trie itself)

This is what distinguishes program families from regimes. Regimes filter by **presence** — does this pattern appear anywhere? Program families filter by **lineage** — what prefix do these trajectories share? The trie is the data structure that makes lineage queries natural. Each answers a different question about the program string; together they exhaust the geometry (overlapping vs partitioning).

### 8.5 The division of labor

| | $\psi$ (state) | $\varphi$ (measure) | $\rho_\pi$ (intent) |
|---|---|---|---|
| **Answers** | *Where* in the task? | *What* did the agent do? | *How* is the policy operating? |
| **Output** | Discrete, monotonic | $\mathbb{R}^d$ | Discrete, non-monotonic |
| **Captures** | Task progress | Behavioral fingerprint | Policy character |
| **Analytical children** | Horizons (nested) | The cloud + metrics | Regimes (overlapping, presence) + program families (partitioning, lineage) |

### 8.6 On naming

**Intent** — the user's hypothesis about the policy's operational purpose at each step. "Exploring," "executing," "recovering," "verifying" are intent labels. The word is close to the policy: intent is *about* the policy, inferred *from* the trajectory. It carries the right epistemic weight — we are hypothesizing about the policy's purpose, not observing it directly.

**Regime** — the sub-field conditioned on a behavioral pattern's presence. From dynamical systems: a qualitatively distinct behavioral pattern. A single label is a length-1 pattern; a sequential motif like `[executing, recovering, executing]` is a longer one. "Under the regime of X" reads at any resolution. Regimes overlap. Comparison across regimes is the analytical motion.

**Program family** — the sub-field of trajectories sharing a common program prefix (or exact program). The word "program" reflects the computational architecture the policy executed. The word "family" reflects that these trajectories are related by structural lineage — they share a common opening sequence of intents, like members of a family sharing ancestry. The program trie is the family tree.

---

## 9. Implementation

### 9.1 The `Field` contract (extended)

```python
class Field(ABC, Generic[T]):

    # -- Abstract: user MUST implement ---------
    # measure(trajectory) -> np.ndarray
    # dimensions() -> list[Dimension]

    # -- Optional: user CAN override -----------
    # state(trajectory, t) -> str            (existing — for horizons)

    def intent(self, trajectory: T, t: int) -> str:
        """rho_pi — the policy's operational character at step t.

        This answers "how is the policy operating?" It is the user's
        hypothesis about the policy's intent, read from the trajectory
        prefix up to step t.

        Unlike state(), intent is non-monotonic — the policy can
        return to an intent it exhibited earlier. The sequence of
        intents forms the program string.

        The default returns a constant — all steps share one intent,
        which means regime() returns the full field. Override to
        define meaningful policy characters.
        """
        return "_"
```

### 9.2 Computed at ingestion

```python
def add(self, trajectory: T, outcome: float) -> np.ndarray:
    vector = self.measure(trajectory)
    self._points.append(vector)
    self._outcomes.append(outcome)
    self._raw.append(trajectory)

    # State sequence (existing)
    state_seq = []
    for t in range(self.trajectory_length(trajectory)):
        state_seq.append(self.state(trajectory, t))
    self._state_sequences.append(state_seq)

    # Intent sequence (new)
    intent_seq = []
    for t in range(self.trajectory_length(trajectory)):
        intent_seq.append(self.intent(trajectory, t))
    self._intent_sequences.append(intent_seq)

    return vector
```

### 9.3 Regime on `Field`

```python
def regime(self, pattern: str | tuple[str, ...]) -> _MaterializedField:
    """The sub-field of trajectories whose program contains pattern.

    A single label is a length-1 pattern — regime("exploring") checks
    if "exploring" appears anywhere in the program string. A tuple is
    a sequential pattern — regime(("executing", "recovering", "executing"))
    checks for that contiguous subsequence.

    Regimes overlap: a trajectory can match many patterns.
    """
    if isinstance(pattern, str):
        pattern = (pattern,)
    mask = np.array([
        self._contains_subsequence(self._program_string(k), pattern)
        for k in range(self.K)
    ])
    return self._materialize(mask)

@property
def intents(self) -> list[str]:
    """All intent labels observed, in first-seen order."""
    seen = {}
    for seq in self._intent_sequences:
        for label in seq:
            if label not in seen:
                seen[label] = len(seen)
    return list(seen.keys())
```

### 9.4 Program trie and families on `Field`

```python
def _program_string(self, k: int) -> tuple[str, ...]:
    """Run-length encode the intent sequence into a program string."""
    seq = self._intent_sequences[k]
    if not seq:
        return ()
    result = [seq[0]]
    for label in seq[1:]:
        if label != result[-1]:
            result.append(label)
    return tuple(result)

def program_family(self, prefix: tuple[str, ...]) -> _MaterializedField:
    """The sub-field of trajectories whose program string starts with prefix.

    At full program length this is exact match. At shorter lengths
    this groups all trajectories sharing the same opening intent
    sequence — a subtree of the program trie.

    Program families partition at any depth. Lineage, not presence.
    """
    mask = np.array([
        self._program_string(k)[:len(prefix)] == prefix
        for k in range(self.K)
    ])
    return self._materialize(mask)

@property
def programs(self) -> list[tuple[str, ...]]:
    """All distinct program strings observed."""
    seen = {}
    for k in range(self.K):
        p = self._program_string(k)
        if p not in seen:
            seen[p] = len(seen)
    return list(seen.keys())

@staticmethod
def _contains_subsequence(program, pattern):
    """Check if pattern appears as a contiguous subsequence in program."""
    n, m = len(program), len(pattern)
    return any(program[i:i+m] == pattern for i in range(n - m + 1))
```

### 9.5 Example

```python
class CodeFixFieldWithIntent(Field[dict]):

    def dimensions(self) -> list[Dimension]:
        # ... same 8 dimensions ...

    def measure(self, trajectory: dict) -> np.ndarray:
        # ... same behavioral vector ...

    def state(self, trajectory: dict, t: int) -> str:
        # ... same monotonic task progress ...

    def intent(self, trajectory: dict, t: int) -> str:
        """Policy's operational character at step t."""
        messages = trajectory["messages"][:t + 1]
        tool_calls = _extract_tool_calls_up_to(messages, t)

        if not tool_calls:
            return "orienting"

        last = tool_calls[-1]
        name = last["name"]

        if name == "Read":
            return "exploring"
        elif name == "Edit":
            return "executing"
        elif name == "Bash":
            cmd = str((last.get("input") or {}).get("command", ""))
            if "python" in cmd or "pytest" in cmd:
                return "verifying"

        # Check if recent edits were preceded by a failure
        if _recent_test_failed(messages, t):
            return "recovering"

        return "exploring"
```

### 9.6 Usage

```python
field = CodeFixFieldWithIntent(agent_cwd=workdir)
for run in runs:
    field.add(run.trajectory, run.outcome)

# Regimes — what does the field look like under each policy character?
for r in field.intents:
    sub = field.regime(r)
    m = sub.metrics()
    print(f"  regime({r}): K={sub.K}, width={m.width():.3f}")

# Program families — exact programs and their success rates
for prog in field.programs:
    sub = field.program_family(prog)
    sr = sub.success_region().K / sub.K if sub.K else 0
    print(f"  {prog}: K={sub.K}, success={sr:.0%}")

# Prefix query (program family) — lineage: started with exploring then executing
started_clean = field.program_family(("exploring", "executing"))
print(f"  started [explore→execute]: K={started_clean.K}")

# Pattern query (regime) — presence: the recovery loop appeared anywhere
recovery_loop = field.regime(("executing", "recovering", "executing"))
print(f"  recovery loop regime: K={recovery_loop.K}")

# Composition: regime x horizon
editing = field.horizon("editing")
for r in field.intents:
    sub = editing.regime(r)
    if sub.K > 0:
        print(f"  editing x {r}: K={sub.K}, width={sub.metrics().width():.3f}")
```

### 9.7 File changes

| File | What |
|------|------|
| `agent_fields/field.py` | Add `intent()`, `regime()`, `intents`, `program_family()`, `programs`, store `_intent_sequences` in `add()` |
| `agent_fields/__init__.py` | No new exports — everything lives on `Field` |

No new module. The intent lives on the field.
