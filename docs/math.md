# Deriving the Field from Trajectory Data: A Model-Agnostic Formulation

> This mathematical formulation is derived from the theory from the essay : [Agents are not thinking, they are searching](https://technoyoda.github.io/agent-search.html). That essay establishes three things this derivation depends on:
>
> 1. **[Pre-training establishes the landscape](https://technoyoda.github.io/agent-search.html#heading-4)** — next-token prediction over internet-scale data produces a distribution over language. The prompt is a conditioning variable that determines which region of this distribution gets sampled. This is what makes the space of reachable behaviors finite (though vast) — the model can only produce trajectories within the distribution it learned.
>
> 2. **[Reinforcement learning establishes the search strategy](https://technoyoda.github.io/agent-search.html#heading-5)** — RL training shapes a policy that navigates trajectories toward reward signals. The policy is what makes the agent favor certain trajectories over others. We never see this policy directly — we only observe the distribution of behaviors it produces.
>
> 3. **[The inference rollout is the trajectory](https://technoyoda.github.io/agent-search.html#heading-6)** — the trained agent starts from an initial context, generates an action, receives environmental feedback that enters the context window, and repeats. Each cycle, the context grows and the space of reachable behaviors shifts. This is the process that produces the trajectories $\tau$ we collect and measure.
>
> The [Agent Field Theory section](https://technoyoda.github.io/agent-search.html#heading-7) introduces the `field` — the space of reachable behaviors conditioned on the context window plus the trained policy. That formulation is per-timestep and intractable: it would require enumerating every possible future trajectory from every context state. Since model providers (OpenAI, Anthropic, etc.) are black-box entities that will never expose the reward functions that shaped their policies, we cannot compute the per-timestep field analytically. This document replaces it with an empirical approximation built from completed trajectories. 
---

## 1. Primitives and Notation

Define what we can observe and collect:

**Trajectory.** An ordered sequence of observation-action pairs:

$$\tau = \left[(o_1, a_1),\; (o_2, a_2),\; \dots,\; (o_n, a_n)\right]$$

**Environment configuration.** The full specification of the controlled environment — files present, tools available, permissions, test suites. Since the experimenter controls it, $E$ is a known vector at setup time:

$$E \in \mathcal{E} \subseteq \mathbb{R}^m$$

Note that $E$ is identical at the start for all runs within an experiment, but $E$ may carry inherent runtime entropy — intermittent API failures, non-deterministic tool outputs, timing jitter. This entropy is not a flaw in the formulation; it is absorbed naturally. When we sample $K$ trajectories from the same $(E, c_0)$, some will encounter perturbations and some will not, some will recover and some will not. The resulting point cloud (Section 3) reflects the distribution of behaviors **inclusive** of environmental noise, which is exactly what we need to predict real-world agent performance in imperfect environments.

**Context seed.** The initial prompt or system instruction that starts the run, denoted $c_0$.

**Outcome label.** A quality signal assigned to a completed trajectory:

$$y(\tau) \in \mathbb{R}$$

This may be binary (success/failure) or a richer scalar (correctness score, reward).

**The black box.** The model is a stochastic function $\mathcal{M}$ that, given $(E, c_0)$, produces trajectories:

$$\tau \sim \mathcal{M}(E, c_0)$$

We never open $\mathcal{M}$. We only observe its input-output behavior. The stochasticity of $\mathcal{M}$ comes from two sources: the [pre-trained distribution](https://technoyoda.github.io/agent-search.html#heading-4) determines what tokens are reachable, and the [RL-shaped policy](https://technoyoda.github.io/agent-search.html#heading-5) determines which trajectories through that space are favored. Both are opaque to us.

---

## 2. The Empirical Field as a Distribution

Run $K$ trajectories for a fixed configuration $(E, c_0)$:

$$\{\tau_1, \tau_2, \dots, \tau_K\} \sim \mathcal{M}(E, c_0)$$

**Definition (Empirical Field).** The field induced by a configuration is the distribution over trajectories:

$$\mathcal{F}(E, c_0) \coloneqq P_{\mathcal{M}}(\tau \mid E, c_0)$$

We do not have the analytical form of $P_{\mathcal{M}}(\tau \mid E, c_0)$. To write it down, we would need three things we do not have:

1. **The pre-trained distribution** — the full conditional distribution over next tokens given any context, shaped by [pre-training](https://technoyoda.github.io/agent-search.html#heading-4) over internet-scale data. This is a function over a vocabulary of tens of thousands of tokens, conditioned on sequences of arbitrary length. Model providers do not publish it.

2. **The RL-shaped policy** — the modifications to that distribution imposed by [reinforcement learning](https://technoyoda.github.io/agent-search.html#heading-5). RLHF, RLAIF, and other post-training procedures reshape which trajectories the model favors, but the reward functions, preference data, and training dynamics are proprietary. The policy is the product of these procedures, and we have no access to it.

3. **The environment's transition dynamics** — how $E$ responds to each action and what observations it returns. Even when we control the environment, the mapping from agent action to environment response can be complex (tool execution, file system state, API calls), and some environments carry runtime entropy (network latency, non-deterministic tool output) that makes the transition function stochastic.

The analytical form of $\mathcal{F}$ would require composing all three — marginalizing the policy over all possible action sequences, conditioned on the environment's responses at each step. This is intractable. But we do not need it. We have samples. Each sample is produced by one [inference rollout](https://technoyoda.github.io/agent-search.html#heading-6): the agent starts from $c_0$, generates actions, receives feedback from $E$ that enters the context window, and repeats until the trajectory terminates. Run $K$ rollouts and we have $K$ samples from $\mathcal{F}$ — enough to approximate its shape empirically.

Since trajectories are high-dimensional objects, we need to project them into a workable space.

---

## 3. Trajectory Embedding

### 3.1 Definition

**Definition (Measurement Function).** A function $\varphi: \mathcal{T} \to \mathbb{R}^d$ that maps a trajectory to a fixed-dimensional vector describing **what the agent did** (its behavior), not how well it did (its outcome):

$$\varphi(\tau) \in \mathbb{R}^d$$

where each component is a behavioral measurement:

| Component | Example |
|-----------|---------|
| $\varphi_1(\tau)$ | number of tool calls |
| $\varphi_2(\tau)$ | number of unique files touched |
| $\varphi_3(\tau)$ | number of backtracking events |
| $\varphi_4(\tau)$ | $\lvert\tau\rvert$ (trajectory length) |
| $\varphi_5(\tau)$ | exploration fraction |
| $\varphi_6(\tau)$ | semantic similarity to reference |
| $\vdots$ | $\vdots$ |

The outcome label $y(\tau)$ is deliberately excluded from $\varphi$ — it is a label painted on each point, not a coordinate of the cloud. This separation is what allows [field metrics](#4-field-metrics) to ask *where in behavioral space do good outcomes cluster?* without conflating behavioral diversity with outcome noise.

### 3.2 The Point Cloud

Now each trajectory is a point in $\mathbb{R}^d$, and the empirical field becomes a point cloud:

$$\mathcal{F}(E, c_0) \approx \lbrace\varphi(\tau_1),\; \varphi(\tau_2),\; \dots,\; \varphi(\tau_K)\rbrace \subset \mathbb{R}^d$$

The field's shape, spread, and center are now measurable geometric properties of this cloud.

### 3.3 The Choice of $\varphi$ Determines the Field

$\varphi$ determines which behavioral properties become dimensions of the point cloud — only those properties exist for metrics to operate on. This means the choice of $\varphi$ determines what the field metrics mean. If $\varphi$ emphasizes tool calls and file touches, field width $W_{\mathcal{F}}$ measures **behavioral diversity**. If $\varphi$ emphasizes semantic similarity to a reference solution, the same $W_{\mathcal{F}}$ measures **solution diversity**. Same cloud machinery, entirely different interpretation.

Any behavior not captured by $\varphi$ is absent from the cloud and invisible to every metric downstream. This also means **multiple fields can coexist for the same trajectory data**: different $\varphi$ functions produce different clouds, different metrics, different views of the same underlying behavior.

### 3.4 Theoretical Contract

$\varphi$ is, at this stage, a **theoretical object**. It is a contract that says: "give me a trajectory, I return a fixed-dimensional vector." The formalism does not prescribe *how* $\varphi$ is implemented — it may be a hand-crafted feature vector, a learned embedding, or a bag of heuristics. It only requires the interface be honored: trajectory in, vector in $\mathbb{R}^d$ out.

The mathematical machinery in Sections 4–7 (field metrics, ablation decomposition, field horizon, intent) operates on the **output** of $\varphi$, not on its internals. The formalism gives us the interface; a physical implementation gives us the data structure.

---

## 4. Field Metrics

From the point cloud, define concrete measurable properties of the field — scalars, vectors, and matrices that characterize its shape. These metrics operate on the cloud geometry ($\varphi$ outputs) and the outcome labels ($y$) as two separate objects — the cloud is the map, the outcome is the terrain elevation painted on top. The semantics of what these metrics derive is explained in [METRICS.md](./METRICS.md) and [API.md](./API.md).

### 4.1 Field Center (Expected Behavior)

$$\mu_{\mathcal{F}} = \frac{1}{K} \sum_{i=1}^{K} \varphi(\tau_i)$$

This is the centroid of the point cloud — the mean of all measured behavioral vectors. It lives in the behavioral space defined by $\varphi$, not in outcome space. See [METRICS.md — center](./METRICS.md#center--mathbbrd) for interpretation.

### 4.2 Field Width (Behavioral Diversity)

$$W_{\mathcal{F}} = \text{tr}\!\left(\text{Cov}\left(\varphi(\tau)\right)\right) = \sum_{j=1}^{d} \text{Var}\left(\varphi_j(\tau)\right)$$

High $W_{\mathcal{F}}$ means the points are dispersed across the cloud. Low $W_{\mathcal{F}}$ means they cluster tightly. Because $y(\tau)$ is excluded from $\varphi$, this metric reflects variance in behavioral coordinates only — it is not inflated by outcome noise. See [METRICS.md — width](./METRICS.md#width--scalar) for interpretation.

### 4.3 Field Convergence (Quality Concentration)

$$C_{\mathcal{F}} = \frac{\mathbb{E}\left[y(\tau)\right]}{\sigma\left[y(\tau)\right]}$$

This is where $y(\tau)$ enters — as an external label over the cloud, not as a coordinate within it. High $C_{\mathcal{F}}$ means the mean outcome is large relative to its standard deviation. Low $C_{\mathcal{F}}$ means outcome variance is high relative to the mean. See [METRICS.md — convergence](./METRICS.md#convergence--scalar) for interpretation.

### 4.4 Field Skew (Outcome-Dimension Correlation)

$$S_{\mathcal{F}} = \text{corr}\!\left(y(\tau),\; \varphi_j(\tau)\right)$$

where $\varphi_j$ is any behavioral dimension from the embedding. This correlates the external outcome label against a behavioral coordinate. Positive $S_{\mathcal{F}}$ means higher values on dimension $j$ co-occur with higher outcomes. Negative means the opposite. Near zero means no linear relationship. See [METRICS.md — skew](./METRICS.md#skewcost_dim--scalar) for interpretation.

### 4.5 Derived: Success and Failure Regions

Because $y(\tau)$ is a label on the cloud rather than a dimension of it, we can partition the cloud and compare regions:

$$\mu_{\mathcal{F}}^{+} = \frac{1}{|\mathcal{S}|} \sum_{\tau_i \in \mathcal{S}} \varphi(\tau_i) \qquad \mu_{\mathcal{F}}^{-} = \frac{1}{|\mathcal{R}|} \sum_{\tau_i \in \mathcal{R}} \varphi(\tau_i)$$

where $\mathcal{S} = \{\tau : y(\tau) \geq \theta_y\}$ (successful trajectories) and $\mathcal{R} = \{\tau : y(\tau) < \theta_y\}$ (failed trajectories). The vector $\mu_{\mathcal{F}}^{+} - \mu_{\mathcal{F}}^{-}$ is the direction of the centroid gap between the two groups in behavioral space. See [METRICS.md — separation](./METRICS.md#separationthreshold--mathbbrd) for interpretation.

---

## 5. Ablation Decomposition — The Core Derivation

This is where the controlled environment pays off. We want to understand how each environmental factor contributes to the field's shape.

### 5.1 Environment Factorization

Represent the environment as a vector of controllable factors:

$$E = (e_1, e_2, \dots, e_m)$$

where each $e_j$ is an independently controllable lever:

- $e_1$: number of irrelevant files present
- $e_2$: test suite strength (none / unit / integration / property)
- $e_3$: permission level (full / scoped / read-only)
- $e_4$: presence of a specific configuration file

### 5.2 Ablation Protocol

For each factor $e_j$, construct two configurations:

$$E_j^{+} = E \text{ with } e_j \text{ present/enabled}$$

$$E_j^{-} = E \text{ with } e_j \text{ absent/disabled}$$

Run $K$ trajectories for each. Compute the field metrics for both:

$$\mathcal{F}_j^{+} = \mathcal{F}(E_j^{+}, c_0) \;\longrightarrow\; \mu_j^{+},\; W_j^{+},\; C_j^{+}$$

$$\mathcal{F}_j^{-} = \mathcal{F}(E_j^{-}, c_0) \;\longrightarrow\; \mu_j^{-},\; W_j^{-},\; C_j^{-}$$

### 5.3 Factor Effects

**Definition (Field Effect of Factor $e_j$).** The marginal impact of factor $e_j$ on each field metric:

$$\Delta W(e_j) = W_j^{+} - W_j^{-} \qquad \text{(width change — narrowing or widening?)}$$

$$\Delta C(e_j) = C_j^{+} - C_j^{-} \qquad \text{(convergence change — quality improvement?)}$$

$$\Delta \mu(e_j) = \left\|\mu_j^{+} - \mu_j^{-}\right\|_2 \qquad \text{(center shift — behavioral change?)}$$

This gives an **empirical decomposition** of how each environmental lever affects the field, without ever looking at the model.

### 5.4 Interaction Effects

For pairs of factors $(e_i, e_j)$, run the four-way ablation:

$$\mathcal{F}(E_{ij}^{++}), \quad \mathcal{F}(E_{ij}^{+-}), \quad \mathcal{F}(E_{ij}^{-+}), \quad \mathcal{F}(E_{ij}^{--})$$

**Definition (Interaction Effect).** The interaction effect of $(e_i, e_j)$ on field width:

$$I_W(e_i, e_j) = \left(W_{ij}^{++} - W_{ij}^{+-}\right) - \left(W_{ij}^{-+} - W_{ij}^{--}\right)$$

**Proposition (Factor Independence).** If $I_W(e_i, e_j) \approx 0$, the factors are independent in their field effects: their contributions are additive and can be reasoned about in isolation. If $I_W(e_i, e_j) \neq 0$, the factors interact — their joint effect is non-decomposable.

---

## 6. Field Horizon

The field is not static — it evolves as context accumulates within a single trajectory. The **field horizon** captures this: standing at a point in the trajectory, looking forward at what behaviors are still reachable.

### 6.1 The State Function

The measurement function $\varphi$ captures *what the agent did* — the full behavioral vector. To track *where in the task the agent is*, we introduce a second function:

**Definition (State Function).** A function $\psi: \mathcal{T} \times \mathbb{N} \to \mathcal{S}$ that reduces the trajectory prefix up to step $t$ into a discrete state label:

$$\psi(\tau, t) \in \mathcal{S}$$

where $\mathcal{S}$ is a discrete set of states. $\psi$ is a reduce function over the accumulated context — a lossy compression of everything the agent has seen and done into a single label representing semantic progress toward the goal.

The state progression is linear:

$$s_0 \to s_1 \to s_2 \to \cdots \to s_n$$

The task has a logical progression — a semantic ordering of what needs to happen to reach the goal. Different agents traverse this progression at different speeds, with different behaviors at each phase, but the progression itself is a line. States are monotonic.

$\psi$ and $\varphi$ serve different roles:

| | $\psi$ (state) | $\varphi$ (measurement) |
|---|---|---|
| Answers | *Where* in the task? | *How* is the agent behaving? |
| Output | Discrete label | Vector in $\mathbb{R}^d$ |
| Dynamics | Linear progression | High-dimensional, varies freely |

Behavioral variation within a phase — "did the agent read the right file or the wrong file during diagnosis?" — is captured by $\varphi$, not by subdividing state. State stays coarse and linear. Measurement stays high-dimensional and detailed.

### 6.2 The Field Horizon

**Definition (Field Horizon).** Given a state $s \in \mathcal{S}$, the field horizon at $s$ is:

$$\mathcal{H}(s) = \lbrace\varphi(\tau_k) : \exists\, t \text{ s.t. } \psi(\tau_k, t) = s\rbrace$$

The cloud of terminal measurements, filtered to only trajectories that passed through state $s$ at some point. The outcome labels carry over from the parent field.

$\mathcal{H}(s)$ is a sub-field. It has all the same structure as the parent — a point cloud in $\mathbb{R}^d$, outcome labels, dimensions. All field metrics apply directly:

$$W_{\mathcal{H}(s)} \quad \text{— how diverse are futures from state } s \text{?}$$

$$C_{\mathcal{H}(s)} \quad \text{— how reliably do trajectories through } s \text{ succeed?}$$

$$\mu_{\mathcal{H}(s)}^{+} - \mu_{\mathcal{H}(s)}^{-} \quad \text{— what separates success from failure among trajectories through } s \text{?}$$

### 6.3 Horizon Over Multiple States

Each trajectory unrolls its own state machine. Different trajectories can produce different state labels — one may pass through `"debugging:core_files"`, another through `"debugging:tests"`. These are different labels from different executions, but they may represent the same semantic phase.

**Definition.** Given a set of states $G \subseteq \mathcal{S}$, the horizon over $G$ is:

$$\mathcal{H}(G) = \lbrace\varphi(\tau_k) : \exists\, t,\; \exists\, s \in G \text{ s.t. } \psi(\tau_k, t) = s\rbrace$$

The union of trajectories that passed through any state in $G$. This is a query-time operation — $\psi$ remains unchanged. The user defines states at full resolution and selects at analysis time.

The single-state horizon is the special case $\mathcal{H}(\lbrace s \rbrace) = \mathcal{H}(s)$.

### 6.4 Composability

Because states are linear, horizons are naturally nested:

$$\mathcal{H}(s_n) \subseteq \mathcal{H}(s_{n-1}) \subseteq \cdots \subseteq \mathcal{H}(s_0) = \mathcal{F}$$

Every trajectory that reached state $s_n$ also passed through every earlier state. The parent field is $\mathcal{H}(s_0)$ — the horizon at the start, containing all trajectories.

Comparing horizons across the progression:

$$\Delta W = W_{\mathcal{H}(s_1)} - W_{\mathcal{H}(s_2)}$$

is the change in width between two consecutive horizons. Positive $\Delta W$ means the cloud is wider at $s_1$ than at $s_2$ — the set of trajectories that reached $s_2$ has less variance than the set that reached $s_1$.

### 6.5 Drift Detector

$$\boxed{\delta(s) = W_{\mathcal{H}(s)} - W_{\mathcal{H}^+(s)}}$$

where $\mathcal{H}^+(s)$ is the horizon restricted to successful trajectories. $\delta(s) > 0$ means the non-successful trajectories contribute additional spread to the horizon at state $s$ beyond what the successful trajectories alone produce.

---

## 7. Intent and the Policy Lens

The measurement function $\varphi$ captures *what the agent did*. The state function $\psi$ captures *where in the task the agent is*. Neither captures *how the policy is operating* — the qualitative character of the policy's behavior at each step. Two trajectories can reach the same task state with identical behavioral measurements yet differ in their operational character: one is in confident execution, the other is recovering from a failed attempt. The **intent function** reads this character from the trajectory.

### 7.1 The Intent Function

**Definition (Intent Function).** A function $\rho_\pi: \mathcal{T} \times \mathbb{N} \to \mathcal{I}$ that assigns an intent label to each step of a trajectory:

$$\rho_\pi(\tau, t) \in \mathcal{I}$$

where $\mathcal{I}$ is a finite alphabet of intent labels defined by the user.

Like $\psi$, the intent function takes a trajectory and a step index. Like $\psi$, it returns a discrete label. The structural difference: **intent is non-monotonic.** The policy can explore, then execute, then explore again when it hits a wall. Recurrence is intrinsic to how policies operate.

| | $\psi$ (state) | $\rho_\pi$ (intent) |
|---|---|---|
| **Serves** | The trajectory | The policy |
| **Answers** | *Where* in the task? | *How* is the policy operating? |
| **Output** | Discrete label (monotonic) | Discrete label (non-monotonic) |
| **Dynamics** | Linear progression | Recurrent, variable |

### 7.2 Intent Sequence and Program String

At ingestion, the field computes the intent at each step, producing the **intent sequence**:

$$[\rho_\pi(\tau_k, 0),\; \rho_\pi(\tau_k, 1),\; \ldots,\; \rho_\pi(\tau_k, T_k - 1)]$$

Collapsing consecutive identical labels via run-length encoding yields the **program string**:

$$\pi(\tau_k) = \text{RLE}([\rho_\pi(\tau_k, 0), \ldots, \rho_\pi(\tau_k, T_k - 1)])$$

The program string is a string over the alphabet $\mathcal{I}$, variable length across trajectories. It discards dwell time and preserves transitions — the policy's architectural skeleton.

### 7.3 The Regime

**Definition (Regime).** Given a pattern $p$ (a contiguous subsequence over $\mathcal{I}$), the regime is:

$$\mathcal{R}(p) = \lbrace\varphi(\tau_k) : p \sqsubseteq \pi(\tau_k)\rbrace$$

where $p \sqsubseteq s$ means $p$ appears as a contiguous subsequence of $s$.

**Key property: regimes overlap.** A trajectory whose program contains multiple patterns belongs to multiple regimes simultaneously. This is structurally different from horizons, which nest ($\mathcal{H}(s_n) \subseteq \mathcal{H}(s_{n-1})$).

$\mathcal{R}(p)$ is a sub-field. All field metrics ($W$, $C$, $\mu$, $S$) apply directly.

### 7.4 The Program Family

**Definition (Program Family).** Given a prefix $G$ over $\mathcal{I}$, the program family is:

$$\mathcal{F}_G = \lbrace\varphi(\tau_k) : \pi(\tau_k) \text{ has prefix } G\rbrace$$

**Key property: families partition at any prefix length $d$.** Every trajectory belongs to exactly one family at depth $d$. No overlap.

The set of all program strings forms a conceptual **prefix tree (trie)**. Each path from root to leaf is one program string. Families are subtrees at a chosen depth — hierarchical grouping without parameters.

### 7.5 Variance Decomposition

At any prefix depth $d$, the program families $\lbrace\mathcal{F}_{G_1}, \mathcal{F}_{G_2}, \ldots\rbrace$ partition the field. The field width decomposes:

$$W = W_{\text{between}} + W_{\text{within}}$$

where $W_{\text{between}}$ is the variance of family centroids around the field centroid, and $W_{\text{within}}$ is the average within-family variance.

If $W_{\text{within}} \ll W$ for all families, the field's variation is **structural** — different programs produce different behaviors. If $W_{\text{within}} \approx W$, the variation is **parametric** — same program, different execution.

### 7.6 Composability

Intent composes with existing field structures:

**Horizon $\times$ Regime:**

$$\mathcal{H}(s) \cap \mathcal{R}(p)$$

Trajectories that reached state $s$ AND where pattern $p$ appeared. This reveals: at a given task milestone, what was the policy doing?

**Horizon $\times$ Family:**

$$\mathcal{H}(s) \cap \mathcal{F}_G$$

Among trajectories running a given program, how many reached state $s$?

**Intent-aware drift:**

$$\Delta\mathcal{I}(s) = P(m \mid s, y=1) - P(m \mid s, y=0)$$

Not just "failures are wider at state $s$," but "failures at state $s$ are in intent $m$ while successes are in intent $m'$." The drift has a cause.

---

## 8. Experimental Pipeline

The concrete steps required to build this:

1. **Environment Parameterization.** Enumerate controllable factors $\{e_1, \dots, e_m\}$ and their levels.
2. **Trajectory Collection.** Run $K$ trajectories per configuration; log full $(o, a)$ sequences.
3. **Labeling.** Score each trajectory via automated metrics and human labels: $y(\tau)$.
4. **Feature Extraction.** Implement $\varphi(\tau)$ to embed trajectories into $\mathbb{R}^d$.
5. **Ablation Execution.** Systematically vary factors per Section 5; collect field samples.
6. **Horizon Analysis.** Define $\psi$, compute horizons at each state, monitor drift $\delta(s)$ per Section 6.
7. **Intent Analysis.** Define $\rho_\pi$, compute program strings, construct regimes and program families per Section 7.

---

## 9. Key Insight: Formalism as Interface, Not Implementation

The model's weights, gradients, and logits are never required. The field is reconstructed entirely from **behavioral observations under controlled perturbation**. The model is treated as a stochastic oracle — what we are doing is **system identification on a black box**, which is classical engineering.

Because each factor is toggled independently with all others held constant, the ablation design isolates per-factor effects. The strength of the causal claim depends on the degree to which factors are truly independent and the environment is fully specified — assumptions that should be validated per experiment.

### 9.1 What the Formalism Buys Us

Everything in Sections 1–7 is pure mathematical formalism. The objects — $\varphi$, $\psi$, $\rho_\pi$, $\mathcal{F}$, $\mathcal{H}$, $\mathcal{R}(p)$, $\mathcal{F}_G$, $\pi(\tau_k)$, $W$, $C$ — are theoretical. They are not data structures, not code, not implementations. This is deliberate, and it buys two things:

1. **It tells us what to build.** We now know we need: a thing that eats trajectories and produces vectors ($\varphi$ — see [METRICS.md](./METRICS.md) for what operates on its output), a thing that reduces prefixes into progress labels ($\psi$), a thing that reads the policy's operational character ($\rho_\pi$), a thing that holds clouds of those vectors ($\hat{\mathcal{F}}$), a thing that filters clouds by state ($\mathcal{H}$), by behavioral pattern ($\mathcal{R}$), or by program lineage ($\mathcal{F}_G$), and a thing that computes metrics on those clouds ($W$, $C$, $\mu$, $S$ — see [METRICS.md](./METRICS.md) for semantics). The math specifies the *interface* — trajectory in, vector out, cloud in, scalar out — not the implementation.

2. **It tells us what questions are answerable.** Before writing a single line of code, we already know: if we can construct $\varphi$ and collect enough trajectories, then ablation decomposition and drift detection are computable from existing objects. Extensions like the response surface and transfer function build on top — see [EXTENSIONS.md](./EXTENSIONS.md).

### 9.2 The Practical Boundary

The next step — the physical implementation — is about choosing concrete data structures for $\varphi$, concrete storage for $\hat{\mathcal{F}}$, and concrete compute for the metrics. The math does not care if $\varphi$ is a hand-crafted feature vector, a learned embedding, or a bag of heuristics. It only requires the contract be honored: trajectory in, fixed-dimension vector out. The construction of $\varphi$ and $\mathcal{F}$ as formalism is what allows us to now reason about practical use cases of these objects — how we actually figure out a physical implementation.

---

## Appendix A: Unpacking the Empirical Field Definition

This appendix provides a detailed, intuitive walkthrough of the central definition from Section 2:

$$\mathcal{F}(E, c_0) \coloneqq P_{\mathcal{M}}(\tau \mid E, c_0)$$

### A.1 Reading the Expression

We parse it left to right:

| Symbol | Meaning |
|---|---|
| $\mathcal{F}(E, c_0)$ | "The field, given a specific environment $E$ and starting prompt $c_0$." This is the object we are defining. |
| $\coloneqq$ | "Is defined as." We are assigning a name, not proving an equality. |
| $P_{\mathcal{M}}$ | A probability distribution whose shape is determined by model $\mathcal{M}$. The subscript says: these probabilities come from *this specific* black-box model. |
| $(\tau \mid E, c_0)$ | "The probability of trajectory $\tau$, *conditioned on* the environment being $E$ and the prompt being $c_0$." |

### A.2 What Kind of Object Is $\mathcal{F}$?

$\mathcal{F}(E, c_0)$ is not a single number. It is a **probability distribution over the entire space of possible trajectories**. Think of it as a function that assigns a likelihood to every conceivable sequence of agent behaviors:

$$\mathcal{F}(E, c_0): \mathcal{T} \to [0, 1] \qquad \text{where} \quad \sum_{\tau \in \mathcal{T}} \mathcal{F}(E, c_0)(\tau) = 1$$

Concretely, imagine enumerating every possible trajectory the agent might produce:

| Trajectory | Behavior | Probability |
|---|---|---|
| $\tau_a$ | reads file A, edits line 3, runs tests, done | $P(\tau_a \mid E, c_0) = 0.12$ |
| $\tau_b$ | reads file A, reads file B, edits line 7, runs tests, done | $P(\tau_b \mid E, c_0) = 0.08$ |
| $\tau_c$ | reads file C, deletes it, crashes | $P(\tau_c \mid E, c_0) = 0.001$ |
| $\tau_d$ | reads file A, gets confused, reads 50 more files, gives up | $P(\tau_d \mid E, c_0) = 0.03$ |
| $\vdots$ | thousands more... | all summing to 1 |

That entire table — the complete assignment of probabilities to every possible behavior — **is** $\mathcal{F}(E, c_0)$.

### A.3 Why the Subscript $\mathcal{M}$ Matters

The subscript on $P_{\mathcal{M}}$ does critical work. It says: a different model would produce a **different** distribution over trajectories given the same $(E, c_0)$. Formally:

$$P_{\mathcal{M}_1}(\tau \mid E, c_0) \neq P_{\mathcal{M}_2}(\tau \mid E, c_0) \qquad \text{in general}$$

The field is model-dependent. But we reconstruct it without opening the model — purely by sampling from it and observing what comes out. The model's internals — [the pre-trained distribution](https://technoyoda.github.io/agent-search.html#heading-4) and the [RL-shaped policy](https://technoyoda.github.io/agent-search.html#heading-5) — are baked into the shape of $P_{\mathcal{M}}$, but we access that shape only through repeated sampling.

### A.4 Why the Conditional $\mid$ Does All the Work

The conditioning bar says: the distribution over trajectories **changes** when you change $E$ or $c_0$:

$$\mathcal{F}(E_1, c_0) \neq \mathcal{F}(E_2, c_0) \qquad \text{when } E_1 \neq E_2$$

This is the entire premise of the formulation:

- Change the environment (add a file, remove a permission) $\longrightarrow$ different distribution $\longrightarrow$ different field
- Change the prompt (vague vs. precise) $\longrightarrow$ different distribution $\longrightarrow$ different field

The field is not a property of the model alone. It is a property of the **model-environment-prompt triple**.

### A.5 Analogy: Ball on a Landscape

| Concept | Analogy |
|---|---|
| Model $\mathcal{M}$ | The ball (mass, friction, elasticity — fixed, unknown internals) |
| Environment $E$ + prompt $c_0$ | The landscape (hills, valleys, walls) |
| Field $\mathcal{F}(E, c_0)$ | The probability map over where the ball ends up and what paths it takes |

You cannot open the ball to inspect its internals. But you can reshape the landscape and roll the ball $K$ times to map out where it tends to go. That empirical map **is** $\mathcal{F}(E, c_0)$.

### A.6 Why Section 3 ($\varphi$) Is Necessary

Trajectory space $\mathcal{T}$ is essentially infinite-dimensional — there are combinatorially many possible sequences of observations and actions. You cannot literally enumerate every $\tau$ and assign it a probability.

The measurement function $\varphi(\tau) \in \mathbb{R}^d$ projects each trajectory into a finite-dimensional vector. This transforms the intractable distribution $P_{\mathcal{M}}(\tau \mid E, c_0)$ over trajectory space into a tractable distribution over $\mathbb{R}^d$:

$$\mathcal{F}(E, c_0) \xrightarrow{\;\varphi\;} \hat{\mathcal{F}}(E, c_0) = P_{\mathcal{M}}\left(\varphi(\tau) \mid E, c_0\right) \quad \text{over } \mathbb{R}^d$$

You lose some information (trajectories with different structures may map to the same point), but you gain a distribution you can actually compute with — means, covariances, density estimates, regression targets. Everything in Sections 4–6 operates on $\hat{\mathcal{F}}$, not $\mathcal{F}$ directly.
