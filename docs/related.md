# Related Work

This document maps AFT against existing work. The honest assessment: AFT is a novel synthesis, not a novel discovery. Each of its core ideas has precedent in at least one existing field. The novelty is the specific combination — a user-defined behavioral projection with separated outcome labels, temporal slicing via state functions, and cloud metrics — packaged as a theoretical foundation for LLM agent engineering.

---

## Very close

### Quality-Diversity / MAP-Elites — the structural twin

**What it is.** MAP-Elites maintains an archive of solutions indexed by a user-defined "behavior descriptor" function that maps behavior to an m-dimensional vector. The archive is a discretized grid over this behavioral space. Fitness (outcome) is tracked separately from behavioral coordinates. Metrics include coverage (fraction of cells filled) and QD-score (sum of fitness across cells).

**What's the same.** A user-defined function projects agent behavior into a fixed-dimensional space. The behavioral space is the primary object. Outcomes are tracked separately from behavioral coordinates. Metrics characterize the distribution. The framing is "what does the distribution of behaviors look like?" — not just "what's the best score?"

**What's different.** MAP-Elites is an *optimization algorithm* — it actively searches for diverse, high-performing solutions. AFT is a *measurement framework* — it characterizes existing behavior without trying to change it. MAP-Elites discretizes the space into a grid. AFT treats it as a continuous point cloud. MAP-Elites has no temporal slicing ($\psi$ / horizons). It comes from evolutionary computation, not LLM agent engineering.

**Degree of relation.** Structurally the closest analogue. Same architecture (user-defined descriptor → space → archive → metrics), different purpose (optimization vs. measurement).

- [Quality-Diversity Papers](https://quality-diversity.github.io/papers.html)
- [MAP-Elites overview](https://www.emergentmind.com/topics/map-elites-algorithm)

---

### "When Agents Disagree With Themselves" (Feb 2025)

**What it is.** Runs the same ReAct agent on the same task 10 times and analyzes the distribution of action sequences. Finds 2.0–4.2 distinct action sequences per 10 runs, that 69% of divergence occurs at step 2, and that early divergence predicts lower accuracy.

**What's the same.** Repeated runs, distribution of behaviors as the primary object, computes where divergence occurs (analogous to horizon analysis), connects behavioral divergence to outcome.

**What's different.** Empirical study, not a theoretical framework. No formal Field object, no user-defined $\varphi$, no point-cloud framing. The analysis is ad-hoc rather than through a general formalism.

**Degree of relation.** Very close in spirit. Asks the same questions AFT asks. Does not propose general machinery to answer them.

- [arXiv:2602.11619](https://arxiv.org/html/2602.11619)

---

### "Capable but Unreliable: Canonical Path Deviation" (Feb 2025)

**What it is.** Defines a "canonical solution envelope" for a task, then measures how much stochastic agent trajectories deviate from it. Finds deviation is gradual and cumulative — each off-canonical step increases the probability of the next being off-canonical by 22.7 percentage points.

**What's the same.** Formalizes a space of trajectories, measures distributional properties across runs, treats behavioral distribution as the primary object. The "canonical envelope" is conceptually close to AFT's field center. The cumulative deviation finding is analogous to what horizon analysis reveals.

**What's different.** Formalized around deviation from a *single canonical path*, not a general point cloud in a user-defined space. No $\varphi$/$\psi$. More of a causal mechanism paper than a general theory.

**Degree of relation.** Very close. Independent discovery of similar ideas, different formalization.

- [arXiv:2602.19008](https://arxiv.org/abs/2602.19008)

---

### "Beyond Expected Return: Policy Reproducibility" (AAAI 2024)

**What it is.** Defines "policy reproducibility" as the dispersion of the return distribution across rollouts. Proposes MAD and IQR as metrics. Advocates Lower Confidence Bound instead of mean return.

**What's the same.** The thesis — "you must look at the distribution, not just the mean" — is the same philosophical move AFT makes. Proposes distributional metrics on repeated rollouts.

**What's different.** 1D return distributions only (not multi-dimensional behavioral space). No user-defined measurement function. No temporal slicing. RL-specific.

**Degree of relation.** Very close on the distributional insight. Narrower scope — outcome distribution only, not behavioral distribution.

- [arXiv:2312.07178](https://arxiv.org/abs/2312.07178)

---

## Moderately close

### Occupancy measures in RL

The distribution over state-action pairs induced by a policy. Standard RL maximizes a linear functional of this distribution. The occupancy measure is a distributional characterization of "what a policy does" — the same kind of object the Field aims to be. But it operates on the native MDP state-action space, not a user-defined projected behavioral space.

- [RL Theory lecture notes](https://rltheory.github.io/lecture-notes/planning-in-mdps/lec2/)

### Distributional RL (Bellemare et al., C51, QR-DQN)

Models the full distribution of returns, not just expected return. The core move — "the shape of the distribution carries information the expectation discards" — is identical to AFT's. But distributional RL models return distributions for *learning better policies*, not for characterizing behavior. 1D return, not multi-dimensional behavioral space.

- [arXiv:1707.06887](https://arxiv.org/abs/1707.06887)

### Novelty Search behavior characterizations

User-defined behavior characterization function mapping behavior to a vector in a metric space. Search is driven toward diversity. Close to AFT's $\varphi$, but used for driving optimization, not for measuring a fixed agent.

- [Learning Behavior Characterizations for Novelty Search](https://dl.acm.org/doi/10.1145/2908812.2908929)

### "Agent Drift: Quantifying Behavioral Degradation" (Jan 2026)

Multi-dimensional characterization of agent behavior over time (12 dimensions including tool usage patterns, reasoning stability). Focuses on drift within a session, not the static distribution across independent runs. Dimensions are fixed, not user-defined.

- [arXiv:2601.04170](https://arxiv.org/abs/2601.04170)

### "Understanding SE Agents: Thought-Action-Result Trajectories" (ASE 2025)

Empirical analysis of 120 trajectories from 3 SE agents, extracting behavioral features that distinguish success from failure. Does what AFT does, but ad-hoc — no general framework, no $\varphi$, no distributional formalism.

- [arXiv:2506.18824](https://arxiv.org/abs/2506.18824)

### STARE — trajectory embeddings via Transformer (Oct 2024)

Learned (neural) projection of agent trajectories into fixed-dimensional embeddings. The projection is learned, not user-defined. Representation learning, not analysis framework.

- [arXiv:2410.09204](https://arxiv.org/html/2410.09204)

### "The Necessity of a Unified Framework for LLM-Based Agent Evaluation" (Feb 2025)

Position paper calling for standardized, multi-dimensional agent evaluation beyond pass/fail. Calls for the thing AFT builds, but does not propose a specific formalism.

- [arXiv:2602.03238](https://arxiv.org/abs/2602.03238)

---

## Tangentially related

| Work | Relation to AFT |
|------|----------------|
| [AgentSpec](https://arxiv.org/abs/2503.18666) / [Agent Behavioral Contracts](https://arxiv.org/abs/2602.22302) | Runtime constraint enforcement — defines what behavior *should* be, not what the distribution *is* |
| [RAGEN / StarPO](https://arxiv.org/abs/2504.20073) | Trajectory-level RL for LLM agents — trajectories as optimization targets, not distributions to characterize |
| [Agentic UQ](https://arxiv.org/html/2601.15703v1) | Uncertainty quantification at decision points, not behavioral distribution characterization |
| [FAIRGAME](https://arxiv.org/abs/2512.07462) | Game-theoretic analysis of agent behavior — domain-specific, not a general measurement framework |
| [Formal-LLM](https://github.com/agiresearch/Formal-LLM) | Automata constraints on agent planning — formal specification, not behavioral measurement |
| [Architecture-Aware Evaluation](https://arxiv.org/abs/2601.19583) | Links architecture to metrics — which metrics for which architecture, not distributional analysis |
| LangSmith, LangFuse, Braintrust, Arize | Observability platforms — per-trace logging and aggregate stats, no distributional behavioral analysis |

---

## What is genuinely novel about AFT

1. **The Field as formal object** — a point cloud in a user-defined projected behavioral space, with outcome labels separated from behavioral coordinates. No prior work packages this exact construction.

2. **The $\varphi$ / $\psi$ contract** — user-defined measurement function for behavioral projection combined with user-defined state function for temporal slicing. MAP-Elites has the closest analogue (behavior descriptors) but in optimization, not measurement, and without temporal slicing.

3. **The specific metric vocabulary** — width, center, convergence, separation, skew as named quantities on the behavioral cloud. Individual distributional metrics exist in prior work, but not this collection applied to behavioral analysis.

4. **Horizons** — temporal slicing of the behavioral distribution by task phase, where each slice is itself a field with full metrics. No prior work has this.

5. **The search-space-bounding framing** — "have I bounded the search space well enough?" as the engineering question. Position papers call for better evaluation; AFT reframes the problem itself.

## What is NOT novel

- Analyzing distributions across repeated agent runs (behavioral consistency paper, policy reproducibility paper)
- User-defined behavioral descriptor functions (MAP-Elites, Novelty Search)
- Distributional thinking beyond expected values (distributional RL, policy reproducibility)
- Trajectory-level analysis of LLM agents (SE agent studies, canonical path deviation)
- The observation that LLM agents are stochastic and need multi-run analysis (multiple 2025 papers)
