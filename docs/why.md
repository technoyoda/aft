# Why Agent Field Theory

## The problem

AI agents are [search programs, not thinking machines](https://technoyoda.github.io/agent-search.html). Each run dynamically unrolls a trajectory shaped by a [trained policy](https://technoyoda.github.io/agent-search.html#heading-5) responding to [environmental feedback](https://technoyoda.github.io/agent-search.html#heading-6) — user queries, API responses, file contents, tool outputs. The state machine assembled by this process is different every time, because the environment is different every time and the policy is stochastic. The prompt constrains which trajectories are likely, but it does not determine a fixed program.

This means "is my agent following instructions?" is not the right framing. The agent may be perceived as executing instructions, but it is navigating a search space that is [conditioned at each timestep](https://technoyoda.github.io/agent-search.html#heading-6) by the growing context window and shaped by [pre-training](https://technoyoda.github.io/agent-search.html#heading-4), [RL](https://technoyoda.github.io/agent-search.html#heading-5), and the environment. The prompt narrows that space. It does not control the program.

The right question is: **have I bounded the search space well enough that the agent's stochastic search consistently lands where I need it?**

Answering that question requires measuring the behavioral distribution across repeated runs — not inspecting any single trajectory, not checking pass/fail, but characterizing the *shape* of what the agent does when you let it run many times. Existing evaluation frameworks measure outcomes. They do not give you a language for measuring the structure of the behavioral space itself — where trajectories cluster, where they scatter, at what phase they diverge, and what behavioral properties separate success from failure.

AFT provides that language.

## What the user provides

The user implements two functions, plus a helper:

- **`measure(trajectory)`** — the measurement function $\varphi$. Defines what behavioral properties to extract from a completed trajectory. This is the projection from raw trajectory data into a fixed-dimensional space where analysis is possible. It determines what the field can see — any behavior not captured by a dimension is invisible to every metric downstream.

- **`state(trajectory, t)`** — the state function $\psi$. Defines what semantic progress looks like for their task. Returns a discrete label at each step: "start", "diagnosed", "fixed", "verified". This provides the vocabulary for slicing the field by phase. This allow us to ask questions like "at what point do trajectories start diverging?" The user defines states at full resolution; queries can group them at analysis time.

- **`trajectory_length(trajectory)`** — tells the framework how many steps the trajectory has, so that `state()` can be evaluated at each step.

## What they get back

Analytical tools grounded in the [mathematical formulation](./math.md):

- **Field metrics** — [width](./METRICS.md#width--scalar) (behavioral diversity), [center](./METRICS.md#center--mathbbrd) (average behavior), [variance](./METRICS.md#variance--mathbbrd) (per-dimension spread), [convergence](./METRICS.md#convergence--scalar) (outcome reliability), [separation](./METRICS.md#separationthreshold--mathbbrd) (what distinguishes success from failure), [skew](./METRICS.md#skewcost_dim--scalar) (whether success is cheap or expensive along a dimension).

- **Horizons** — the field at a specific [state](./math.md#62-the-field-horizon). Every horizon is itself a field with full metrics. Walk the horizon chain to find where trajectories diverge. Compare horizons across configurations to see whether an intervention changed behavior at the phase that matters.

- **Ablation decomposition** — [compare fields](./math.md#5-ablation-decomposition--the-core-derivation) across different environments, prompts, or models. Isolate which lever moved the distribution and how.

These metrics are a vocabulary. They are the language the framework provides for reasoning about the behavioral point cloud. The user defines the cloud's content (via `measure()`), but the grammar for talking about it — center, width, convergence, separation, skew, horizons — is a design choice made by the framework. Different metrics would make different questions speakable. These metrics are optimized for an engineering workflow: is there a problem (convergence), where is it (separation, horizons), what kind is it (skew), what should I change (ablation). That workflow is a choice, not a mathematical inevitability. The user works backwards from decisions to dimensions. The metrics give them the language to evaluate whether those dimensions are doing their job.

The intention of these tools is a different frame of thinking. Not "did the agent succeed?" but "what does the behavioral distribution look like, and how is it shaped by the boundaries I've set?"

## The Field as logical abstraction

The [Field](./math.md#2-the-empirical-field-as-a-distribution) is a logical abstraction over the search space. Since we cannot know the full search space of the policy — the [pre-trained distribution](https://technoyoda.github.io/agent-search.html#heading-4) and the [RL-shaped policy](https://technoyoda.github.io/agent-search.html#heading-5) are opaque — we construct fields as a way to measure how much the agent aligns with the search space we are bounding via the environment and the system prompt.

The choice of `measure()` determines the space. The choice of `state()` determines the temporal vocabulary. Different choices produce different fields from the same trajectory data, answering different questions. This is by design — the formalism is general, the measurement is specific to the task and the decision the user needs to make.

### The state function as projection

Every task has logical structure — phases, dependencies, prerequisite knowledge, decision points. "Fix the bugs" implies: read the file, identify the problems, fix them, verify. That structure is a graph, potentially complex, and it is dynamic — every token that enters the context window can reshape what is reachable, effectively creating a new graph at every step.

The state function is the user's projection of that structure onto the linear trace. The task's graph might branch, loop, or shift with each observation. But the trace already happened — it is a chain. `state()` labels positions on that chain using the user's understanding of the task's logical phases. The linearity is not a claim that the task is simple. It is a compression. The user decides which aspects of the task's logical progression matter for their analysis.

This makes `state()` an invariant. The underlying graph shifts with every token. The state labels remain stable — "I don't care how the graph changed between steps, I care whether the agent has reached the 'diagnosed' phase." That stability is what makes horizons useful: you can slice the behavioral distribution by phase and ask questions about each slice, regardless of how the agent arrived there.

The Field gives users hooks to define their own state machines over agent trajectories, and the right APIs to evaluate how an agent is behaving within those definitions. The engineering problem becomes: **shape the search space** — via the prompt, the environment, the available tools — **so that the distribution of trajectories concentrates where you need it.** AFT gives you the measurements to evaluate whether your shaping is working, and the diagnostics to find where it breaks down.
