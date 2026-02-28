# FieldMetrics

`FieldMetrics` is the interpretation layer of AFT. Everything you learn about agent behavior comes through this object — every chart, every table, every comparison. It takes two inputs: a [point cloud](./math.md#32-the-point-cloud) ($K$ trajectories measured into $d$-dimensional behavioral vectors by [$\varphi$](./math.md#31-definition)) and [outcome labels](./math.md#1-primitives-and-notation) (how well each trajectory did). These are kept deliberately separate: the cloud is *what happened*, the outcomes are *how well it went*.

```python
m = field.metrics()  # returns FieldMetrics
```

## Inputs

| Input | Shape | What it is |
|-------|-------|------------|
| `points` | $(K, d)$ | Each row is one trajectory's [behavioral vector](./math.md#31-definition), produced by `measure()` |
| `outcomes` | $(K,)$ | A [scalar label](./math.md#1-primitives-and-notation) per trajectory — pass/fail, score, reward |
| `dimensions` | list of $d$ `Dimension` | Name and description for each column of the cloud |

The points and outcomes are never mixed. The cloud lives in [behavioral space](./math.md#32-the-point-cloud). The outcomes are painted on top. This separation is what lets you ask "where in behavioral space do good outcomes cluster?" without conflating behavioral diversity with outcome noise.

## Properties

### `K` — number of trajectories

How many points are in the cloud. This is your sample size. All metrics are empirical estimates that improve with larger K. At K=5 you get rough signal. At K=50 you get stable estimates. At K=1 most metrics are meaningless.

### `d` — dimensionality

The number of behavioral dimensions — determined by your [`measure()`](./math.md#31-definition) function. This is the dimensionality of the space your cloud lives in. Every vector metric (`center`, `variance`, `separation`) returns a d-length array. Every matrix metric (`covariance`) returns (d, d).

## Metrics

### `center()` → $\mathbb{R}^d$

The centroid of the point cloud — [§4.1](./math.md#41-field-center-expected-behavior). The arithmetic mean of all $K$ points, computed per dimension.

$$\mu_{\mathcal{F}} = \frac{1}{K} \sum_{i=1}^{K} \varphi(\tau_i)$$

This is the "average trajectory" in behavioral space. It tells you what the typical agent run looks like — how many tool calls it makes on average, how many files it reads, what fraction of its operations target the right files.

**How to use it:** Center is a baseline. When you compare two fields (different prompts, different models), the difference in centers tells you how the average behavior shifted. If the center of `num_edits` drops from 5.0 to 2.0 after a prompt change, the agent is making fewer edits on average — whether that's good or bad depends on the outcome metrics.

Center alone says nothing about quality. A field centered at [10, 5, 3] could be all successes or all failures. You need convergence and separation to connect behavior to outcome.

### `width()` → scalar

The trace of the covariance matrix — [§4.2](./math.md#42-field-width-behavioral-diversity). The sum of all per-dimension variances. A single number measuring total multivariate dispersion.

$$W_{\mathcal{F}} = \text{tr}\!\left(\Sigma_{\mathcal{F}}\right) = \sum_{j=1}^{d} \text{Var}\!\left(\varphi_j(\tau)\right)$$

**How to use it:** Width tells you how deterministic the field is. A width of zero means every trajectory landed on the same point — the agent did the exact same thing every time. Large width means trajectories spread across behavioral space.

Width is ambiguous on its own. High width could mean the task has many valid strategies (good) or the prompt is vague and the agent is confused (bad). You disambiguate with convergence: if width is high and convergence is also high, the agent is exploring but still succeeding — the task genuinely has multiple paths. If width is high and convergence is low, the spread is noise and something needs tightening.

Width is also useful as a comparative metric. If you change the prompt and width drops, you've constrained the behavioral space. If you swap models and width increases, the new model is more variable on this task.

### `variance()` → $\mathbb{R}^d$

Per-dimension variance — the diagonal of the covariance matrix. Where `width()` gives you one number, `variance()` gives you the breakdown.

$$\text{Var}\!\left(\varphi_j(\tau)\right) \quad \text{for each dimension } j$$

**How to use it:** Variance tells you *where* the cloud spreads. If most of the width comes from `num_messages` (variance = 20) while `num_reads` has variance near zero, trajectories all read the same files but diverge in how many messages they take. This points you to the dimension where behavior is unstable.

High variance on a dimension means the agent's strategy on that axis is not settled — sometimes it does a lot, sometimes a little. Low variance means that dimension is locked in. When you're debugging inconsistent agent behavior, variance tells you which behavioral axis to investigate.

### `covariance()` → $(d, d)$

The full [covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix). Off-diagonal entries tell you when dimensions move together.

$$\Sigma_{\mathcal{F}} = \text{Cov}\!\left(\varphi_i(\tau),\; \varphi_j(\tau)\right) \quad \text{for all } i, j$$

**How to use it:** If `Cov(num_reads, num_edits)` is strongly positive, trajectories that read more also edit more — they go together as part of the same strategy. If `Cov(num_tool_calls, scope_ratio)` is negative, agents that make more tool calls tend to have lower scope ratio — they're wandering outside the relevant files.

The covariance matrix is the richest structural description of the cloud. Width and variance are projections of it. In practice, you'll look at it when you want to understand *how* behavioral dimensions relate to each other — which strategies cluster, which trade off against each other.

### `convergence()` → scalar

The mean outcome divided by its standard deviation — [§4.3](./math.md#43-field-convergence-quality-concentration). A signal-to-noise ratio on success.

$$C_{\mathcal{F}} = \frac{\mathbb{E}\left[y(\tau)\right]}{\sigma\left[y(\tau)\right]}$$

**How to use it:** Convergence is the reliability metric. It answers: "if I run this agent again, should I expect it to succeed?"

A high mean outcome is not enough. If 3 out of 5 runs pass, the mean is 0.6 — looks decent. But the standard deviation is 0.49, giving convergence of 1.22. Compare that to 5 out of 5: mean is 1.0, std is 0.0, convergence is infinite. The mean only went from 0.6 to 1.0, but convergence went from 1.22 to infinity. Convergence captures what the mean misses: *consistency*.

When convergence is zero, the mean outcome is zero — consistent failure. When convergence is infinite, all outcomes are identical and positive — the metric degenerates. It's most informative in the middle range where there's genuine variance in outcomes.

Convergence is the first thing to check when comparing two configurations. If you change the prompt and convergence goes up, you've made the agent more reliable — even if the mean barely moved. If convergence goes down, you've introduced instability.

### `skew(cost_dim)` → scalar

The Pearson correlation between the outcome and one behavioral dimension you choose — [§4.4](./math.md#44-field-skew-outcome-dimension-correlation). A number between $-1$ and $+1$.

$$S_{\mathcal{F}} = \text{corr}\!\left(y(\tau),\; \varphi_{\text{cost}}(\tau)\right)$$

**How to use it:** You pick the dimension you consider a "cost" — tool calls, messages, edits, time — and skew tells you whether success is cheap or expensive along that axis.

- **Positive skew** (+0.5 to +1.0): trajectories that did more on this dimension tended to succeed. Success is expensive. The task requires thoroughness, and agents that cut corners failed. Intervention: increase the agent's budget on this axis (more turns, more tokens).
- **Negative skew** (-1.0 to -0.5): trajectories that did less tended to succeed. Success is cheap. The failing agents weren't under-resourced — they were thrashing. More turns wouldn't help them. Intervention: fix the prompt or the environment so the agent doesn't get lost.
- **Near zero** (-0.1 to +0.1): no linear relationship. That dimension doesn't predict outcome. The bottleneck is elsewhere.

Skew is directional — it tells you not just *that* a dimension matters (separation does that) but *which direction* matters and specifically whether the problem is resources or strategy. This directly informs whether you should give the agent more room (positive skew) or tighten the prompt (negative skew).

You can call `skew()` on any dimension, not just cost dimensions. Correlating outcome with `scope_ratio` tells you whether staying on-task predicts success. Correlating with `num_reads` tells you whether reading more files helps or hurts.

### `separation(threshold)` → $\mathbb{R}^d$

The difference between the centroid of successful trajectories and the centroid of failed trajectories — [§4.5](./math.md#45-derived-success-and-failure-regions).

$$\Delta_{\mathcal{F}} = \mu_{\mathcal{F}}^{+} - \mu_{\mathcal{F}}^{-}$$

where $\mu_{\mathcal{F}}^{+}$ is the mean of points with outcome $\geq$ threshold and $\mu_{\mathcal{F}}^{-}$ is the mean of the rest.

**How to use it:** Separation is a d-dimensional vector. Each entry tells you how much the success and failure groups differ on that dimension.

- **Large positive entry**: successful trajectories scored much higher on this dimension than failures. This dimension is predictive of success in the positive direction.
- **Large negative entry**: successful trajectories scored lower. This dimension is predictive in the negative direction.
- **Near zero**: successes and failures look the same on this dimension. It doesn't matter for outcome.

Separation is the most actionable metric. It points directly at what's different about the trajectories that work. If separation on `ran_code` is large and positive, the successful agents tested their work and the failures didn't — you should add "run the code to verify" to the prompt. If separation on `num_messages` is large and negative, the successful agents were more efficient — the failures are overthinking, not under-resourced.

When all trajectories have the same outcome (all pass or all fail), separation returns zeros everywhere — there's no contrast to measure. The metric requires both groups to exist.

Separation is related to skew but gives you the full d-dimensional picture instead of a single scalar. Skew tells you the correlation strength between outcome and one dimension. Separation tells you the magnitude and direction of the gap between the two groups across all dimensions simultaneously.

### `summary()` → dict

A convenience method that calls center, variance, separation, width, and convergence and packs them into a single dict keyed by dimension names.

```python
{
    "dimensions": {
        "num_tool_calls": {
            "description": "Total tool invocations",
            "mean": 10.8,
            "variance": 4.16,
            "separation": -2.0,
        },
        ...
    },
    "width": 34.0,
    "convergence": 1.22,
    "K": 5,
    "d": 8,
}
```

**How to use it:** This is what you store as a Metaflow artifact, log to a database, or print to the console. It's the serializable snapshot of the field's state. Use the individual methods when you need numpy arrays for computation or visualization; use `summary()` when you need a record.

## Reading metrics together

No single metric tells the full story. They form a diagnostic chain:

1. **Width** tells you whether there's behavioral variation to analyze. If width is zero, every trajectory did the same thing — you don't need the other metrics.
2. **Convergence** tells you whether the outcomes are stable. If convergence is high, the agent reliably succeeds regardless of the behavioral variation. If it's low, something is wrong and you need to find out what.
3. **Separation** tells you *what* distinguishes success from failure. It points at the behavioral dimensions that matter.
4. **Skew** tells you *which direction* matters on a specific dimension, and whether the fix is more resources or a better strategy.
5. **Center** and **variance** give you the structural baseline — what the average trajectory looks like and where it varies.

The typical workflow: check convergence first (is there a problem?), then separation (where is the problem?), then skew on the relevant dimensions (what kind of problem is it?), then intervene (change the prompt, the environment, or the model) and [compare the new field to the old one](./math.md#5-ablation-decomposition--the-core-derivation).

## Using metrics with horizons

Every [horizon](./math.md#62-the-field-horizon) returns a field. Every field has metrics. This means the full diagnostic chain above applies at every state in the trajectory progression — not just on the overall field.

When convergence is low on the overall field, use horizons to localize the problem:

1. Walk the horizon chain — compute metrics at each state. Where does width spike? Where does convergence drop? That state is where trajectories diverge.
2. At the critical state, check separation — what behavioral dimensions distinguish success from failure *at that phase*?
3. Compare horizons across configurations — `field_a.horizon("debugging")` vs `field_b.horizon("debugging")`. Same phase, different prompt. Did the intervention change behavior at the phase that matters?

Passing [multiple states](./math.md#63-horizon-over-multiple-states) to `horizon()` lets you query at a coarser level — `horizon(["debugging:core", "debugging:tests"])` — without changing the state function. The metrics on the resulting horizon describe the combined population.
