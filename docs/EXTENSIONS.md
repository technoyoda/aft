# Extensions — Future Work

These are formalisms that build on the core objects in [math.md](./math.md) but are not yet implemented in the library. They are recorded here so the derivations are not lost. If we care about these, they become future work.

---

## The Field Response Function — Fitting

With sufficient ablation data (see [math.md §5](./math.md#5-ablation-decomposition--the-core-derivation)), we could fit a function that predicts the field from the environment configuration.

### Linear Field Model

Start with a second-order response surface:

$$\boxed{W_{\mathcal{F}}(E) \approx w_0 + \sum_{j=1}^{m} w_j \, e_j + \sum_{i < j} w_{ij} \, e_i \, e_j}$$

Fit the coefficients $\{w_0, w_j, w_{ij}\}$ via ordinary least squares on the ablation data. The same form applies to [convergence](./METRICS.md#convergence--scalar):

$$C_{\mathcal{F}}(E) \approx c_0^{*} + \sum_{j=1}^{m} c_j \, e_j + \sum_{i < j} c_{ij} \, e_i \, e_j$$

### Nonlinear Field Model

If the linear model yields poor residuals, introduce a learned link function $g$:

$$W_{\mathcal{F}}(E) = g\!\left(\sum_{j=1}^{m} w_j \, e_j + \sum_{i < j} w_{ij} \, e_i \, e_j\right)$$

where $g: \mathbb{R} \to \mathbb{R}^{+}$ is a monotonic function (e.g., a small MLP, or a Gaussian process for uncertainty bounds). The key constraint: $g$ is fitted on trajectory data, not on model internals. It is a function of the environment mapping to field properties.

---

## The Transfer Function

Combine the ablation decomposition ([math.md §5](./math.md#5-ablation-decomposition--the-core-derivation)) with the response surface above and the [drift detector](./math.md#65-drift-detector) into a single predictive object.

**Definition (Field Transfer Function).**

$$\boxed{\Psi: (E, c_0) \;\longrightarrow\; \left(W_{\mathcal{F}},\; C_{\mathcal{F}},\; \mu_{\mathcal{F}},\; \lbrace\delta(s)\rbrace_{s \in \mathcal{S}}\right)}$$

$\Psi$ maps observable, controllable inputs to predicted field properties, fitted entirely from trajectory data.

This would enable direct answers to engineering questions:

| Question | How $\Psi$ Answers It |
|---|---|
| Will adding these tests help? | $\Delta C_{\mathcal{F}} = \Psi(E^{+}_{\text{tests}}) - \Psi(E^{-}_{\text{tests}})$ |
| Is the environment too noisy? | $W_{\mathcal{F}}(E) > \theta_{\text{noise}}$ — see [width](./METRICS.md#width--scalar) |
| Which factor matters most? | $\displaystyle\arg\max_j \left\lvert\frac{\partial W_{\mathcal{F}}}{\partial e_j}\right\rvert$ from fitted coefficients |
| Are two changes redundant? | $I_W(e_i, e_j) \approx 0 \implies$ yes — see [math.md §5.4](./math.md#54-interaction-effects) |
| When to intervene mid-run? | When $\delta(s) > \theta$ — see [drift detector](./math.md#65-drift-detector) |

### Dependencies

The transfer function requires:
- Sufficient ablation data across multiple environment configurations
- The response surface fitting above
- All core objects from [math.md](./math.md): $\varphi$ ([§3](./math.md#3-trajectory-embedding)), field metrics ([§4](./math.md#4-field-metrics), [METRICS.md](./METRICS.md)), ablation protocol ([§5](./math.md#5-ablation-decomposition--the-core-derivation)), horizons ([§6](./math.md#6-field-horizon))
