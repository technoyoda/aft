from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from typing import Any

from .dimension import Dimension
from .metrics import FieldMetrics

T = TypeVar("T")


class Field(ABC, Generic[T]):
    """The central object in Agent Mechanics.

    agent_fields (aft) is built on the observation that agents are not thinking — they are
    searching (see https://technoyoda.github.io/agent-search.html). Given a
    task, an environment, and a prompt, an agent navigates a probabilistic
    trajectory toward reward signals shaped during training.

    The original formulation defines three primitives: the environment (the
    real-world state — repo, tools, permissions), the context window (everything
    the model has seen — system prompt, conversation history, tool outputs),
    and the field (the space of reachable behaviors conditioned on the context
    window plus the trained policy). Critically, the field is not static. It
    shifts every time a token enters the context window. During an inference
    rollout, the cycle is: the policy produces an action shaped by training
    and the current context, the environment returns feedback that enters the
    context window, the new tokens reshape the field, the system prompt
    persists and continuously constrains it, and the process repeats. Each
    cycle, the context grows and the field shifts. A precise prompt narrows
    it. Noise warps it. Permissions bound it from outside. A bad observation
    at step 2 stays in the context window for every subsequent step, warping
    the field each time.

    This per-timestep field is a theoretical object — it would require
    enumerating every possible future trajectory from every possible context
    state, which is infinite. In practice, we approximate it. Run an agent K
    times on the same task, record what it did each time, and measure each
    completed trajectory with a user-defined function. The resulting point
    cloud in behavioral space is an empirical approximation of the field —
    not the field at any single timestep, but the distribution of behaviors
    the field produced across all timesteps of all runs.

    This class is that approximation. A Field is a point cloud of behavioral
    vectors derived from agent trajectories, with outcome labels (success,
    failure, or a richer score) painted on each point as a separate quantity —
    not a coordinate of the cloud. The cloud describes *what trajectories did* ie 
    what the agent did in an aggreate way. The labels describe *how well they did*. 
    This separation is deliberate: it lets you ask where in behavioral space good 
    outcomes cluster, rather than conflating behavioral diversity with outcome noise.

    The Field exposes metrics (via `.metrics()`) that characterize the cloud's
    shape: how wide it is (behavioral diversity), where it centers (average
    behavior), how reliably trajectories succeed (convergence), and what behavioral
    dimensions separate success from failure (the separation vector). These
    metrics are the engineering interface to the field — they tell you what to
    change in the environment or prompt to improve agent performance.

    Because the Field is constructed from observed trajectories and never looks
    inside the model, it works as system identification on a black box. Swap
    the model, change the prompt, modify the environment — construct a new
    Field each time and compare the metrics. The Field is the unit of
    comparison.

    Usage:
        Subclass Field and implement two methods:

        - ``measure(trajectory)`` — the measurement function (phi). Takes a
          trajectory (your data, any shape) and returns a numpy vector in R^d.
          Each dimension should capture a behavioral property of the trajectory.
          This function determines which behavioral properties become
          dimensions of the point cloud. Only those properties exist for
          metrics to operate on — center, width, variance, separation, skew
          all compute over these dimensions. Any behavior not captured here
          is absent from the cloud and invisible to every metric downstream.
          Different measure functions produce different fields from the same
          trajectory data.

        - ``dimensions()`` — returns a list of Dimension objects, one per
          dimension. Each Dimension has a name and description that make
          metrics interpretable.

        Optionally override ``state(trajectory, t)`` — the state function
        (psi). It reduces the trajectory prefix up to step t into a discrete
        label representing semantic progress toward the goal. This enables
        horizon analysis: ``field.horizon("diagnosed")`` returns the sub-field
        of trajectories that reached the "diagnosed" state, with full metrics.

        Optionally override ``intent(trajectory, t)`` — the intent function
        (rho_pi). It reads the policy's operational character at each step.
        Unlike state, intent is non-monotonic — the policy can return to
        modes it exhibited earlier. This enables regime and program family
        analysis: ``field.regime("exploring")`` returns the sub-field where
        the exploring pattern appeared, ``field.program_family(("exploring",
        "executing"))`` returns trajectories sharing that program prefix.

        Then feed it trajectories via ``add()`` or ``ingest()``, and call
        ``metrics()`` to compute field metrics on the accumulated cloud.
    """

    def __init__(self):
        self._points: list[np.ndarray] = []
        self._outcomes: list[float] = []
        self._raw: list[T] = []
        self._state_sequences: list[list[str]] = []
        self._intent_sequences: list[list[str]] = []

    # ── The contract: user implements these ─────────────────────────────

    @abstractmethod
    def measure(self, trajectory: T) -> np.ndarray:
        """Map a trajectory to a behavioral vector in R^d.

        This is phi — the measurement function. It determines which
        behavioral properties become dimensions of the point cloud.
        Only those properties exist for metrics to operate on — any
        behavior not captured here is absent from the cloud and invisible
        to every metric downstream. Every dimension should capture a
        behavioral property — what the agent *did*, not how well it did.
        Outcome quality belongs in the `outcome` label, not here.

        Returns:
            np.ndarray of shape (d,)
        """
        ...

    @abstractmethod
    def dimensions(self) -> list[Dimension]:
        """Metadata for each dimension of measure()'s output.

        Returns a list of Dimension objects — one per dimension. Each
        Dimension has a name (short identifier) and description (what
        it measures). These make metrics interpretable.

        Returns:
            list of d Dimension objects.
        """
        ...

    # ── Optional: override for horizon analysis ──────────────────────

    def state(self, trajectory: T, t: int) -> str:  # noqa: ARG002
        """Reduce the trajectory prefix up to step t into a discrete state label.

        This is psi — the state function. It answers "where in the task
        is the agent?" by reducing the accumulated context into a single
        discrete label representing semantic progress toward the goal.

        The state progression should be linear:
            start → phase_1 → phase_2 → ... → done

        Behavioral variation within a phase is captured by measure(),
        not by subdividing state. State tracks *where*, measure tracks *how*.

        The default returns a constant — all trajectories share one state,
        meaning horizon() returns the full field. Override to define
        meaningful task phases and enable horizon analysis.

        Args:
            trajectory: the full trajectory.
            t: the step index (0-based).

        Returns:
            A discrete label (any hashable, typically a string).
        """
        return "_"

    def trajectory_length(self, trajectory: T) -> int:
        """Return the number of steps in the trajectory.

        Required when ``state()`` or ``intent()`` is overridden. The
        framework calls this to know how many times to invoke
        ``state(trajectory, t)`` and ``intent(trajectory, t)`` — once for
        each ``t`` in ``range(trajectory_length(trajectory))``.

        The default raises NotImplementedError. Override it to return the
        step count for your trajectory type.

        Args:
            trajectory: the raw trajectory (same type passed to measure/state).

        Returns:
            The number of steps.
        """
        raise NotImplementedError(
            "trajectory_length() must be implemented when state() is overridden. "
            "Return the number of steps in your trajectory type."
        )

    def intent(self, trajectory: T, t: int) -> str:  # noqa: ARG002
        """The policy's operational character at step t.

        This is rho_pi — the intent function. It answers "how is the policy
        operating?" by reading the qualitative character of the policy's
        behavior from the trajectory prefix up to step t.

        Unlike state(), intent is non-monotonic — the policy can return to
        an intent it exhibited earlier (e.g. explore → execute → explore).
        The sequence of intents, collapsed by run-length encoding, produces
        the program string — the skeleton of the policy's computational
        architecture.

        The default returns a constant — all steps share one intent,
        meaning regime() returns the full field. Override to define
        meaningful policy characters (e.g. "exploring", "executing",
        "recovering", "verifying").

        Args:
            trajectory: the full trajectory.
            t: the step index (0-based).

        Returns:
            A discrete label (typically a string).
        """
        return "_"

    # ── Ingestion ──────────────────────────────────────────────────────

    def add(self, trajectory: T, outcome: float) -> np.ndarray:
        """Measure a single trajectory and add it to the field.

        Args:
            trajectory: the raw trajectory (any type — your data).
            outcome: y(tau) — the quality label for this trajectory.

        Returns:
            The measured vector (the point added to the cloud).
        """
        vector = self.measure(trajectory)
        self._points.append(vector)
        self._outcomes.append(outcome)
        self._raw.append(trajectory)

        # Compute and store the state sequence for horizon analysis
        try:
            n = self.trajectory_length(trajectory)
        except NotImplementedError:
            n = 1  # default state() returns "_" — one step suffices
        seq = [self.state(trajectory, t) for t in range(n)]
        self._state_sequences.append(seq)

        intent_seq = [self.intent(trajectory, t) for t in range(n)]
        self._intent_sequences.append(intent_seq)

        return vector

    def ingest(self, trajectories: list[T], outcomes: list[float]) -> None:
        """Bulk-add trajectories to the field.

        Args:
            trajectories: list of raw trajectories.
            outcomes: list of y(tau) labels, same length.
        """
        if len(trajectories) != len(outcomes):
            raise ValueError(
                f"trajectories ({len(trajectories)}) and outcomes "
                f"({len(outcomes)}) must have the same length"
            )
        for traj, outcome in zip(trajectories, outcomes):
            self.add(traj, outcome)

    # ── The cloud ──────────────────────────────────────────────────────

    @property
    def points(self) -> np.ndarray:
        """The point cloud — (K, d) array of measured behavioral vectors."""
        if not self._points:
            return np.empty((0, self.d))
        return np.stack(self._points)

    @property
    def outcomes(self) -> np.ndarray:
        """Outcome labels — (K,) array of y(tau) values."""
        return np.array(self._outcomes)

    @property
    def K(self) -> int:
        """Number of trajectories in the field."""
        return len(self._points)

    @property
    def d(self) -> int:
        """Dimensionality of the behavioral space."""
        return len(self.dimensions())

    # ── Metrics ────────────────────────────────────────────────────────

    def metrics(self) -> FieldMetrics:
        """Compute all field metrics on the current cloud.

        Returns a FieldMetrics object with center(), width(), variance(),
        convergence(), skew(), separation(), and summary().
        """
        if self.K == 0:
            raise ValueError("Cannot compute metrics on an empty field.")
        return FieldMetrics(self.points, self.outcomes, self.dimensions())

    # ── Subsetting ─────────────────────────────────────────────────────

    def subset(self, mask: np.ndarray) -> _MaterializedField:
        """Return a new Field containing only the points matching the mask.

        Args:
            mask: boolean array of shape (K,).
        """
        pts = self.points[mask]
        outs = self.outcomes[mask]
        sub = _MaterializedField(pts, outs, self.dimensions())
        # Carry through sequences so horizon/regime/program_family chain
        sub._state_sequences = [
            s for s, m in zip(self._state_sequences, mask) if m
        ]
        sub._intent_sequences = [
            s for s, m in zip(self._intent_sequences, mask) if m
        ]
        return sub

    def success_region(self, threshold: float = 0.5) -> _MaterializedField:
        """The region of the field where trajectories succeeded."""
        return self.subset(self.outcomes >= threshold)

    def failure_region(self, threshold: float = 0.5) -> _MaterializedField:
        """The region of the field where trajectories failed."""
        return self.subset(self.outcomes < threshold)

    # ── Horizon ────────────────────────────────────────────────────────

    def horizon(self, state: str | list[str]) -> _MaterializedField:
        """The field horizon at one or more states.

        When given a single state, returns the sub-field of trajectories
        that passed through that state at some point during execution.

        When given a list of states, returns the sub-field of trajectories
        that passed through any of the listed states — treating multiple
        fine-grained state labels as a single semantic phase at query
        time without changing the state function.

        The horizon is itself a field. All metrics (center, width,
        convergence, separation, skew) work on it directly.

        Args:
            state: a single state label, or a list of state labels.
        """
        if isinstance(state, str):
            mask = self._trajectories_through(state)
        else:
            mask = self._trajectories_through_any(state)
        return self.subset(mask)

    @property
    def states(self) -> list[str]:
        """All states observed across trajectories, in first-seen order."""
        seen: dict[str, int] = {}
        for seq in self._state_sequences:
            for s in seq:
                if s not in seen:
                    seen[s] = len(seen)
        return list(seen.keys())

    def horizon_at(self, t: int) -> _MaterializedField:
        """The field horizon at step t.

        Returns the sub-field containing only trajectories that were
        still alive (had at least t+1 steps) at step t. This is the
        step-index proxy for state — useful when you don't have a
        meaningful state() override, or when you want to see the field
        at a specific point in raw execution time.

        Args:
            t: the step index (0-based).
        """
        mask = np.array([len(seq) > t for seq in self._state_sequences])
        return self.subset(mask)

    def _trajectories_through(self, state: str) -> np.ndarray:
        """Boolean mask: which trajectories passed through this state."""
        mask = np.zeros(self.K, dtype=bool)
        for k, seq in enumerate(self._state_sequences):
            if state in seq:
                mask[k] = True
        return mask

    def _trajectories_through_any(self, states: list[str]) -> np.ndarray:
        """Boolean mask: which trajectories passed through any of the given states."""
        state_set = set(states)
        mask = np.zeros(self.K, dtype=bool)
        for k, seq in enumerate(self._state_sequences):
            if state_set.intersection(seq):
                mask[k] = True
        return mask

    # ── Intent / Program ──────────────────────────────────────────────

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

    @staticmethod
    def _contains_subsequence(
        program: tuple[str, ...], pattern: tuple[str, ...]
    ) -> bool:
        """Check if pattern appears as a contiguous subsequence of program."""
        n, m = len(program), len(pattern)
        return any(program[i : i + m] == pattern for i in range(n - m + 1))

    def regime(self, pattern: str | tuple[str, ...]) -> _MaterializedField:
        """The sub-field of trajectories whose program string contains *pattern*.

        A single label is a length-1 pattern — ``regime("exploring")``
        checks if ``"exploring"`` appears anywhere in the program string.
        A tuple is a sequential motif — ``regime(("executing", "recovering",
        "executing"))`` checks for that contiguous subsequence.

        Regimes overlap: a trajectory can match many patterns.

        Args:
            pattern: a single intent label, or a tuple of labels.
        """
        if isinstance(pattern, str):
            pattern = (pattern,)
        mask = np.array(
            [
                self._contains_subsequence(self._program_string(k), pattern)
                for k in range(self.K)
            ]
        )
        return self.subset(mask)

    @property
    def intents(self) -> list[str]:
        """All intent labels observed across trajectories, in first-seen order."""
        seen: dict[str, int] = {}
        for seq in self._intent_sequences:
            for label in seq:
                if label not in seen:
                    seen[label] = len(seen)
        return list(seen.keys())

    def program_family(self, prefix: tuple[str, ...]) -> _MaterializedField:
        """The sub-field of trajectories whose program string starts with *prefix*.

        At full program length this is exact match. At shorter lengths
        this groups all trajectories sharing the same opening intent
        sequence — a subtree of the program trie.

        Program families partition at any prefix depth.

        Args:
            prefix: tuple of intent labels defining the required prefix.
        """
        plen = len(prefix)
        mask = np.array(
            [self._program_string(k)[:plen] == prefix for k in range(self.K)]
        )
        return self.subset(mask)

    @property
    def programs(self) -> list[tuple[str, ...]]:
        """All distinct program strings observed, in first-seen order."""
        seen: dict[tuple[str, ...], int] = {}
        for k in range(self.K):
            p = self._program_string(k)
            if p not in seen:
                seen[p] = len(seen)
        return list(seen.keys())


class _MaterializedField(Field[Any]):
    """A Field backed by pre-computed numpy arrays (no measure needed).

    Created by Field.subset(), success_region(), failure_region(), and
    Field.from_arrays(). Not intended for direct subclassing.
    """

    def __init__(
        self,
        points: np.ndarray,
        outcomes: np.ndarray,
        dims: list[Dimension],
    ):
        super().__init__()
        self._dims = dims
        # Directly populate the internal storage
        for i in range(len(points)):
            self._points.append(points[i])
            self._outcomes.append(float(outcomes[i]))

    def measure(self, trajectory: Any) -> np.ndarray:
        raise NotImplementedError(
            "A materialized field has no measure(). "
            "It was created from pre-computed points."
        )

    def dimensions(self) -> list[Dimension]:
        return self._dims

    @classmethod
    def from_arrays(
        cls,
        points: np.ndarray,
        outcomes: np.ndarray,
        dimensions: list[Dimension],
    ) -> _MaterializedField:
        """Construct a Field directly from numpy arrays.

        Use this when you already have embedded vectors and don't need
        the measure() pipeline.

        Args:
            points: (K, d) array.
            outcomes: (K,) array.
            dimensions: list of d Dimension objects.
        """
        return cls(points, outcomes, dimensions)


# Expose from_arrays as a top-level factory on Field for convenience.
Field.from_arrays = _MaterializedField.from_arrays
