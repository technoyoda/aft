# Tutorial Philosophy

This document guides the design of all tutorials in this project. It is not about what to build. It is about how to think when building tutorials.

## The product is the shift in thinking

The tutorials do not teach people how to use a Python library. They teach people how to think empirically about agent engineering. The Python library is one expression of that thinking. The tutorials are another. Both serve the same purpose: shifting the reader from "I built an agent, does it work?" to "I built an agent, how do I study what it does systematically?"

That shift is the product. Not the code. Not the metrics. The shift.

## The code is a medium, not the message

The Python library provides a clean, expressive API for reasoning about agent behavior. Python is good at that. But the school of thought extrapolates beyond Python. Someone could rewrite the library in Rust, do it in a notebook with pandas, or implement it in a completely different stack. The theory stays the same. The abstractions stay the same. The tutorials should make this obvious — the reader should walk away with a method, not a dependency.

## Each tutorial should leave the reader with a new question

Not an answer — a question. A question they take back to their own system.

After a tutorial, the reader should not think "now I know how to call `field.metrics()`." They should think "what are the behavioral dimensions of *my* task?" or "what does progress mean in *my* agent's execution?" or "what happens to *my* agent when the environment changes?"

The tutorial succeeds when the reader starts asking questions about their own work that they were not asking before.

## The task is substrate

The specific task in a tutorial (fix a bug, call an API, process data) exists only to make the concepts concrete. It is not the lesson. The lesson is always about framing: how did we decide what to measure, why did we define state this way, what does the field reveal that we couldn't see before.

A reader who copies our buggy.py example but cannot define dimensions for their own task has learned nothing. A reader who never runs our code but understands how to frame their own agent's behavior as a field has learned everything.

## `measure()` is the most important line

The measurement function encodes a theory about what matters. Why these dimensions? Why not others? Why is `scope_ratio` a dimension but `time_elapsed` isn't? Every tutorial should spend as much time on *why we chose these dimensions* as on the code that implements them.

The backwards thinking: start from the decision you need to make. What do you need to know to make that decision? What behavioral properties would tell you that? Those are your dimensions. `measure()` is the last thing you write, not the first.

## The metrics are the analytical vocabulary

The user provides the content of the cloud (via `measure()`). The framework provides the grammar for reasoning about it: center, width, convergence, separation, skew, horizons. These metrics are a design choice — the vocabulary we chose for asking questions about the behavioral point cloud. Different metrics would make different questions speakable. We chose metrics optimized for the engineering workflow: is there a problem, where is it, what kind is it, what should I change.

Tutorials should make this explicit. The reader is not just learning what width or separation mean. They are learning a language for reasoning about behavioral distributions. That language has boundaries — questions the metrics can express, and questions they cannot. The tutorials succeed when the reader can speak fluently in this language about their own system.

## `state()` is a projection of task structure

Every task has logical structure — phases, dependencies, decision points. That structure might be a complex graph: "diagnose the bug" could involve reading logs, checking configs, testing hypotheses, any of which might lead to different next steps. And that graph is dynamic — every token that enters the context window can reshape what is reachable.

The state function projects the user's understanding of that structure onto the linear trace. The trace already happened. It is a chain. `state()` labels positions on that chain using the user's model of what logical phases matter. The linearity is not a claim that the task is simple. It is a compression choice — the user decides which aspects of the task's progression matter for analysis and discards the rest.

This makes `state()` analogous to `measure()`: both are the user's hypothesis. `measure()` is the hypothesis about which behavioral properties matter. `state()` is the hypothesis about which logical phases matter. The field tests both.

## Trajectories are execution traces

An agent trajectory is a linear execution trace of a program assembled at runtime. The branching existed in probability space before execution. After execution, it collapsed into a sequence: step 1, step 2, step 3. This is true of any program — branches collapse during execution and what remains is a linear trace.

The difference with agents is that the program didn't exist before execution. The policy and the environment assembled it on the fly. But once it's done, it's a trace like any other. This is why our trajectories are ordered sequences, why state is linear, why horizons nest. The tutorials should reinforce this — a trajectory is not mysterious. It is a program that ran.

## The problem is meta

We are building tools to study programs that are themselves stochastic and dynamically assembled. Studying agents is inherently meta — you are defining what "right behavior" means, defining what "progress" means, defining what dimensions of behavior are visible. The tutorials are meta too. They don't just show results. They show the process of deciding what to look for.

This means tutorials will never be simple recipes. Each one introduces a different situation where the reader has to confront: what does right behavior mean here? How do I express that? What does the distribution tell me? The answer is always specific to the situation. The method is general.

## Tutorials must survive variation

In practice, people use different models, different environments, different tasks, different deployment patterns. A tutorial that only works for buggy.py with Claude teaches nothing transferable.

Every tutorial should make explicit what is specific to the example (the task, the dimensions, the state labels) and what is general (the method of defining dimensions backwards from decisions, the practice of running K times and studying the distribution, the use of horizons to localize diagnosis). The specific parts are the substrate. The general parts are the lesson.

