# Tutorial 8: Comparing Across Tasks

**The idea:** Some behavioral properties are stable across tasks (the agent's signature). Others shift with the task. A task-agnostic field separates the two.

This tutorial runs the same agent on two different tasks (bug-fix from tutorial 1 and data-processing from tutorial 3) and measures both with a `TaskAgnosticField` -- dimensions that apply to any agent task without task-specific knowledge.

The question: which dimensions characterize the *agent* and which characterize the *task*?

## The framing problem

`scope_ratio` (fraction of operations on buggy.py) only makes sense for the bug-fix task. `exploration_ratio` (reads vs total tool calls) makes sense for any task. Designing task-agnostic dimensions forces you to separate what the agent *always does* from what it does *because of this task*. The stable dimensions are the agent's behavioral fingerprint. The shifting dimensions are the task's influence on the field.

## Prerequisites

Completed runs from both tutorial-1 and tutorial-3:

```bash
cd ../tutorial-1
python agent_field_flow.py run

cd ../tutorial-3
python data_flow.py run
```

## Running

```bash
cd examples/tutorial-8

# Uses latest GelloFlow and EnvShapeFlow runs
python cross_task_analysis.py

# Or specify run IDs
python cross_task_analysis.py \
  --run-id-1 12345 --run-id-3 67890
```

## What to look for

The script prints a comparison table with each dimension's mean across both tasks, the delta, and a signal classification:

- **STABLE:** Effect size < 0.5. This dimension doesn't change across tasks -- it's part of the agent's signature.
- **TASK-DEP:** Effect size > 1.0. This dimension shifts significantly between tasks -- the task shapes this behavior.
- **shifting:** In between. Partially task-dependent.

Look for:
- Is `verification_effort` stable? If so, the agent has a consistent testing strategy regardless of task.
- Does `commit_speed` shift? If the agent commits earlier on one task, the task difficulty or structure influences when the agent starts editing.
- Does `exploration_ratio` stay the same? If the agent reads the same fraction of the time on both tasks, that's a signature.

## Files

| File | What it does |
|------|-------------|
| `cross_task_analysis.py` | Pulls tutorial-1 and tutorial-3 data, builds TaskAgnosticField for each, compares |
| `cross_task_field.py` | `TaskAgnosticField` -- 6 task-agnostic dimensions |
