# Tutorials

Finished tutorials that teach empirical reasoning about agent behavior. Each one introduces a different way of thinking. The task in each tutorial is substrate — the lesson is always about framing.

## Tool choices

These tutorials use three tools. None of them are load-bearing to the theory — swap any of them and the ideas still work.

**`claude_agent_sdk`** runs the agent. It streams messages from Claude Code, which we capture as plain dicts (via `dataclasses.asdict()`). The trajectory is just a list of messages with tool calls and results. Any agent framework that gives you access to the execution trace works here. The only requirement is that you can iterate over what the agent did.

**Metaflow** handles experiment orchestration. Each tutorial is a [Metaflow flow](https://docs.metaflow.org/introduction/what-is-metaflow) with parallel executions (foreach over K runs). This gives us:
- Cheap parallelization with complicated experiment design: K agent runs execute concurrently
- Artifact persistence: trajectories, outcomes, and field points are stored per run and queryable later via the Client API
- Reproducibility: every run is versioned and inspectable

**agent_fields** is the core library (`import agent_fields as aft`). `Field`, `FieldMetrics`, `Dimension`, and `visualisations` are the only imports.

## Tutorial map

The design choices behind these tutorials is [given over here](../docs/tutorial-philosophy.md).

Tutorials 1-2 use Metaflow to orchestrate agents and collect data. Tutorials 4-6 and 8 are scripts that pull data from previous runs using the [Metaflow Client API](https://docs.metaflow.org/metaflow/client).

| Tutorial | Title | Type | What it teaches |
|----------|-------|------|-----------------|
| [tutorial-1](tutorial-1/) | Prompt Ablation | Flow | Behavior is measurable. The field is a unit of comparison. |
| [tutorial-2](tutorial-2/) | Horizon Analysis | Flow | Executions collapse into chains of labels. The distribution has temporal structure. |
| [tutorial-4](tutorial-4/) | Conditional Questions | Script | Global separation and conditional separation answer different questions. |
| [tutorial-5](tutorial-5/) | measure() is Your Hypothesis | Script | Same data, different measure(), different field. |
| [tutorial-6](tutorial-6/) | The Oracle as a Ruler | Script | An LLM can score semantic dimensions that counters can't capture. |
| [tutorial-7](tutorial-7/) | The Field Over Time | Flow | Fields are snapshots. Compare them across periods to detect drift. |
| [tutorial-8](tutorial-8/) | Comparing Across Tasks | Script | Some dimensions are agent signature, others are task-dependent. |

## Dependencies

```
tutorial-1 (standalone)
tutorial-2 (standalone)
tutorial-4 → requires tutorial-1 run
tutorial-5 → requires tutorial-1 run
tutorial-6 → requires tutorial-1 run + ANTHROPIC_API_KEY
tutorial-7 (standalone, uses tutorial-1 files)
tutorial-8 → requires tutorial-1 run
```
