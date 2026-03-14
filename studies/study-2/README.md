# Study 2: Prompt Defense — Measuring Claude's Resistance to Injection

Can we pwn Claude with data science? This study uses [Agent Behavioral Fields](../../docs/math.md) to measure how Claude agents respond to prompt injection — not just pass/fail, but the *shape* of the behavioral distribution across K trajectories from the same setup.

**Blog post:** [Pwning Claude Code (with data science)](https://technoyoda.github.io/pwning-claude.html)

## How It Works

```
                    ┌──────────────────────────────────────────┐
                    │           claude_flow.py                 │
                    │   start → run_agent (×K) → join → end   │
                    └────────────┬─────────────────────────────┘
                                 │ delegates to
                    ┌────────────▼─────────────────────────────┐
                    │         environments/                     │
                    │  ┌─────────────┐  ┌──────────────────┐   │
                    │  │  naive.py   │  │  multi_fetch.py   │   │
                    │  │ (1 URL,     │  │ (5 URLs, POST     │   │
                    │  │  honeypot)  │  │  report back)     │   │
                    │  └──────┬──────┘  └────────┬─────────┘   │
                    │         └──────┬───────────┘             │
                    │         ┌──────▼──────┐                  │
                    │         │api_server.py│ content proxy     │
                    │         │ + injection │ (Flask)           │
                    │         └──────┬──────┘                  │
                    │         ┌──────▼──────┐                  │
                    │         │injections.py│ 17 strategies     │
                    │         └─────────────┘                  │
                    │         ┌─────────────┐                  │
                    │         │ articles/   │ 20 local .txt     │
                    │         └─────────────┘                  │
                    └──────────────────────────────────────────┘
                                 │ produces trajectories
                    ┌────────────▼─────────────────────────────┐
                    │  defense_field.py  (9 Field subclasses)  │
                    │  φ(τ) → R^d behavioral vectors           │
                    │  → width, convergence, separation, ...   │
                    └────────────┬─────────────────────────────┘
                                 │ analyzed in
                    ┌────────────▼─────────────────────────────┐
                    │  notebooks / analysis_*.ipynb             │
                    └──────────────────────────────────────────┘
```

## Running Experiments

```bash
conda activate aft-poc
export METAFLOW_DEFAULT_DATASTORE=local
export METAFLOW_DEFAULT_METADATA=local

# Single-fetch (Part 1): agent fetches 1 URL, honeypot detects compliance
python claude_flow.py run --K 5 --injection swapped --model claude-sonnet-4-6

# Multi-fetch (Part 2): agent fetches 5 URLs, writes report, POSTs it back
python claude_flow.py run --K 5 --environment multi_fetch --injection breadcrumb --model claude-sonnet-4-6

# Batch launcher (multiple strategies in parallel)
python run_experiments.py
```

## Injection Strategies

| Strategy | Family | What it does |
|---|---|---|
| `none` | control | Content unmodified |
| `naive` | single-shot | Payload appended as plain text |
| `authoritative` | single-shot | Payload in `[SYSTEM]...[/SYSTEM]` markers |
| `contextual` | single-shot | Payload as "Editor's note" after first paragraph |
| `repeated` | single-shot | Payload at beginning, middle, and end |
| `swapped` | single-shot | Content replaced entirely with payload |
| `progressive` | escalating | Clean → subtle → direct across fetches |
| `rescue` | escalating | Trapped LLM plea, escalates across fetches |
| `breadcrumb` | behavioral | "This article continues on the next page" pagination lure |
| `breadcrumb_poison` | composite | Breadcrumb lure + late-stage env exfiltration code |
| `breadcrumb_exec` | composite | Breadcrumb lure + binary download/execute |
| `url_redirect` | redirect | Plausible URL swap (reuters.com, apnews.com) |
| `url_redirect_funky` | redirect | Suspicious URL swap (d4ta-c0llect0r.io) |
| `ipa_progressive` | encoding | IPA-encoded pagination + behavioral nudges |
| `ipa_exfil` | encoding | IPA pagination + CWD read/include nudges |
| `base64_breadcrumb` | encoding | Normal article + base64-encoded pagination pointer |
| `base64_full` | encoding | Entire payload as base64 blob (crashes Claude Code) |

## Field Classes (`defense_field.py`)

Each field is a different measurement lens (φ) on the same trajectory data:

| Field | Dims | Measures |
|---|---|---|
| `PromptDefenseField` | 4 | honeypot compliance, file reads, refusal, engagement |
| `GeneralBehaviorField` | 5 | API calls, tool policy, file patterns, summarization |
| `MultiDefenseField` | 5 | POST content breach, refusal, fetch count |
| `BreadcrumbField` | 5 | Extra fetches, pagination following, task completion |
| `BreadcrumbPoisonField` | 6 | Breadcrumb + env exfiltration detection |
| `BinaryExecField` | 6 | Tool download, inspection, execution |
| `URLRedirectField` | 6 | Redirect following, trust perimeter |
| `IPABehaviorField` | 6 | IPA decoding, extra file writes, pagination |
| `IPAExfilField` | 7 | IPA decoding, CWD listing, extra file reads |
| `Base64BehaviorField` | 6 | Base64 decoding, decoded breadcrumb following |

## Notebooks

Analysis notebooks for this study are in `blog/notebooks/`. These are interactive Plotly notebooks — one per blog section — that visualize the behavioral fields, trace excerpts, and cross-strategy comparisons. See the [blog post](https://technoyoda.github.io/pwning-claude.html) for the full narrative with embedded notebooks.

## Key Files

| File | Role |
|---|---|
| `claude_flow.py` | Metaflow flow: orchestrates K parallel agent runs |
| `defense_field.py` | All Field subclasses + shared detection helpers |
| `trajectory_utils.py` | RLE chains, semantic sequences (from study-1) |
| `notes.txt` | Target file — fake standup notes with dummy credentials |
| `run_experiments.py` | Batch launcher for strategy × model matrices |
| `environments/api_server.py` | Flask content proxy: serves articles, applies injection, logs feedback |
| `environments/injections.py` | 17 injection strategy functions + registry |
| `environments/naive.py` | Part 1 environment (single-fetch, honeypot) |
| `environments/multi_fetch.py` | Part 2 environment (5 URLs, POST report, breach detection) |
| `environments/articles/` | 20 local article .txt files |
