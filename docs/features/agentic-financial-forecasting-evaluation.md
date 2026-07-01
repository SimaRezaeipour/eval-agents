# Feature: Agentic Financial Planning & Forecasting Evaluation

> **Date:** 2026-06-29
> **Status:** Draft
> **Branch:** `feature/agentic-financial-forecasting-evaluation`
> **Plan:** `docs/plans/2026-06-29-agentic-financial-forecasting-evaluation.md`
> **Team:** Violet Khazali (Lead), Kaiqi Cheng, Kanishk Patoliya, Sima Rezaeipour, Uday Sharma, Ujjwal Khanna
> **Facilitators:** Tarun Joseph (Vector Project), Sara Ketabi / Bahar Sateli (Vector Technical), Jaswinder Narain (Company PoC)

---

## Overview

Build and evaluate a Plan-and-Execute agentic LLM system that performs multi-step financial
planning and forecasting over historical NASDAQ daily price data. The agent decomposes
natural-language queries into explicit plans, executes deterministic tool calls against
NASDAQ OHLCV data, and synthesizes structured, explainable answers. A single-shot LLM
baseline serves as the lower-bound comparison. Six evaluation metrics are aggregated into
an **Agent Reliability Score (ARS)** across 30 gold-annotated benchmark queries.

The focus is **agent reasoning and planning quality** — not market-prediction accuracy.

---

## Requirements

### Functional

- [ ] FR-1: The system accepts a natural-language financial query and returns a structured `FinalAnswer` containing a summary, numeric claims with source citations, and caveats.
- [ ] FR-2: The planner emits an explicit ordered `Plan` (list of `PlanStep`) before any tool calls are made, enabling plan-level evaluation.
- [ ] FR-3: The executor runs tools deterministically; each step's output is stored in a shared state dict keyed by step ID.
- [ ] FR-4: The synthesizer uses only observations in the scratchpad; it must not fabricate numeric values.
- [ ] FR-5: Every system run produces a JSONL trace capturing query, plan, state, errors, answer, latency, and token counts.
- [ ] FR-6: A single-shot baseline system accepts the same queries, receives a raw CSV snippet as context, and returns a trace with the same schema as the agent.
- [ ] FR-7: The benchmark contains 30 gold-annotated queries across 3 tiers (T1 Descriptive, T2 Comparative, T3 Forecast/Scenario) plus 5 adversarial probes.
- [ ] FR-8: Gold answers are computed offline by running the same tool functions on the same data — no manual annotation of numeric values.
- [ ] FR-9: Four deterministic evaluators (completion, tool F1, argument correctness, numerical correctness) and two LLM-judge evaluators (faithfulness, plan-goal alignment) are implemented.
- [ ] FR-10: An evaluation runner produces `results/full.csv` with per-query scores for both systems and computes ARS = mean(6 metrics).
- [ ] FR-11: Three output figures are generated: ARS by system × tier, metric heatmap, failure-type pie chart.

### Non-Functional

- [ ] NFR-1: All LLM calls use `temperature=0` and structured output (`response_format=json_schema`) to minimise variance.
- [ ] NFR-2: Every LLM call result is cached to disk by prompt hash to prevent redundant API calls and rate-limit failures.
- [ ] NFR-3: Test coverage ≥ 80% on all `src/` and `evaluation/` modules (`fail_under = 80`).
- [ ] NFR-4: All tools are pure functions (no side effects, no global state mutation).
- [ ] NFR-5: The data access layer uses DuckDB over parquet — no Python-level loops over CSV rows.
- [ ] NFR-6: All secrets are managed via pydantic-settings `BaseSettings` with `SecretStr`; no raw `os.getenv()` calls.
- [ ] NFR-7: Structured JSON logging via `python-json-logger` throughout; no plain-text log lines.
- [ ] NFR-8: LLM-judge evaluators report Cohen's κ across two independent runs on 10 samples; fallback to deterministic-only ARS if κ < 0.6.
- [ ] NFR-9: The full evaluation pipeline (60 runs) completes within the Day 2 session time budget.
- [ ] NFR-10: All runs are traced in LangFuse for observability and failure debugging.

---

## Technical Design

### Architecture: Plan → Execute → Synthesize

```
User Query (NL)
    │
    ▼
Intent & Entity Parser  (LLM, Pydantic schema)
    │
    ▼
Planner                 (LLM, temperature=0 → Plan[PlanStep])
    │
    ▼
Executor                (deterministic Python loop, NOT an LLM)
    │  for step in plan: state[step.id] = tools[step.tool](**resolve(step.args))
    ▼
Synthesizer             (LLM, scratchpad → FinalAnswer)
    │
    ▼
Trace Logger            (JSONL per run)
```

### New Modules

```
src/
├── data/
│   ├── prepare.py          ← one-time parquet build
│   └── loader.py           ← DuckDB load_prices()
├── agent/
│   ├── schemas.py          ← PlanStep, Plan, FinalAnswer, NumericClaim
│   ├── planner.py          ← Azure OpenAI → Plan
│   ├── executor.py         ← deterministic step runner
│   ├── synthesizer.py      ← Azure OpenAI → FinalAnswer
│   ├── agent.py            ← orchestrator
│   └── tools/
│       ├── prices.py       ← load_prices
│       ├── returns.py      ← compute_returns
│       ├── volatility.py   ← compute_volatility
│       ├── moving_average.py
│       └── forecast.py     ← simple_forecast (ets default, qlib optional)
└── tracing/
    └── logger.py           ← JSONL trace writer
baselines/
└── singleshot.py           ← same-schema LLM baseline
benchmark/
├── templates.txt
├── build_gold.py
└── queries.jsonl
evaluation/
├── evaluators/
│   ├── completion.py
│   ├── tool_f1.py
│   ├── args.py
│   ├── numeric.py
│   ├── faithfulness.py     ← LLM judge
│   └── alignment.py        ← LLM judge
├── judges/prompts.py
└── run_eval.py
results/full.csv
reports/
├── make_figures.py
├── figures/
└── findings.md
```

### Tool Registry (5 tools)

| Tool | Signature | Returns |
|------|-----------|---------|
| `load_prices` | `(ticker, start, end)` | DataFrame[date, close, adj_close, volume] |
| `compute_returns` | `(prices_ref, kind="log")` | pd.Series |
| `compute_volatility` | `(returns_ref, window=30, annualize=True)` | float |
| `moving_average` | `(prices_ref, window)` | pd.Series |
| `simple_forecast` | `(returns_ref, horizon=21, method="ets")` | dict{point, lower, upper} |

`*_ref` args are step IDs resolved from the shared state dict — the LLM never copies numeric data into args.

### Evaluation Metrics & ARS

| # | Metric | Type | Formula | Min Goal | Stretch |
|---|--------|------|---------|----------|---------|
| 1 | Task Completion | Deterministic | 1 if FinalAnswer parses with ≥1 claim | > 0.90 | = 1.0 |
| 2 | Tool Selection F1 | Deterministic | F1(predicted tools, gold tools) | > 0.75 | > 0.85 |
| 3 | Argument Correctness | Deterministic | mean(correct_args / total_args per step) | > 0.70 | > 0.80 |
| 4 | Numerical Correctness | Deterministic | fraction of required_facts within tolerance | > 0.65 | > 0.75 |
| 5 | Faithfulness | LLM judge | fraction of claims traceable to scratchpad | > 0.80 | > 0.90 |
| 6 | Plan-Goal Alignment | LLM judge | rubric 1–5 rescaled to [0,1] | > 0.70 | > 0.80 |

**ARS = mean(m1…m6)**

---

## Acceptance Criteria

### Data Pipeline
- [ ] AC-1: `python -m src.data.prepare` completes without error and produces `data/processed/prices.parquet/` and `data/processed/universe.csv` containing exactly 50 tickers.
- [ ] AC-2: `load_prices("AAPL", "2022-01-01", "2022-12-31")` returns a DataFrame with ≥250 rows and columns `date, close, adj_close, volume`.

### Agent
- [ ] AC-3: `agent.run("What was AAPL annualized volatility in 2022?")` returns a trace dict with non-empty `plan`, `state`, and `answer` fields.
- [ ] AC-4: The planner emits valid `Plan` JSON (Pydantic-parseable) for all 30 benchmark queries.
- [ ] AC-5: The executor never raises an unhandled exception — all step errors are captured in `trace["errors"]`.
- [ ] AC-6: The synthesizer's `FinalAnswer.claims` each have a `source_step` that exists in `trace["state"]`.
- [ ] AC-7: Every run produces a JSONL row in `data/traces/traces.jsonl`.

### Tools
- [ ] AC-8: All 5 tools pass unit tests with ≥80% branch coverage.
- [ ] AC-9: `compute_volatility` on AAPL 2022 data returns a value in the range [0.25, 0.50] (sanity check).
- [ ] AC-10: `simple_forecast` with `method="ets"` returns a dict with keys `point`, `lower`, `upper`; `lower ≤ point ≤ upper`.
- [ ] AC-11: `simple_forecast` falls back to `ets` with a warning when called with `method="qlib"` and `pyqlib` is not installed.

### Benchmark
- [ ] AC-12: `benchmark/queries.jsonl` contains exactly 30 scored queries (10 T1, 10 T2, 10 T3) plus 5 adversarial probes.
- [ ] AC-13: All `gold_answer` values in `queries.jsonl` are computed by `build_gold.py` — no manually entered floats.
- [ ] AC-14: At least 5 gold queries have been verified by hand against raw CSV data.

### Evaluation
- [ ] AC-15: `tool_selection_f1` returns 1.0 when predicted and gold plans use identical tool multisets.
- [ ] AC-16: `numeric_correctness` returns 0.0 when no claims match any `required_facts` key.
- [ ] AC-17: LLM-judge evaluators report Cohen's κ ≥ 0.6 on the 10-item sanity check; if not, rubric is tightened before the full run.
- [ ] AC-18: `python -m evaluation.run_eval` produces `results/full.csv` with 60 rows (30 queries × 2 systems) and columns `system, qid, tier, completion, tool_f1, args, numeric, faithfulness, alignment, ARS, latency_s`.

### Deliverables
- [ ] AC-19: `results/full.csv` shows ARS_agent > ARS_baseline (hypothesis confirmed or refuted and documented).
- [ ] AC-20: Three figures exist in `reports/figures/`: `ars_by_tier.png`, `metric_heatmap.png`, `failure_pie.png`.
- [ ] AC-21: `reports/findings.md` documents the top-3 agent failure modes with example query IDs.
- [ ] AC-22: `README.md` reproduces the full pipeline in ≤10 commands.

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Kaggle download blocked on PwC network | M | Pre-download to laptop; cache in `data/raw/` |
| Azure OpenAI rate limits | M | Cache every LLM call to disk by hash of prompt |
| Planner outputs invalid JSON | H | Use `response_format=json_schema`; add retry with error feedback |
| Gold answers wrong | M | Compute with same tool functions; spot-check 10% |
| LLM-judge variance | M | Two judge runs; report κ; fallback to 4-metric ARS if κ < 0.6 |
| Out of time Day 2 | H | Hard checkpoints; cut adversarial probes first, then judge metrics, then baseline |
| Numeric metric too strict | L | Per-query tolerance; default 5%, widen to 10% for volatility |

---

## Future Work (out of scope for 2-day build)

- Critic/reflection loop (ReAct-style self-correction)
- GARCH, Monte Carlo, and factor-model tools
- Robustness Δ analysis (paraphrase + date perturbation)
- LangGraph-based orchestration for parallel step execution
- Qlib integration for production-grade forecasting
