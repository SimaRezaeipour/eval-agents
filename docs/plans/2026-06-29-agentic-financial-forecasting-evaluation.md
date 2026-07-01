# Agentic Financial Planning & Forecasting Evaluation — Implementation Plan

**Goal:** Build and evaluate a Plan-and-Execute agentic LLM system that performs multi-step
financial planning and forecasting over historical NASDAQ daily price data. The focus is
**agent reasoning and planning quality** (not predictive accuracy). Compare against a
single-shot LLM baseline using 6 metrics aggregated into an Agent Reliability Score (ARS).

**Tech Stack:**
- Python 3.11+ via `uv`
- `pandas`, `duckdb`, `pyarrow` — data layer
- `pydantic` — schemas and structured I/O
- `openai` (Azure OpenAI) — planner, synthesizer, LLM judge
- `statsmodels` — ETS forecasting
- `matplotlib`, `scikit-learn` — evaluation plots and metrics
- `pytest` + `pytest-cov` — testing (≥80% coverage)
- `langfuse` — tracing and observability
- `ruff`, `mypy` — linting and type checking

**Dataset:** Kaggle — Stock Market Dataset (NASDAQ OHLCV daily, ~8 000 tickers). Top 50
tickers by 2023 average volume used as the evaluation universe.

**Hypothesis:** A Plan-and-Execute agent with explicit tool calls outperforms a single-shot
LLM on tool-selection accuracy, numerical correctness, and faithfulness — even when both
have access to identical raw context.

**Success criteria (2-day MVP):**
- 30 gold-annotated benchmark queries (10 per tier: Descriptive / Comparative / Forecast)
- 2 systems compared: Plan-and-Execute agent vs. single-shot baseline
- 6 metrics computed, aggregated into ARS
- Error analysis + 3 plots + README

**Metric targets (from team tracker):**

| Metric | Minimum Goal | Stretch Goal |
|--------|-------------|--------------|
| Task Completion | > 0.90 | = 1.0 |
| Tool Selection F1 | > 0.75 | > 0.85 |
| Argument Correctness | > 0.70 | > 0.80 |
| Numerical Correctness | > 0.65 | > 0.75 |
| Faithfulness | > 0.80 | > 0.90 |
| Plan-Goal Alignment | > 0.70 | > 0.80 |

**Team (PwC):** Violet Khazali (Lead), Kaiqi Cheng, Kanishk Patoliya, Sima Rezaeipour, Uday Sharma, Ujjwal Khanna
**Facilitators:** Tarun Joseph (Vector Project), Sara Ketabi / Bahar Sateli (Vector Technical), Jaswinder Narain (Company PoC)

---

### Task 1: Project Setup & Data Pipeline

**Files:**
- Create `pyproject.toml` with all dependencies
- Create `.env.example` with `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`
- Create `src/data/prepare.py`
- Create `src/data/loader.py`

Steps:
- Initialise repo with `uv init`; add all dependencies
- `prepare.py`: rank tickers by 2023 avg volume → top 50; concatenate into `data/processed/prices.parquet` partitioned by ticker; write `data/processed/universe.csv`
- `loader.py`: expose `load_prices(ticker, start, end) -> pd.DataFrame` via DuckDB view over parquet
- Verify parquet is readable; smoke-test `load_prices("AAPL", "2022-01-01", "2022-12-31")`

---

### Task 2: Pydantic Schemas

**Files:**
- Create `src/agent/schemas.py`

Steps:
- Define `ToolName` Literal with 5 tool names
- Define `PlanStep(BaseModel)`: id, tool, args, depends_on, rationale
- Define `Plan(BaseModel)`: steps list
- Define `NumericClaim(BaseModel)`: statement, value, source_step
- Define `FinalAnswer(BaseModel)`: summary, claims, caveats

---

### Task 3: Tool Registry

**Files:**
- Create `src/agent/tools/__init__.py` — `TOOLS` dict
- Create `src/agent/tools/prices.py` — `load_prices`
- Create `src/agent/tools/returns.py` — `compute_returns`
- Create `src/agent/tools/volatility.py` — `compute_volatility`
- Create `src/agent/tools/moving_average.py` — `moving_average`
- Create `src/agent/tools/forecast.py` — `simple_forecast`

Steps:
- All tools are pure functions (no side effects), strictly typed
- `load_prices`: wraps loader.py; returns DataFrame
- `compute_returns`: accepts `prices_ref` (step-id key into state), `kind` ∈ {simple, log}; returns pd.Series
- `compute_volatility`: accepts `returns_ref`, `window=30`, `annualize=True`; returns float (rolling std * √252 if annualized)
- `moving_average`: accepts `prices_ref`, `window`; returns pd.Series
- `simple_forecast`: accepts `returns_ref`, `horizon=21`, `method` ∈ {mean, ets, linear, qlib}; returns dict {point, lower, upper}; defaults to `ets` via statsmodels; falls back to `ets` with warning if `qlib` not installed
- `TOOLS` dict maps ToolName → callable; executor imports this

---

### Task 4: Planner, Executor, Synthesizer

**Files:**
- Create `src/agent/planner.py`
- Create `src/agent/executor.py`
- Create `src/agent/synthesizer.py`
- Create `src/agent/agent.py`
- Create `src/tracing/logger.py`

Steps:
- `planner.py`: call Azure OpenAI with structured output (`response_format=json_schema`); system prompt enforces `Plan` schema; `temperature=0`; returns `Plan`
- `executor.py`: deterministic loop over plan steps; resolve `{"ref": "step_id"}` args from state dict; catch per-step exceptions, store `None` + error; return `{state, errors}`
- `synthesizer.py`: call Azure OpenAI; system prompt passes `<scratchpad>` (serialised state); returns `FinalAnswer`; `temperature=0`
- `agent.py`: orchestrate plan → execute → synthesise; call `log_run`; return trace dict
- `logger.py`: append JSONL row per run to `data/traces/traces.jsonl`; fields: query, plan, state (repr-truncated to 500 chars per key), errors, answer, latency_s, tokens

---

### Task 5: Single-Shot Baseline

**Files:**
- Create `baselines/singleshot.py`

Steps:
- Dump last 200 rows of each mentioned ticker as inline CSV snippet
- Single LLM call with full prompt; `temperature=0`
- Return trace dict with same shape as agent (plan=[], state={}, answer={...}) so evaluators work on both systems

---

### Task 6: Benchmark Construction

**Files:**
- Create `benchmark/templates.txt` — 10 query templates
- Create `benchmark/build_gold.py` — script that runs tool chain and fills `gold_answer`
- Create `benchmark/queries.jsonl` — 30 gold-annotated queries + 5 adversarial probes

Steps:
- Draft 10 templates, fill with ticker × date-range × metric permutations → 30 queries
- Tiers: 10 × T1 (1–2 tools), 10 × T2 (3–4 tools, multi-ticker), 10 × T3 (4–6 tools, includes `simple_forecast`)
- 5 adversarial probes: unknown ticker, date outside coverage, ambiguous query, conflicting dates, paraphrase of T2
- `build_gold.py`: run tool sequences offline; store `gold_answer`, `gold_plan`, `tolerance`, `required_facts` per query
- Spot-check 5 queries by hand; widen tolerance to 10% for volatility queries

---

### Task 7: Deterministic Evaluators

**Files:**
- Create `evaluation/evaluators/completion.py`
- Create `evaluation/evaluators/tool_f1.py`
- Create `evaluation/evaluators/args.py`
- Create `evaluation/evaluators/numeric.py`

Steps:
- `completion`: 1 if FinalAnswer parses and has ≥1 claim, else 0
- `tool_f1`: Counter-based F1 between predicted vs. gold multiset of tool names
- `args`: mean over steps of (correct args / total args), normalised 0–1
- `numeric`: fraction of `required_facts` where |pred − gold| / |gold| ≤ tolerance

---

### Task 8: LLM-as-Judge Evaluators

**Files:**
- Create `evaluation/evaluators/faithfulness.py`
- Create `evaluation/evaluators/alignment.py`
- Create `evaluation/judges/prompts.py`

Steps:
- `faithfulness`: for each claim check if value appears (within 1%) in cited step's observation; returns fraction supported
- `alignment`: rubric 1–5 (1=irrelevant, 5=fully addresses with no waste); rescaled to [0,1]
- Both judges use `temperature=0`, structured output
- Run each judge **twice** on 10 random items; compute Cohen's κ; if κ < 0.6, tighten rubric

---

### Task 9: Evaluation Runner & Results

**Files:**
- Create `evaluation/run_eval.py`
- Create `results/` directory

Steps:
- Runner loops 30 queries × 2 systems = 60 runs
- Compute all 6 metrics per row; add `ARS = mean(6 metrics)`
- Output `results/full.csv`
- Print grouped summary by system

---

### Task 10: Reporting & Deliverables

**Files:**
- Create `reports/make_figures.py`
- Create `reports/figures/` directory
- Create `reports/findings.md`
- Create `README.md`

Steps:
- Figure 1: bar chart — ARS by system × tier
- Figure 2: heatmap — metric × tier for the agent
- Figure 3: failure-type pie chart from error bucket analysis
- `findings.md` (≤2 pages): headline ARS numbers, top-3 failure modes, judge vs. deterministic disagreements
- `README.md`: reproduce in <10 commands; reference quickstart from PROJECT_PLAN.md

---

### Optional: Robustness Analysis

**Files:** inline in `evaluation/run_eval.py` or `evaluation/robustness.py`

Steps:
- Take 5 T2/T3 queries; paraphrase + shift dates ±60 days
- Re-run agent; report **Robustness Δ** = ARS_perturbed − ARS_original

---

## Execution Timeline

| Day | Hours | Tasks |
|-----|-------|-------|
| Day 1 | 0–1 | Task 1 (setup + data pipeline) |
| Day 1 | 1–2 | Task 1 verification |
| Day 1 | 2–4 | Tasks 2 + 3 (schemas + tools) |
| Day 1 | 4–6 | Task 4 (agent wiring) |
| Day 1 | 6–7 | Task 5 (baseline) |
| Day 1 | 7–9 | Task 6 (benchmark) |
| Day 1 | 9–10 | Smoke test 5 queries; eyeball traces |
| Day 2 | 0–2 | Tasks 7 (deterministic evaluators + unit tests) |
| Day 2 | 2–4 | Task 8 (judge evaluators + κ check) |
| Day 2 | 4–5 | Task 9 (eval runner → CSV) |
| Day 2 | 5–7 | Full 60-run eval |
| Day 2 | 7–9 | Task 10 (error analysis + 3 plots) |
| Day 2 | 9–10 | README, conclusions, future work |

**Hard checkpoints:**
- Day 1 end of hour 6 → agent runs one query end-to-end
- Day 1 end of hour 10 → 10 of 30 gold queries ready
- Day 2 end of hour 5 → eval runner produces a CSV row for one system
- Day 2 end of hour 7 → both systems fully scored
