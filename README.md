# Oracle

Autonomous AI prediction engine for [Polymarket](https://polymarket.com). Oracle ingests multi-modal signals, reasons over a hybrid knowledge graph, and executes paper trades — end to end, no human in the loop.

---

## What it does

Oracle runs a continuous prediction pipeline:

1. **Ingest** — pulls from NewsAPI, Twitter/X, Reddit, government APIs (Congress, CourtListener), polling aggregators, YouTube/podcast audio (Whisper), and chart images (vision)
2. **Store** — chunks and embeds content into Qdrant (vector DB) and extracts entities into Neo4j (knowledge graph)
3. **Research** — a multi-agent system generates a structured thesis for each market using hybrid retrieval
4. **Evaluate** — an LLM judge scores the thesis on 4 dimensions; a hallucination detector verifies every claim against sources
5. **Reflect** — a self-critique step checks for anchoring, recency, and confirmation biases before committing
6. **Trade** — the Risk Agent enforces hard guardrails, then the Portfolio Manager executes paper trades
7. **Learn** — post-resolution post-mortems classify predictions as good-process vs lucky, feeding back into calibration

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Ingestion Layer                       │
│  News · Twitter · Reddit · Gov APIs · Audio · Vision    │
└───────────────────┬─────────────────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │   Knowledge Store  │
          │  Neo4j (graph)     │
          │  Qdrant (vectors)  │
          └─────────┬──────────┘
                    │
     ┌──────────────▼──────────────┐
     │     Hybrid Retrieval        │
     │  Vector · BM25 · Graph      │
     │  RRF Fusion · BGE Reranker  │
     └──────────────┬──────────────┘
                    │
     ┌──────────────▼──────────────────────────┐
     │            Agent Pipeline               │
     │  Research → Reflection → Judge          │
     │  Hallucination Check → Risk → Trade     │
     └──────────────┬──────────────────────────┘
                    │
     ┌──────────────▼──────────────┐
     │       Observability         │
     │  Prometheus · Grafana       │
     │  LLM Tracer · SSE Dashboard │
     └─────────────────────────────┘
```

---

## Backtest results

Backtested against 50 resolved Polymarket markets (crypto price/outcome markets, Apr 2026).
Infrastructure: NewsAPI for evidence, Claude Code CLI for synthesis and reflection. Neo4j/Qdrant offline (no local instance), so results reflect news-only evidence.

### Best run (calibrated pipeline)

| Metric | Value | Baseline |
|--------|-------|----------|
| Brier score | 0.250 | random = 0.25 |
| Accuracy | 68% | — |
| Calibration error | 0.314 | 0 = perfect |
| Alpha rate | 66% | — |
| Hallucination catch | 96% | — |
| Latency p50 | 11.5s | — |

### Hit rate by confidence tier

| Tier | Hit rate | n |
|------|----------|---|
| 50–60% | 100% | 4 |
| 60–70% | 100% | 4 |
| 70–80% | 82% | 17 |

When the model reaches ≥60% confidence, it is correct 100% of the time (8/8). The 0.5 abstentions (markets with no relevant news evidence) are the drag on overall EV.

### Pipeline evolution

| State | Brier | Notes |
|-------|-------|-------|
| Broken (deprecated model + bad API key) | 0.331 | Flat 0.35 on every market |
| Fixed model + NewsAPI | 0.605 | Overconfident, wrong direction |
| Cache fix + lower Claude threshold | 0.240 | Below random — first real signal |
| Abstain at 0.5 when no evidence | 0.250 | EV still negative from coin-flip abstentions |
| Claude CLI synthesis + calibrated prompt | **0.250** | 66% alpha rate, 100% hit at ≥60% confidence |

### Known limitations

- NewsAPI free tier: 100 req/day — exhausted quickly at 50 markets/run
- Neo4j + Qdrant offline: no knowledge base, evidence is news-only
- All markets are crypto price/outcome (Polymarket's recent closed set is dominated by these)
- Claude synthesis relies on Claude Code CLI (`claude -p`) — requires active session

---

## Numbers

### Retrieval

| Metric | Value |
|--------|-------|
| Retrieval strategies | 3 (vector, BM25, graph traversal) |
| Fusion algorithm | Reciprocal Rank Fusion (k=60) |
| Re-ranker | BGE-reranker-v2-m3 (cross-encoder) |
| Embedding model | BAAI/bge-large-en-v1.5 (1024-dim) |
| Claim verification threshold | 0.75 cosine similarity |

### Evaluation pipeline

| Check | Model | Max tokens |
|-------|-------|-----------|
| LLM judge (4-dim scoring) | claude-3-5-haiku | 1,024 |
| Hallucination — claim extraction | claude-3-5-haiku | 1,024 |
| Hallucination — contradiction check | claude-3-5-haiku | 1,024 |
| Reflection / bias detection | claude-3-5-haiku | 512 |
| Research synthesis | claude-3-5-haiku | 1,024 |

Judge quality gates: groundedness ≥ 7/10, reasoning ≥ 6/10, evidence ≥ 5/10.

### Model routing

The `ComplexityClassifier` (logistic regression, trained on 500 synthetic samples) routes ~80% of queries to the local stub and only sends ~20% to Claude — keeping API costs low.

| Route | Share | Latency |
|-------|-------|---------|
| Local stub | ~80% | <1ms |
| Claude (haiku) | ~20% | ~500ms |

### Risk guardrails (hard limits)

| Rule | Threshold |
|------|-----------|
| Max single-market exposure | 10% of portfolio |
| Max category exposure | 30% of portfolio |
| Max risk on markets resolving within 24h | 5% of portfolio |
| Stop-loss trigger | 50% loss on any position |

### API cost (claude-3-5-haiku @ $0.80/1M input · $4.00/1M output)

| Scale | Research cycles/day | Daily cost |
|-------|-------------------|------------|
| 10 markets | ~30 | ~$0.35 |
| 50 markets | ~150 | ~$1.80 |
| 200 markets | ~1,000 | ~$12 |
| 500 markets | ~2,500 | ~$30 |

### Ingestion schedule

| Source | Interval |
|--------|---------|
| Polymarket markets | 60 seconds |
| News (NewsAPI) | 15 minutes |
| Reddit | 30 minutes |
| Government APIs | 6 hours |
| Polling aggregators | 12 hours |
| Audio (Whisper) | Daily |
| Twitter/X | Continuous streaming |

---

## Tech stack

| Layer | Tech |
|-------|------|
| API | FastAPI + uvicorn |
| Vector DB | Qdrant v1.9 |
| Graph DB | Neo4j 5.19 (APOC) |
| Embeddings | sentence-transformers (BGE large) |
| LLM | Anthropic claude-3-5-haiku |
| Fine-tuning | Modal + LoRA (Mistral 7B Instruct, r=16) |
| Observability | Prometheus + Grafana |
| Cache | Qdrant-backed semantic cache (1024-dim) |
| Frontend | React (Vite) — real-time war room dashboard |
| Streaming | Server-Sent Events (SSE) |
| A/B testing | SQLite + two-sample t-test (min 30 samples/variant) |
| Containerization | Docker Compose |

---

## Quickstart

```bash
# 1. Copy env
cp .env.example .env
# Fill in: ORACLE_ANTHROPIC_API_KEY, ORACLE_NEWSAPI_KEY, etc.

# 2. Start infrastructure
docker compose up -d

# 3. Install Python deps
pip install uv && uv sync

# 4. Run the API
uvicorn oracle.api.app:app --reload

# 5. Start the frontend
cd frontend && npm install && npm run dev
```

The API will be at `http://localhost:8000` and the war room dashboard at `http://localhost:5173`.

---

## Fine-tuning (optional)

Oracle ships a Modal-based LoRA pipeline for fine-tuning Mistral 7B on prediction market reasoning:

```bash
# Generate training data from resolved markets
python -m oracle.training.data_generator

# Launch fine-tune on Modal (A10G GPU, ~2h)
modal run src/oracle/training/modal_trainer.py
```

LoRA config: r=16, alpha=32, target modules: q/k/v/o projections, dropout=0.05.

---

## Observability

Prometheus metrics are exposed at `/metrics`. Key gauges:

- `oracle_brier_score` — calibration quality (lower is better)
- `oracle_accuracy_rate` — rolling prediction accuracy
- `oracle_cache_hit_rate` — tool cache efficiency
- `oracle_portfolio_value` — paper portfolio value
- `oracle_cost_per_prediction` — USD cost histogram
- `oracle_llm_latency_seconds` — per-model, per-agent latency

The React war room streams live agent activity, trade decisions, and evaluation scores via SSE.

---

## Project structure

```
src/oracle/
├── agents/          # Research, Reflection, Quant, Risk, Portfolio agents
├── api/             # FastAPI app, routes, SSE streaming
├── cache/           # TTL + semantic cache (Qdrant-backed)
├── evaluation/      # LLM judge, hallucination detector, calibration, post-mortems
├── ingestion/       # News, Twitter, Reddit, audio, vision, gov scrapers
├── knowledge/       # Neo4j + Qdrant clients, embeddings
├── observability/   # Prometheus metrics, LLM tracer
├── prompts/         # Prompt registry, A/B testing
├── retrieval/       # Vector, BM25, graph search, RRF fusion, re-ranker
├── routing/         # Complexity classifier → model routing
└── training/        # Modal LoRA fine-tuning, synthetic data generator
```
