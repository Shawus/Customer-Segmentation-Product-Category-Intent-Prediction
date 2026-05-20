# AI Sales Compass — Customer Segmentation & Product Recommendation

A multi-agent AI system that performs customer segmentation via spectral clustering and product recommendation via LightGBM learning-to-rank, orchestrated through LangGraph.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│           LangGraph Orchestrator            │
│                                             │
│  Intent Agent → Extraction Agent            │
│       → Clustering Agent                    │
│       → Classification Agent                │
│       → Summary Agent                       │
└─────────────────────────────────────────────┘
    │                           │
    ▼                           ▼
┌──────────────┐      ┌──────────────────┐
│  Clustering  │      │  Classification  │
│  (Spectral)  │      │  (LightGBM Rank) │
└──────────────┘      └──────────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `agents/` | LangGraph agent nodes (intent, extraction, clustering, classification, summary) |
| `clustering/` | Spectral clustering with augmented similarity (RFM + temporal + product features) |
| `classification/` | LightGBM LambdaRank product recommender with Word2Vec embeddings |
| `utils/` | LLM client, async task tracking |

## Key Technical Highlights

- **Spectral Clustering**: Custom augmented similarity graph combining KNN on numerical features with categorical edge weights
- **LightGBM LambdaRank**: Learning-to-rank for product recommendations with group-aware training
- **Word2Vec Embeddings**: Skip-gram model trained on product category hierarchies (32-dim)
- **Multi-Agent Orchestration**: LangGraph StateGraph with conditional routing based on extracted intent

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # Fill in your Azure OpenAI credentials
```

## Environment Variables

See `.env.example` for required configuration.

## Running

```bash
# Development
uvicorn app:app --reload --port 8000

# Production
gunicorn -c gunicorn_conf.py app:app
```

## API Endpoints

- `POST /api/run_agent` — Run the multi-agent pipeline for a customer query
- `GET /api/status_check/{task_id}` — Check async task status
- `POST /clustering/train` — Train clustering model
- `POST /clustering/inference` — Run clustering inference
- `POST /classification/train` — Train ranking model
- `GET /classification/health` — Health check
