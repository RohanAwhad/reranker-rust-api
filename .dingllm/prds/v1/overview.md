# overview.md

OpenAI-compatible rerank API in Rust using ONNX Runtime. Serves `gte-reranker-modernbert-base` (149M cross-encoder) via axum, scoring query-document pairs for result reordering after first-pass retrieval.

**Inherits architecture from** `embedding_generation_api_rust` — same dependency stack, server skeleton, ONNX loading, and project conventions.

---

## Model

| Field | Value |
|-------|-------|
| Model | `Alibaba-NLP/gte-reranker-modernbert-base` |
| Architecture | Cross-encoder (ModernBERT-base, 12-layer) |
| Parameters | 149M |
| Tokenizer | ModernBERT tokenizer (not XLM-RoBERTa) |
| Max sequence | 8,192 tokens (query + doc concatenated) |
| Language | English |
| License | Apache-2.0 |
| ONNX size (FP32) | ~570 MB |
| Output | Single logit per (query, doc) pair → sigmoid → `[0, 1]` score |

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/rerank` | Score & sort documents by relevance to query |
| GET | `/healthcheck` | `{ status: "ok", model: "..." }` |
| GET | `/v1/models` | List available models |
| GET | `/docs` | Swagger UI (OpenAPI) |

---

## Quick Example

```bash
curl http://localhost:3000/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gte-reranker-modernbert-base",
    "query": "What is the capital of France?",
    "documents": [
      "Paris is the capital of France.",
      "France is a country in Europe."
    ]
  }'
```

→ `{ results: [{ index: 0, relevance_score: 0.98 }, { index: 1, relevance_score: 0.42 }] }`

---

## Configuration (env vars)

| Var | Default | Description |
|-----|---------|-------------|
| `MODEL_PATH` | `models/gte-reranker-modernbert-base` | Dir with `model.onnx` + `tokenizer.json` |
| `MODEL_NAME` | `gte-reranker-modernbert-base` | Model name in API responses |
| `BIND_ADDR` | `0.0.0.0:3000` | Server bind address |

---

## More Details

- [Features & Use Cases](./features-and-use-cases.md) — user stories, v1/v2 feature tables
- [API Spec](./api-spec.md) — full schemas, error codes, limits, edge cases
- [Architecture](./architecture.md) — file tree, data flow, ONNX export, deps
- [Implementation Plan](./implementation-plan.md) — phases, feature-phase mapping, test strategy
