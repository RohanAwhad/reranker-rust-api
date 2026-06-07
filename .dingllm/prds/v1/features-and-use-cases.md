# features-and-use-cases.md

## Primary Use Cases

1. **RAG pipeline reranking** — The canonical use case. First-pass vector/BM25 retrieval returns top-K candidates (often 20-100). Cross-encoder scores each candidate against the query. The top-N most relevant are fed into the LLM context window, improving answer quality by filtering noise.

2. **Search result reordering** — Hybrid search merges results from multiple indexes (vector, keyword, semantic). The cross-encoder provides a unified relevance score to produce a single ranked list.

3. **Question answering passage selection** — Given a question, score candidate passages from a knowledge base. The highest-scoring passage contains the answer.

4. **Multi-query batch reranking** — Same document set scored against multiple queries in separate calls. Enables A/B testing different query formulations or batching user questions through a pipeline.

5. **Document filtering by threshold** — Keep only documents with `relevance_score > 0.7`. Use the API as a binary classifier: relevant vs. not-relevant.

6. **Code search reranking** — Score code snippets against natural language queries. The model scores 79.99 on CoIR (Code Retrieval), so it's viable for IDE/developer tooling.

---

## User Stories

| # | As a... | I want to... | So that... |
|---|---------|--------------|------------|
| US1 | RAG developer | Send `{ query, documents[] }` and get back `{ index, score }` pairs | I can feed only the top-N most relevant chunks into my LLM prompt |
| US2 | Search engineer | Reorder my first-pass retrieval results by relevance | Users see the best results first, not noise |
| US3 | Platform developer | Drop an OpenAI-compatible rerank endpoint into my pipeline | I don't need to write model-serving infrastructure |
| US4 | DevOps | Configure model path + name via environment variables | I can deploy the same binary with different models in different environments |
| US5 | API consumer | Get scores in `[0, 1]` via sigmoid | Scores are calibrated and comparable across different queries |
| US6 | Backend developer | Get a model list and healthcheck | My orchestration layer can detect when the service is ready |
| US7 | Data scientist | Get scores for 50+ documents in a single call | I avoid N+1 network overhead when evaluating many documents |
| US8 | RAG pipeline | Detect when a query+doc exceeds the token limit | Documents get truncated gracefully without crashing the request |
| US9 | QA engineer | See the API schema via Swagger UI at `/docs` | I can test endpoints without reading source code |

---

## v1 Features

| Feature | Priority | Delivered In |
|---------|----------|-------------|
| `POST /v1/rerank` — score & sort documents | P0 | Phase 3 |
| Sigmoid calibration (scores in `[0, 1]`) | P0 | Phase 2 |
| Multi-document batching (single ONNX forward pass) | P0 | Phase 2 |
| Results sorted by score descending | P0 | Phase 2 |
| `GET /healthcheck` — liveness + loaded model | P0 | Phase 3 |
| `GET /v1/models` — list available models | P0 | Phase 3 |
| `GET /docs` — Swagger UI | P1 | Phase 3 |
| OpenAPI schema generation via `utoipa` | P1 | Phase 3 |
| Token usage in response (`total_tokens`) | P1 | Phase 3 |
| `query + doc > 8192` truncation with log warning | P1 | Phase 2 |
| ONNX model export script (`export_model.py`) | P0 | Phase 4 |
| CI + pre-commit hooks (fmt, clippy, check) | P0 | Phase 1 |
| README with usage examples | P1 | Phase 5 |

---

## v2 Features (Deferred)

| Feature | Rationale |
|---------|-----------|
| `return_documents: bool` — include doc text in response | Useful for debugging relevance, but adds response size |
| `top_n: u32` — return only top N results | Caller can truncate client-side; server-side saves serialization |
| `max_chunks_per_doc: u32` — split long docs into chunks | Model supports 8192 tokens; long docs could be chunked and scored independently |
| Streaming response (SSE) | Only matters for 1000+ document batches |
| `encoding_format: "base64"` support | Only if scores need embedding-style transport |
| Multi-model support (load >1 model, select by name) | Complexity jump; single model covers 90% of use cases |
| Authentication (API key header) | Not needed for internal/VPC deployments |
| Metrics endpoint (`/metrics` in Prometheus format) | Useful once deployed at scale |
| GPU execution provider support | CPU is fine for 149M model; GPU needed only at very high throughput |
