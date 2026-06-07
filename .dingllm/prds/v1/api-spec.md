# api-spec.md

## Endpoints

### `POST /v1/rerank`

Score a set of documents against a query. Returns results sorted by `relevance_score` descending.

**Request:**
```json
{
  "model": "gte-reranker-modernbert-base",
  "query": "What is the capital of France?",
  "documents": [
    "Paris is the capital of France.",
    "France is a country in Europe."
  ]
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | `string` | Yes | Must match the loaded model name |
| `query` | `string` | Yes | The search query (1..N chars) |
| `documents` | `string[]` | Yes | Array of document texts to score (0..N) |

**Response (200 OK):**
```json
{
  "object": "list",
  "model": "gte-reranker-modernbert-base",
  "results": [
    { "index": 0, "relevance_score": 0.98 },
    { "index": 1, "relevance_score": 0.42 }
  ],
  "usage": {
    "total_tokens": 24
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `object` | `"list"` | Always `"list"` |
| `model` | `string` | Model that produced the results |
| `results` | `array` | Sorted by `relevance_score` descending |
| `results[].index` | `u32` | Original position in the input `documents` array |
| `results[].relevance_score` | `f32` | Sigmoid-calibrated score in `[0, 1]` |
| `usage.total_tokens` | `u32` | Total tokens across all query+doc pairs |

**Error Responses:**

`400 Bad Request` — wrong model name:
```json
{
  "error": {
    "message": "Model 'foo' not found. Available model: 'gte-reranker-modernbert-base'",
    "type": "invalid_request_error"
  }
}
```

`500 Internal Server Error` — inference failure:
```json
{
  "error": {
    "message": "Reranking failed: <error details>",
    "type": "server_error"
  }
}
```

---

### `GET /healthcheck`

**Response (200 OK):**
```json
{
  "status": "ok",
  "model": "gte-reranker-modernbert-base"
}
```

---

### `GET /v1/models`

**Response (200 OK):**
```json
{
  "object": "list",
  "data": [
    {
      "id": "gte-reranker-modernbert-base",
      "object": "model",
      "owned_by": "local"
    }
  ]
}
```

---

### `GET /docs`

Swagger UI served at `/docs` with OpenAPI JSON at `/api-docs/openapi.json`.

---

## Happy Path Flow

```
Client                          Reranker API                     ONNX Runtime
  │                                  │                                │
  │  POST /v1/rerank                 │                                │
  │  { query, documents: [...] }     │                                │
  │ ───────────────────────────────> │                                │
  │                                  │  Validate model name           │
  │                                  │  Tokenize query+doc pairs × N  │
  │                                  │ ──────────────────────────────>│
  │                                  │  Run ONNX inference (batch)    │
  │                                  │ <──────────────────────────────│
  │                                  │  Extract logits → sigmoid      │
  │                                  │  Sort by score descending      │
  │  { results: [{ index, score }] } │                                │
  │ <─────────────────────────────── │                                │
  │                                  │                                │
```

---

## Limits & Edge Cases

| Scenario | Behavior |
|----------|----------|
| Empty documents array | 200 with `results: [], total_tokens: 0` |
| Empty query string | Tokenized as-is (may produce nonsense results; no special handling) |
| Single document | 200 with single result, score computed normally |
| query + doc > 8,192 tokens | Truncate doc to fit limit; log warning via `tracing::warn!` |
| Unsupported model name in request | 400 with `invalid_request_error` |
| ONNX inference fails | 500 with `server_error` |
| Single document per batch | Minimal overhead (ONNX still runs batch=1) |
| 100+ documents | Accepted, single batch forward pass; all results returned sorted |
| Unicode / emoji in query or doc | Tokenizer handles — no special processing |
| All documents score identically | Natural behavior; tie-breaking by input index (stable sort) |
| Very short query + very short doc | Handled — padding fills to model's expected sequence length |
