# Reranker API

OpenAI-compatible rerank API in Rust using ONNX Runtime. Serves `gte-reranker-modernbert-base` (149M cross-encoder) via axum, scoring query-document pairs for result reordering after first-pass retrieval.

## Quickstart

### 1. Export model to ONNX

```bash
pip install torch transformers onnx
python3 export_model.py
```

Downloads `Alibaba-NLP/gte-reranker-modernbert-base` (149M params, English, 8192 max tokens) and exports to `models/gte-reranker-modernbert-base/`.

### 2. Run the server

```bash
cargo run --release
```

Server starts on `http://0.0.0.0:3000`.

### 3. Rerank documents

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

Response:

```json
{
  "object": "list",
  "model": "gte-reranker-modernbert-base",
  "results": [
    { "index": 0, "relevance_score": 0.898 },
    { "index": 1, "relevance_score": 0.774 }
  ],
  "usage": { "total_tokens": 24 }
}
```

Results sorted by `relevance_score` descending. Scores are sigmoid-calibrated in `(0, 1)`.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/rerank` | Score & sort documents by relevance to query |
| GET | `/healthcheck` | Service health + loaded model name |
| GET | `/v1/models` | List available models |
| GET | `/docs` | OpenAPI Swagger UI |

### POST `/v1/rerank`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | `string` | Yes | Must match the loaded model name |
| `query` | `string` | Yes | The search query |
| `documents` | `string[]` | Yes | Array of document texts to score (max 500) |

Errors: `400` for wrong model name, `500` for inference failure.

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `MODEL_PATH` | `models/gte-reranker-modernbert-base` | Directory containing `model.onnx` + `tokenizer.json` |
| `MODEL_NAME` | `gte-reranker-modernbert-base` | Model name returned in API responses |
| `BIND_ADDR` | `0.0.0.0:3000` | Address to bind the server |
| `RUST_LOG` | `info` | Log level (trace, debug, info, warn, error) |

## Architecture

```
POST /v1/rerank
  → axum handler (JSON deserialization, model name validation)
  → tokenizer: encode [query + SEP + doc] × N as EncodeInput::Dual pairs
  → session.run() (ONNX Runtime, graph optimization level 3)
  → extract [B, 1] logits → sigmoid → sort by score descending
  → JSON response { results: [{ index, relevance_score }] }
```

- **Model**: `Alibaba-NLP/gte-reranker-modernbert-base` (ModernBERT cross-encoder, 149M params)
- **Backend**: ONNX Runtime (`ort` v2.0.0-rc.12) on CPU
- **Tokenizer**: HuggingFace `tokenizers` crate v0.21 (pair encoding via `EncodeInput::Dual`)
- **Server**: axum 0.8 + tokio
- **Docs**: utoipa OpenAPI + Swagger UI
- **Max sequence**: 8,192 tokens. Longer sequences are truncated with a log warning.
- **Max documents**: 500 per request.

## Benchmarks

Apple M4, `cargo run --release`, CPU inference:

| Documents | Latency (p50) |
|-----------|---------------|
| 10 docs | ~51ms |
| 50 docs | ~115ms |

Throughput: ~1 request blocks the Mutex at a time (single ONNX session). For concurrent requests, use multiple instances or a load balancer.

## Build

```bash
cargo build --release
```

Requires Rust 1.88+.

### Tests

```bash
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

GitHub Actions CI runs fmt + clippy + test on push to master.

## Model Details

| Field | Value |
|-------|-------|
| Model | `Alibaba-NLP/gte-reranker-modernbert-base` |
| Architecture | Cross-encoder (ModernBERT-base, 12-layer) |
| Parameters | 149M |
| Max tokens | 8,192 (query + doc concatenated) |
| ONNX size | ~599 MB (FP32) |
| Language | English |
| License | Apache-2.0 |
| BEIR nDCG@10 | ~56.73 |
