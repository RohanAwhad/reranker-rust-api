# Devlogs

## 2026-06-07 — Initial implementation

- Scaffolded project from `embedding_generation_api_rust` patterns (axum + ort + tokenizers + utoipa)
- Model: `Alibaba-NLP/gte-reranker-modernbert-base` (149M cross-encoder, ModernBERT, English, 8192 max tokens)
- ONNX export via `export_model.py` (AutoModelForSequenceClassification, opset 14, 3 inputs, [B,1] output)
  - Verified: output shape `[B, 1]`, `num_labels=1`, tokenizer produces no `token_type_ids` (ModernBERT ignores them but ONNX graph expects the input — pass zeros)
  - ONNX file: ~599 MB
- Pair tokenization via `EncodeInput::Dual(query, doc)` — Rust `tokenizers` crate v0.21 uses `InputSequence` type (requires `.into()`)
- Inference: 3 input tensors (`input_ids`, `attention_mask`, `token_type_ids`) → sigmoid → sort by score descending
- 4 endpoints: `/healthcheck`, `/v1/models`, `/v1/rerank`, `/docs` (Swagger)
- API follows OpenAI rerank format: `{ model, query, documents }` → `{ results: [{ index, relevance_score }] }`
- `RUST_LOG` env var for log level control (via `tracing-subscriber` with `env-filter` feature)
- Limits: `MAX_DOCUMENTS=500`, `MAX_SEQ_LEN=8192` (truncation with `tracing::warn!`)
- Token counting: pre-padding tokens via individual `encode()` calls per pair
- Sorting: `sort_by` with `partial_cmp` (stable in Rust ≥1.77, ties preserve input order)
- Threading: `Arc<Mutex<>>` wrapping `RerankerModel` since `Session::run` requires `&mut self`
- Healthcheck acquires lock (blocks during inference — negligible at 51ms, notable for large batches)

### Benchmarks (Apple M4, CPU)

| Documents | p50 latency |
|-----------|-------------|
| 10 docs | ~51ms |
| 50 docs | ~115ms |

Port 3000 was occupied (SSH tunnel to Langfuse). Used port 8765 for testing. Default `BIND_ADDR` stays at 3000 in config.

### Resolved PRD gaps

| GAP | Resolution |
|-----|-----------|
| GAP-01 (pair encoding) | `EncodeInput::Dual` in Rust, matching Python `tokenizer(q, doc)` |
| GAP-02 (token_type_ids) | Not produced by tokenizer but required by ONNX — pass zeros |
| GAP-03 (output shape) | Confirmed `[B, 1]` via ONNX inspection |
| GAP-05 (input names) | Confirmed 3 inputs match export names |
| GAP-10 (MAX_DOCUMENTS) | Set to 500, returns error if exceeded |
| GAP-18 (tokenization consistency) | Export script uses tokenizer pairs, Rust matches via Dual |
| GAP-17 (LOGGING_LEVEL) | Added `RUST_LOG` env var via `tracing-subscriber` with `env-filter` feature |
| GAP-24 (phase ordering) | Moved export to Phase 0 before coding inference |

### Pending
- Add benchmark report in `docs/benchmark-report.md`
