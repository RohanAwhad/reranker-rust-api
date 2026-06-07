# implementation-plan.md

## Phase 0: ONNX Model Export + Inspection **[COMPLETED]**

**Resolves:** GAP-01 through GAP-05, GAP-18

- Export `gte-reranker-modernbert-base` to ONNX via `export_model.py`
- Inspect exported model to confirm:
  - Output shape: `[B, 1]` (confirmed — single logit per pair)
  - Input names: `input_ids`, `attention_mask`, `token_type_ids` (confirmed)
  - `num_labels=1` — single output logit, not binary classification
  - ModernBERT tokenizer produces no `token_type_ids` but ONNX graph expects the input (pass all zeros)
- Model file size: ~599 MB (FP32)

## Phase 1: Scaffold + Model Loading **[COMPLETED]**

**Features delivered:** CI, pre-commit, model loading

- Copy `Cargo.toml` from embedding API (renamed to `reranker_api`, same deps + `env-filter` for RUST_LOG)
- Copy CI workflow, pre-commit hook, `.gitignore`
- `src/reranker.rs`: `RerankerModel::load()` — ONNX session + tokenizer, graph opt level 3
- `src/main.rs` skeleton: `AppState`, healthcheck, list_models, env var config
- Verified: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo check` clean

## Phase 2: Inference Core **[COMPLETED]**

**Features delivered:** Sigmoid scoring, multi-doc batching, sorting, truncation, token counting

- `RerankerModel::rerank(&mut self, query, documents) -> Vec<(usize, f32)>`:
  1. Build `EncodeInput::Dual(query, doc)` for each document (matching Python `tokenizer(q, d)`)
  2. Tokenize batch via `encode_batch()`
  3. Build padded `Array2<i64>` arrays (`input_ids`, `attention_mask`, `token_type_ids` — all zeros for type ids)
  4. Run ONNX inference via `session.run()`
  5. Extract `[B, 1]` logits, apply `sigmoid()`
  6. Stable sort by score descending
- `RerankerModel::count_tokens()` — pre-padding token count per pair
- MAX_DOCUMENTS=500 (returns error if exceeded)
- MAX_SEQ_LEN=8192 (truncates with `tracing::warn!`)

## Phase 3: HTTP Layer **[COMPLETED]**

**Features delivered:** Full API surface

- Routes: `POST /v1/rerank`, `GET /healthcheck`, `GET /v1/models`, `GET /docs` (Swagger)
- Request/response types with `utoipa::ToSchema` derive
- Model name validation (exact match) → 400 on mismatch
- Inference errors → 500 with `server_error`
- OpenAPI generation via `ApiDoc` struct + Swagger UI at `/docs`
- `RUST_LOG` env var for tracing log level (via `tracing-subscriber` with `env-filter`)

## Phase 4: Export Script **[COMPLETED]**

- `export_model.py`: downloads model, exports to ONNX (opset 14), saves tokenizer
- Includes shape inspection, input/output name verification

## Phase 5: Polish **[COMPLETED]**

- README with quickstart, API docs, config, benchmarks, model details
- devlogs.md with implementation notes, resolved GAPs, benchmarks
- Benchmark: Apple M4 — 10 docs ~51ms, 50 docs ~115ms

---

## Resolved PRD Gaps

| GAP | Resolution | Where |
|-----|-----------|-------|
| GAP-01/18 | Use `EncodeInput::Dual` in Rust (matching Python pair encoding) | `reranker.rs:61` |
| GAP-02 | `token_type_ids` absent from tokenizer; ONNX expects it → pass zeros | `reranker.rs:80-85` |
| GAP-03 | Output `[B, 1]` confirmed via ONNX inspection → iterate `data[i]` | `reranker.rs:107-110` |
| GAP-04 | Dynamic axes correct per ONNX export | `export_model.py:65-70` |
| GAP-05 | Input names match: `input_ids`, `attention_mask`, `token_type_ids` | `reranker.rs:88-96` |
| GAP-10 | `MAX_DOCUMENTS=500` with error on exceed | `reranker.rs:11,48-52` |
| GAP-17 | `RUST_LOG` env var via `tracing-subscriber` with `env-filter` feature | `main.rs:200-204` |
| GAP-24 | Export moved to Phase 0 — model inspected before inference code written | This doc |

---

## Feature-Phase Matrix

| Feature | Phase 0 | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|---------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| ONNX export + inspection | x | | | | | |
| CI + pre-commit | | x | | | | |
| Model loading (ONNX + tokenizer) | | x | | | | |
| Sigmoid scoring | | | x | | | |
| Multi-document batching | | | x | | | |
| Sort by score descending | | | x | | | |
| `>8192` token truncation | | | x | | | |
| Token counting (pre-padding) | | | x | | | |
| MAX_DOCUMENTS limit | | | x | | | |
| `POST /v1/rerank` | | | | x | | |
| `GET /healthcheck` | | | | x | | |
| `GET /v1/models` | | | | x | | |
| `GET /docs` (Swagger) | | | | x | | |
| OpenAPI utoipa schemas | | | | x | | |
| Error responses (400, 500) | | | | x | | |
| RUST_LOG log level | | | | x | | |
| README + benchmarks | | | | | | x |
