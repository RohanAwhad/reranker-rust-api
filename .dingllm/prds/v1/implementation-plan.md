# implementation-plan.md

## Phase 1: Scaffold + Model Loading

**Features delivered:** CI, pre-commit, model loading

- Copy `Cargo.toml` from embedding API (rename crate to `reranker_api`, same deps)
- Copy CI workflow, pre-commit hook, `.gitignore`
- Implement `src/reranker.rs`:
  - `RerankerModel` struct: `session: Session`, `tokenizer: Tokenizer`, `name: String`
  - `RerankerModel::load(model_dir: &Path, name: &str) -> Result<Self>`
    - Load `model.onnx` via `Session::builder()` → `GraphOptimizationLevel::Level3` → `commit_from_file()`
    - Load `tokenizer.json` via `Tokenizer::from_file()`
    - Disable automatic padding/truncation (batching handled manually)
- Implement `src/main.rs` skeleton:
  - `AppState { model: Arc<Mutex<RerankerModel>> }`
  - Load model from env vars at startup
  - Placeholder routes (return 501)
  - `#[tokio::main]` + `tracing_subscriber::fmt::init()`
- Verify: `cargo build`, `cargo fmt --check`, `cargo clippy -- -D warnings`

**Test strategy:** No unit tests in this phase — model file absent, no inference contract yet. CI must pass (fmt + clippy + check).

---

## Phase 2: Inference Core

**Features delivered:** Sigmoid scoring, multi-doc batching, sorting, truncation

- Implement `RerankerModel::rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<(usize, f32)>>`:
  1. Build pair texts: for each doc, format `query [SEP] doc` (tokenizer handles CLS/SEP)
  2. Tokenize batch via `encode_batch()` — tokenizer auto-inserts `[CLS]` and `[SEP]` when given two texts
  3. Build padded `Array2<i64>` for `input_ids`, `attention_mask`, `token_type_ids`
  4. Run `session.run(ort::inputs![...])?`
  5. Extract `[B, 1]` logit tensor, apply `sigmoid()` per entry
  6. Sort `(index, score)` pairs by score descending (stable sort for tie-breaking)
- Implement `sigmoid(x: f32) -> f32` helper
- Truncation: if any pair exceeds model max tokens, truncate doc to fit and `tracing::warn!`
- Edge cases: empty documents → empty vec; single document → single result
- Verify: `cargo build` + no regressions from Phase 1

**Test strategy:** No unit tests — requires ONNX model file to run inference. Manual smoke test via `curl` after Phase 3. Start writing `devlogs.md` entries after this phase.

---

## Phase 3: HTTP Layer

**Features delivered:** Full API surface (`/v1/rerank`, `/healthcheck`, `/v1/models`, `/docs`)

- Implement request/response types:
  - `RerankRequest { model: String, query: String, documents: Vec<String> }`
  - `RerankResponse { object: String, model: String, results: Vec<ResultItem>, usage: Usage }`
  - `ResultItem { index: u32, relevance_score: f32 }`
  - `Usage { total_tokens: u32 }`
  - `HealthResponse`, `ModelsResponse`, `ErrorResponse` (same pattern as embedding API)
- Implement route handlers:
  - `POST /v1/rerank`: validate model name, extract texts, call `model.rerank()`, wrap response
    - 200: success
    - 400: wrong model name → `invalid_request_error`
    - 500: inference failure → `server_error`
  - `GET /healthcheck`: lock model, return `{ status, model }`
  - `GET /v1/models`: return single-element model list
- OpenAPI/Swagger:
  - `ApiDoc` struct with `#[openapi(paths(...), components(schemas(...)))]`
  - `#[utoipa::path(...)]` on each handler
  - Swagger UI at `/docs`
- Verify: `cargo build`, `cargo fmt --check`, `cargo clippy -- -D warnings`

**Test strategy:** Manual smoke tests via `curl` against running server:
- `POST /v1/rerank` with valid request → 200, sorted results
- `POST /v1/rerank` with wrong model → 400
- `GET /healthcheck` → 200
- `GET /v1/models` → 200
- `GET /docs` → Swagger UI loads
- Empty documents → 200, empty results
- Single document → 200, single result

---

## Phase 4: Export Script

**Features delivered:** Reproducible ONNX model export

- Create `export_model.py`:
  - Downloads `Alibaba-NLP/gte-reranker-modernbert-base` via transformers
  - Exports to `models/gte-reranker-modernbert-base/model.onnx` + tokenizer files
  - Uses `AutoModelForSequenceClassification`, opset 14, dynamic axes
- Verify: run `python3 export_model.py` → produces `model.onnx` + `tokenizer.json` + `tokenizer_config.json`

**Test strategy:** End-to-end: run export, start server, send real `curl` requests, verify scores make sense (known pair: "Paris is the capital of France" should score higher than "France is a country" for query "capital of France").

---

## Phase 5: Polish

**Features delivered:** README, benchmarks, devlog completion

- Write `README.md`: quickstart, endpoints, config, model details, build/test commands
- Benchmark: score 50 documents, measure latency (p50/p95), report tokens/sec
- Update `devlogs.md`: summary of all phases, known issues, future work
- Remove `return_documents: bool` reference from Phase 5 (deferred to v2)
- Verify: `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check`

**Test strategy:** Benchmark numbers included in README/benchmark report. All CI checks green.

---

## Feature-Phase Mapping

| Feature | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|---------|:-------:|:-------:|:-------:|:-------:|:-------:|
| CI + pre-commit | x | | | | |
| Model loading (ONNX + tokenizer) | x | | | | |
| Sigmoid scoring | | x | | | |
| Multi-document batching | | x | | | |
| Sort by score descending | | x | | | |
| `>8192` token truncation | | x | | | |
| `POST /v1/rerank` | | | x | | |
| `GET /healthcheck` | | | x | | |
| `GET /v1/models` | | | x | | |
| `GET /docs` (Swagger) | | | x | | |
| OpenAPI utoipa schemas | | | x | | |
| Token usage response | | | x | | |
| Error responses (400, 500) | | | x | | |
| `export_model.py` | | | | x | |
| README + benchmarks | | | | | x |
