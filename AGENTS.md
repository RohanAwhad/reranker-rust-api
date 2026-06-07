# AGENTS.md

## Build & check

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo build --release
```

CI runs these as separate jobs on push/PR to `master`.

## Tests require model files

Tests load the ONNX model from `models/gte-reranker-modernbert-base/`. These files are gitignored (~599 MB) and must be exported first:

```bash
pip install torch transformers onnx
python3 export_model.py
```

Without model files, `cargo test` will panic at `setup_model()`.

## Project layout

- `src/main.rs` — axum server, routes, request/response types, OpenAPI schemas
- `src/reranker.rs` — `RerankerModel`: load, tokenize, ONNX inference, sigmoid, sort
- `src/lib.rs` — re-exports `RerankerModel` (used by both binary and tests)
- `tests/parity_test.rs` — integration tests that verify ONNX output matches Python ORT scores
- `export_model.py` — downloads HF model and exports to ONNX (gitignored but present on disk)

## Quirks

- **`ort` pinned to `=2.0.0-rc.12`** — exact version pin on a release candidate. Don't upgrade without testing.
- **`token_type_ids` must be zeros** — ModernBERT tokenizer doesn't produce them, but the ONNX graph expects the input tensor. Pass zeros.
- **`Session::run` takes `&mut self`** — model is wrapped in `Arc<Mutex<>>`. Healthcheck also acquires the lock.
- **Default branch is `master`**, not `main`.

## Env vars

| Var | Default |
|-----|---------|
| `MODEL_PATH` | `models/gte-reranker-modernbert-base` |
| `MODEL_NAME` | `gte-reranker-modernbert-base` |
| `BIND_ADDR` | `0.0.0.0:3000` |
| `RUST_LOG` | `info` |
