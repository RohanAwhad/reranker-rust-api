# architecture.md

## File Structure

```
reranker_api_rust/
├── .dingllm/prds/v1/
│   ├── overview.md
│   ├── features-and-use-cases.md
│   ├── api-spec.md
│   ├── architecture.md              ← this file
│   └── implementation-plan.md
├── .github/workflows/ci.yml
├── .git/hooks/pre-commit
├── .gitignore
├── Cargo.toml
├── README.md
├── devlogs.md
├── export_model.py                  ← cross-encoder ONNX export
├── models/
│   └── gte-reranker-modernbert-base/
│       ├── model.onnx               (gitignored)
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── src/
    ├── main.rs                      ← axum server, routes, OpenAPI, env config
    └── reranker.rs                  ← RerankerModel: load, rerank, sigmoid
```

---

## Data Flow

```
POST /v1/rerank
  → axum handler (JSON deserialization, model name validation)
  → tokenizer: encode [query + SEP + doc1, query + SEP + doc2, ...]  (pair-batch)
  → session.run()  (ONNX Runtime, graph opt level 3)
  → extract logits → sigmoid per pair
  → sort by score descending (stable: ties broken by input index)
  → JSON response with { index, relevance_score }
```

---

## Crate Structure

Single binary crate. No `lib.rs` — both modules declared in `main.rs`:

```
main.rs ── mod reranker;     (RerankerModel, sigmoid)
        ├─ mod main          (axum types, handlers, server boot)
        └─ #[tokio::main]
```

---

## Key Differences from Embedding API

| Aspect | Embedding API | Reranker API |
|--------|--------------|--------------|
| ONNX input | `N` independent texts | `N` pairs: `[query + SEP + doc] × N` |
| ONNX output | `[B, L, 768]` hidden states | `[B, 1]` logit per pair |
| ONNX model class | `AutoModel` | `AutoModelForSequenceClassification` |
| Post-processing | Mean pooling + L2 normalize | Sigmoid on raw logit |
| Rust post-processing | `mean_pooling()` + `l2_normalize()` functions | `sigmoid()` function only |
| API input fields | `input: string \| string[]` | `query: string`, `documents: string[]` |
| API output fields | `data: [{ embedding }]` | `results: [{ index, relevance_score }]` |
| Token counting | Sum tokens per text | Sum tokens per pair (query + sep + doc) |
| Sorting | None (preserve input order) | Sort results by score descending |

---

## ONNX Export

Cross-encoder export uses `AutoModelForSequenceClassification` (not `AutoModel`):

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_id = "Alibaba-NLP/gte-reranker-modernbert-base"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

inputs = tokenizer("query text", "document text", return_tensors="pt")
# Produces: [CLS] query [SEP] doc [SEP]

with torch.no_grad():
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"]),
        "models/gte-reranker-modernbert-base/model.onnx",
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        dynamo=False,
    )
```

### Sigmoid

Cross-encoders trained with cross-entropy loss output a raw logit. Apply sigmoid for a probability-like score in `[0, 1]`:

```rust
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
```

---

## Dependencies

```toml
[dependencies]
axum = "0.8"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
ort = { version = "=2.0.0-rc.12", features = ["download-binaries"] }
tokenizers = "0.21"
ndarray = "0.17"
utoipa = { version = "5", features = ["axum_extras"] }
utoipa-swagger-ui = { version = "9", features = ["axum"] }
anyhow = "1"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

---

## State Model

```rust
struct AppState {
    model: Arc<Mutex<RerankerModel>>,  // Mutex because Session::run needs &mut self
}
```

Single model, single session. Loaded at startup, shared across all handlers via `Arc`.

Note: `ndarray` version must match `ort`'s internal ndarray dependency version. `ort = "=2.0.0-rc.12"` requires `ndarray = "0.17"`.

### Resolved Design Decisions

| Decision | Resolution |
|----------|-----------|
| Pair tokenization | `EncodeInput::Dual(query.into(), doc.into())` in Rust matches Python `tokenizer(query, doc)` |
| `token_type_ids` | ModernBERT tokenizer produces none; ONNX graph still expects the tensor → pass all zeros |
| Output shape | Confirmed `[B, 1]` via ONNX inspection → iterate `data[i]` directly (flat output) |
| MAX_DOCUMENTS | 500 per request; exceeded → error returned before tokenization |
| Log level | `RUST_LOG` env var via `tracing-subscriber` with `env-filter` feature |
