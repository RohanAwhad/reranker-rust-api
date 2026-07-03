use std::path::Path;

use reranker_api::RerankerModel;

fn setup_model() -> RerankerModel {
    RerankerModel::load(
        Path::new("models/gte-reranker-modernbert-base"),
        "gte-reranker-modernbert-base",
    )
    .expect("failed to load model")
}

#[test]
fn parity_ordering_paris_query() {
    let mut model = setup_model();
    let results = model
        .rerank(
            "What is the capital of France?",
            &[
                "Paris is the capital of France.".into(),
                "France is a country in Europe.".into(),
                "The capital of France is Paris, a major European city.".into(),
            ],
        )
        .unwrap();

    let paris_idx = results.iter().position(|r| r.0 == 0).unwrap();
    let france_idx = results.iter().position(|r| r.0 == 1).unwrap();
    assert!(
        paris_idx < france_idx,
        "Paris doc should rank above France-as-country doc"
    );
}

#[test]
fn parity_ordering_pasta_query() {
    let mut model = setup_model();
    let results = model
        .rerank(
            "How do I cook pasta?",
            &[
                "Boil water, add salt, cook pasta for 8-10 minutes.".into(),
                "The Eiffel Tower is in Paris.".into(),
            ],
        )
        .unwrap();

    let recipe_idx = results.iter().position(|r| r.0 == 0).unwrap();
    let eiffel_idx = results.iter().position(|r| r.0 == 1).unwrap();
    assert!(
        recipe_idx < eiffel_idx,
        "Pasta recipe should outrank Eiffel Tower"
    );
}

#[test]
fn parity_onnx_vs_python_ort_same() {
    // Verify Rust ONNX Runtime produces the same scores as Python ONNX Runtime
    // (both use the same ONNX model file). Scores confirmed via:
    //   python3 -c "import onnxruntime; session = ort.InferenceSession('model.onnx'); ..."
    let mut model = setup_model();
    let results = model
        .rerank(
            "machine learning definition",
            &[
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.".into(),
                "Python is a high-level programming language.".into(),
            ],
        )
        .unwrap();

    // Python ORT raw logits: ML doc → 1.516696, Python doc → 1.651557
    // After sigmoid: ML doc → 0.820048, Python doc → 0.839089
    let ml_score = results.iter().find(|r| r.0 == 0).unwrap().1;
    let py_score = results.iter().find(|r| r.0 == 1).unwrap().1;
    assert!((ml_score - 0.820048).abs() < 0.001, "ML score: {ml_score}");
    assert!(
        (py_score - 0.839089).abs() < 0.001,
        "Python score: {py_score}"
    );
}

#[test]
fn parity_logit_scale() {
    let mut model = setup_model();

    let documents: Vec<String> = vec![
        "Paris is the capital of France.".into(),
        "How do I cook pasta?".into(),
        "Machine learning is a subset of artificial intelligence.".into(),
    ];

    // Run twice with same inputs to check determinism
    let results1 = model
        .rerank("What is the capital of France?", &documents)
        .unwrap();
    let results2 = model
        .rerank("What is the capital of France?", &documents)
        .unwrap();

    assert_eq!(results1.len(), results2.len());
    for i in 0..results1.len() {
        assert!(
            (results1[i].1 - results2[i].1).abs() < 1e-6,
            "Non-deterministic scores: run1={} run2={}",
            results1[i].1,
            results2[i].1
        );
    }
}

#[test]
fn parity_batch_size_invariant() {
    let mut model = setup_model();

    let query = "What is deep learning?";
    let doc = "Deep learning uses neural networks.".to_string();

    let results_single = model.rerank(query, &[doc.clone()]).unwrap();
    let results_batch = model
        .rerank(query, &[doc, "Unrelated text about cooking pasta.".into()])
        .unwrap();

    let single_score = results_single[0].1;
    let batch_score = results_batch.iter().find(|(i, _)| *i == 0).unwrap().1;

    assert!(
        (single_score - batch_score).abs() < 0.02,
        "Score varies with batch size: single={single_score} batch={batch_score}"
    );
}
