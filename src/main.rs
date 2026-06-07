mod reranker;

use axum::{routing::get, Router};
use std::sync::{Arc, Mutex};

use reranker::RerankerModel;

#[derive(Clone)]
#[allow(dead_code)]
struct AppState {
    model: Arc<Mutex<RerankerModel>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| "models/gte-reranker-modernbert-base".to_string());
    let model_name =
        std::env::var("MODEL_NAME").unwrap_or_else(|_| "gte-reranker-modernbert-base".to_string());

    tracing::info!("Loading model from: {}", model_path);
    let model = Arc::new(Mutex::new(RerankerModel::load(
        std::path::Path::new(&model_path),
        &model_name,
    )?));
    tracing::info!("Model '{}' loaded successfully", model_name);

    let state = Arc::new(AppState { model });

    let app = Router::new()
        .route("/healthcheck", get(healthcheck))
        .route("/v1/models", get(list_models))
        .with_state(state);

    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".to_string());
    tracing::info!("Starting server on {}", bind_addr);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn healthcheck() -> &'static str {
    "ok"
}

async fn list_models() -> &'static str {
    "[]"
}
