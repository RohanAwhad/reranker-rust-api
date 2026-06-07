mod reranker;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::{Arc, Mutex};
use utoipa::{OpenApi, ToSchema};
use utoipa_swagger_ui::SwaggerUi;

use reranker::RerankerModel;

#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<RerankerModel>>,
}

#[derive(OpenApi)]
#[openapi(
    paths(healthcheck, list_models, rerank),
    components(schemas(
        RerankRequest, RerankResponse, ResultItem, Usage,
        ModelsResponse, ModelInfo, HealthResponse,
    )),
    tags(
        (name = "Rerank", description = "Reranking API")
    )
)]
struct ApiDoc;

#[derive(Serialize, ToSchema)]
struct HealthResponse {
    status: String,
    model: String,
}

#[derive(Serialize, ToSchema)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelInfo>,
}

#[derive(Serialize, ToSchema)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Deserialize, ToSchema)]
struct RerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
}

#[derive(Serialize, ToSchema)]
struct RerankResponse {
    object: String,
    model: String,
    results: Vec<ResultItem>,
    usage: Usage,
}

#[derive(Serialize, ToSchema)]
struct ResultItem {
    index: u32,
    relevance_score: f32,
}

#[derive(Serialize, ToSchema)]
struct Usage {
    total_tokens: u32,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

#[utoipa::path(
    get,
    path = "/healthcheck",
    tag = "Rerank",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse)
    )
)]
async fn healthcheck(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let model = state.model.lock().unwrap();
    Json(HealthResponse {
        status: "ok".into(),
        model: model.name.clone(),
    })
}

#[utoipa::path(
    get,
    path = "/v1/models",
    tag = "Rerank",
    responses(
        (status = 200, description = "List available models", body = ModelsResponse)
    )
)]
async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelsResponse> {
    let model = state.model.lock().unwrap();
    Json(ModelsResponse {
        object: "list".into(),
        data: vec![ModelInfo {
            id: model.name.clone(),
            object: "model".into(),
            owned_by: "local".into(),
        }],
    })
}

#[utoipa::path(
    post,
    path = "/v1/rerank",
    tag = "Rerank",
    request_body = RerankRequest,
    responses(
        (status = 200, description = "Reranking completed successfully", body = RerankResponse),
        (status = 400, description = "Invalid request"),
        (status = 500, description = "Inference error")
    )
)]
async fn rerank(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RerankRequest>,
) -> impl IntoResponse {
    let mut model = state.model.lock().unwrap();

    if req.model != model.name {
        let error = ErrorResponse {
            error: ErrorDetail {
                message: format!(
                    "Model '{}' not found. Available model: '{}'",
                    req.model, model.name
                ),
                error_type: "invalid_request_error".into(),
            },
        };
        return (StatusCode::BAD_REQUEST, Json(error)).into_response();
    }

    let total_tokens: u32 = req
        .documents
        .iter()
        .map(|doc| model.count_tokens(&req.query, doc).unwrap_or(0))
        .sum();

    match model.rerank(&req.query, &req.documents) {
        Ok(results) => {
            let results: Vec<ResultItem> = results
                .into_iter()
                .map(|(index, relevance_score)| ResultItem {
                    index: index as u32,
                    relevance_score,
                })
                .collect();

            let response = RerankResponse {
                object: "list".into(),
                model: model.name.clone(),
                results,
                usage: Usage { total_tokens },
            };

            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            let error = ErrorResponse {
                error: ErrorDetail {
                    message: format!("Reranking failed: {e}"),
                    error_type: "server_error".into(),
                },
            };
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
        }
    }
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
        Path::new(&model_path),
        &model_name,
    )?));
    tracing::info!("Model '{}' loaded successfully", model_name);

    let state = Arc::new(AppState { model });

    let app = Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .route("/healthcheck", get(healthcheck))
        .route("/v1/models", get(list_models))
        .route("/v1/rerank", post(rerank))
        .with_state(state);

    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".to_string());
    tracing::info!("Starting server on {}", bind_addr);

    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
