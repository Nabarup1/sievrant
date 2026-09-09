use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Json, Response},
};
use serde_json::json;
use tracing::warn;

use crate::api::rest::AppState;

/// Resolved tenant context injected by auth middleware into every request.
#[derive(Debug, Clone)]
pub struct TenantContext {
    pub tenant_id: String,
    pub api_key_prefix: String,
}

/// Auth middleware: validates `Authorization: Bearer <token>` and injects
/// a `TenantContext` extension into the request on success.
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    let token = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::to_owned);

    match token {
        None => {
            warn!("request missing Authorization header");
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": "missing_token",
                    "message": "Authorization: Bearer <token> header required"
                })),
            )
                .into_response()
        }
        Some(raw) => match state.verify_token(&raw) {
            None => {
                warn!("invalid API key");
                (
                    StatusCode::UNAUTHORIZED,
                    Json(json!({
                        "error": "invalid_token",
                        "message": "API key is invalid or revoked"
                    })),
                )
                    .into_response()
            }
            Some(tenant_id) => {
                let ctx = TenantContext {
                    tenant_id,
                    api_key_prefix: raw.chars().take(8).collect(),
                };
                req.extensions_mut().insert(ctx);
                next.run(req).await
            }
        },
    }
}
