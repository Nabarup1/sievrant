use axum::{
    extract::{Request, State},
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use tracing::{warn, Instrument};

use crate::api::{auth::AuthError, error::ApiError, state::AppState};

/// Resolved tenant context injected by [`auth_middleware`].
#[derive(Debug, Clone)]
pub struct TenantContext {
    pub tenant_id: String,
}

/// Validates `Authorization: Bearer <key>` and injects a [`TenantContext`].
pub async fn auth_middleware(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    let header = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    match state.keys.authenticate(header) {
        Ok(tenant_id) => {
            req.extensions_mut().insert(TenantContext { tenant_id });
            next.run(req).await
        }
        Err(AuthError::Missing) => {
            ApiError::unauthorized("Authorization: Bearer <api key> header required")
                .into_response()
        }
        Err(AuthError::Invalid) => {
            warn!(path = %req.uri().path(), "rejected invalid API key");
            ApiError::unauthorized("API key is invalid or revoked").into_response()
        }
    }
}

/// Load shedding, per-request timeout and request id.
///
/// When `max_concurrent_requests` are in flight, new requests get 503
/// immediately rather than queueing without bound.
pub async fn limits_middleware(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    let request_id = uuid::Uuid::new_v4().to_string();
    let Ok(_permit) = state.limiter.clone().try_acquire_owned() else {
        let mut resp = ApiError::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "overloaded",
            "too many concurrent requests; retry with backoff",
        )
        .into_response();
        resp.headers_mut()
            .insert("retry-after", HeaderValue::from_static("1"));
        return resp;
    };

    let timeout = if req.uri().path().starts_with("/v1/calibration") {
        state.calibration_timeout
    } else {
        state.request_timeout
    };
    let span = tracing::info_span!("request", request_id = %request_id, method = %req.method(), path = %req.uri().path());
    let mut resp = match tokio::time::timeout(timeout, next.run(req).instrument(span)).await {
        Ok(r) => r,
        Err(_) => ApiError::new(
            StatusCode::GATEWAY_TIMEOUT,
            "timeout",
            format!("request exceeded {}s", timeout.as_secs()),
        )
        .into_response(),
    };
    if let Ok(v) = HeaderValue::from_str(&request_id) {
        resp.headers_mut().insert("x-request-id", v);
    }
    resp
}
