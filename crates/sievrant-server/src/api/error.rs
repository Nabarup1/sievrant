//! One error type for every REST handler: `{"error": code, "message": text}`.

use axum::{
    async_trait,
    extract::{rejection::JsonRejection, FromRequest, Request},
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::de::DeserializeOwned;
use sievrant_core::error::{CalibrationError, ConfigError, IndexError, LlmError, SievrantError};

#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    pub code: &'static str,
    pub message: String,
}

impl ApiError {
    pub fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    pub fn validation(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "validation_error",
            message,
        )
    }

    pub fn corpus_not_found(name: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            "corpus_not_found",
            format!("corpus '{name}' does not exist for this tenant"),
        )
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "unauthorized", message)
    }
}

impl From<SievrantError> for ApiError {
    fn from(e: SievrantError) -> Self {
        use SievrantError as E;
        match &e {
            E::CorpusNotFound { name } => Self::corpus_not_found(name),
            E::DocumentNotFound { id } => Self::new(
                StatusCode::NOT_FOUND,
                "document_not_found",
                format!("document '{id}' not found"),
            ),
            E::Config(ConfigError::InvalidValue { .. } | ConfigError::MissingField { .. })
            | E::Calibration(CalibrationError::InsufficientData { .. }) => {
                Self::validation(e.to_string())
            }
            E::Calibration(CalibrationError::NotFound { .. }) => Self::new(
                StatusCode::NOT_FOUND,
                "calibration_not_found",
                e.to_string(),
            ),
            E::AccessDenied { .. } => Self::new(StatusCode::FORBIDDEN, "forbidden", e.to_string()),
            // Upstream model / provider problems: the client may retry.
            E::Llm(LlmError::RateLimited { .. } | LlmError::Timeout { .. })
            | E::Index(IndexError::Unavailable { .. }) => Self::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                e.to_string(),
            ),
            E::Llm(_) | E::Rerank(_) | E::Index(IndexError::Embedding(_)) | E::Query(_) => {
                Self::new(StatusCode::BAD_GATEWAY, "upstream_error", e.to_string())
            }
            // Storage, audit, internal: log the detail, return a generic message.
            _ => {
                tracing::error!(error = %e, "internal error");
                Self::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_error",
                    "internal server error; see server logs",
                )
            }
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "error": self.code, "message": self.message })),
        )
            .into_response()
    }
}

/// `axum::Json` with JSON-shaped rejections (bad JSON, wrong content type,
/// body too large) instead of plain-text ones.
pub struct ApiJson<T>(pub T);

#[async_trait]
impl<S, T> FromRequest<S> for ApiJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(v)) => Ok(ApiJson(v)),
            Err(rej) => {
                let status = rej.status();
                let code = match &rej {
                    JsonRejection::BytesRejection(_) if status == StatusCode::PAYLOAD_TOO_LARGE => {
                        "payload_too_large"
                    }
                    _ if status == StatusCode::UNPROCESSABLE_ENTITY => "validation_error",
                    _ => "bad_request",
                };
                Err(ApiError::new(status, code, rej.body_text()))
            }
        }
    }
}
