use serde::Serialize;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, SievrantError>;

/// Top-level error type. Consumers pattern-match on variants to handle specific failure modes.
/// Every variant carries enough context to diagnose the problem without needing to read logs.
#[derive(Debug, Error)]
pub enum SievrantError {
    #[error("Index error: {0}")]
    Index(#[from] IndexError),

    #[error("LLM provider error: {0}")]
    Llm(#[from] LlmError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Access denied: {reason}")]
    AccessDenied { reason: String },

    #[error("Calibration error: {0}")]
    Calibration(#[from] CalibrationError),

    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    #[error("Reranking error: {0}")]
    Rerank(#[from] RerankError),

    #[error("Query processing error: {0}")]
    Query(#[from] QueryError),

    #[error("Cache error: {0}")]
    Cache(#[from] CacheError),

    #[error("Audit error: {0}")]
    Audit(#[from] AuditError),

    #[error("Corpus not found: {name}")]
    CorpusNotFound { name: String },

    #[error("Document not found: {id}")]
    DocumentNotFound { id: String },

    #[error("Internal error: {0}")]
    Internal(String),
}

impl SievrantError {
    /// Convert to a JSON-serializable error body suitable for HTTP responses.
    pub fn to_error_response(&self) -> ErrorResponse {
        ErrorResponse {
            error: self.variant_name(),
            message: self.to_string(),
        }
    }

    fn variant_name(&self) -> &'static str {
        match self {
            Self::Index(_) => "INDEX_ERROR",
            Self::Llm(_) => "LLM_ERROR",
            Self::Config(_) => "CONFIG_ERROR",
            Self::AccessDenied { .. } => "ACCESS_DENIED",
            Self::Calibration(_) => "CALIBRATION_ERROR",
            Self::Storage(_) => "STORAGE_ERROR",
            Self::Rerank(_) => "RERANK_ERROR",
            Self::Query(_) => "QUERY_ERROR",
            Self::Cache(_) => "CACHE_ERROR",
            Self::Audit(_) => "AUDIT_ERROR",
            Self::CorpusNotFound { .. } => "CORPUS_NOT_FOUND",
            Self::DocumentNotFound { .. } => "DOCUMENT_NOT_FOUND",
            Self::Internal(_) => "INTERNAL_ERROR",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: &'static str,
    pub message: String,
}

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("Vector index error: {0}")]
    Vector(String),

    #[error("BM25 index error: {0}")]
    Bm25(String),

    #[error("Tree index error: {0}")]
    Tree(String),

    #[error("Graph index error: {0}")]
    Graph(String),

    #[error("Index not available: {index_type} is currently unavailable")]
    Unavailable { index_type: String },

    #[error("Embedding error: {0}")]
    Embedding(String),
}

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("Provider {provider} returned HTTP {status}: {body}")]
    ApiError {
        provider: String,
        status: u16,
        body: String,
    },

    #[error("Provider {provider} rate limited, retry after {retry_after_ms}ms")]
    RateLimited {
        provider: String,
        retry_after_ms: u64,
    },

    #[error("Provider {provider} request timed out after {timeout_ms}ms")]
    Timeout { provider: String, timeout_ms: u64 },

    #[error("Failed to parse response from {provider}: {reason}")]
    ParseError { provider: String, reason: String },

    #[error("Context window exceeded: {tokens_needed} tokens needed, {tokens_available} available")]
    ContextWindowExceeded {
        tokens_needed: usize,
        tokens_available: usize,
    },

    #[error("No LLM provider configured")]
    NoProviderConfigured,

    #[error("HTTP client error: {0}")]
    Http(String),
}

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Missing required field: {field}")]
    MissingField { field: String },

    #[error("Invalid value for {field}: {value}")]
    InvalidValue { field: String, value: String },

    #[error("Failed to read config file {path}: {reason}")]
    FileRead { path: String, reason: String },

    #[error("Config parse error: {0}")]
    Parse(String),
}

#[derive(Debug, Error)]
pub enum CalibrationError {
    #[error("Insufficient calibration data: need at least {min_required} examples, got {provided}")]
    InsufficientData { min_required: usize, provided: usize },

    #[error("No calibration set found for corpus {corpus}")]
    NotFound { corpus: String },

    #[error("Calibration file IO error: {0}")]
    Io(String),

    #[error("Calibration data parse error: {0}")]
    Parse(String),

    #[error("Coverage drift detected: empirical {empirical:.3} vs nominal {nominal:.3}")]
    CoverageDrift { empirical: f64, nominal: f64 },
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("WAL corrupted at segment {segment}: {reason}")]
    WalCorrupted { segment: u64, reason: String },

    #[error("WAL write failed: {0}")]
    WalWrite(String),

    #[error("Disk IO error: {0}")]
    Io(String),

    #[error("Checksum mismatch for file {path}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        path: String,
        expected: String,
        actual: String,
    },

    #[error("Merge operation failed at level {level}: {reason}")]
    MergeFailed { level: usize, reason: String },

    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl From<std::io::Error> for StorageError {
    fn from(e: std::io::Error) -> Self {
        StorageError::Io(e.to_string())
    }
}

impl From<serde_json::Error> for StorageError {
    fn from(e: serde_json::Error) -> Self {
        StorageError::Serialization(e.to_string())
    }
}

#[derive(Debug, Error)]
pub enum RerankError {
    #[error("Reranker model not found at {path}")]
    ModelNotFound { path: String },

    #[error("Failed to load reranker model: {0}")]
    ModelLoad(String),

    #[error("ONNX inference error: {0}")]
    Inference(String),

    #[error("ONNX runtime error: {0}")]
    Onnx(String),

    #[error("Reranker API error: {0}")]
    Api(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Reranker unavailable: {0}")]
    Unavailable(String),
}

#[derive(Debug, Error)]
pub enum QueryError {
    #[error("Query expansion failed: {0}")]
    Expansion(String),

    #[error("Query decomposition failed: {0}")]
    Decomposition(String),

    #[error("Embedding generation failed: {0}")]
    Embedding(String),

    #[error("Empty query string")]
    EmptyQuery,
}

#[derive(Debug, Error)]
pub enum CacheError {
    #[error("Cache IO error: {0}")]
    Io(String),

    #[error("Cache serialization error: {0}")]
    Serialization(String),

    #[error("Cache capacity exceeded")]
    CapacityExceeded,
}

impl From<serde_json::Error> for CacheError {
    fn from(e: serde_json::Error) -> Self {
        CacheError::Serialization(e.to_string())
    }
}

#[derive(Debug, Error)]
pub enum AuditError {
    #[error("Audit log IO error: {0}")]
    Io(String),

    #[error("Audit log integrity check failed for {path}: tamper detected")]
    TamperDetected { path: String },

    #[error("Audit serialization error: {0}")]
    Serialization(String),
}

impl From<std::io::Error> for AuditError {
    fn from(e: std::io::Error) -> Self {
        AuditError::Io(e.to_string())
    }
}

/// Extension trait for adding context messages to any error type, converting it to
/// SievrantError::Internal. Used in internal pipeline code where the specific error
/// type is not meaningful to the caller.
pub trait ErrorContext<T> {
    fn context(self, msg: &str) -> Result<T>;
    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T>;
}

impl<T, E: std::error::Error> ErrorContext<T> for std::result::Result<T, E> {
    fn context(self, msg: &str) -> Result<T> {
        self.map_err(|e| SievrantError::Internal(format!("{msg}: {e}")))
    }

    fn with_context<F: FnOnce() -> String>(self, f: F) -> Result<T> {
        self.map_err(|e| SievrantError::Internal(format!("{}: {e}", f())))
    }
}

/// Represents the outcome of a pipeline stage that can partially fail without
/// stopping the entire retrieval. The orchestrator uses this to continue with
/// fewer indexes when one is unavailable.
#[derive(Debug)]
pub enum DegradedResult<T> {
    /// Stage completed without issues.
    Ok(T),
    /// Stage completed with reduced quality; pipeline continues with a warning attached.
    Degraded { partial: T, warning: String },
    /// Stage failed in a way that makes continuing unsafe or meaningless.
    Fatal(SievrantError),
}

impl<T> DegradedResult<T> {
    pub fn is_fatal(&self) -> bool {
        matches!(self, DegradedResult::Fatal(_))
    }

    pub fn warning(&self) -> Option<&str> {
        match self {
            DegradedResult::Degraded { warning, .. } => Some(warning),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_index_error_converts_correctly() {
        let idx_err = IndexError::Unavailable {
            index_type: "vector".into(),
        };
        let siev_err: SievrantError = idx_err.into();
        assert!(matches!(siev_err, SievrantError::Index(_)));
        assert!(siev_err.to_string().contains("vector"));
    }

    #[test]
    fn from_llm_error_converts_correctly() {
        let llm_err = LlmError::NoProviderConfigured;
        let siev_err: SievrantError = llm_err.into();
        assert!(matches!(siev_err, SievrantError::Llm(_)));
    }

    #[test]
    fn error_response_serializes_to_json() {
        let err = SievrantError::CorpusNotFound {
            name: "test_corpus".into(),
        };
        let response = err.to_error_response();
        let json = serde_json::to_string(&response).expect("serialization failed");
        assert!(json.contains("CORPUS_NOT_FOUND"));
        assert!(json.contains("test_corpus"));
    }

    #[test]
    fn error_context_trait_wraps_errors() {
        let result: std::result::Result<(), std::io::Error> =
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "file missing"));
        let siev_result = result.context("reading config");
        assert!(matches!(siev_result, Err(SievrantError::Internal(_))));
        assert!(siev_result
            .unwrap_err()
            .to_string()
            .contains("reading config"));
    }

    #[test]
    fn storage_error_converts_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let storage_err: StorageError = io_err.into();
        assert!(matches!(storage_err, StorageError::Io(_)));
    }

    #[test]
    fn degraded_result_is_fatal_check() {
        let ok: DegradedResult<i32> = DegradedResult::Ok(42);
        assert!(!ok.is_fatal());

        let fatal: DegradedResult<i32> = DegradedResult::Fatal(SievrantError::Internal("oops".into()));
        assert!(fatal.is_fatal());
    }
}
