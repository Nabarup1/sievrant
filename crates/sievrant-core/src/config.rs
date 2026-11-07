//! Runtime configuration.
//!
//! Sources, lowest to highest precedence:
//! 1. built-in defaults,
//! 2. an optional `sievrant.toml` (or the path passed to [`SievrantConfig::load_from`]),
//! 3. nested environment variables `SIEVRANT_<SECTION>__<FIELD>` (e.g. `SIEVRANT_LLM__MODEL`),
//! 4. the flat environment variables documented in `.env.example`
//!    (e.g. `SIEVRANT_LLM_PROVIDER`, `QDRANT_URL`).
//!
//! Provider API keys may also come from each provider's conventional variable
//! (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...); see `sievrant_llm::registry`.
//! Every secret is wrapped in [`Secret`], whose `Debug` output is redacted, so
//! printing or logging a config can never leak a key.

use serde::{Deserialize, Serialize};
use tracing::error;

use crate::error::ConfigError;

/// A secret string. Serializes transparently but never prints its value.
#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Secret(String);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Access the underlying value. Keep the call sites few and obvious.
    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_empty() {
            f.write_str("<empty>")
        } else {
            f.write_str("<redacted>")
        }
    }
}

/// Top-level configuration for a Sievrant instance.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SievrantConfig {
    pub llm: LlmConfig,
    /// Optional cheaper/faster model for query expansion, LLM reranking and
    /// synthetic calibration. Empty provider means "same as `llm`".
    pub fast_llm: LlmConfig,
    pub embedding: EmbeddingConfig,
    pub reranker: RerankerConfig,
    pub vector: VectorConfig,
    pub qdrant: QdrantConfig,
    pub retrieval: RetrievalDefaults,
    pub cache: CacheConfig,
    pub storage: StorageConfig,
    pub server: ServerConfig,
    pub observability: ObservabilityConfig,
    pub audit: AuditConfig,
    pub calibration: CalibrationConfig,
    pub routing: RoutingConfig,
}

impl SievrantConfig {
    /// Load configuration from defaults, optional `sievrant.toml`, and the environment.
    pub fn load() -> Result<Self, ConfigError> {
        let cfg = config::Config::builder()
            .add_source(config::File::with_name("sievrant").required(false))
            .add_source(
                config::Environment::with_prefix("SIEVRANT")
                    .separator("__")
                    .try_parsing(true),
            )
            .build()
            .map_err(|e| ConfigError::Parse(e.to_string()))?;

        let mut loaded: SievrantConfig = match cfg.try_deserialize() {
            Ok(c) => c,
            Err(e) => {
                error!(error = %e, "config deserialization failed");
                return Err(ConfigError::Parse(e.to_string()));
            }
        };

        loaded.apply_env(&|name| std::env::var(name).ok())?;
        loaded.validate()?;
        Ok(loaded)
    }

    /// Load configuration with a specific TOML file path.
    pub fn load_from(path: &str) -> Result<Self, ConfigError> {
        let cfg = config::Config::builder()
            .add_source(config::File::with_name(path).required(true))
            .add_source(
                config::Environment::with_prefix("SIEVRANT")
                    .separator("__")
                    .try_parsing(true),
            )
            .build()
            .map_err(|e| ConfigError::FileRead {
                path: path.into(),
                reason: e.to_string(),
            })?;

        let mut loaded: SievrantConfig = cfg
            .try_deserialize()
            .map_err(|e| ConfigError::Parse(e.to_string()))?;

        loaded.apply_env(&|name| std::env::var(name).ok())?;
        loaded.validate()?;
        Ok(loaded)
    }

    /// Apply the flat environment variables. `get` is injectable for tests.
    pub fn apply_env(&mut self, get: &dyn Fn(&str) -> Option<String>) -> Result<(), ConfigError> {
        let env = |name: &str| {
            get(name)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };

        fn parse<T: std::str::FromStr>(name: &str, v: String) -> Result<T, ConfigError> {
            v.parse().map_err(|_| ConfigError::InvalidValue {
                field: name.into(),
                value: v,
            })
        }
        fn parse_bool(name: &str, v: String) -> Result<bool, ConfigError> {
            match v.to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                _ => Err(ConfigError::InvalidValue {
                    field: name.into(),
                    value: v,
                }),
            }
        }

        macro_rules! set_str {
            ($name:expr, $target:expr) => {
                if let Some(v) = env($name) {
                    $target = v;
                }
            };
        }
        macro_rules! set_opt {
            ($name:expr, $target:expr) => {
                if let Some(v) = env($name) {
                    $target = Some(v);
                }
            };
        }
        macro_rules! set_secret {
            ($name:expr, $target:expr) => {
                if let Some(v) = env($name) {
                    $target = Some(Secret::new(v));
                }
            };
        }
        macro_rules! set_parsed {
            ($name:expr, $target:expr) => {
                if let Some(v) = env($name) {
                    $target = parse($name, v)?;
                }
            };
        }
        macro_rules! set_bool {
            ($name:expr, $target:expr) => {
                if let Some(v) = env($name) {
                    $target = parse_bool($name, v)?;
                }
            };
        }

        // LLM
        set_str!("SIEVRANT_LLM_PROVIDER", self.llm.provider);
        set_opt!("SIEVRANT_LLM_MODEL", self.llm.model);
        set_secret!("SIEVRANT_LLM_API_KEY", self.llm.api_key);
        set_opt!("SIEVRANT_LLM_BASE_URL", self.llm.base_url);
        set_opt!("SIEVRANT_LLM_EFFORT", self.llm.effort);
        set_parsed!("SIEVRANT_LLM_MAX_TOKENS", self.llm.max_answer_tokens);
        set_parsed!("SIEVRANT_LLM_MAX_RETRIES", self.llm.max_retries);
        if let Some(v) = env("SIEVRANT_ANTHROPIC_FALLBACKS") {
            self.llm.anthropic_fallbacks = !matches!(v.as_str(), "off" | "false" | "0" | "none");
        }
        set_opt!("SIEVRANT_APP_URL", self.llm.app_url);
        set_opt!("SIEVRANT_APP_NAME", self.llm.app_name);

        // Fast LLM
        set_str!("SIEVRANT_FAST_LLM_PROVIDER", self.fast_llm.provider);
        set_opt!("SIEVRANT_FAST_LLM_MODEL", self.fast_llm.model);
        set_secret!("SIEVRANT_FAST_LLM_API_KEY", self.fast_llm.api_key);
        set_opt!("SIEVRANT_FAST_LLM_BASE_URL", self.fast_llm.base_url);
        set_opt!("SIEVRANT_FAST_LLM_EFFORT", self.fast_llm.effort);

        // Embeddings
        set_str!("SIEVRANT_EMBEDDING_PROVIDER", self.embedding.provider);
        set_opt!("SIEVRANT_EMBEDDING_MODEL", self.embedding.model);
        set_secret!("SIEVRANT_EMBEDDING_API_KEY", self.embedding.api_key);
        set_opt!("SIEVRANT_EMBEDDING_BASE_URL", self.embedding.base_url);
        if let Some(v) = env("SIEVRANT_EMBEDDING_DIMENSIONS") {
            self.embedding.dimensions = Some(parse("SIEVRANT_EMBEDDING_DIMENSIONS", v)?);
        }
        set_parsed!("SIEVRANT_EMBEDDING_BATCH_SIZE", self.embedding.batch_size);

        // Reranker
        set_str!("SIEVRANT_RERANKER_PROVIDER", self.reranker.provider);
        set_opt!("SIEVRANT_RERANKER_MODEL", self.reranker.model);
        set_secret!("SIEVRANT_RERANKER_API_KEY", self.reranker.api_key);
        set_opt!("SIEVRANT_RERANKER_BASE_URL", self.reranker.base_url);

        // Vector store
        set_str!("SIEVRANT_VECTOR_BACKEND", self.vector.backend);
        set_str!("QDRANT_URL", self.qdrant.url);
        set_secret!("QDRANT_API_KEY", self.qdrant.api_key);
        set_str!("QDRANT_COLLECTION_PREFIX", self.qdrant.collection_prefix);

        // Storage
        set_str!("SIEVRANT_DATA_DIR", self.storage.data_dir);

        // Retrieval
        set_parsed!("SIEVRANT_DEFAULT_COVERAGE", self.retrieval.coverage);
        set_parsed!("SIEVRANT_TOKEN_BUDGET", self.retrieval.token_budget);
        set_parsed!("SIEVRANT_SEARCH_LIMIT", self.retrieval.search_limit);
        set_parsed!("SIEVRANT_RERANK_TOP_N", self.retrieval.rerank_top_n);
        set_parsed!("SIEVRANT_RRF_K", self.retrieval.rrf_k);
        set_bool!("SIEVRANT_QUERY_EXPANSION", self.retrieval.query_expansion);
        set_bool!(
            "SIEVRANT_EXPERIMENTAL_INDEXES",
            self.retrieval.experimental_indexes
        );

        // Server
        set_str!("SIEVRANT_HOST", self.server.host);
        set_parsed!("SIEVRANT_PORT", self.server.port);
        set_parsed!("SIEVRANT_GRPC_PORT", self.server.grpc_port);
        set_bool!("SIEVRANT_GRPC_ENABLED", self.server.grpc_enabled);
        if let Some(v) = env("SIEVRANT_API_KEYS") {
            self.server.api_keys = Some(Secret::new(v));
        }
        if let Some(v) = env("SIEVRANT_API_KEY") {
            self.server.api_key = Some(Secret::new(v));
        }
        set_opt!("SIEVRANT_API_KEY_HASHES", self.server.api_key_hashes);
        set_bool!("SIEVRANT_AUTH_DISABLED", self.server.auth_disabled);
        set_opt!("SIEVRANT_CORS_ORIGINS", self.server.cors_origins);
        set_parsed!("SIEVRANT_MAX_BODY_BYTES", self.server.max_body_bytes);
        set_parsed!(
            "SIEVRANT_MAX_CONCURRENT_REQUESTS",
            self.server.max_concurrent_requests
        );

        // Audit / receipts
        set_bool!("SIEVRANT_AUDIT_ENABLED", self.audit.enabled);
        set_secret!("SIEVRANT_SIGNING_KEY", self.audit.signing_key);

        // Calibration
        set_parsed!(
            "SIEVRANT_COVERAGE_DRIFT_TOLERANCE",
            self.calibration.drift_tolerance
        );

        // Observability
        set_str!("SIEVRANT_LOG_LEVEL", self.observability.log_level);
        set_str!("SIEVRANT_LOG_FORMAT", self.observability.log_format);
        set_bool!("SIEVRANT_OTEL_ENABLED", self.observability.otlp_enabled);
        set_str!("SIEVRANT_OTEL_ENDPOINT", self.observability.otlp_endpoint);

        Ok(())
    }

    /// Validate that all configuration values are within acceptable ranges.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |field: &str, value: String| {
            Err(ConfigError::InvalidValue {
                field: field.into(),
                value,
            })
        };
        if self.retrieval.coverage <= 0.0 || self.retrieval.coverage >= 1.0 {
            return bad("retrieval.coverage", self.retrieval.coverage.to_string());
        }
        if self.retrieval.token_budget == 0 {
            return bad("retrieval.token_budget", "0".into());
        }
        if self.retrieval.rrf_k == 0 {
            return bad("retrieval.rrf_k", "0".into());
        }
        if self.retrieval.search_limit == 0 || self.retrieval.rerank_top_n == 0 {
            return bad("retrieval.search_limit/rerank_top_n", "0".into());
        }
        if !(0.0..=1.0).contains(&self.cache.similarity_threshold) {
            return bad(
                "cache.similarity_threshold",
                self.cache.similarity_threshold.to_string(),
            );
        }
        if self.server.port == 0 {
            return bad("server.port", "0".into());
        }
        if !matches!(self.vector.backend.as_str(), "memory" | "qdrant") {
            return bad("vector.backend", self.vector.backend.clone());
        }
        Ok(())
    }
}

/// Chat model settings. `provider` is a provider id from
/// `sievrant_llm::registry::PROVIDERS`, `none` (evidence-only mode) or
/// `auto` (first provider whose API key is present in the environment).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    pub provider: String,
    /// Model id; each provider has a default.
    pub model: Option<String>,
    /// Overrides the provider's conventional key variable.
    pub api_key: Option<Secret>,
    /// Overrides the provider's default endpoint (required for `custom`).
    pub base_url: Option<String>,
    pub max_retries: u32,
    /// Upper bound on answer length in tokens. Thinking models count their
    /// reasoning against this, so do not set it too low.
    pub max_answer_tokens: usize,
    /// Anthropic `output_config.effort` (low | medium | high | xhigh | max).
    pub effort: Option<String>,
    /// Anthropic server-side refusal fallbacks (`fallbacks: "default"`).
    pub anthropic_fallbacks: bool,
    /// OpenRouter attribution headers.
    pub app_url: Option<String>,
    pub app_name: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: "auto".into(),
            model: None,
            api_key: None,
            base_url: None,
            max_retries: 3,
            max_answer_tokens: 4096,
            effort: None,
            anthropic_fallbacks: true,
            app_url: None,
            app_name: None,
        }
    }
}

/// Embedding settings. `provider` accepts a provider id, `local` (offline
/// hashing embedder) or `auto`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<Secret>,
    pub base_url: Option<String>,
    /// Vector size. Learned from the first response when unset.
    pub dimensions: Option<usize>,
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "auto".into(),
            model: None,
            api_key: None,
            base_url: None,
            dimensions: None,
            batch_size: 128,
        }
    }
}

/// Reranker settings: cohere | jina | voyage | custom | embedding | llm | none | auto.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankerConfig {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<Secret>,
    /// Full URL of a Cohere-compatible rerank endpoint (for `custom`).
    pub base_url: Option<String>,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            provider: "auto".into(),
            model: None,
            api_key: None,
            base_url: None,
        }
    }
}

/// Vector store selection: `memory` (embedded, persisted to the data dir) or `qdrant`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VectorConfig {
    pub backend: String,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            backend: "memory".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<Secret>,
    /// Collections are named `<prefix><tenant>__<corpus>`.
    pub collection_prefix: String,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6334".into(),
            api_key: None,
            collection_prefix: "sievrant_".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RetrievalDefaults {
    /// Default conformal coverage target for all retrieve() calls.
    pub coverage: f64,
    pub token_budget: usize,
    pub rrf_k: usize,
    /// Candidates requested from each index.
    pub search_limit: usize,
    /// Fused candidates passed to the reranker (and scored by the gate).
    pub rerank_top_n: usize,
    /// Maximum number of evidence chunks returned per call.
    pub max_evidence_chunks: usize,
    /// LLM query expansion. Off by default: it adds latency, cost and
    /// nondeterminism to every query.
    pub query_expansion: bool,
    /// Enable the tree and graph indexes (in-memory, not persisted).
    pub experimental_indexes: bool,
}

impl Default for RetrievalDefaults {
    fn default() -> Self {
        Self {
            coverage: 0.95,
            token_budget: 4000,
            rrf_k: 60,
            search_limit: 100,
            rerank_top_n: 50,
            max_evidence_chunks: 15,
            query_expansion: false,
            experimental_indexes: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    pub enabled: bool,
    pub max_entries: usize,
    /// Cosine similarity threshold for a semantic cache hit. Range [0.0, 1.0].
    pub similarity_threshold: f64,
    /// TTL for cache entries in seconds.
    pub ttl_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_entries: 10_000,
            similarity_threshold: 0.95,
            ttl_secs: 3600,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    pub data_dir: String,
    /// WAL segment is rotated when it exceeds this size in bytes.
    pub wal_max_segment_bytes: u64,
    /// Number of sorted runs at a level before triggering a merge.
    pub merge_fanout: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: "data".into(),
            wal_max_segment_bytes: 64 * 1024 * 1024,
            merge_fanout: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub grpc_port: u16,
    pub grpc_enabled: bool,
    pub workers: usize,
    pub request_timeout_secs: u64,
    /// Comma-separated `tenant:key` pairs.
    pub api_keys: Option<Secret>,
    /// A single key for the `default` tenant.
    pub api_key: Option<Secret>,
    /// Comma-separated `tenant:sha256hex` pairs, so raw keys never touch disk.
    pub api_key_hashes: Option<String>,
    /// Development only: accept every request as tenant `default`.
    pub auth_disabled: bool,
    /// Comma-separated allowed origins. Unset means no browser CORS access.
    pub cors_origins: Option<String>,
    pub max_body_bytes: usize,
    pub max_concurrent_requests: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8080,
            grpc_port: 50051,
            grpc_enabled: true,
            workers: 4,
            request_timeout_secs: 120,
            api_keys: None,
            api_key: None,
            api_key_hashes: None,
            auth_disabled: false,
            cors_origins: None,
            max_body_bytes: 32 * 1024 * 1024,
            max_concurrent_requests: 256,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ObservabilityConfig {
    pub log_level: String,
    pub log_format: String,
    pub otlp_enabled: bool,
    pub otlp_endpoint: String,
    pub metrics_enabled: bool,
    pub metrics_port: u16,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: "info".into(),
            log_format: "json".into(),
            otlp_enabled: false,
            otlp_endpoint: "http://localhost:4317".into(),
            metrics_enabled: false,
            metrics_port: 9090,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AuditConfig {
    pub enabled: bool,
    pub audit_dir: String,
    /// How often to write a checksum of the audit log, in seconds.
    pub checksum_interval_secs: u64,
    /// Base64-encoded 32-byte Ed25519 seed used to sign retrieval receipts.
    /// When unset, receipts are not signed.
    pub signing_key: Option<Secret>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            audit_dir: "data/audit".into(),
            checksum_interval_secs: 3600,
            signing_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CalibrationConfig {
    pub calibration_dir: String,
    pub min_calibration_examples: usize,
    /// Alert when observed coverage is significantly below nominal by more than this.
    pub drift_tolerance: f64,
    /// Mark calibration stale when the corpus grows beyond this multiple of
    /// its size at calibration time.
    pub recalibration_corpus_growth_factor: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            calibration_dir: "data/calibration".into(),
            min_calibration_examples: 50,
            drift_tolerance: 0.02,
            recalibration_corpus_growth_factor: 1.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RoutingConfig {
    pub enabled: bool,
    /// Route classification defaults to Standard when disabled.
    pub default_route: String,
    /// Below this classifier confidence, always route to Standard.
    pub confidence_threshold: f64,
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_route: "standard".into(),
            confidence_threshold: 0.6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env_from(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn default_config_passes_validation() {
        SievrantConfig::default()
            .validate()
            .expect("default config should be valid");
    }

    #[test]
    fn secrets_never_appear_in_debug_output() {
        let mut cfg = SievrantConfig::default();
        cfg.llm.api_key = Some(Secret::new("sk-live-SUPERSECRET"));
        cfg.qdrant.api_key = Some(Secret::new("qdrant-SUPERSECRET"));
        cfg.server.api_keys = Some(Secret::new("acme:sk-SUPERSECRET"));
        cfg.audit.signing_key = Some(Secret::new("c2VlZA=="));
        let printed = format!("{cfg:?} {cfg:#?}");
        assert!(!printed.contains("SUPERSECRET"), "{printed}");
        assert!(!printed.contains("c2VlZA=="));
        assert!(printed.contains("<redacted>"));
    }

    #[test]
    fn flat_env_vars_override_defaults() {
        let mut cfg = SievrantConfig::default();
        cfg.apply_env(&env_from(&[
            ("SIEVRANT_LLM_PROVIDER", "openrouter"),
            ("SIEVRANT_LLM_MODEL", "anthropic/claude-opus-5"),
            ("SIEVRANT_LLM_API_KEY", "k"),
            ("SIEVRANT_EMBEDDING_PROVIDER", "voyage"),
            ("SIEVRANT_EMBEDDING_DIMENSIONS", "1024"),
            ("SIEVRANT_RERANKER_PROVIDER", "cohere"),
            ("SIEVRANT_VECTOR_BACKEND", "qdrant"),
            ("QDRANT_URL", "http://qdrant:6334"),
            ("SIEVRANT_PORT", "9000"),
            ("SIEVRANT_QUERY_EXPANSION", "true"),
            ("SIEVRANT_ANTHROPIC_FALLBACKS", "off"),
            ("SIEVRANT_API_KEYS", "acme:sk-1,globex:sk-2"),
        ]))
        .unwrap();
        assert_eq!(cfg.llm.provider, "openrouter");
        assert_eq!(cfg.llm.model.as_deref(), Some("anthropic/claude-opus-5"));
        assert_eq!(cfg.llm.api_key.as_ref().unwrap().expose(), "k");
        assert_eq!(cfg.embedding.provider, "voyage");
        assert_eq!(cfg.embedding.dimensions, Some(1024));
        assert_eq!(cfg.reranker.provider, "cohere");
        assert_eq!(cfg.vector.backend, "qdrant");
        assert_eq!(cfg.qdrant.url, "http://qdrant:6334");
        assert_eq!(cfg.server.port, 9000);
        assert!(cfg.retrieval.query_expansion);
        assert!(!cfg.llm.anthropic_fallbacks);
        assert!(cfg.server.api_keys.is_some());
        cfg.validate().unwrap();
    }

    #[test]
    fn malformed_env_values_are_rejected() {
        let mut cfg = SievrantConfig::default();
        let err = cfg
            .apply_env(&env_from(&[("SIEVRANT_PORT", "eighty")]))
            .unwrap_err();
        assert!(err.to_string().contains("SIEVRANT_PORT"));
        let err = cfg
            .apply_env(&env_from(&[("SIEVRANT_QUERY_EXPANSION", "maybe")]))
            .unwrap_err();
        assert!(err.to_string().contains("SIEVRANT_QUERY_EXPANSION"));
    }

    #[test]
    fn blank_env_values_are_ignored() {
        let mut cfg = SievrantConfig::default();
        cfg.apply_env(&env_from(&[("SIEVRANT_LLM_MODEL", "  ")]))
            .unwrap();
        assert!(cfg.llm.model.is_none());
    }

    #[test]
    fn coverage_out_of_range_fails_validation() {
        let mut cfg = SievrantConfig::default();
        cfg.retrieval.coverage = 1.5;
        assert!(cfg.validate().is_err());

        cfg.retrieval.coverage = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_token_budget_fails_validation() {
        let mut cfg = SievrantConfig::default();
        cfg.retrieval.token_budget = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn invalid_cache_threshold_fails_validation() {
        let mut cfg = SievrantConfig::default();
        cfg.cache.similarity_threshold = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_port_fails_validation() {
        let mut cfg = SievrantConfig::default();
        cfg.server.port = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_rrf_k_fails_validation() {
        let mut cfg = SievrantConfig::default();
        cfg.retrieval.rrf_k = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn unknown_vector_backend_fails_validation() {
        let mut cfg = SievrantConfig::default();
        cfg.vector.backend = "lancedb".into();
        assert!(cfg.validate().is_err());
    }
}
