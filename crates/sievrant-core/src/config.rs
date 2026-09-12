use serde::{Deserialize, Serialize};

use crate::error::ConfigError;

/// Top-level configuration for a Sievrant instance.
/// Loaded in priority order: defaults, sievrant.toml, then SIEVRANT_* environment variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SievrantConfig {
    pub llm: LlmConfig,
    pub embedding: EmbeddingConfig,
    pub qdrant: QdrantConfig,
    pub reranker: RerankerConfig,
    pub retrieval: RetrievalDefaults,
    pub cache: CacheConfig,
    pub storage: StorageConfig,
    pub server: ServerConfig,
    pub observability: ObservabilityConfig,
    pub audit: AuditConfig,
    pub calibration: CalibrationConfig,
    pub routing: RoutingConfig,
}

impl Default for SievrantConfig {
    fn default() -> Self {
        Self {
            llm: LlmConfig::default(),
            embedding: EmbeddingConfig::default(),
            qdrant: QdrantConfig::default(),
            reranker: RerankerConfig::default(),
            retrieval: RetrievalDefaults::default(),
            cache: CacheConfig::default(),
            storage: StorageConfig::default(),
            server: ServerConfig::default(),
            observability: ObservabilityConfig::default(),
            audit: AuditConfig::default(),
            calibration: CalibrationConfig::default(),
            routing: RoutingConfig::default(),
        }
    }
}

impl SievrantConfig {
    /// Load configuration from defaults, optional sievrant.toml, and environment variables.
    /// Environment variables take highest precedence and use the SIEVRANT_ prefix.
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
                println!("CONFIG DESERIALIZATION ERROR: {:?}", e);
                SievrantConfig::default()
            }
        };

        if let Ok(model) = std::env::var("SIEVRANT_EMBEDDING__MODEL") {
            loaded.embedding.model = model;
        }
        if let Ok(provider) = std::env::var("SIEVRANT_EMBEDDING__PROVIDER") {
            loaded.embedding.provider = provider;
        }
        if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
            loaded.llm.openai_base_url = Some(base_url);
        }
        if let Ok(model) = std::env::var("OPENAI_MODEL") {
            loaded.llm.openai_model = model;
        }

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

        let loaded: SievrantConfig = cfg
            .try_deserialize()
            .map_err(|e| ConfigError::Parse(e.to_string()))?;

        loaded.validate()?;
        Ok(loaded)
    }

    /// Validate that all configuration values are within acceptable ranges.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.retrieval.coverage <= 0.0 || self.retrieval.coverage >= 1.0 {
            return Err(ConfigError::InvalidValue {
                field: "retrieval.coverage".into(),
                value: self.retrieval.coverage.to_string(),
            });
        }
        if self.retrieval.token_budget == 0 {
            return Err(ConfigError::InvalidValue {
                field: "retrieval.token_budget".into(),
                value: "0".into(),
            });
        }
        if self.retrieval.rrf_k == 0 {
            return Err(ConfigError::InvalidValue {
                field: "retrieval.rrf_k".into(),
                value: "0".into(),
            });
        }
        if !(0.0..=1.0).contains(&self.cache.similarity_threshold) {
            return Err(ConfigError::InvalidValue {
                field: "cache.similarity_threshold".into(),
                value: self.cache.similarity_threshold.to_string(),
            });
        }
        if self.server.port == 0 {
            return Err(ConfigError::InvalidValue {
                field: "server.port".into(),
                value: "0".into(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// SIEVRANT_LLM__DEFAULT_PROVIDER. Values: "openai", "anthropic", "gemini".
    pub default_provider: String,
    pub openai_api_key: Option<String>,
    pub anthropic_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub openai_model: String,
    pub anthropic_model: String,
    pub gemini_model: String,
    pub openai_base_url: Option<String>,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    pub max_retries: u32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            default_provider: "openai".into(),
            openai_api_key: None,
            anthropic_api_key: None,
            gemini_api_key: None,
            openai_model: "gpt-4o".into(),
            anthropic_model: "claude-sonnet-4-5".into(),
            gemini_model: "gemini-2.5-pro".into(),
            openai_base_url: None,
            timeout_secs: 60,
            max_retries: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EmbeddingConfig {
    pub provider: String,
    pub model: String,
    pub dimensions: usize,
    pub batch_size: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: "openai".into(),
            model: "text-embedding-3-large".into(),
            dimensions: 3072,
            batch_size: 512,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QdrantConfig {
    pub url: String,
    pub grpc_port: u16,
    pub api_key: Option<String>,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: "http://localhost:6333".into(),
            grpc_port: 6334,
            api_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankerConfig {
    /// "onnx" for local model, "cohere" for API-based.
    pub reranker_type: String,
    pub model_path: String,
    pub cohere_api_key: Option<String>,
    pub cohere_model: String,
    /// Number of candidates to pass to reranker. Higher is more accurate but slower.
    pub top_n: usize,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            reranker_type: "onnx".into(),
            model_path: "models/bge-reranker-v2-m3".into(),
            cohere_api_key: None,
            cohere_model: "rerank-v3.5".into(),
            top_n: 100,
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
    pub rerank_top_n: usize,
    /// Maximum number of evidence chunks returned per call.
    pub max_evidence_chunks: usize,
}

impl Default for RetrievalDefaults {
    fn default() -> Self {
        Self {
            coverage: 0.95,
            token_budget: 4000,
            rrf_k: 60,
            rerank_top_n: 100,
            max_evidence_chunks: 15,
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
            enabled: true,
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
    pub workers: usize,
    pub request_timeout_secs: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8080,
            workers: 4,
            request_timeout_secs: 120,
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
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            audit_dir: "data/audit".into(),
            checksum_interval_secs: 3600,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CalibrationConfig {
    pub calibration_dir: String,
    pub min_calibration_examples: usize,
    /// If empirical coverage drifts more than this from nominal, alert and recalibrate.
    pub drift_tolerance: f64,
    /// Recalibrate when corpus grows beyond this multiple of its size at last calibration.
    pub recalibration_corpus_growth_factor: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            calibration_dir: "data/calibration".into(),
            min_calibration_examples: 100,
            drift_tolerance: 0.02,
            recalibration_corpus_growth_factor: 2.0,
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

    #[test]
    fn default_config_passes_validation() {
        let cfg = SievrantConfig::default();
        cfg.validate().expect("default config should be valid");
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
}
