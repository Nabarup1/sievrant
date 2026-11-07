//! Providers shared by every corpus in a process: chat LLM, fast LLM,
//! embeddings, reranker, receipt signer and audit log. Built once from
//! [`SievrantConfig`].

use std::sync::Arc;

use serde::Serialize;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use sievrant_audit::{AuditLogger, ReceiptSigner};
use sievrant_core::config::{LlmConfig, SievrantConfig};
use sievrant_core::error::{ConfigError, Result, SievrantError};
use sievrant_llm::registry::{provider_info, resolve_api_key, ChatProtocol, PROVIDERS};
use sievrant_llm::{
    build_embedder, build_llm, EmbeddingProvider, EmbeddingSettings, LlmProvider, LlmSettings,
};
use sievrant_rerank::{EmbeddingReranker, HttpReranker, LlmReranker, NoReranker, Reranker};

/// Chat providers tried, in order, when `provider = "auto"`.
const AUTO_LLM_ORDER: &[&str] = &[
    "anthropic",
    "openai",
    "gemini",
    "openrouter",
    "mistral",
    "groq",
    "deepseek",
    "together",
    "fireworks",
    "xai",
];
/// Embedding providers tried, in order, when `provider = "auto"`.
const AUTO_EMBEDDING_ORDER: &[&str] = &[
    "openai",
    "gemini",
    "voyage",
    "mistral",
    "jina",
    "together",
    "fireworks",
];

fn config_err(field: &str, value: impl Into<String>) -> SievrantError {
    SievrantError::Config(ConfigError::InvalidValue {
        field: field.into(),
        value: value.into(),
    })
}

fn has_key(provider: &str) -> bool {
    provider_info(provider)
        .map(|p| resolve_api_key(p, &None).is_some())
        .unwrap_or(false)
}

fn resolve_auto(requested: &str, order: &[&str], fallback: &str) -> String {
    let requested = requested.trim().to_ascii_lowercase();
    if requested.is_empty() || requested == "auto" {
        order
            .iter()
            .find(|p| has_key(p))
            .map(|p| p.to_string())
            .unwrap_or_else(|| fallback.to_string())
    } else {
        requested
    }
}

fn llm_settings(cfg: &LlmConfig, provider: String) -> LlmSettings {
    LlmSettings {
        provider,
        model: cfg.model.clone(),
        api_key: cfg.api_key.as_ref().map(|s| s.expose().to_string()),
        base_url: cfg.base_url.clone(),
        max_retries: cfg.max_retries,
        effort: cfg.effort.clone(),
        anthropic_fallbacks: cfg.anthropic_fallbacks,
        app_url: cfg.app_url.clone(),
        app_name: cfg.app_name.clone(),
    }
}

/// Human-readable summary of the active providers (no secrets).
#[derive(Debug, Clone, Serialize)]
pub struct ProviderSummary {
    pub llm: String,
    pub fast_llm: String,
    pub embedding: String,
    pub reranker: String,
    pub vector_backend: String,
    pub receipts_signed: bool,
}

pub struct Services {
    pub config: SievrantConfig,
    pub llm: Option<Arc<dyn LlmProvider>>,
    /// Used for query expansion, LLM reranking and synthetic calibration.
    pub fast_llm: Option<Arc<dyn LlmProvider>>,
    pub embedder: Arc<dyn EmbeddingProvider>,
    pub reranker: Arc<dyn Reranker>,
    pub reranker_id: String,
    pub signer: Option<Arc<ReceiptSigner>>,
    pub audit: Option<Arc<AuditLogger>>,
    pub summary: ProviderSummary,
}

impl Services {
    pub fn from_config(config: SievrantConfig) -> Result<Self> {
        // Chat LLM
        let llm_provider = resolve_auto(&config.llm.provider, AUTO_LLM_ORDER, "none");
        let llm = build_llm(&llm_settings(&config.llm, llm_provider.clone()))?;

        // Fast LLM (defaults to the main one)
        let fast_llm = if config.fast_llm.provider.trim().is_empty()
            || config.fast_llm.provider == "auto" && config.fast_llm.model.is_none()
        {
            llm.clone()
        } else {
            let p = resolve_auto(&config.fast_llm.provider, AUTO_LLM_ORDER, "none");
            let mut fast_cfg = config.fast_llm.clone();
            fast_cfg.max_retries = config.llm.max_retries;
            build_llm(&llm_settings(&fast_cfg, p))?
        };

        // Embeddings
        let emb_provider = resolve_auto(&config.embedding.provider, AUTO_EMBEDDING_ORDER, "local");
        if emb_provider == "local" {
            warn!(
                "using the offline 'local' hashing embedder; set SIEVRANT_EMBEDDING_PROVIDER \
                 (or an OPENAI/GEMINI/VOYAGE/... key) for production-quality retrieval"
            );
        }
        let embedder = build_embedder(&EmbeddingSettings {
            provider: emb_provider,
            model: config.embedding.model.clone(),
            api_key: config
                .embedding
                .api_key
                .as_ref()
                .map(|s| s.expose().to_string()),
            base_url: config.embedding.base_url.clone(),
            dimensions: config.embedding.dimensions,
            batch_size: config.embedding.batch_size,
        })?;

        // Reranker
        let (reranker, reranker_id) = Self::build_reranker(&config, &embedder, &fast_llm)?;

        // Receipts and audit log
        let signer = match &config.audit.signing_key {
            Some(k) if !k.is_empty() => {
                Some(Arc::new(ReceiptSigner::from_base64_seed(k.expose())?))
            }
            _ => None,
        };
        let audit = if config.audit.enabled {
            let dir = std::path::Path::new(&config.storage.data_dir).join("audit");
            Some(Arc::new(AuditLogger::new(dir)?))
        } else {
            None
        };

        let summary = ProviderSummary {
            llm: llm
                .as_ref()
                .map(|l| format!("{llm_provider}:{}", l.model_id()))
                .unwrap_or_else(|| "none (evidence-only)".into()),
            fast_llm: fast_llm
                .as_ref()
                .map(|l| l.model_id().to_string())
                .unwrap_or_else(|| "none".into()),
            embedding: embedder.model_id(),
            reranker: reranker_id.clone(),
            vector_backend: config.vector.backend.clone(),
            receipts_signed: signer.is_some(),
        };
        info!(?summary, "providers configured");

        Ok(Self {
            config,
            llm,
            fast_llm,
            embedder,
            reranker,
            reranker_id,
            signer,
            audit,
            summary,
        })
    }

    fn build_reranker(
        config: &SievrantConfig,
        embedder: &Arc<dyn EmbeddingProvider>,
        fast_llm: &Option<Arc<dyn LlmProvider>>,
    ) -> Result<(Arc<dyn Reranker>, String)> {
        let rc = &config.reranker;
        let explicit_key = rc.api_key.as_ref().map(|s| s.expose().to_string());
        let key_for = |env: &str| {
            explicit_key
                .clone()
                .or_else(|| std::env::var(env).ok().filter(|v| !v.trim().is_empty()))
        };

        let mut provider = rc.provider.trim().to_ascii_lowercase();
        if provider.is_empty() || provider == "auto" {
            let env_set = |name: &str| std::env::var(name).is_ok_and(|v| !v.trim().is_empty());
            provider = if env_set("COHERE_API_KEY") {
                "cohere"
            } else if env_set("VOYAGE_API_KEY") {
                "voyage"
            } else if env_set("JINA_API_KEY") {
                "jina"
            } else {
                "embedding"
            }
            .into();
        }

        let model = |default: &str| rc.model.clone().unwrap_or_else(|| default.to_string());
        let need_key = |env: &str, name: &str| {
            key_for(env).ok_or_else(|| {
                SievrantError::Config(ConfigError::MissingField {
                    field: format!("{name} reranker key (SIEVRANT_RERANKER_API_KEY or {env})"),
                })
            })
        };

        let built: (Arc<dyn Reranker>, String) = match provider.as_str() {
            "cohere" => {
                let m = model("rerank-v3.5");
                (
                    Arc::new(HttpReranker::cohere(
                        &need_key("COHERE_API_KEY", "cohere")?,
                        &m,
                    )),
                    format!("cohere:{m}"),
                )
            }
            "jina" => {
                let m = model("jina-reranker-v2-base-multilingual");
                (
                    Arc::new(HttpReranker::jina(&need_key("JINA_API_KEY", "jina")?, &m)),
                    format!("jina:{m}"),
                )
            }
            "voyage" => {
                let m = model("rerank-2");
                (
                    Arc::new(HttpReranker::voyage(
                        &need_key("VOYAGE_API_KEY", "voyage")?,
                        &m,
                    )),
                    format!("voyage:{m}"),
                )
            }
            "custom" => {
                let url = rc
                    .base_url
                    .clone()
                    .ok_or_else(|| config_err("reranker.base_url", "required for custom"))?;
                let m = model("default");
                (
                    Arc::new(HttpReranker::new(
                        "custom",
                        &url,
                        &explicit_key.unwrap_or_default(),
                        &m,
                    )),
                    format!("custom:{url}:{m}"),
                )
            }
            "embedding" => (
                Arc::new(EmbeddingReranker::new(embedder.clone())),
                format!("embedding:{}", embedder.model_id()),
            ),
            "llm" => {
                let llm = fast_llm.clone().ok_or_else(|| {
                    config_err("reranker.provider", "llm requires a configured LLM")
                })?;
                let id = format!("llm:{}", llm.model_id());
                (Arc::new(LlmReranker::new(llm)), id)
            }
            "none" => (Arc::new(NoReranker), "none".into()),
            other => return Err(config_err("reranker.provider", other)),
        };
        Ok(built)
    }

    /// Stable hash of every setting that affects candidate scores.
    pub fn fingerprint(&self) -> String {
        let r = &self.config.retrieval;
        let raw = format!(
            "v1|emb={}|rerank={}|limit={}|topn={}|rrf={}|expand={}|exp_idx={}",
            self.embedder.model_id(),
            self.reranker_id,
            r.search_limit,
            r.rerank_top_n,
            r.rrf_k,
            r.query_expansion && self.fast_llm.is_some(),
            r.experimental_indexes,
        );
        format!("{:x}", Sha256::digest(raw.as_bytes()))[..16].to_string()
    }
}

/// Catalog of providers for `sievrant providers` and `/v1/providers`.
#[derive(Debug, Clone, Serialize)]
pub struct ProviderCatalogEntry {
    pub id: &'static str,
    pub chat: bool,
    pub embeddings: bool,
    pub key_env: Vec<&'static str>,
    pub key_present: bool,
    pub default_chat_model: Option<&'static str>,
    pub default_embedding_model: Option<&'static str>,
}

pub fn provider_catalog() -> Vec<ProviderCatalogEntry> {
    PROVIDERS
        .iter()
        .map(|p| ProviderCatalogEntry {
            id: p.id,
            chat: p.chat != ChatProtocol::Unsupported,
            embeddings: p.embedding_base_url.is_some(),
            key_env: p.key_env.to_vec(),
            key_present: !p.requires_key || resolve_api_key(p, &None).is_some(),
            default_chat_model: p.default_chat_model,
            default_embedding_model: p.default_embedding_model,
        })
        .collect()
}
