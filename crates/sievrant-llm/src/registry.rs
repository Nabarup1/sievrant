//! Provider catalog and builders.
//!
//! Every LLM and embedding backend Sievrant can talk to is listed in
//! [`PROVIDERS`]. Switching providers is a configuration change:
//!
//! ```text
//! SIEVRANT_LLM_PROVIDER=openrouter
//! SIEVRANT_LLM_MODEL=anthropic/claude-opus-5     # optional, has a default
//! OPENROUTER_API_KEY=...                         # or SIEVRANT_LLM_API_KEY
//! ```
//!
//! API keys resolve in order: the explicit `api_key` setting, then the
//! provider's conventional environment variable (e.g. `OPENAI_API_KEY`).

use std::sync::Arc;

use sievrant_core::error::{ConfigError, Result, SievrantError};

use crate::anthropic::AnthropicNative;
use crate::embeddings::{EmbeddingProvider, HashingEmbeddingProvider, OpenAiEmbeddingProvider};
use crate::gemini::GeminiNative;
use crate::openai::OpenAiCompatible;
use crate::provider::LlmProvider;

/// Wire protocol a provider speaks for chat completions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatProtocol {
    OpenAiCompatible,
    Anthropic,
    Gemini,
    /// Provider offers no chat endpoint Sievrant can use.
    Unsupported,
}

/// Static description of a provider.
#[derive(Debug, Clone, Copy)]
pub struct ProviderInfo {
    pub id: &'static str,
    pub chat: ChatProtocol,
    /// Base URL for chat (OpenAI-compatible providers include the `/v1` part).
    pub base_url: &'static str,
    /// Base URL for the OpenAI-compatible `/embeddings` endpoint, if any.
    pub embedding_base_url: Option<&'static str>,
    /// Environment variables checked (in order) for the API key.
    pub key_env: &'static [&'static str],
    pub default_chat_model: Option<&'static str>,
    pub default_embedding_model: Option<&'static str>,
    /// Local servers need no key.
    pub requires_key: bool,
}

/// All supported providers. Defaults can lag new releases; set
/// `SIEVRANT_LLM_MODEL` / `SIEVRANT_EMBEDDING_MODEL` to pick a specific model.
pub const PROVIDERS: &[ProviderInfo] = &[
    ProviderInfo {
        id: "openai",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "https://api.openai.com/v1",
        embedding_base_url: Some("https://api.openai.com/v1"),
        key_env: &["OPENAI_API_KEY"],
        default_chat_model: Some("gpt-5-mini"),
        default_embedding_model: Some("text-embedding-3-small"),
        requires_key: true,
    },
    ProviderInfo {
        id: "anthropic",
        chat: ChatProtocol::Anthropic,
        base_url: "https://api.anthropic.com",
        embedding_base_url: None,
        key_env: &["ANTHROPIC_API_KEY"],
        default_chat_model: Some("claude-opus-5"),
        default_embedding_model: None,
        requires_key: true,
    },
    ProviderInfo {
        id: "gemini",
        chat: ChatProtocol::Gemini,
        base_url: "https://generativelanguage.googleapis.com/v1beta",
        embedding_base_url: Some("https://generativelanguage.googleapis.com/v1beta/openai"),
        key_env: &["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        default_chat_model: Some("gemini-2.5-flash"),
        default_embedding_model: Some("gemini-embedding-001"),
        requires_key: true,
    },
    ProviderInfo {
        id: "openrouter",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "https://openrouter.ai/api/v1",
        embedding_base_url: None,
        key_env: &["OPENROUTER_API_KEY"],
        default_chat_model: Some("openai/gpt-5-mini"),
        default_embedding_model: None,
        requires_key: true,
    },
    ProviderInfo {
        id: "deepseek",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "https://api.deepseek.com/v1",
        embedding_base_url: None,
        key_env: &["DEEPSEEK_API_KEY"],
        default_chat_model: Some("deepseek-chat"),
        default_embedding_model: None,
        requires_key: true,
    },
    ProviderInfo {
        id: "groq",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "https://api.groq.com/openai/v1",
        embedding_base_url: None,
        key_env: &["GROQ_API_KEY"],
        default_chat_model: Some("llama-3.3-70b-versatile"),
        default_embedding_model: None,
        requires_key: true,
    },
    ProviderInfo {
        id: "mistral",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "https://api.mistral.ai/v1",
        embedding_base_url: Some("https://api.mistral.ai/v1"),
        key_env: &["MISTRAL_API_KEY"],
        default_chat_model: Some("mistral-small-latest"),
        default_embedding_model: Some("mistral-embed"),
        requires_key: true,
    },
    ProviderInfo {
        id: "together",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "https://api.together.xyz/v1",
        embedding_base_url: Some("https://api.together.xyz/v1"),
        key_env: &["TOGETHER_API_KEY"],
        default_chat_model: Some("meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        default_embedding_model: Some("BAAI/bge-large-en-v1.5"),
        requires_key: true,
    },
    ProviderInfo {
        id: "fireworks",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "https://api.fireworks.ai/inference/v1",
        embedding_base_url: Some("https://api.fireworks.ai/inference/v1"),
        key_env: &["FIREWORKS_API_KEY"],
        default_chat_model: Some("accounts/fireworks/models/llama-v3p3-70b-instruct"),
        default_embedding_model: Some("nomic-ai/nomic-embed-text-v1.5"),
        requires_key: true,
    },
    ProviderInfo {
        id: "xai",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "https://api.x.ai/v1",
        embedding_base_url: None,
        key_env: &["XAI_API_KEY"],
        default_chat_model: Some("grok-3-mini"),
        default_embedding_model: None,
        requires_key: true,
    },
    ProviderInfo {
        id: "voyage",
        chat: ChatProtocol::Unsupported,
        base_url: "https://api.voyageai.com/v1",
        embedding_base_url: Some("https://api.voyageai.com/v1"),
        key_env: &["VOYAGE_API_KEY"],
        default_chat_model: None,
        default_embedding_model: Some("voyage-3.5"),
        requires_key: true,
    },
    ProviderInfo {
        id: "jina",
        chat: ChatProtocol::Unsupported,
        base_url: "https://api.jina.ai/v1",
        embedding_base_url: Some("https://api.jina.ai/v1"),
        key_env: &["JINA_API_KEY"],
        default_chat_model: None,
        default_embedding_model: Some("jina-embeddings-v3"),
        requires_key: true,
    },
    ProviderInfo {
        id: "ollama",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "http://localhost:11434/v1",
        embedding_base_url: Some("http://localhost:11434/v1"),
        key_env: &["OLLAMA_API_KEY"],
        default_chat_model: Some("llama3.1"),
        default_embedding_model: Some("nomic-embed-text"),
        requires_key: false,
    },
    ProviderInfo {
        id: "lmstudio",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "http://localhost:1234/v1",
        embedding_base_url: Some("http://localhost:1234/v1"),
        key_env: &["LMSTUDIO_API_KEY"],
        default_chat_model: None,
        default_embedding_model: None,
        requires_key: false,
    },
    ProviderInfo {
        id: "custom",
        chat: ChatProtocol::OpenAiCompatible,
        base_url: "",
        embedding_base_url: Some(""),
        key_env: &[],
        default_chat_model: None,
        default_embedding_model: None,
        requires_key: false,
    },
];

/// Look up a provider by id (case-insensitive).
pub fn provider_info(id: &str) -> Option<&'static ProviderInfo> {
    let id = id.trim().to_ascii_lowercase();
    PROVIDERS.iter().find(|p| p.id == id)
}

/// Settings for building an LLM client. Mirrors `sievrant_core::config::LlmConfig`.
#[derive(Debug, Clone, Default)]
pub struct LlmSettings {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub max_retries: u32,
    /// Anthropic `output_config.effort`.
    pub effort: Option<String>,
    /// Anthropic server-side refusal fallbacks for models that support them.
    pub anthropic_fallbacks: bool,
    /// Sent as OpenRouter's `HTTP-Referer` / `X-Title` attribution headers.
    pub app_url: Option<String>,
    pub app_name: Option<String>,
}

/// Settings for building an embedding client.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingSettings {
    pub provider: String,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub dimensions: Option<usize>,
    pub batch_size: usize,
}

fn invalid(field: &str, value: impl Into<String>) -> SievrantError {
    SievrantError::Config(ConfigError::InvalidValue {
        field: field.into(),
        value: value.into(),
    })
}

fn missing(field: &str) -> SievrantError {
    SievrantError::Config(ConfigError::MissingField {
        field: field.into(),
    })
}

fn non_empty(v: &Option<String>) -> Option<String> {
    v.as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Resolve an API key: explicit setting first, then the provider's env vars.
pub fn resolve_api_key(info: &ProviderInfo, explicit: &Option<String>) -> Option<String> {
    non_empty(explicit).or_else(|| {
        info.key_env.iter().find_map(|name| {
            std::env::var(name)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        })
    })
}

fn key_hint(info: &ProviderInfo, setting: &str) -> String {
    let mut names: Vec<&str> = vec![setting];
    names.extend(info.key_env.iter().copied());
    names.join(" or ")
}

/// Build the configured chat LLM. Returns `Ok(None)` for provider `none`,
/// which runs Sievrant in evidence-only mode (no answer generation).
pub fn build_llm(s: &LlmSettings) -> Result<Option<Arc<dyn LlmProvider>>> {
    let id = s.provider.trim().to_ascii_lowercase();
    if id.is_empty() || id == "none" {
        return Ok(None);
    }
    let info = provider_info(&id).ok_or_else(|| invalid("llm.provider", &s.provider))?;
    if info.chat == ChatProtocol::Unsupported {
        return Err(invalid(
            "llm.provider",
            format!("{} (embeddings/rerank only, no chat endpoint)", info.id),
        ));
    }

    let model = non_empty(&s.model)
        .or_else(|| info.default_chat_model.map(str::to_string))
        .ok_or_else(|| missing("llm.model (SIEVRANT_LLM_MODEL)"))?;
    let base_url = non_empty(&s.base_url).unwrap_or_else(|| info.base_url.to_string());
    if base_url.is_empty() {
        return Err(missing("llm.base_url (SIEVRANT_LLM_BASE_URL)"));
    }
    let api_key = resolve_api_key(info, &s.api_key).unwrap_or_default();
    if info.requires_key && api_key.is_empty() {
        return Err(missing(&format!(
            "API key for provider '{}' ({})",
            info.id,
            key_hint(info, "SIEVRANT_LLM_API_KEY")
        )));
    }
    let retries = s.max_retries;

    let provider: Arc<dyn LlmProvider> = match info.chat {
        ChatProtocol::Anthropic => Arc::new(
            AnthropicNative::new(&api_key, &model)
                .with_base_url(&base_url)
                .with_retries(retries)
                .with_effort(s.effort.clone())
                .with_fallbacks(s.anthropic_fallbacks),
        ),
        ChatProtocol::Gemini => Arc::new(
            GeminiNative::new(&api_key, &model)
                .with_base_url(&base_url)
                .with_retries(retries),
        ),
        ChatProtocol::OpenAiCompatible => {
            let mut client = OpenAiCompatible::custom(&base_url, &api_key, &model, 128_000)
                .named(info.id)
                .with_retries(retries);
            if info.id == "openrouter" {
                if let Some(url) = non_empty(&s.app_url) {
                    client = client.with_header("HTTP-Referer", &url);
                }
                client = client.with_header(
                    "X-Title",
                    &non_empty(&s.app_name).unwrap_or_else(|| "Sievrant".into()),
                );
            }
            Arc::new(client)
        }
        ChatProtocol::Unsupported => unreachable!(),
    };
    Ok(Some(provider))
}

/// Build the configured embedding provider.
///
/// Provider `local` is a deterministic lexical hashing embedder that needs no
/// network or key. It is meant for offline development, tests and
/// reproducible benchmarks, not for production retrieval quality.
pub fn build_embedder(s: &EmbeddingSettings) -> Result<Arc<dyn EmbeddingProvider>> {
    let id = s.provider.trim().to_ascii_lowercase();
    if id == "local" || id == "hash" {
        return Ok(Arc::new(HashingEmbeddingProvider::new(
            s.dimensions.unwrap_or(1024),
        )));
    }
    let info = provider_info(&id).ok_or_else(|| invalid("embedding.provider", &s.provider))?;
    let base_url = non_empty(&s.base_url)
        .or_else(|| info.embedding_base_url.map(str::to_string))
        .filter(|u| !u.is_empty())
        .ok_or_else(|| {
            invalid(
                "embedding.provider",
                format!(
                    "{} has no embeddings endpoint; choose openai, gemini, mistral, voyage, jina, together, fireworks, ollama, lmstudio, custom or local",
                    info.id
                ),
            )
        })?;
    let model = non_empty(&s.model)
        .or_else(|| info.default_embedding_model.map(str::to_string))
        .ok_or_else(|| missing("embedding.model (SIEVRANT_EMBEDDING_MODEL)"))?;
    let api_key = resolve_api_key(info, &s.api_key).unwrap_or_default();
    if info.requires_key && api_key.is_empty() {
        return Err(missing(&format!(
            "API key for embedding provider '{}' ({})",
            info.id,
            key_hint(info, "SIEVRANT_EMBEDDING_API_KEY")
        )));
    }
    let batch = if s.batch_size == 0 { 128 } else { s.batch_size };
    Ok(Arc::new(
        OpenAiEmbeddingProvider::custom(base_url, api_key, model)
            .named(info.id)
            .with_batch_size(batch)
            .with_dimensions(s.dimensions)
            .with_float_encoding(info.id == "openai"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_provider_id_is_unique_and_lowercase() {
        let mut ids: Vec<&str> = PROVIDERS.iter().map(|p| p.id).collect();
        assert!(ids.iter().all(|id| *id == id.to_ascii_lowercase()));
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), PROVIDERS.len());
    }

    #[test]
    fn none_provider_means_no_llm() {
        let s = LlmSettings {
            provider: "none".into(),
            ..Default::default()
        };
        assert!(build_llm(&s).unwrap().is_none());
    }

    #[test]
    fn explicit_key_builds_each_chat_provider_with_its_default_model() {
        for p in PROVIDERS {
            if p.chat == ChatProtocol::Unsupported || p.default_chat_model.is_none() {
                continue;
            }
            let s = LlmSettings {
                provider: p.id.to_uppercase(),
                api_key: Some("test-key".into()),
                ..Default::default()
            };
            let llm = build_llm(&s).unwrap().unwrap();
            assert_eq!(llm.model_id(), p.default_chat_model.unwrap(), "{}", p.id);
        }
    }

    #[test]
    fn missing_key_is_a_clear_error() {
        let s = LlmSettings {
            provider: "xai".into(),
            ..Default::default()
        };
        // XAI_API_KEY is not set in the test environment.
        if std::env::var("XAI_API_KEY").is_err() {
            let err = build_llm(&s).err().unwrap().to_string();
            assert!(err.contains("XAI_API_KEY"), "{err}");
        }
    }

    #[test]
    fn custom_provider_requires_base_url_and_model() {
        let s = LlmSettings {
            provider: "custom".into(),
            model: Some("m".into()),
            ..Default::default()
        };
        assert!(build_llm(&s).is_err());
        let s = LlmSettings {
            provider: "custom".into(),
            model: Some("m".into()),
            base_url: Some("http://localhost:8000/v1".into()),
            ..Default::default()
        };
        assert!(build_llm(&s).unwrap().is_some());
    }

    #[test]
    fn embedding_only_providers_reject_chat() {
        let s = LlmSettings {
            provider: "voyage".into(),
            api_key: Some("k".into()),
            ..Default::default()
        };
        assert!(build_llm(&s).is_err());
    }

    #[test]
    fn chat_only_providers_reject_embeddings() {
        let s = EmbeddingSettings {
            provider: "anthropic".into(),
            api_key: Some("k".into()),
            ..Default::default()
        };
        let err = build_embedder(&s).err().unwrap().to_string();
        assert!(err.contains("no embeddings endpoint"), "{err}");
    }

    #[test]
    fn local_embedder_needs_nothing() {
        let s = EmbeddingSettings {
            provider: "local".into(),
            ..Default::default()
        };
        assert_eq!(build_embedder(&s).unwrap().dimensions(), 1024);
    }
}
