use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use sievrant_core::error::{ConfigError, Result, SievrantError};

/// Contract for all LLM providers. Implementations are cheaply cloneable
/// via Arc so they can be shared across tokio tasks.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse>;
    fn supports_vision(&self) -> bool;
    fn context_window(&self) -> usize;
    fn model_id(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub messages: Vec<Message>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: String,
    pub usage: TokenUsage,
    pub model: String,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Default)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    Unknown,
}

/// Configuration for LLM provider selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub provider: String,
    pub api_key: String,
    pub model: String,
    pub context_window: Option<usize>,
    pub base_url: Option<String>,
}

/// Instantiate an LLM provider from configuration.
///
/// Thin wrapper over [`crate::registry::build_llm`] kept for API compatibility;
/// accepts every provider id in [`crate::registry::PROVIDERS`].
pub fn create_provider(config: &LlmConfig) -> Result<Box<dyn LlmProvider>> {
    let settings = crate::registry::LlmSettings {
        provider: config.provider.clone(),
        model: Some(config.model.clone()),
        api_key: Some(config.api_key.clone()),
        base_url: config.base_url.clone(),
        max_retries: 3,
        anthropic_fallbacks: true,
        ..Default::default()
    };
    match crate::registry::build_llm(&settings)? {
        Some(p) => Ok(Box::new(ArcProvider(p))),
        None => Err(SievrantError::Config(ConfigError::InvalidValue {
            field: "llm.provider".into(),
            value: config.provider.clone(),
        })),
    }
}

struct ArcProvider(std::sync::Arc<dyn LlmProvider>);

#[async_trait]
impl LlmProvider for ArcProvider {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse> {
        self.0.complete(req).await
    }
    fn supports_vision(&self) -> bool {
        self.0.supports_vision()
    }
    fn context_window(&self) -> usize {
        self.0.context_window()
    }
    fn model_id(&self) -> &str {
        self.0.model_id()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn openai_config() -> LlmConfig {
        LlmConfig {
            provider: "openai".into(),
            api_key: "sk-test".into(),
            model: "gpt-4o".into(),
            context_window: None,
            base_url: None,
        }
    }

    #[test]
    fn unknown_provider_returns_error() {
        let cfg = LlmConfig {
            provider: "unknown_model_xyz".into(),
            api_key: "key".into(),
            model: "m".into(),
            context_window: None,
            base_url: None,
        };
        assert!(create_provider(&cfg).is_err());
    }

    #[test]
    fn openai_provider_created_successfully() {
        let cfg = openai_config();
        let provider = create_provider(&cfg);
        assert!(provider.is_ok());
        assert_eq!(provider.unwrap().model_id(), "gpt-4o");
    }

    #[test]
    fn custom_provider_requires_base_url() {
        let cfg = LlmConfig {
            provider: "custom".into(),
            api_key: "key".into(),
            model: "llama3".into(),
            context_window: Some(8192),
            base_url: None,
        };
        assert!(create_provider(&cfg).is_err());
    }
}
