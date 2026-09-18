use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use sievrant_core::error::{ConfigError, Result, SievrantError};

use crate::{anthropic::AnthropicNative, gemini::GeminiNative, openai::OpenAiCompatible};

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
pub fn create_provider(config: &LlmConfig) -> Result<Box<dyn LlmProvider>> {
    match config.provider.as_str() {
        "openai" => Ok(Box::new(OpenAiCompatible::openai(
            &config.api_key,
            &config.model,
        ))),
        "anthropic" => Ok(Box::new(AnthropicNative::new(
            &config.api_key,
            &config.model,
        ))),
        "gemini" => Ok(Box::new(GeminiNative::new(&config.api_key, &config.model))),
        "deepseek" => Ok(Box::new(OpenAiCompatible::deepseek(
            &config.api_key,
            &config.model,
        ))),
        "custom" => {
            let base_url = config.base_url.as_deref().ok_or_else(|| {
                SievrantError::Config(ConfigError::MissingField {
                    field: "llm.base_url".into(),
                })
            })?;
            Ok(Box::new(OpenAiCompatible::custom(
                base_url,
                &config.api_key,
                &config.model,
                config.context_window.unwrap_or(8192),
            )))
        }
        other => Err(SievrantError::Config(ConfigError::InvalidValue {
            field: "llm.provider".into(),
            value: other.into(),
        })),
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
