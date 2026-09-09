use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use sievrant_core::error::{LlmError, Result};

use crate::provider::{FinishReason, LlmProvider, LlmRequest, LlmResponse, Role, TokenUsage};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Native Anthropic Messages API adapter.
/// System prompt is sent as a top-level field, not inside the messages array.
pub struct AnthropicNative {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl AnthropicNative {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: usize,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<AnthropicMessage<'a>>,
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
    model: String,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[async_trait]
impl LlmProvider for AnthropicNative {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse> {
        let messages: Vec<AnthropicMessage> = req
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| AnthropicMessage {
                role: match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "user",
                },
                content: &m.content,
            })
            .collect();

        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            system: req.system_prompt.as_deref(),
            messages,
        };

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                provider: "anthropic".into(),
                status: status.as_u16(),
                body: body_text,
            }
            .into());
        }

        let parsed: AnthropicResponse = response.json().await.map_err(|e| {
            LlmError::ParseError {
                provider: "anthropic".into(),
                reason: e.to_string(),
            }
        })?;

        let content = parsed
            .content
            .into_iter()
            .next()
            .map(|c| c.text)
            .unwrap_or_default();

        let finish_reason = match parsed.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::Length,
            _ => FinishReason::Unknown,
        };

        Ok(LlmResponse {
            content,
            usage: TokenUsage {
                input_tokens: parsed.usage.input_tokens,
                output_tokens: parsed.usage.output_tokens,
            },
            model: parsed.model,
            finish_reason,
        })
    }

    fn supports_vision(&self) -> bool {
        self.model.contains("claude-3")
    }

    fn context_window(&self) -> usize {
        200_000
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anthropic_response_parses_correctly() {
        let json_str = r#"{
            "content": [{"type": "text", "text": "The answer is 42."}],
            "usage": {"input_tokens": 30, "output_tokens": 8},
            "model": "claude-opus-4-5",
            "stop_reason": "end_turn"
        }"#;
        let parsed: AnthropicResponse = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed.content[0].text, "The answer is 42.");
        assert_eq!(parsed.usage.input_tokens, 30);
        assert_eq!(parsed.stop_reason.as_deref(), Some("end_turn"));
    }
}
