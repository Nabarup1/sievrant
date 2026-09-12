use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use sievrant_core::error::{LlmError, Result};

use crate::provider::{FinishReason, LlmProvider, LlmRequest, LlmResponse, Role, TokenUsage};

/// OpenAI-compatible adapter. Works with OpenAI, DeepSeek, GLM, local vllm,
/// and any gateway that exposes the standard /v1/chat/completions endpoint.
pub struct OpenAiCompatible {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
    context_window: usize,
}

impl OpenAiCompatible {
    pub fn openai(api_key: &str, model: &str) -> Self {
        Self::custom("https://api.openai.com/v1", api_key, model, 128_000)
    }

    pub fn deepseek(api_key: &str, model: &str) -> Self {
        Self::custom("https://api.deepseek.com/v1", api_key, model, 64_000)
    }

    pub fn custom(base_url: &str, api_key: &str, model: &str, context_window: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            context_window,
        }
    }

    fn build_request_body(&self, req: &LlmRequest) -> serde_json::Value {
        let mut messages: Vec<serde_json::Value> = Vec::new();

        if let Some(system) = &req.system_prompt {
            messages.push(json!({ "role": "system", "content": system }));
        }

        for msg in &req.messages {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            messages.push(json!({ "role": role, "content": msg.content }));
        }

        json!({
            "model": self.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        })
    }
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
    model: String,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[async_trait]
impl LlmProvider for OpenAiCompatible {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = self.build_request_body(&req);

        // Exponential backoff: 3 attempts for rate limits and server errors.
        let max_attempts = 3u32;
        let mut attempt = 0u32;

        loop {
            attempt += 1;
            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .map_err(|e| LlmError::Http(e.to_string()))?;

            let status = response.status();

            if status == 429 {
                if attempt < max_attempts {
                    let backoff_ms = 500u64 * (1 << (attempt - 1));
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    continue;
                }
                return Err(LlmError::RateLimited {
                    provider: "openai-compatible".into(),
                    retry_after_ms: 1000,
                }
                .into());
            }

            if status.is_server_error() {
                if attempt < max_attempts {
                    let backoff_ms = 500u64 * (1 << (attempt - 1));
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    continue;
                }
                let body_text = response.text().await.unwrap_or_default();
                return Err(LlmError::ApiError {
                    provider: "openai-compatible".into(),
                    status: status.as_u16(),
                    body: body_text,
                }
                .into());
            }

            if !status.is_success() {
                let body_text = response.text().await.unwrap_or_default();
                return Err(LlmError::ApiError {
                    provider: "openai-compatible".into(),
                    status: status.as_u16(),
                    body: body_text,
                }
                .into());
            }

            let parsed: OpenAiResponse = response.json().await.map_err(|e| {
                LlmError::ParseError {
                    provider: "openai-compatible".into(),
                    reason: e.to_string(),
                }
            })?;

            let choice = parsed
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| LlmError::ParseError {
                    provider: "openai-compatible".into(),
                    reason: "empty choices array".into(),
                })?;

            let finish_reason = match choice.finish_reason.as_deref() {
                Some("stop") => FinishReason::Stop,
                Some("length") => FinishReason::Length,
                Some("content_filter") => FinishReason::ContentFilter,
                _ => FinishReason::Unknown,
            };

            let usage = parsed.usage.map(|u| TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
            }).unwrap_or_default();

            return Ok(LlmResponse {
                content: choice.message.content.unwrap_or_default(),
                usage,
                model: parsed.model,
                finish_reason,
            });
        }
    }

    fn supports_vision(&self) -> bool {
        self.model.contains("vision") || self.model.contains("gpt-4o")
    }

    fn context_window(&self) -> usize {
        self.context_window
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_includes_system_prompt_as_first_message() {
        let adapter = OpenAiCompatible::openai("sk-test", "gpt-4o");
        let req = LlmRequest {
            messages: vec![crate::provider::Message {
                role: Role::User,
                content: "Hello".into(),
            }],
            max_tokens: 100,
            temperature: 0.0,
            system_prompt: Some("You are an assistant.".into()),
        };
        let body = adapter.build_request_body(&req);
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are an assistant.");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn openai_response_parses_correctly() {
        let json_str = r#"{
            "choices": [{"message": {"content": "Answer text", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            "model": "gpt-4o"
        }"#;
        let parsed: OpenAiResponse = serde_json::from_str(json_str).unwrap();
        assert_eq!(parsed.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(parsed.choices[0].message.content.as_deref(), Some("Answer text"));
        assert_eq!(parsed.usage.unwrap().prompt_tokens, 20);
    }
}
