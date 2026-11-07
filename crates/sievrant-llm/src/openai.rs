use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;

use sievrant_core::error::{LlmError, Result};

use crate::client::send_with_retry;
use crate::provider::{FinishReason, LlmProvider, LlmRequest, LlmResponse, Role, TokenUsage};

/// OpenAI-compatible `/chat/completions` adapter. Works with OpenAI,
/// OpenRouter, DeepSeek, Groq, Mistral, Together, Fireworks, xAI, Ollama,
/// LM Studio, vLLM and any other gateway exposing the same endpoint.
pub struct OpenAiCompatible {
    client: reqwest::Client,
    provider: String,
    base_url: String,
    api_key: String,
    model: String,
    context_window: usize,
    max_retries: u32,
    extra_headers: Vec<(String, String)>,
}

impl OpenAiCompatible {
    pub fn openai(api_key: &str, model: &str) -> Self {
        Self::custom("https://api.openai.com/v1", api_key, model, 128_000).named("openai")
    }

    pub fn deepseek(api_key: &str, model: &str) -> Self {
        Self::custom("https://api.deepseek.com/v1", api_key, model, 64_000).named("deepseek")
    }

    pub fn custom(base_url: &str, api_key: &str, model: &str, context_window: usize) -> Self {
        Self {
            client: reqwest::Client::new(),
            provider: "openai-compatible".into(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            context_window,
            max_retries: 3,
            extra_headers: Vec::new(),
        }
    }

    /// Provider name used in logs and error messages.
    pub fn named(mut self, provider: &str) -> Self {
        self.provider = provider.to_string();
        self
    }

    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Extra headers, e.g. OpenRouter's `HTTP-Referer` / `X-Title`.
    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        self.extra_headers
            .push((name.to_string(), value.to_string()));
        self
    }

    /// OpenAI's reasoning models (o-series, gpt-5 family) reject `max_tokens`
    /// and any temperature other than the default.
    fn is_openai_reasoning_model(&self) -> bool {
        let m = self.model.as_str();
        self.base_url.contains("api.openai.com") && (m.starts_with('o') || m.starts_with("gpt-5"))
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

        if self.is_openai_reasoning_model() {
            json!({
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": req.max_tokens,
            })
        } else {
            json!({
                "model": self.model,
                "messages": messages,
                "max_tokens": req.max_tokens,
                "temperature": req.temperature,
            })
        }
    }
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
    #[serde(default)]
    model: Option<String>,
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

        let response = send_with_retry(&self.provider, self.max_retries, || {
            let mut rb = self.client.post(&url).json(&body);
            if !self.api_key.is_empty() {
                rb = rb.bearer_auth(&self.api_key);
            }
            for (k, v) in &self.extra_headers {
                rb = rb.header(k.as_str(), v.as_str());
            }
            rb
        })
        .await?;

        let parsed: OpenAiResponse = response.json().await.map_err(|e| LlmError::ParseError {
            provider: self.provider.clone(),
            reason: e.to_string(),
        })?;

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ParseError {
                provider: self.provider.clone(),
                reason: "empty choices array".into(),
            })?;

        let finish_reason = match choice.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Unknown,
        };

        let usage = parsed
            .usage
            .map(|u| TokenUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
            })
            .unwrap_or_default();

        Ok(LlmResponse {
            content: choice.message.content.unwrap_or_default(),
            usage,
            model: parsed.model.unwrap_or_else(|| self.model.clone()),
            finish_reason,
        })
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

    fn req() -> LlmRequest {
        LlmRequest {
            messages: vec![crate::provider::Message {
                role: Role::User,
                content: "Hello".into(),
            }],
            max_tokens: 100,
            temperature: 0.0,
            system_prompt: Some("You are an assistant.".into()),
        }
    }

    #[test]
    fn request_body_includes_system_prompt_as_first_message() {
        let adapter = OpenAiCompatible::openai("sk-test", "gpt-4o");
        let body = adapter.build_request_body(&req());
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are an assistant.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(body["max_tokens"], 100);
    }

    #[test]
    fn openai_reasoning_models_use_max_completion_tokens() {
        let body = OpenAiCompatible::openai("k", "gpt-5-mini").build_request_body(&req());
        assert_eq!(body["max_completion_tokens"], 100);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn other_gateways_keep_standard_params_for_same_model_name() {
        let body = OpenAiCompatible::custom("https://openrouter.ai/api/v1", "k", "gpt-5-mini", 8)
            .build_request_body(&req());
        assert_eq!(body["max_tokens"], 100);
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
        assert_eq!(
            parsed.choices[0].message.content.as_deref(),
            Some("Answer text")
        );
        assert_eq!(parsed.usage.unwrap().prompt_tokens, 20);
    }
}
