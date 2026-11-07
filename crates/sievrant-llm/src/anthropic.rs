use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use sievrant_core::error::{LlmError, Result};

use crate::client::send_with_retry;
use crate::provider::{FinishReason, LlmProvider, LlmRequest, LlmResponse, Role, TokenUsage};

pub const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";

/// Native Anthropic Messages API adapter (raw HTTP; Rust has no official SDK).
///
/// Current models (Opus 5, Sonnet 5, Opus 4.7/4.8, Fable) reject sampling
/// parameters and think adaptively by default, so this adapter:
/// - sends `temperature` only to older models that still accept it,
/// - concatenates `text` content blocks and skips `thinking` blocks,
/// - maps the `refusal` stop reason to [`FinishReason::ContentFilter`],
/// - opts `claude-opus-5` into server-side refusal fallbacks unless disabled.
pub struct AnthropicNative {
    client: reqwest::Client,
    api_key: String,
    model: String,
    url: String,
    max_retries: u32,
    effort: Option<String>,
    fallbacks: bool,
}

impl AnthropicNative {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
            model: model.to_string(),
            url: ANTHROPIC_API_URL.to_string(),
            max_retries: 3,
            effort: None,
            fallbacks: true,
        }
    }

    /// Point at a different endpoint (e.g. a gateway exposing the Messages API).
    pub fn with_base_url(mut self, base_url: &str) -> Self {
        let base = base_url.trim_end_matches('/');
        self.url = if base.ends_with("/messages") {
            base.to_string()
        } else if base.ends_with("/v1") {
            format!("{base}/messages")
        } else {
            format!("{base}/v1/messages")
        };
        self
    }

    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// `output_config.effort`: low | medium | high | xhigh | max.
    pub fn with_effort(mut self, effort: Option<String>) -> Self {
        self.effort = effort.filter(|e| !e.is_empty());
        self
    }

    /// Enable or disable server-side refusal fallbacks (`fallbacks: "default"`).
    pub fn with_fallbacks(mut self, enabled: bool) -> Self {
        self.fallbacks = enabled;
        self
    }

    /// Older models accept `temperature`; current ones return a 400 for it.
    /// Unknown (newer) models are treated as not accepting it.
    fn accepts_temperature(&self) -> bool {
        const LEGACY: &[&str] = &[
            "claude-3",
            "claude-haiku-4-5",
            "claude-sonnet-4-5",
            "claude-sonnet-4-6",
            "claude-opus-4-5",
            "claude-opus-4-6",
            "claude-opus-4-1",
            "claude-opus-4-0",
            "claude-sonnet-4-0",
        ];
        LEGACY.iter().any(|p| self.model.starts_with(p))
    }

    fn uses_fallbacks(&self) -> bool {
        self.fallbacks && self.model == "claude-opus-5"
    }

    fn build_body(&self, req: &LlmRequest) -> Value {
        let messages: Vec<Value> = req
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| {
                let role = if m.role == Role::Assistant {
                    "assistant"
                } else {
                    "user"
                };
                json!({ "role": role, "content": m.content })
            })
            .collect();

        let mut body = json!({
            "model": self.model,
            "max_tokens": req.max_tokens,
            "messages": messages,
        });
        if let Some(system) = &req.system_prompt {
            body["system"] = json!(system);
        }
        if self.accepts_temperature() {
            body["temperature"] = json!(req.temperature);
        }
        if let Some(effort) = &self.effort {
            body["output_config"] = json!({ "effort": effort });
        }
        if self.uses_fallbacks() {
            body["fallbacks"] = json!("default");
        }
        body
    }
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
    usage: AnthropicUsage,
    model: String,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

fn map_stop_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn") | Some("stop_sequence") => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("refusal") => FinishReason::ContentFilter,
        _ => FinishReason::Unknown,
    }
}

#[async_trait]
impl LlmProvider for AnthropicNative {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse> {
        let body = self.build_body(&req);
        let fallbacks = self.uses_fallbacks();

        let response = send_with_retry("anthropic", self.max_retries, || {
            let mut rb = self
                .client
                .post(&self.url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_VERSION)
                .json(&body);
            if fallbacks {
                rb = rb.header("anthropic-beta", FALLBACK_BETA);
            }
            rb
        })
        .await?;

        let parsed: AnthropicResponse =
            response.json().await.map_err(|e| LlmError::ParseError {
                provider: "anthropic".into(),
                reason: e.to_string(),
            })?;

        let content = parsed
            .content
            .iter()
            .filter(|b| b.kind == "text")
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join("");

        Ok(LlmResponse {
            content,
            usage: TokenUsage {
                input_tokens: parsed.usage.input_tokens,
                output_tokens: parsed.usage.output_tokens,
            },
            model: parsed.model,
            finish_reason: map_stop_reason(parsed.stop_reason.as_deref()),
        })
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn context_window(&self) -> usize {
        if self.model.starts_with("claude-haiku") || self.model.starts_with("claude-3") {
            200_000
        } else {
            1_000_000
        }
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Message;

    fn req() -> LlmRequest {
        LlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: "hi".into(),
            }],
            max_tokens: 64,
            temperature: 0.0,
            system_prompt: Some("sys".into()),
        }
    }

    #[test]
    fn response_skips_thinking_blocks_and_joins_text() {
        let json_str = r#"{
            "content": [
                {"type": "thinking", "thinking": "", "signature": "x"},
                {"type": "text", "text": "The answer "},
                {"type": "text", "text": "is 42."}
            ],
            "usage": {"input_tokens": 30, "output_tokens": 8},
            "model": "claude-opus-5",
            "stop_reason": "end_turn"
        }"#;
        let parsed: AnthropicResponse = serde_json::from_str(json_str).unwrap();
        let text: String = parsed
            .content
            .iter()
            .filter(|b| b.kind == "text")
            .filter_map(|b| b.text.as_deref())
            .collect();
        assert_eq!(text, "The answer is 42.");
    }

    #[test]
    fn refusal_maps_to_content_filter() {
        assert_eq!(
            map_stop_reason(Some("refusal")),
            FinishReason::ContentFilter
        );
        assert_eq!(map_stop_reason(Some("max_tokens")), FinishReason::Length);
    }

    #[test]
    fn current_models_omit_temperature_and_opt_into_fallbacks() {
        let body = AnthropicNative::new("k", "claude-opus-5").build_body(&req());
        assert!(body.get("temperature").is_none());
        assert_eq!(body["fallbacks"], "default");
        assert_eq!(body["system"], "sys");
    }

    #[test]
    fn legacy_models_keep_temperature_and_skip_fallbacks() {
        let body = AnthropicNative::new("k", "claude-haiku-4-5").build_body(&req());
        assert_eq!(body["temperature"], 0.0);
        assert!(body.get("fallbacks").is_none());
    }

    #[test]
    fn fallbacks_can_be_disabled_and_effort_set() {
        let body = AnthropicNative::new("k", "claude-opus-5")
            .with_fallbacks(false)
            .with_effort(Some("low".into()))
            .build_body(&req());
        assert!(body.get("fallbacks").is_none());
        assert_eq!(body["output_config"]["effort"], "low");
    }

    #[test]
    fn base_url_is_normalised() {
        let a = AnthropicNative::new("k", "m").with_base_url("https://gw.example.com/");
        assert_eq!(a.url, "https://gw.example.com/v1/messages");
        let b = AnthropicNative::new("k", "m").with_base_url("https://gw.example.com/v1");
        assert_eq!(b.url, "https://gw.example.com/v1/messages");
    }
}
