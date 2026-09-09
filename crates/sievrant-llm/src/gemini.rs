use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use sievrant_core::error::{LlmError, Result};

use crate::provider::{FinishReason, LlmProvider, LlmRequest, LlmResponse, Role, TokenUsage};

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Native Gemini generateContent API adapter.
pub struct GeminiNative {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl GeminiNative {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }
}

#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiSystemInstruction>,
    #[serde(rename = "generationConfig")]
    generation_config: GeminiGenerationConfig,
}

#[derive(Serialize)]
struct GeminiContent {
    role: String,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Serialize)]
struct GeminiSystemInstruction {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    max_output_tokens: usize,
    temperature: f32,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsage>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: GeminiContent2,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct GeminiContent2 {
    parts: Vec<GeminiPart2>,
}

#[derive(Deserialize)]
struct GeminiPart2 {
    text: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsage {
    prompt_token_count: Option<u32>,
    candidates_token_count: Option<u32>,
}

#[async_trait]
impl LlmProvider for GeminiNative {
    async fn complete(&self, req: LlmRequest) -> Result<LlmResponse> {
        let url = format!(
            "{}/{}:generateContent?key={}",
            GEMINI_API_BASE, self.model, self.api_key
        );

        let contents: Vec<GeminiContent> = req
            .messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| GeminiContent {
                role: match m.role {
                    Role::User => "user".into(),
                    Role::Assistant => "model".into(),
                    Role::System => "user".into(),
                },
                parts: vec![GeminiPart { text: m.content.clone() }],
            })
            .collect();

        let system_instruction = req.system_prompt.as_ref().map(|s| GeminiSystemInstruction {
            parts: vec![GeminiPart { text: s.clone() }],
        });

        let body = GeminiRequest {
            contents,
            system_instruction,
            generation_config: GeminiGenerationConfig {
                max_output_tokens: req.max_tokens,
                temperature: req.temperature,
            },
        };

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Http(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(LlmError::ApiError {
                provider: "gemini".into(),
                status: status.as_u16(),
                body: body_text,
            }
            .into());
        }

        let parsed: GeminiResponse = response.json().await.map_err(|e| {
            LlmError::ParseError {
                provider: "gemini".into(),
                reason: e.to_string(),
            }
        })?;

        let candidate = parsed
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::ParseError {
                provider: "gemini".into(),
                reason: "empty candidates".into(),
            })?;

        let content = candidate
            .content
            .parts
            .into_iter()
            .filter_map(|p| p.text)
            .collect::<Vec<_>>()
            .join("");

        let finish_reason = match candidate.finish_reason.as_deref() {
            Some("STOP") => FinishReason::Stop,
            Some("MAX_TOKENS") => FinishReason::Length,
            Some("SAFETY") => FinishReason::ContentFilter,
            _ => FinishReason::Unknown,
        };

        let usage = parsed.usage_metadata.map(|u| TokenUsage {
            input_tokens: u.prompt_token_count.unwrap_or(0),
            output_tokens: u.candidates_token_count.unwrap_or(0),
        }).unwrap_or_default();

        Ok(LlmResponse {
            content,
            usage,
            model: self.model.clone(),
            finish_reason,
        })
    }

    fn supports_vision(&self) -> bool {
        self.model.contains("vision") || self.model.contains("pro")
    }

    fn context_window(&self) -> usize {
        1_000_000
    }

    fn model_id(&self) -> &str {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_response_parses_correctly() {
        let json_str = r#"{
            "candidates": [{
                "content": {"parts": [{"text": "The lease term is 60 months."}], "role": "model"},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 12}
        }"#;
        let parsed: GeminiResponse = serde_json::from_str(json_str).unwrap();
        let text: String = parsed.candidates[0]
            .content
            .parts
            .iter()
            .filter_map(|p| p.text.as_deref())
            .collect();
        assert_eq!(text, "The lease term is 60 months.");
        assert_eq!(parsed.candidates[0].finish_reason.as_deref(), Some("STOP"));
        assert_eq!(parsed.usage_metadata.unwrap().prompt_token_count, Some(40));
    }
}
