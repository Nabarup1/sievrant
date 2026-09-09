use std::time::Duration;

use reqwest::StatusCode;
use serde_json::Value;
use tracing::warn;

use sievrant_core::error::{LlmError, Result, SievrantError};

/// Shared, resilient HTTP client used by all LLM and embedding provider implementations.
///
/// Handles 429 (rate limit) and 5xx (server error) responses with full
/// exponential backoff plus jitter. The jitter prevents thundering-herd
/// problems when many concurrent queries hit a rate-limited endpoint simultaneously.
pub struct ResilientHttpClient {
    inner: reqwest::Client,
    max_retries: u32,
    /// Base delay in milliseconds before applying exponential growth.
    base_backoff_ms: u64,
}

impl ResilientHttpClient {
    pub fn new(max_retries: u32, base_backoff_ms: u64) -> Self {
        let inner = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("failed to build reqwest client");
        Self { inner, max_retries, base_backoff_ms }
    }

    /// POST a JSON body to `url`, retrying on 429/5xx with exponential backoff.
    /// On success, returns the parsed JSON response body.
    pub async fn post_json(&self, url: &str, body: &Value, bearer_token: &str) -> Result<Value> {
        let mut attempt = 0u32;

        loop {
            let resp = self
                .inner
                .post(url)
                .bearer_auth(bearer_token)
                .json(body)
                .send()
                .await
                .map_err(|e| SievrantError::Llm(LlmError::Http(e.to_string())))?;

            let status = resp.status();

            if status.is_success() {
                let json = resp
                    .json::<Value>()
                    .await
                    .map_err(|e| SievrantError::Llm(LlmError::ParseError { provider: "http".into(), reason: e.to_string() }))?;
                return Ok(json);
            }

            // 429 or 5xx -- check retry budget and back off
            let retryable = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
            if !retryable || attempt >= self.max_retries {
                let body_text = resp.text().await.unwrap_or_default();
                return Err(SievrantError::Llm(LlmError::ApiError {
                    provider: "http".into(),
                    status: status.as_u16(),
                    body: body_text,
                }));
            }

            // Exponential backoff: base * 2^attempt, capped at 30 s
            let delay_ms = (self.base_backoff_ms * (1 << attempt)).min(30_000)
                + (rand_jitter() % 200);
            warn!(
                attempt,
                delay_ms,
                %status,
                "retrying request after transient error"
            );
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            attempt += 1;
        }
    }

    /// GET with Bearer auth, retrying on transient errors.
    pub async fn get_json(&self, url: &str, bearer_token: &str) -> Result<Value> {
        let mut attempt = 0u32;

        loop {
            let resp = self
                .inner
                .get(url)
                .bearer_auth(bearer_token)
                .send()
                .await
                .map_err(|e| SievrantError::Llm(LlmError::Http(e.to_string())))?;

            let status = resp.status();
            if status.is_success() {
                return resp
                    .json::<Value>()
                    .await
                    .map_err(|e| SievrantError::Llm(LlmError::ParseError { provider: "http".into(), reason: e.to_string() }));
            }

            let retryable = status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
            if !retryable || attempt >= self.max_retries {
                let text = resp.text().await.unwrap_or_default();
                return Err(SievrantError::Llm(LlmError::ApiError {
                    provider: "http".into(),
                    status: status.as_u16(),
                    body: text,
                }));
            }

            let delay_ms = (self.base_backoff_ms * (1 << attempt)).min(30_000)
                + (rand_jitter() % 200);
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            attempt += 1;
        }
    }
}

/// Simple non-crypto jitter using bit tricks to avoid pulling in a rand crate.
fn rand_jitter() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    t as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn client_constructs_with_defaults() {
        let c = ResilientHttpClient::new(3, 200);
        assert_eq!(c.max_retries, 3);
        assert_eq!(c.base_backoff_ms, 200);
    }
}
