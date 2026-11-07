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
        Self {
            inner,
            max_retries,
            base_backoff_ms,
        }
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
                .map_err(|e| SievrantError::Llm(LlmError::Http(e.without_url().to_string())))?;

            let status = resp.status();

            if status.is_success() {
                let json = resp.json::<Value>().await.map_err(|e| {
                    SievrantError::Llm(LlmError::ParseError {
                        provider: "http".into(),
                        reason: e.to_string(),
                    })
                })?;
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
            let delay_ms =
                (self.base_backoff_ms * (1 << attempt)).min(30_000) + (rand_jitter() % 200);
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
                .map_err(|e| SievrantError::Llm(LlmError::Http(e.without_url().to_string())))?;

            let status = resp.status();
            if status.is_success() {
                return resp.json::<Value>().await.map_err(|e| {
                    SievrantError::Llm(LlmError::ParseError {
                        provider: "http".into(),
                        reason: e.to_string(),
                    })
                });
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

            let delay_ms =
                (self.base_backoff_ms * (1 << attempt)).min(30_000) + (rand_jitter() % 200);
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            attempt += 1;
        }
    }
}

/// Send a request built by `make`, retrying transient failures.
///
/// Retries connection errors, 408, 429, 529 (Anthropic "overloaded") and
/// other 5xx responses with exponential backoff plus jitter, honouring a
/// `retry-after` header when the provider sends one. Transport errors are
/// stripped of their URL so query-string credentials can never leak into
/// logs or API responses.
pub(crate) async fn send_with_retry<F>(
    provider: &str,
    max_retries: u32,
    mut make: F,
) -> Result<reqwest::Response>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    let mut attempt = 0u32;
    loop {
        let outcome = make().send().await;
        let retry_after = match &outcome {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(outcome.unwrap());
                }
                let retryable = status == StatusCode::REQUEST_TIMEOUT
                    || status == StatusCode::TOO_MANY_REQUESTS
                    || status.as_u16() == 529
                    || status.is_server_error();
                if !retryable || attempt >= max_retries {
                    let resp = outcome.unwrap();
                    let body = resp.text().await.unwrap_or_default();
                    return Err(SievrantError::Llm(LlmError::ApiError {
                        provider: provider.to_string(),
                        status: status.as_u16(),
                        body: truncate(&body, 2000),
                    }));
                }
                resp.headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.trim().parse::<u64>().ok())
                    .map(|secs| secs.min(60) * 1000)
            }
            Err(e) => {
                if attempt >= max_retries || !(e.is_timeout() || e.is_connect()) {
                    let e = outcome.unwrap_err().without_url();
                    return Err(SievrantError::Llm(LlmError::Http(format!(
                        "{provider}: {e}"
                    ))));
                }
                None
            }
        };

        let backoff = (500u64 << attempt.min(6)).min(30_000) + rand_jitter() % 250;
        let delay_ms = retry_after.unwrap_or(backoff);
        warn!(provider, attempt, delay_ms, "retrying provider request");
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        attempt += 1;
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &s[..end])
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
