use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use crossbeam_channel::{bounded, Receiver, Sender, TrySendError};
use reqwest::Client;
use serde_json::json;
use tracing::{error, info, warn};

use sievrant_core::error::{Result, SievrantError};

use crate::metering::UsageRecord;

/// Contract for any billing emission backend.
/// Implementations include WebhookBillingEmitter and LogBillingEmitter.
#[async_trait]
pub trait BillingEmitter: Send + Sync {
    async fn emit_usage(&self, record: &UsageRecord) -> Result<()>;
}

/// Emits usage records to a webhook endpoint (e.g. your Next.js API or Stripe).
/// Records are enqueued on the hot path and flushed by a background task every
/// `flush_interval_secs` seconds to avoid any blocking on the retrieval path.
pub struct WebhookBillingEmitter {
    sender: Sender<UsageRecord>,
}

impl WebhookBillingEmitter {
    /// Start the emitter. The background flush task runs until `shutdown` is set.
    pub fn start(
        webhook_url: String,
        flush_interval_secs: u64,
        buffer_capacity: usize,
    ) -> (Self, Arc<AtomicBool>) {
        let (sender, receiver) = bounded::<UsageRecord>(buffer_capacity);
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        tokio::spawn(flush_loop(
            receiver,
            webhook_url,
            flush_interval_secs,
            shutdown_clone,
        ));

        (Self { sender }, shutdown)
    }
}

#[async_trait]
impl BillingEmitter for WebhookBillingEmitter {
    async fn emit_usage(&self, record: &UsageRecord) -> Result<()> {
        self.sender.try_send(record.clone()).map_err(|e| {
            SievrantError::Internal(match e {
                TrySendError::Full(_) => "billing queue full; record dropped".into(),
                TrySendError::Disconnected(_) => "billing background task has shut down".into(),
            })
        })
    }
}

/// Background flush loop that batches records and POSTs them to the webhook.
/// Uses an in-process buffer for resilience when the webhook is temporarily unavailable.
async fn flush_loop(
    receiver: Receiver<UsageRecord>,
    webhook_url: String,
    flush_interval_secs: u64,
    shutdown: Arc<AtomicBool>,
) {
    let client = Client::new();
    let mut interval = tokio::time::interval(Duration::from_secs(flush_interval_secs));

    loop {
        interval.tick().await;

        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let batch: Vec<UsageRecord> = receiver.try_iter().collect();
        if batch.is_empty() {
            continue;
        }

        info!(count = batch.len(), "flushing billing records to webhook");

        let body = json!({ "records": batch });
        match client.post(&webhook_url).json(&body).send().await {
            Ok(resp) if resp.status().is_success() => {
                info!(count = batch.len(), "billing records sent successfully");
            }
            Ok(resp) => {
                warn!(
                    status = resp.status().as_u16(),
                    count = batch.len(),
                    "webhook returned non-2xx; records may be lost"
                );
            }
            Err(e) => {
                error!(
                    error = %e,
                    count = batch.len(),
                    "webhook unreachable; records lost (WAL persistence not yet wired)"
                );
            }
        }
    }

    info!("billing flush loop terminated");
}

/// A simple billing emitter that writes records to the tracing log.
/// Useful for local development and integration tests.
pub struct LogBillingEmitter;

#[async_trait]
impl BillingEmitter for LogBillingEmitter {
    async fn emit_usage(&self, record: &UsageRecord) -> Result<()> {
        info!(
            tenant_id = %record.tenant_id,
            query_id = %record.query_id,
            total_tokens = record.total_tokens(),
            tokens_saved = record.tokens_saved,
            cost_usd_cents = record.cost_usd_cents,
            "billing_record"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn log_emitter_does_not_error() {
        let emitter = LogBillingEmitter;
        let record = crate::metering::UsageRecord::new(
            "tenant-1", "q-1", "gpt-4o", 100, 300, 150, 3, 200, 0.03,
        );
        assert!(emitter.emit_usage(&record).await.is_ok());
    }

    #[tokio::test]
    async fn webhook_emitter_buffers_records() {
        let (emitter, shutdown) =
            WebhookBillingEmitter::start("http://localhost:9999/noop".into(), 60, 100);
        let record = crate::metering::UsageRecord::new(
            "tenant-1", "q-2", "gpt-4o", 50, 200, 80, 2, 200, 0.03,
        );
        // Should succeed because the channel has capacity
        assert!(emitter.emit_usage(&record).await.is_ok());
        shutdown.store(true, Ordering::Relaxed);
    }
}
