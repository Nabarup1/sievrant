//! Shared server state: provider services (built once), API keys, and a
//! tenant-scoped registry of open corpora.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};

use sievrant::{validate_name, Corpus, Services};
use sievrant_core::error::{Result, SievrantError};
use tokio::sync::Semaphore;

use crate::api::auth::ApiKeys;

type Key = (String, String);

/// Open corpora keyed by `(tenant, corpus)`. A corpus is opened at most once
/// per process: two `Corpus` instances over the same files would each hold
/// their own ingest lock and could corrupt the chunk store.
pub struct CorpusRegistry {
    services: Arc<Services>,
    open: RwLock<HashMap<Key, Arc<Corpus>>>,
    opening: tokio::sync::Mutex<()>,
}

impl CorpusRegistry {
    pub fn new(services: Arc<Services>) -> Self {
        Self {
            services,
            open: RwLock::new(HashMap::new()),
            opening: tokio::sync::Mutex::new(()),
        }
    }

    fn cached(&self, key: &Key) -> Option<Arc<Corpus>> {
        self.open
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(key)
            .cloned()
    }

    /// Open an existing corpus. `CorpusNotFound` if it does not exist.
    pub async fn get(&self, tenant: &str, name: &str) -> Result<Arc<Corpus>> {
        self.open_inner(tenant, name, false).await
    }

    /// Open a corpus, creating it if needed.
    pub async fn get_or_create(&self, tenant: &str, name: &str) -> Result<Arc<Corpus>> {
        self.open_inner(tenant, name, true).await
    }

    async fn open_inner(&self, tenant: &str, name: &str, create: bool) -> Result<Arc<Corpus>> {
        validate_name("tenant", tenant)?;
        validate_name("corpus", name)?;
        let key = (tenant.to_string(), name.to_string());
        if let Some(c) = self.cached(&key) {
            return Ok(c);
        }
        let _guard = self.opening.lock().await;
        if let Some(c) = self.cached(&key) {
            return Ok(c);
        }
        if !create && !Corpus::exists(&self.services, tenant, name) {
            return Err(SievrantError::CorpusNotFound { name: name.into() });
        }
        let corpus =
            Arc::new(Corpus::open_with(self.services.clone(), tenant, name, create).await?);
        self.open
            .write()
            .unwrap_or_else(|p| p.into_inner())
            .insert(key, corpus.clone());
        Ok(corpus)
    }

    /// Names of corpora that exist on disk for a tenant, sorted.
    pub fn list(&self, tenant: &str) -> Vec<String> {
        if validate_name("tenant", tenant).is_err() {
            return Vec::new();
        }
        let dir = PathBuf::from(&self.services.config.storage.data_dir)
            .join("tenants")
            .join(tenant);
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().join("corpus.json").exists())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| validate_name("corpus", n).is_ok())
            .collect();
        names.sort();
        names
    }
}

#[derive(Clone)]
pub struct AppState {
    pub services: Arc<Services>,
    pub keys: Arc<ApiKeys>,
    pub corpora: Arc<CorpusRegistry>,
    pub limiter: Arc<Semaphore>,
    pub request_timeout: Duration,
    /// Calibration runs the full pipeline once per example and can take minutes.
    pub calibration_timeout: Duration,
}

impl AppState {
    pub fn new(services: Arc<Services>, keys: ApiKeys) -> Self {
        let srv = &services.config.server;
        let request_timeout = Duration::from_secs(srv.request_timeout_secs.max(1));
        Self {
            limiter: Arc::new(Semaphore::new(srv.max_concurrent_requests.max(1))),
            request_timeout,
            calibration_timeout: request_timeout.max(Duration::from_secs(30 * 60)),
            corpora: Arc::new(CorpusRegistry::new(services.clone())),
            keys: Arc::new(keys),
            services,
        }
    }
}
