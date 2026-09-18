use std::{path::PathBuf, sync::Arc};

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use sievrant_core::error::{Result, StorageError};

/// Persisted metadata for a corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusMetadata {
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub document_count: usize,
    pub chunk_count: usize,
    pub index_generation: u64,
    pub calibrated: bool,
    pub description: Option<String>,
}

/// Tracked document record inside a corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub doc_id: String,
    pub source_path: String,
    pub version_hash: String,
    pub chunk_ids: Vec<String>,
    pub ingested_at: DateTime<Utc>,
    pub deleted_at: Option<DateTime<Utc>>,
}

/// In-memory handle for a corpus. Each corpus owns its metadata and its set of
/// live document records; the indexes themselves are owned by the orchestrator.
pub struct CorpusState {
    pub metadata: CorpusMetadata,
    /// All documents, including logically deleted ones (deleted_at is set).
    documents: DashMap<String, DocumentRecord>,
    /// Chunk IDs that have been logically deleted and must be filtered from results.
    deleted_chunk_ids: DashMap<String, ()>,
}

impl CorpusState {
    fn new(metadata: CorpusMetadata) -> Self {
        Self {
            metadata,
            documents: DashMap::new(),
            deleted_chunk_ids: DashMap::new(),
        }
    }

    /// True if this chunk_id has been logically deleted and should be excluded from results.
    pub fn is_chunk_deleted(&self, chunk_id: &str) -> bool {
        self.deleted_chunk_ids.contains_key(chunk_id)
    }
}

/// Corpus manager: lifecycle operations for named corpora stored under `data_dir`.
pub struct CorpusManager {
    data_dir: PathBuf,
    corpora: DashMap<String, Arc<CorpusState>>,
}

impl CorpusManager {
    pub fn new(data_dir: impl Into<PathBuf>) -> Result<Self> {
        let data_dir: PathBuf = data_dir.into();
        std::fs::create_dir_all(&data_dir).map_err(|e| StorageError::Io(e.to_string()))?;

        let manager = Self {
            data_dir,
            corpora: DashMap::new(),
        };

        manager.load_existing_corpora()?;
        Ok(manager)
    }

    pub fn create(&self, name: &str, description: Option<String>) -> Result<()> {
        if self.corpora.contains_key(name) {
            return Err(StorageError::Io(format!("corpus '{name}' already exists")).into());
        }

        let metadata = CorpusMetadata {
            name: name.to_string(),
            created_at: Utc::now(),
            document_count: 0,
            chunk_count: 0,
            index_generation: 0,
            calibrated: false,
            description,
        };

        let corpus_dir = self.corpus_dir(name);
        std::fs::create_dir_all(&corpus_dir).map_err(|e| StorageError::Io(e.to_string()))?;
        self.persist_metadata(name, &metadata)?;

        self.corpora
            .insert(name.to_string(), Arc::new(CorpusState::new(metadata)));
        Ok(())
    }

    pub fn delete(&self, name: &str) -> Result<()> {
        if !self.corpora.contains_key(name) {
            return Err(StorageError::Io(format!("corpus '{name}' not found")).into());
        }

        let corpus_dir = self.corpus_dir(name);
        if corpus_dir.exists() {
            std::fs::remove_dir_all(&corpus_dir).map_err(|e| StorageError::Io(e.to_string()))?;
        }

        self.corpora.remove(name);
        Ok(())
    }

    pub fn list(&self) -> Vec<CorpusMetadata> {
        self.corpora
            .iter()
            .map(|e| e.value().metadata.clone())
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<Arc<CorpusState>> {
        self.corpora.get(name).map(|e| Arc::clone(e.value()))
    }

    /// Register a new document in a corpus. The caller is responsible for
    /// dispatching the chunks to the actual indexes.
    pub fn add_document(
        &self,
        corpus: &str,
        doc_id: &str,
        source_path: &str,
        content: &str,
        chunk_ids: Vec<String>,
    ) -> Result<DocumentRecord> {
        let corpus_state = self
            .corpora
            .get(corpus)
            .ok_or_else(|| StorageError::Io(format!("corpus '{corpus}' not found")))?;

        let version_hash = sha256_of(content.as_bytes());
        let record = DocumentRecord {
            doc_id: doc_id.to_string(),
            source_path: source_path.to_string(),
            version_hash,
            chunk_ids: chunk_ids.clone(),
            ingested_at: Utc::now(),
            deleted_at: None,
        };

        corpus_state
            .documents
            .insert(doc_id.to_string(), record.clone());
        Ok(record)
    }

    /// Logically delete a document and all its chunks. The deleted chunk IDs
    /// are stored in the corpus so that post-retrieval filtering can exclude them.
    pub fn remove_document(&self, corpus: &str, doc_id: &str) -> Result<()> {
        let corpus_state = self
            .corpora
            .get(corpus)
            .ok_or_else(|| StorageError::Io(format!("corpus '{corpus}' not found")))?;

        if let Some(mut record) = corpus_state.documents.get_mut(doc_id) {
            record.deleted_at = Some(Utc::now());
            for chunk_id in &record.chunk_ids {
                corpus_state.deleted_chunk_ids.insert(chunk_id.clone(), ());
            }
        }

        Ok(())
    }

    /// Atomically replace a document: check if the content hash changed, and if
    /// so, mark the old version deleted and register the new one.
    pub fn update_document(
        &self,
        corpus: &str,
        doc_id: &str,
        source_path: &str,
        new_content: &str,
        new_chunk_ids: Vec<String>,
    ) -> Result<bool> {
        let new_hash = sha256_of(new_content.as_bytes());

        let corpus_state = self
            .get(corpus)
            .ok_or_else(|| StorageError::Io(format!("corpus '{corpus}' not found")))?;

        let has_same_hash = corpus_state
            .documents
            .get(doc_id)
            .map(|existing| existing.version_hash == new_hash)
            .unwrap_or(false);

        if has_same_hash {
            // No content change; skip expensive reingestion
            return Ok(false);
        }

        self.remove_document(corpus, doc_id)?;
        self.add_document(corpus, doc_id, source_path, new_content, new_chunk_ids)?;
        Ok(true)
    }

    /// Compute how many hours have passed since a chunk was ingested.
    pub fn compute_freshness(ingested_at: &DateTime<Utc>) -> f64 {
        let duration = Utc::now() - *ingested_at;
        duration.num_minutes() as f64 / 60.0
    }

    fn corpus_dir(&self, name: &str) -> PathBuf {
        self.data_dir.join(name)
    }

    fn persist_metadata(&self, name: &str, metadata: &CorpusMetadata) -> Result<()> {
        let path = self.corpus_dir(name).join("metadata.json");
        let json = serde_json::to_string_pretty(metadata)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        std::fs::write(&path, json).map_err(|e| StorageError::Io(e.to_string()))?;
        Ok(())
    }

    fn load_existing_corpora(&self) -> Result<()> {
        let entries =
            std::fs::read_dir(&self.data_dir).map_err(|e| StorageError::Io(e.to_string()))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let meta_path = path.join("metadata.json");
            if !meta_path.exists() {
                continue;
            }
            let json =
                std::fs::read_to_string(&meta_path).map_err(|e| StorageError::Io(e.to_string()))?;
            let metadata: CorpusMetadata = serde_json::from_str(&json)
                .map_err(|e| StorageError::Serialization(e.to_string()))?;
            let name = metadata.name.clone();
            self.corpora
                .insert(name, Arc::new(CorpusState::new(metadata)));
        }

        Ok(())
    }
}

fn sha256_of(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn manager() -> (TempDir, CorpusManager) {
        let dir = TempDir::new().unwrap();
        let mgr = CorpusManager::new(dir.path()).unwrap();
        (dir, mgr)
    }

    #[test]
    fn create_and_list_corpus() {
        let (_dir, mgr) = manager();
        mgr.create("legal-contracts", Some("Lease agreements".into()))
            .unwrap();
        let list = mgr.list();
        assert!(list.iter().any(|m| m.name == "legal-contracts"));
    }

    #[test]
    fn delete_removes_corpus() {
        let (_dir, mgr) = manager();
        mgr.create("tmp", None).unwrap();
        mgr.delete("tmp").unwrap();
        assert!(mgr.get("tmp").is_none());
    }

    #[test]
    fn duplicate_create_returns_error() {
        let (_dir, mgr) = manager();
        mgr.create("corp", None).unwrap();
        assert!(mgr.create("corp", None).is_err());
    }

    #[test]
    fn document_version_hash_detects_changes() {
        let (_dir, mgr) = manager();
        mgr.create("corp", None).unwrap();

        mgr.add_document(
            "corp",
            "d1",
            "doc.pdf",
            "original content",
            vec!["c1".into()],
        )
        .unwrap();
        let changed = mgr
            .update_document("corp", "d1", "doc.pdf", "new content", vec!["c2".into()])
            .unwrap();
        assert!(changed, "different content should trigger update");

        let not_changed = mgr
            .update_document("corp", "d1", "doc.pdf", "new content", vec!["c2".into()])
            .unwrap();
        assert!(!not_changed, "same content should not trigger update");
    }

    #[test]
    fn removed_document_chunks_are_marked_deleted() {
        let (_dir, mgr) = manager();
        mgr.create("corp", None).unwrap();
        mgr.add_document(
            "corp",
            "d1",
            "doc.pdf",
            "text",
            vec!["c1".into(), "c2".into()],
        )
        .unwrap();
        mgr.remove_document("corp", "d1").unwrap();

        let state = mgr.get("corp").unwrap();
        assert!(state.is_chunk_deleted("c1"));
        assert!(state.is_chunk_deleted("c2"));
        assert!(!state.is_chunk_deleted("c99"));
    }

    #[test]
    fn corpus_survives_reload() {
        let dir = TempDir::new().unwrap();
        {
            let mgr = CorpusManager::new(dir.path()).unwrap();
            mgr.create("persistent-corp", Some("test".into())).unwrap();
        }
        let mgr2 = CorpusManager::new(dir.path()).unwrap();
        assert!(
            mgr2.get("persistent-corp").is_some(),
            "corpus must survive manager reload"
        );
    }
}
