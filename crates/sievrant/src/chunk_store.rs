//! Persisted record of every live document and chunk in a corpus.
//!
//! The indexes answer "which chunk ids match", this store answers "what is
//! that chunk": text, provenance and ACL metadata. Index hits are hydrated
//! from here, so a chunk that is not in the store (deleted, or orphaned by a
//! crash mid-ingest) can never be returned.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, RwLock};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use sievrant_core::error::{Result, SievrantError, StorageError};
use sievrant_core::types::ChunkMetadata;

fn io_err(e: impl std::fmt::Display) -> SievrantError {
    StorageError::Io(e.to_string()).into()
}

/// A stored chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunk {
    pub chunk_id: String,
    pub text: String,
    pub metadata: ChunkMetadata,
}

/// A stored document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentEntry {
    pub doc_id: String,
    pub version_hash: String,
    pub title: Option<String>,
    pub source: Option<String>,
    pub acl_groups: Vec<String>,
    pub ingested_at: DateTime<Utc>,
    pub chunk_ids: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum LogRecord {
    Put {
        doc: DocumentEntry,
        chunks: Vec<StoredChunk>,
    },
    Del {
        doc_id: String,
    },
}

#[derive(Default)]
struct State {
    docs: HashMap<String, DocumentEntry>,
    chunks: HashMap<String, StoredChunk>,
}

pub struct ChunkStore {
    path: PathBuf,
    state: RwLock<State>,
    log: Mutex<BufWriter<File>>,
}

impl ChunkStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(io_err)?;
        }
        let (state, deletes) = Self::load(&path)?;
        if deletes > 0 && deletes * 3 > state.docs.len() {
            Self::compact(&path, &state)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(io_err)?;
        Ok(Self {
            path,
            state: RwLock::new(state),
            log: Mutex::new(BufWriter::new(file)),
        })
    }

    fn apply(state: &mut State, rec: LogRecord) -> bool {
        match rec {
            LogRecord::Put { doc, chunks } => {
                if let Some(old) = state.docs.remove(&doc.doc_id) {
                    for id in old.chunk_ids {
                        state.chunks.remove(&id);
                    }
                }
                for c in chunks {
                    state.chunks.insert(c.chunk_id.clone(), c);
                }
                state.docs.insert(doc.doc_id.clone(), doc);
                false
            }
            LogRecord::Del { doc_id } => {
                if let Some(old) = state.docs.remove(&doc_id) {
                    for id in old.chunk_ids {
                        state.chunks.remove(&id);
                    }
                }
                true
            }
        }
    }

    fn load(path: &PathBuf) -> Result<(State, usize)> {
        let mut state = State::default();
        let mut deletes = 0;
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((state, 0)),
            Err(e) => return Err(io_err(e)),
        };
        for line in BufReader::new(file).lines() {
            let line = line.map_err(io_err)?;
            if line.trim().is_empty() {
                continue;
            }
            // A torn final line after a crash is skipped.
            if let Ok(rec) = serde_json::from_str::<LogRecord>(&line) {
                if Self::apply(&mut state, rec) {
                    deletes += 1;
                }
            }
        }
        Ok((state, deletes))
    }

    fn compact(path: &PathBuf, state: &State) -> Result<()> {
        let tmp = path.with_extension("compact");
        {
            let mut w = BufWriter::new(File::create(&tmp).map_err(io_err)?);
            for doc in state.docs.values() {
                let chunks = doc
                    .chunk_ids
                    .iter()
                    .filter_map(|id| state.chunks.get(id).cloned())
                    .collect();
                let rec = LogRecord::Put {
                    doc: doc.clone(),
                    chunks,
                };
                let line = serde_json::to_string(&rec).map_err(io_err)?;
                writeln!(w, "{line}").map_err(io_err)?;
            }
            w.flush().map_err(io_err)?;
        }
        fs::rename(&tmp, path).map_err(io_err)
    }

    fn append(&self, rec: &LogRecord) -> Result<()> {
        let line = serde_json::to_string(rec).map_err(io_err)?;
        let mut w = self.log.lock().unwrap();
        writeln!(w, "{line}").map_err(io_err)?;
        w.flush().map_err(io_err)
    }

    /// Store a document and its chunks, replacing any previous version.
    pub fn put(&self, doc: DocumentEntry, chunks: Vec<StoredChunk>) -> Result<()> {
        let rec = LogRecord::Put { doc, chunks };
        self.append(&rec)?;
        Self::apply(&mut self.state.write().unwrap(), rec);
        Ok(())
    }

    /// Remove a document. Returns its chunk ids if it existed.
    pub fn delete(&self, doc_id: &str) -> Result<Option<Vec<String>>> {
        let ids = self.doc(doc_id).map(|d| d.chunk_ids);
        if ids.is_some() {
            let rec = LogRecord::Del {
                doc_id: doc_id.to_string(),
            };
            self.append(&rec)?;
            Self::apply(&mut self.state.write().unwrap(), rec);
        }
        Ok(ids)
    }

    pub fn doc(&self, doc_id: &str) -> Option<DocumentEntry> {
        self.state.read().unwrap().docs.get(doc_id).cloned()
    }

    pub fn chunk(&self, chunk_id: &str) -> Option<StoredChunk> {
        self.state.read().unwrap().chunks.get(chunk_id).cloned()
    }

    pub fn has_doc(&self, doc_id: &str) -> bool {
        self.state.read().unwrap().docs.contains_key(doc_id)
    }

    pub fn document_count(&self) -> usize {
        self.state.read().unwrap().docs.len()
    }

    pub fn chunk_count(&self) -> usize {
        self.state.read().unwrap().chunks.len()
    }

    /// Documents sorted by id, paginated.
    pub fn list(&self, offset: usize, limit: usize) -> Vec<DocumentEntry> {
        let state = self.state.read().unwrap();
        let mut ids: Vec<&String> = state.docs.keys().collect();
        ids.sort();
        ids.into_iter()
            .skip(offset)
            .take(limit)
            .filter_map(|id| state.docs.get(id).cloned())
            .collect()
    }

    /// Deterministic sample of up to `n` chunks with at least `min_words` words.
    pub fn sample(&self, n: usize, min_words: usize, seed: u64) -> Vec<StoredChunk> {
        let state = self.state.read().unwrap();
        let mut ids: Vec<&String> = state
            .chunks
            .iter()
            .filter(|(_, c)| c.text.split_whitespace().count() >= min_words)
            .map(|(id, _)| id)
            .collect();
        ids.sort();
        let mut x = seed | 1;
        for i in (1..ids.len()).rev() {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let j = (x % (i as u64 + 1)) as usize;
            ids.swap(i, j);
        }
        ids.into_iter()
            .take(n)
            .filter_map(|id| state.chunks.get(id).cloned())
            .collect()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(doc: &str) -> ChunkMetadata {
        ChunkMetadata {
            doc_id: doc.into(),
            source_doc: doc.into(),
            doc_version_hash: "h".into(),
            path: vec![],
            page: None,
            char_span: None,
            ingested_at: Utc::now(),
            index_generation: 0,
            acl_groups: vec!["public".into()],
            token_count: 3,
        }
    }

    fn put(store: &ChunkStore, doc: &str, n: usize, hash: &str) {
        let chunks: Vec<StoredChunk> = (0..n)
            .map(|i| StoredChunk {
                chunk_id: format!("{doc}:{i}"),
                text: format!("chunk {i} of {doc} with several words in it"),
                metadata: meta(doc),
            })
            .collect();
        store
            .put(
                DocumentEntry {
                    doc_id: doc.into(),
                    version_hash: hash.into(),
                    title: None,
                    source: None,
                    acl_groups: vec!["public".into()],
                    ingested_at: Utc::now(),
                    chunk_ids: chunks.iter().map(|c| c.chunk_id.clone()).collect(),
                },
                chunks,
            )
            .unwrap();
    }

    #[test]
    fn put_replace_delete_and_reload() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("chunks.jsonl");
        {
            let s = ChunkStore::open(&path).unwrap();
            put(&s, "a", 3, "h1");
            put(&s, "b", 2, "h1");
            put(&s, "a", 1, "h2");
            assert_eq!(s.chunk_count(), 3);
            assert!(s.chunk("a:2").is_none());
            assert_eq!(s.delete("b").unwrap().unwrap().len(), 2);
            assert!(s.delete("missing").unwrap().is_none());
        }
        let s = ChunkStore::open(&path).unwrap();
        assert_eq!(s.document_count(), 1);
        assert_eq!(s.chunk_count(), 1);
        assert_eq!(s.doc("a").unwrap().version_hash, "h2");
    }

    #[test]
    fn sample_is_deterministic_and_respects_min_words() {
        let dir = tempfile::TempDir::new().unwrap();
        let s = ChunkStore::open(dir.path().join("c.jsonl")).unwrap();
        for d in ["a", "b", "c", "d"] {
            put(&s, d, 5, "h");
        }
        let one = s.sample(6, 3, 7);
        let two = s.sample(6, 3, 7);
        assert_eq!(one.len(), 6);
        assert_eq!(
            one.iter().map(|c| &c.chunk_id).collect::<Vec<_>>(),
            two.iter().map(|c| &c.chunk_id).collect::<Vec<_>>()
        );
        assert!(s.sample(5, 100, 7).is_empty());
    }
}
