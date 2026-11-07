//! Embedded vector store: exact cosine search over an in-memory map,
//! optionally persisted to an append-only log so vectors survive restarts.
//!
//! Log record layout (little endian):
//! - upsert: `0x01 | u32 meta_len | meta_json | u32 dim | f32 * dim`
//! - delete: `0x02 | u32 id_len | id_utf8`
//!
//! The log is compacted on open when deletions make up a large share of it.
//! Exact search is O(N) per query; use the Qdrant backend beyond ~1M chunks.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::{IndexChunk, VectorStore};
use sievrant_core::{
    error::{IndexError, Result},
    types::{ChunkMetadata, RankedCandidate, RetrieverType},
};

const OP_UPSERT: u8 = 1;
const OP_DELETE: u8 = 2;

#[derive(Serialize, Deserialize)]
struct StoredMeta {
    chunk_id: String,
    text: String,
    metadata: ChunkMetadata,
}

struct Point {
    chunk: IndexChunk,
    embedding: Vec<f32>,
    norm: f32,
}

fn io_err(e: impl std::fmt::Display) -> sievrant_core::error::SievrantError {
    IndexError::Vector(e.to_string()).into()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub struct MemoryStore {
    _dimensions: usize,
    points: Arc<RwLock<HashMap<String, Point>>>,
    log: Option<Mutex<BufWriter<File>>>,
}

impl MemoryStore {
    /// Volatile store (tests, ephemeral use).
    pub fn new(dimensions: usize) -> Self {
        Self {
            _dimensions: dimensions,
            points: Arc::new(RwLock::new(HashMap::new())),
            log: None,
        }
    }

    /// Store persisted at `path`, loading any existing contents.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(io_err)?;
        }
        let (points, deletes) = Self::load(&path)?;
        if deletes > 0 && deletes * 3 > points.len() {
            Self::compact(&path, &points)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(io_err)?;
        Ok(Self {
            _dimensions: 0,
            points: Arc::new(RwLock::new(points)),
            log: Some(Mutex::new(BufWriter::new(file))),
        })
    }

    fn load(path: &PathBuf) -> Result<(HashMap<String, Point>, usize)> {
        let mut points = HashMap::new();
        let mut deletes = 0usize;
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((points, 0)),
            Err(e) => return Err(io_err(e)),
        };
        let mut r = BufReader::new(file);
        let mut op = [0u8; 1];
        loop {
            match r.read_exact(&mut op) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(io_err(e)),
            }
            // A torn record at the tail (crash mid-write) ends the log.
            let Ok(len) = read_u32(&mut r) else { break };
            let mut buf = vec![0u8; len as usize];
            if r.read_exact(&mut buf).is_err() {
                break;
            }
            match op[0] {
                OP_UPSERT => {
                    let Ok(dim) = read_u32(&mut r) else { break };
                    let mut raw = vec![0u8; dim as usize * 4];
                    if r.read_exact(&mut raw).is_err() {
                        break;
                    }
                    let embedding: Vec<f32> = raw
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect();
                    let meta: StoredMeta = match serde_json::from_slice(&buf) {
                        Ok(m) => m,
                        Err(_) => break,
                    };
                    let n = norm(&embedding);
                    points.insert(
                        meta.chunk_id.clone(),
                        Point {
                            chunk: IndexChunk {
                                chunk_id: meta.chunk_id,
                                text: meta.text,
                                embedding: None,
                                metadata: meta.metadata,
                            },
                            embedding,
                            norm: n,
                        },
                    );
                }
                OP_DELETE => {
                    let id = String::from_utf8_lossy(&buf).to_string();
                    points.remove(&id);
                    deletes += 1;
                }
                _ => break,
            }
        }
        Ok((points, deletes))
    }

    fn compact(path: &PathBuf, points: &HashMap<String, Point>) -> Result<()> {
        let tmp = path.with_extension("compact");
        {
            let mut w = BufWriter::new(File::create(&tmp).map_err(io_err)?);
            for p in points.values() {
                write_upsert(&mut w, &p.chunk, &p.embedding)?;
            }
            w.flush().map_err(io_err)?;
        }
        fs::rename(&tmp, path).map_err(io_err)
    }

    /// Number of stored vectors.
    pub async fn len(&self) -> usize {
        self.points.read().await.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.points.read().await.is_empty()
    }
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn write_upsert(w: &mut impl Write, chunk: &IndexChunk, embedding: &[f32]) -> Result<()> {
    let meta = serde_json::to_vec(&StoredMeta {
        chunk_id: chunk.chunk_id.clone(),
        text: chunk.text.clone(),
        metadata: chunk.metadata.clone(),
    })
    .map_err(io_err)?;
    let mut rec = Vec::with_capacity(9 + meta.len() + embedding.len() * 4);
    rec.push(OP_UPSERT);
    rec.extend_from_slice(&(meta.len() as u32).to_le_bytes());
    rec.extend_from_slice(&meta);
    rec.extend_from_slice(&(embedding.len() as u32).to_le_bytes());
    for x in embedding {
        rec.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&rec).map_err(io_err)
}

#[async_trait]
impl VectorStore for MemoryStore {
    async fn upsert_vectors(&self, chunks: &[IndexChunk]) -> Result<()> {
        if let Some(log) = &self.log {
            let mut w = log.lock().unwrap();
            for chunk in chunks {
                if let Some(e) = &chunk.embedding {
                    write_upsert(&mut *w, chunk, e)?;
                }
            }
            w.flush().map_err(io_err)?;
        }
        let mut guard = self.points.write().await;
        for chunk in chunks {
            if let Some(e) = &chunk.embedding {
                guard.insert(
                    chunk.chunk_id.clone(),
                    Point {
                        chunk: IndexChunk {
                            embedding: None,
                            ..chunk.clone()
                        },
                        norm: norm(e),
                        embedding: e.clone(),
                    },
                );
            }
        }
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        acl_filter: &[String],
    ) -> Result<Vec<RankedCandidate>> {
        let qn = norm(query_embedding);
        if qn == 0.0 {
            return Ok(vec![]);
        }
        let guard = self.points.read().await;
        let mut results: Vec<(f32, &Point)> = guard
            .values()
            .filter(|p| {
                acl_filter.is_empty()
                    || acl_filter
                        .iter()
                        .any(|g| p.chunk.metadata.acl_groups.contains(g))
            })
            .filter(|p| p.embedding.len() == query_embedding.len() && p.norm > 0.0)
            .map(|p| {
                let dot: f32 = p
                    .embedding
                    .iter()
                    .zip(query_embedding)
                    .map(|(a, b)| a * b)
                    .sum();
                (dot / (p.norm * qn), p)
            })
            .collect();

        results.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(top_k)
            .map(|(sim, p)| RankedCandidate {
                chunk_id: p.chunk.chunk_id.clone(),
                score: sim as f64,
                retriever: RetrieverType::VectorIndex,
                text: p.chunk.text.clone(),
                metadata: p.chunk.metadata.clone(),
            })
            .collect())
    }

    async fn delete_documents(&self, chunk_ids: &[String]) -> Result<()> {
        if let Some(log) = &self.log {
            let mut w = log.lock().unwrap();
            for id in chunk_ids {
                w.write_all(&[OP_DELETE]).map_err(io_err)?;
                w.write_all(&(id.len() as u32).to_le_bytes())
                    .map_err(io_err)?;
                w.write_all(id.as_bytes()).map_err(io_err)?;
            }
            w.flush().map_err(io_err)?;
        }
        let mut guard = self.points.write().await;
        for id in chunk_ids {
            guard.remove(id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::types::ChunkMetadata;

    fn test_chunk(id: &str, groups: Vec<&str>, embedding: Vec<f32>) -> IndexChunk {
        IndexChunk {
            chunk_id: id.to_string(),
            text: format!("text of {id}"),
            embedding: Some(embedding),
            metadata: ChunkMetadata {
                doc_id: "doc1".to_string(),
                source_doc: "doc1.txt".to_string(),
                doc_version_hash: "hash".to_string(),
                path: vec!["Section 1".into()],
                page: Some(3),
                char_span: Some([0, 10]),
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups: groups.into_iter().map(|s| s.to_string()).collect(),
                token_count: 10,
            },
        }
    }

    #[tokio::test]
    async fn memory_store_enforces_acl() {
        let store = MemoryStore::new(2);
        store
            .upsert_vectors(&[
                test_chunk("c1", vec!["engineering"], vec![1.0, 0.0]),
                test_chunk("c2", vec!["hr"], vec![1.0, 0.0]),
            ])
            .await
            .unwrap();

        let r = store
            .search(&[1.0, 0.0], 10, &["engineering".into()])
            .await
            .unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].chunk_id, "c1");
        let r = store.search(&[1.0, 0.0], 10, &["hr".into()]).await.unwrap();
        assert_eq!(r[0].chunk_id, "c2");
        assert!(store
            .search(&[1.0, 0.0], 10, &["executive".into()])
            .await
            .unwrap()
            .is_empty());
        // No filter means see all (internal system query)
        assert_eq!(store.search(&[1.0, 0.0], 10, &[]).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn upsert_replaces_and_search_ranks_by_cosine() {
        let store = MemoryStore::new(2);
        store
            .upsert_vectors(&[
                test_chunk("a", vec!["p"], vec![1.0, 0.0]),
                test_chunk("b", vec!["p"], vec![0.0, 1.0]),
            ])
            .await
            .unwrap();
        store
            .upsert_vectors(&[test_chunk("b", vec!["p"], vec![0.9, 0.1])])
            .await
            .unwrap();
        assert_eq!(store.len().await, 2);
        let r = store.search(&[1.0, 0.0], 2, &[]).await.unwrap();
        assert_eq!(r[0].chunk_id, "a");
        assert!(r[1].score > 0.9);
    }

    #[tokio::test]
    async fn persisted_store_survives_reopen_with_deletes_and_metadata() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");
        {
            let s = MemoryStore::open(&path).unwrap();
            s.upsert_vectors(&[
                test_chunk("a", vec!["p"], vec![1.0, 0.0]),
                test_chunk("b", vec!["p"], vec![0.0, 1.0]),
                test_chunk("c", vec!["p"], vec![0.5, 0.5]),
            ])
            .await
            .unwrap();
            s.delete_documents(&["b".into()]).await.unwrap();
        }
        let s = MemoryStore::open(&path).unwrap();
        assert_eq!(s.len().await, 2);
        let r = s.search(&[1.0, 0.0], 5, &[]).await.unwrap();
        assert_eq!(r[0].chunk_id, "a");
        assert_eq!(r[0].metadata.page, Some(3));
        assert!(r.iter().all(|c| c.chunk_id != "b"));
    }

    #[tokio::test]
    async fn torn_tail_record_is_ignored() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("vectors.bin");
        {
            let s = MemoryStore::open(&path).unwrap();
            s.upsert_vectors(&[test_chunk("a", vec!["p"], vec![1.0, 0.0])])
                .await
                .unwrap();
        }
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(&[OP_UPSERT, 200, 0, 0, 0, b'{']).unwrap();
        let s = MemoryStore::open(&path).unwrap();
        assert_eq!(s.len().await, 1);
    }
}
