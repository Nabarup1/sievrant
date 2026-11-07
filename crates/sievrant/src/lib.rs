#![warn(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

//! # Sievrant
//!
//! Retrieval with a stated, testable guarantee: after calibration, the
//! evidence Sievrant returns contains a relevant passage for at least the
//! requested fraction of answerable queries, and it abstains instead of
//! guessing when nothing clears the calibrated threshold.
//!
//! The entry point is [`Corpus`]: ingest documents, calibrate once
//! (labelled, synthetic or from feedback), then [`Corpus::retrieve`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde::Serialize;

pub use sievrant_core::config::SievrantConfig;
pub use sievrant_core::error::{Result, SievrantError};
pub use sievrant_core::types::{
    CalibrationSource, Decision, Document, DocumentMetadata, EvidencePacket, GuaranteeStatus,
    RetrievalConfig, Route, RouteStrategy,
};

mod calibrate;
mod chunk_store;
mod noop;
mod services;

pub use calibrate::{CalibrationExample, CalibrationReport, CalibrationStatus, ThresholdReport};
pub use chunk_store::{ChunkStore, DocumentEntry, StoredChunk};
pub use services::{provider_catalog, ProviderCatalogEntry, ProviderSummary, Services};
pub use sievrant_calibration::{FeedbackOutcome, GapEntry};

use sievrant_calibration::CorpusCalibration;
use sievrant_core::error::{ConfigError, StorageError};
use sievrant_core::types::{RankedCandidate, RetrieverType};
use sievrant_index::{
    bm25::{Bm25Index, Bm25IndexConfig},
    graph::{GraphIndex, GraphIndexConfig},
    tree::{TreeIndex, TreeIndexConfig},
    vector::{IndexChunk, IndexQuery, VectorBackendConfig, VectorIndex, VectorIndexConfig},
    RetrievalIndex,
};
use sievrant_math::conformal::nonconformity;
use sievrant_orchestrator::pipeline::{IndexSet, PipelineSettings, RetrievalOrchestrator};
use sievrant_query::{
    processing::{QueryProcessingConfig, QueryProcessor},
    router::{QueryRouter, RoutingConfig},
};

/// Tenant used by the single-tenant embedded API ([`Corpus::create`], [`Corpus::open`]).
pub const DEFAULT_TENANT: &str = "default";

/// Validate a tenant or corpus name: 1-64 chars of `[A-Za-z0-9_-]`.
/// Names become directory and collection names, so nothing else is allowed.
pub fn validate_name(kind: &str, name: &str) -> Result<()> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if ok {
        Ok(())
    } else {
        Err(SievrantError::Config(ConfigError::InvalidValue {
            field: kind.into(),
            value: format!("{name:?} (use 1-64 characters: letters, digits, '-' or '_')"),
        }))
    }
}

/// Adapter from the llm crate's embedding trait to the query crate's.
struct EmbeddingBridge(Arc<dyn sievrant_llm::EmbeddingProvider>);

#[async_trait]
impl sievrant_query::EmbeddingProvider for EmbeddingBridge {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.0.embed_batch(texts).await
    }
    fn dimensions(&self) -> usize {
        self.0.dimensions()
    }
}

/// Wraps an index so every hit is re-read from the chunk store: text and
/// metadata are authoritative, and deleted or orphaned chunks are dropped.
struct HydratingIndex {
    inner: Arc<dyn RetrievalIndex>,
    store: Arc<ChunkStore>,
}

#[async_trait]
impl RetrievalIndex for HydratingIndex {
    async fn search(
        &self,
        query: &IndexQuery,
        acl_filter: &[String],
        limit: usize,
    ) -> Result<Vec<RankedCandidate>> {
        let hits = self.inner.search(query, acl_filter, limit).await?;
        Ok(hits
            .into_iter()
            .filter_map(|mut hit| {
                let stored = self.store.chunk(&hit.chunk_id)?;
                // Defence in depth: re-check ACLs against stored metadata.
                if !acl_filter.is_empty()
                    && !stored
                        .metadata
                        .acl_groups
                        .iter()
                        .any(|g| acl_filter.contains(g))
                {
                    return None;
                }
                hit.text = stored.text;
                hit.metadata = stored.metadata;
                Some(hit)
            })
            .collect())
    }
    async fn insert(&self, chunks: &[IndexChunk]) -> Result<()> {
        self.inner.insert(chunks).await
    }
    async fn delete(&self, chunk_ids: &[String]) -> Result<()> {
        self.inner.delete(chunk_ids).await
    }
    fn index_type(&self) -> RetrieverType {
        self.inner.index_type()
    }
    async fn ingest_graph_entities(
        &self,
        chunk_id: &str,
        entities: &[String],
        relations: &[(String, String, String)],
    ) -> Result<()> {
        self.inner
            .ingest_graph_entities(chunk_id, entities, relations)
            .await
    }
    async fn ingest_tree_summary(&self, node_id: &str, summary: String) -> Result<()> {
        self.inner.ingest_tree_summary(node_id, summary).await
    }
}

/// Result of an ingest call.
#[derive(Debug, Clone, Default, Serialize)]
pub struct IngestSummary {
    /// Documents added or replaced.
    pub ingested: usize,
    /// Documents skipped because their content was unchanged.
    pub unchanged: usize,
    /// Documents rejected, with reasons.
    pub failed: Vec<IngestFailure>,
    /// Chunks written.
    pub chunks: usize,
}

/// A rejected document.
#[derive(Debug, Clone, Serialize)]
pub struct IngestFailure {
    /// Document id as submitted.
    pub doc_id: String,
    /// Why it was rejected.
    pub reason: String,
}

/// One ranked candidate from [`Corpus::search`].
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    /// Chunk id.
    pub chunk_id: String,
    /// Owning document.
    pub doc_id: String,
    /// Source name.
    pub source_doc: String,
    /// Chunk text.
    pub text: String,
    /// Reranker score.
    pub rerank_score: f64,
    /// Reciprocal rank fusion score.
    pub rrf_score: f64,
    /// Conformal nonconformity (lower is better).
    pub nonconformity: f64,
    /// Index that surfaced the chunk.
    pub retriever: RetrieverType,
    /// Whether the chunk passes the calibrated gate at the requested coverage.
    pub in_conformal_set: Option<bool>,
}

/// Counts and configuration for a corpus.
#[derive(Debug, Clone, Serialize)]
pub struct CorpusStats {
    /// Tenant that owns the corpus.
    pub tenant: String,
    /// Corpus name.
    pub name: String,
    /// Live documents.
    pub document_count: usize,
    /// Live chunks.
    pub chunk_count: usize,
    /// Calibration status at the default coverage.
    pub guarantee_status: GuaranteeStatus,
    /// Calibration set id, if calibrated.
    pub calibration_set_id: Option<String>,
    /// Pipeline fingerprint.
    pub pipeline_fingerprint: String,
}

/// A named, tenant-scoped document collection with its indexes and calibration.
pub struct Corpus {
    /// Owning tenant.
    pub tenant: String,
    /// Corpus name.
    pub name: String,
    /// The retrieval pipeline.
    pub orchestrator: Arc<RetrievalOrchestrator>,
    store: Arc<ChunkStore>,
    services: Arc<Services>,
    dir: PathBuf,
    chunk_count: Arc<AtomicU64>,
    ingest_lock: tokio::sync::Mutex<()>,
}

impl Corpus {
    /// Directory holding a corpus's data.
    pub fn dir_for(data_dir: &Path, tenant: &str, name: &str) -> PathBuf {
        data_dir.join("tenants").join(tenant).join(name)
    }

    /// Whether a corpus exists on disk.
    pub fn exists(services: &Services, tenant: &str, name: &str) -> bool {
        Self::dir_for(Path::new(&services.config.storage.data_dir), tenant, name)
            .join("corpus.json")
            .exists()
    }

    /// Open a corpus, creating it when `create` is true and it does not exist.
    pub async fn open_with(
        services: Arc<Services>,
        tenant: &str,
        name: &str,
        create: bool,
    ) -> Result<Self> {
        validate_name("tenant", tenant)?;
        validate_name("corpus", name)?;
        let data_dir = PathBuf::from(&services.config.storage.data_dir);
        let dir = Self::dir_for(&data_dir, tenant, name);
        let marker = dir.join("corpus.json");
        if !marker.exists() {
            if !create {
                return Err(StorageError::Io(format!("corpus '{name}' not found")).into());
            }
            std::fs::create_dir_all(&dir).map_err(|e| StorageError::Io(e.to_string()))?;
            let meta = serde_json::json!({
                "tenant": tenant,
                "name": name,
                "created_at": chrono::Utc::now(),
            });
            std::fs::write(&marker, meta.to_string())
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }

        let cfg = &services.config;
        let store = Arc::new(ChunkStore::open(dir.join("chunks.jsonl"))?);

        let backend = match cfg.vector.backend.as_str() {
            "qdrant" => VectorBackendConfig::Qdrant {
                url: cfg.qdrant.url.clone(),
                api_key: cfg.qdrant.api_key.as_ref().map(|k| k.expose().to_string()),
                collection_name: format!("{}{}__{}", cfg.qdrant.collection_prefix, tenant, name),
            },
            _ => VectorBackendConfig::PersistentMemory {
                path: dir.join("vectors.bin"),
            },
        };
        let vector = VectorIndex::new(VectorIndexConfig {
            backend,
            embedding_dimensions: services.embedder.dimensions(),
            ..Default::default()
        })
        .await?;
        let bm25 = Bm25Index::new(Bm25IndexConfig {
            index_dir: dir.join("bm25"),
            ..Bm25IndexConfig::default()
        })?;

        let hydrate = |inner: Arc<dyn RetrievalIndex>| -> Arc<dyn RetrievalIndex> {
            Arc::new(HydratingIndex {
                inner,
                store: store.clone(),
            })
        };
        let experimental = cfg.retrieval.experimental_indexes;
        let indexes = IndexSet {
            vector: hydrate(Arc::new(vector)),
            bm25: hydrate(Arc::new(bm25)),
            tree: experimental
                .then(|| hydrate(Arc::new(TreeIndex::new(TreeIndexConfig::default())))),
            graph: experimental
                .then(|| hydrate(Arc::new(GraphIndex::new(GraphIndexConfig::default())))),
        };

        let chunk_count = Arc::new(AtomicU64::new(store.chunk_count() as u64));
        let orchestrator = RetrievalOrchestrator {
            cache: Arc::new(noop::NoopCache),
            router: Arc::new(QueryRouter::new(RoutingConfig::default())),
            query_processor: Arc::new(QueryProcessor::new(
                services.fast_llm.clone(),
                Arc::new(EmbeddingBridge(services.embedder.clone())),
                QueryProcessingConfig {
                    expansion_enabled: cfg.retrieval.query_expansion,
                    ..Default::default()
                },
            )),
            acl_resolver: Arc::new(noop::PublicAclResolver),
            indexes,
            reranker: services.reranker.clone(),
            calibration: Arc::new(CorpusCalibration::open(dir.join("calibration"))?),
            llm: services.llm.clone(),
            signer: services.signer.clone(),
            settings: PipelineSettings {
                search_limit: cfg.retrieval.search_limit,
                rerank_top_n: cfg.retrieval.rerank_top_n,
                rrf_k: cfg.retrieval.rrf_k,
                max_evidence_chunks: cfg.retrieval.max_evidence_chunks,
                max_answer_tokens: cfg.llm.max_answer_tokens,
                growth_factor: cfg.calibration.recalibration_corpus_growth_factor,
                drift_tolerance: cfg.calibration.drift_tolerance,
                reranker_name: services.reranker_id.clone(),
                embedding_model: services.embedder.model_id(),
                fingerprint: services.fingerprint(),
            },
            chunk_count: chunk_count.clone(),
            tenant: tenant.to_string(),
        };

        Ok(Self {
            tenant: tenant.to_string(),
            name: name.to_string(),
            orchestrator: Arc::new(orchestrator),
            store,
            services,
            dir,
            chunk_count,
            ingest_lock: tokio::sync::Mutex::new(()),
        })
    }

    /// Create a corpus in the default tenant (embedded, single-tenant use).
    pub async fn create(name: &str, config: SievrantConfig) -> Result<Self> {
        let services = Arc::new(Services::from_config(config)?);
        if Self::exists(&services, DEFAULT_TENANT, name) {
            return Err(StorageError::Io(format!("corpus '{name}' already exists")).into());
        }
        Self::open_with(services, DEFAULT_TENANT, name, true).await
    }

    /// Open an existing corpus in the default tenant.
    pub async fn open(name: &str, config: SievrantConfig) -> Result<Self> {
        let services = Arc::new(Services::from_config(config)?);
        Self::open_with(services, DEFAULT_TENANT, name, false).await
    }

    /// Delete the corpus and all its data (vectors, chunks, calibration).
    pub async fn destroy(self) -> Result<()> {
        let _guard = self.ingest_lock.lock().await;
        let ids: Vec<String> = self
            .store
            .list(0, usize::MAX)
            .into_iter()
            .flat_map(|d| d.chunk_ids)
            .collect();
        // Remote vector stores are cleared explicitly; local files go with the directory.
        let _ = self.orchestrator.indexes.vector.delete(&ids).await;
        drop(_guard);
        let dir = self.dir.clone();
        drop(self);
        std::fs::remove_dir_all(&dir).map_err(|e| StorageError::Io(e.to_string()).into())
    }

    /// Ingest documents. Changed documents replace their previous version;
    /// unchanged ones are skipped. Documents without ACL groups are `public`.
    pub async fn ingest(&self, docs: Vec<Document>) -> Result<IngestSummary> {
        let _guard = self.ingest_lock.lock().await;
        let mut summary = IngestSummary::default();
        let pipeline = sievrant_orchestrator::ingestor::IngestionPipeline::new(Arc::new(
            self.services.config.clone(),
        ));

        struct Pending {
            entry: DocumentEntry,
            chunks: Vec<StoredChunk>,
            replaced: Vec<String>,
        }
        let mut pending: Vec<Pending> = Vec::new();

        for mut doc in docs {
            if doc.id.trim().is_empty() || doc.id.len() > 256 {
                summary.failed.push(IngestFailure {
                    doc_id: doc.id.clone(),
                    reason: "id must be 1-256 characters".into(),
                });
                continue;
            }
            if doc.content.trim().is_empty() {
                summary.failed.push(IngestFailure {
                    doc_id: doc.id.clone(),
                    reason: "content is empty".into(),
                });
                continue;
            }
            doc.acl_groups.retain(|g| !g.trim().is_empty());
            if doc.acl_groups.is_empty() {
                doc.acl_groups = vec!["public".into()];
            }
            let hash = sievrant_core::ingestion::content_hash(&doc.content);
            let acl_key = doc.acl_groups.join(",");
            let version = format!("{hash}|{acl_key}");
            let existing = self.store.doc(&doc.id);
            if existing.as_ref().is_some_and(|e| e.version_hash == version) {
                summary.unchanged += 1;
                continue;
            }
            let (_, enriched) = pipeline.ingest(doc.clone(), &self.name).await?;
            if enriched.is_empty() {
                summary.failed.push(IngestFailure {
                    doc_id: doc.id.clone(),
                    reason: "document produced no chunks".into(),
                });
                continue;
            }
            let chunks: Vec<StoredChunk> = enriched
                .into_iter()
                .map(|ec| StoredChunk {
                    chunk_id: ec.chunk.id,
                    text: ec.chunk.text,
                    metadata: ec.metadata,
                })
                .collect();
            pending.push(Pending {
                entry: DocumentEntry {
                    doc_id: doc.id.clone(),
                    version_hash: version,
                    title: doc.metadata.title.clone(),
                    source: doc
                        .metadata
                        .filename
                        .clone()
                        .or(doc.metadata.source_url.clone()),
                    acl_groups: doc.acl_groups.clone(),
                    ingested_at: chrono::Utc::now(),
                    chunk_ids: chunks.iter().map(|c| c.chunk_id.clone()).collect(),
                },
                chunks,
                replaced: existing.map(|e| e.chunk_ids).unwrap_or_default(),
            });
        }

        if pending.is_empty() {
            return Ok(summary);
        }

        let texts: Vec<String> = pending
            .iter()
            .flat_map(|p| p.chunks.iter().map(|c| c.text.clone()))
            .collect();
        let embeddings = self.services.embedder.embed_batch(&texts).await?;
        let mut emb_iter = embeddings.into_iter();

        let replaced: Vec<String> = pending.iter().flat_map(|p| p.replaced.clone()).collect();
        if !replaced.is_empty() {
            self.orchestrator.indexes.vector.delete(&replaced).await?;
            self.orchestrator.indexes.bm25.delete(&replaced).await?;
        }

        let index_chunks: Vec<IndexChunk> = pending
            .iter()
            .flat_map(|p| p.chunks.iter())
            .map(|c| IndexChunk {
                chunk_id: c.chunk_id.clone(),
                text: c.text.clone(),
                embedding: emb_iter.next(),
                metadata: c.metadata.clone(),
            })
            .collect();
        self.orchestrator
            .indexes
            .vector
            .insert(&index_chunks)
            .await?;
        self.orchestrator.indexes.bm25.insert(&index_chunks).await?;

        // The store is written last: a crash before this leaves only orphans
        // in the indexes, which hydration filters out.
        for p in pending {
            summary.chunks += p.chunks.len();
            summary.ingested += 1;
            self.store.put(p.entry, p.chunks)?;
        }
        self.chunk_count
            .store(self.store.chunk_count() as u64, Ordering::Relaxed);
        Ok(summary)
    }

    /// Ingest a single document (compatibility helper).
    pub async fn add_document(&self, doc: Document) -> Result<IngestSummary> {
        self.ingest(vec![doc]).await
    }

    /// Delete a document and its chunks from every index. Returns false if absent.
    pub async fn delete_document(&self, doc_id: &str) -> Result<bool> {
        let _guard = self.ingest_lock.lock().await;
        let Some(ids) = self.store.doc(doc_id).map(|d| d.chunk_ids) else {
            return Ok(false);
        };
        self.orchestrator.indexes.vector.delete(&ids).await?;
        self.orchestrator.indexes.bm25.delete(&ids).await?;
        self.store.delete(doc_id)?;
        self.chunk_count
            .store(self.store.chunk_count() as u64, Ordering::Relaxed);
        Ok(true)
    }

    /// Full pipeline: retrieve, gate, (optionally) answer, audit and sign.
    pub async fn retrieve(&self, query: &str, config: RetrievalConfig) -> Result<EvidencePacket> {
        let packet = self
            .orchestrator
            .retrieve(query, &self.name, config)
            .await?;
        if let Some(audit) = &self.services.audit {
            if let Err(e) = audit.log_retrieval(&self.tenant, &self.name, &packet).await {
                tracing::error!(error = %e, "audit log write failed");
            }
        }
        Ok(packet)
    }

    /// Ranked candidates without gating or generation (evaluation, debugging,
    /// and clients that apply their own policy). ACLs are enforced.
    pub async fn search(&self, query: &str, config: RetrievalConfig) -> Result<Vec<SearchHit>> {
        let acl: Vec<String> = match &config.acl_groups {
            Some(g) if !g.is_empty() => g.clone(),
            _ => vec!["public".into()],
        };
        let scored = self.orchestrator.score(query, &config, Some(&acl)).await?;
        let tau = self
            .orchestrator
            .calibration
            .decide(
                config.coverage,
                &self.orchestrator.settings.fingerprint,
                self.store.chunk_count() as u64,
                self.orchestrator.settings.growth_factor,
            )
            .tau
            .filter(|_| !scored.degraded_rerank);
        Ok(scored
            .candidates
            .into_iter()
            .map(|c| {
                let nc = nonconformity(c.rerank_score);
                SearchHit {
                    retriever: scored
                        .retrievers
                        .get(&c.chunk_id)
                        .cloned()
                        .unwrap_or(RetrieverType::VectorIndex),
                    chunk_id: c.chunk_id,
                    doc_id: c.metadata.doc_id,
                    source_doc: c.metadata.source_doc,
                    text: c.text,
                    rerank_score: c.rerank_score,
                    rrf_score: c.fused_score,
                    nonconformity: nc,
                    in_conformal_set: tau.map(|t| nc <= t),
                }
            })
            .collect())
    }

    /// Record which documents or chunks were relevant to a served query.
    pub fn feedback(
        &self,
        query_id: &str,
        relevant_doc_ids: &[String],
        relevant_chunk_ids: &[String],
    ) -> Result<Option<FeedbackOutcome>> {
        self.orchestrator.calibration.submit_feedback(
            query_id,
            relevant_doc_ids,
            relevant_chunk_ids,
        )
    }

    /// Queries the corpus could not answer, most frequent first.
    pub fn gaps(&self, limit: usize) -> Vec<GapEntry> {
        self.orchestrator.calibration.gaps(limit)
    }

    /// Documents in the corpus, sorted by id.
    pub fn documents(&self, offset: usize, limit: usize) -> Vec<DocumentEntry> {
        self.store.list(offset, limit)
    }

    /// Counts and calibration summary.
    pub fn stats(&self) -> CorpusStats {
        let status = self.calibration_status(self.services.config.retrieval.coverage);
        CorpusStats {
            tenant: self.tenant.clone(),
            name: self.name.clone(),
            document_count: self.store.document_count(),
            chunk_count: self.store.chunk_count(),
            guarantee_status: status.status,
            calibration_set_id: status.set_id,
            pipeline_fingerprint: status.pipeline_fingerprint,
        }
    }

    /// Shared provider services.
    pub fn services(&self) -> &Arc<Services> {
        &self.services
    }
}

/// Retrieve with default settings.
pub async fn retrieve(query: &str, corpus: &Corpus) -> Result<EvidencePacket> {
    corpus.retrieve(query, RetrievalConfig::default()).await
}

/// Evidence-only retrieval: gated evidence, no LLM answer.
pub async fn evidence(query: &str, corpus: &Corpus) -> Result<EvidencePacket> {
    corpus
        .retrieve(
            query,
            RetrievalConfig {
                generate_answer: false,
                ..RetrievalConfig::default()
            },
        )
        .await
}

/// Fluent builder for a retrieval request.
pub struct RetrieveBuilder<'a> {
    query: &'a str,
    corpus: &'a Corpus,
    config: RetrievalConfig,
}

impl<'a> RetrieveBuilder<'a> {
    /// Start building a retrieval request.
    pub fn new(query: &'a str, corpus: &'a Corpus) -> Self {
        Self {
            query,
            corpus,
            config: RetrievalConfig::default(),
        }
    }

    /// Coverage target, e.g. 0.95.
    pub fn coverage(mut self, coverage: f64) -> Self {
        self.config.coverage = coverage;
        self
    }

    /// Maximum tokens across returned evidence.
    pub fn budget(mut self, tokens: usize) -> Self {
        self.config.token_budget = tokens;
        self
    }

    /// Override the adaptive router.
    pub fn route(mut self, route: RouteStrategy) -> Self {
        self.config.route = route;
        self
    }

    /// End-user identity (recorded; ACLs come from [`Self::acl_groups`]).
    pub fn user(mut self, user_id: &str) -> Self {
        self.config.user_id = Some(user_id.to_string());
        self
    }

    /// ACL groups the end user may read.
    pub fn acl_groups(mut self, groups: Vec<String>) -> Self {
        self.config.acl_groups = Some(groups);
        self
    }

    /// Return evidence only, without an LLM answer.
    pub fn evidence_only(mut self) -> Self {
        self.config.generate_answer = false;
        self
    }

    /// Bypass the semantic cache.
    pub fn skip_cache(mut self, skip: bool) -> Self {
        self.config.skip_cache = skip;
        self
    }

    /// Execute the request.
    pub async fn await_result(self) -> Result<EvidencePacket> {
        self.corpus.retrieve(self.query, self.config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::config::Secret;

    /// Offline configuration: local embeddings, embedding reranker, no LLM.
    fn offline_config(dir: &Path) -> SievrantConfig {
        let mut cfg = SievrantConfig::default();
        cfg.storage.data_dir = dir.to_string_lossy().to_string();
        cfg.llm.provider = "none".into();
        cfg.embedding.provider = "local".into();
        cfg.reranker.provider = "embedding".into();
        cfg.audit.signing_key = Some(Secret::new(sievrant_audit::ReceiptSigner::generate_seed()));
        cfg
    }

    fn doc(id: &str, text: &str, acl: &[&str]) -> Document {
        Document {
            id: id.into(),
            content: text.into(),
            metadata: DocumentMetadata::default(),
            acl_groups: acl.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn topics() -> Vec<(&'static str, &'static str)> {
        vec![
            ("lease", "The lease term is sixty months starting on the commencement date of the agreement between landlord and tenant."),
            ("rent", "Monthly rent is four thousand dollars payable on the first business day of each calendar month by bank transfer."),
            ("deposit", "The security deposit equals two months of rent and is refundable within thirty days after the tenant vacates."),
            ("pets", "Pets are permitted only with written consent of the landlord and an additional nonrefundable cleaning fee."),
            ("parking", "Each tenant receives one assigned parking space in the underground garage at no additional monthly charge."),
            ("repairs", "The landlord is responsible for structural repairs while the tenant handles minor maintenance and light bulbs."),
            ("termination", "Either party may terminate early with ninety days written notice and payment of a two month penalty."),
            ("insurance", "The tenant must maintain renters insurance covering personal property and liability of at least one million dollars."),
            ("utilities", "Water and trash are included in rent while electricity gas and internet are paid directly by the tenant."),
            ("sublease", "Subleasing any portion of the premises requires prior written approval that the landlord may not unreasonably withhold."),
        ]
    }

    fn questions() -> Vec<(&'static str, &'static str)> {
        vec![
            ("how long is the lease term in months", "lease"),
            ("what is the monthly rent payable amount", "rent"),
            ("how much is the security deposit refundable", "deposit"),
            ("are pets permitted with consent", "pets"),
            ("do tenants get an assigned parking space", "parking"),
            ("who handles structural repairs and maintenance", "repairs"),
            (
                "can either party terminate early with notice",
                "termination",
            ),
            (
                "what renters insurance must the tenant maintain",
                "insurance",
            ),
            ("which utilities are included in rent", "utilities"),
            ("is subleasing the premises allowed", "sublease"),
        ]
    }

    async fn seeded(dir: &Path) -> Corpus {
        let c = Corpus::create("leases", offline_config(dir)).await.unwrap();
        let docs = topics()
            .into_iter()
            .map(|(id, t)| doc(id, t, &[]))
            .collect();
        let s = c.ingest(docs).await.unwrap();
        assert_eq!(s.ingested, 10);
        c
    }

    #[test]
    fn names_are_validated() {
        assert!(validate_name("corpus", "legal-docs_2026").is_ok());
        for bad in ["", "../etc", "a/b", "sp ace", &"x".repeat(65)] {
            assert!(validate_name("corpus", bad).is_err(), "{bad}");
        }
    }

    #[tokio::test]
    async fn ingest_is_idempotent_and_replaces_changed_documents() {
        let dir = tempfile::TempDir::new().unwrap();
        let c = seeded(dir.path()).await;
        let again = c
            .ingest(
                topics()
                    .into_iter()
                    .map(|(id, t)| doc(id, t, &[]))
                    .collect(),
            )
            .await
            .unwrap();
        assert_eq!(again.unchanged, 10);
        assert_eq!(again.ingested, 0);

        c.ingest(vec![doc("lease", "The lease term is twelve months.", &[])])
            .await
            .unwrap();
        let hits = c
            .search("lease term months", RetrievalConfig::default())
            .await
            .unwrap();
        let lease: Vec<&SearchHit> = hits.iter().filter(|h| h.doc_id == "lease").collect();
        assert_eq!(lease.len(), 1);
        assert!(lease[0].text.contains("twelve"));
    }

    #[tokio::test]
    async fn uncalibrated_retrieval_is_honest_and_signed() {
        let dir = tempfile::TempDir::new().unwrap();
        let c = seeded(dir.path()).await;
        let p = c
            .retrieve("how long is the lease term", RetrievalConfig::default())
            .await
            .unwrap();
        assert_eq!(p.decision, Decision::EvidenceOnly);
        assert_eq!(p.coverage_guarantee, None);
        assert_eq!(p.guarantee.status, GuaranteeStatus::Uncalibrated);
        assert_eq!(p.evidence[0].source_doc, "lease");
        assert!(p.receipt.is_some());
    }

    #[tokio::test]
    async fn calibration_enables_the_guarantee_and_feedback_is_tracked() {
        let dir = tempfile::TempDir::new().unwrap();
        let c = seeded(dir.path()).await;
        let mut examples = Vec::new();
        for _ in 0..3 {
            for (q, d) in questions() {
                examples.push(CalibrationExample {
                    query: q.into(),
                    relevant_doc_ids: vec![d.into()],
                    relevant_chunk_ids: vec![],
                });
            }
        }
        examples.push(CalibrationExample {
            query: "what is the capital of mongolia".into(),
            relevant_doc_ids: vec![],
            relevant_chunk_ids: vec![],
        });
        let report = c
            .calibrate(examples, CalibrationSource::Labeled)
            .await
            .unwrap();
        assert_eq!(report.answerable, 30);
        assert_eq!(report.unanswerable, 1);
        assert!(report.retrieval_recall > 0.99, "{report:?}");

        let cfg = RetrievalConfig {
            coverage: 0.9,
            ..Default::default()
        };
        let p = c
            .retrieve("how long is the lease term", cfg.clone())
            .await
            .unwrap();
        assert_eq!(p.guarantee.status, GuaranteeStatus::Calibrated);
        assert_eq!(p.coverage_guarantee, Some(0.9));
        assert!(p.evidence.iter().any(|e| e.source_doc == "lease"));

        let out = c
            .feedback(&p.query_id.0, &["lease".into()], &[])
            .unwrap()
            .unwrap();
        assert_eq!(out.covered, Some(true));
        let status = c.calibration_status(0.9);
        assert_eq!(status.observed.unwrap().observations, 1);

        // New corpus content beyond the growth factor makes calibration stale.
        let extra: Vec<Document> = (0..20)
            .map(|i| {
                doc(
                    &format!("extra{i}"),
                    &format!("Unrelated appendix number {i} about fire safety drills."),
                    &[],
                )
            })
            .collect();
        c.ingest(extra).await.unwrap();
        let p = c.retrieve("how long is the lease term", cfg).await.unwrap();
        assert_eq!(p.guarantee.status, GuaranteeStatus::Stale);
        assert_eq!(p.coverage_guarantee, None);
    }

    #[tokio::test]
    async fn acl_groups_isolate_documents() {
        let dir = tempfile::TempDir::new().unwrap();
        let c = Corpus::create("acl", offline_config(dir.path()))
            .await
            .unwrap();
        c.ingest(vec![
            doc("pub", "Office hours are nine to five on weekdays.", &[]),
            doc(
                "sec",
                "Office merger codename is bluebird, confidential.",
                &["legal"],
            ),
        ])
        .await
        .unwrap();
        let hits = c
            .search("office codename", RetrievalConfig::default())
            .await
            .unwrap();
        assert!(hits.iter().all(|h| h.doc_id != "sec"));
        let legal = RetrievalConfig {
            acl_groups: Some(vec!["legal".into()]),
            ..Default::default()
        };
        let hits = c.search("office codename", legal).await.unwrap();
        assert!(hits.iter().all(|h| h.doc_id == "sec"));
    }

    #[tokio::test]
    async fn delete_and_reopen_persist() {
        let dir = tempfile::TempDir::new().unwrap();
        {
            let c = seeded(dir.path()).await;
            assert!(c.delete_document("rent").await.unwrap());
            assert!(!c.delete_document("rent").await.unwrap());
        }
        let c = Corpus::open("leases", offline_config(dir.path()))
            .await
            .unwrap();
        assert_eq!(c.stats().document_count, 9);
        let hits = c
            .search("monthly rent amount", RetrievalConfig::default())
            .await
            .unwrap();
        assert!(hits.iter().all(|h| h.doc_id != "rent"));
        assert!(hits.iter().any(|h| h.doc_id == "lease"));
    }

    #[tokio::test]
    async fn create_twice_fails_and_open_missing_fails() {
        let dir = tempfile::TempDir::new().unwrap();
        let _c = Corpus::create("x", offline_config(dir.path()))
            .await
            .unwrap();
        assert!(Corpus::create("x", offline_config(dir.path()))
            .await
            .is_err());
        assert!(Corpus::open("nope", offline_config(dir.path()))
            .await
            .is_err());
    }
}
