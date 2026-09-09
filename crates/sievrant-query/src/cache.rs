use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, RwLock,
};

use chrono::{DateTime, Utc};
use dashmap::DashMap;

use sievrant_core::{
    types::{EvidencePacket, Route},
};

/// Configuration for the semantic cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Seconds before a cache entry is considered stale.
    pub ttl_seconds: u64,
    /// Maximum number of entries before LRU eviction runs.
    pub max_entries: usize,
    /// Cosine similarity threshold for semantic cache hit (Tier 2).
    pub semantic_similarity_threshold: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            ttl_seconds: 3600,
            max_entries: 10_000,
            semantic_similarity_threshold: 0.96,
        }
    }
}

/// A single cached EvidencePacket, annotated with access metadata for eviction.
pub struct CacheEntry {
    pub query: String,
    pub corpus: String,
    pub packet: EvidencePacket,
    /// Dense embedding of the query for semantic (Tier 2) lookups.
    pub embedding: Vec<f32>,
    pub created_at: DateTime<Utc>,
    pub last_accessed: AtomicU64,
    pub hit_count: AtomicU64,
    /// Tenant scope. Lookups MUST match tenant_id to prevent data leakage.
    pub tenant_id: Option<String>,
}

/// A lightweight record in the semantic index used to find similar past queries.
struct SemanticRecord {
    cache_key: u64,
    embedding: Vec<f32>,
    tenant_id: Option<String>,
}

struct SemanticCacheIndex {
    records: Vec<SemanticRecord>,
}

impl SemanticCacheIndex {
    fn new() -> Self {
        Self { records: Vec::new() }
    }

    fn insert(&mut self, cache_key: u64, embedding: Vec<f32>, tenant_id: Option<String>) {
        self.records.push(SemanticRecord { cache_key, embedding, tenant_id });
    }

    /// Find the cache key of the most similar stored query above the threshold,
    /// restricted to entries with a matching tenant_id.
    fn find_similar(
        &self,
        query_embedding: &[f32],
        tenant_id: &Option<String>,
        threshold: f32,
    ) -> Option<u64> {
        let mut best_sim = threshold;
        let mut best_key = None;

        for record in &self.records {
            if record.tenant_id.as_ref() != tenant_id.as_ref() {
                continue;
            }
            let sim = cosine_similarity(query_embedding, &record.embedding);
            if sim >= best_sim {
                best_sim = sim;
                best_key = Some(record.cache_key);
            }
        }

        best_key
    }

    fn remove(&mut self, cache_key: u64) {
        self.records.retain(|r| r.cache_key != cache_key);
    }
}

/// Three-tier semantic cache.
///
/// Tier 1: Exact hash match (sub-millisecond DashMap lookup).
/// Tier 2: Cosine similarity match against stored query embeddings (~2ms linear scan).
/// Eviction: LRU-by-last-access when max_entries is exceeded.
///
/// Tenant isolation: every lookup compares tenant_id; a mismatch is a cache miss.
pub struct SemanticCache {
    exact_cache: DashMap<u64, Arc<CacheEntry>>,
    semantic_index: RwLock<SemanticCacheIndex>,
    config: CacheConfig,
}

impl SemanticCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            exact_cache: DashMap::new(),
            semantic_index: RwLock::new(SemanticCacheIndex::new()),
            config,
        }
    }

    pub fn lookup(
        &self,
        query: &str,
        corpus: &str,
        user_id: &Option<String>,
    ) -> Option<EvidencePacket> {
        let key = self.compute_hash(query, corpus, user_id);
        let now_epoch = Utc::now().timestamp() as u64;

        // Tier 1: exact match
        if let Some(entry) = self.exact_cache.get(&key) {
            if !self.is_expired(&entry) && entry.tenant_id.as_ref() == user_id.as_ref() {
                entry.hit_count.fetch_add(1, Ordering::Relaxed);
                entry.last_accessed.store(now_epoch, Ordering::Relaxed);
                let mut packet = entry.packet.clone();
                packet.cache_hit = true;
                packet.route = Route::Cached;
                return Some(packet);
            }
        }

        // Tier 2: semantic similarity (no embedding provided here — the orchestrator
        // must call lookup_with_embedding when it has the query vector available).
        None
    }

    /// Tier 2 lookup: requires the caller to provide the query embedding.
    /// The orchestrator calls this after embedding the query on a cache miss.
    pub fn lookup_with_embedding(
        &self,
        query_embedding: &[f32],
        user_id: &Option<String>,
    ) -> Option<EvidencePacket> {
        let index = self.semantic_index.read().unwrap();
        let candidate_key = index.find_similar(
            query_embedding,
            user_id,
            self.config.semantic_similarity_threshold,
        )?;
        drop(index);

        if let Some(entry) = self.exact_cache.get(&candidate_key) {
            if !self.is_expired(&entry) && entry.tenant_id.as_ref() == user_id.as_ref() {
                entry.hit_count.fetch_add(1, Ordering::Relaxed);
                entry.last_accessed.store(Utc::now().timestamp() as u64, Ordering::Relaxed);
                let mut packet = entry.packet.clone();
                packet.cache_hit = true;
                packet.route = Route::Cached;
                return Some(packet);
            }
        }

        None
    }

    pub fn store(
        &self,
        query: &str,
        corpus: &str,
        user_id: &Option<String>,
        embedding: Vec<f32>,
        packet: EvidencePacket,
    ) {
        // Evict before inserting if at capacity
        if self.exact_cache.len() >= self.config.max_entries {
            self.evict_lru();
        }

        let key = self.compute_hash(query, corpus, user_id);
        let entry = Arc::new(CacheEntry {
            query: query.to_string(),
            corpus: corpus.to_string(),
            packet,
            embedding: embedding.clone(),
            created_at: Utc::now(),
            last_accessed: AtomicU64::new(Utc::now().timestamp() as u64),
            hit_count: AtomicU64::new(0),
            tenant_id: user_id.clone(),
        });

        self.exact_cache.insert(key, entry);

        let mut index = self.semantic_index.write().unwrap();
        index.insert(key, embedding, user_id.clone());
    }

    pub fn invalidate_corpus(&self, corpus: &str) {
        let keys_to_remove: Vec<u64> = self
            .exact_cache
            .iter()
            .filter(|e| e.value().corpus == corpus)
            .map(|e| *e.key())
            .collect();

        let mut index = self.semantic_index.write().unwrap();
        for key in keys_to_remove {
            self.exact_cache.remove(&key);
            index.remove(key);
        }
    }

    fn is_expired(&self, entry: &CacheEntry) -> bool {
        let age = Utc::now() - entry.created_at;
        // use >= so that a TTL of 0 immediately expires entries
        age.num_milliseconds() >= (self.config.ttl_seconds as i64) * 1000
    }

    fn compute_hash(&self, query: &str, corpus: &str, user_id: &Option<String>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        corpus.hash(&mut hasher);
        user_id.hash(&mut hasher);
        hasher.finish()
    }

    /// Evict the least-recently-used entry by last_accessed timestamp.
    fn evict_lru(&self) {
        let oldest = self
            .exact_cache
            .iter()
            .min_by_key(|e| e.value().last_accessed.load(Ordering::Relaxed))
            .map(|e| *e.key());

        if let Some(key) = oldest {
            self.exact_cache.remove(&key);
            let mut index = self.semantic_index.write().unwrap();
            index.remove(key);
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::types::{AuditBlock, Decision, RetrievalStagesMs};

    fn dummy_packet(answer: &str) -> EvidencePacket {
        EvidencePacket {
            query_id: sievrant_core::types::QueryId::new(),
            trace_id: None,
            answer: Some(answer.to_string()),
            decision: Decision::Answered,
            coverage_guarantee: 0.95,
            empirical_coverage_last_30d: None,
            route: Route::Standard,
            cache_hit: false,
            evidence: vec![],
            audit: AuditBlock {
                model_version: None,
                retrieval_stages_ms: RetrievalStagesMs::default(),
                total_candidates_before_rerank: 0,
                total_candidates_after_rerank: 0,
                total_candidates_after_conformal: 0,
                calibration_set_id: "default".into(),
                session_hash: "sha256:test".into(),
            },
            latency_ms: 42,
            cost_usd: None,
        }
    }

    #[test]
    fn exact_cache_hit_and_miss() {
        let cache = SemanticCache::new(CacheConfig::default());
        let packet = dummy_packet("The lease is 60 months.");

        cache.store("what is the lease term?", "corp", &None, vec![0.1, 0.2], packet);

        let hit = cache.lookup("what is the lease term?", "corp", &None);
        assert!(hit.is_some(), "should be a cache hit");
        assert!(hit.unwrap().cache_hit);

        let miss = cache.lookup("different question", "corp", &None);
        assert!(miss.is_none(), "different query should miss");
    }

    #[test]
    fn expired_entry_returns_none() {
        let cache = SemanticCache::new(CacheConfig {
            ttl_seconds: 0,
            ..Default::default()
        });
        let packet = dummy_packet("ans");
        cache.store("q", "c", &None, vec![], packet);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(cache.lookup("q", "c", &None).is_none());
    }

    #[test]
    fn tenant_isolation_prevents_cross_user_hits() {
        let cache = SemanticCache::new(CacheConfig::default());
        let packet = dummy_packet("private answer");

        let user_a = Some("user_a".to_string());
        let user_b = Some("user_b".to_string());

        cache.store("secret query", "corp", &user_a, vec![], packet);

        let user_b_hit = cache.lookup("secret query", "corp", &user_b);
        assert!(
            user_b_hit.is_none(),
            "user B must not see user A's cached result"
        );

        let user_a_hit = cache.lookup("secret query", "corp", &user_a);
        assert!(user_a_hit.is_some(), "user A should hit their own entry");
    }

    #[test]
    fn semantic_tier2_hit_on_similar_embedding() {
        let cache = SemanticCache::new(CacheConfig {
            semantic_similarity_threshold: 0.99,
            ..Default::default()
        });
        let embedding = vec![1.0f32, 0.0, 0.0];
        let packet = dummy_packet("answer");
        cache.store("q", "corp", &None, embedding.clone(), packet);

        // Slightly perturbed embedding: cosine similarity is still 1.0 for this vector
        let hit = cache.lookup_with_embedding(&embedding, &None);
        assert!(hit.is_some(), "identical embedding should be a Tier 2 hit");
    }

    #[test]
    fn lru_eviction_removes_oldest_entry() {
        let cache = SemanticCache::new(CacheConfig {
            max_entries: 2,
            ..Default::default()
        });
        cache.store("q1", "c", &None, vec![], dummy_packet("a1"));
        cache.store("q2", "c", &None, vec![], dummy_packet("a2"));
        // Third insert should evict one of the first two
        cache.store("q3", "c", &None, vec![], dummy_packet("a3"));
        assert!(cache.exact_cache.len() <= 2);
    }
}
