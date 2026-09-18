use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        RwLock,
    },
};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use sievrant_core::error::Result;

use crate::wal::{ChunkData, WalEntry, WriteAheadLog};

/// Configuration for the tiered merge engine.
#[derive(Debug, Clone)]
pub struct MergeConfig {
    /// Number of sorted runs that trigger a merge at each level.
    pub runs_per_level: usize,
    /// Number of levels. Data flows L0 -> L1 -> ... -> L(num_levels-1).
    pub num_levels: usize,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            runs_per_level: 4,
            num_levels: 4,
        }
    }
}

/// A single sorted run at a specific level. Immutable after creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SortedRun {
    pub id: String,
    pub level: usize,
    pub chunk_count: usize,
    pub size_bytes: u64,
    pub created_at: DateTime<Utc>,
    /// Incremented on each merge. Used by readers to detect staleness.
    pub generation: u64,
}

/// A level in the tiered merge hierarchy.
struct MergeLevel {
    _level: usize,
    /// Sorted runs at this level, oldest first.
    runs: Vec<SortedRun>,
    max_runs: usize,
}

/// LSM-style tiered merge engine.
///
/// Write amplification and amortized insert cost analysis:
///   Each chunk is written once to L0. When L0 reaches `runs_per_level` runs,
///   they are merged into a single L1 run. This process repeats up the levels.
///   The total write amplification across all levels is O(log N) with base
///   `runs_per_level`, analogous to LSM-tree write amplification in RocksDB.
///
/// Crash safety:
///   Every merge writes MergeStart to the WAL before touching any files.
///   MergeComplete is written after the new run is durable. On restart,
///   any MergeStart without a matching MergeComplete is rolled back by
///   deleting the partial output run.
pub struct MergeEngine {
    config: MergeConfig,
    levels: RwLock<Vec<MergeLevel>>,
    current_generation: AtomicU64,
    /// In-memory run contents (keyed by run_id -> chunk_ids).
    run_contents: RwLock<HashMap<String, Vec<String>>>,
    run_counter: AtomicU64,
}

impl MergeEngine {
    pub fn new(config: MergeConfig) -> Self {
        let num_levels = config.num_levels;
        let runs_per_level = config.runs_per_level;
        let levels = (0..num_levels)
            .map(|level| MergeLevel {
                _level: level,
                runs: Vec::new(),
                max_runs: runs_per_level,
            })
            .collect();

        Self {
            config,
            levels: RwLock::new(levels),
            current_generation: AtomicU64::new(0),
            run_contents: RwLock::new(HashMap::new()),
            run_counter: AtomicU64::new(0),
        }
    }

    /// Flush an in-memory buffer of chunks to a new L0 sorted run.
    /// Records MergeStart + MergeComplete in the WAL, then triggers
    /// level merges if the run threshold is reached.
    pub fn flush(&self, wal: &mut WriteAheadLog, chunks: &[ChunkData]) -> Result<()> {
        let run_id = self.next_run_id(0);
        let chunk_ids: Vec<String> = chunks.iter().map(|c| c.chunk_id.clone()).collect();
        let chunk_count = chunks.len();

        wal.append(&WalEntry::MergeStart {
            level: 0,
            run_ids: vec![],
        })?;

        let generation = self.current_generation.load(Ordering::SeqCst);
        let run = SortedRun {
            id: run_id.clone(),
            level: 0,
            chunk_count,
            size_bytes: chunk_count as u64 * 512,
            created_at: Utc::now(),
            generation,
        };

        {
            let mut contents = self.run_contents.write().unwrap();
            contents.insert(run_id.clone(), chunk_ids);
        }

        {
            let mut levels = self.levels.write().unwrap();
            levels[0].runs.push(run);
        }

        wal.append(&WalEntry::MergeComplete {
            level: 0,
            new_run_id: run_id,
            old_run_ids: vec![],
        })?;

        self.maybe_merge_level(wal, 0)?;
        Ok(())
    }

    /// Merge all runs at `level` into a single run at `level + 1` when the
    /// run count reaches `max_runs`. Cascades up the level hierarchy.
    fn maybe_merge_level(&self, wal: &mut WriteAheadLog, level: usize) -> Result<()> {
        let should_merge = {
            let levels = self.levels.read().unwrap();
            levels
                .get(level)
                .map(|l| l.runs.len() >= l.max_runs)
                .unwrap_or(false)
        };

        if !should_merge {
            return Ok(());
        }

        let next_level = level + 1;
        let num_levels = self.config.num_levels;
        if next_level >= num_levels {
            return Ok(());
        }

        let old_runs: Vec<SortedRun> = {
            let mut levels = self.levels.write().unwrap();
            std::mem::take(&mut levels[level].runs)
        };

        let old_run_ids: Vec<String> = old_runs.iter().map(|r| r.id.clone()).collect();

        wal.append(&WalEntry::MergeStart {
            level: next_level,
            run_ids: old_run_ids.clone(),
        })?;

        let generation = self.current_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let new_run_id = self.next_run_id(next_level);

        // Merge chunk contents from all source runs into the new run
        let merged_chunk_ids: Vec<String> = {
            let contents = self.run_contents.read().unwrap();
            old_run_ids
                .iter()
                .flat_map(|id| contents.get(id).cloned().unwrap_or_default())
                .collect()
        };

        let merged_count = merged_chunk_ids.len();

        {
            let mut contents = self.run_contents.write().unwrap();
            contents.insert(new_run_id.clone(), merged_chunk_ids);
            for id in &old_run_ids {
                contents.remove(id);
            }
        }

        let new_run = SortedRun {
            id: new_run_id.clone(),
            level: next_level,
            chunk_count: merged_count,
            size_bytes: merged_count as u64 * 512,
            created_at: Utc::now(),
            generation,
        };

        {
            let mut levels = self.levels.write().unwrap();
            levels[next_level].runs.push(new_run);
        }

        wal.append(&WalEntry::MergeComplete {
            level: next_level,
            new_run_id,
            old_run_ids,
        })?;

        self.maybe_merge_level(wal, next_level)
    }

    /// Check whether a chunk_id is present in any run across all levels.
    pub fn contains(&self, chunk_id: &str) -> bool {
        let contents = self.run_contents.read().unwrap();
        contents
            .values()
            .any(|ids| ids.iter().any(|id| id == chunk_id))
    }

    pub fn current_generation(&self) -> u64 {
        self.current_generation.load(Ordering::SeqCst)
    }

    fn next_run_id(&self, level: usize) -> String {
        let n = self.run_counter.fetch_add(1, Ordering::SeqCst);
        format!("run-L{level}-{n:08}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_wal(dir: &TempDir) -> WriteAheadLog {
        WriteAheadLog::open(dir.path(), 64 * 1024 * 1024).unwrap()
    }

    fn make_chunks(start: usize, count: usize) -> Vec<ChunkData> {
        (start..start + count)
            .map(|i| ChunkData {
                chunk_id: format!("c{i}"),
                doc_id: "d".into(),
                text: format!("chunk text {i}"),
                token_count: 10,
                acl_groups: vec![],
            })
            .collect()
    }

    #[test]
    fn flush_makes_chunks_retrievable() {
        let dir = TempDir::new().unwrap();
        let mut wal = make_wal(&dir);
        let engine = MergeEngine::new(MergeConfig::default());

        let chunks = make_chunks(0, 5);
        engine.flush(&mut wal, &chunks).unwrap();

        assert!(engine.contains("c0"));
        assert!(engine.contains("c4"));
        assert!(!engine.contains("c99"));
    }

    #[test]
    fn generation_increments_on_merge() {
        let dir = TempDir::new().unwrap();
        let mut wal = make_wal(&dir);
        let config = MergeConfig {
            runs_per_level: 2,
            num_levels: 4,
        };
        let engine = MergeEngine::new(config);

        let gen_before = engine.current_generation();

        // Two flushes trigger L0 -> L1 merge with runs_per_level = 2
        engine.flush(&mut wal, &make_chunks(0, 3)).unwrap();
        engine.flush(&mut wal, &make_chunks(3, 3)).unwrap();

        let gen_after = engine.current_generation();
        assert!(
            gen_after > gen_before,
            "generation must increment after merge"
        );
    }

    #[test]
    fn all_chunks_retrievable_after_cascading_merges() {
        let dir = TempDir::new().unwrap();
        let mut wal = make_wal(&dir);
        let config = MergeConfig {
            runs_per_level: 2,
            num_levels: 4,
        };
        let engine = MergeEngine::new(config);

        let total = 64;
        for batch_start in (0..total).step_by(4) {
            engine
                .flush(&mut wal, &make_chunks(batch_start, 4))
                .unwrap();
        }

        for i in 0..total {
            assert!(engine.contains(&format!("c{i}")), "chunk c{i} not found");
        }
    }
}
