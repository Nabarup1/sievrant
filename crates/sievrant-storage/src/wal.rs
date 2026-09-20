use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use sievrant_core::error::{Result, StorageError};

const WAL_MAGIC: u32 = 0x57414C30; // "WAL0"
const ENTRY_HEADER_SIZE: usize = 12; // 4 magic + 4 len + 4 checksum

/// A raw chunk being written to the WAL.
/// Carries only the fields needed for replay; does not hold embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkData {
    pub chunk_id: String,
    pub doc_id: String,
    pub text: String,
    pub token_count: usize,
    pub acl_groups: Vec<String>,
}

/// Every durably committed operation is recorded as a WAL entry.
/// Merge operations require both a Start and a Complete entry; a Start
/// with no matching Complete is treated as an incomplete merge on restart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalEntry {
    InsertChunks {
        corpus: String,
        chunks: Vec<ChunkData>,
    },
    DeleteChunks {
        corpus: String,
        chunk_ids: Vec<String>,
    },
    MergeStart {
        level: usize,
        run_ids: Vec<String>,
    },
    MergeComplete {
        level: usize,
        new_run_id: String,
        old_run_ids: Vec<String>,
    },
    /// Written after all previous entries have been flushed to the indexes.
    /// On restart, replay begins from the entry after the last Checkpoint.
    Checkpoint {
        generation: u64,
    },
}

/// Write-Ahead Log for durable ingestion and merge crash safety.
///
/// File layout per segment:
///   [WAL_MAGIC: 4 bytes] [entry_len: 4 bytes LE] [checksum: 4 bytes] [payload: entry_len bytes] ...
///
/// Segment files are named `wal-{segment_counter:08}.log` and rotated
/// when the current segment exceeds `max_segment_size` bytes.
pub struct WriteAheadLog {
    log_dir: PathBuf,
    current_segment: BufWriter<File>,
    segment_counter: u64,
    current_size: u64,
    max_segment_size: u64,
}

impl WriteAheadLog {
    /// Open or create a WAL at the given directory.
    pub fn open(log_dir: impl Into<PathBuf>, max_segment_size: u64) -> Result<Self> {
        let log_dir: PathBuf = log_dir.into();
        fs::create_dir_all(&log_dir).map_err(|e| StorageError::Io(e.to_string()))?;

        let segment_counter = find_latest_segment(&log_dir);
        let segment_path = segment_path(&log_dir, segment_counter);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&segment_path)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        let current_size = file.metadata().map(|m| m.len()).unwrap_or(0);

        Ok(Self {
            log_dir,
            current_segment: BufWriter::new(file),
            segment_counter,
            current_size,
            max_segment_size,
        })
    }

    /// Serialize and append a WAL entry. Flushes to the OS buffer on every call.
    pub fn append(&mut self, entry: &WalEntry) -> Result<()> {
        if self.current_size >= self.max_segment_size {
            self.rotate_segment()?;
        }

        let payload =
            serde_json::to_vec(entry).map_err(|e| StorageError::Serialization(e.to_string()))?;

        let payload_len = payload.len() as u32;
        let checksum = crc32_checksum(&payload);

        self.current_segment
            .write_all(&WAL_MAGIC.to_le_bytes())
            .map_err(|e| StorageError::WalWrite(e.to_string()))?;
        self.current_segment
            .write_all(&payload_len.to_le_bytes())
            .map_err(|e| StorageError::WalWrite(e.to_string()))?;
        self.current_segment
            .write_all(&checksum.to_le_bytes())
            .map_err(|e| StorageError::WalWrite(e.to_string()))?;
        self.current_segment
            .write_all(&payload)
            .map_err(|e| StorageError::WalWrite(e.to_string()))?;

        self.current_segment
            .flush()
            .map_err(|e| StorageError::WalWrite(e.to_string()))?;

        self.current_size += ENTRY_HEADER_SIZE as u64 + payload.len() as u64;
        Ok(())
    }

    /// Write a Checkpoint entry and flush. All entries before the most recent
    /// Checkpoint can be safely truncated after compaction.
    pub fn checkpoint(&mut self, generation: u64) -> Result<()> {
        self.append(&WalEntry::Checkpoint { generation })
    }

    /// Replay all WAL entries across all segments in order, starting from the
    /// entry after the last Checkpoint (if any). Returns ordered list of entries.
    ///
    /// Used on startup to recover committed but unflushed writes and to detect
    /// incomplete merge operations (MergeStart without matching MergeComplete).
    pub fn replay(&self) -> Result<Vec<WalEntry>> {
        let mut segments = list_segment_paths(&self.log_dir);
        segments.sort();

        let mut all_entries: Vec<WalEntry> = Vec::new();

        for segment_path in &segments {
            let file = File::open(segment_path).map_err(|e| StorageError::Io(e.to_string()))?;
            let mut reader = BufReader::new(file);
            let entries = read_entries_from_segment(&mut reader, segment_path)?;
            all_entries.extend(entries);
        }

        // Find the last checkpoint and keep only entries after it.
        let last_checkpoint_pos = all_entries
            .iter()
            .rposition(|e| matches!(e, WalEntry::Checkpoint { .. }));

        if let Some(pos) = last_checkpoint_pos {
            Ok(all_entries.into_iter().skip(pos + 1).collect())
        } else {
            Ok(all_entries)
        }
    }

    /// Remove all WAL segments that were written before the last Checkpoint.
    /// Safe to call after all entries have been flushed to the durable indexes.
    pub fn truncate_before_checkpoint(&self) -> Result<()> {
        let mut segments = list_segment_paths(&self.log_dir);
        segments.sort();

        let mut last_checkpoint_segment = None;

        for segment_path in &segments {
            let file = File::open(segment_path).map_err(|e| StorageError::Io(e.to_string()))?;
            let mut reader = BufReader::new(file);
            if let Ok(entries) = read_entries_from_segment(&mut reader, segment_path) {
                if entries
                    .iter()
                    .any(|e| matches!(e, WalEntry::Checkpoint { .. }))
                {
                    last_checkpoint_segment = Some(segment_path.clone());
                }
            }
        }

        if let Some(last) = last_checkpoint_segment {
            for segment_path in &segments {
                if segment_path < &last {
                    let _ = fs::remove_file(segment_path);
                }
            }
        }

        Ok(())
    }

    fn rotate_segment(&mut self) -> Result<()> {
        self.current_segment
            .flush()
            .map_err(|e| StorageError::WalWrite(e.to_string()))?;

        self.segment_counter += 1;
        let path = segment_path(&self.log_dir, self.segment_counter);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        self.current_segment = BufWriter::new(file);
        self.current_size = 0;
        Ok(())
    }
}

fn segment_path(dir: &Path, counter: u64) -> PathBuf {
    dir.join(format!("wal-{counter:08}.log"))
}

fn find_latest_segment(dir: &Path) -> u64 {
    list_segment_paths(dir)
        .iter()
        .filter_map(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.strip_prefix("wal-"))
                .and_then(|n| n.parse::<u64>().ok())
        })
        .max()
        .unwrap_or(0)
}

fn list_segment_paths(dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension().map(|e| e == "log").unwrap_or(false)
                        && p.file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.starts_with("wal-"))
                            .unwrap_or(false)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn read_entries_from_segment(reader: &mut BufReader<File>, path: &Path) -> Result<Vec<WalEntry>> {
    let mut entries = Vec::new();

    loop {
        let mut header = [0u8; ENTRY_HEADER_SIZE];
        match reader.read_exact(&mut header) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(StorageError::Io(e.to_string()).into()),
        }

        let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
        if magic != WAL_MAGIC {
            return Err(StorageError::WalCorrupted {
                segment: 0,
                reason: format!("bad magic in {}", path.display()),
            }
            .into());
        }

        let payload_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let expected_checksum = u32::from_le_bytes(header[8..12].try_into().unwrap());

        let mut payload = vec![0u8; payload_len];
        reader
            .read_exact(&mut payload)
            .map_err(|e| StorageError::Io(e.to_string()))?;

        let actual_checksum = crc32_checksum(&payload);
        if actual_checksum != expected_checksum {
            return Err(StorageError::ChecksumMismatch {
                path: path.display().to_string(),
                expected: expected_checksum.to_string(),
                actual: actual_checksum.to_string(),
            }
            .into());
        }

        let entry: WalEntry = serde_json::from_slice(&payload)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;

        entries.push(entry);
    }

    Ok(entries)
}

/// Simple CRC-32/Adler-style checksum for WAL entry integrity.
/// Not a cryptographic hash; purpose is to detect accidental corruption.
fn crc32_checksum(data: &[u8]) -> u32 {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    u32::from_le_bytes(result[..4].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tmp_wal() -> (TempDir, WriteAheadLog) {
        let dir = TempDir::new().unwrap();
        let wal = WriteAheadLog::open(dir.path(), 64 * 1024 * 1024).unwrap();
        (dir, wal)
    }

    #[test]
    fn append_and_replay_round_trips_entries() {
        let (_dir, mut wal) = tmp_wal();

        let entry = WalEntry::InsertChunks {
            corpus: "test_corpus".into(),
            chunks: vec![ChunkData {
                chunk_id: "c1".into(),
                doc_id: "d1".into(),
                text: "The lease term is 60 months.".into(),
                token_count: 8,
                acl_groups: vec!["public".into()],
            }],
        };

        wal.append(&entry).unwrap();
        wal.append(&WalEntry::Checkpoint { generation: 1 }).unwrap();
        wal.append(&WalEntry::InsertChunks {
            corpus: "test_corpus".into(),
            chunks: vec![],
        })
        .unwrap();

        let replayed = wal.replay().unwrap();
        // Only the entry after the checkpoint should be replayed
        assert_eq!(replayed.len(), 1);
        assert!(matches!(replayed[0], WalEntry::InsertChunks { .. }));
    }

    #[test]
    fn corrupt_checksum_returns_error() {
        let dir = TempDir::new().unwrap();
        let segment = dir.path().join("wal-00000000.log");

        // Write a valid header + bad payload
        let mut file = File::create(&segment).unwrap();
        file.write_all(&WAL_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap(); // wrong checksum
        file.write_all(b"junk").unwrap();

        let wal = WriteAheadLog::open(dir.path(), 64 * 1024 * 1024).unwrap();
        let result = wal.replay();
        assert!(result.is_err());
    }

    #[test]
    fn segment_rotates_on_size_limit() {
        let dir = TempDir::new().unwrap();
        let mut wal = WriteAheadLog::open(dir.path(), 200).unwrap(); // tiny limit

        for i in 0..10 {
            wal.append(&WalEntry::Checkpoint { generation: i }).unwrap();
        }

        let segments = list_segment_paths(dir.path());
        assert!(
            segments.len() > 1,
            "segment rotation should have created multiple files"
        );
    }

    #[test]
    fn replay_after_crash_recovers_all_committed_entries() {
        let dir = TempDir::new().unwrap();

        {
            let mut wal = WriteAheadLog::open(dir.path(), 64 * 1024 * 1024).unwrap();
            wal.append(&WalEntry::InsertChunks {
                corpus: "corp".into(),
                chunks: vec![ChunkData {
                    chunk_id: "x".into(),
                    doc_id: "d".into(),
                    text: "content".into(),
                    token_count: 1,
                    acl_groups: vec![],
                }],
            })
            .unwrap();
            // Simulate crash: no checkpoint written, no explicit close
        }

        // Re-open as if recovering after crash
        let wal2 = WriteAheadLog::open(dir.path(), 64 * 1024 * 1024).unwrap();
        let recovered = wal2.replay().unwrap();
        assert_eq!(
            recovered.len(),
            1,
            "uncommitted entry should be replayed on restart"
        );
    }

    #[test]
    fn test_append_replay_checkpoint_roundtrip() {
        let dir = TempDir::new().unwrap();
        let mut wal = WriteAheadLog::open(dir.path(), 64 * 1024 * 1024).unwrap();

        let chunk = ChunkData {
            chunk_id: "c1".into(),
            doc_id: "d1".into(),
            text: "text".into(),
            token_count: 1,
            acl_groups: vec![],
        };
        wal.append(&WalEntry::InsertChunks {
            corpus: "corp".into(),
            chunks: vec![chunk.clone()],
        })
        .unwrap();

        // Append checkpoint
        wal.append(&WalEntry::Checkpoint { generation: 42 })
            .unwrap();

        // Replay should return 0 entries since all previous are checkpointed
        let replayed = wal.replay().unwrap();
        assert_eq!(replayed.len(), 0);

        // Append one more
        wal.append(&WalEntry::InsertChunks {
            corpus: "corp".into(),
            chunks: vec![chunk.clone()],
        })
        .unwrap();
        let replayed2 = wal.replay().unwrap();
        assert_eq!(replayed2.len(), 1);
    }

    #[test]
    fn test_segment_rotation() {
        let dir = TempDir::new().unwrap();
        // Use a tiny limit so it rotates quickly
        let mut wal = WriteAheadLog::open(dir.path(), 128).unwrap();

        for i in 0..10 {
            wal.append(&WalEntry::Checkpoint { generation: i }).unwrap();
        }

        let segments = list_segment_paths(dir.path());
        assert!(
            segments.len() >= 2,
            "Expected multiple segments due to rotation, got {}",
            segments.len()
        );
    }
}
