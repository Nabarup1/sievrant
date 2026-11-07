use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

use chrono::{Datelike, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use sievrant_core::{
    error::{AuditError, Result, SievrantError},
    types::{Decision, EvidencePacket, GuaranteeStatus, QueryId, Route},
};

/// Hash that anchors the first record of each monthly file.
pub const GENESIS_HASH: &str = "sha256:genesis";

fn io_err(e: impl std::fmt::Display) -> SievrantError {
    SievrantError::Audit(AuditError::Io(e.to_string()))
}

fn ser_err(e: impl std::fmt::Display) -> SievrantError {
    SievrantError::Audit(AuditError::Serialization(e.to_string()))
}

/// One retrieval decision in the audit log.
///
/// Records form a hash chain: `record_hash` covers every field including
/// `prev_hash`, the hash of the preceding record. Editing, deleting,
/// inserting or reordering any record breaks every hash after it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecord {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub tenant: String,
    pub corpus: String,
    pub query_id: QueryId,
    pub trace_id: Option<String>,
    pub decision: Decision,
    pub evidence_count: usize,
    pub evidence_ids: Vec<String>,
    pub model_version: Option<String>,
    pub calibration_set_id: String,
    pub guarantee_status: GuaranteeStatus,
    pub coverage_guarantee: Option<f64>,
    pub session_hash: String,
    pub latency_ms: u64,
    pub route: Route,
    pub prev_hash: String,
    pub record_hash: String,
}

impl AuditRecord {
    fn compute_record_hash(&self) -> String {
        let mut unsigned = self.clone();
        unsigned.record_hash = String::new();
        // Struct field order is fixed, so this serialisation is stable.
        let bytes = serde_json::to_vec(&unsigned).unwrap_or_default();
        format!("sha256:{:x}", Sha256::digest(&bytes))
    }
}

struct Writer {
    month_key: (i32, u32),
    file: BufWriter<File>,
    last_hash: String,
}

/// Append-only, hash-chained audit log rotated monthly (`audit_YYYY_MM.log`).
pub struct AuditLogger {
    audit_dir: PathBuf,
    writer: Mutex<Option<Writer>>,
}

impl AuditLogger {
    pub fn new(audit_dir: impl Into<PathBuf>) -> Result<Self> {
        let audit_dir: PathBuf = audit_dir.into();
        fs::create_dir_all(&audit_dir).map_err(io_err)?;
        Ok(Self {
            audit_dir,
            writer: Mutex::new(None),
        })
    }

    fn path_for(&self, year: i32, month: u32) -> PathBuf {
        self.audit_dir.join(format!("audit_{year}_{month:02}.log"))
    }

    /// Hash of the last complete record in `path`, or the genesis hash.
    fn last_hash_in(path: &Path) -> Result<String> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(GENESIS_HASH.to_string())
            }
            Err(e) => return Err(io_err(e)),
        };
        let mut last = GENESIS_HASH.to_string();
        for line in BufReader::new(file).lines() {
            let line = line.map_err(io_err)?;
            if let Ok(rec) = serde_json::from_str::<AuditRecord>(&line) {
                last = rec.record_hash;
            }
        }
        Ok(last)
    }

    /// Append a record for a retrieval.
    pub async fn log_retrieval(
        &self,
        tenant: &str,
        corpus: &str,
        packet: &EvidencePacket,
    ) -> Result<()> {
        let now = Utc::now();
        let key = (now.year(), now.month());
        let mut guard = self.writer.lock().unwrap();

        if guard.as_ref().map(|w| w.month_key) != Some(key) {
            let path = self.path_for(key.0, key.1);
            let last_hash = Self::last_hash_in(&path)?;
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(io_err)?;
            *guard = Some(Writer {
                month_key: key,
                file: BufWriter::new(file),
                last_hash,
            });
        }
        let writer = guard.as_mut().unwrap();

        let mut record = AuditRecord {
            timestamp: now,
            tenant: tenant.to_string(),
            corpus: corpus.to_string(),
            query_id: packet.query_id.clone(),
            trace_id: packet.trace_id.clone(),
            decision: packet.decision.clone(),
            evidence_count: packet.evidence.len(),
            evidence_ids: packet.evidence.iter().map(|e| e.chunk_id.clone()).collect(),
            model_version: packet.audit.model_version.clone(),
            calibration_set_id: packet.audit.calibration_set_id.clone(),
            guarantee_status: packet.guarantee.status,
            coverage_guarantee: packet.coverage_guarantee,
            session_hash: packet.audit.session_hash.clone(),
            latency_ms: packet.latency_ms,
            route: packet.route.clone(),
            prev_hash: writer.last_hash.clone(),
            record_hash: String::new(),
        };
        record.record_hash = record.compute_record_hash();

        let json = serde_json::to_string(&record).map_err(ser_err)?;
        writeln!(writer.file, "{json}").map_err(io_err)?;
        writer.file.flush().map_err(io_err)?;
        writer.last_hash = record.record_hash;
        Ok(())
    }

    /// Flush buffered records to disk.
    pub async fn checkpoint(&self) -> Result<()> {
        if let Some(w) = self.writer.lock().unwrap().as_mut() {
            w.file.flush().map_err(io_err)?;
        }
        Ok(())
    }
}

pub struct IntegrityVerifier {
    _audit_dir: PathBuf,
}

#[derive(Debug)]
pub struct VerificationResult {
    pub file: PathBuf,
    pub is_valid: bool,
    pub record_count: usize,
    pub errors: Vec<String>,
}

impl IntegrityVerifier {
    pub fn new(audit_dir: impl Into<PathBuf>) -> Self {
        Self {
            _audit_dir: audit_dir.into(),
        }
    }

    /// Verify every record hash and the chain linking them.
    pub fn verify_file(&self, path: &Path) -> Result<VerificationResult> {
        let file = File::open(path).map_err(io_err)?;
        let mut count = 0;
        let mut errors = Vec::new();
        let mut expected_prev = GENESIS_HASH.to_string();

        for (i, line) in BufReader::new(file).lines().enumerate() {
            let line = line.map_err(io_err)?;
            let record: AuditRecord = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    errors.push(format!("line {}: unparseable record: {e}", i + 1));
                    continue;
                }
            };
            let expected_hash = record.compute_record_hash();
            if record.record_hash != expected_hash {
                errors.push(format!("line {}: record contents were modified", i + 1));
            }
            if record.prev_hash != expected_prev {
                errors.push(format!(
                    "line {}: chain broken (record removed, inserted or reordered before it)",
                    i + 1
                ));
            }
            expected_prev = record.record_hash.clone();
            count += 1;
        }

        Ok(VerificationResult {
            file: path.to_path_buf(),
            is_valid: errors.is_empty(),
            record_count: count,
            errors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::types::{AuditBlock, GuaranteeInfo, EVIDENCE_PACKET_SCHEMA_VERSION};
    use tempfile::TempDir;

    fn dummy_packet() -> EvidencePacket {
        EvidencePacket {
            schema_version: EVIDENCE_PACKET_SCHEMA_VERSION.into(),
            query_id: QueryId::new(),
            trace_id: None,
            answer: None,
            decision: Decision::Abstained,
            coverage_guarantee: None,
            guarantee: GuaranteeInfo::none(GuaranteeStatus::Uncalibrated, 0.95, "n/a"),
            empirical_coverage_last_30d: None,
            route: Route::Fast,
            cache_hit: false,
            evidence: vec![],
            audit: AuditBlock {
                session_hash: "sha256:abc".into(),
                ..Default::default()
            },
            latency_ms: 10,
            cost_usd: None,
            receipt: None,
        }
    }

    fn current_log(dir: &Path) -> PathBuf {
        let now = Utc::now();
        dir.join(format!("audit_{}_{:02}.log", now.year(), now.month()))
    }

    #[tokio::test]
    async fn logs_retrieval_and_verifies_integrity() {
        let dir = TempDir::new().unwrap();
        let logger = AuditLogger::new(dir.path()).unwrap();
        logger
            .log_retrieval("t", "c", &dummy_packet())
            .await
            .unwrap();
        logger
            .log_retrieval("t", "c", &dummy_packet())
            .await
            .unwrap();
        logger.checkpoint().await.unwrap();

        let result = IntegrityVerifier::new(dir.path())
            .verify_file(&current_log(dir.path()))
            .unwrap();
        assert!(result.is_valid, "{:?}", result.errors);
        assert_eq!(result.record_count, 2);
    }

    #[tokio::test]
    async fn chain_continues_across_logger_restarts() {
        let dir = TempDir::new().unwrap();
        AuditLogger::new(dir.path())
            .unwrap()
            .log_retrieval("t", "c", &dummy_packet())
            .await
            .unwrap();
        AuditLogger::new(dir.path())
            .unwrap()
            .log_retrieval("t", "c", &dummy_packet())
            .await
            .unwrap();
        let result = IntegrityVerifier::new(dir.path())
            .verify_file(&current_log(dir.path()))
            .unwrap();
        assert!(result.is_valid, "{:?}", result.errors);
    }

    #[tokio::test]
    async fn detects_modified_record() {
        let dir = TempDir::new().unwrap();
        let logger = AuditLogger::new(dir.path()).unwrap();
        logger
            .log_retrieval("t", "c", &dummy_packet())
            .await
            .unwrap();
        let path = current_log(dir.path());
        let content = fs::read_to_string(&path).unwrap();
        fs::write(
            &path,
            content.replace("\"latency_ms\":10", "\"latency_ms\":99"),
        )
        .unwrap();

        let result = IntegrityVerifier::new(dir.path())
            .verify_file(&path)
            .unwrap();
        assert!(!result.is_valid);
    }

    #[tokio::test]
    async fn detects_deleted_record() {
        let dir = TempDir::new().unwrap();
        let logger = AuditLogger::new(dir.path()).unwrap();
        for _ in 0..3 {
            logger
                .log_retrieval("t", "c", &dummy_packet())
                .await
                .unwrap();
        }
        let path = current_log(dir.path());
        let lines: Vec<String> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        fs::write(&path, format!("{}\n{}\n", lines[0], lines[2])).unwrap();

        let result = IntegrityVerifier::new(dir.path())
            .verify_file(&path)
            .unwrap();
        assert!(!result.is_valid);
        assert!(result.errors[0].contains("chain broken"));
    }
}
