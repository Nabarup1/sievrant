use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

/// Compute a tamper-evident session hash.
/// The hash covers all inputs and outputs of a retrieval:
/// SHA256(query_id || query_text || evidence_chunk_ids || answer || model_version || timestamp)
///
/// This enforces EU AI Act Article 12 (Record-keeping) by cryptographically binding
/// the exact state of the pipeline decision to the evidence packet.
pub fn compute_session_hash(
    query_id: &str,
    query: &str,
    evidence_ids: &[String],
    answer: &Option<String>,
    model_version: &Option<String>,
    timestamp: &DateTime<Utc>,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(query_id.as_bytes());
    hasher.update(query.as_bytes());

    for id in evidence_ids {
        hasher.update(id.as_bytes());
    }

    if let Some(a) = answer {
        hasher.update(a.as_bytes());
    }

    if let Some(m) = model_version {
        hasher.update(m.as_bytes());
    }

    // RFC3339 string format ensures deterministic time serialization
    hasher.update(timestamp.to_rfc3339().as_bytes());

    format!("sha256:{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_hash_is_deterministic() {
        let ts = Utc::now();
        let evidence_ids = vec!["chunk1".to_string(), "chunk2".to_string()];

        let h1 = compute_session_hash(
            "q123",
            "What is the revenue?",
            &evidence_ids,
            &Some("The revenue is $10M.".to_string()),
            &Some("claude-3.5-sonnet".to_string()),
            &ts,
        );

        let h2 = compute_session_hash(
            "q123",
            "What is the revenue?",
            &evidence_ids,
            &Some("The revenue is $10M.".to_string()),
            &Some("claude-3.5-sonnet".to_string()),
            &ts,
        );

        assert_eq!(h1, h2, "Hash must be deterministic for identical inputs");

        let h3 = compute_session_hash(
            "q123",
            "What is the revenue?",
            &evidence_ids,
            &Some("The revenue is $11M.".to_string()), // Altered answer
            &Some("claude-3.5-sonnet".to_string()),
            &ts,
        );

        assert_ne!(h1, h3, "Hash must change if answer changes");
    }
}
