//! Signed retrieval receipts.
//!
//! A receipt is an Ed25519 signature over a JSON payload that binds the
//! query, the evidence (by text hash), the answer and the coverage guarantee
//! that applied. The payload is stored verbatim next to the signature, so
//! verifiers in any language only need an Ed25519 library and SHA-256; there
//! are no canonicalisation rules to reproduce.
//!
//! Key: `SIEVRANT_SIGNING_KEY` = base64 of a 32-byte Ed25519 seed. Generate
//! one with `sievrant keygen`.

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use sievrant_core::error::{AuditError, Result, SievrantError};
use sievrant_core::types::{EvidencePacket, Receipt};

fn err(msg: impl Into<String>) -> SievrantError {
    SievrantError::Audit(AuditError::Serialization(msg.into()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Short identifier for a public key: first 16 hex chars of its SHA-256.
pub fn key_id(public: &VerifyingKey) -> String {
    sha256_hex(public.as_bytes())[..16].to_string()
}

/// Evidence entry inside a receipt payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReceiptEvidence {
    pub chunk_id: String,
    pub doc_version_hash: String,
    pub text_sha256: String,
}

/// What a receipt attests to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReceiptPayload {
    pub v: u32,
    pub issued_at: String,
    pub tenant: String,
    pub corpus: String,
    pub query_id: String,
    pub query_sha256: String,
    pub decision: String,
    pub answer_sha256: Option<String>,
    pub coverage_guarantee: Option<f64>,
    pub guarantee_status: String,
    pub threshold: Option<f64>,
    pub calibration_set_id: Option<String>,
    pub model_version: Option<String>,
    pub pipeline_fingerprint: Option<String>,
    pub evidence: Vec<ReceiptEvidence>,
}

impl ReceiptPayload {
    pub fn from_packet(tenant: &str, corpus: &str, query: &str, p: &EvidencePacket) -> Self {
        Self {
            v: 1,
            issued_at: chrono::Utc::now().to_rfc3339(),
            tenant: tenant.to_string(),
            corpus: corpus.to_string(),
            query_id: p.query_id.0.clone(),
            query_sha256: sha256_hex(query.as_bytes()),
            decision: serde_json::to_value(&p.decision)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default(),
            answer_sha256: p.answer.as_ref().map(|a| sha256_hex(a.as_bytes())),
            coverage_guarantee: p.coverage_guarantee,
            guarantee_status: serde_json::to_value(p.guarantee.status)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default(),
            threshold: p.guarantee.threshold,
            calibration_set_id: p.guarantee.calibration_set_id.clone(),
            model_version: p.audit.model_version.clone(),
            pipeline_fingerprint: p.audit.pipeline_fingerprint.clone(),
            evidence: p
                .evidence
                .iter()
                .map(|e| ReceiptEvidence {
                    chunk_id: e.chunk_id.clone(),
                    doc_version_hash: e.doc_version_hash.clone(),
                    text_sha256: sha256_hex(e.text.as_bytes()),
                })
                .collect(),
        }
    }
}

/// Holds the signing key.
pub struct ReceiptSigner {
    key: SigningKey,
    key_id: String,
}

impl ReceiptSigner {
    /// Parse a base64-encoded 32-byte seed.
    pub fn from_base64_seed(seed_b64: &str) -> Result<Self> {
        let bytes = B64
            .decode(seed_b64.trim())
            .map_err(|e| err(format!("signing key is not valid base64: {e}")))?;
        let seed: [u8; 32] = bytes
            .try_into()
            .map_err(|_| err("signing key must decode to exactly 32 bytes"))?;
        let key = SigningKey::from_bytes(&seed);
        let key_id = key_id(&key.verifying_key());
        Ok(Self { key, key_id })
    }

    /// Generate a fresh seed, returned as base64.
    pub fn generate_seed() -> String {
        let key = SigningKey::generate(&mut rand_core::OsRng);
        B64.encode(key.to_bytes())
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    /// Base64 of the 32-byte public key.
    pub fn public_key_base64(&self) -> String {
        B64.encode(self.key.verifying_key().as_bytes())
    }

    pub fn sign(&self, payload: &ReceiptPayload) -> Result<Receipt> {
        let payload = serde_json::to_string(payload).map_err(|e| err(e.to_string()))?;
        let signature = self.key.sign(payload.as_bytes());
        Ok(Receipt {
            alg: "ed25519".into(),
            key_id: self.key_id.clone(),
            payload,
            signature: B64.encode(signature.to_bytes()),
        })
    }
}

/// Outcome of verifying a packet's receipt.
#[derive(Debug, Clone, Serialize)]
pub struct ReceiptCheck {
    pub signature_valid: bool,
    /// Payload matches the packet's answer, evidence, decision and guarantee.
    pub packet_matches: bool,
    /// `Some(true/false)` when the original query text was supplied.
    pub query_matches: Option<bool>,
    pub problems: Vec<String>,
}

impl ReceiptCheck {
    pub fn ok(&self) -> bool {
        self.signature_valid && self.packet_matches && self.query_matches != Some(false)
    }
}

/// Verify a packet's receipt against a base64 public key.
pub fn verify_packet(
    packet: &EvidencePacket,
    public_key_b64: &str,
    query: Option<&str>,
) -> Result<ReceiptCheck> {
    let receipt = packet
        .receipt
        .as_ref()
        .ok_or_else(|| err("packet has no receipt"))?;
    if receipt.alg != "ed25519" {
        return Err(err(format!("unsupported receipt alg {}", receipt.alg)));
    }
    let pk_bytes: [u8; 32] = B64
        .decode(public_key_b64.trim())
        .map_err(|e| err(format!("public key is not valid base64: {e}")))?
        .try_into()
        .map_err(|_| err("public key must decode to 32 bytes"))?;
    let public = VerifyingKey::from_bytes(&pk_bytes).map_err(|e| err(e.to_string()))?;
    let sig_bytes: [u8; 64] = B64
        .decode(&receipt.signature)
        .map_err(|e| err(format!("signature is not valid base64: {e}")))?
        .try_into()
        .map_err(|_| err("signature must decode to 64 bytes"))?;

    let mut problems = Vec::new();
    let signature_valid = public
        .verify(
            receipt.payload.as_bytes(),
            &Signature::from_bytes(&sig_bytes),
        )
        .is_ok();
    if !signature_valid {
        problems.push("signature does not match payload and key".into());
    }
    if receipt.key_id != key_id(&public) {
        problems.push("receipt key_id does not match the supplied public key".into());
    }

    let signed: ReceiptPayload =
        serde_json::from_str(&receipt.payload).map_err(|e| err(e.to_string()))?;
    let fresh = ReceiptPayload::from_packet(&signed.tenant, &signed.corpus, "", packet);
    let mut check = |ok: bool, what: &str| {
        if !ok {
            problems.push(format!("{what} differs from the signed receipt"));
        }
        ok
    };
    let packet_matches = [
        check(signed.query_id == fresh.query_id, "query_id"),
        check(signed.decision == fresh.decision, "decision"),
        check(signed.answer_sha256 == fresh.answer_sha256, "answer"),
        check(signed.evidence == fresh.evidence, "evidence"),
        check(
            signed.coverage_guarantee == fresh.coverage_guarantee,
            "coverage_guarantee",
        ),
        check(
            signed.guarantee_status == fresh.guarantee_status,
            "guarantee status",
        ),
        check(signed.threshold == fresh.threshold, "threshold"),
    ]
    .iter()
    .all(|x| *x);

    let query_matches = query.map(|q| {
        let ok = sha256_hex(q.as_bytes()) == signed.query_sha256;
        if !ok {
            problems.push("query text does not match the signed query hash".into());
        }
        ok
    });

    Ok(ReceiptCheck {
        signature_valid: signature_valid && receipt.key_id == key_id(&public),
        packet_matches,
        query_matches,
        problems,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sievrant_core::types::*;

    fn packet() -> EvidencePacket {
        EvidencePacket {
            schema_version: EVIDENCE_PACKET_SCHEMA_VERSION.into(),
            query_id: QueryId("q-1".into()),
            trace_id: None,
            answer: Some("60 months [d1:0:0]".into()),
            decision: Decision::Answered,
            coverage_guarantee: Some(0.95),
            guarantee: GuaranteeInfo {
                threshold: Some(0.4),
                calibration_set_id: Some("cal".into()),
                ..GuaranteeInfo::none(GuaranteeStatus::Calibrated, 0.95, "ok")
            },
            empirical_coverage_last_30d: None,
            route: Route::Fast,
            cache_hit: false,
            evidence: vec![EvidenceChunk {
                chunk_id: "d1:0:0".into(),
                source_doc: "lease.pdf".into(),
                doc_version_hash: "sha256:abc".into(),
                path: vec![],
                page: None,
                char_span: None,
                text: "The lease term is 60 months.".into(),
                retriever: RetrieverType::VectorIndex,
                rrf_score: 0.03,
                rerank_score: Some(0.8),
                nonconformity_score: 0.2,
                submodular_marginal_gain: 0.5,
                ingested_at: chrono::Utc::now(),
                index_generation: 1,
                acl_groups: vec!["public".into()],
                freshness_hours: 0.0,
                token_count: 8,
            }],
            audit: AuditBlock::default(),
            latency_ms: 5,
            cost_usd: None,
            receipt: None,
        }
    }

    fn signed(signer: &ReceiptSigner) -> EvidencePacket {
        let mut p = packet();
        let payload = ReceiptPayload::from_packet("acme", "contracts", "What is the term?", &p);
        p.receipt = Some(signer.sign(&payload).unwrap());
        p
    }

    #[test]
    fn valid_receipt_verifies_and_survives_json_round_trip() {
        let signer = ReceiptSigner::from_base64_seed(&ReceiptSigner::generate_seed()).unwrap();
        let p = signed(&signer);
        let json = serde_json::to_string(&p).unwrap();
        let back: EvidencePacket = serde_json::from_str(&json).unwrap();
        let check = verify_packet(
            &back,
            &signer.public_key_base64(),
            Some("What is the term?"),
        )
        .unwrap();
        assert!(check.ok(), "{:?}", check.problems);
    }

    #[test]
    fn tampered_answer_or_evidence_is_detected() {
        let signer = ReceiptSigner::from_base64_seed(&ReceiptSigner::generate_seed()).unwrap();
        let mut p = signed(&signer);
        p.answer = Some("120 months".into());
        let check = verify_packet(&p, &signer.public_key_base64(), None).unwrap();
        assert!(check.signature_valid);
        assert!(!check.packet_matches);

        let mut p = signed(&signer);
        p.evidence[0].text = "The lease term is 12 months.".into();
        assert!(!verify_packet(&p, &signer.public_key_base64(), None)
            .unwrap()
            .ok());
    }

    #[test]
    fn wrong_key_or_query_fails() {
        let signer = ReceiptSigner::from_base64_seed(&ReceiptSigner::generate_seed()).unwrap();
        let other = ReceiptSigner::from_base64_seed(&ReceiptSigner::generate_seed()).unwrap();
        let p = signed(&signer);
        assert!(
            !verify_packet(&p, &other.public_key_base64(), None)
                .unwrap()
                .signature_valid
        );
        let check =
            verify_packet(&p, &signer.public_key_base64(), Some("another question")).unwrap();
        assert_eq!(check.query_matches, Some(false));
        assert!(!check.ok());
    }

    #[test]
    fn bad_seed_is_rejected() {
        assert!(ReceiptSigner::from_base64_seed("not base64!").is_err());
        assert!(ReceiptSigner::from_base64_seed(&B64.encode([0u8; 16])).is_err());
    }
}
