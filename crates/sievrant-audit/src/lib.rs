pub mod hasher;
pub mod receipt;
pub mod trail;

pub use receipt::{verify_packet, ReceiptCheck, ReceiptPayload, ReceiptSigner};
pub use trail::{AuditLogger, IntegrityVerifier};

// EU AI Act Compliance Mapping:
// - Article 12 (Record-keeping): trail.rs (AuditLogger) writes a hash-chained,
//   append-only log of every retrieval decision; IntegrityVerifier detects edits,
//   deletions, insertions and reordering.
// - Article 13 (Transparency): receipt.rs signs each Evidence Packet so a third
//   party can verify which evidence, answer and guarantee were produced.
// - Article 14 (Human oversight): The conformal prediction gate's refusal to answer
//   (tracked via Decision in the audit log) acts as the statistical guardrail enabling
//   safe human oversight when confidence is low.
