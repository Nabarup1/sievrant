pub mod hasher;
pub mod trail;

// EU AI Act Compliance Mapping:
// - Article 12 (Record-keeping): Handled by trail.rs (AuditLogger), which logs 
//   events including the trace ID, decision, timestamp, latency, and session hash.
// - Article 13 (Transparency): The session hash in hasher.rs cryptographically binds 
//   the inputs and outputs, ensuring the system operation is transparent and tamper-evident.
// - Article 14 (Human oversight): The conformal prediction gate's refusal to answer
//   (tracked via Decision in the audit log) acts as the statistical guardrail enabling
//   safe human oversight when confidence is low.
