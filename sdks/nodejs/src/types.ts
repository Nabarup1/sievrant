/**
 * TypeScript type definitions for Sievrant SDK.
 * These mirror the Rust type definitions in sievrant-core verbatim.
 */

/** Outcome of a single retrieval call. */
export type Decision = 'ANSWERED' | 'ABSTAINED' | 'EVIDENCE_ONLY';

/** Which pipeline path the query took. */
export type Route = 'FAST' | 'STANDARD' | 'SLOW' | 'CACHED';

/** Which index contributed a specific evidence chunk. */
export type RetrieverType = 'vector_index' | 'bm25_index' | 'tree_index' | 'graph_index';

/** Per-pipeline-stage latency breakdown in milliseconds. */
export interface RetrievalStagesMs {
  cache_check: number;
  routing: number;
  query_processing: number;
  acl_resolution: number;
  vector_search: number;
  bm25_search: number;
  tree_search: number;
  graph_search: number;
  rrf_fusion: number;
  reranking: number;
  submodular_selection: number;
  conformal_gate: number;
  llm_generation: number;
}

/** Compliance and debugging block attached to every EvidencePacket. */
export interface AuditBlock {
  /** LLM model identifier, e.g. 'gpt-4o-2024-08-06' */
  model_version: string | null;
  /** Per-stage wall-clock latency breakdown across 13 pipeline stages. */
  retrieval_stages_ms: RetrievalStagesMs;
  /** Candidate count entering the reranker (post-RRF). */
  total_candidates_before_rerank: number;
  /** Candidate count after reranking, before submodular selection. */
  total_candidates_after_rerank: number;
  /** Evidence count that passed the conformal gate. */
  total_candidates_after_conformal: number;
  /** Identifies the calibration set used to derive the conformal threshold. */
  calibration_set_id: string;
  /**
   * SHA-256 of (query_id || query_text || evidence_ids || answer || model || timestamp).
   * Tamper-evident hash for audit log integrity verification.
   */
  session_hash: string;
}

/** A single retrieved evidence chunk with full provenance and scoring metadata. */
export interface EvidenceChunk {
  /** Globally unique identifier: 'doc_id:section_idx:chunk_idx' */
  chunk_id: string;
  /** Source document filename or URI. */
  source_doc: string;
  /** SHA-256 of source document content at ingestion time. */
  doc_version_hash: string;
  /** Hierarchical path within the document. */
  path: string[];
  /** Page number in the source document, if the format supports pagination. */
  page: number | null;
  /** Character span [start, end] in the source document for inline highlighting. */
  char_span: [number, number] | null;
  /** The actual text content of this chunk. */
  text: string;
  /** Which index surfaced this chunk. */
  retriever: RetrieverType;
  /** Score from Reciprocal Rank Fusion. Always positive. */
  rrf_score: number;
  /** Score from the cross-encoder reranker. null if reranking was skipped. */
  rerank_score: number | null;
  /** Conformal non-conformity score. Lower is more trustworthy. */
  nonconformity_score: number;
  /** Marginal information gain assigned by submodular selection. */
  submodular_marginal_gain: number;
  /** ISO 8601 timestamp when this chunk was written into the corpus. */
  ingested_at: string;
  /** LSM merge generation at the time this chunk was indexed. */
  index_generation: number;
  /** ACL groups this chunk belongs to. */
  acl_groups: string[];
  /** Hours elapsed since this chunk was last ingested or updated. */
  freshness_hours: number;
  /** Approximate token count used for budget accounting. */
  token_count: number;
}

/** The primary output of every Sievrant retrieval call. */
export interface EvidencePacket {
  /** Unique ID for this retrieval request. */
  query_id: string;
  /** OpenTelemetry trace ID. Set when OTel is enabled. */
  trace_id: string | null;
  /** Generated answer text. null when decision is ABSTAINED or EVIDENCE_ONLY. */
  answer: string | null;
  /** Retrieval outcome. Drives agent branching. */
  decision: Decision;
  /** Nominal conformal coverage target configured for this call. */
  coverage_guarantee: number;
  /** Empirical coverage rate over the last 30 days. null until enough data accumulates. */
  empirical_coverage_last_30d: number | null;
  /** Which pipeline path this query took. */
  route: Route;
  /** True when this packet was served from the semantic cache. */
  cache_hit: boolean;
  /** Evidence chunks that passed submodular selection and the conformal gate. */
  evidence: EvidenceChunk[];
  /** Full compliance and debugging audit block. */
  audit: AuditBlock;
  /** Total wall-clock latency from query receipt to packet serialization. */
  latency_ms: number;
  /** Estimated USD cost. null when cost tracking is disabled or a local model is used. */
  cost_usd: number | null;
}

/** Response from the /v1/health endpoint. */
export interface HealthResponse {
  status: string;
  version: string;
}

/** Response from the /v1/ingest endpoint. */
export interface IngestResponse {
  corpus: string;
  ingested: number;
  failed: number;
}

/** Metadata about a document corpus. */
export interface CorpusMetadata {
  name: string;
  created_at: string;
  document_count: number;
  chunk_count: number;
  index_generation: number;
  calibration_set_id: string | null;
}

/** Options for the query() method. */
export interface QueryOptions {
  corpus: string;
  query: string;
  coverage?: number;
  tokenBudget?: number;
  aclGroups?: string[];
  skipCache?: boolean;
  route?: 'AUTO' | 'FAST' | 'STANDARD' | 'SLOW';
  full?: boolean;
}

/** Options for the ingest() method. */
export interface IngestOptions {
  corpus: string;
  documents: IngestDocument[];
}

/** A document to ingest. */
export interface IngestDocument {
  id: string;
  content: string;
  metadata?: Record<string, unknown>;
  acl_groups?: string[];
}

/** Options for SievrantClient constructor. */
export interface SievrantClientOptions {
  baseUrl?: string;
  apiKey?: string;
  timeoutMs?: number;
  maxRetries?: number;
  headers?: Record<string, string>;
}
