"""
Pydantic models that mirror the Rust type definitions in sievrant-core verbatim.
Every field name matches the JSON wire format exactly.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class Decision(str, Enum):
    """Outcome of a single retrieval call."""

    ANSWERED = "ANSWERED"
    """The conformal gate passed and an answer was generated."""

    ABSTAINED = "ABSTAINED"
    """The conformal gate rejected all evidence; no answer produced."""

    EVIDENCE_ONLY = "EVIDENCE_ONLY"
    """Evidence retrieved but LLM generation was skipped."""


class Route(str, Enum):
    """Which pipeline path the query took."""

    FAST = "FAST"
    """Single-pass hybrid search, no query expansion."""

    STANDARD = "STANDARD"
    """Full pipeline with expansion but no multi-hop decomposition."""

    SLOW = "SLOW"
    """Multi-hop decomposition with iterative sub-query retrieval."""

    CACHED = "CACHED"
    """Served from semantic cache; no retrieval executed."""


class RetrieverType(str, Enum):
    """Which index contributed a specific evidence chunk."""

    VECTOR_INDEX = "vector_index"
    BM25_INDEX = "bm25_index"
    TREE_INDEX = "tree_index"
    GRAPH_INDEX = "graph_index"


class RetrievalStagesMs(BaseModel):
    """Per-pipeline-stage latency breakdown in milliseconds."""

    cache_check: int = 0
    routing: int = 0
    query_processing: int = 0
    acl_resolution: int = 0
    vector_search: int = 0
    bm25_search: int = 0
    tree_search: int = 0
    graph_search: int = 0
    rrf_fusion: int = 0
    reranking: int = 0
    submodular_selection: int = 0
    conformal_gate: int = 0
    llm_generation: int = 0


class AuditBlock(BaseModel):
    """Compliance and debugging block attached to every EvidencePacket.

    Satisfies EU AI Act Articles 12 and 13 record-keeping requirements.
    """

    model_version: str | None = None
    """LLM model identifier, e.g. 'gpt-4o-2024-08-06'."""

    retrieval_stages_ms: RetrievalStagesMs = Field(default_factory=RetrievalStagesMs)
    """Per-stage wall-clock latency breakdown across 13 pipeline stages."""

    total_candidates_before_rerank: int = 0
    """Candidate count entering the reranker (post-RRF)."""

    total_candidates_after_rerank: int = 0
    """Candidate count after reranking, before submodular selection."""

    total_candidates_after_conformal: int = 0
    """Evidence count that passed the conformal gate."""

    calibration_set_id: str = "default"
    """Identifies the calibration set used to derive the conformal threshold."""

    session_hash: str = ""
    """SHA-256 of (query_id || query_text || evidence_ids || answer || model || timestamp).
    Tamper-evident hash for audit log integrity verification.
    """


class EvidenceChunk(BaseModel):
    """A single retrieved evidence chunk with full provenance and scoring metadata.

    This is the atomic unit of the EvidencePacket. Every field has a named consumer.
    """

    chunk_id: str
    """Globally unique identifier: 'doc_id:section_idx:chunk_idx'."""

    source_doc: str
    """Source document filename or URI."""

    doc_version_hash: str
    """SHA-256 of source document content at ingestion time."""

    path: list[str] = Field(default_factory=list)
    """Hierarchical path within the document, e.g. ['Section 4', '4.2 Term', 'paragraph 3']."""

    page: int | None = None
    """Page number in the source document, if the format supports pagination."""

    char_span: tuple[int, int] | None = None
    """Character span [start, end] in the source document for inline highlighting."""

    text: str
    """The actual text content of this chunk."""

    retriever: RetrieverType
    """Which index surfaced this chunk."""

    rrf_score: float
    """Score from Reciprocal Rank Fusion. Always positive."""

    rerank_score: float | None = None
    """Score from the cross-encoder reranker. None if reranking was skipped."""

    nonconformity_score: float
    """Conformal non-conformity score. Lower is more trustworthy."""

    submodular_marginal_gain: float
    """Marginal information gain assigned by submodular selection."""

    ingested_at: str
    """ISO 8601 timestamp when this chunk was written into the corpus."""

    index_generation: int
    """LSM merge generation at the time this chunk was indexed."""

    acl_groups: list[str] = Field(default_factory=list)
    """ACL groups this chunk belongs to. Populated at ingestion; verified at query time."""

    freshness_hours: float
    """Hours elapsed since this chunk was last ingested or updated."""

    token_count: int
    """Approximate token count used for budget accounting in submodular selection."""


class EvidencePacket(BaseModel):
    """The primary output of every Sievrant retrieval call.

    This struct is both the API response and the compliance artifact (EU AI Act Art. 12/13).
    """

    query_id: str
    """Unique ID for this retrieval request."""

    trace_id: str | None = None
    """OpenTelemetry trace ID. Set when OTel is enabled."""

    answer: str | None = None
    """Generated answer text. None when decision is ABSTAINED or EVIDENCE_ONLY."""

    decision: Decision
    """Retrieval outcome. Drives agent branching."""

    coverage_guarantee: float
    """Nominal conformal coverage target configured for this call."""

    empirical_coverage_last_30d: float | None = None
    """Empirical coverage rate over the last 30 days. None until enough data accumulates."""

    route: Route
    """Which pipeline path this query took."""

    cache_hit: bool
    """True when this packet was served from the semantic cache."""

    evidence: list[EvidenceChunk] = Field(default_factory=list)
    """Evidence chunks that passed submodular selection and the conformal gate."""

    audit: AuditBlock = Field(default_factory=AuditBlock)
    """Full compliance and debugging audit block."""

    latency_ms: int
    """Total wall-clock latency from query receipt to packet serialization."""

    cost_usd: float | None = None
    """Estimated USD cost. None when cost tracking is disabled or a local model is used."""

    @field_validator("decision", "route", mode="before")
    @classmethod
    def uppercase_enums(cls, v: str) -> str:
        if isinstance(v, str):
            return v.upper()
        return v


class HealthResponse(BaseModel):
    """Response from the /v1/health endpoint."""

    status: str
    version: str


class IngestResponse(BaseModel):
    """Response from the /v1/ingest endpoint."""

    corpus: str
    ingested: int
    failed: int


class CorpusMetadata(BaseModel):
    """Metadata about a document corpus."""

    name: str
    created_at: str
    document_count: int
    chunk_count: int
    index_generation: int
    calibration_set_id: str | None = None
