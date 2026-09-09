"""Tests for Pydantic models."""

from __future__ import annotations

from sievrant.models import (
    Decision,
    Route,
    RetrieverType,
    EvidencePacket,
    EvidenceChunk,
    AuditBlock,
    RetrievalStagesMs,
)


def _make_chunk(chunk_id: str = "doc:sec:p") -> dict:
    return {
        "chunk_id": chunk_id,
        "source_doc": "test.pdf",
        "doc_version_hash": "sha256:abc",
        "path": ["Section 1"],
        "page": 1,
        "char_span": None,
        "text": "Test content.",
        "retriever": "vector_index",
        "rrf_score": 0.016,
        "rerank_score": 0.91,
        "nonconformity_score": 0.09,
        "submodular_marginal_gain": 0.5,
        "ingested_at": "2025-01-01T00:00:00Z",
        "index_generation": 1,
        "acl_groups": ["public"],
        "freshness_hours": 24.0,
        "token_count": 5,
    }


def test_evidence_chunk_parses_correctly() -> None:
    chunk = EvidenceChunk.model_validate(_make_chunk())
    assert chunk.retriever == RetrieverType.VECTOR_INDEX
    assert chunk.rrf_score == 0.016
    assert chunk.acl_groups == ["public"]


def test_decision_enum_values() -> None:
    assert Decision.ANSWERED == "ANSWERED"
    assert Decision.ABSTAINED == "ABSTAINED"
    assert Decision.EVIDENCE_ONLY == "EVIDENCE_ONLY"


def test_route_enum_values() -> None:
    assert Route.FAST == "FAST"
    assert Route.CACHED == "CACHED"


def test_retriever_type_enum_values() -> None:
    assert RetrieverType.VECTOR_INDEX == "vector_index"
    assert RetrieverType.BM25_INDEX == "bm25_index"
    assert RetrieverType.GRAPH_INDEX == "graph_index"


def test_evidence_packet_round_trip() -> None:
    data = {
        "query_id": "test-001",
        "trace_id": None,
        "decision": "ANSWERED",
        "answer": "The rent is $4,500.",
        "coverage_guarantee": 0.95,
        "empirical_coverage_last_30d": None,
        "route": "FAST",
        "cache_hit": False,
        "latency_ms": 50,
        "cost_usd": None,
        "evidence": [_make_chunk()],
        "audit": {
            "model_version": "gpt-4o",
            "retrieval_stages_ms": {},
            "total_candidates_before_rerank": 100,
            "total_candidates_after_rerank": 10,
            "total_candidates_after_conformal": 1,
            "calibration_set_id": "default",
            "session_hash": "sha256:test",
        },
    }
    packet = EvidencePacket.model_validate(data)
    assert packet.query_id == "test-001"
    assert packet.decision == Decision.ANSWERED
    assert len(packet.evidence) == 1
    assert packet.audit.session_hash == "sha256:test"
    serialized = packet.model_dump()
    assert serialized["query_id"] == "test-001"
