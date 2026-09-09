"""
Tests for the Sievrant Python SDK client.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from pytest_httpx import HTTPXMock

from sievrant.client import SievrantClient
from sievrant.models import Decision, Route, RetrieverType
from sievrant.exceptions import (
    AuthError,
    CorpusNotFoundError,
    SievrantAbstainedError,
    SievrantServerError,
)


BASE_URL = "http://localhost:8080"
API_KEY = "sk-test-key"


MOCK_PACKET = {
    "query_id": "01932a7b-0000-0000-0000-000000000001",
    "trace_id": None,
    "decision": "ANSWERED",
    "answer": "The monthly rent is $4,500.",
    "coverage_guarantee": 0.95,
    "empirical_coverage_last_30d": None,
    "route": "FAST",
    "cache_hit": False,
    "latency_ms": 47,
    "cost_usd": None,
    "evidence": [
        {
            "chunk_id": "doc-001:sec_1:p1",
            "source_doc": "lease.pdf",
            "doc_version_hash": "sha256:abc123",
            "path": ["Section 1"],
            "page": 1,
            "char_span": None,
            "text": "Monthly rent is $4,500.",
            "retriever": "vector_index",
            "rrf_score": 0.0164,
            "rerank_score": 0.92,
            "nonconformity_score": 0.08,
            "submodular_marginal_gain": 0.41,
            "ingested_at": "2025-01-01T00:00:00Z",
            "index_generation": 1,
            "acl_groups": ["legal"],
            "freshness_hours": 24.0,
            "token_count": 8,
        }
    ],
    "audit": {
        "model_version": "gpt-4o",
        "retrieval_stages_ms": {
            "cache_check": 1,
            "routing": 2,
            "query_processing": 4,
            "acl_resolution": 1,
            "vector_search": 18,
            "bm25_search": 6,
            "tree_search": 4,
            "graph_search": 8,
            "rrf_fusion": 2,
            "reranking": 21,
            "submodular_selection": 3,
            "conformal_gate": 1,
            "llm_generation": 340,
        },
        "total_candidates_before_rerank": 100,
        "total_candidates_after_rerank": 25,
        "total_candidates_after_conformal": 1,
        "calibration_set_id": "default",
        "session_hash": "sha256:deadbeef",
    },
}

MOCK_HEALTH = {"status": "ok", "version": "0.1.0"}
MOCK_INGEST = {"corpus": "test-corpus", "ingested": 1, "failed": 0}


@pytest.fixture
def client(httpx_mock: HTTPXMock) -> SievrantClient:
    return SievrantClient(base_url=BASE_URL, api_key=API_KEY)


@pytest.mark.asyncio
async def test_health_returns_ok(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/v1/health", json=MOCK_HEALTH)
    async with SievrantClient(base_url=BASE_URL, api_key=API_KEY) as client:
        health = await client.health()
    assert health.status == "ok"
    assert health.version == "0.1.0"


@pytest.mark.asyncio
async def test_query_returns_evidence_packet(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/v1/query", json=MOCK_PACKET)
    async with SievrantClient(base_url=BASE_URL, api_key=API_KEY) as client:
        result = await client.query(corpus="test-corpus", query="What is the rent?")
    assert result.decision == Decision.ANSWERED
    assert result.answer == "The monthly rent is $4,500."
    assert len(result.evidence) == 1
    assert result.evidence[0].chunk_id == "doc-001:sec_1:p1"
    assert result.evidence[0].retriever == RetrieverType.VECTOR_INDEX
    assert result.latency_ms == 47
    assert result.route == Route.FAST


@pytest.mark.asyncio
async def test_query_full_endpoint(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/v1/query/full", json=MOCK_PACKET)
    async with SievrantClient(base_url=BASE_URL, api_key=API_KEY) as client:
        result = await client.query(corpus="test-corpus", query="What is the rent?", full=True)
    assert result.audit.model_version == "gpt-4o"
    assert result.audit.session_hash == "sha256:deadbeef"
    assert result.audit.retrieval_stages_ms.vector_search == 18


@pytest.mark.asyncio
async def test_query_raises_auth_error_on_401(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/query",
        status_code=401,
        json={"error": "unauthorized", "message": "Invalid token"},
    )
    async with SievrantClient(base_url=BASE_URL, api_key=API_KEY) as client:
        with pytest.raises(AuthError):
            await client.query(corpus="test", query="test")


@pytest.mark.asyncio
async def test_query_raises_corpus_not_found_on_404(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/query",
        status_code=404,
        json={"error": "corpus_not_found", "message": "Corpus 'missing' not found"},
    )
    async with SievrantClient(base_url=BASE_URL, api_key=API_KEY) as client:
        with pytest.raises(CorpusNotFoundError):
            await client.query(corpus="missing", query="test")


@pytest.mark.asyncio
async def test_raise_on_abstain(httpx_mock: HTTPXMock) -> None:
    abstained_packet = {**MOCK_PACKET, "decision": "ABSTAINED", "answer": None}
    httpx_mock.add_response(url=f"{BASE_URL}/v1/query", json=abstained_packet)
    async with SievrantClient(base_url=BASE_URL, api_key=API_KEY) as client:
        with pytest.raises(SievrantAbstainedError) as exc_info:
            await client.query(
                corpus="test", query="unanswerable question", raise_on_abstain=True
            )
    assert "01932a7b" in exc_info.value.query_id


@pytest.mark.asyncio
async def test_ingest_returns_response(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{BASE_URL}/v1/ingest", json=MOCK_INGEST)
    async with SievrantClient(base_url=BASE_URL, api_key=API_KEY) as client:
        result = await client.ingest(
            corpus="test-corpus",
            documents=[{"id": "doc-001", "content": "Monthly rent is $4,500."}],
        )
    assert result.ingested == 1
    assert result.failed == 0


@pytest.mark.asyncio
async def test_server_error_raises_sievrant_server_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{BASE_URL}/v1/query",
        status_code=500,
        json={"error": "internal_error", "message": "Something went wrong"},
    )
    async with SievrantClient(base_url=BASE_URL, api_key=API_KEY) as client:
        with pytest.raises(SievrantServerError) as exc_info:
            await client.query(corpus="test", query="test")
    assert exc_info.value.status_code == 500


def test_client_requires_base_url() -> None:
    import os

    os.environ.pop("SIEVRANT_BASE_URL", None)
    with pytest.raises(ValueError, match="base_url is required"):
        SievrantClient(api_key="sk-test")


def test_client_requires_api_key() -> None:
    import os

    os.environ.pop("SIEVRANT_API_KEY", None)
    with pytest.raises(ValueError, match="api_key is required"):
        SievrantClient(base_url="http://localhost:8080")
