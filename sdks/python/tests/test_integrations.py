from unittest.mock import AsyncMock, patch

import pytest

from sievrant.integrations.dspy import SievrantRM
from sievrant.integrations.langchain import SievrantRetriever as LangChainRetriever
from sievrant.integrations.llama_index import SievrantRetriever as LlamaIndexRetriever
from sievrant.models import EvidenceChunk, EvidencePacket


@pytest.fixture
def mock_packet():
    return EvidencePacket(
        evidence=[
            EvidenceChunk(
                chunk_id="test_chunk",
                text="Integration test context",
                document_id="test_doc",
                score=0.9,
                metadata={},
                source_doc="test_doc",
                doc_version_hash="hash",
                retriever="vector_index",
                rrf_score=0.9,
                nonconformity_score=0.1,
                submodular_marginal_gain=0.9,
                ingested_at="2023-01-01T00:00:00Z",
                index_generation=1,
                freshness_hours=0.0,
                token_count=10,
            )
        ],
        query="test",
        processing_time_ms=10.0,
        tokens_saved=0,
        generation="1",
        query_id="qid",
        decision="EVIDENCE_ONLY",
        coverage_guarantee=0.9,
        route="FAST",
        cache_hit=False,
        latency_ms=10.0,
    )


@pytest.mark.asyncio
async def test_langchain_retriever(mock_packet):
    with patch("sievrant.integrations.langchain.SievrantClient", autospec=True) as MockClient:
        instance = MockClient.return_value
        instance.__aenter__.return_value = instance
        instance.query = AsyncMock(return_value=mock_packet)
        retriever = LangChainRetriever(base_url="http://mock", api_key="mock", corpus="test_corpus")
        docs = await retriever.ainvoke("test query")
        assert len(docs) == 1
        assert docs[0].page_content == "Integration test context"
        assert docs[0].metadata["rrf_score"] == 0.9


@pytest.mark.asyncio
async def test_llamaindex_retriever(mock_packet):
    with patch("sievrant.integrations.llama_index.SievrantClient", autospec=True) as MockClient:
        instance = MockClient.return_value
        instance.__aenter__.return_value = instance
        instance.query = AsyncMock(return_value=mock_packet)
        retriever = LlamaIndexRetriever(
            base_url="http://mock", api_key="mock", corpus="test_corpus"
        )
        nodes = await retriever.aretrieve("test query")
        assert len(nodes) == 1
        assert nodes[0].get_content() == "Integration test context"
        assert nodes[0].score == 0.9


@pytest.mark.asyncio
async def test_dspy_rm(mock_packet):
    with patch("sievrant.integrations.dspy.SievrantClient", autospec=True) as MockClient:
        instance = MockClient.return_value
        instance.__aenter__.return_value = instance
        instance.query = AsyncMock(return_value=mock_packet)
        rm = SievrantRM(base_url="http://mock", api_key="mock", corpus="test_corpus")
        # Just simple test to ensure it initiates
        assert rm is not None
