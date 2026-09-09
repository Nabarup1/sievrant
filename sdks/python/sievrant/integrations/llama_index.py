"""
LlamaIndex integration for Sievrant.

Provides SievrantQueryEngine and SievrantRetriever for LlamaIndex,
implementing BaseQueryEngine and BaseRetriever respectively.
"""

from __future__ import annotations

from typing import Any, Optional, List

from sievrant.client import SievrantClient
from sievrant.models import EvidenceChunk, Decision

try:
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.base.base_query_engine import BaseQueryEngine
    from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.response.schema import Response
except ImportError as exc:
    raise ImportError(
        "LlamaIndex integration requires llama-index-core. "
        "Install it with: pip install 'sievrant[llama-index]'"
    ) from exc


def _chunk_to_node(chunk: EvidenceChunk) -> NodeWithScore:
    """Convert a Sievrant EvidenceChunk to a LlamaIndex NodeWithScore."""
    node = TextNode(
        text=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "source_doc": chunk.source_doc,
            "doc_version_hash": chunk.doc_version_hash,
            "path": " > ".join(chunk.path),
            "page": chunk.page,
            "retriever": chunk.retriever.value,
            "rrf_score": chunk.rrf_score,
            "rerank_score": chunk.rerank_score,
            "nonconformity_score": chunk.nonconformity_score,
            "submodular_marginal_gain": chunk.submodular_marginal_gain,
            "acl_groups": chunk.acl_groups,
            "freshness_hours": chunk.freshness_hours,
            "token_count": chunk.token_count,
        },
    )
    return NodeWithScore(node=node, score=chunk.rerank_score or chunk.rrf_score)


class SievrantRetriever(BaseRetriever):
    """LlamaIndex BaseRetriever that queries the Sievrant server.

    Returns NodeWithScore objects compatible with all LlamaIndex pipelines.

    Example::

        retriever = SievrantRetriever(
            base_url="http://localhost:8080",
            api_key="sk-your-key",
            corpus="legal-contracts",
        )
        nodes = retriever.retrieve("What are the payment terms?")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        corpus: str,
        *,
        coverage: float = 0.95,
        token_budget: int = 4000,
        acl_groups: Optional[list[str]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._corpus = corpus
        self._coverage = coverage
        self._token_budget = token_budget
        self._acl_groups = acl_groups or []
        super().__init__(callback_manager=callback_manager)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._aretrieve(query_bundle))
        finally:
            loop.close()

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        async with SievrantClient(
            base_url=self._base_url, api_key=self._api_key
        ) as client:
            packet = await client.query(
                corpus=self._corpus,
                query=query_bundle.query_str,
                coverage=self._coverage,
                token_budget=self._token_budget,
                acl_groups=self._acl_groups or None,
                full=True,
            )

        return [_chunk_to_node(c) for c in packet.evidence]


class SievrantQueryEngine(BaseQueryEngine):
    """LlamaIndex BaseQueryEngine that uses Sievrant for retrieval.

    Returns a Response with source_nodes and response text, fully compatible
    with all LlamaIndex query engine consumers.

    Example::

        engine = SievrantQueryEngine(
            base_url="http://localhost:8080",
            api_key="sk-your-key",
            corpus="legal-contracts",
        )
        response = engine.query("What is the rent?")
        print(response.response)
        print(response.source_nodes)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        corpus: str,
        *,
        coverage: float = 0.95,
        token_budget: int = 4000,
        acl_groups: Optional[list[str]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._corpus = corpus
        self._coverage = coverage
        self._token_budget = token_budget
        self._acl_groups = acl_groups or []
        super().__init__(callback_manager=callback_manager)

    def _query(self, query_bundle: QueryBundle) -> Response:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._aquery(query_bundle))
        finally:
            loop.close()

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        async with SievrantClient(
            base_url=self._base_url, api_key=self._api_key
        ) as client:
            packet = await client.query(
                corpus=self._corpus,
                query=query_bundle.query_str,
                coverage=self._coverage,
                token_budget=self._token_budget,
                acl_groups=self._acl_groups or None,
                full=True,
            )

        source_nodes = [_chunk_to_node(c) for c in packet.evidence]

        if packet.decision == Decision.ABSTAINED:
            response_text = (
                "I was unable to find sufficient evidence to answer this question "
                "with the required confidence level. Please consult the source documents."
            )
        else:
            response_text = packet.answer or ""

        return Response(
            response=response_text,
            source_nodes=source_nodes,
            metadata={
                "query_id": packet.query_id,
                "decision": packet.decision.value,
                "coverage_guarantee": packet.coverage_guarantee,
                "latency_ms": packet.latency_ms,
                "route": packet.route.value,
                "session_hash": packet.audit.session_hash,
            },
        )
