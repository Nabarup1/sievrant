"""
LangChain integration for Sievrant.

Provides SievrantRetriever, a drop-in BaseRetriever implementation
that wraps the Sievrant query API and maps EvidenceChunk objects to
LangChain Document objects with full metadata preserved.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from sievrant.client import SievrantClient
from sievrant.models import Decision, EvidenceChunk
from sievrant.exceptions import SievrantAbstainedError

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from pydantic import Field as PydanticField
except ImportError as exc:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install it with: pip install 'sievrant[langchain]'"
    ) from exc


def _chunk_to_document(chunk: EvidenceChunk) -> Document:
    """Convert a Sievrant EvidenceChunk to a LangChain Document."""
    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "source_doc": chunk.source_doc,
            "doc_version_hash": chunk.doc_version_hash,
            "path": chunk.path,
            "page": chunk.page,
            "char_span": list(chunk.char_span) if chunk.char_span else None,
            "retriever": chunk.retriever.value,
            "rrf_score": chunk.rrf_score,
            "rerank_score": chunk.rerank_score,
            "nonconformity_score": chunk.nonconformity_score,
            "submodular_marginal_gain": chunk.submodular_marginal_gain,
            "acl_groups": chunk.acl_groups,
            "freshness_hours": chunk.freshness_hours,
            "token_count": chunk.token_count,
            "ingested_at": chunk.ingested_at,
            "index_generation": chunk.index_generation,
        },
    )


class SievrantRetriever(BaseRetriever):
    """LangChain BaseRetriever that queries the Sievrant server.

    Drop-in replacement for any LangChain retriever. Works with RetrievalQA,
    ConversationalRetrievalChain, LCEL pipelines, and any other LangChain construct
    that accepts a BaseRetriever.

    Example::

        retriever = SievrantRetriever(
            base_url="http://localhost:8080",
            api_key=os.environ["SIEVRANT_API_KEY"],
            corpus="legal-contracts",
            coverage=0.95,
        )
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    """

    base_url: str
    """URL of the running Sievrant server."""

    api_key: str
    """Bearer token for authentication."""

    corpus: str
    """The corpus to retrieve from."""

    coverage: float = PydanticField(default=0.95, ge=0.0, le=1.0)
    """Conformal coverage target. Default 0.95 (95% guarantee)."""

    token_budget: int = PydanticField(default=4000, gt=0)
    """Maximum total tokens across all returned evidence. Default 4000."""

    k: Optional[int] = PydanticField(default=None)
    """Maximum number of documents to return. None = all that pass the conformal gate."""

    acl_groups: list[str] = PydanticField(default_factory=list)
    """ACL groups to restrict results to."""

    skip_cache: bool = False
    """Skip the semantic cache for this retrieval."""

    route: str = "AUTO"
    """Pipeline route: AUTO, FAST, STANDARD, or SLOW."""

    raise_on_abstain: bool = False
    """If True, raise SievrantAbstainedError when decision is ABSTAINED."""

    timeout: float = 30.0
    """Request timeout in seconds."""

    class Config:
        arbitrary_types_allowed = True

    def _get_client(self) -> SievrantClient:
        return SievrantClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._aget_relevant_documents(query, run_manager=run_manager)  # type: ignore[arg-type]
            )
        finally:
            loop.close()

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        async with self._get_client() as client:
            packet = await client.query(
                corpus=self.corpus,
                query=query,
                coverage=self.coverage,
                token_budget=self.token_budget,
                acl_groups=self.acl_groups or None,
                skip_cache=self.skip_cache,
                route=self.route,
                full=True,
                raise_on_abstain=self.raise_on_abstain,
            )

        chunks = packet.evidence
        if self.k is not None:
            chunks = chunks[: self.k]

        docs = [_chunk_to_document(c) for c in chunks]

        # Attach top-level EvidencePacket metadata to each document
        for doc in docs:
            doc.metadata["sievrant_decision"] = packet.decision.value
            doc.metadata["sievrant_query_id"] = packet.query_id
            doc.metadata["sievrant_coverage_guarantee"] = packet.coverage_guarantee
            doc.metadata["sievrant_route"] = packet.route.value
            doc.metadata["sievrant_cache_hit"] = packet.cache_hit
            doc.metadata["sievrant_session_hash"] = packet.audit.session_hash

        return docs
