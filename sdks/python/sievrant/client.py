"""
SievrantClient: async HTTP client for the Sievrant server.

Uses httpx for async HTTP transport with automatic retries,
timeout handling, and clean error mapping to SDK exceptions.
"""

from __future__ import annotations

import os
from typing import Optional, Any

import httpx

from sievrant.models import (
    EvidencePacket,
    HealthResponse,
    IngestResponse,
    CorpusMetadata,
    Decision,
    Route,
)
from sievrant.exceptions import (
    AuthError,
    CorpusNotFoundError,
    SievrantServerError,
    SievrantTimeoutError,
    SievrantAbstainedError,
    ValidationError,
    RateLimitError,
    SievrantError,
)

_DEFAULT_TIMEOUT = 30.0
_DEFAULT_MAX_RETRIES = 3


class SievrantClient:
    """Async HTTP client for the Sievrant retrieval server.

    Usage (async context manager, recommended)::

        async with SievrantClient(base_url="http://localhost:8080", api_key="sk-...") as client:
            result = await client.query(corpus="my-corpus", query="What is the rent?")

    Usage (manual lifecycle)::

        client = SievrantClient(base_url="...", api_key="...")
        result = await client.query(...)
        await client.close()
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: float = 1.5,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        resolved_url = base_url or os.environ.get("SIEVRANT_BASE_URL")
        resolved_key = api_key or os.environ.get("SIEVRANT_API_KEY")

        if not resolved_url:
            raise ValueError(
                "base_url is required. Pass it explicitly or set SIEVRANT_BASE_URL env var."
            )
        if not resolved_key:
            raise ValueError(
                "api_key is required. Pass it explicitly or set SIEVRANT_API_KEY env var."
            )

        self._base_url = resolved_url.rstrip("/")
        self._api_key = resolved_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

        default_headers = {
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
            "User-Agent": "sievrant-python/0.1.0",
        }
        if headers:
            default_headers.update(headers)

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=default_headers,
            timeout=httpx.Timeout(timeout),
        )

    async def __aenter__(self) -> "SievrantClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def health(self) -> HealthResponse:
        """Check server health. No authentication required.

        Returns:
            HealthResponse with status and version fields.
        """
        resp = await self._get("/v1/health", auth=False)
        return HealthResponse.model_validate(resp)

    async def query(
        self,
        *,
        corpus: str,
        query: str,
        coverage: float = 0.95,
        token_budget: int = 4000,
        acl_groups: Optional[list[str]] = None,
        skip_cache: bool = False,
        route: str = "AUTO",
        full: bool = False,
        raise_on_abstain: bool = False,
    ) -> EvidencePacket:
        """Execute a retrieval query against a corpus.

        Args:
            corpus: The corpus to search.
            query: The natural language query.
            coverage: Conformal coverage target in (0, 1). Default 0.95.
            token_budget: Maximum total tokens across evidence chunks. Default 4000.
            acl_groups: ACL groups to restrict results to. Defaults to caller's groups.
            skip_cache: Skip the semantic cache for this call. Default False.
            route: Pipeline route. One of AUTO, FAST, STANDARD, SLOW. Default AUTO.
            full: If True, use the /v1/query/full endpoint to get the complete EvidencePacket
                  with all evidence chunks and audit block. Default False.
            raise_on_abstain: If True, raise SievrantAbstainedError when decision is ABSTAINED.

        Returns:
            EvidencePacket with decision, answer, evidence, and audit block.

        Raises:
            AuthError: Invalid or missing API key.
            CorpusNotFoundError: Corpus does not exist.
            SievrantAbstainedError: Decision was ABSTAINED and raise_on_abstain=True.
            SievrantServerError: Server-side error.
            SievrantTimeoutError: Request timed out.
        """
        endpoint = "/v1/query/full" if full else "/v1/query"
        payload: dict[str, Any] = {
            "query": query,
            "corpus": corpus,
            "coverage": coverage,
            "token_budget": token_budget,
            "skip_cache": skip_cache,
            "route": route,
        }
        if acl_groups is not None:
            payload["acl_groups"] = acl_groups

        data = await self._post(endpoint, json=payload)
        packet = EvidencePacket.model_validate(data)

        if raise_on_abstain and packet.decision == Decision.ABSTAINED:
            raise SievrantAbstainedError(
                query_id=packet.query_id,
                audit=packet.audit,
            )

        return packet

    async def ingest(
        self,
        *,
        corpus: str,
        documents: list[dict[str, Any]],
    ) -> IngestResponse:
        """Ingest documents into a corpus.

        The corpus is created automatically if it does not exist.

        Args:
            corpus: Target corpus name.
            documents: List of document dicts. Each must have 'id' and 'content'.
                       Optional fields: 'metadata' (dict), 'acl_groups' (list[str]).

        Returns:
            IngestResponse with ingested and failed counts.

        Raises:
            AuthError: Invalid or missing API key.
            SievrantServerError: Server-side error.
        """
        payload = {"corpus": corpus, "documents": documents}
        data = await self._post("/v1/ingest", json=payload)
        return IngestResponse.model_validate(data)

    async def list_corpora(self) -> list[CorpusMetadata]:
        """List all corpora accessible to this API key.

        Returns:
            List of CorpusMetadata objects.
        """
        data = await self._get("/v1/corpora")
        return [CorpusMetadata.model_validate(c) for c in data.get("corpora", [])]

    async def get_corpus(self, corpus: str) -> CorpusMetadata:
        """Get metadata for a specific corpus.

        Args:
            corpus: Corpus name.

        Returns:
            CorpusMetadata for the named corpus.

        Raises:
            CorpusNotFoundError: Corpus does not exist.
        """
        data = await self._get(f"/v1/corpora/{corpus}")
        return CorpusMetadata.model_validate(data)

    async def delete_document(self, corpus: str, doc_id: str) -> bool:
        """Delete a specific document from a corpus.

        Args:
            corpus: Corpus name.
            doc_id: Document ID to delete.

        Returns:
            True if deleted successfully.
        """
        data = await self._delete(f"/v1/corpora/{corpus}/documents/{doc_id}")
        return bool(data.get("deleted", False))

    async def _get(self, path: str, *, auth: bool = True) -> dict[str, Any]:
        headers = {} if auth else {"Authorization": ""}
        try:
            resp = await self._client.get(path, headers=headers if not auth else None)
        except httpx.TimeoutException:
            raise SievrantTimeoutError(self._timeout)
        return self._handle_response(resp)

    async def _post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._client.post(path, json=json)
        except httpx.TimeoutException:
            raise SievrantTimeoutError(self._timeout)
        return self._handle_response(resp)

    async def _delete(self, path: str) -> dict[str, Any]:
        try:
            resp = await self._client.delete(path)
        except httpx.TimeoutException:
            raise SievrantTimeoutError(self._timeout)
        return self._handle_response(resp)

    def _handle_response(self, resp: httpx.Response) -> dict[str, Any]:
        if resp.is_success:
            return resp.json()

        body: dict[str, Any] = {}
        try:
            body = resp.json()
        except Exception:
            pass

        code = resp.status_code
        message = body.get("message", resp.text)
        error_code = body.get("error")

        if code == 401:
            raise AuthError(message)
        elif code == 404:
            # Extract corpus name from the error message if possible
            corpus = body.get("corpus", "unknown")
            raise CorpusNotFoundError(corpus)
        elif code == 422:
            raise ValidationError(message, field=body.get("field"))
        elif code == 429:
            retry_after = resp.headers.get("Retry-After")
            raise RateLimitError(int(retry_after) if retry_after else None)
        elif code >= 500:
            raise SievrantServerError(message, status_code=code, error_code=error_code)
        else:
            raise SievrantError(message, status_code=code)

    @classmethod
    def sync(
        cls,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> "_SyncSievrantClient":
        """Create a synchronous wrapper around the async client.

        For use in Jupyter notebooks, simple scripts, or synchronous frameworks.
        """
        return _SyncSievrantClient(base_url=base_url, api_key=api_key, **kwargs)


class _SyncSievrantClient:
    """Synchronous wrapper. Runs the async client in a dedicated event loop."""

    def __init__(self, **kwargs: Any) -> None:
        import asyncio
        self._async = SievrantClient(**kwargs)
        self._loop = asyncio.new_event_loop()

    def query_sync(self, **kwargs: Any) -> EvidencePacket:
        return self._loop.run_until_complete(self._async.query(**kwargs))

    def ingest_sync(self, **kwargs: Any) -> IngestResponse:
        return self._loop.run_until_complete(self._async.ingest(**kwargs))

    def health_sync(self) -> HealthResponse:
        return self._loop.run_until_complete(self._async.health())

    def close(self) -> None:
        self._loop.run_until_complete(self._async.close())
        self._loop.close()
