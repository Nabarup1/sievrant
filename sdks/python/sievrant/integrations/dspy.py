"""
DSPy integration for Sievrant.

Provides SievrantRM, a dspy.Retrieve-compatible retrieval module
that queries the Sievrant server and returns passages for DSPy programs.
"""

from __future__ import annotations

from sievrant.client import SievrantClient

try:
    import dspy
except ImportError as exc:
    raise ImportError(
        "DSPy integration requires dspy-ai. Install it with: pip install 'sievrant[dspy]'"
    ) from exc


class SievrantRM(dspy.Retrieve):
    """DSPy Retrieve module backed by the Sievrant server.

    Compatible with dspy.settings.configure(rm=...) and usable
    in any DSPy program via dspy.Retrieve(k=...).

    Example::

        import dspy
        from sievrant.integrations.dspy import SievrantRM

        dspy.settings.configure(
            rm=SievrantRM(
                base_url="http://localhost:8080",
                api_key=os.environ["SIEVRANT_API_KEY"],
                corpus="legal-contracts",
            )
        )

        class RAG(dspy.Module):
            def __init__(self):
                self.retrieve = dspy.Retrieve(k=3)
                self.generate = dspy.ChainOfThought("context, question -> answer")

            def forward(self, question):
                context = self.retrieve(question).passages
                return self.generate(context=context, question=question)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        corpus: str,
        *,
        k: int = 3,
        coverage: float = 0.95,
        token_budget: int = 4000,
        acl_groups: list[str] | None = None,
    ) -> None:
        super().__init__(k=k)
        self._base_url = base_url
        self._api_key = api_key
        self._corpus = corpus
        self._coverage = coverage
        self._token_budget = token_budget
        self._acl_groups = acl_groups or []

    def forward(
        self,
        query_or_queries: str | list[str],
        k: int | None = None,
    ) -> dspy.Prediction:
        import asyncio

        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries
        k = k or self.k

        loop = asyncio.new_event_loop()
        try:
            all_passages = loop.run_until_complete(self._retrieve_all(queries, k))
        finally:
            loop.close()

        return dspy.Prediction(passages=all_passages)

    async def _retrieve_all(self, queries: list[str], k: int) -> list[str]:
        passages: list[str] = []
        async with SievrantClient(base_url=self._base_url, api_key=self._api_key) as client:
            for query in queries:
                packet = await client.query(
                    corpus=self._corpus,
                    query=query,
                    coverage=self._coverage,
                    token_budget=self._token_budget,
                    acl_groups=self._acl_groups or None,
                    full=True,
                )
                passages.extend(chunk.text for chunk in packet.evidence[:k])
        return passages
