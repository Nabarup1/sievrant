import aiohttp

try:
    import litellm

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


class TreeSummarizer:
    """
    Implements RAPTOR-style tree summarization for Sievrant.
    Recursively summarizes chunks to generate high-level semantic nodes.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8080",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key

        if not HAS_LITELLM:
            print(
                "Warning: litellm not installed. "
                "TreeSummarizer requires litellm. Run `pip install litellm`."
            )

    async def summarize_nodes(self, corpus: str, node_texts: dict[str, list[str]], auth_token: str):
        """
        Generates summaries for a set of nodes based on their children's texts,
        and pushes them to the Sievrant backend.

        Args:
            corpus: The target corpus.
            node_texts: Dict mapping node_id -> List of texts (from children).
            auth_token: Bearer token for API access.
        """
        if not HAS_LITELLM:
            return

        requests = []
        for node_id, texts in node_texts.items():
            summary = await self._generate_summary(texts)
            if summary:
                requests.append({"node_id": node_id, "summary": summary})

        if requests:
            await self._push_summaries(corpus, requests, auth_token)

    async def _generate_summary(self, texts: list[str]) -> str | None:
        litellm.suppress_debug_info = True

        combined_text = "\n\n---\n\n".join(texts)
        # Prevent context window overflow
        if len(combined_text) > 12000:
            combined_text = combined_text[:12000] + "... [truncated]"

        prompt = (
            "You are a hierarchical summarization agent. Read the following detailed text chunks "
            "and produce a single, cohesive, high-level summary that captures the core meaning. "
            "The summary should be dense with information but easily understandable.\n\n"
            f"TEXTS:\n{combined_text}"
        )

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key

        try:
            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Failed to generate summary: {e}")
            return None

    async def _push_summaries(self, corpus: str, requests: list[dict[str, str]], auth_token: str):
        payload = {"corpus": corpus, "requests": requests}

        headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json"}

        url = f"{self.api_base}/v1/tree/summary"

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(url, json=payload, headers=headers) as response,
            ):
                if response.status == 202:
                    print(f"Successfully pushed {len(requests)} tree summaries.")
                else:
                    text = await response.text()
                    print(f"Failed to push summaries. Status: {response.status}. Body: {text}")
        except Exception as e:
            print(f"Error connecting to Sievrant server: {e}")
