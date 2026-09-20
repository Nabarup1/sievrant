import json
from typing import Any

try:
    import litellm

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


class EntityExtractor:
    """
    Extracts entities and relations from text chunks for GraphRAG.
    If litellm is installed and an API key is present for the chosen model,
    it uses an LLM (OpenAI, Anthropic, Gemini, Groq, etc).
    Otherwise, it falls back to a mock extraction suitable for local testing.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key
        # Check if litellm is available and if the user wants to use it
        self.use_llm = HAS_LITELLM

        if not self.use_llm:
            print(
                "Warning: litellm not installed. Using mock extractor for GraphRAG. "
                "Run `pip install litellm` to use actual LLMs."
            )

    async def extract(self, chunk_id: str, text: str) -> dict[str, Any]:
        """
        Extracts entities and relations.
        Returns:
            {"chunk_id": "...", "entities": [...], "relations": [...]}
        """
        if self.use_llm:
            return await self._extract_llm(chunk_id, text)
        else:
            return self._extract_mock(chunk_id, text)

    async def _extract_llm(self, chunk_id: str, text: str) -> dict[str, Any]:
        prompt = (
            "You are a GraphRAG extraction system. "
            "Given the following text, extract the key entities "
            "(people, organizations, concepts, locations) and the relations between them.\n"
            "Output strictly as a JSON object with 'entities' (list of strings) and 'relations' "
            "(list of objects with 'source', 'target', and 'relation' strings).\n\n"
            f"Text: {text}"
        )

        try:
            # Drop litellm default logging to avoid clutter
            litellm.suppress_debug_info = True

            kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key

            response = await litellm.acompletion(**kwargs)

            content = response.choices[0].message.content
            # Handle potential markdown code blocks returned by some models (like Claude)
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]

            data = json.loads(content.strip())

            return {
                "chunk_id": chunk_id,
                "entities": data.get("entities", []),
                "relations": data.get("relations", []),
            }
        except Exception as e:
            print(f"LLM extraction failed for chunk {chunk_id}: {e}")
            return self._extract_mock(chunk_id, text)

    def _extract_mock(self, chunk_id: str, text: str) -> dict[str, Any]:
        """Simple mock extractor that splits capitalized words for testing."""
        words = text.split()
        entities = list(set([w.strip(".,!?\"'") for w in words if w.istitle() and len(w) > 3]))

        relations = []
        if len(entities) >= 2:
            relations.append(
                {"source": entities[0], "target": entities[1], "relation": "co-occurs_with"}
            )

        return {"chunk_id": chunk_id, "entities": entities, "relations": relations}
