import json
from collections.abc import Callable
from typing import Any

try:
    import litellm

    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


class PromptOptimizer:
    """
    A lightweight DSPy-style optimizer for Sievrant extraction prompts.
    It takes an initial prompt, a set of few-shot examples (dataset), and an evaluation metric.
    It iteratively runs the prompt against the dataset, scores it, and if the score is low,
    asks the LLM to rewrite the prompt to fix the failures.
    """

    def __init__(self, model: str = "gpt-4o", api_key: str | None = None):
        self.model = model
        self.api_key = api_key

        if not HAS_LITELLM:
            print(
                "Warning: litellm not installed. "
                "PromptOptimizer requires litellm. Run `pip install litellm`."
            )

    async def optimize(
        self,
        initial_prompt: str,
        dataset: list[dict[str, Any]],
        eval_fn: Callable[[dict[str, Any], dict[str, Any]], float],
        max_iterations: int = 3,
    ) -> str:
        """
        Optimizes the prompt over `max_iterations`.

        Args:
            initial_prompt: The base instructions.
            dataset: List of dicts, e.g., [{"text": "...", "expected": {...}}, ...]
            eval_fn: A function that takes (prediction, expected) and returns a float
                score 0.0 - 1.0.
            max_iterations: Max optimization loops.

        Returns:
            The optimized prompt string.
        """
        if not HAS_LITELLM:
            return initial_prompt

        current_prompt = initial_prompt
        best_prompt = current_prompt
        best_score = -1.0

        for iteration in range(max_iterations):
            print(f"[Optimizer] Iteration {iteration + 1}/{max_iterations}")

            # 1. Evaluate current prompt
            total_score = 0.0
            failures = []

            for example in dataset:
                text = example["text"]
                expected = example["expected"]

                prediction = await self._run_prompt(current_prompt, text)
                score = eval_fn(prediction, expected)
                total_score += score

                if score < 1.0:
                    failures.append(
                        {"text": text, "expected": expected, "actual": prediction, "score": score}
                    )

            avg_score = total_score / max(len(dataset), 1)
            print(f"[Optimizer] Score: {avg_score:.2f}")

            if avg_score > best_score:
                best_score = avg_score
                best_prompt = current_prompt

            if avg_score >= 1.0 or not failures:
                print("[Optimizer] Perfect score achieved!")
                break

            # 2. Rewrite prompt based on failures
            if iteration < max_iterations - 1:
                print("[Optimizer] Rewriting prompt based on failures...")
                current_prompt = await self._rewrite_prompt(current_prompt, failures[:3])

        print(f"[Optimizer] Finished. Best score: {best_score:.2f}")
        return best_prompt

    async def _run_prompt(self, prompt: str, text: str) -> dict[str, Any]:
        """Runs the extraction prompt for evaluation."""
        litellm.suppress_debug_info = True

        full_prompt = f"{prompt}\n\nText: {text}"
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": full_prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key

        try:
            response = await litellm.acompletion(**kwargs)
            content = response.choices[0].message.content

            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]

            return json.loads(content.strip())
        except Exception as e:
            return {"error": str(e)}

    async def _rewrite_prompt(self, current_prompt: str, failures: list[dict[str, Any]]) -> str:
        """Asks the LLM to improve the prompt."""
        litellm.suppress_debug_info = True

        failures_text = json.dumps(failures, indent=2)

        meta_prompt = (
            "You are a Prompt Engineer. "
            "Your task is to improve a data extraction prompt based on its failures.\n"
            f"CURRENT PROMPT:\n{current_prompt}\n\n"
            f"FAILURES (Sample):\n{failures_text}\n\n"
            "Analyze why the current prompt failed on the expected outputs. "
            "Write a NEW, IMPROVED prompt that will prevent these failures. "
            "Return ONLY the new prompt text without markdown wrapping or explanations."
        )

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": meta_prompt}],
            "temperature": 0.2,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key

        try:
            response = await litellm.acompletion(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception:
            # Fallback to current if rewrite fails
            return current_prompt
