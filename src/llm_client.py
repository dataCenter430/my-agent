"""LLM client with automatic provider selection and retry logic."""

from __future__ import annotations

import os
import json
import time
from typing import Any

import litellm
from litellm import completion

litellm.drop_params = True
litellm.set_verbose = False


def _get_model_and_base() -> tuple[str, str | None]:
    """
    Priority:
    1. LLM_PROXY_URL env var (set by validators during evaluation) → use proxy
    2. OPENROUTER_API_KEY → openrouter/anthropic/claude-sonnet-4-5
    3. ANTHROPIC_API_KEY  → anthropic/claude-sonnet-4-5-20251101
    4. OPENAI_API_KEY     → gpt-4o
    """
    proxy_url = os.environ.get("LLM_PROXY_URL")
    if proxy_url:
        model = os.environ.get("LLM_MODEL", "openrouter/anthropic/claude-sonnet-4-5")
        return model, proxy_url

    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter/anthropic/claude-sonnet-4-5", None

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic/claude-sonnet-4-5-20251101", None

    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o", None

    raise RuntimeError(
        "No LLM API key found. Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY. "
        "During evaluation, LLM_PROXY_URL is provided automatically."
    )


class LLMClient:
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.model, self.api_base = _get_model_and_base()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 8096,
        temperature: float = 0.2,
    ) -> str:
        """Send messages and return assistant reply text, with retry on transient errors."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = completion(**kwargs)
                return response.choices[0].message.content or ""
            except (litellm.RateLimitError, litellm.ServiceUnavailableError) as exc:
                last_exc = exc
                wait = self.retry_delay * (2 ** attempt)
                time.sleep(wait)
            except litellm.APIConnectionError as exc:
                last_exc = exc
                time.sleep(self.retry_delay)
            except Exception as exc:
                raise

        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_exc}")

    def extract_json(self, text: str) -> dict:
        """Extract JSON from LLM response, handling markdown code fences."""
        text = text.strip()

        # Strip markdown code fences
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                break

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object within the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        # Fallback: treat entire text as a thinking response with no command
        return {"thinking": text, "command": "", "done": False}
