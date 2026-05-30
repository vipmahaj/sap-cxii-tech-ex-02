"""LlmClient Protocol + OpenAI and Fixture implementations.

Per ai-layer.md §3: the API layer never imports openai directly. It depends
only on the Protocol, which makes tests deterministic and Part 4d's
model-agnostic-prompt-layer argument concrete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from orders.prompts import SYSTEM_PROMPT, retry_user_message


@dataclass
class LlmResult:
    """Structured envelope returned by every LlmClient.generate_sql() call."""

    out_of_scope: bool
    reason: str | None
    sql: str | None
    input_tokens: int
    output_tokens: int


class LlmClient(Protocol):
    """Single method protocol. Both impls below conform."""

    def generate_sql(
        self,
        question: str,
        retry_context: tuple[str, str] | None = None,
    ) -> LlmResult:
        """Generate a SQL query for `question`.

        retry_context is (bad_sql, error_message) when this is a retry call.
        """
        ...


# --------------------------------------------------------------------------
# Production implementation — OpenAI gpt-4o-mini with JSON mode
# --------------------------------------------------------------------------


class OpenAiLlmClient:
    """Calls OpenAI chat completions with response_format=json_object.

    The openai SDK import is deferred to __init__ so the FixtureLlmClient
    path (used by tests) doesn't pull openai in.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 30.0,
    ) -> None:
        from openai import OpenAI  # local import — see class docstring

        self.model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout_seconds)

    def generate_sql(
        self,
        question: str,
        retry_context: tuple[str, str] | None = None,
    ) -> LlmResult:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        if retry_context is not None:
            bad_sql, error = retry_context
            # Reconstruct the prior assistant turn so the model sees its own
            # bad output, then append the corrective user turn.
            messages.append({
                "role": "assistant",
                "content": json.dumps({
                    "out_of_scope": False,
                    "reason": None,
                    "sql": bad_sql,
                }),
            })
            messages.append({
                "role": "user",
                "content": retry_user_message(bad_sql, error),
            })

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
        )

        content = response.choices[0].message.content or "{}"
        try:
            envelope = json.loads(content)
        except json.JSONDecodeError as e:
            # Treat malformed JSON as a SQL-shaped failure. The orchestrator
            # will retry once.
            return LlmResult(
                out_of_scope=False,
                reason=None,
                sql=f"-- MALFORMED LLM RESPONSE: {e}",
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
            )

        return LlmResult(
            out_of_scope=bool(envelope.get("out_of_scope", False)),
            reason=envelope.get("reason"),
            sql=envelope.get("sql"),
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )


# --------------------------------------------------------------------------
# Test implementation — deterministic, no API key required
# --------------------------------------------------------------------------


class FixtureLlmClient:
    """Deterministic LLM client backed by a JSON fixture file.

    Used by the test suite so the 'retry loop fires' and 'retry exhausted'
    scenarios are CI-runnable without an API key.

    Fixture file shape (see tests/fixtures/llm_responses.json):
        {
          "<question>": {
            "first":  { "out_of_scope": false, "sql": "SELECT ..." },
            "retry":  { "out_of_scope": false, "sql": "SELECT ..." }
          }
        }
    """

    def __init__(self, fixture_path: str) -> None:
        path = Path(fixture_path)
        if not path.is_file():
            raise FileNotFoundError(f"LLM fixture not found: {fixture_path}")
        self._data = json.loads(path.read_text())

    def generate_sql(
        self,
        question: str,
        retry_context: tuple[str, str] | None = None,
    ) -> LlmResult:
        entry = self._data.get(question)
        if entry is None:
            raise KeyError(
                f"FixtureLlmClient has no entry for question: {question!r}. "
                f"Add it to tests/fixtures/llm_responses.json."
            )

        key = "retry" if retry_context is not None else "first"
        payload = entry.get(key)
        if payload is None:
            raise KeyError(
                f"FixtureLlmClient: question {question!r} is missing the {key!r} "
                f"entry. retry_context={retry_context!r}"
            )

        return LlmResult(
            out_of_scope=bool(payload.get("out_of_scope", False)),
            reason=payload.get("reason"),
            sql=payload.get("sql"),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
        )


# --------------------------------------------------------------------------
# Factory used by app.py via FastAPI Depends
# --------------------------------------------------------------------------


def build_client(settings) -> LlmClient:
    """Pick the right implementation based on settings.llm_client."""
    if settings.llm_client == "fixture":
        return FixtureLlmClient("tests/fixtures/llm_responses.json")
    if settings.llm_client == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_CLIENT=openai")
        return OpenAiLlmClient(
            api_key=settings.openai_api_key,
            model=settings.llm_model_name,
        )
    raise ValueError(f"Unknown LLM_CLIENT: {settings.llm_client!r}")
