"""Small, bounded provider interface for Google Gemini structured JSON calls."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Base class for failures that can be handled without fabricated output."""


class LLMDisabled(LLMError):
    pass


class MissingAPIKey(LLMError):
    pass


class LLMTimeout(LLMError):
    pass


class LLMRateLimited(LLMError):
    pass


class LLMUnavailable(LLMError):
    pass


class LLMRequestFailed(LLMError):
    pass


class MalformedLLMResponse(LLMError):
    pass


@dataclass(frozen=True)
class GeminiSettings:
    enabled: bool
    api_key: str | None
    model: str
    timeout_seconds: int = 20
    max_response_characters: int = 20_000
    max_output_tokens: int = 1_024

    @classmethod
    def from_environment(cls) -> GeminiSettings:
        enabled_value = os.environ.get("LLM_ENABLED", "false").strip().casefold()
        if enabled_value not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("LLM_ENABLED must be true or false")
        enabled = enabled_value in {"true", "1", "yes"}
        api_key = os.environ.get("GEMINI_API_KEY")
        model = os.environ.get("GEMINI_MODEL", "").strip()
        if enabled and (api_key is None or not api_key.strip()):
            raise MissingAPIKey("LLM_ENABLED is true but GEMINI_API_KEY is missing")
        if enabled and not model:
            raise ValueError("LLM_ENABLED is true but GEMINI_MODEL is missing")
        return cls(
            enabled=enabled,
            api_key=api_key if enabled else None,
            model=model,
        )


class JSONProvider(Protocol):
    def generate_json(
        self,
        *,
        system_instruction: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]: ...


class GeminiProvider:
    """Gemini Developer API adapter; makes one request and never retries."""

    def __init__(self, settings: GeminiSettings, client: Any | None = None) -> None:
        if not settings.enabled:
            raise LLMDisabled("Gemini is disabled")
        if not settings.api_key:
            raise MissingAPIKey("GEMINI_API_KEY is required when Gemini is enabled")
        if not settings.model:
            raise ValueError("GEMINI_MODEL is required when Gemini is enabled")
        if settings.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")
        self._settings = settings
        if client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as error:
                raise LLMUnavailable(
                    "Install the google-genai dependency to enable Gemini"
                ) from error
            self._client = genai.Client(
                api_key=settings.api_key,
                http_options=types.HttpOptions(
                    timeout=settings.timeout_seconds * 1_000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        else:
            self._client = client

    def generate_json(
        self,
        *,
        system_instruction: str,
        payload: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        from google.genai import types
        import httpx
        from google.genai.errors import APIError

        try:
            response = self._client.models.generate_content(
                model=self._settings.model,
                contents=json.dumps(payload, ensure_ascii=False),
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0,
                    max_output_tokens=self._settings.max_output_tokens,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                ),
            )
        except httpx.TimeoutException as error:
            raise LLMTimeout("Gemini request timed out") from error
        except httpx.TransportError as error:
            raise LLMUnavailable("Gemini network request failed") from error
        except APIError as error:
            if error.code == 429:
                raise LLMRateLimited("Gemini rate limit reached") from error
            if error.code in {408, 504}:
                raise LLMTimeout("Gemini request timed out") from error
            if error.code >= 500:
                raise LLMUnavailable("Gemini service is temporarily unavailable") from error
            raise LLMRequestFailed(
                f"Gemini rejected the request with status {error.code}"
            ) from error

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise MalformedLLMResponse("Gemini returned no structured text")
        if len(text) > self._settings.max_response_characters:
            raise MalformedLLMResponse("Gemini response exceeded the size limit")
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as error:
            raise MalformedLLMResponse("Gemini response was not valid JSON") from error
        if not isinstance(decoded, dict):
            raise MalformedLLMResponse("Gemini response must be a JSON object")
        return decoded

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()


def provider_from_environment() -> GeminiProvider | None:
    settings = GeminiSettings.from_environment()
    if not settings.enabled:
        return None
    return GeminiProvider(settings)
