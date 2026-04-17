from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from runtime_config import get_settings


DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenRouterCompletion:
    content: str
    model_used: str


def _extract_message_text(choice: Dict[str, Any]) -> str:
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        fragments = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_value = block.get("text")
                if text_value:
                    fragments.append(str(text_value).strip())
        return " ".join(fragment for fragment in fragments if fragment).strip()

    return ""


def call_mistral_with_metadata(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: Optional[str] = None,
) -> OpenRouterCompletion:
    settings = get_settings()
    if not settings.has_openrouter:
        raise OpenRouterUnavailableError("OPENROUTER_API_KEY is not configured.")

    try:
        import requests
    except Exception as exc:  # pragma: no cover - import guard
        raise OpenRouterUnavailableError("The 'requests' dependency is not installed.") from exc

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model or settings.openrouter_model or DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=settings.openrouter_timeout_seconds,
        )
    except requests.Timeout as exc:
        raise OpenRouterUnavailableError(
            f"OpenRouter request timed out after {settings.openrouter_timeout_seconds}s."
        ) from exc
    except requests.RequestException as exc:
        raise OpenRouterUnavailableError(f"OpenRouter request failed: {exc}") from exc

    if response.status_code != 200:
        raise OpenRouterUnavailableError(
            f"OpenRouter returned {response.status_code}: {response.text[:200]}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise OpenRouterUnavailableError("OpenRouter returned invalid JSON.") from exc

    content = _extract_message_text((data.get("choices") or [{}])[0])
    if not content:
        raise OpenRouterUnavailableError("OpenRouter returned an empty completion.")

    model_used = str(data.get("model") or payload["model"])
    return OpenRouterCompletion(content=content, model_used=model_used)


def call_mistral(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: Optional[str] = None,
) -> str:
    completion = call_mistral_with_metadata(prompt=prompt, system=system, model=model)
    return completion.content
