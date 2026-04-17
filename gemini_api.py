from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from runtime_config import get_settings


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeminiCompletion:
    content: str
    model_used: str


def _extract_text(data: Dict[str, Any]) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""

    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    fragments = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if text:
            fragments.append(str(text).strip())
    return " ".join(fragment for fragment in fragments if fragment).strip()


def call_gemini_with_metadata(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: Optional[str] = None,
) -> GeminiCompletion:
    settings = get_settings()
    if not settings.has_gemini:
        raise GeminiUnavailableError("GEMINI_API_KEY is not configured.")

    try:
        import requests
    except Exception as exc:  # pragma: no cover - import guard
        raise GeminiUnavailableError("The 'requests' dependency is not installed.") from exc

    model_name = model or settings.gemini_model
    url = f"{GEMINI_BASE_URL}/{model_name}:generateContent"
    params = {"key": settings.gemini_api_key}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            f"System instruction:\n{system}\n\n"
                            f"User message:\n{prompt}"
                        )
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url,
            params=params,
            json=payload,
            timeout=settings.gemini_timeout_seconds,
        )
    except requests.Timeout as exc:
        raise GeminiUnavailableError(
            f"Gemini request timed out after {settings.gemini_timeout_seconds}s."
        ) from exc
    except requests.RequestException as exc:
        raise GeminiUnavailableError(f"Gemini request failed: {exc}") from exc

    if response.status_code != 200:
        raise GeminiUnavailableError(f"Gemini returned {response.status_code}: {response.text[:240]}")

    try:
        data = response.json()
    except ValueError as exc:
        raise GeminiUnavailableError("Gemini returned invalid JSON.") from exc

    content = _extract_text(data)
    if not content:
        raise GeminiUnavailableError("Gemini returned an empty completion.")

    return GeminiCompletion(content=content, model_used=model_name)


def call_gemini(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: Optional[str] = None,
) -> str:
    completion = call_gemini_with_metadata(prompt=prompt, system=system, model=model)
    return completion.content
