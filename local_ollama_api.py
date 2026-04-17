from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from runtime_config import get_settings


class LocalLLMUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class LocalLLMCompletion:
    content: str
    model_used: str


def call_local_llm_with_metadata(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: Optional[str] = None,
) -> LocalLLMCompletion:
    settings = get_settings()
    if not settings.has_local_llm:
        raise LocalLLMUnavailableError("LOCAL_LLM_ENABLED is false.")

    try:
        import requests
    except Exception as exc:  # pragma: no cover - import guard
        raise LocalLLMUnavailableError("The 'requests' dependency is not installed.") from exc

    model_name = model or settings.local_llm_model
    endpoint = f"{settings.local_llm_base_url.rstrip('/')}/api/chat"
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=settings.local_llm_timeout_seconds,
        )
    except requests.Timeout as exc:
        raise LocalLLMUnavailableError(
            f"Local Ollama request timed out after {settings.local_llm_timeout_seconds}s."
        ) from exc
    except requests.ConnectionError as exc:
        raise LocalLLMUnavailableError(
            "Could not connect to local Ollama. Ensure `ollama serve` is running."
        ) from exc
    except requests.RequestException as exc:
        raise LocalLLMUnavailableError(f"Local Ollama request failed: {exc}") from exc

    if response.status_code != 200:
        raise LocalLLMUnavailableError(f"Local Ollama returned {response.status_code}: {response.text[:240]}")

    try:
        data = response.json()
    except ValueError as exc:
        raise LocalLLMUnavailableError("Local Ollama returned invalid JSON.") from exc

    content = ((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise LocalLLMUnavailableError("Local Ollama returned an empty completion.")

    return LocalLLMCompletion(content=content, model_used=str(data.get("model") or model_name))


def call_local_llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: Optional[str] = None,
) -> str:
    completion = call_local_llm_with_metadata(prompt=prompt, system=system, model=model)
    return completion.content
