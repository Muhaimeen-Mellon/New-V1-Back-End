from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from runtime_config import get_settings

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]


@dataclass(frozen=True)
class EmbeddingRuntime:
    enabled: bool
    available: bool
    model_name: str
    provider: str = "sentence-transformers"
    dimension: int = 0
    error: Optional[str] = None


def _clean_text(text: str) -> str:
    return " ".join((text or "").split())


def _rounded_vector(vector: Sequence[float], precision: int = 6) -> List[float]:
    return [round(float(value), precision) for value in vector]


@lru_cache(maxsize=1)
def _load_model() -> Tuple[Optional[object], Optional[str]]:
    settings = get_settings()
    if not settings.embedding_enabled:
        return None, "embeddings_disabled"
    if SentenceTransformer is None:
        return None, "sentence_transformers_unavailable"

    try:
        model = SentenceTransformer(settings.embedding_model)
        return model, None
    except Exception as exc:  # pragma: no cover - depends on runtime cache/network
        logger.warning("Embedding model load failed for '%s': %s", settings.embedding_model, exc)
        return None, f"{type(exc).__name__}: {exc}"


def get_embedding_runtime() -> EmbeddingRuntime:
    settings = get_settings()
    model, error = _load_model()
    if model is None:
        return EmbeddingRuntime(
            enabled=settings.embedding_enabled,
            available=False,
            model_name=settings.embedding_model,
            error=error,
        )

    try:
        dimension = int(model.get_sentence_embedding_dimension() or 0)
    except Exception:
        dimension = 0
    return EmbeddingRuntime(
        enabled=settings.embedding_enabled,
        available=True,
        model_name=settings.embedding_model,
        dimension=dimension,
    )


def get_embedding_model_name() -> str:
    return get_settings().embedding_model


def embeddings_available() -> bool:
    return get_embedding_runtime().available


def embed_text(text: str) -> Optional[List[float]]:
    vectors = embed_texts([text])
    return vectors[0] if vectors else None


def embed_texts(texts: Sequence[str]) -> List[Optional[List[float]]]:
    cleaned_texts = [_clean_text(text) for text in texts]
    model, error = _load_model()
    if model is None:
        if error and error != "embeddings_disabled":
            logger.debug("Embedding encode skipped: %s", error)
        return [None for _ in cleaned_texts]

    active_indices = [index for index, text in enumerate(cleaned_texts) if text]
    if not active_indices:
        return [None for _ in cleaned_texts]

    active_texts = [cleaned_texts[index] for index in active_indices]
    try:
        vectors = model.encode(active_texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception as exc:  # pragma: no cover - runtime/model specific
        logger.warning("Embedding encode failed: %s", exc)
        return [None for _ in cleaned_texts]

    results: List[Optional[List[float]]] = [None for _ in cleaned_texts]
    for index, vector in zip(active_indices, vectors):
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        results[index] = _rounded_vector(vector)
    return results


def cosine_similarity(left: Optional[Sequence[float]], right: Optional[Sequence[float]]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        return 0.0

    dot_product = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        left_float = float(left_value)
        right_float = float(right_value)
        dot_product += left_float * right_float
        left_norm += left_float * left_float
        right_norm += right_float * right_float
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0

    similarity = dot_product / (math.sqrt(left_norm) * math.sqrt(right_norm))
    return round(max(0.0, min(similarity, 1.0)), 4)
