from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "what",
    "when",
    "where",
    "who",
    "why",
    "you",
    "your",
}

NEGATION_MARKERS = {"no", "not", "never", "cannot", "can't", "wont", "won't", "don't", "false"}
AFFIRMATION_MARKERS = {"yes", "can", "will", "do", "true", "always"}
ATTRIBUTE_CONTRADICTION_MARKERS = {
    "favorite",
    "prefer",
    "prefers",
    "preferred",
    "color",
    "language",
    "name",
    "belief",
    "goal",
    "role",
    "likes",
    "loves",
}
COMPLEXITY_MARKERS = {
    "why",
    "how",
    "compare",
    "difference",
    "conflict",
    "contradict",
    "future",
    "scenario",
    "simulate",
    "what if",
    "suppose",
    "if",
    "plan",
    "trajectory",
}
HYBRID_MARKERS = {
    "and",
    "also",
    "plus",
    "while",
    "alongside",
    "remember",
    "know",
    "because",
}
COMPARISON_MARKERS = {
    "compare",
    "versus",
    "vs",
    "difference",
    "conflict",
    "contradiction",
    "changed my mind",
}


@dataclass(frozen=True)
class QueryRetrievalPlan:
    complexity: str
    keywords: List[str]
    target_layers: List[str]
    leaf_limit: int
    top_leaf_count: int
    max_context_hits: int
    allow_pattern_layers: bool
    allow_dream_simulation: bool
    profile_required: bool
    comparison_seeking: bool


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"\b[a-z0-9']+\b", normalize_text(text))
        if len(token) > 2 and token not in STOPWORDS
    ]


def unique_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def _extract_bigrams(tokens: Sequence[str], limit: int = 4) -> List[str]:
    bigrams: List[str] = []
    for index in range(max(0, len(tokens) - 1)):
        left = tokens[index]
        right = tokens[index + 1]
        if left in STOPWORDS or right in STOPWORDS:
            continue
        bigram = f"{left} {right}"
        if bigram not in bigrams:
            bigrams.append(bigram)
        if len(bigrams) >= limit:
            break
    return bigrams


def extract_lexical_keywords(text: str, limit: int = 6, bigram_limit: int = 4) -> List[str]:
    tokens = tokenize(text)
    if not tokens:
        return []

    token_counts = Counter(tokens)
    ordered_unique = sorted(token_counts, key=lambda token: (-token_counts[token], tokens.index(token)))
    keywords = ordered_unique[:limit]
    return keywords + _extract_bigrams(tokens, limit=bigram_limit)


def build_query_retrieval_plan(
    text: str,
    *,
    input_type: str = "general",
    tags: Optional[Sequence[str]] = None,
) -> QueryRetrievalPlan:
    normalized = normalize_text(text)
    tags_set = {tag for tag in (tags or []) if tag}
    content_tokens = tokenize(normalized)
    comparison_seeking = any(marker in normalized for marker in COMPARISON_MARKERS)
    explicit_complex = any(marker in normalized for marker in COMPLEXITY_MARKERS)
    mild_hybrid = any(marker in normalized for marker in HYBRID_MARKERS)
    clause_count = len(re.findall(r"[,;:?!]", normalized)) + normalized.count(" and ") + normalized.count(" but ")
    mixed_intent = input_type in {"general", "personal"} and any(tag in tags_set for tag in {"factual", "introspective"})

    if (
        len(content_tokens) > 16
        or clause_count >= 3
        or explicit_complex
        or input_type in {"future_modeling", "symbolic"}
    ):
        complexity = "complex"
    elif len(content_tokens) >= 9 or clause_count >= 2 or mild_hybrid or mixed_intent:
        complexity = "hybrid"
    else:
        complexity = "simple"

    layer_map = {
        "simple": ["factual", "profile"],
        "hybrid": ["factual", "partial_pattern", "profile"],
        "complex": ["factual", "full_pattern", "profile"],
    }
    sizing = {
        "simple": {"leaf_limit": 48, "top_leaf_count": 4, "max_context_hits": 6},
        "hybrid": {"leaf_limit": 72, "top_leaf_count": 6, "max_context_hits": 8},
        "complex": {"leaf_limit": 96, "top_leaf_count": 8, "max_context_hits": 10},
    }[complexity]

    return QueryRetrievalPlan(
        complexity=complexity,
        keywords=extract_lexical_keywords(text),
        target_layers=layer_map[complexity],
        leaf_limit=sizing["leaf_limit"],
        top_leaf_count=sizing["top_leaf_count"],
        max_context_hits=sizing["max_context_hits"],
        allow_pattern_layers=complexity != "simple",
        allow_dream_simulation=complexity != "simple" and input_type in {"future_modeling", "symbolic"},
        profile_required=True,
        comparison_seeking=comparison_seeking,
    )


def source_alignment_prior(input_type: str, source_kind: str) -> float:
    lookup = {
        "factual": {
            "self_model": 0.1,
            "architecture": 0.09,
            "constraint": 0.08,
            "knowledge": 0.08,
            "user_model": 0.07,
            "codex": 0.04,
            "memory": 0.02,
            "reflection": -0.04,
            "dream": -0.06,
            "simulation": -0.03,
        },
        "introspective": {
            "self_model": 0.09,
            "user_model": 0.09,
            "reflection": 0.08,
            "memory": 0.07,
            "constraint": 0.04,
            "codex": 0.03,
            "knowledge": -0.03,
        },
        "personal": {
            "user_model": 0.1,
            "self_model": 0.08,
            "memory": 0.08,
            "reflection": 0.05,
            "codex": 0.02,
            "knowledge": -0.04,
        },
        "symbolic": {"dream": 0.08, "simulated_dream": 0.07, "simulation": 0.05, "codex": 0.02, "knowledge": -0.05},
        "future_modeling": {"simulation": 0.08, "simulated_dream": 0.07, "dream": 0.05, "memory": 0.02, "knowledge": -0.05},
        "general": {
            "self_model": 0.09,
            "architecture": 0.08,
            "user_model": 0.07,
            "constraint": 0.07,
            "codex": 0.04,
            "memory": 0.03,
            "knowledge": 0.02,
        },
    }
    return lookup.get(input_type, lookup["general"]).get(source_kind, 0.0)


def compute_semantic_proxy_score(
    query: str,
    candidate_text: str,
    *,
    summary: str = "",
    source_kind: str = "",
    input_type: str = "general",
    query_keywords: Optional[Sequence[str]] = None,
    recency_rank: int = 0,
) -> float:
    query_text = normalize_text(query)
    candidate_text = normalize_text(candidate_text)
    summary_text = normalize_text(summary)
    if not query_text or not candidate_text:
        return 0.0

    query_tokens = tokenize(query_text)
    candidate_tokens = tokenize(candidate_text)
    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap = set(query_tokens) & set(candidate_tokens)
    if not overlap and query_text not in candidate_text:
        return 0.0

    query_bigrams = set(_extract_bigrams(query_tokens, limit=max(1, len(query_tokens) - 1)))
    candidate_bigrams = set(_extract_bigrams(candidate_tokens, limit=max(1, len(candidate_tokens) - 1)))
    bigram_overlap = (len(query_bigrams & candidate_bigrams) / len(query_bigrams)) if query_bigrams else 0.0

    query_coverage = len(overlap) / max(1, len(set(query_tokens)))
    candidate_coverage = len(overlap) / max(1, len(set(candidate_tokens)))

    summary_bonus = 0.0
    if summary_text:
        summary_tokens = set(tokenize(summary_text))
        summary_overlap = len(summary_tokens & set(query_tokens)) / max(1, len(set(query_tokens)))
        summary_bonus = min(0.08, summary_overlap * 0.08)

    exact_bonus = 0.12 if query_text in candidate_text or candidate_text in query_text else 0.0
    keyword_phrases = [keyword for keyword in (query_keywords or []) if " " in keyword]
    if not exact_bonus and any(phrase in candidate_text for phrase in keyword_phrases):
        exact_bonus = 0.06

    recency_bonus = max(0.0, 0.05 - (recency_rank * 0.002))
    score = (
        (query_coverage * 0.46)
        + (candidate_coverage * 0.16)
        + (bigram_overlap * 0.2)
        + summary_bonus
        + exact_bonus
        + max(-0.04, source_alignment_prior(input_type, source_kind))
        + recency_bonus
    )
    return round(max(0.0, min(score, 1.0)), 4)


def compute_bm25_lexical_scores(
    query_terms: Sequence[str],
    documents: Sequence[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> List[float]:
    normalized_documents = [normalize_text(document) for document in documents]
    tokenized_documents = [tokenize(document) for document in normalized_documents]
    if not normalized_documents or not query_terms:
        return [0.0 for _ in documents]

    average_doc_length = sum(len(tokens) for tokens in tokenized_documents) / max(1, len(tokenized_documents))
    query_terms = [normalize_text(term) for term in query_terms if normalize_text(term)]
    if not query_terms:
        return [0.0 for _ in documents]

    doc_frequencies: dict[str, int] = {}
    for term in query_terms:
        if " " in term:
            doc_frequencies[term] = sum(1 for document in normalized_documents if term in document)
        else:
            doc_frequencies[term] = sum(1 for tokens in tokenized_documents if term in tokens)

    scores: List[float] = []
    total_documents = max(1, len(tokenized_documents))
    for tokens, document in zip(tokenized_documents, normalized_documents):
        document_length = max(1, len(tokens))
        term_counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if " " in term:
                term_frequency = document.count(term)
            else:
                term_frequency = term_counts.get(term, 0)
            if term_frequency <= 0:
                continue

            doc_frequency = doc_frequencies.get(term, 0)
            idf = math.log(1 + ((total_documents - doc_frequency + 0.5) / (doc_frequency + 0.5)))
            denominator = term_frequency + k1 * (1 - b + b * (document_length / max(1.0, average_doc_length)))
            score += idf * ((term_frequency * (k1 + 1)) / denominator)
        scores.append(score)

    max_score = max(scores, default=0.0)
    if max_score <= 0.0:
        return [0.0 for _ in scores]
    return [round(max(0.0, min(score / max_score, 1.0)), 4) for score in scores]


def fuse_relevance_scores(semantic_score: float, lexical_score: float, semantic_weight: float = 0.9) -> float:
    return round(
        max(0.0, min((semantic_weight * semantic_score) + ((1.0 - semantic_weight) * lexical_score), 1.0)),
        4,
    )


def compute_relevance_score(query: str, candidate: str, recency_rank: int = 0) -> float:
    return compute_semantic_proxy_score(query, candidate, recency_rank=recency_rank)


def compute_temporal_coherence(
    candidate_created_at: Optional[str],
    anchor_created_ats: Sequence[Optional[str]],
    *,
    pillar_memory: bool = False,
    profile_memory: bool = False,
) -> float:
    if profile_memory or pillar_memory:
        return 0.82

    candidate_dt = _parse_iso(candidate_created_at)
    anchor_dts = [_parse_iso(value) for value in anchor_created_ats if value]
    anchor_dts = [value for value in anchor_dts if value is not None]
    if not candidate_dt or not anchor_dts:
        return 0.58

    average_hours = sum(abs((candidate_dt - anchor).total_seconds()) / 3600.0 for anchor in anchor_dts) / len(anchor_dts)
    if average_hours <= 24:
        return 1.0
    if average_hours <= 72:
        return 0.9
    if average_hours <= 168:
        return 0.8
    if average_hours <= 720:
        return 0.68
    return 0.54


def top_n_average(values: Sequence[float], count: int) -> float:
    if not values:
        return 0.0
    top_values = sorted(values, reverse=True)[:count]
    return round(sum(top_values) / len(top_values), 4)


def build_preview(text: str, limit: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def pairwise_conflict_detected(texts: Sequence[str]) -> bool:
    normalized = [normalize_text(text) for text in texts if normalize_text(text)]
    for index, left in enumerate(normalized):
        left_tokens = unique_tokens(left)
        if not left_tokens:
            continue
        left_negated = bool(left_tokens & NEGATION_MARKERS)
        left_affirmed = bool(left_tokens & AFFIRMATION_MARKERS)
        for right in normalized[index + 1 :]:
            right_tokens = unique_tokens(right)
            if not right_tokens:
                continue
            overlap = left_tokens & right_tokens
            if len(overlap) < 3:
                continue
            right_negated = bool(right_tokens & NEGATION_MARKERS)
            right_affirmed = bool(right_tokens & AFFIRMATION_MARKERS)
            if (left_negated and right_affirmed) or (left_affirmed and right_negated):
                return True
            shared_attribute_markers = overlap & ATTRIBUTE_CONTRADICTION_MARKERS
            left_unique = left_tokens - overlap
            right_unique = right_tokens - overlap
            if (
                shared_attribute_markers
                and 0 < len(left_unique) <= 2
                and 0 < len(right_unique) <= 2
                and left_unique.isdisjoint(right_unique)
            ):
                return True
    return False


def distinct_texts(texts: Iterable[str], limit: int = 3) -> List[str]:
    results: List[str] = []
    seen: set[str] = set()
    for text in texts:
        normalized = normalize_text(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        results.append(text.strip())
        if len(results) >= limit:
            break
    return results


def _parse_iso(raw_value: Optional[str]) -> Optional[datetime]:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None
