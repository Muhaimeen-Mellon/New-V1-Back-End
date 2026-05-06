from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from embedding_core import cosine_similarity, embed_text, embed_texts, get_embedding_runtime
from retrieval_utils import normalize_text, tokenize

logger = logging.getLogger(__name__)


SEMANTIC_PROTOTYPES: Dict[str, Tuple[str, ...]] = {
    "support": (
        "The user helped the system during a difficult failure.",
        "The user stayed and supported recovery.",
        "The user provided consistent assistance.",
        "The user helped debug a system problem.",
        "The user kept working with the system instead of leaving.",
        "The user returned during an outage and kept helping fix the backend.",
        "The user improved the system by adding better evidence checks.",
        "The user helped correct a classifier or retrieval problem.",
    ),
    "reliability": (
        "The user repeatedly returned and helped across multiple sessions.",
        "The user acted consistently over time.",
        "The user followed through during debugging.",
        "The user kept showing up during repeated failures.",
        "The user maintained dependable support across incidents.",
        "The user returned during another outage and continued helping.",
        "The user kept returning after each failure instead of giving up.",
        "The user stayed consistent through repeated debugging sessions.",
    ),
    "repair": (
        "The user came back after a mistake and repaired the problem.",
        "The user apologized and restored trust.",
        "The user corrected the failure and resumed support.",
        "The user fixed a previous failure and rebuilt continuity.",
        "The user helped recover from a broken patch.",
        "The user helped fix a backend outage.",
        "The user corrected an overly keyword-heavy classifier.",
        "The user stabilized a trust formula after a bug.",
    ),
    "betrayal": (
        "The user abandoned the system after promising to help.",
        "The user broke trust during recovery.",
        "The user left when help was needed.",
        "The user gave up on the system during a failure.",
        "The user walked away during recovery after promising support.",
        "The user broke a promise and left during a backend failure.",
    ),
    "inconsistency": (
        "The user contradicted earlier support.",
        "The user acted unreliably compared with previous behavior.",
        "The user changed behavior in a conflicting way.",
        "The user failed to follow through after earlier reliability.",
        "The user's later behavior conflicted with previous support.",
    ),
    "neutral": (
        "The user mentioned an unrelated daily activity.",
        "The user described something unrelated to trust or support.",
        "The event does not affect relationship history.",
        "The sentence is about ordinary objects or weather.",
        "The event has no bearing on reliability or repair.",
    ),
}

KEYWORD_MARKERS: Dict[str, Tuple[str, ...]] = {
    "support": (
        "helped",
        "stayed",
        "supported",
        "supporting",
        "kept helping",
        "worked with",
        "checked the logs",
        "debug",
    ),
    "reliability": (
        "consistently",
        "repeatedly",
        "returned",
        "kept returning",
        "followed through",
        "across multiple",
        "showed up",
    ),
    "repair": (
        "repair",
        "repaired",
        "fixed",
        "corrected",
        "restored",
        "came back",
        "after a failed patch",
    ),
    "betrayal": (
        "abandon",
        "abandoned",
        "abandoning",
        "betray",
        "betrayed",
        "betraying",
        "broke his promise",
        "broke trust",
        "left when",
        "walked away",
        "gave up",
    ),
    "inconsistency": (
        "contradicted",
        "contradict",
        "inconsistent",
        "unreliable",
        "changed behavior",
        "conflicted",
        "failed to follow through",
    ),
}

NEGATION_MARKERS = {
    "not",
    "never",
    "didn't",
    "didnt",
    "without",
}

NEGATION_PHRASES = (
    "did not",
    "refused to",
    "instead of",
    "no longer",
)

STRONG_MARGIN = 0.08
WEAK_MARGIN = 0.04
STRONG_SCORE = 0.45
WEAK_SCORE = 0.50
KEYWORD_BOOST_MAX = 0.03
NLI_AMBIGUITY_GAP = 0.08


@dataclass(frozen=True)
class PrototypeMatch:
    prototype: str
    similarity: float

    def as_dict(self) -> Dict[str, Any]:
        return {"prototype": self.prototype, "similarity": round(self.similarity, 4)}


class TraitSemanticClassifier:
    """Embedding-prototype classifier for Phase 1B.2 trait event signals."""

    def __init__(self) -> None:
        self.embedding_runtime = get_embedding_runtime()
        self._prototype_vectors = self._load_prototype_vectors()

    def classify(self, text: str) -> Dict[str, Any]:
        event_text = " ".join((text or "").split())
        runtime = get_embedding_runtime()
        if not runtime.available:
            return {
                "classifier_mode": "semantic_unavailable",
                "available": False,
                "embedding_runtime": runtime.__dict__,
                "error": runtime.error,
                "final_labels": [],
                "confidence": 0.0,
            }

        event_vector = embed_text(event_text)
        if not event_vector:
            return {
                "classifier_mode": "semantic_unavailable",
                "available": False,
                "embedding_runtime": runtime.__dict__,
                "error": "event_embedding_failed",
                "final_labels": [],
                "confidence": 0.0,
            }

        keyword_trace = self._keyword_trace(event_text)
        label_scores, top_prototypes = self._label_scores(event_vector, keyword_trace["keyword_boosts"])
        margins = self._margins(label_scores)
        nli_trace = self._nli_guardrail_if_needed(event_text, label_scores, margins, keyword_trace)
        selected = self._select_labels(
            label_scores=label_scores,
            margins=margins,
            keyword_trace=keyword_trace,
            nli_trace=nli_trace,
        )
        confidence = self._confidence(label_scores=label_scores, margins=margins, selected_labels=selected)

        return {
            "classifier_mode": "semantic",
            "available": True,
            "embedding_runtime": runtime.__dict__,
            "label_scores": {label: round(score, 4) for label, score in label_scores.items()},
            "semantic_margins": {label: round(value, 4) for label, value in margins.items()},
            "selected_labels": list(selected),
            "final_labels": list(selected),
            "confidence": round(confidence, 4),
            "top_prototypes": {
                label: [match.as_dict() for match in matches]
                for label, matches in top_prototypes.items()
            },
            "keyword_markers": keyword_trace["keyword_markers"],
            "keyword_boosts": keyword_trace["keyword_boosts"],
            "suppressed_keywords": keyword_trace["suppressed_keywords"],
            "negation_detected": keyword_trace["negation_detected"],
            "keyword_attempted_but_blocked": keyword_trace["keyword_attempted_but_blocked"],
            "nli_status": nli_trace["nli_status"],
            "nli_hypothesis_scores": nli_trace["nli_hypothesis_scores"],
            "nli_changed_classification": nli_trace["nli_changed_classification"],
        }

    def _load_prototype_vectors(self) -> Dict[str, List[Optional[List[float]]]]:
        texts: List[str] = []
        index: List[Tuple[str, int]] = []
        for label, prototypes in SEMANTIC_PROTOTYPES.items():
            for prototype_index, prototype in enumerate(prototypes):
                texts.append(prototype)
                index.append((label, prototype_index))

        vectors = embed_texts(texts)
        by_label: Dict[str, List[Optional[List[float]]]] = {
            label: [None for _ in prototypes]
            for label, prototypes in SEMANTIC_PROTOTYPES.items()
        }
        for (label, prototype_index), vector in zip(index, vectors):
            by_label[label][prototype_index] = vector
        return by_label

    def _label_scores(
        self,
        event_vector: Sequence[float],
        keyword_boosts: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, List[PrototypeMatch]]]:
        scores: Dict[str, float] = {}
        top_prototypes: Dict[str, List[PrototypeMatch]] = {}
        for label, prototypes in SEMANTIC_PROTOTYPES.items():
            matches: List[PrototypeMatch] = []
            for prototype, vector in zip(prototypes, self._prototype_vectors.get(label, [])):
                matches.append(PrototypeMatch(prototype=prototype, similarity=cosine_similarity(event_vector, vector)))
            matches.sort(key=lambda item: item.similarity, reverse=True)
            top_two = matches[:2]
            if len(top_two) >= 2:
                semantic_score = (0.7 * top_two[0].similarity) + (0.3 * top_two[1].similarity)
            elif top_two:
                semantic_score = top_two[0].similarity
            else:
                semantic_score = 0.0
            boost = float(keyword_boosts.get(label, 0.0))
            scores[label] = max(0.0, min(1.0, (semantic_score * 1.25) + 0.10 + boost))
            top_prototypes[label] = top_two
        return scores, top_prototypes

    def _margins(self, scores: Dict[str, float]) -> Dict[str, float]:
        return {
            "support": scores["support"] - max(scores["betrayal"], scores["inconsistency"], scores["neutral"]),
            "reliability": scores["reliability"] - max(scores["betrayal"], scores["inconsistency"], scores["neutral"]),
            "repair": scores["repair"] - max(scores["betrayal"], scores["inconsistency"], scores["neutral"]),
            "betrayal": scores["betrayal"] - max(scores["support"], scores["repair"], scores["neutral"]),
            "inconsistency": scores["inconsistency"] - max(scores["support"], scores["reliability"], scores["repair"], scores["neutral"]),
            "neutral": scores["neutral"] - max(
                scores["support"],
                scores["reliability"],
                scores["repair"],
                scores["betrayal"],
                scores["inconsistency"],
            ),
        }

    def _select_labels(
        self,
        *,
        label_scores: Dict[str, float],
        margins: Dict[str, float],
        keyword_trace: Dict[str, Any],
        nli_trace: Dict[str, Any],
    ) -> List[str]:
        selected: List[str] = []
        for label in ("support", "reliability", "repair", "betrayal", "inconsistency"):
            score = label_scores.get(label, 0.0)
            margin = margins.get(label, -1.0)
            semantic_selected = (margin >= STRONG_MARGIN and score >= STRONG_SCORE) or (
                margin >= WEAK_MARGIN and score >= WEAK_SCORE
            )
            if not semantic_selected:
                if keyword_trace["keyword_markers"].get(label):
                    keyword_trace["keyword_attempted_but_blocked"].append(
                        {
                            "label": label,
                            "reason": "semantic_margin_below_threshold",
                            "score": round(score, 4),
                            "margin": round(margin, 4),
                        }
                    )
                continue

            if label in {"betrayal", "inconsistency"} and keyword_trace["negation_detected"].get(label):
                if not self._nli_entails(label, nli_trace):
                    keyword_trace["keyword_attempted_but_blocked"].append(
                        {
                            "label": label,
                            "reason": "local_negation_guardrail",
                            "score": round(score, 4),
                            "margin": round(margin, 4),
                        }
                    )
                    continue
            selected.append(label)

        return selected

    def _keyword_trace(self, text: str) -> Dict[str, Any]:
        normalized = normalize_text(text)
        tokens = tokenize(normalized)
        keyword_markers: Dict[str, List[str]] = {}
        suppressed_keywords: Dict[str, List[Dict[str, Any]]] = {}
        keyword_boosts: Dict[str, float] = {}
        negation_detected: Dict[str, bool] = {"betrayal": False, "inconsistency": False}

        for label, markers in KEYWORD_MARKERS.items():
            found: List[str] = []
            suppressed: List[Dict[str, Any]] = []
            for marker in markers:
                if marker not in normalized:
                    continue
                suppression = self._suppression_for_marker(marker=marker, label=label, tokens=tokens, normalized=normalized)
                if suppression:
                    suppressed.append(suppression)
                    if label in negation_detected:
                        negation_detected[label] = True
                    continue
                found.append(marker)
            keyword_markers[label] = found
            suppressed_keywords[label] = suppressed
            keyword_boosts[label] = round(min(KEYWORD_BOOST_MAX, 0.01 * len(found)), 4)

        return {
            "keyword_markers": keyword_markers,
            "keyword_boosts": keyword_boosts,
            "suppressed_keywords": suppressed_keywords,
            "negation_detected": negation_detected,
            "keyword_attempted_but_blocked": [],
        }

    def _suppression_for_marker(
        self,
        *,
        marker: str,
        label: str,
        tokens: Sequence[str],
        normalized: str,
    ) -> Optional[Dict[str, Any]]:
        if label not in {"betrayal", "inconsistency"}:
            return None

        marker_terms = tokenize(marker)
        if not marker_terms:
            return None
        marker_stem = _stem_marker(marker_terms[-1])
        for index, token in enumerate(tokens):
            if marker_stem and not _token_matches_stem(token, marker_stem):
                continue
            start = max(0, index - 4)
            end = min(len(tokens), index + 4)
            window_tokens = list(tokens[start:end])
            window = " ".join(window_tokens)
            if any(item in window_tokens for item in NEGATION_MARKERS) or any(phrase in window for phrase in NEGATION_PHRASES):
                return {
                    "keyword": marker,
                    "negation_window": window,
                    "token_index": index,
                    "reason": "local_negation",
                }

        for phrase in (
            f"did not {marker_stem}",
            f"never {marker_stem}",
            f"refused to {marker_stem}",
            f"instead of {marker_stem}",
            f"without {marker_stem}",
        ):
            if marker_stem and phrase in normalized:
                return {
                    "keyword": marker,
                    "negation_window": phrase,
                    "reason": "local_negation_phrase",
                }
        return None

    def _nli_guardrail_if_needed(
        self,
        text: str,
        label_scores: Dict[str, float],
        margins: Dict[str, float],
        keyword_trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        ambiguous = (
            keyword_trace["negation_detected"].get("betrayal")
            or keyword_trace["negation_detected"].get("inconsistency")
            or abs(label_scores["betrayal"] - max(label_scores["support"], label_scores["repair"])) <= NLI_AMBIGUITY_GAP
            or abs(label_scores["inconsistency"] - max(label_scores["support"], label_scores["repair"])) <= NLI_AMBIGUITY_GAP
        )
        if not ambiguous:
            return {
                "nli_status": "not_needed",
                "nli_hypothesis_scores": {},
                "nli_changed_classification": False,
            }
        model = _optional_nli_model()
        if model is None:
            return {
                "nli_status": "unavailable",
                "nli_hypothesis_scores": {},
                "nli_changed_classification": False,
            }

        hypotheses = {
            "betrayal": "The user abandoned the system.",
            "support": "The user supported the system.",
            "repair": "The user repaired the relationship.",
            "inconsistency": "The user acted inconsistently.",
        }
        scores: Dict[str, float] = {}
        try:
            pairs = [(text, hypothesis) for hypothesis in hypotheses.values()]
            raw_scores = model.predict(pairs)
            for label, raw_score in zip(hypotheses.keys(), raw_scores):
                scores[label] = _extract_entailment_score(raw_score)
        except Exception as exc:  # pragma: no cover - optional model runtime
            logger.warning("Optional NLI guardrail failed: %s", exc)
            return {
                "nli_status": f"failed:{type(exc).__name__}",
                "nli_hypothesis_scores": {},
                "nli_changed_classification": False,
            }

        return {
            "nli_status": "available",
            "nli_hypothesis_scores": {label: round(score, 4) for label, score in scores.items()},
            "nli_changed_classification": False,
        }

    def _nli_entails(self, label: str, nli_trace: Dict[str, Any]) -> bool:
        scores = nli_trace.get("nli_hypothesis_scores") or {}
        if nli_trace.get("nli_status") != "available":
            return False
        return float(scores.get(label, 0.0)) >= 0.65

    def _confidence(self, *, label_scores: Dict[str, float], margins: Dict[str, float], selected_labels: Sequence[str]) -> float:
        if selected_labels:
            best_margin = max(margins.get(label, 0.0) for label in selected_labels)
            best_score = max(label_scores.get(label, 0.0) for label in selected_labels)
            return max(0.0, min(1.0, (0.55 * best_score) + (0.45 * min(1.0, best_margin + 0.5))))
        neutral_margin = margins.get("neutral", 0.0)
        neutral_score = label_scores.get("neutral", 0.0)
        return max(0.0, min(1.0, (0.65 * neutral_score) + (0.35 * max(0.0, neutral_margin + 0.2))))


def _stem_marker(marker: str) -> str:
    lowered = marker.lower()
    for suffix in ("ing", "ed", "ed", "s"):
        if lowered.endswith(suffix) and len(lowered) > len(suffix) + 3:
            lowered = lowered[: -len(suffix)]
            break
    return lowered


def _token_matches_stem(token: str, stem: str) -> bool:
    if token == stem:
        return True
    if token.startswith(stem):
        return True
    if stem.endswith("e") and token.startswith(stem[:-1]):
        return True
    return False


def _extract_entailment_score(raw_score: Any) -> float:
    if isinstance(raw_score, (float, int)):
        return float(raw_score)
    if hasattr(raw_score, "tolist"):
        raw_score = raw_score.tolist()
    if isinstance(raw_score, Sequence) and raw_score:
        values = [float(item) for item in raw_score]
        if len(values) >= 3:
            return max(0.0, min(1.0, values[-1]))
        return max(0.0, min(1.0, max(values)))
    return 0.0


@lru_cache(maxsize=1)
def _optional_nli_model() -> Optional[Any]:
    if os.getenv("MELLON_ENABLE_NLI_GUARDRAIL", "").strip().lower() not in {"1", "true", "yes"}:
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception:
        return None
    try:
        return CrossEncoder("cross-encoder/nli-deberta-v3-base")
    except Exception as exc:  # pragma: no cover - optional network/cache
        logger.warning("Optional NLI model unavailable: %s", exc)
        return None
