from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from retrieval_utils import (
    QueryRetrievalPlan,
    build_preview,
    compute_semantic_proxy_score,
    normalize_text,
    tokenize,
    top_n_average,
)

logger = logging.getLogger(__name__)


IDENTITY_QUERY_MARKERS = {
    "identity",
    "prefer",
    "prefers",
    "preference",
    "favorite",
    "favourite",
    "belief",
    "beliefs",
    "role",
}

STRICT_ATTRIBUTE_QUERY_MARKERS = {
    "favorite",
    "favourite",
    "prefer",
    "prefers",
    "preferred",
    "color",
    "language",
    "planet",
    "name",
    "backend",
}

NON_DISCRIMINATIVE_QUERY_TOKENS = {
    "favorite",
    "favourite",
    "prefer",
    "prefers",
    "preferred",
    "already",
    "current",
    "view",
    "role",
}

GENERIC_SUPPORT_TOKENS = {
    "mellon",
    "mellon's",
    "remember",
    "memory",
    "internal",
    "about",
    "what",
    "does",
    "have",
    "with",
    "from",
    "main",
    "work",
    "project",
}

MIN_ANCHOR_COVERAGE = 0.3


@dataclass
class MemoryReviewResult:
    review_state: str
    review_reason: str
    memory_support_strength: float
    memory_conflict_detected: bool
    memory_gap_detected: bool
    recalled_vs_inferred: str
    reflection_bank_used: bool
    reflection_ids_used: List[str]
    recommended_strategy: str
    recommended_fallback_reason: Optional[str]
    issue_tags: List[str] = field(default_factory=list)
    reflection_support: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "review_state": self.review_state,
            "review_reason": self.review_reason,
            "memory_support_strength": self.memory_support_strength,
            "memory_conflict_detected": self.memory_conflict_detected,
            "memory_gap_detected": self.memory_gap_detected,
            "recalled_vs_inferred": self.recalled_vs_inferred,
            "reflection_bank_used": self.reflection_bank_used,
            "reflection_ids_used": list(self.reflection_ids_used),
            "recommended_strategy": self.recommended_strategy,
            "recommended_fallback_reason": self.recommended_fallback_reason,
            "issue_tags": list(self.issue_tags),
            "reflection_support": list(self.reflection_support),
        }


class MemoryReviewEngine:
    def __init__(self, memory_tree: Optional[Any] = None):
        self.memory_tree = memory_tree

    def review_memory(
        self,
        *,
        query: str,
        user_id: str,
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
        hits: Sequence[Dict[str, Any]],
        layer_coverage: Sequence[str],
        conflict_detected: bool,
        baseline_decision: Dict[str, Any],
    ) -> MemoryReviewResult:
        scores = [float(hit.get("score", 0.0)) for hit in hits]
        top_score = scores[0] if scores else 0.0
        avg3 = top_n_average(scores, 3)
        avg4 = top_n_average(scores, 4)
        hit_count = len(hits)
        coverage = set(layer_coverage)
        target_layers = set(retrieval_plan.target_layers)
        coverage_ratio = len(coverage & target_layers) / max(1, len(target_layers))
        support_anchor_tokens = self._extract_support_anchor_tokens(query=query, retrieval_plan=retrieval_plan)
        anchor_coverage, matched_anchor_tokens = self._measure_anchor_coverage(
            hits=hits,
            anchor_tokens=support_anchor_tokens,
        )
        phrase_anchor_coverage, matched_phrase_anchors = self._measure_phrase_anchor_coverage(
            hits=hits,
            retrieval_plan=retrieval_plan,
        )
        specific_anchor_tokens = self._extract_specific_anchor_tokens(support_anchor_tokens)
        specific_anchor_coverage, matched_specific_anchor_tokens = self._measure_anchor_coverage(
            hits=hits,
            anchor_tokens=specific_anchor_tokens,
        )

        factual_covered = "factual" in coverage
        profile_covered = "profile" in coverage
        pattern_covered = bool({"partial_pattern", "full_pattern"} & coverage)
        missing_layers = sorted(target_layers - coverage)

        structured_future_support = sum(1 for hit in hits[:4] if self._is_structured_future_hit(hit))
        vague_future_support = (
            input_type == "future_modeling"
            and any((hit.get("source") in {"dream", "simulated_dream", "simulation", "reflection"}) for hit in hits[:3])
            and structured_future_support == 0
        )

        recalled_vs_inferred = self._recalled_vs_inferred(
            top_score=top_score,
            avg3=avg3,
            hit_count=hit_count,
            coverage_ratio=coverage_ratio,
        )
        memory_support_strength = self._support_strength(
            top_score=top_score,
            avg3=avg3,
            avg4=avg4,
            hit_count=hit_count,
            coverage_ratio=coverage_ratio,
            conflict_detected=conflict_detected,
            recalled_vs_inferred=recalled_vs_inferred,
            structured_future_support=structured_future_support,
            anchor_coverage=anchor_coverage,
        )

        identity_query = any(marker in normalize_text(query) for marker in IDENTITY_QUERY_MARKERS)
        strict_attribute_query = any(marker in normalize_text(query) for marker in STRICT_ATTRIBUTE_QUERY_MARKERS)
        gap_detected = bool(missing_layers) or hit_count == 0 or (input_type == "factual" and not factual_covered)
        issue_tags: List[str] = []

        if not hits:
            review_state = "insufficient_memory"
            review_reason = "no_relevant_memory"
            issue_tags.append("missing_memory")
        elif identity_query and not profile_covered and memory_support_strength < 0.72:
            review_state = "insufficient_memory"
            review_reason = "identity_support_gap"
            issue_tags.extend(["identity_support_gap", "missing_profile_layer"])
        elif input_type == "factual" and not factual_covered:
            review_state = "insufficient_memory"
            review_reason = "missing_factual_layer"
            issue_tags.append("missing_factual_layer")
        elif (
            strict_attribute_query
            and specific_anchor_tokens
            and specific_anchor_coverage < 0.5
            and phrase_anchor_coverage < 0.5
        ):
            review_state = "insufficient_memory"
            review_reason = "subject_gap"
            gap_detected = True
            issue_tags.extend(["subject_gap", *specific_anchor_tokens[:3]])
        elif support_anchor_tokens and anchor_coverage < MIN_ANCHOR_COVERAGE:
            review_state = "insufficient_memory"
            review_reason = "subject_gap"
            gap_detected = True
            issue_tags.extend(["subject_gap", *support_anchor_tokens[:3]])
        elif conflict_detected:
            review_state = "conflicting_memory"
            review_reason = "conflicting_traces"
            issue_tags.append("conflicting_traces")
        elif vague_future_support:
            review_state = "reasoning_risk"
            review_reason = "vague_future_traces"
            issue_tags.extend(["vague_future_traces", "weak_future_support"])
        elif recalled_vs_inferred == "mostly_inferred" and top_score < 0.58:
            review_state = "reasoning_risk" if hit_count > 0 else "insufficient_memory"
            review_reason = "mostly_inferred_from_sparse_hits"
            issue_tags.append("weak_support_inference")
        elif gap_detected or baseline_decision.get("strategy") == "internal_memory_plus_llm":
            review_state = "partial_memory"
            if missing_layers:
                review_reason = f"weak_layer_coverage:{','.join(missing_layers)}"
                issue_tags.extend(["weak_layer_coverage", *missing_layers])
            else:
                review_reason = "memory_partial"
                issue_tags.append("memory_partial")
        else:
            review_state = "stable_memory"
            review_reason = "strong_recalled_support"
            issue_tags.append("stable_support")

        if review_state == "insufficient_memory":
            memory_support_strength = min(memory_support_strength, 0.36 if hit_count else 0.18)
        elif review_state == "reasoning_risk":
            memory_support_strength = min(memory_support_strength, 0.48)

        reflection_support = self._lookup_reflection_support(
            query=query,
            user_id=user_id,
            input_type=input_type,
            retrieval_plan=retrieval_plan,
            review_state=review_state,
            review_reason=review_reason,
            issue_tags=issue_tags,
        )
        reflection_ids_used = [item["id"] for item in reflection_support if item.get("id")]
        reflection_bank_used = bool(reflection_ids_used)

        if review_reason == "memory_partial" and any("identity_support_gap" in item.get("issue_tags", []) for item in reflection_support):
            review_state = "insufficient_memory"
            review_reason = "identity_support_gap"
            issue_tags.extend(["identity_support_gap", "missing_profile_layer"])

        recommended_strategy, recommended_fallback_reason = self._plan_response(
            review_state=review_state,
            review_reason=review_reason,
            memory_support_strength=memory_support_strength,
            hit_count=hit_count,
            input_type=input_type,
            retrieval_plan=retrieval_plan,
            baseline_decision=baseline_decision,
        )

        return MemoryReviewResult(
            review_state=review_state,
            review_reason=review_reason,
            memory_support_strength=memory_support_strength,
            memory_conflict_detected=bool(conflict_detected),
            memory_gap_detected=gap_detected,
            recalled_vs_inferred=recalled_vs_inferred,
            reflection_bank_used=reflection_bank_used,
            reflection_ids_used=reflection_ids_used,
            recommended_strategy=recommended_strategy,
            recommended_fallback_reason=recommended_fallback_reason,
            issue_tags=list(dict.fromkeys(issue_tags)),
            reflection_support=reflection_support,
        )

    def store_review_trace(
        self,
        *,
        user_id: str,
        prompt: str,
        memory_bundle: Dict[str, Any],
        review_result: MemoryReviewResult,
        response_payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.memory_tree:
            return None
        if review_result.review_state == "stable_memory" and not review_result.reflection_bank_used:
            return None

        pattern_key = f"{memory_bundle.get('input_type', 'general')}:{review_result.review_state}:{review_result.review_reason}"
        for row in self.memory_tree.get_recent_nodes(user_id=user_id, limit=120):
            node = self._load_node(row)
            if not node:
                continue
            metadata = node.get("metadata") or {}
            if metadata.get("review_trace") and metadata.get("pattern_key") == pattern_key:
                return None

        top_hits = list(memory_bundle.get("hits") or [])[:3]
        top_seed_ids: List[str] = []
        top_summaries: List[str] = []
        for hit in top_hits:
            node = hit.get("node") or {}
            metadata = node.get("metadata") or {}
            if metadata.get("seed_memory_id"):
                top_seed_ids.append(str(metadata["seed_memory_id"]))
            summary = node.get("summary") or hit.get("summary")
            if summary:
                top_summaries.append(str(summary))

        summary = (
            f"Memory review: {review_result.review_state.replace('_', ' ')} "
            f"due to {review_result.review_reason.replace('_', ' ')}."
        )
        trace_text = self._build_review_trace_text(
            prompt=prompt,
            memory_bundle=memory_bundle,
            review_result=review_result,
            response_payload=response_payload,
            top_summaries=top_summaries,
        )
        return self.memory_tree.remember(
            user_id=user_id,
            source_kind="reflection",
            text=trace_text,
            related_input=prompt,
            emotion_tag="metacognitive",
            summary=build_preview(summary, limit=120),
            importance_score=0.78 if review_result.review_state in {"conflicting_memory", "insufficient_memory", "reasoning_risk"} else 0.68,
            emotional_weight=0.24,
            identity_relevance=0.42,
            pillar_memory=False,
            cluster_id=f"reflection:memory_review:{review_result.review_state}",
            metadata={
                "review_trace": True,
                "pattern_key": pattern_key,
                "review_state": review_result.review_state,
                "review_reason": review_result.review_reason,
                "input_type": memory_bundle.get("input_type"),
                "memory_support_strength": review_result.memory_support_strength,
                "recalled_vs_inferred": review_result.recalled_vs_inferred,
                "issue_tags": list(review_result.issue_tags),
                "recommended_strategy": review_result.recommended_strategy,
                "recommended_fallback_reason": review_result.recommended_fallback_reason,
                "reflection_ids_used": list(review_result.reflection_ids_used),
                "response_origin": response_payload.get("response_origin"),
                "llm_called": bool(response_payload.get("llm_called")),
                "top_seed_ids": top_seed_ids,
                "top_summaries": top_summaries,
            },
        )

    def _plan_response(
        self,
        *,
        review_state: str,
        review_reason: str,
        memory_support_strength: float,
        hit_count: int,
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
        baseline_decision: Dict[str, Any],
    ) -> tuple[str, Optional[str]]:
        if review_state == "stable_memory":
            return "internal_memory_only", None
        if review_state == "conflicting_memory":
            if retrieval_plan.comparison_seeking:
                return "internal_memory_plus_llm", "conflicting_memory_compare"
            return "llm_fallback", "conflicting_memory"
        if review_state == "insufficient_memory":
            baseline_reason = baseline_decision.get("fallback_reason")
            return "llm_fallback", baseline_reason or review_reason or "insufficient_internal_memory"
        if review_state == "reasoning_risk":
            if memory_support_strength >= 0.46 and hit_count >= 2:
                return "internal_memory_plus_llm", review_reason
            return "llm_fallback", review_reason
        return "internal_memory_plus_llm", "memory_partial"

    def _support_strength(
        self,
        *,
        top_score: float,
        avg3: float,
        avg4: float,
        hit_count: int,
        coverage_ratio: float,
        conflict_detected: bool,
        recalled_vs_inferred: str,
        structured_future_support: int,
        anchor_coverage: float,
    ) -> float:
        strength = (
            (top_score * 0.48)
            + (avg3 * 0.18)
            + (avg4 * 0.08)
            + (min(hit_count, 4) * 0.05)
            + (coverage_ratio * 0.14)
            + (anchor_coverage * 0.12)
            + (0.04 if structured_future_support else 0.0)
        )
        if recalled_vs_inferred == "mostly_recalled":
            strength += 0.05
        elif recalled_vs_inferred == "mostly_inferred":
            strength -= 0.08
        if anchor_coverage < MIN_ANCHOR_COVERAGE:
            strength -= 0.18
        if conflict_detected:
            strength -= 0.14
        return round(max(0.0, min(strength, 1.0)), 4)

    def _recalled_vs_inferred(
        self,
        *,
        top_score: float,
        avg3: float,
        hit_count: int,
        coverage_ratio: float,
    ) -> str:
        if top_score >= 0.74 and avg3 >= 0.58 and hit_count >= 2 and coverage_ratio >= 0.67:
            return "mostly_recalled"
        if top_score >= 0.52 and hit_count >= 2:
            return "mixed"
        return "mostly_inferred"

    def _lookup_reflection_support(
        self,
        *,
        query: str,
        user_id: str,
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
        review_state: str,
        review_reason: str,
        issue_tags: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if not self.memory_tree or review_state == "stable_memory":
            return []

        query_text = " ".join(
            fragment
            for fragment in [
                query,
                input_type,
                review_state,
                review_reason,
                " ".join(issue_tags),
            ]
            if fragment
        ).strip()

        results: List[Dict[str, Any]] = []
        for recency_rank, row in enumerate(self.memory_tree.get_recent_nodes(user_id=user_id, limit=160)):
            node = self._load_node(row)
            if not node or node.get("source_kind") != "reflection":
                continue
            metadata = node.get("metadata") or {}
            if not metadata.get("review_trace"):
                continue

            candidate_text = " ".join(
                value
                for value in [
                    node.get("summary", ""),
                    node.get("text", ""),
                    str(metadata.get("review_state", "")),
                    str(metadata.get("review_reason", "")),
                    " ".join(str(item) for item in metadata.get("issue_tags") or []),
                    str(metadata.get("recommended_strategy", "")),
                    str(metadata.get("recommended_fallback_reason", "")),
                ]
                if value
            ).strip()
            score = compute_semantic_proxy_score(
                query_text,
                candidate_text,
                summary=node.get("summary", ""),
                source_kind="reflection",
                input_type=input_type,
                query_keywords=retrieval_plan.keywords,
                recency_rank=recency_rank,
            )
            if metadata.get("review_state") == review_state:
                score += 0.08
            if metadata.get("review_reason") == review_reason:
                score += 0.1
            if set(issue_tags) & set(metadata.get("issue_tags") or []):
                score += 0.08
            score = round(min(score, 1.0), 4)
            if score < 0.34:
                continue

            results.append(
                {
                    "id": row.get("id"),
                    "score": score,
                    "summary": node.get("summary") or build_preview(node.get("text", ""), limit=120),
                    "review_state": metadata.get("review_state"),
                    "review_reason": metadata.get("review_reason"),
                    "issue_tags": list(metadata.get("issue_tags") or []),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:2]

    def _build_review_trace_text(
        self,
        *,
        prompt: str,
        memory_bundle: Dict[str, Any],
        review_result: MemoryReviewResult,
        response_payload: Dict[str, Any],
        top_summaries: Sequence[str],
    ) -> str:
        support_phrase = (
            "The strongest supporting memories were "
            + "; ".join(summary.rstrip(".!?") for summary in top_summaries[:2])
            if top_summaries
            else "There were no strong supporting memories."
        )
        fix = self._recommended_fix(review_result)
        return (
            f"Memory review for prompt '{build_preview(prompt, limit=120)}' found {review_result.review_state} "
            f"because {review_result.review_reason}. {support_phrase}. "
            f"The fix that worked was to respond with {response_payload.get('response_origin', review_result.recommended_strategy)} "
            f"and {fix}."
        )

    def _recommended_fix(self, review_result: MemoryReviewResult) -> str:
        if review_result.review_state == "conflicting_memory":
            return "state the conflict explicitly instead of collapsing it into one claim"
        if review_result.review_state == "insufficient_memory":
            return "say the point is not established in internal memory and avoid filling the gap with invented detail"
        if review_result.review_reason == "vague_future_traces":
            return "prefer only structured future traces and mark the answer as partial rather than confident"
        if review_result.review_state == "reasoning_risk":
            return "keep the wording cautious because the answer is more inferred than recalled"
        return "anchor the answer in the strongest surviving memories and mark it as partial"

    def _is_structured_future_hit(self, hit: Dict[str, Any]) -> bool:
        node = hit.get("node") or {}
        metadata = node.get("metadata") or {}
        return bool(
            node.get("source_kind") in {"simulation", "dream", "simulated_dream"}
            and (
                metadata.get("scenario_summary")
                or metadata.get("key_variables")
                or metadata.get("predicted_outcomes")
                )
        )

    def _extract_support_anchor_tokens(
        self,
        *,
        query: str,
        retrieval_plan: QueryRetrievalPlan,
    ) -> List[str]:
        anchors: List[str] = []
        seen: set[str] = set()
        for keyword in list(retrieval_plan.keywords) + tokenize(query):
            normalized = normalize_text(keyword)
            if not normalized or " " in normalized:
                continue
            if normalized in GENERIC_SUPPORT_TOKENS:
                continue
            if len(normalized) < 3:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            anchors.append(normalized)
            if len(anchors) >= 4:
                break
        return anchors

    def _extract_specific_anchor_tokens(self, anchor_tokens: Sequence[str]) -> List[str]:
        return [
            token
            for token in anchor_tokens
            if token not in NON_DISCRIMINATIVE_QUERY_TOKENS and token not in GENERIC_SUPPORT_TOKENS
        ]

    def _measure_anchor_coverage(
        self,
        *,
        hits: Sequence[Dict[str, Any]],
        anchor_tokens: Sequence[str],
    ) -> tuple[float, List[str]]:
        if not anchor_tokens:
            return 1.0, []

        candidate_tokens: set[str] = set()
        candidate_texts: List[str] = []
        for hit in list(hits)[:4]:
            if hit.get("source") == "codex":
                continue
            node = hit.get("node") or {}
            metadata = node.get("metadata") or {}
            fragments = [
                node.get("summary", ""),
                node.get("text", ""),
                " ".join(node.get("keywords") or []),
                " ".join(metadata.get("top_summaries") or []),
            ]
            candidate_text = " ".join(fragment for fragment in fragments if fragment).strip()
            if not candidate_text:
                continue
            candidate_texts.append(normalize_text(candidate_text))
            candidate_tokens.update(tokenize(candidate_text))

        matched: List[str] = []
        for anchor in anchor_tokens:
            if self._anchor_matches(anchor=anchor, candidate_tokens=candidate_tokens, candidate_texts=candidate_texts):
                matched.append(anchor)

        return round(len(matched) / max(1, len(anchor_tokens)), 4), matched

    def _measure_phrase_anchor_coverage(
        self,
        *,
        hits: Sequence[Dict[str, Any]],
        retrieval_plan: QueryRetrievalPlan,
    ) -> tuple[float, List[str]]:
        phrase_anchors = [
            normalize_text(keyword)
            for keyword in retrieval_plan.keywords
            if " " in normalize_text(keyword)
        ]
        if not phrase_anchors:
            return 1.0, []

        candidate_texts: List[str] = []
        for hit in list(hits)[:4]:
            if hit.get("source") == "codex":
                continue
            node = hit.get("node") or {}
            metadata = node.get("metadata") or {}
            fragments = [
                node.get("summary", ""),
                node.get("text", ""),
                " ".join(node.get("keywords") or []),
                " ".join(metadata.get("top_summaries") or []),
            ]
            candidate_text = " ".join(fragment for fragment in fragments if fragment).strip()
            if candidate_text:
                candidate_texts.append(normalize_text(candidate_text))

        matched = [phrase for phrase in phrase_anchors if any(phrase in text for text in candidate_texts)]
        return round(len(matched) / max(1, len(phrase_anchors)), 4), matched

    def _anchor_matches(
        self,
        *,
        anchor: str,
        candidate_tokens: Sequence[str],
        candidate_texts: Sequence[str],
    ) -> bool:
        if anchor in candidate_tokens:
            return True
        anchor_prefix = anchor[:5]
        for token in candidate_tokens:
            if token.startswith(anchor_prefix) or anchor.startswith(token[:5]):
                return True
        return any(anchor in text for text in candidate_texts)

    def _load_node(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw = row.get("memory_node")
        if not raw:
            return None
        if isinstance(raw, dict):
            return dict(raw)
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
