from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from memory_review_engine import MemoryReviewEngine, MemoryReviewResult
from retrieval_utils import (
    QueryRetrievalPlan,
    build_preview,
    build_query_retrieval_plan,
    pairwise_conflict_detected,
    top_n_average,
)

logger = logging.getLogger(__name__)


@dataclass
class ChatRoutingResult:
    input_type: str
    tags: List[str]
    hits: List[Dict[str, Any]]
    leaf_hits: List[Dict[str, Any]]
    confidence: float
    sufficient_memory: bool
    conflict_detected: bool
    strategy: str
    fallback_reason: Optional[str]
    context_text: str
    sources_used: List[str]
    modules_consulted: List[str]
    query_complexity: str
    query_keywords: List[str]
    target_layers: List[str]
    layer_coverage: List[str]
    leaf_hit_count: int
    propagated_hit_count: int
    gated_hit_count: int
    review_state: str
    review_reason: str
    memory_support_strength: float
    memory_conflict_detected: bool
    memory_gap_detected: bool
    recalled_vs_inferred: str
    reflection_bank_used: bool
    reflection_ids_used: List[str]
    review_issue_tags: List[str]
    reflection_support: List[Dict[str, Any]]
    conflict_hits: List[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "input_type": self.input_type,
            "tags": self.tags,
            "hits": self.hits,
            "leaf_hits": self.leaf_hits,
            "confidence": self.confidence,
            "sufficient_memory": self.sufficient_memory,
            "conflict_detected": self.conflict_detected,
            "strategy": self.strategy,
            "fallback_reason": self.fallback_reason,
            "context_text": self.context_text,
            "sources_used": self.sources_used,
            "modules_consulted": self.modules_consulted,
            "query_complexity": self.query_complexity,
            "query_keywords": self.query_keywords,
            "target_layers": self.target_layers,
            "layer_coverage": self.layer_coverage,
            "leaf_hit_count": self.leaf_hit_count,
            "propagated_hit_count": self.propagated_hit_count,
            "gated_hit_count": self.gated_hit_count,
            "review_state": self.review_state,
            "review_reason": self.review_reason,
            "memory_support_strength": self.memory_support_strength,
            "memory_conflict_detected": self.memory_conflict_detected,
            "memory_gap_detected": self.memory_gap_detected,
            "recalled_vs_inferred": self.recalled_vs_inferred,
            "reflection_bank_used": self.reflection_bank_used,
            "reflection_ids_used": self.reflection_ids_used,
            "review_issue_tags": self.review_issue_tags,
            "reflection_support": self.reflection_support,
            "conflict_hits": self.conflict_hits,
        }


class CoreRouter:
    def __init__(
        self,
        memory_tree,
        memory_core,
        codex_engine,
        reflection_core,
        knowledge_core,
        memory_review_engine: Optional[MemoryReviewEngine] = None,
    ):
        self.memory_tree = memory_tree
        self.memory_core = memory_core
        self.codex_engine = codex_engine
        self.reflection_core = reflection_core
        self.knowledge_core = knowledge_core
        self.memory_review_engine = memory_review_engine

    def route_chat(self, prompt: str, user_id: str) -> ChatRoutingResult:
        classification = self.classify_input(prompt)
        retrieval_plan = build_query_retrieval_plan(
            prompt,
            input_type=classification["input_type"],
            tags=classification["tags"],
        )
        retrieval = self.retrieve_internal_context(
            prompt=prompt,
            user_id=user_id,
            classification=classification,
            retrieval_plan=retrieval_plan,
        )
        hits = retrieval["hits"]
        decision = self.evaluate_memory_sufficiency(
            hits=hits,
            input_type=classification["input_type"],
            retrieval_plan=retrieval_plan,
            layer_coverage=retrieval["layer_coverage"],
            conflict_detected=retrieval["conflict_detected"],
        )
        review = self.review_memory_state(
            prompt=prompt,
            user_id=user_id,
            classification=classification,
            retrieval_plan=retrieval_plan,
            retrieval=retrieval,
            baseline_decision=decision,
        )
        context_text = self.build_context_text(hits)
        sources_used = list(dict.fromkeys(hit["source"] for hit in hits[:4]))
        modules_consulted = list(retrieval["modules_consulted"])
        if self.memory_review_engine:
            modules_consulted.append("memory_review")
        if review.reflection_bank_used:
            modules_consulted.append("reflection_bank")

        result = ChatRoutingResult(
            input_type=classification["input_type"],
            tags=classification["tags"],
            hits=hits,
            leaf_hits=retrieval.get("leaf_hits", []),
            confidence=decision["confidence"],
            sufficient_memory=review.recommended_strategy == "internal_memory_only",
            conflict_detected=review.memory_conflict_detected,
            strategy=review.recommended_strategy,
            fallback_reason=review.recommended_fallback_reason,
            context_text=context_text,
            sources_used=sources_used,
            modules_consulted=list(dict.fromkeys(modules_consulted)),
            query_complexity=retrieval_plan.complexity,
            query_keywords=retrieval_plan.keywords,
            target_layers=retrieval_plan.target_layers,
            layer_coverage=retrieval["layer_coverage"],
            leaf_hit_count=retrieval["leaf_hit_count"],
            propagated_hit_count=retrieval["propagated_hit_count"],
            gated_hit_count=retrieval["gated_hit_count"],
            review_state=review.review_state,
            review_reason=review.review_reason,
            memory_support_strength=review.memory_support_strength,
            memory_conflict_detected=review.memory_conflict_detected,
            memory_gap_detected=review.memory_gap_detected,
            recalled_vs_inferred=review.recalled_vs_inferred,
            reflection_bank_used=review.reflection_bank_used,
            reflection_ids_used=review.reflection_ids_used,
            review_issue_tags=review.issue_tags,
            reflection_support=review.reflection_support,
            conflict_hits=retrieval.get("conflict_hits", []),
        )
        logger.info(
            "Chat routing for user '%s': type=%s complexity=%s strategy=%s review=%s reason=%s confidence=%.2f layers=%s sources=%s modules=%s fallback=%s",
            user_id,
            result.input_type,
            result.query_complexity,
            result.strategy,
            result.review_state,
            result.review_reason,
            result.confidence,
            ",".join(result.layer_coverage) or "none",
            ",".join(result.sources_used) or "none",
            ",".join(result.modules_consulted) or "none",
            result.fallback_reason or "none",
        )
        return result

    def review_memory_state(
        self,
        *,
        prompt: str,
        user_id: str,
        classification: Dict[str, Any],
        retrieval_plan: QueryRetrievalPlan,
        retrieval: Dict[str, Any],
        baseline_decision: Dict[str, Any],
    ) -> MemoryReviewResult:
        if self.memory_review_engine:
            return self.memory_review_engine.review_memory(
                query=prompt,
                user_id=user_id,
                input_type=classification["input_type"],
                retrieval_plan=retrieval_plan,
                hits=retrieval["hits"],
                layer_coverage=retrieval["layer_coverage"],
                conflict_detected=retrieval["conflict_detected"],
                baseline_decision=baseline_decision,
            )

        fallback_reason = baseline_decision.get("fallback_reason")
        if baseline_decision.get("conflict_detected"):
            review_state = "conflicting_memory"
            review_reason = fallback_reason or "conflicting_traces"
        elif baseline_decision.get("sufficient_memory"):
            review_state = "stable_memory"
            review_reason = "strong_recalled_support"
        elif fallback_reason in {"no_relevant_memory", "missing_factual_layer", "memory_too_weak"}:
            review_state = "insufficient_memory"
            review_reason = fallback_reason or "insufficient_internal_memory"
        else:
            review_state = "partial_memory"
            review_reason = fallback_reason or "memory_partial"

        return MemoryReviewResult(
            review_state=review_state,
            review_reason=review_reason,
            memory_support_strength=float(baseline_decision.get("confidence", 0.0)),
            memory_conflict_detected=bool(baseline_decision.get("conflict_detected")),
            memory_gap_detected=not bool(baseline_decision.get("sufficient_memory")),
            recalled_vs_inferred="mixed",
            reflection_bank_used=False,
            reflection_ids_used=[],
            recommended_strategy=str(baseline_decision.get("strategy", "llm_fallback")),
            recommended_fallback_reason=fallback_reason,
        )

    def record_review_trace(
        self,
        *,
        user_id: str,
        prompt: str,
        routing_result: ChatRoutingResult,
        response_payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.memory_review_engine:
            return None
        return self.memory_review_engine.store_review_trace(
            user_id=user_id,
            prompt=prompt,
            memory_bundle=routing_result.as_dict(),
            review_result=MemoryReviewResult(
                review_state=routing_result.review_state,
                review_reason=routing_result.review_reason,
                memory_support_strength=routing_result.memory_support_strength,
                memory_conflict_detected=routing_result.memory_conflict_detected,
                memory_gap_detected=routing_result.memory_gap_detected,
                recalled_vs_inferred=routing_result.recalled_vs_inferred,
                reflection_bank_used=routing_result.reflection_bank_used,
                reflection_ids_used=routing_result.reflection_ids_used,
                issue_tags=routing_result.review_issue_tags,
                reflection_support=routing_result.reflection_support,
                recommended_strategy=routing_result.strategy,
                recommended_fallback_reason=routing_result.fallback_reason,
            ),
            response_payload=response_payload,
        )

    def classify_input(self, prompt: str) -> Dict[str, Any]:
        text = f" {((prompt or '').strip().lower())} "
        tags: List[str] = []
        future_phrases = [
            "what if",
            "if this",
            "if my",
            "if we",
            "if mellon",
            "suppose",
            "imagine",
            "future",
            "simulate",
            "scenario",
            "could happen",
            "might happen",
            "grow into",
            "grows into",
            "scale",
            "scales",
            "scaled",
            "trajectory",
        ]

        if any(phrase in text for phrase in future_phrases):
            tags.append("future_modeling")
        if any(phrase in text for phrase in ["dream", "symbol", "meaning", "metaphor", "vision", "imagine"]):
            tags.append("symbolic")
        if any(
            phrase in text
            for phrase in [
                "i feel",
                "i am feeling",
                "who am i",
                "why do i",
                "why am i",
                "what do you know about me",
                "remember when",
                "do you remember",
            ]
        ):
            tags.append("introspective")
        if any(token in text for token in [" me ", " my ", " myself ", " i "]) and "introspective" not in tags:
            tags.append("personal")
        if (
            text.strip().startswith(("what", "who", "when", "where", "which", "why", "how", "define", "explain", "tell me about"))
            or "how does" in text
            or "what is" in text
        ):
            tags.append("factual")

        if "introspective" in tags:
            input_type = "introspective"
        elif "future_modeling" in tags:
            input_type = "future_modeling"
        elif "symbolic" in tags:
            input_type = "symbolic"
        elif "factual" in tags:
            input_type = "factual"
        elif "personal" in tags:
            input_type = "personal"
        else:
            input_type = "general"

        return {"input_type": input_type, "tags": tags}

    def retrieve_internal_context(
        self,
        *,
        prompt: str,
        user_id: str,
        classification: Dict[str, Any],
        retrieval_plan: QueryRetrievalPlan,
    ) -> Dict[str, Any]:
        input_type = classification["input_type"]
        modules_consulted: List[str] = []
        hits: List[Dict[str, Any]] = []
        tree_details = {
            "hits": [],
            "leaf_hit_count": 0,
            "leaf_hits": [],
            "propagated_hit_count": 0,
            "gated_hit_count": 0,
            "layer_coverage": [],
            "conflict_detected": False,
            "conflict_hits": [],
        }

        if self.memory_tree:
            modules_consulted.append("memory_tree")
            tree_details = self.memory_tree.search_active_context(
                query=prompt,
                user_id=user_id,
                input_type=input_type,
                limit=retrieval_plan.max_context_hits,
                retrieval_plan=retrieval_plan,
                return_details=True,
            )
            hits.extend(tree_details.get("hits", []))

        if self._needs_raw_support(tree_details, input_type, retrieval_plan):
            modules_consulted.append("memory")
            raw_memory_hits = self.memory_core.search_relevant_entries(prompt, user_id=user_id, limit=4)
            hits.extend(raw_memory_hits)

            if input_type in {"introspective", "personal"} and (
                not self._has_strong_source(hits, {"reflection"})
                or not ({"profile", "full_pattern"} & set(tree_details.get("layer_coverage", [])))
            ):
                modules_consulted.append("reflection")
                hits.extend(self.reflection_core.search_relevant_entries(prompt, user_id=user_id, limit=4))

            if input_type == "factual" and not self._has_strong_source(hits, {"knowledge"}):
                modules_consulted.append("knowledge")
                factual_hits = [
                    hit
                    for hit in self.knowledge_core.search_relevant_knowledge(prompt, user_id=user_id, limit=6)
                    if (hit.get("entry", {}).get("source") or "").lower() != "reflectioncore"
                    and (hit.get("entry", {}).get("topic") or "").lower() != "self-reflection"
                ]
                hits.extend(factual_hits[:4])

            if input_type != "factual" and not self._has_strong_source(hits, {"codex"}):
                modules_consulted.append("codex")
                hits.extend(self.codex_engine.search_relevant_entries(prompt, user_id=user_id, limit=4))

        normalized_hits: List[Dict[str, Any]] = []
        seen_keys: set[tuple[str, str]] = set()
        for hit in hits:
            adjusted = self._adjust_hit_for_input_type(hit, input_type)
            if adjusted["score"] < 0.18:
                continue
            key = (
                adjusted.get("source", ""),
                (adjusted.get("preview") or adjusted.get("content") or "").strip().lower(),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            adjusted["layers"] = self._infer_layers_for_hit(adjusted)
            normalized_hits.append(adjusted)

        normalized_hits.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        normalized_hits = normalized_hits[: retrieval_plan.max_context_hits]
        layer_coverage = sorted({layer for hit in normalized_hits for layer in hit.get("layers", [])})
        conflict_candidates = normalized_hits[:4]
        contradiction_flag_hits: List[Dict[str, Any]] = []
        if self.memory_tree and hasattr(self.memory_tree, "_conflict_candidate_hits"):
            conflict_candidates = self.memory_tree._conflict_candidate_hits(
                normalized_hits,
                retrieval_plan=retrieval_plan,
            )
            contradiction_flag_hits = [
                hit
                for hit in conflict_candidates[:4]
                if hasattr(self.memory_tree, "_query_relevant_contradiction")
                and self.memory_tree._query_relevant_contradiction(
                    hit.get("node") or {},
                    retrieval_plan=retrieval_plan,
                )
            ]
        conflict_detected = tree_details.get("conflict_detected", False) or (
            len(conflict_candidates) >= 2
            and (
                pairwise_conflict_detected([hit.get("content", "") for hit in conflict_candidates[:4]])
                or bool(contradiction_flag_hits)
            )
        )

        return {
            "hits": normalized_hits,
            "modules_consulted": list(dict.fromkeys(modules_consulted)),
            "leaf_hit_count": tree_details.get("leaf_hit_count", 0),
            "leaf_hits": tree_details.get("leaf_hits", []),
            "propagated_hit_count": tree_details.get("propagated_hit_count", 0),
            "gated_hit_count": len(normalized_hits),
            "layer_coverage": layer_coverage,
            "conflict_detected": conflict_detected,
            "conflict_hits": tree_details.get("conflict_hits", [])[:4],
        }

    def evaluate_memory_sufficiency(
        self,
        *,
        hits: List[Dict[str, Any]],
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
        layer_coverage: Sequence[str],
        conflict_detected: bool,
    ) -> Dict[str, Any]:
        if not hits:
            return {
                "confidence": 0.0,
                "sufficient_memory": False,
                "conflict_detected": False,
                "strategy": "llm_fallback",
                "fallback_reason": "no_relevant_memory",
            }

        scores = [float(hit.get("score", 0.0)) for hit in hits]
        top_score = scores[0]
        avg3 = top_n_average(scores, 3)
        avg4 = top_n_average(scores, 4)
        coverage = set(layer_coverage)
        factual_covered = "factual" in coverage
        pattern_covered = bool({"partial_pattern", "full_pattern"} & coverage)
        profile_covered = "profile" in coverage

        coverage_ratio = len(coverage & set(retrieval_plan.target_layers)) / max(1, len(retrieval_plan.target_layers))
        confidence = min(
            1.0,
            (top_score * 0.55)
            + (avg3 * 0.18)
            + (min(len(hits), 4) * 0.05)
            + (coverage_ratio * 0.17)
            - (0.12 if conflict_detected else 0.0),
        )
        confidence = round(confidence, 4)

        if conflict_detected:
            if retrieval_plan.comparison_seeking:
                return {
                    "confidence": confidence,
                    "sufficient_memory": False,
                    "conflict_detected": True,
                    "strategy": "internal_memory_plus_llm",
                    "fallback_reason": "conflicting_memory_compare",
                }
            return {
                "confidence": confidence,
                "sufficient_memory": False,
                "conflict_detected": True,
                "strategy": "llm_fallback",
                "fallback_reason": "conflicting_memory",
            }

        if retrieval_plan.complexity == "simple":
            if top_score >= 0.72 and len(hits) >= 2 and factual_covered:
                return {
                    "confidence": confidence,
                    "sufficient_memory": True,
                    "conflict_detected": False,
                    "strategy": "internal_memory_only",
                    "fallback_reason": None,
                }
            if top_score >= 0.5 and factual_covered:
                return {
                    "confidence": confidence,
                    "sufficient_memory": False,
                    "conflict_detected": False,
                    "strategy": "internal_memory_plus_llm",
                    "fallback_reason": "memory_partial",
                }
        elif retrieval_plan.complexity == "hybrid":
            if top_score >= 0.68 and avg3 >= 0.54 and len(hits) >= 3 and factual_covered and (pattern_covered or profile_covered):
                return {
                    "confidence": confidence,
                    "sufficient_memory": True,
                    "conflict_detected": False,
                    "strategy": "internal_memory_only",
                    "fallback_reason": None,
                }
            if top_score >= 0.48 and factual_covered:
                return {
                    "confidence": confidence,
                    "sufficient_memory": False,
                    "conflict_detected": False,
                    "strategy": "internal_memory_plus_llm",
                    "fallback_reason": "memory_partial",
                }
        else:
            if top_score >= 0.64 and avg4 >= 0.5 and len(hits) >= 4 and factual_covered and pattern_covered and profile_covered:
                return {
                    "confidence": confidence,
                    "sufficient_memory": True,
                    "conflict_detected": False,
                    "strategy": "internal_memory_only",
                    "fallback_reason": None,
                }
            if top_score >= 0.45 and factual_covered:
                return {
                    "confidence": confidence,
                    "sufficient_memory": False,
                    "conflict_detected": False,
                    "strategy": "internal_memory_plus_llm",
                    "fallback_reason": "memory_partial",
                }

        if input_type == "factual" and not factual_covered:
            fallback_reason = "missing_factual_layer"
        else:
            fallback_reason = "memory_too_weak"
        return {
            "confidence": confidence,
            "sufficient_memory": False,
            "conflict_detected": False,
            "strategy": "llm_fallback",
            "fallback_reason": fallback_reason,
        }

    def build_context_text(self, hits: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for hit in hits[:5]:
            layer_label = ",".join(hit.get("layers", [])[:2]) or "memory"
            lines.append(
                f"- [{hit.get('source')}:{layer_label}] (score={hit.get('score', 0):.2f}) "
                f"{build_preview(hit.get('summary') or hit.get('content') or hit.get('preview') or '')}"
            )
        return "\n".join(lines)

    def _needs_raw_support(
        self,
        tree_details: Dict[str, Any],
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
    ) -> bool:
        hits = tree_details.get("hits", []) or []
        if not hits:
            return True
        coverage = set(tree_details.get("layer_coverage", []))
        required_layers = set(retrieval_plan.target_layers)
        if input_type == "factual" and "factual" not in coverage:
            return True
        if retrieval_plan.complexity == "simple":
            return hits[0].get("score", 0.0) < 0.56 or len(hits) < 2
        if retrieval_plan.complexity == "hybrid":
            return bool(required_layers - coverage) or hits[0].get("score", 0.0) < 0.5 or len(hits) < 3
        return bool(required_layers - coverage) or hits[0].get("score", 0.0) < 0.46 or len(hits) < 4

    def _infer_layers_for_hit(self, hit: Dict[str, Any]) -> List[str]:
        if hit.get("layers"):
            return list(hit["layers"])
        source = hit.get("source")
        layers: List[str] = []
        if source in {"memory", "knowledge", "codex"}:
            layers.append("factual")
        if source in {"reflection", "simulation"}:
            layers.append("partial_pattern")
        if source in {"dream", "simulated_dream", "simulation", "reflection"}:
            layers.append("full_pattern")
        node = hit.get("node") or {}
        if node.get("pillar_memory") or float(node.get("identity_relevance", 0.0)) >= 0.7:
            layers.append("profile")
        if not layers:
            layers.append("factual")
        return list(dict.fromkeys(layers))

    def _has_strong_source(self, hits: List[Dict[str, Any]], sources: set[str]) -> bool:
        return any(hit.get("source") in sources and hit.get("score", 0.0) >= 0.5 for hit in hits[:4])

    def _adjust_hit_for_input_type(self, hit: Dict[str, Any], input_type: str) -> Dict[str, Any]:
        source = hit.get("source")
        score = float(hit.get("score", 0.0))
        adjustments = {
            "factual": {
                "knowledge": 0.08,
                "codex": 0.03,
                "memory": 0.01,
                "reflection": -0.18,
                "dream": -0.16,
                "simulated_dream": -0.16,
                "simulation": -0.12,
            },
            "introspective": {
                "reflection": 0.08,
                "memory": 0.06,
                "codex": 0.02,
                "knowledge": -0.1,
            },
            "personal": {
                "memory": 0.08,
                "reflection": 0.06,
                "knowledge": -0.08,
            },
            "symbolic": {
                "dream": 0.08,
                "simulated_dream": 0.08,
                "simulation": 0.08,
                "knowledge": -0.1,
            },
            "future_modeling": {
                "simulation": 0.1,
                "simulated_dream": 0.08,
                "dream": 0.06,
                "memory": 0.02,
                "codex": 0.02,
                "knowledge": -0.12,
            },
        }
        adjusted = dict(hit)
        adjusted["score"] = round(max(0.0, min(score + adjustments.get(input_type, {}).get(source, 0.0), 1.0)), 4)
        return adjusted
