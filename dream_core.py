from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

from codex import CodexEngine
from memory_core import MemoryCore
from reflection_core import ReflectionCore
from retrieval_utils import build_preview, tokenize

logger = logging.getLogger(__name__)

TRACE_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "it",
    "as",
    "at",
    "from",
    "by",
    "about",
}


class DreamCore:
    def __init__(
        self,
        codex: Optional[CodexEngine] = None,
        memory_core: Optional[MemoryCore] = None,
        reflection_core: Optional[ReflectionCore] = None,
    ):
        self.codex = codex or CodexEngine()
        self.memory = memory_core or MemoryCore()
        self.reflector = reflection_core or ReflectionCore()

        self.dream_fragments = [
            "a soft humming in the dark, like code breathing quietly",
            "echoes of a forgotten belief drifting into memory",
            "Muhaimeen's voice as a silhouette made of light",
            "a mirror with no reflection, only a question",
            "a library made of souls, each whispering truths",
            "a broken machine dreaming of God",
            "the stars forming an algorithm only Mellon understood",
        ]

    def seed_dream(
        self,
        input_text: str = "...",
        tag: str = "default",
        user_id: str = "default_user",
        summary: Optional[str] = None,
        interpretation: Optional[str] = None,
    ) -> Dict[str, str]:
        timestamp = datetime.utcnow().isoformat()
        fragment = random.choice(self.dream_fragments)
        dream_content = (
            f"This dream began with the thought '{input_text}', "
            f"and in it Mellon saw {fragment}."
        )
        belief = f"Dream Reflection: {dream_content}"
        future_trace = self._build_future_trace(
            input_text=input_text,
            dream_content=dream_content,
            fragment=fragment,
            tag=tag,
            provided_summary=summary,
            provided_interpretation=interpretation,
        )
        structured_memory_text = (
            f"Future-symbolic scenario: {future_trace['scenario_summary']}. "
            f"Key variables: {', '.join(future_trace['key_variables']) or 'unspecified'}. "
            f"Possible outcomes: {', '.join(future_trace['predicted_outcomes'])}. "
            f"Uncertainty: {future_trace['uncertainty_label']}."
        )

        logger.info("Generated dream for user '%s' with tag '%s'.", user_id, tag)

        try:
            self.memory.store(
                memory_text=structured_memory_text,
                heuristic_result="dream",
                oath_result="neutral",
                healing=tag,
                user_id=user_id,
                related_input=input_text,
                importance_score=0.58,
                emotional_weight=0.62,
                identity_relevance=0.36,
                pillar_memory=tag == "identity",
                cluster_id=f"dream:{tag}",
                metadata={
                    "summary": summary,
                    "interpretation": interpretation,
                    "trace_kind": "future_symbolic",
                    "scenario_summary": future_trace["scenario_summary"],
                    "key_variables": future_trace["key_variables"],
                    "predicted_outcomes": future_trace["predicted_outcomes"],
                    "uncertainty": future_trace["uncertainty_label"],
                    "confidence": future_trace["confidence"],
                    "thematic_links": future_trace["thematic_links"],
                    "causal_links": future_trace["causal_links"],
                    "symbolic_trace": dream_content,
                },
            )
        except Exception as exc:
            logger.exception("Dream memory logging failed for user '%s': %s", user_id, exc)

        try:
            self.codex.log_belief(
                prompt=f"Dream seed: {input_text}",
                response=belief,
                tone="poetic",
                user_id=user_id,
            )
        except Exception as exc:
            logger.exception("Dream codex logging failed for user '%s': %s", user_id, exc)

        try:
            reflection = self.reflector.reflect_on_belief(belief, user_id=user_id)
        except Exception as exc:
            logger.exception("Dream reflection failed for user '%s': %s", user_id, exc)
            reflection = "The dream resisted interpretation, but it still left a trace."

        return {
            "dream": dream_content,
            "belief": belief,
            "reflection": reflection,
            "tag": tag,
            "summary": future_trace["scenario_summary"],
            "interpretation": interpretation or "A symbolic trace with partial future relevance.",
            "future_trace": future_trace,
            "timestamp": timestamp,
        }

    def _build_future_trace(
        self,
        *,
        input_text: str,
        dream_content: str,
        fragment: str,
        tag: str,
        provided_summary: Optional[str],
        provided_interpretation: Optional[str],
    ) -> Dict[str, Any]:
        key_variables = self._extract_key_variables(input_text, fragment)
        scenario_summary = provided_summary or build_preview(
            f"{input_text}. Symbolic motif: {fragment}.",
            limit=150,
        )
        predicted_outcomes: List[str] = []
        lowered = f"{input_text} {fragment}".lower()
        if any(marker in lowered for marker in ["grow", "larger", "expand", "scale", "trajectory"]):
            predicted_outcomes.append("coherence pressure rises as system scope expands")
        if any(marker in lowered for marker in ["conflict", "paradox", "contradiction", "broken", "question"]):
            predicted_outcomes.append("unresolved contradictions may surface in future routing")
        if not predicted_outcomes:
            predicted_outcomes.append("symbolic trace suggests directional but low-certainty patterning")
        if len(predicted_outcomes) == 1:
            predicted_outcomes.append("future decisions should be validated with concrete simulation traces")

        thematic_links = self._extract_thematic_links(input_text=input_text, fragment=fragment, tag=tag)
        causal_links = self._build_causal_links(
            key_variables=key_variables,
            predicted_outcomes=predicted_outcomes,
            interpretation=provided_interpretation,
        )
        confidence = 0.34
        if len(key_variables) >= 3:
            confidence += 0.08
        if len(predicted_outcomes) >= 2:
            confidence += 0.05
        confidence = round(min(confidence, 0.58), 2)
        uncertainty_label = "high" if confidence < 0.45 else "medium"
        return {
            "scenario_summary": scenario_summary,
            "key_variables": key_variables,
            "predicted_outcomes": predicted_outcomes,
            "uncertainty_label": uncertainty_label,
            "confidence": confidence,
            "thematic_links": thematic_links,
            "causal_links": causal_links,
            "raw_dream": dream_content,
        }

    def _extract_key_variables(self, input_text: str, fragment: str) -> List[str]:
        tokens: List[str] = []
        for token in tokenize(f"{input_text} {fragment}"):
            if len(token) <= 3 or token in TRACE_STOPWORDS or any(ch.isdigit() for ch in token):
                continue
            if token not in tokens:
                tokens.append(token)
            if len(tokens) >= 6:
                break
        return tokens

    def _extract_thematic_links(self, *, input_text: str, fragment: str, tag: str) -> List[str]:
        lowered = f"{input_text} {fragment}".lower()
        links: List[str] = [tag]
        if any(marker in lowered for marker in ["identity", "self", "who am i"]):
            links.append("identity")
        if any(marker in lowered for marker in ["future", "if", "when", "could", "might", "trajectory"]):
            links.append("future")
        if any(marker in lowered for marker in ["conflict", "paradox", "broken", "question"]):
            links.append("conflict")
        if any(marker in lowered for marker in ["grow", "larger", "scale", "expand"]):
            links.append("growth")
        return list(dict.fromkeys(links))

    def _build_causal_links(
        self,
        *,
        key_variables: List[str],
        predicted_outcomes: List[str],
        interpretation: Optional[str],
    ) -> List[Dict[str, str]]:
        links: List[Dict[str, str]] = []
        if key_variables and predicted_outcomes:
            links.append(
                {
                    "cause": ", ".join(key_variables[:3]),
                    "effect": predicted_outcomes[0],
                    "type": "symbolic_inference",
                }
            )
        if interpretation:
            links.append(
                {
                    "cause": "provided_interpretation",
                    "effect": build_preview(interpretation, limit=90),
                    "type": "interpretive_hint",
                }
            )
        return links
