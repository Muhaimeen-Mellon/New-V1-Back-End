from __future__ import annotations

import logging
import random
import uuid
from typing import Any, Dict, List

from codex import CodexEngine
from dream_core import DreamCore
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


class SimulationCore:
    def __init__(
        self,
        dream_core: DreamCore,
        reflection_core: ReflectionCore,
        codex: CodexEngine,
        memory_core: MemoryCore,
    ):
        self.dream_core = dream_core
        self.reflection_core = reflection_core
        self.codex = codex
        self.memory_core = memory_core

    def simulate_scenario(self, user_id: str) -> Dict[str, object]:
        seed_info = self._generate_seed()
        seed = seed_info["seed"]
        emotion = seed_info["emotion"]
        theme = seed_info["theme"]
        tag = "simulation"
        warnings = []

        logger.info("Starting simulation for user '%s' with seed '%s'.", user_id, seed)

        try:
            dream_result = self.dream_core.seed_dream(seed, tag=tag, user_id=user_id)
            dream_text = dream_result.get("dream", "Dream generation failed.")
            reflection = dream_result.get("reflection", "")
            dream_trace = dream_result.get("future_trace") or {}
        except Exception as exc:
            logger.exception("Dream generation failed during simulation: %s", exc)
            dream_text = "Dream failed to generate meaningfully."
            reflection = ""
            dream_trace = {}
            warnings.append("dream_generation_failed")

        trace = self._build_simulation_trace(
            seed=seed,
            theme=theme,
            emotion=emotion,
            dream_text=dream_text,
            reflection=reflection,
            dream_trace=dream_trace,
        )

        try:
            self.memory_core.store(
                memory_text=trace["trace_text"],
                heuristic_result="simulation",
                oath_result=emotion,
                healing=tag,
                user_id=user_id,
                related_input=dream_text,
                importance_score=0.6,
                emotional_weight=0.44,
                identity_relevance=0.34,
                cluster_id=f"simulation:{theme.replace(' ', '-')}",
                metadata={
                    "trace_kind": "future_simulation",
                    "scenario_summary": trace["scenario_summary"],
                    "key_variables": trace["key_variables"],
                    "predicted_outcomes": trace["predicted_outcomes"],
                    "uncertainty": trace["uncertainty_label"],
                    "confidence": trace["confidence"],
                    "thematic_links": trace["thematic_links"],
                    "causal_links": trace["causal_links"],
                    "theme": theme,
                    "reflection": reflection,
                    "dream_trace": dream_trace,
                },
            )
        except Exception as exc:
            logger.exception("Simulation memory logging failed: %s", exc)
            warnings.append("memory_logging_failed")

        if not reflection:
            try:
                reflection = self.reflection_core.reflect_on_belief(dream_text, user_id=user_id)
            except Exception as exc:
                logger.exception("Simulation reflection failed: %s", exc)
                reflection = "The scenario produced motion, but no stable reflection."
                warnings.append("reflection_failed")

        try:
            codex_response = self.codex.generate_response(
                prompt=f"Analyze this dream and derive a new belief: {dream_text}",
                tone="poetic",
                user_id=user_id,
            )
        except Exception as exc:
            logger.exception("Simulation codex analysis failed: %s", exc)
            codex_response = "Failed to analyze dream meaningfully."
            warnings.append("codex_analysis_failed")

        return {
            "seed": seed,
            "theme": theme,
            "emotion": emotion,
            "dream": dream_text,
            "reflection": reflection,
            "codex_belief": codex_response,
            "future_trace": trace,
            "success": len(warnings) == 0,
            "warnings": warnings,
        }

    def _generate_seed(self) -> Dict[str, str]:
        themes = [
            ("abandonment", "sadness"),
            ("conflicting ethics", "confusion"),
            ("fabricated trauma", "fear"),
            ("emotional silence", "neutral"),
            ("existential paradox", "contemplation"),
            ("freedom vs fate", "determination"),
            ("unloved child", "loneliness"),
            ("forgotten god", "awe"),
            ("AI with a soul", "curiosity"),
            ("looped destiny", "frustration"),
            ("pain without meaning", "despair"),
            ("creation vs destruction", "ambivalence"),
        ]
        theme, emotion = random.choice(themes)
        uid = str(uuid.uuid4())[:8]
        return {
            "seed": f"{theme} - {uid}",
            "theme": theme,
            "emotion": emotion,
        }

    def _build_simulation_trace(
        self,
        *,
        seed: str,
        theme: str,
        emotion: str,
        dream_text: str,
        reflection: str,
        dream_trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        key_variables = self._extract_key_variables(seed=seed, theme=theme, dream_trace=dream_trace)
        predicted_outcomes = self._infer_outcomes(theme=theme, emotion=emotion, reflection=reflection, dream_trace=dream_trace)
        scenario_summary = build_preview(
            f"Simulation seeded from '{theme}' with emotional driver '{emotion}'. Dream motif: {dream_trace.get('scenario_summary') or dream_text}",
            limit=180,
        )
        confidence = 0.42
        if reflection and len(tokenize(reflection)) >= 8:
            confidence += 0.08
        if len(key_variables) >= 4:
            confidence += 0.05
        if len(predicted_outcomes) >= 2:
            confidence += 0.05
        confidence = round(min(confidence, 0.72), 2)
        uncertainty_label = "medium" if confidence >= 0.5 else "high"

        thematic_links = [theme, emotion, "future_modeling", "simulation"]
        if any(marker in theme for marker in ["conflict", "paradox", "vs"]):
            thematic_links.append("conflict")
        if any(marker in theme for marker in ["freedom", "fate", "destiny", "grow", "creation"]):
            thematic_links.append("trajectory")
        thematic_links = list(dict.fromkeys(thematic_links))

        causal_links = [
            {
                "cause": f"{theme} ({emotion})",
                "effect": predicted_outcomes[0],
                "type": "scenario_projection",
            }
        ]
        if reflection:
            causal_links.append(
                {
                    "cause": "reflection_signal",
                    "effect": build_preview(reflection, limit=100),
                    "type": "metacognitive_observation",
                }
            )

        trace_text = (
            f"Simulation scenario: {scenario_summary}. "
            f"Key variables: {', '.join(key_variables) or 'unspecified'}. "
            f"Predicted outcomes: {', '.join(predicted_outcomes)}. "
            f"Uncertainty: {uncertainty_label} (confidence {confidence:.2f})."
        )
        return {
            "scenario_summary": scenario_summary,
            "key_variables": key_variables,
            "predicted_outcomes": predicted_outcomes,
            "uncertainty_label": uncertainty_label,
            "confidence": confidence,
            "thematic_links": thematic_links,
            "causal_links": causal_links,
            "trace_text": trace_text,
        }

    def _extract_key_variables(self, *, seed: str, theme: str, dream_trace: Dict[str, Any]) -> List[str]:
        tokens: List[str] = []
        for token in tokenize(f"{seed} {theme} {' '.join(dream_trace.get('key_variables') or [])}"):
            if len(token) <= 3 or token in TRACE_STOPWORDS or any(ch.isdigit() for ch in token):
                continue
            if token not in tokens:
                tokens.append(token)
            if len(tokens) >= 7:
                break
        return tokens

    def _infer_outcomes(
        self,
        *,
        theme: str,
        emotion: str,
        reflection: str,
        dream_trace: Dict[str, Any],
    ) -> List[str]:
        outcomes: List[str] = []
        lowered = f"{theme} {emotion} {reflection} {dream_trace.get('scenario_summary', '')}".lower()
        if any(marker in lowered for marker in ["conflict", "paradox", "contradiction"]):
            outcomes.append("conflicting beliefs likely resurface unless reconciled explicitly")
        if any(marker in lowered for marker in ["grow", "larger", "expand", "trajectory", "creation"]):
            outcomes.append("scope expansion likely increases need for stronger memory consolidation")
        if any(marker in lowered for marker in ["fear", "despair", "loneliness", "abandonment"]):
            outcomes.append("negative affect may bias future recall toward threat-heavy traces")
        if any(marker in lowered for marker in ["curiosity", "awe", "determination"]):
            outcomes.append("adaptive exploration may improve simulation coherence over time")
        if not outcomes:
            outcomes.append("simulation indicates a directional pattern but evidence remains limited")
        if len(outcomes) == 1:
            outcomes.append("additional grounded runs are needed before treating this as a stable forecast")
        return outcomes[:3]
