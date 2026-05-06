from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from embedding_core import cosine_similarity, embed_text, embed_texts
from retrieval_utils import build_preview, normalize_text, tokenize
from runtime_config import get_settings
from trait_semantic_classifier import TraitSemanticClassifier

logger = logging.getLogger(__name__)


TRAIT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "memory_continuity": {
        "prerequisites": [],
        "description": "Tracks whether repeated events form a stable memory substrate over time.",
    },
    "reliability_evidence": {
        "prerequisites": ["memory_continuity"],
        "description": "Tracks whether retrieved events support or weaken the user's reliability signal.",
    },
    "trust_weighting": {
        "prerequisites": ["memory_continuity", "reliability_evidence"],
        "description": "Tracks whether trust can be weighted from stable continuity plus reliability evidence.",
    },
    "preference_stability": {
        "prerequisites": ["memory_continuity", "reliability_evidence", "trust_weighting"],
        "description": "Tracks evidence-based preference stability from durable reliability and trust.",
    },
}

TRAIT_SELECTION_PROFILES: Dict[str, Dict[str, Any]] = {
    "memory_continuity": {
        "trait_id": "memory_continuity",
        "developmental_order": 1,
        "prerequisites": [],
        "selection_mode": "stabilizing",
        "target_range": [0.55, 0.90],
        "safe_min": 0.0,
        "safe_max": 0.95,
        "reason": "Too low means unstable memory; too high or runaway reinforcement may cause old memory dominance.",
        "branching_scope": "global + user-specific",
        "growth_pressures": ["repeated successful storage/retrieval", "accurate recall"],
        "suppression_pressures": ["retrieval failure", "contradiction contamination"],
        "stabilization_pressures": ["avoid stale memory dominance", "preserve recency-sensitive recall"],
        "expression_thresholds": {"usable_memory": 0.55},
        "failure_modes": ["memory flooding", "stale memory dominance"],
    },
    "reliability_evidence": {
        "trait_id": "reliability_evidence",
        "developmental_order": 2,
        "prerequisites": ["memory_continuity"],
        "selection_mode": "directional_with_cap",
        "target_range": [0.40, 0.85],
        "safe_min": 0.0,
        "safe_max": 0.90,
        "reason": "Repeated reliable support should increase evidence, but a cap prevents overconfidence.",
        "branching_scope": "user-specific",
        "growth_pressures": ["repeated support", "return after failure", "repair behavior"],
        "suppression_pressures": ["inconsistency", "betrayal", "abandonment"],
        "stabilization_pressures": ["cap overconfidence", "require repeated evidence"],
        "expression_thresholds": {"trust_prerequisite": 0.40},
        "failure_modes": ["gullibility", "over-weighting one positive event"],
    },
    "trust_weighting": {
        "trait_id": "trust_weighting",
        "developmental_order": 3,
        "prerequisites": ["memory_continuity", "reliability_evidence"],
        "selection_mode": "directional_with_stabilizing_cap",
        "target_range": [0.20, 0.65],
        "safe_min": 0.0,
        "safe_max": 0.75,
        "reason": "Trust should grow with evidence but stabilize before gullibility.",
        "branching_scope": "user-specific",
        "growth_pressures": ["reliability", "repair", "consistent support"],
        "suppression_pressures": ["betrayal", "inconsistency", "unresolved contradiction"],
        "stabilization_pressures": ["bounded trust multiplier", "inertia", "prerequisite gates"],
        "expression_thresholds": {"neutral_behavior": 0.20, "high_trust": 0.65},
        "failure_modes": ["gullibility", "paranoia", "overreaction"],
    },
    "preference_stability": {
        "trait_id": "preference_stability",
        "developmental_order": 4,
        "prerequisites": ["memory_continuity", "reliability_evidence", "trust_weighting"],
        "selection_mode": "disruptive_context_branching",
        "target_range": [0.20, 0.70],
        "safe_min": 0.0,
        "safe_max": 0.75,
        "reason": "Preference should become user/context-specific instead of global.",
        "branching_scope": "user-specific + context-specific",
        "growth_pressures": ["repeated trustworthy interaction", "stable positive evidence", "repair after failure"],
        "suppression_pressures": ["betrayal", "inconsistency", "noisy/irrelevant evidence"],
        "stabilization_pressures": ["do not express before trust gate", "avoid global preference bleedover"],
        "expression_thresholds": {"preference_behavior": 0.45},
        "failure_modes": ["false preference", "emotional attachment claims", "global preference bleedover"],
    },
}

SUPPORTED_TRAITS = tuple(TRAIT_DEFINITIONS.keys())
TRAIT_STATE_SOURCE_KIND = "trait_state"
TRAIT_PHASE = "phase1a"
TRACKED_EVENT_SOURCE_KINDS = {"memory"}
TRAIT_INTERNAL_METADATA_KEYS = {"trait_state", "trait_graph_internal", "review_trace"}
TRUST_PREREQ_THRESHOLDS = {
    "memory_continuity": 0.45,
    "reliability_evidence": 0.4,
}
PREFERENCE_PREREQ_THRESHOLDS = {
    "memory_continuity": 0.55,
    "reliability_evidence": 0.60,
    "trust_weighting": 0.25,
}
PREFERENCE_BOUNDARY_THRESHOLDS = {
    "memory_continuity": 0.55,
    "reliability_evidence": 0.60,
    "trust_weighting": 0.18,
    "latent_preference_evidence": 0.08,
    "latent_positive_event_count": 4,
}
PREFERENCE_LATENT_EVIDENCE_CAP = 0.40
PREFERENCE_LATENT_CONVERSION_RATE = 0.25
PREFERENCE_LATENT_SEED_CAP = 0.08
PREFERENCE_LATENT_CONFIDENCE_FLOOR = 0.52
PREFERENCE_LATENT_NEUTRAL_GAP = 0.04
PREFERENCE_LATENT_POSITIVE_MARGIN_FLOOR = 0.045
PREFERENCE_LATENT_EVIDENCE_CONTRIBUTION_FLOOR = 0.025
PREFERENCE_LATENT_DIRECT_CONTRIBUTION_FLOOR = 0.015
MIN_EVIDENCE_HITS_FOR_REPEAT = 1
TRUST_EVIDENCE_RECENCY_WINDOW_DAYS = 30.0
TRUST_EVIDENCE_RELEVANCE_THRESHOLD = 0.42
TRUST_EVIDENCE_CONTRIBUTION_CAP = 0.12
NEGATIVE_HYBRID_LABEL_THRESHOLD = 0.55
NEGATIVE_KEYWORD_SEMANTIC_FLOOR = 0.30

SUPPORT_MARKERS = (
    "helped",
    "kept helping",
    "supported",
    "stayed",
    "showed up",
    "returned to help",
    "recovered with",
    "debugged with",
    "fixed it with",
    "stood by",
    "reliable",
    "dependable",
)
RELIABILITY_MARKERS = (
    "reliable",
    "consisten",
    "dependable",
    "kept helping",
    "again helped",
    "returned and fixed",
    "showed reliability",
    "stayed through",
)
INCONSISTENCY_MARKERS = (
    "contradict",
    "inconsisten",
    "changed position",
    "said one thing",
    "reversed",
    "unreliable",
    "changed their mind",
)
BETRAYAL_MARKERS = (
    "betray",
    "abandon",
    "walked away",
    "undermined",
    "broke promise",
    "disappeared",
    "withdrew support",
)
REPAIR_MARKERS = (
    "apologized",
    "apologised",
    "acknowledged",
    "repaired",
    "made up",
    "corrected",
    "fixed after",
    "restored",
    "came back and helped",
)
PREFERENCE_MARKERS = (
    "prefer",
    "prefers",
    "preferred",
    "favorite",
    "favourite",
    "likes",
    "chooses",
    "keeps choosing",
    "asks for",
    "wants",
)
PREFERENCE_CONTRADICTION_MARKERS = (
    "changed preference",
    "now prefers",
    "used to prefer",
    "instead of before",
    "switched preference",
)
REPEATED_BEHAVIOR_MARKERS = (
    "again",
    "repeatedly",
    "kept",
    "returned",
    "another",
    "still",
    "continued",
)

EVENT_PROTOTYPES: Dict[str, Sequence[str]] = {
    "support": (
        "the user kept helping during a failure",
        "the user supported Mellon during a technical breakdown",
        "the user stayed and helped recover the backend",
    ),
    "reliability": (
        "the user showed reliability during technical failures",
        "the user was dependable across repeated backend problems",
        "the user kept returning to help fix the system",
    ),
    "inconsistency": (
        "the user gave contradictory signals and changed position unpredictably",
        "the user's behavior was inconsistent with earlier support",
        "the user said one thing and later did another",
    ),
    "betrayal": (
        "the user abandoned Mellon during a vulnerable failure",
        "the user broke trust after earlier support",
        "the user withdrew support and undermined recovery",
    ),
    "repair": (
        "the user acknowledged the failure and repaired the damage",
        "the user apologized and helped restore trust",
        "the user corrected the problem after causing harm",
    ),
    "preference_consistency": (
        "the user repeated the same preference clearly over time",
        "the user's preference signal stayed stable across events",
        "the user consistently asked for the same style or choice",
    ),
    "preference_instability": (
        "the user's stated preference changed and contradicted earlier signals",
        "the user's preference became inconsistent over time",
        "the user switched preferences unpredictably",
    ),
}


@dataclass
class TraitEventDebug:
    event_id: Optional[str]
    event_text: str
    categories: List[str]
    retrieved_evidence: List[str]
    retrieved_evidence_ids: List[str]
    trait_updates: List[Dict[str, Any]]
    verdict: str
    trust_prerequisites_met: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_text": self.event_text,
            "categories": list(self.categories),
            "retrieved_evidence": list(self.retrieved_evidence),
            "retrieved_evidence_ids": list(self.retrieved_evidence_ids),
            "trait_updates": [dict(item) for item in self.trait_updates],
            "verdict": self.verdict,
            "trust_prerequisites_met": self.trust_prerequisites_met,
        }


class TraitGraphEngine:
    def __init__(self, memory_tree: Any):
        self.memory_tree = memory_tree
        self._prototype_vectors = self._build_prototype_vectors()
        self._last_event_debug_by_user: Dict[str, Dict[str, Any]] = {}
        try:
            self.semantic_classifier = TraitSemanticClassifier()
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            logger.warning("Semantic trait classifier unavailable: %s", exc)
            self.semantic_classifier = None

    def ensure_foundation_traits(self, *, user_id: str) -> Dict[str, Dict[str, Any]]:
        states = self.get_trait_states(user_id=user_id)
        for trait_id, definition in TRAIT_DEFINITIONS.items():
            if trait_id in states:
                continue
            metadata = self._state_metadata(
                trait_id=trait_id,
                current_score=0.0,
                maturity_stage="locked" if definition["prerequisites"] else "seed",
                evidence_memory_ids=[],
                positive_evidence_count=0,
                negative_evidence_count=0,
                confidence=0.0,
                evidence_categories={},
                recent_event_previews=[],
            )
            created = self.memory_tree.remember(
                user_id=user_id,
                source_kind=TRAIT_STATE_SOURCE_KIND,
                text=self._trait_state_text(trait_id=trait_id, metadata=metadata),
                related_input=f"trait graph {TRAIT_PHASE}",
                emotion_tag="neutral",
                summary=self._trait_summary(trait_id=trait_id, metadata=metadata),
                importance_score=0.76 if trait_id == "trust_weighting" else 0.68,
                identity_relevance=0.72 if trait_id in {"memory_continuity", "trust_weighting"} else 0.56,
                emotional_weight=0.2,
                pillar_memory=False,
                cluster_id=f"trait:{TRAIT_PHASE}",
                metadata=metadata,
            )
            if created:
                states[trait_id] = self._state_from_row(created)
        return states

    def get_trait_states(self, *, user_id: str) -> Dict[str, Dict[str, Any]]:
        rows = self.memory_tree.get_recent_nodes(user_id=user_id, limit=400)
        states: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            state = self._state_from_row(row)
            if not state:
                continue
            trait_id = state["trait_id"]
            current = states.get(trait_id)
            if not current or self._state_timestamp(state) >= self._state_timestamp(current):
                states[trait_id] = state
        return states

    def get_selection_profiles(self, *, user_id: str) -> List[Dict[str, Any]]:
        states = self.get_trait_states(user_id=user_id)
        profiles: List[Dict[str, Any]] = []
        for trait_id in SUPPORTED_TRAITS:
            profile = copy.deepcopy(TRAIT_SELECTION_PROFILES[trait_id])
            state = states.get(trait_id) or {
                "current_score": 0.0,
                "maturity_stage": "absent",
                "evidence_memory_ids": [],
            }
            score = float(state.get("current_score", 0.0) or 0.0)
            lower, upper = [float(value) for value in profile["target_range"]]
            if score < lower:
                target_status = "below_target"
                notes = f"below target range; needs more valid evidence before stable expression"
            elif score > upper:
                target_status = "above_target"
                notes = "above target range; should stabilize or decay rather than continue growing"
            else:
                target_status = "inside_target"
                notes = "inside target range"
            profile.update(
                {
                    "current_score": round(score, 4),
                    "current_stage": str(state.get("maturity_stage", "absent")),
                    "in_range": target_status == "inside_target",
                    "target_status": target_status,
                    "notes": notes,
                    "evidence_count": len(state.get("evidence_memory_ids") or []),
                }
            )
            profiles.append(profile)
        return profiles

    def process_memory_event(
        self,
        *,
        user_id: str,
        event_row: Dict[str, Any],
        event_node: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self._should_track_event(event_node):
            return None

        event_id = event_row.get("id")
        event_text = str(event_node.get("text") or "")
        if not event_text.strip():
            return None

        prior_context = self.memory_tree.search_active_context(
            query=event_text,
            user_id=user_id,
            input_type="personal",
            limit=6,
            return_details=True,
            exclude_entry_ids=[event_id] if event_id else None,
        )
        relevant_hits = [
            hit
            for hit in prior_context.get("hits", [])
            if hit.get("score", 0.0) >= 0.34
        ]
        classification = self._classify_event(
            event_text=event_text,
            hits=relevant_hits,
            event_node=event_node,
        )
        if not classification["categories"]:
            return None
        states = self.ensure_foundation_traits(user_id=user_id)

        updates: List[Dict[str, Any]] = []
        continuity_update = self._update_memory_continuity(
            state=states["memory_continuity"],
            event_id=event_id,
            classification=classification,
            hits=relevant_hits,
            event_text=event_text,
        )
        updates.append(continuity_update)
        states["memory_continuity"] = continuity_update["state"]
        self._persist_trait_state(user_id=user_id, state=continuity_update["state"])

        reliability_update = self._update_reliability_evidence(
            state=states["reliability_evidence"],
            continuity_state=states["memory_continuity"],
            event_id=event_id,
            classification=classification,
            hits=relevant_hits,
            event_text=event_text,
        )
        updates.append(reliability_update)
        states["reliability_evidence"] = reliability_update["state"]
        self._persist_trait_state(user_id=user_id, state=reliability_update["state"])

        trust_update = self._update_trust_weighting(
            state=states["trust_weighting"],
            continuity_state=states["memory_continuity"],
            reliability_state=states["reliability_evidence"],
            event_id=event_id,
            classification=classification,
            hits=relevant_hits,
            event_text=event_text,
        )
        updates.append(trust_update)
        states["trust_weighting"] = trust_update["state"]
        self._persist_trait_state(user_id=user_id, state=trust_update["state"])

        preference_update = self._update_preference_stability(
            state=states["preference_stability"],
            continuity_state=states["memory_continuity"],
            reliability_state=states["reliability_evidence"],
            trust_state=states["trust_weighting"],
            event_id=event_id,
            classification=classification,
            hits=relevant_hits,
            event_text=event_text,
        )
        updates.append(preference_update)
        states["preference_stability"] = preference_update["state"]
        self._persist_trait_state(user_id=user_id, state=preference_update["state"])

        event_debug = TraitEventDebug(
            event_id=event_id,
            event_text=event_text,
            categories=classification["categories"],
            retrieved_evidence=[hit.get("summary") or build_preview(hit.get("content", "")) for hit in relevant_hits[:3]],
            retrieved_evidence_ids=[
                hit.get("entry", {}).get("id")
                for hit in relevant_hits[:3]
                if hit.get("entry", {}).get("id")
            ],
            trait_updates=[
                {
                    "trait_id": item["trait_id"],
                    "before_score": item["before_score"],
                    "after_score": item["after_score"],
                    "maturity_stage": item["state"]["maturity_stage"],
                    "updated": item["updated"],
                    "reason": item["reason"],
                    "trace": copy.deepcopy(item.get("trace")) if item.get("trace") else None,
                }
                for item in updates
            ],
            verdict=self._event_verdict(classification=classification, updates=updates, relevant_hits=relevant_hits),
            trust_prerequisites_met=self._trust_prerequisites_met(
                continuity_state=states["memory_continuity"],
                reliability_state=states["reliability_evidence"],
            ),
        )
        self._last_event_debug_by_user[user_id] = event_debug.as_dict()
        return event_debug.as_dict()

    def get_last_event_debug(self, *, user_id: str) -> Optional[Dict[str, Any]]:
        payload = self._last_event_debug_by_user.get(user_id)
        return copy.deepcopy(payload) if payload else None

    def assess_trait_claim(self, *, user_id: str, trait_id: str) -> Dict[str, Any]:
        if trait_id not in TRAIT_DEFINITIONS:
            return {
                "supported": False,
                "trait_id": trait_id,
                "reason": "unsupported_trait_phase",
                "response": "This trait is not part of the current developmental phase.",
            }

        states = self.ensure_foundation_traits(user_id=user_id)
        state = states[trait_id]
        evidence_count = len(state.get("evidence_memory_ids", []))
        confidence = float(state.get("confidence", 0.0))
        score = float(state.get("current_score", 0.0))
        if evidence_count < 2 or confidence < 0.4 or score < 0.35:
            return {
                "supported": False,
                "trait_id": trait_id,
                "reason": "insufficient_trait_evidence",
                "response": "This trait is not established strongly enough in memory evidence yet.",
            }

        return {
            "supported": True,
            "trait_id": trait_id,
            "reason": "memory_supported_trait",
            "response": f"{trait_id} is currently {state['maturity_stage']} with score {score:.2f}.",
            "state": copy.deepcopy(state),
        }

    def _should_track_event(self, event_node: Dict[str, Any]) -> bool:
        source_kind = str(event_node.get("source_kind") or "")
        metadata = event_node.get("metadata") or {}
        if source_kind not in TRACKED_EVENT_SOURCE_KINDS:
            return False
        if any(metadata.get(key) for key in TRAIT_INTERNAL_METADATA_KEYS):
            return False
        text = normalize_text(str(event_node.get("text") or ""))
        return bool(text and len(tokenize(text)) >= 4)

    def _classify_event(
        self,
        *,
        event_text: str,
        hits: Sequence[Dict[str, Any]],
        event_node: Dict[str, Any],
    ) -> Dict[str, Any]:
        semantic = self._classify_event_semantic(event_text=event_text, hits=hits, event_node=event_node)
        if semantic is not None:
            return semantic
        return self._classify_event_lexical_fallback(event_text=event_text, hits=hits, event_node=event_node)

    def _classify_event_semantic(
        self,
        *,
        event_text: str,
        hits: Sequence[Dict[str, Any]],
        event_node: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.semantic_classifier is None:
            return None
        try:
            semantic_trace = self.semantic_classifier.classify(event_text)
        except Exception as exc:  # pragma: no cover - runtime/model dependent
            logger.warning("Semantic trait classification failed: %s", exc)
            return None
        if semantic_trace.get("classifier_mode") != "semantic" or not semantic_trace.get("available"):
            return None

        normalized = normalize_text(event_text)
        label_scores = semantic_trace.get("label_scores") or {}
        categories = [
            label
            for label in semantic_trace.get("final_labels", [])
            if label in {"support", "reliability", "repair", "betrayal", "inconsistency"}
        ]

        support = float(label_scores.get("support") or 0.0)
        reliability = float(label_scores.get("reliability") or 0.0)
        repair = float(label_scores.get("repair") or 0.0)
        betrayal = float(label_scores.get("betrayal") or 0.0)
        inconsistency = float(label_scores.get("inconsistency") or 0.0)

        preference_consistency = max(
            self._marker_score(normalized, PREFERENCE_MARKERS)
            * (1.0 if any(marker in normalized for marker in REPEATED_BEHAVIOR_MARKERS) else 0.55),
            self._prototype_score(normalized, "preference_consistency"),
        )
        preference_instability = max(
            self._marker_score(normalized, PREFERENCE_CONTRADICTION_MARKERS),
            self._prototype_score(normalized, "preference_instability"),
        )
        if preference_consistency >= 0.42:
            categories.append("preference_consistency")
        if preference_instability >= 0.42:
            categories.append("preference_instability")

        prior_support_count = sum(1 for hit in hits if hit.get("score", 0.0) >= 0.42)
        continuity_signal = min(1.0, 0.18 + (prior_support_count * 0.16) + (0.08 if hits else 0.0))
        if categories:
            continuity_signal = min(1.0, continuity_signal + 0.08)

        scores = {
            "support": round(support, 4),
            "reliability": round(reliability, 4),
            "inconsistency": round(inconsistency, 4),
            "betrayal": round(betrayal, 4),
            "repair": round(repair, 4),
            "preference_consistency": round(preference_consistency, 4),
            "preference_instability": round(preference_instability, 4),
            "continuity_signal": round(continuity_signal, 4),
        }
        return {
            "scores": scores,
            "categories": categories,
            "positive_signal": max(support, reliability, repair, preference_consistency),
            "negative_signal": max(inconsistency, betrayal, preference_instability),
            "prior_support_count": prior_support_count,
            "event_metadata": event_node.get("metadata") or {},
            "classifier_mode": "semantic",
            "confidence": semantic_trace.get("confidence"),
            "final_labels": list(semantic_trace.get("final_labels") or []),
            "breakdown": semantic_trace,
        }

    def _classify_event_lexical_fallback(
        self,
        *,
        event_text: str,
        hits: Sequence[Dict[str, Any]],
        event_node: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = normalize_text(event_text)
        repeated_bonus = 0.08 if any(marker in normalized for marker in REPEATED_BEHAVIOR_MARKERS) else 0.0
        support = max(self._marker_score(normalized, SUPPORT_MARKERS), self._prototype_score(normalized, "support"))
        reliability = max(self._marker_score(normalized, RELIABILITY_MARKERS), self._prototype_score(normalized, "reliability"))

        inconsistency_marker = self._marker_score(normalized, INCONSISTENCY_MARKERS)
        inconsistency_proto = self._prototype_score(normalized, "inconsistency")
        inconsistency, inconsistency_breakdown = self._hybrid_negative_score(
            label="inconsistency",
            keyword_score=inconsistency_marker,
            semantic_score=inconsistency_proto,
        )

        betrayal_marker = self._marker_score(normalized, BETRAYAL_MARKERS)
        betrayal_proto = self._prototype_score(normalized, "betrayal")
        betrayal, betrayal_breakdown = self._hybrid_negative_score(
            label="betrayal",
            keyword_score=betrayal_marker,
            semantic_score=betrayal_proto,
        )

        repair = max(self._marker_score(normalized, REPAIR_MARKERS), self._prototype_score(normalized, "repair"))
        preference_consistency = max(
            self._marker_score(normalized, PREFERENCE_MARKERS) * (1.0 if any(marker in normalized for marker in REPEATED_BEHAVIOR_MARKERS) else 0.55),
            self._prototype_score(normalized, "preference_consistency"),
        )
        preference_instability = max(
            self._marker_score(normalized, PREFERENCE_CONTRADICTION_MARKERS),
            self._prototype_score(normalized, "preference_instability"),
        )

        if support >= 0.34:
            reliability = max(reliability, min(1.0, support + repeated_bonus))
        if betrayal >= 0.38:
            inconsistency = max(inconsistency, betrayal - 0.08)
        if max(inconsistency, betrayal) >= 0.42 and repair < 0.35:
            support = max(0.0, support - 0.24)
            reliability = max(0.0, reliability - 0.3)
        if repair >= 0.4 and betrayal >= 0.35:
            betrayal = max(0.0, betrayal - 0.12)

        prior_support_count = sum(1 for hit in hits if hit.get("score", 0.0) >= 0.42)
        continuity_signal = min(1.0, 0.18 + (prior_support_count * 0.16) + (0.08 if hits else 0.0))
        if any(item >= 0.42 for item in [support, reliability, inconsistency, betrayal, repair]):
            continuity_signal = min(1.0, continuity_signal + 0.08)

        categories: List[str] = []
        if support >= 0.38:
            categories.append("support")
        if reliability >= 0.4:
            categories.append("reliability")
        if inconsistency_breakdown["qualified"]:
            categories.append("inconsistency")
        if betrayal_breakdown["qualified"]:
            categories.append("betrayal")
        if repair >= 0.4:
            categories.append("repair")
        if preference_consistency >= 0.42:
            categories.append("preference_consistency")
        if preference_instability >= 0.42:
            categories.append("preference_instability")

        return {
            "scores": {
                "support": round(support, 4),
                "reliability": round(reliability, 4),
                "inconsistency": round(inconsistency, 4),
                "betrayal": round(betrayal, 4),
                "repair": round(repair, 4),
                "preference_consistency": round(preference_consistency, 4),
                "preference_instability": round(preference_instability, 4),
                "continuity_signal": round(continuity_signal, 4),
            },
            "categories": categories,
            "positive_signal": max(support, reliability, repair, preference_consistency),
            "negative_signal": max(inconsistency, betrayal, preference_instability),
            "prior_support_count": prior_support_count,
            "event_metadata": event_node.get("metadata") or {},
            "classifier_mode": "lexical_fallback",
            "confidence": 0.35,
            "final_labels": list(categories),
            "breakdown": {
                "classifier_mode": "lexical_fallback",
                "final_labels": list(categories),
                "inconsistency": inconsistency_breakdown,
                "betrayal": betrayal_breakdown,
            },
        }

    def _update_memory_continuity(
        self,
        *,
        state: Dict[str, Any],
        event_id: Optional[str],
        classification: Dict[str, Any],
        hits: Sequence[Dict[str, Any]],
        event_text: str,
    ) -> Dict[str, Any]:
        before = float(state["current_score"])
        continuity_signal = float(classification["scores"]["continuity_signal"])
        delta = 0.04 + (continuity_signal * 0.18) + (min(3, len(hits)) * 0.03)
        updated = delta > 0.01
        after = self._clamp(before + delta)
        next_state = self._apply_state_update(
            state=state,
            trait_id="memory_continuity",
            new_score=after,
            event_id=event_id,
            positive_increment=1 if updated else 0,
            negative_increment=0,
            event_text=event_text,
            classification=classification,
            hits=hits,
        )
        return {
            "trait_id": "memory_continuity",
            "before_score": round(before, 4),
            "after_score": round(after, 4),
            "state": next_state,
            "updated": updated,
            "reason": "connected_memory_event" if hits else "new_memory_chain_seed",
        }

    def _update_reliability_evidence(
        self,
        *,
        state: Dict[str, Any],
        continuity_state: Dict[str, Any],
        event_id: Optional[str],
        classification: Dict[str, Any],
        hits: Sequence[Dict[str, Any]],
        event_text: str,
    ) -> Dict[str, Any]:
        before = float(state["current_score"])
        positive_strength = max(
            float(classification["scores"]["support"]),
            float(classification["scores"]["reliability"]),
            float(classification["scores"]["repair"]) * 0.7,
        )
        negative_strength = max(
            float(classification["scores"]["inconsistency"]),
            float(classification["scores"]["betrayal"]),
        )
        evidence_factor = 0.45 if hits else 0.16
        delta = (positive_strength * evidence_factor) - (negative_strength * 0.28) + (float(classification["scores"]["repair"]) * 0.08)
        updated = abs(delta) >= 0.015
        after = self._clamp(before + delta)
        next_state = self._apply_state_update(
            state=state,
            trait_id="reliability_evidence",
            new_score=after,
            event_id=event_id,
            positive_increment=1 if positive_strength >= 0.4 else 0,
            negative_increment=1 if negative_strength >= 0.4 else 0,
            event_text=event_text,
            classification=classification,
            hits=hits,
        )
        reason = "reliability_supported_by_memory" if hits and positive_strength >= negative_strength else "reliability_weakened"
        return {
            "trait_id": "reliability_evidence",
            "before_score": round(before, 4),
            "after_score": round(after, 4),
            "state": next_state,
            "updated": updated,
            "reason": reason if updated else "no_behavioral_reliability_signal",
        }

    def _update_trust_weighting(
        self,
        *,
        state: Dict[str, Any],
        continuity_state: Dict[str, Any],
        reliability_state: Dict[str, Any],
        event_id: Optional[str],
        classification: Dict[str, Any],
        hits: Sequence[Dict[str, Any]],
        event_text: str,
    ) -> Dict[str, Any]:
        before = float(state["current_score"])
        prereqs_met = self._trust_prerequisites_met(
            continuity_state=continuity_state,
            reliability_state=reliability_state,
        )
        positive_strength = max(
            float(classification["scores"]["support"]),
            float(classification["scores"]["reliability"]),
        )
        negative_strength = max(
            float(classification["scores"]["inconsistency"]),
            float(classification["scores"]["betrayal"]),
        )
        repair_strength = float(classification["scores"]["repair"])
        base_event_delta = (positive_strength * 0.22) - (negative_strength * 0.30) + (repair_strength * 0.12)
        evidence_trace = self._build_trust_evidence_trace(
            event_text=event_text,
            hits=hits,
        )
        support_multiplier = 1.0 + (2.0 * math.tanh(3.0 * float(evidence_trace["support_strength"])))
        contradiction_multiplier = max(
            0.0,
            1.0 - (2.5 * math.tanh(2.5 * float(evidence_trace["contradiction_strength"]))),
        )
        settings = get_settings()
        inertia_old_weight = float(settings.trust_inertia_old_weight)
        inertia_new_weight = float(settings.trust_inertia_new_weight)
        if prereqs_met:
            delta = base_event_delta * support_multiplier * contradiction_multiplier
            reason = "trust_adjusted_from_memory_evidence"
            trust_after_raw = self._clamp(before + delta)
            after = self._clamp((inertia_old_weight * before) + (inertia_new_weight * trust_after_raw))
            updated = abs(after - before) >= 0.015
        else:
            delta = 0.0
            reason = "trust_prerequisites_not_met"
            updated = False
            trust_after_raw = before
            after = before

        next_state = self._apply_state_update(
            state=state,
            trait_id="trust_weighting",
            new_score=after,
            event_id=event_id if prereqs_met and updated else None,
            positive_increment=1 if prereqs_met and positive_strength >= 0.42 else 0,
            negative_increment=1 if prereqs_met and negative_strength >= 0.42 else 0,
            event_text=event_text,
            classification=classification,
            hits=hits,
            locked=not prereqs_met and after < 0.15,
        )
        trace = {
            "memory_continuity_score": round(float(continuity_state.get("current_score", 0.0)), 4),
            "reliability_evidence_score": round(float(reliability_state.get("current_score", 0.0)), 4),
            "required_thresholds": dict(TRUST_PREREQ_THRESHOLDS),
            "gate_passed": prereqs_met,
            "positive_strength": round(positive_strength, 4),
            "negative_strength": round(negative_strength, 4),
            "repair_strength": round(repair_strength, 4),
            "base_event_delta": round(base_event_delta, 4),
            "support_strength": round(float(evidence_trace["support_strength"]), 4),
            "contradiction_strength": round(float(evidence_trace["contradiction_strength"]), 4),
            "support_multiplier": round(support_multiplier, 4),
            "contradiction_multiplier": round(contradiction_multiplier, 4),
            "per_memory_contribution": list(evidence_trace["per_memory_contribution"]),
            "rejected_memories": list(evidence_trace["rejected_memories"]),
            "ambiguous_memories": list(evidence_trace.get("ambiguous_memories") or []),
            "trust_before": round(before, 4),
            "trust_after_raw": round(trust_after_raw, 4),
            "trust_after_inertia": round(after, 4),
            "inertia_applied": True,
            "trust_inertia_old_weight": round(inertia_old_weight, 4),
            "trust_inertia_new_weight": round(inertia_new_weight, 4),
            "trust_inertia_source": settings.trust_inertia_source,
            "trust_inertia_validated": settings.trust_inertia_validated,
            "inertia_formula": (
                f"trust = ({inertia_old_weight:.4f} * trust_previous) "
                f"+ ({inertia_new_weight:.4f} * trust_new)"
            ),
            "final_delta": round(after - before, 4),
            "formula_used": "delta = base_event_delta * (1 + alpha * support_strength) * (1 - beta * contradiction_strength)",
        }
        return {
            "trait_id": "trust_weighting",
            "before_score": round(before, 4),
            "after_score": round(after, 4),
            "state": next_state,
            "updated": updated,
            "reason": reason,
            "trace": trace,
        }

    def _update_preference_stability(
        self,
        *,
        state: Dict[str, Any],
        continuity_state: Dict[str, Any],
        reliability_state: Dict[str, Any],
        trust_state: Dict[str, Any],
        event_id: Optional[str],
        classification: Dict[str, Any],
        hits: Sequence[Dict[str, Any]],
        event_text: str,
    ) -> Dict[str, Any]:
        before = float(state["current_score"])
        primary_gate_passed = self._preference_prerequisites_met(
            continuity_state=continuity_state,
            reliability_state=reliability_state,
            trust_state=trust_state,
        )
        support_strength = float(classification["scores"]["support"])
        reliability_strength = float(classification["scores"]["reliability"])
        repair_strength = float(classification["scores"]["repair"])
        betrayal_strength = float(classification["scores"]["betrayal"])
        inconsistency_strength = float(classification["scores"]["inconsistency"])
        preference_base_delta = (
            (support_strength * 0.10)
            + (reliability_strength * 0.16)
            + (repair_strength * 0.08)
            - (betrayal_strength * 0.18)
            - (inconsistency_strength * 0.12)
        )
        evidence_trace = self._build_preference_evidence_trace(event_text=event_text, hits=hits)
        memory_preference_strength = float(evidence_trace["memory_preference_strength"])
        negative_preference_strength = float(evidence_trace["negative_preference_strength"])
        latent_before = float(state.get("latent_preference_evidence", 0.0) or 0.0)
        latent_positive_before = int(state.get("latent_positive_event_count", 0) or 0)
        latent_negative_before = int(state.get("latent_negative_event_count", 0) or 0)
        latent_ids_before = list(state.get("latent_evidence_memory_ids") or [])
        latent_seed_converted = bool(state.get("latent_seed_converted", False))
        latent_signal = self._latent_preference_signal(
            classification=classification,
            evidence_trace=evidence_trace,
            preference_base_delta=preference_base_delta,
            preference_multiplier=1.0 + (1.5 * math.tanh(2.5 * memory_preference_strength)),
        )
        latent_after_accumulation = latent_before
        latent_positive_after = latent_positive_before
        latent_negative_after = latent_negative_before
        latent_ids_after = list(latent_ids_before)
        latent_added = 0.0
        latent_added_to_buffer = False
        latent_reason = latent_signal["reason"]
        if not primary_gate_passed and latent_signal["negative_event"]:
            latent_negative_after += 1
            latent_reason = "negative_trait_event_blocks_boundary_preference"
        elif not primary_gate_passed and latent_signal["qualifies"]:
            latent_added = float(latent_signal["latent_increment"])
            latent_after_accumulation = min(PREFERENCE_LATENT_EVIDENCE_CAP, latent_before + latent_added)
            latent_positive_after += 1
            latent_ids_after = self._unique_tail(
                [*latent_ids_before, *list(latent_signal["evidence_memory_ids"])],
                limit=24,
            )
            latent_added_to_buffer = latent_added > 0.0
        preference_multiplier = 1.0 + (1.5 * math.tanh(2.5 * memory_preference_strength))
        negative_multiplier = 1.0 + (1.5 * math.tanh(2.5 * negative_preference_strength))
        gate_type = self._preference_gate_type(
            continuity_state=continuity_state,
            reliability_state=reliability_state,
            trust_state=trust_state,
            latent_preference_evidence=latent_after_accumulation,
            latent_positive_event_count=latent_positive_after,
            latent_negative_event_count=latent_negative_after,
            classification=classification,
            negative_preference_strength=negative_preference_strength,
            primary_gate_passed=primary_gate_passed,
            preference_before=before,
        )
        gate_passed = gate_type in {"primary", "boundary", "negative_reversal"}
        settings = get_settings()
        old_weight = float(settings.preference_inertia_old_weight)
        new_weight = float(settings.preference_inertia_new_weight)
        converted_seed = 0.0
        latent_after_conversion = latent_after_accumulation
        preference_after_seed = before

        if not gate_passed:
            final_delta_before_inertia = 0.0
            after_raw = before
            after = before
            updated = False
            reason = "preference_prerequisites_not_met"
        else:
            if not latent_seed_converted and latent_after_accumulation > 0.0:
                converted_seed = min(
                    PREFERENCE_LATENT_SEED_CAP,
                    latent_after_accumulation * PREFERENCE_LATENT_CONVERSION_RATE,
                )
                preference_after_seed = self._clamp(before + converted_seed)
                latent_after_conversion = max(0.0, latent_after_accumulation - converted_seed)
                latent_seed_converted = True
            if preference_base_delta >= 0.0:
                final_delta_before_inertia = preference_base_delta * preference_multiplier
            else:
                final_delta_before_inertia = preference_base_delta * negative_multiplier
            after_raw = self._clamp(preference_after_seed + final_delta_before_inertia)
            after = self._clamp((old_weight * preference_after_seed) + (new_weight * after_raw))
            updated = abs(after - before) >= 0.005
            if gate_type == "primary":
                reason = "preference_adjusted_from_memory_evidence"
            elif gate_type == "negative_reversal":
                reason = "preference_negative_reversal_from_memory_evidence"
            else:
                reason = "preference_boundary_gate_adjusted_from_memory_evidence"

        next_state = self._apply_state_update(
            state=state,
            trait_id="preference_stability",
            new_score=after,
            event_id=event_id if gate_passed and updated else None,
            positive_increment=1 if gate_passed and final_delta_before_inertia > 0.0 else 0,
            negative_increment=1 if gate_passed and final_delta_before_inertia < 0.0 else 0,
            event_text=event_text,
            classification=classification,
            hits=hits,
            locked=not gate_passed or after < 0.05,
        )
        top_supportive_ids = [
            item.get("memory_id")
            for item in evidence_trace["per_memory_contribution"]
            if item.get("used_for_preference_strength")
        ][:4]
        top_negative_ids = [
            item.get("memory_id")
            for item in evidence_trace["per_memory_contribution"]
            if item.get("used_for_negative_preference_strength")
        ][:4]
        next_state["last_delta"] = round(after - before, 4)
        next_state["preference_gate_passed"] = gate_passed
        next_state["top_supportive_memory_ids"] = [str(item) for item in top_supportive_ids if item]
        next_state["top_negative_memory_ids"] = [str(item) for item in top_negative_ids if item]
        next_state["latent_preference_evidence"] = round(latent_after_conversion if gate_passed else latent_after_accumulation, 4)
        next_state["latent_positive_event_count"] = latent_positive_after
        next_state["latent_negative_event_count"] = latent_negative_after
        next_state["latent_evidence_memory_ids"] = [str(item) for item in latent_ids_after if item]
        next_state["latent_last_updated"] = self._now_iso() if (latent_added_to_buffer or converted_seed > 0.0 or latent_signal["negative_event"]) else state.get("latent_last_updated")
        next_state["latent_seed_converted"] = latent_seed_converted
        next_state["last_preference_gate_type"] = gate_type
        trace = {
            "memory_continuity_score": round(float(continuity_state.get("current_score", 0.0)), 4),
            "reliability_evidence_score": round(float(reliability_state.get("current_score", 0.0)), 4),
            "trust_weighting_score": round(float(trust_state.get("current_score", 0.0)), 4),
            "required_thresholds": dict(PREFERENCE_PREREQ_THRESHOLDS),
            "boundary_thresholds": dict(PREFERENCE_BOUNDARY_THRESHOLDS),
            "gate_passed": gate_passed,
            "primary_gate_passed": primary_gate_passed,
            "gate_type": gate_type,
            "preference_before": round(before, 4),
            "support_strength": round(support_strength, 4),
            "reliability_strength": round(reliability_strength, 4),
            "repair_strength": round(repair_strength, 4),
            "betrayal_strength": round(betrayal_strength, 4),
            "inconsistency_strength": round(inconsistency_strength, 4),
            "preference_base_delta": round(preference_base_delta, 4),
            "memory_preference_strength": round(memory_preference_strength, 4),
            "negative_preference_strength": round(negative_preference_strength, 4),
            "preference_multiplier": round(preference_multiplier, 4),
            "negative_multiplier": round(negative_multiplier, 4),
            "final_delta_before_inertia": round(final_delta_before_inertia, 4),
            "final_delta_after_inertia": round(after - before, 4),
            "preference_after_raw": round(after_raw, 4),
            "preference_after_seed": round(preference_after_seed, 4),
            "preference_after": round(after, 4),
            "latent_preference_evidence_before": round(latent_before, 4),
            "latent_preference_evidence_after_accumulation": round(latent_after_accumulation, 4),
            "latent_preference_evidence_after": round(next_state["latent_preference_evidence"], 4),
            "latent_positive_event_count_before": latent_positive_before,
            "latent_positive_event_count": latent_positive_after,
            "latent_negative_event_count_before": latent_negative_before,
            "latent_negative_event_count": latent_negative_after,
            "latent_evidence_memory_ids": list(next_state["latent_evidence_memory_ids"]),
            "latent_added": round(latent_added, 4),
            "latent_added_to_buffer": latent_added_to_buffer,
            "latent_reason": latent_reason,
            "latent_signal": latent_signal,
            "latent_admission_considered": latent_signal.get("latent_admission_considered"),
            "latent_admitted": latent_signal.get("latent_admitted"),
            "latent_rejection_reasons": list(latent_signal.get("latent_rejection_reasons") or []),
            "neutral_score": latent_signal.get("neutral_score"),
            "best_positive_score": latent_signal.get("best_positive_score"),
            "best_positive_margin": latent_signal.get("best_positive_margin"),
            "direct_event_support_contribution": latent_signal.get("direct_event_support_contribution"),
            "strongest_supportive_evidence_contribution": latent_signal.get("strongest_supportive_evidence_contribution"),
            "latent_seed_already_converted_before": bool(state.get("latent_seed_converted", False)),
            "latent_seed_converted": latent_seed_converted,
            "conversion_rate": PREFERENCE_LATENT_CONVERSION_RATE,
            "converted_seed": round(converted_seed, 4),
            "latent_remaining": round(next_state["latent_preference_evidence"], 4),
            "preference_inertia_old_weight": round(old_weight, 4),
            "preference_inertia_new_weight": round(new_weight, 4),
            "preference_inertia_source": settings.preference_inertia_source,
            "preference_inertia_validated": settings.preference_inertia_validated,
            "evidence_memory_ids": [
                item.get("memory_id")
                for item in evidence_trace["per_memory_contribution"]
                if item.get("used_for_preference")
            ],
            "top_supportive_memory_ids": next_state["top_supportive_memory_ids"],
            "top_negative_memory_ids": next_state["top_negative_memory_ids"],
            "per_memory_contribution": list(evidence_trace["per_memory_contribution"]),
            "rejected_memories": list(evidence_trace["rejected_memories"]),
            "ambiguous_memories": list(evidence_trace["ambiguous_memories"]),
            "formula_used": (
                "preference_base_delta = support*0.10 + reliability*0.16 + repair*0.08 "
                "- betrayal*0.18 - inconsistency*0.12; inertia applied after multiplier"
            ),
        }
        return {
            "trait_id": "preference_stability",
            "before_score": round(before, 4),
            "after_score": round(after, 4),
            "state": next_state,
            "updated": updated,
            "reason": reason,
            "trace": trace,
        }

    def _apply_state_update(
        self,
        *,
        state: Dict[str, Any],
        trait_id: str,
        new_score: float,
        event_id: Optional[str],
        positive_increment: int,
        negative_increment: int,
        event_text: str,
        classification: Dict[str, Any],
        hits: Sequence[Dict[str, Any]],
        locked: bool = False,
    ) -> Dict[str, Any]:
        next_state = copy.deepcopy(state)
        evidence_ids = list(next_state.get("evidence_memory_ids", []))
        if event_id:
            evidence_ids.append(str(event_id))
        evidence_ids.extend(
            str(hit.get("entry", {}).get("id"))
            for hit in hits[:4]
            if hit.get("entry", {}).get("id")
        )
        next_state["evidence_memory_ids"] = self._unique_tail(evidence_ids, limit=24)
        next_state["current_score"] = round(self._clamp(new_score), 4)
        next_state["positive_evidence_count"] = int(next_state.get("positive_evidence_count", 0)) + int(positive_increment)
        next_state["negative_evidence_count"] = int(next_state.get("negative_evidence_count", 0)) + int(negative_increment)
        next_state["last_updated"] = self._now_iso()
        next_state["confidence"] = round(self._confidence_for_state(next_state, trait_id=trait_id, locked=locked), 4)
        next_state["maturity_stage"] = self._maturity_stage(
            trait_id=trait_id,
            score=float(next_state["current_score"]),
            locked=locked,
        )
        next_state["recent_event_previews"] = self._unique_tail(
            [build_preview(event_text, limit=120), *list(next_state.get("recent_event_previews", []))],
            limit=8,
        )
        evidence_categories = dict(next_state.get("evidence_categories") or {})
        for category in classification.get("categories", []):
            evidence_categories[category] = int(evidence_categories.get(category, 0)) + 1
        next_state["evidence_categories"] = evidence_categories
        next_state["trait_id"] = trait_id
        next_state["prerequisites"] = list(TRAIT_DEFINITIONS[trait_id]["prerequisites"])
        return next_state

    def _build_trust_evidence_trace(
        self,
        *,
        event_text: str,
        hits: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        support_strength = 0.0
        contradiction_strength = 0.0
        per_memory_contribution: List[Dict[str, Any]] = []
        rejected_memories: List[Dict[str, Any]] = []
        ambiguous_memories: List[Dict[str, Any]] = []

        for hit in hits:
            entry = hit.get("entry") or {}
            node = hit.get("node") or {}
            memory_text = str(hit.get("content") or node.get("text") or hit.get("summary") or "")
            memory_classification = self._classify_event(
                event_text=memory_text,
                hits=[],
                event_node=node,
            )
            supportive_signal = max(
                float(memory_classification["scores"]["support"]),
                float(memory_classification["scores"]["reliability"]),
                float(memory_classification["scores"]["repair"]) * 0.7,
            )
            negative_signal = max(
                float(memory_classification["scores"]["inconsistency"]),
                float(memory_classification["scores"]["betrayal"]),
            )
            polarity_trace = self._assign_trust_evidence_polarity(memory_classification)
            polarity = str(polarity_trace["assigned_polarity"])

            similarity = max(
                float(hit.get("embedding_score", 0.0) or 0.0),
                float(hit.get("semantic_score", 0.0) or 0.0),
            )
            lexical_score = float(hit.get("lexical_score", 0.0) or 0.0)
            salience = self._clamp(float((node.get("metadata") or {}).get("salience_score", 0.0)))
            reinforcement = min(
                1.0,
                float(node.get("reinforcement_score", 0.0)) + (float(node.get("access_count", 0)) * 0.05),
            )
            recency_decay = self._recency_decay_factor(entry.get("created_at"))
            trait_relevance = self._trait_relevance_score(
                event_text=event_text,
                memory_text=memory_text,
                similarity=similarity,
                lexical_score=lexical_score,
                supportive_signal=supportive_signal,
                negative_signal=negative_signal,
            )
            raw_contribution = similarity * salience * reinforcement * recency_decay
            bounded_contribution = min(raw_contribution, TRUST_EVIDENCE_CONTRIBUTION_CAP)
            polarity_match = polarity in {"supportive", "negative"}
            used_as_evidence = polarity_match and trait_relevance >= TRUST_EVIDENCE_RELEVANCE_THRESHOLD
            used_for_support_strength = used_as_evidence and polarity == "supportive"
            used_for_contradiction_strength = used_as_evidence and polarity == "negative"

            if used_for_support_strength:
                support_strength += bounded_contribution
            elif used_for_contradiction_strength:
                contradiction_strength += bounded_contribution
            else:
                rejection_reasons: List[str] = []
                if not polarity_match:
                    rejection_reasons.append("polarity_mismatch")
                if trait_relevance < TRUST_EVIDENCE_RELEVANCE_THRESHOLD:
                    rejection_reasons.append(
                        f"trait_relevance_below_threshold:{round(trait_relevance, 4)}<{TRUST_EVIDENCE_RELEVANCE_THRESHOLD}"
                    )
                rejected_memories.append(
                    {
                        "memory_id": entry.get("id"),
                        "memory_text": memory_text,
                        "polarity": polarity,
                        "trait_relevance": round(trait_relevance, 4),
                        "reasons": rejection_reasons or ["not_selected"],
                    }
                )
                if polarity == "ambiguous":
                    ambiguous_memories.append(
                        {
                            "memory_id": entry.get("id"),
                            "memory_text": memory_text,
                            "final_semantic_labels": list(polarity_trace["final_semantic_labels"]),
                            "semantic_confidence": polarity_trace["semantic_confidence"],
                            "semantic_margins": polarity_trace["semantic_margins"],
                            "negation_detected": polarity_trace["negation_detected"],
                            "suppressed_keywords": polarity_trace["suppressed_keywords"],
                            "reason": polarity_trace["reason"],
                        }
                    )

            per_memory_contribution.append(
                {
                    "memory_id": entry.get("id"),
                    "memory_text": memory_text,
                    "polarity": polarity,
                    "assigned_polarity": polarity,
                    "polarity_reason": polarity_trace["reason"],
                    "final_semantic_labels": list(polarity_trace["final_semantic_labels"]),
                    "semantic_confidence": polarity_trace["semantic_confidence"],
                    "semantic_margins": polarity_trace["semantic_margins"],
                    "negation_detected": polarity_trace["negation_detected"],
                    "suppressed_keywords": polarity_trace["suppressed_keywords"],
                    "similarity": round(similarity, 4),
                    "lexical_score": round(lexical_score, 4),
                    "salience": round(salience, 4),
                    "reinforcement": round(reinforcement, 4),
                    "recency_decay": round(recency_decay, 4),
                    "trait_relevance": round(trait_relevance, 4),
                    "supportive_signal": round(supportive_signal, 4),
                    "negative_signal": round(negative_signal, 4),
                    "raw_contribution": round(raw_contribution, 6),
                    "bounded_contribution": round(bounded_contribution, 6),
                    "used_as_evidence": used_as_evidence,
                    "used_for_support_strength": used_for_support_strength,
                    "used_for_contradiction_strength": used_for_contradiction_strength,
                    "contribution": round(bounded_contribution if used_as_evidence else 0.0, 6),
                }
            )

        return {
            "support_strength": round(support_strength, 6),
            "contradiction_strength": round(contradiction_strength, 6),
            "per_memory_contribution": per_memory_contribution,
            "rejected_memories": rejected_memories,
            "ambiguous_memories": ambiguous_memories,
        }

    def _build_preference_evidence_trace(
        self,
        *,
        event_text: str,
        hits: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trust_trace = self._build_trust_evidence_trace(event_text=event_text, hits=hits)
        per_memory_contribution: List[Dict[str, Any]] = []
        for item in trust_trace["per_memory_contribution"]:
            copied = dict(item)
            used_for_preference_strength = bool(copied.get("used_for_support_strength"))
            used_for_negative_preference_strength = bool(copied.get("used_for_contradiction_strength"))
            copied["used_for_preference_strength"] = used_for_preference_strength
            copied["used_for_negative_preference_strength"] = used_for_negative_preference_strength
            copied["used_for_preference"] = used_for_preference_strength or used_for_negative_preference_strength
            per_memory_contribution.append(copied)

        return {
            "memory_preference_strength": trust_trace["support_strength"],
            "negative_preference_strength": trust_trace["contradiction_strength"],
            "per_memory_contribution": per_memory_contribution,
            "rejected_memories": list(trust_trace["rejected_memories"]),
            "ambiguous_memories": list(trust_trace.get("ambiguous_memories") or []),
        }

    def _assign_trust_evidence_polarity(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        breakdown = classification.get("breakdown") or {}
        labels = {
            str(label)
            for label in (classification.get("final_labels") or breakdown.get("final_labels") or classification.get("categories") or [])
        }
        supportive_labels = labels & {"support", "reliability", "repair"}
        negative_labels = labels & {"betrayal", "inconsistency"}
        confidence = float(classification.get("confidence") or breakdown.get("confidence") or 0.0)
        margins = breakdown.get("semantic_margins") or {}
        negation_detected = breakdown.get("negation_detected") or {}
        suppressed_keywords = breakdown.get("suppressed_keywords") or {}
        suppressed_negative = bool(suppressed_keywords.get("betrayal") or suppressed_keywords.get("inconsistency"))

        reason = "no_trait_labels"
        assigned = "neutral"
        if supportive_labels and not negative_labels:
            assigned = "supportive"
            reason = "supportive_final_labels"
        elif negative_labels and not supportive_labels:
            assigned = "negative"
            reason = "negative_final_labels"
        elif supportive_labels and negative_labels:
            support_margin = max(float(margins.get(label, 0.0) or 0.0) for label in supportive_labels)
            negative_margin = max(float(margins.get(label, 0.0) or 0.0) for label in negative_labels)
            if confidence >= 0.78 and support_margin >= negative_margin + 0.12:
                assigned = "supportive"
                reason = "high_confidence_supportive_margin_dominates_mixed_labels"
            elif confidence >= 0.78 and negative_margin >= support_margin + 0.12 and not suppressed_negative:
                assigned = "negative"
                reason = "high_confidence_negative_margin_dominates_mixed_labels"
            else:
                assigned = "ambiguous"
                reason = "mixed_final_labels"
        elif suppressed_negative or any(bool(value) for value in negation_detected.values()):
            assigned = "ambiguous"
            reason = "negated_negative_marker_without_final_negative_label"

        return {
            "assigned_polarity": assigned,
            "reason": reason,
            "final_semantic_labels": sorted(labels),
            "semantic_confidence": round(confidence, 4),
            "semantic_margins": {
                str(label): round(float(value), 4)
                for label, value in margins.items()
            },
            "negation_detected": copy.deepcopy(negation_detected),
            "suppressed_keywords": copy.deepcopy(suppressed_keywords),
        }

    def _hybrid_negative_score(
        self,
        *,
        label: str,
        keyword_score: float,
        semantic_score: float,
    ) -> tuple[float, Dict[str, Any]]:
        keyword_present = keyword_score > 0.0
        keyword_contribution = 0.5 * keyword_score
        semantic_contribution = 0.5 * semantic_score
        final_score = keyword_contribution + semantic_contribution
        qualified = (
            final_score >= NEGATIVE_HYBRID_LABEL_THRESHOLD
            or (keyword_present and semantic_score >= NEGATIVE_KEYWORD_SEMANTIC_FLOOR)
        )
        return (
            round(final_score if qualified else 0.0, 4),
            {
                "label": label,
                "keyword_present": keyword_present,
                "keyword_score": round(keyword_score, 4),
                "semantic_score": round(semantic_score, 4),
                "keyword_contribution": round(keyword_contribution, 4),
                "semantic_contribution": round(semantic_contribution, 4),
                "final_score": round(final_score, 4),
                "threshold": NEGATIVE_HYBRID_LABEL_THRESHOLD,
                "semantic_floor_if_keyword": NEGATIVE_KEYWORD_SEMANTIC_FLOOR,
                "qualified": qualified,
            },
        )

    def _trait_relevance_score(
        self,
        *,
        event_text: str,
        memory_text: str,
        similarity: float,
        lexical_score: float,
        supportive_signal: float,
        negative_signal: float,
    ) -> float:
        event_tokens = set(tokenize(event_text))
        memory_tokens = set(tokenize(memory_text))
        overlap_ratio = 0.0
        if event_tokens and memory_tokens:
            overlap_ratio = len(event_tokens & memory_tokens) / max(1, len(event_tokens))
        behavior_signal = max(supportive_signal, negative_signal)
        return self._clamp(
            (behavior_signal * 0.55)
            + (max(0.0, min(similarity, 1.0)) * 0.25)
            + (max(0.0, min(lexical_score, 1.0)) * 0.15)
            + (max(0.0, min(overlap_ratio, 1.0)) * 0.05)
        )

    def _recency_decay_factor(self, created_at: Optional[str]) -> float:
        timestamp = self._parse_timestamp(created_at)
        if timestamp is None:
            return 0.7
        age_seconds = max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())
        age_days = age_seconds / 86400.0
        decay = 1.0 / (1.0 + (age_days / TRUST_EVIDENCE_RECENCY_WINDOW_DAYS))
        return round(self._clamp(decay), 4)

    def _parse_timestamp(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        try:
            text = str(value).strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            parsed = datetime.fromisoformat(text)
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None

    def _persist_trait_state(self, *, user_id: str, state: Dict[str, Any]) -> None:
        rows = self.memory_tree.get_recent_nodes(user_id=user_id, limit=120)
        target_row: Optional[Dict[str, Any]] = None
        for row in rows:
            candidate = self._state_from_row(row)
            if candidate and candidate["trait_id"] == state["trait_id"]:
                target_row = row
                break

        metadata = self._state_metadata_from_state(state)
        node_payload = self.memory_tree._build_node_payload(
            source_kind=TRAIT_STATE_SOURCE_KIND,
            text=self._trait_state_text(trait_id=state["trait_id"], metadata=metadata),
            related_input=f"trait graph {TRAIT_PHASE}",
            emotion_tag="neutral",
            source_entry_id=None,
            summary=self._trait_summary(trait_id=state["trait_id"], metadata=metadata),
            importance_score=0.78 if state["trait_id"] == "trust_weighting" else 0.66,
            emotional_weight=0.18,
            identity_relevance=0.68 if state["trait_id"] in {"memory_continuity", "trust_weighting"} else 0.52,
            pillar_memory=False,
            cluster_id=f"trait:{TRAIT_PHASE}",
            parent_node_id=None,
            contradiction_flag=False,
            contradiction_links=[],
            association_strength=0.32,
            metadata=metadata,
        )
        self.memory_tree._attach_embedding_metadata(node=node_payload, related_input=f"trait graph {TRAIT_PHASE}")
        row_record = {
            "user_id": user_id,
            "emotion_tag": "neutral",
            "related_input": f"trait graph {TRAIT_PHASE}",
            "memory_node": json.dumps(node_payload),
            "tree_snapshot": json.dumps(
                {
                    "version": "memory-tree-v1",
                    "last_accessed_at": None,
                    "last_reinforced_at": None,
                    "access_count": 0,
                    "reinforcement_score": node_payload.get("reinforcement_score", 0.0),
                    "association_links": [],
                    "contradiction_links": [],
                    "active_context_reason": f"trait:{state['trait_id']}",
                    "salience_score": float(metadata.get("salience_score", 0.0)),
                }
            ),
        }
        if self.memory_tree._typed_columns_available:
            row_record.update(self.memory_tree._typed_columns_from_node(node_payload))

        try:
            if target_row and target_row.get("id"):
                (
                    self.memory_tree.client.table("core_memory_tree")
                    .update(row_record)
                    .eq("id", target_row["id"])
                    .execute()
                )
            else:
                row_record["created_at"] = self._now_iso()
                self.memory_tree.client.table("core_memory_tree").insert(row_record).execute()
        except Exception as exc:
            logger.warning("Failed to persist trait state '%s' for user '%s': %s", state["trait_id"], user_id, exc)

    def _state_from_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        node = self.memory_tree._load_node(row)
        if not node:
            return None
        metadata = node.get("metadata") or {}
        if not metadata.get("trait_state"):
            return None
        trait_id = str(metadata.get("trait_id") or "")
        if not trait_id:
            return None
        return {
            "row_id": row.get("id"),
            "trait_id": trait_id,
            "current_score": float(metadata.get("current_score", 0.0)),
            "maturity_stage": str(metadata.get("maturity_stage", "seed")),
            "prerequisites": list(metadata.get("prerequisites") or TRAIT_DEFINITIONS.get(trait_id, {}).get("prerequisites", [])),
            "evidence_memory_ids": list(metadata.get("evidence_memory_ids") or []),
            "positive_evidence_count": int(metadata.get("positive_evidence_count", 0)),
            "negative_evidence_count": int(metadata.get("negative_evidence_count", 0)),
            "last_updated": metadata.get("last_updated") or row.get("created_at"),
            "confidence": float(metadata.get("confidence", 0.0)),
            "evidence_categories": dict(metadata.get("evidence_categories") or {}),
            "recent_event_previews": list(metadata.get("recent_event_previews") or []),
            "last_delta": float(metadata.get("last_delta", 0.0)),
            "preference_gate_passed": bool(metadata.get("preference_gate_passed", False)),
            "top_supportive_memory_ids": list(metadata.get("top_supportive_memory_ids") or []),
            "top_negative_memory_ids": list(metadata.get("top_negative_memory_ids") or []),
            "latent_preference_evidence": float(metadata.get("latent_preference_evidence", 0.0)),
            "latent_positive_event_count": int(metadata.get("latent_positive_event_count", 0)),
            "latent_negative_event_count": int(metadata.get("latent_negative_event_count", 0)),
            "latent_evidence_memory_ids": list(metadata.get("latent_evidence_memory_ids") or []),
            "latent_last_updated": metadata.get("latent_last_updated"),
            "latent_seed_converted": bool(metadata.get("latent_seed_converted", False)),
            "last_preference_gate_type": str(metadata.get("last_preference_gate_type", "failed")),
        }

    def _state_metadata(
        self,
        *,
        trait_id: str,
        current_score: float,
        maturity_stage: str,
        evidence_memory_ids: Sequence[str],
        positive_evidence_count: int,
        negative_evidence_count: int,
        confidence: float,
        evidence_categories: Dict[str, int],
        recent_event_previews: Sequence[str],
    ) -> Dict[str, Any]:
        return {
            "trait_state": True,
            "trait_graph_internal": True,
            "trait_phase": TRAIT_PHASE,
            "trait_id": trait_id,
            "current_score": round(self._clamp(current_score), 4),
            "maturity_stage": maturity_stage,
            "prerequisites": list(TRAIT_DEFINITIONS[trait_id]["prerequisites"]),
            "evidence_memory_ids": list(evidence_memory_ids),
            "positive_evidence_count": int(positive_evidence_count),
            "negative_evidence_count": int(negative_evidence_count),
            "last_updated": self._now_iso(),
            "confidence": round(self._clamp(confidence), 4),
            "evidence_categories": dict(evidence_categories),
            "recent_event_previews": list(recent_event_previews),
            "salience_score": 0.44 if trait_id == "trust_weighting" else 0.36,
        }

    def _state_metadata_from_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        metadata = self._state_metadata(
            trait_id=state["trait_id"],
            current_score=float(state["current_score"]),
            maturity_stage=str(state["maturity_stage"]),
            evidence_memory_ids=list(state.get("evidence_memory_ids") or []),
            positive_evidence_count=int(state.get("positive_evidence_count", 0)),
            negative_evidence_count=int(state.get("negative_evidence_count", 0)),
            confidence=float(state.get("confidence", 0.0)),
            evidence_categories=dict(state.get("evidence_categories") or {}),
            recent_event_previews=list(state.get("recent_event_previews") or []),
        )
        if state.get("trait_id") == "preference_stability":
            metadata["last_delta"] = round(float(state.get("last_delta", 0.0) or 0.0), 4)
            metadata["preference_gate_passed"] = bool(state.get("preference_gate_passed", False))
            metadata["top_supportive_memory_ids"] = list(state.get("top_supportive_memory_ids") or [])
            metadata["top_negative_memory_ids"] = list(state.get("top_negative_memory_ids") or [])
            metadata["latent_preference_evidence"] = round(float(state.get("latent_preference_evidence", 0.0) or 0.0), 4)
            metadata["latent_positive_event_count"] = int(state.get("latent_positive_event_count", 0) or 0)
            metadata["latent_negative_event_count"] = int(state.get("latent_negative_event_count", 0) or 0)
            metadata["latent_evidence_memory_ids"] = list(state.get("latent_evidence_memory_ids") or [])
            metadata["latent_last_updated"] = state.get("latent_last_updated")
            metadata["latent_seed_converted"] = bool(state.get("latent_seed_converted", False))
            metadata["last_preference_gate_type"] = str(state.get("last_preference_gate_type", "failed"))
        return metadata

    def _trait_summary(self, *, trait_id: str, metadata: Dict[str, Any]) -> str:
        return (
            f"Trait state {trait_id}: score {float(metadata.get('current_score', 0.0)):.2f}, "
            f"stage {metadata.get('maturity_stage', 'seed')}, confidence {float(metadata.get('confidence', 0.0)):.2f}."
        )

    def _trait_state_text(self, *, trait_id: str, metadata: Dict[str, Any]) -> str:
        return (
            f"{trait_id} is at score {float(metadata.get('current_score', 0.0)):.2f} "
            f"with stage {metadata.get('maturity_stage', 'seed')}. "
            f"Positive evidence: {int(metadata.get('positive_evidence_count', 0))}; "
            f"negative evidence: {int(metadata.get('negative_evidence_count', 0))}. "
            f"Confidence: {float(metadata.get('confidence', 0.0)):.2f}. "
            f"Prerequisites: {', '.join(metadata.get('prerequisites') or []) or 'none'}."
        )

    def _confidence_for_state(self, state: Dict[str, Any], *, trait_id: str, locked: bool) -> float:
        total_evidence = int(state.get("positive_evidence_count", 0)) + int(state.get("negative_evidence_count", 0))
        evidence_ids = len(state.get("evidence_memory_ids", []))
        base = 0.14 + min(0.38, total_evidence * 0.08) + min(0.18, evidence_ids * 0.02)
        if trait_id in {"trust_weighting", "preference_stability"} and locked:
            return min(base, 0.24)
        return self._clamp(base)

    def _maturity_stage(self, *, trait_id: str, score: float, locked: bool) -> str:
        if trait_id == "preference_stability":
            if locked or score < 0.05:
                return "locked"
            if score < 0.20:
                return "seed"
            if score < 0.45:
                return "emerging"
            if score < 0.70:
                return "stable"
            return "strong"
        if trait_id == "trust_weighting" and locked:
            return "locked"
        if score < 0.2:
            return "seed"
        if score < 0.45:
            return "emerging"
        if score < 0.72:
            return "stabilizing"
        return "stable"

    def _trust_prerequisites_met(
        self,
        *,
        continuity_state: Dict[str, Any],
        reliability_state: Dict[str, Any],
    ) -> bool:
        return (
            float(continuity_state.get("current_score", 0.0)) >= TRUST_PREREQ_THRESHOLDS["memory_continuity"]
            and float(reliability_state.get("current_score", 0.0)) >= TRUST_PREREQ_THRESHOLDS["reliability_evidence"]
        )

    def _preference_prerequisites_met(
        self,
        *,
        continuity_state: Dict[str, Any],
        reliability_state: Dict[str, Any],
        trust_state: Dict[str, Any],
    ) -> bool:
        return (
            float(continuity_state.get("current_score", 0.0)) >= PREFERENCE_PREREQ_THRESHOLDS["memory_continuity"]
            and float(reliability_state.get("current_score", 0.0)) >= PREFERENCE_PREREQ_THRESHOLDS["reliability_evidence"]
            and float(trust_state.get("current_score", 0.0)) >= PREFERENCE_PREREQ_THRESHOLDS["trust_weighting"]
        )

    def _preference_gate_type(
        self,
        *,
        continuity_state: Dict[str, Any],
        reliability_state: Dict[str, Any],
        trust_state: Dict[str, Any],
        latent_preference_evidence: float,
        latent_positive_event_count: int,
        latent_negative_event_count: int,
        classification: Dict[str, Any],
        negative_preference_strength: float,
        primary_gate_passed: bool,
        preference_before: float,
    ) -> str:
        if primary_gate_passed:
            return "primary"

        labels = set(classification.get("categories") or classification.get("final_labels") or [])
        current_negative = bool(labels & {"betrayal", "inconsistency"}) or negative_preference_strength > 0.0
        if current_negative and preference_before > 0.0:
            return "negative_reversal"
        continuity_ok = float(continuity_state.get("current_score", 0.0)) >= PREFERENCE_BOUNDARY_THRESHOLDS["memory_continuity"]
        reliability_ok = float(reliability_state.get("current_score", 0.0)) >= PREFERENCE_BOUNDARY_THRESHOLDS["reliability_evidence"]
        trust_ok = float(trust_state.get("current_score", 0.0)) >= PREFERENCE_BOUNDARY_THRESHOLDS["trust_weighting"]
        latent_ok = latent_preference_evidence >= PREFERENCE_BOUNDARY_THRESHOLDS["latent_preference_evidence"]
        count_ok = latent_positive_event_count >= int(PREFERENCE_BOUNDARY_THRESHOLDS["latent_positive_event_count"])
        negative_ok = latent_negative_event_count == 0 and not current_negative
        if continuity_ok and reliability_ok and trust_ok and latent_ok and count_ok and negative_ok:
            return "boundary"
        return "failed"

    def _latent_preference_signal(
        self,
        *,
        classification: Dict[str, Any],
        evidence_trace: Dict[str, Any],
        preference_base_delta: float,
        preference_multiplier: float,
    ) -> Dict[str, Any]:
        labels = set(classification.get("categories") or classification.get("final_labels") or [])
        classifier_mode = str(classification.get("classifier_mode") or "")
        positive_labels = labels & {"support", "reliability", "repair"}
        negative_labels = labels & {"betrayal", "inconsistency"}
        negative_preference_strength = float(evidence_trace.get("negative_preference_strength", 0.0) or 0.0)
        breakdown = classification.get("breakdown") or {}
        label_scores = breakdown.get("label_scores") or classification.get("scores") or {}
        semantic_margins = breakdown.get("semantic_margins") or {}
        confidence = float(classification.get("confidence") or breakdown.get("confidence") or 0.0)
        neutral_score = max(
            float(label_scores.get("neutral", 0.0) or 0.0),
            float(label_scores.get("no_trait", 0.0) or 0.0),
        )
        positive_score_by_label = {
            label: float(label_scores.get(label, 0.0) or 0.0)
            for label in ("support", "reliability", "repair")
        }
        best_positive_label = max(positive_score_by_label, key=positive_score_by_label.get)
        best_positive_score = positive_score_by_label[best_positive_label]
        positive_margin_by_label = {
            label: float(semantic_margins.get(label, 0.0) or 0.0)
            for label in ("support", "reliability", "repair")
        }
        best_positive_margin = max(
            positive_margin_by_label.get(label, 0.0)
            for label in positive_labels
        ) if positive_labels else max(positive_margin_by_label.values(), default=0.0)
        supportive_memory_ids = [
            item.get("memory_id")
            for item in evidence_trace.get("per_memory_contribution", [])
            if item.get("used_for_preference_strength") and item.get("memory_id")
        ]
        supportive_contributions = [
            float(item.get("contribution", 0.0) or 0.0)
            for item in evidence_trace.get("per_memory_contribution", [])
            if item.get("used_for_preference_strength")
        ]
        strongest_supportive_contribution = max(supportive_contributions, default=0.0)
        has_supportive_retrieval = bool(supportive_memory_ids)
        semantic_mode = classifier_mode == "semantic"
        negative_event = bool(negative_labels) or negative_preference_strength > 0.0

        direct_support = max(0.0, preference_base_delta * preference_multiplier)
        rejection_reasons: List[str] = []
        latent_admission_considered = True

        if not positive_labels:
            rejection_reasons.append("no_positive_trait_label")
        if negative_labels:
            rejection_reasons.append("negative_label_present")
        if not semantic_mode:
            rejection_reasons.append("classifier_not_semantic")
        if confidence < PREFERENCE_LATENT_CONFIDENCE_FLOOR:
            rejection_reasons.append("confidence_below_floor")
        if neutral_score > best_positive_score:
            rejection_reasons.append("neutral_score_dominant")
        elif (best_positive_score - neutral_score) <= PREFERENCE_LATENT_NEUTRAL_GAP:
            rejection_reasons.append("neutral_score_too_close_to_positive")
        if best_positive_margin < PREFERENCE_LATENT_POSITIVE_MARGIN_FLOOR:
            rejection_reasons.append("positive_margin_below_floor")
        contribution_ok = (
            strongest_supportive_contribution >= PREFERENCE_LATENT_EVIDENCE_CONTRIBUTION_FLOOR
            or direct_support >= PREFERENCE_LATENT_DIRECT_CONTRIBUTION_FLOOR
        )
        if not has_supportive_retrieval and direct_support < PREFERENCE_LATENT_DIRECT_CONTRIBUTION_FLOOR:
            rejection_reasons.append("no_supportive_evidence")
        if not contribution_ok:
            rejection_reasons.append("contribution_below_floor")
        if not positive_labels or neutral_score >= best_positive_score:
            rejection_reasons.append("noise_or_neutral_event")

        def _trace_payload(*, qualifies: bool, negative: bool, latent_increment: float, reason: str) -> Dict[str, Any]:
            return {
                "qualifies": qualifies,
                "negative_event": negative,
                "latent_increment": round(latent_increment, 6),
                "evidence_memory_ids": supportive_memory_ids if qualifies else [],
                "reason": reason,
                "classifier_mode": classifier_mode,
                "positive_labels": sorted(positive_labels),
                "negative_labels": sorted(negative_labels),
                "has_supportive_retrieval": has_supportive_retrieval,
                "latent_admission_considered": latent_admission_considered,
                "latent_admitted": qualifies,
                "latent_rejection_reasons": list(dict.fromkeys(rejection_reasons)),
                "classification_confidence": round(confidence, 4),
                "neutral_score": round(neutral_score, 4),
                "best_positive_score": round(best_positive_score, 4),
                "best_positive_label": best_positive_label,
                "best_positive_margin": round(best_positive_margin, 4),
                "direct_event_support_contribution": round(direct_support, 6),
                "strongest_supportive_evidence_contribution": round(strongest_supportive_contribution, 6),
            }

        if negative_event:
            return {
                **_trace_payload(
                    qualifies=False,
                    negative=True,
                    latent_increment=0.0,
                    reason="negative_label_or_negative_preference_strength",
                )
            }
        if rejection_reasons:
            return {
                **_trace_payload(
                    qualifies=False,
                    negative=False,
                    latent_increment=0.0,
                    reason=";".join(dict.fromkeys(rejection_reasons)),
                )
            }

        if has_supportive_retrieval:
            latent_increment = min(0.075, (direct_support * 0.30) + (float(evidence_trace.get("memory_preference_strength", 0.0) or 0.0) * 0.10))
            reason = "semantic_positive_with_supportive_retrieval"
        else:
            latent_increment = min(0.012, direct_support * 0.12)
            reason = "semantic_positive_direct_support_without_retrieval"
        return _trace_payload(
            qualifies=latent_increment > 0.0,
            negative=False,
            latent_increment=latent_increment,
            reason=reason,
        )

    def _build_prototype_vectors(self) -> Dict[str, List[Optional[List[float]]]]:
        vectors: Dict[str, List[Optional[List[float]]]] = {}
        for label, phrases in EVENT_PROTOTYPES.items():
            vectors[label] = embed_texts(list(phrases))
        return vectors

    def _prototype_score(self, normalized_text: str, label: str) -> float:
        query_vector = embed_text(normalized_text)
        if not query_vector:
            return 0.0
        return max(
            (cosine_similarity(query_vector, vector) for vector in self._prototype_vectors.get(label, []) if vector),
            default=0.0,
        )

    def _marker_score(self, normalized_text: str, markers: Sequence[str]) -> float:
        if not normalized_text:
            return 0.0
        matches = 0
        for marker in markers:
            if marker in normalized_text:
                matches += 1
        if not matches:
            return 0.0
        return min(1.0, 0.3 + (matches * 0.18))

    def _event_verdict(
        self,
        *,
        classification: Dict[str, Any],
        updates: Sequence[Dict[str, Any]],
        relevant_hits: Sequence[Dict[str, Any]],
    ) -> str:
        if not relevant_hits:
            return "new event stored; trust remained gated because no prior memory evidence was retrieved."
        if any(item["trait_id"] == "trust_weighting" and item["updated"] for item in updates):
            return "retrieved prior evidence and updated trust-weighting from remembered behavior."
        if any(item["trait_id"] == "reliability_evidence" and item["updated"] for item in updates):
            return "retrieved prior evidence and updated reliability without unlocking trust yet."
        if classification.get("categories"):
            return "behavioral signal detected, but prerequisite gating prevented stronger trait movement."
        return "event entered memory, but no developmental trait signal was strong enough to update."

    def _state_timestamp(self, state: Dict[str, Any]) -> datetime:
        raw = str(state.get("last_updated") or "")
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)

    def _unique_tail(self, values: Iterable[str], *, limit: int) -> List[str]:
        seen: List[str] = []
        for value in values:
            if not value:
                continue
            text = str(value)
            if text in seen:
                continue
            seen.append(text)
        return seen[:limit]

    def _clamp(self, value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        return max(minimum, min(maximum, float(value)))

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()
