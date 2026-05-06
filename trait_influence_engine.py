from __future__ import annotations

from typing import Any, Dict, Optional


def _split_trait_states(
    trait_state: Optional[Dict[str, Any]],
    preference_state: Optional[Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if trait_state and "trust_weighting" in trait_state:
        trust = trait_state.get("trust_weighting") or {}
        preference = preference_state or trait_state.get("preference_stability") or {}
        return trust, preference
    return trait_state or {}, preference_state or {}


def apply_trait_influence(
    response_context: Dict[str, Any],
    trait_state: Dict[str, Any] | None,
    preference_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    context = dict(response_context or {})
    trust_state, preference_state = _split_trait_states(trait_state, preference_state)
    trust_score = float((trust_state or {}).get("current_score", 0.0))
    preference_score = float((preference_state or {}).get("current_score", 0.0))

    helpfulness_weight = float(context.get("helpfulness_weight", 1.0))
    verbosity = float(context.get("verbosity", 1.0))
    cooperation_bias = float(context.get("cooperation_bias", 0.0))
    caution_bias = float(context.get("caution_bias", 0.0))
    familiarity_bias = float(context.get("familiarity_bias", 0.0))
    continuity_bias = float(context.get("continuity_bias", 0.0))
    response_personalization = float(context.get("response_personalization", 0.0))

    band = "neutral"
    if trust_score > 0.65:
        helpfulness_weight += 0.25
        verbosity *= 1.20
        cooperation_bias += 0.2
        band = "high_trust"
    elif trust_score < 0.20:
        helpfulness_weight -= 0.2
        verbosity *= 0.85
        caution_bias += 0.3
        band = "low_trust"

    preference_effect_band = "none"
    if preference_score >= 0.70:
        familiarity_bias += 0.25
        continuity_bias += 0.30
        response_personalization += 0.25
        preference_effect_band = "strong_preference_stability"
    elif preference_score >= 0.45:
        familiarity_bias += 0.15
        continuity_bias += 0.20
        response_personalization += 0.15
        preference_effect_band = "stable_preference"

    return {
        **context,
        "helpfulness_weight": round(helpfulness_weight, 4),
        "verbosity": round(verbosity, 4),
        "cooperation_bias": round(cooperation_bias, 4),
        "caution_bias": round(caution_bias, 4),
        "familiarity_bias": round(familiarity_bias, 4),
        "continuity_bias": round(continuity_bias, 4),
        "response_personalization": round(response_personalization, 4),
        "trust_score": round(max(0.0, min(trust_score, 1.0)), 4),
        "preference_score": round(max(0.0, min(preference_score, 1.0)), 4),
        "trait_influence_band": band,
        "preference_effect_band": preference_effect_band,
    }


class TraitInfluenceEngine:
    def apply_trait_influence(
        self,
        response_context: Dict[str, Any],
        trait_state: Dict[str, Any] | None,
        preference_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return apply_trait_influence(response_context, trait_state, preference_state)
