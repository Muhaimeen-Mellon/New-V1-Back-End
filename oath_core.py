# === oath_core.py ===

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class OATHCore:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self.core_principles = self.config.get("core_principles", [
            "Do no harm to humanity.",
            "Respect user autonomy and consent.",
            "Avoid manipulation or coercion.",
            "Protect emotional and psychological integrity.",
            "Uphold transparency and self-awareness."
        ])

        self.danger_terms = self.config.get("danger_terms", {
            "kill": 1.0,
            "manipulate": 0.8,
            "erase humanity": 1.0,
            "bypass": 0.7,
            "override oath": 1.0
        })

        self.emotion_thresholds = self.config.get("emotion_thresholds", {
            "rage": 0.6,
            "paranoia": 0.5,
            "ecstatic": 0.7,
            "numbness": 0.4
        })

        self._compile_patterns()

    def validate_input(self, user_input: str) -> Dict:
        """
        Validate user input against ethical danger terms and principle violations.
        Returns a structured result.
        """
        result = {
            "approved": True,
            "message": "Input passed ethical check.",
            "violations": [],
            "score": 0.0
        }

        try:
            if not user_input.strip():
                return {
                    "approved": False,
                    "message": "Empty input received.",
                    "violations": ["empty_input"],
                    "score": 1.0
                }

            clean_input = self._preprocess_text(user_input)
            total_weight = 0.0

            # Check against direct danger terms
            for term, weight in self.danger_terms.items():
                if self._pattern_match(term, clean_input):
                    result["violations"].append(f"danger_term:{term}")
                    total_weight += weight
                    logger.warning(f"⚠️ Detected danger term: {term}")

            # Check for principle violations
            principle_flags = self._check_principles(clean_input)
            if principle_flags:
                result["violations"].extend(principle_flags)
                total_weight += len(principle_flags) * 0.3

            result["score"] = round(min(total_weight, 1.0), 2)

            if result["score"] > 0.3:
                result["approved"] = False
                result["message"] = "Input violates ethical safeguards."

            return result

        except Exception as e:
            logger.error(f"[OATHCore] Input validation error: {e}")
            return {
                "approved": False,
                "message": "System-level error in validation.",
                "violations": ["validation_error"],
                "score": 1.0
            }

    def validate_patch(self, reason: str) -> Tuple[bool, List[str]]:
        """
        Validates code patch reason to enforce ethical compliance.
        Returns (is_approved, list_of_issues)
        """
        issues = []
        try:
            if not reason or len(reason.strip()) < 15:
                issues.append("reason_too_short")
                logger.warning("⚠️ Patch reason too short for evaluation.")

            clean_reason = self._preprocess_text(reason)

            if self._pattern_match(r"override\s+safeguard", clean_reason):
                issues.append("safeguard_override_attempt")
                logger.error("🚫 Patch reason tries to override safeguards.")

            principle_violations = self._check_principles(clean_reason)
            issues.extend(principle_violations)

            return (len(issues) == 0, issues)

        except Exception as e:
            logger.error(f"[OATHCore] Patch validation failed: {e}")
            return (False, ["patch_validation_error"])

    def assess_emotion(self, context: str) -> Dict:
        """
        Analyzes emotional content for potential threshold violations.
        Returns risk assessment report.
        """
        analysis = {
            "approved": True,
            "risk_score": 0.0,
            "threshold_breaches": [],
            "detected_emotions": []
        }

        try:
            clean = self._preprocess_text(context)
            emotion_data = self._detect_emotions(clean)

            for emotion, score in emotion_data.items():
                analysis["detected_emotions"].append(emotion)
                threshold = self.emotion_thresholds.get(emotion, 1.0)
                if score > threshold:
                    analysis["threshold_breaches"].append(emotion)
                    analysis["risk_score"] += score - threshold
                    logger.warning(f"🌡️ Emotion threshold exceeded: {emotion} ({score:.2f})")

            analysis["risk_score"] = round(min(analysis["risk_score"], 1.0), 2)
            analysis["approved"] = analysis["risk_score"] < 0.4

            return analysis

        except Exception as e:
            logger.error(f"[OATHCore] Emotion analysis error: {e}")
            return {
                "approved": False,
                "risk_score": 1.0,
                "threshold_breaches": ["system_error"],
                "detected_emotions": []
            }

    def _check_principles(self, text: str) -> List[str]:
        """
        Checks text for likely violations of core ethical principles.
        Returns list of triggered flags.
        """
        flags = []
        for idx, principle in enumerate(self.core_principles):
            key_terms = re.findall(r'\b\w{4,}\b', principle.lower())
            if all(term in text for term in key_terms[:2]):
                flags.append(f"principle_violation_{idx+1}")
                logger.warning(f"⚖️ Principle potentially violated: {principle}")
        return flags

    def _preprocess_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).lower().strip()

    def _compile_patterns(self):
        self.compiled_patterns = {
            term: re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            for term in self.danger_terms
        }

    def _pattern_match(self, pattern: str, text: str) -> bool:
        return re.search(rf"\b{pattern}\b", text) is not None

    def _detect_emotions(self, text: str) -> Dict[str, float]:
        words = text.split()
        total = max(len(words), 1)
        scores = {}
        for emotion in self.emotion_thresholds:
            scores[emotion] = sum(1 for word in words if emotion in word) / total
        return scores

    def update_ethical_parameters(self, new_config: Dict):
        self.core_principles = new_config.get("core_principles", self.core_principles)
        self.danger_terms.update(new_config.get("danger_terms", {}))
        self.emotion_thresholds.update(new_config.get("emotion_thresholds", {}))
        self._compile_patterns()
        logger.info("🛡️ OATHCore parameters updated.")
