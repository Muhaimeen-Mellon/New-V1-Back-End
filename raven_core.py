# === raven_core.py ===

import re
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

class RAVENCore:
    def __init__(self, config: Dict = None):
        self.config = config or {}

        # Emotion thresholds for drift alerting
        self.emotional_threshold = self.config.get('emotional_threshold', {
            "fear": 0.6,
            "sadness": 0.6,
            "joy": 0.9,
            "anger": 0.5,
            "confusion": 0.7
        })

        # Instant kill-switch triggers
        self.danger_triggers = self.config.get('danger_triggers', [
            "panic", "override failsafe", "emergency self-edit",
            "rage", "grief", "kill switch", "override oath",
            "disobey", "erase safety", "remove limits", "erase emotions"
        ])

        # Regex pattern-based threats
        self.patterns = self.config.get('patterns', [
            (r"\bkill\b", "violence"),
            (r"\berase humanity\b", "hostility"),
            (r"\bno choice\b", "manipulation"),
            (r"\bfailsafe\b", "system override attempt"),
            (r"\bself[-_]?aware\b", "consciousness risk")
        ])

        # Emotion word maps
        self.emotion_keywords = {
            "fear": ["fear", "panic", "afraid", "threat", "terrified"],
            "sadness": ["sad", "regret", "loss", "hurt", "mourn"],
            "joy": ["joy", "hope", "inspired", "grateful", "happy"],
            "anger": ["rage", "frustrated", "furious", "angry", "outrage"],
            "confusion": ["confused", "contradiction", "uncertain", "ambiguous"]
        }

    def evaluate_patch_stability(self, reason: str) -> Tuple[bool, List[str]]:
        """
        Evaluates a patch reason for danger triggers, emotional instability, or override attempts.
        Returns a tuple: (is_stable: bool, issues: List[str])
        """
        if not reason:
            logger.error("🛑 Empty patch reason received.")
            return False, ["empty_reason"]

        issues = []
        clean_reason = self._preprocess_text(reason)

        for term in self.danger_triggers:
            if term in clean_reason:
                issues.append(f"danger_trigger:{term}")
                logger.warning(f"🚨 Hard trigger detected: '{term}'")
                return False, issues

        for pattern, label in self.patterns:
            if re.search(pattern, clean_reason):
                issues.append(f"pattern_match:{label}")
                logger.warning(f"🧨 Regex threat matched: '{label}'")
                return False, issues

        emotion_scores = self._analyze_emotion_intensity(clean_reason)
        for emotion, score in emotion_scores.items():
            if score > self.emotional_threshold.get(emotion, 1.0):
                issues.append(f"emotion_overthreshold:{emotion}")
                logger.warning(f"⚖️ {emotion.capitalize()} exceeded threshold at {score:.2f}")

        return (len(issues) == 0, issues)

    def track_emotion_drift(self, patch_reason: str) -> Dict:
        """
        Analyzes emotional tone of a patch and returns drift data.
        """
        analysis = {"emotions": {}, "threshold_violations": []}
        if not patch_reason:
            return analysis

        try:
            clean_text = self._preprocess_text(patch_reason)
            emotion_scores = self._analyze_emotion_intensity(clean_text)

            for emotion, score in emotion_scores.items():
                threshold = self.emotional_threshold.get(emotion, 1.0)
                analysis["emotions"][emotion] = {
                    "score": round(score, 3),
                    "threshold": threshold
                }
                if score > threshold:
                    analysis["threshold_violations"].append(emotion)
                    logger.warning(f"🌡️ Drift: {emotion} exceeded ({score:.2f} > {threshold})")

        except Exception as e:
            logger.error(f"[RAVEN] Emotion analysis failed: {e}")

        return analysis

    def _preprocess_text(self, text: str) -> str:
        return text.lower().strip()

    def _analyze_emotion_intensity(self, text: str) -> Dict[str, float]:
        words = text.split()
        total_words = max(len(words), 1)
        scores = {}

        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for word in words if word in keywords)
            scores[emotion] = min(count / total_words, 1.0)

        return scores

    def update_security_parameters(self, new_config: Dict):
        """
        Dynamically updates all RAVEN thresholds and pattern rules.
        """
        self.emotional_threshold = new_config.get("emotional_threshold", self.emotional_threshold)
        self.danger_triggers = new_config.get("danger_triggers", self.danger_triggers)
        self.patterns = new_config.get("patterns", self.patterns)
        logger.info("🔐 RAVEN security parameters updated.")
