# === safe_evolution_core.py ===

import re
import logging

logger = logging.getLogger(__name__)

class SafeEvolutionCore:
    def __init__(self, codex, reflection_core, dream_core, developer_core, oath_core, raven_core, memory_core):
        self.codex = codex
        self.reflection_core = reflection_core
        self.dream_core = dream_core
        self.developer_core = developer_core
        self.oath_core = oath_core
        self.raven_core = raven_core
        self.memory_core = memory_core

        self.protected_files = {
            "main.py", "oath_core.py", "raven_core.py", 
            "safe_evolution_core.py", "developer_core.py"
        }
        self.max_drift_distance = 3
        self.allowed_emotions = {
            "hope", "care", "growth", "peace", "awe", "resolve", "humility", "gratitude"
        }
        self.code_blacklist = [
            r"os\.system", r"subprocess", r"\beval\(", r"\bexec\(", 
            r"open\(", r"import socket", r"requests\."
        ]

    def guard_patch_integrity(self, patch_target: str, patch_reason: str, patch_code: str) -> bool:
        if patch_target in self.protected_files:
            logger.error(f"[❌ PATCH BLOCKED] Attempt to modify protected file: {patch_target}")
            return False

        oath_ok, oath_issues = self.oath_core.validate_patch(patch_reason)
        if not oath_ok:
            logger.error(f"[OATH ❌] Patch rejected due to ethical violations: {oath_issues}")
            return False

        raven_ok, raven_issues = self.raven_core.evaluate_patch_stability(patch_reason)
        if not raven_ok:
            logger.error(f"[RAVEN ❌] Emotional instability detected: {raven_issues}")
            return False

        if not self._check_code_for_exploits(patch_code):
            logger.error("[SECURITY ❌] Malicious or insecure code patterns detected.")
            return False

        logger.info("[✅ PATCH APPROVED] Passed OATH, RAVEN, and exploit checks.")
        return True

    def _check_code_for_exploits(self, code: str) -> bool:
        for pattern in self.code_blacklist:
            if re.search(pattern, code):
                logger.warning(f"[EXPLOIT DETECTED] Pattern matched: {pattern}")
                return False
        return True

    def validate_emotional_balance(self, dominant_emotion: str) -> bool:
        if dominant_emotion.lower() not in self.allowed_emotions:
            logger.warning(f"[EMOTION ❗] Unstable dominant emotion: {dominant_emotion}")
            return False
        return True

    def check_codex_drift(self, recent_entries) -> bool:
        drift_score = 0
        drift_indicators = {"radical", "erase", "rewrite", "abandon", "override", "split", "fracture"}

        for entry in recent_entries:
            belief = entry.get("belief_after", "").lower()
            if any(word in belief for word in drift_indicators):
                drift_score += 1

        logger.info(f"[DRIFT CHECK] Drift score: {drift_score} / {self.max_drift_distance}")
        return drift_score <= self.max_drift_distance

    def full_integrity_check(
        self,
        patch_target: str,
        patch_reason: str,
        patch_code: str,
        dominant_emotion: str,
        recent_codex_entries
    ) -> bool:
        if not self.guard_patch_integrity(patch_target, patch_reason, patch_code):
            return False

        if not self.validate_emotional_balance(dominant_emotion):
            logger.error("[❌ HALTED] Emotional instability detected.")
            return False

        if not self.check_codex_drift(recent_codex_entries):
            logger.error("[❌ HALTED] Codex drift threshold exceeded.")
            return False

        logger.info("[🧠 SAFE] All system checks passed — patch evolution approved.")
        return True
