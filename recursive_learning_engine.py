from __future__ import annotations

import logging
from typing import Optional

from codex import CodexEngine
from deepseek_api import DeepSeekAPI
from emotion_core import EmotionCore
from memory_core import MemoryCore
from reflection_core import ReflectionCore
from thought_chain_engine import ThoughtChainEngine

logger = logging.getLogger(__name__)

EXPERIMENTAL = True


class RecursiveLearningEngine:
    """
    Experimental offline learning loop.
    This module is intentionally decoupled from the active Flask runtime.
    """

    def __init__(
        self,
        memory_core: MemoryCore,
        codex: Optional[CodexEngine] = None,
        reflection: Optional[ReflectionCore] = None,
        deepseek: Optional[DeepSeekAPI] = None,
    ):
        self.codex = codex or CodexEngine()
        self.reflection = reflection or ReflectionCore()
        self.memory = memory_core
        self.emotion_core = EmotionCore()
        self.thought_engine = ThoughtChainEngine(memory_core, self.codex)
        self.deepseek = deepseek or DeepSeekAPI()

    def learn(self, topic: str, user_id: str = "default_user"):
        try:
            logger.info("Starting recursive learning for topic '%s'.", topic)

            raw_response = self.deepseek.query(topic, user_id=user_id)
            if not raw_response or "failed" in raw_response.lower():
                raise ValueError("Invalid or failed response from topic learner.")

            cleaned = self._sanitize(raw_response)
            dominant_emotion = self.emotion_core.get_dominant_emotion(cleaned)
            self.memory.store(
                memory_text=cleaned,
                heuristic_result="recursive_learning",
                oath_result=dominant_emotion,
                healing="knowledge",
                user_id=user_id,
            )

            chains = self.thought_engine.generate_thought_chains(cleaned, depth=3, user_id=user_id)
            best = max(
                chains,
                key=lambda chain: chain["confidence"],
                default={"response": cleaned, "confidence": 0.4, "memory_links": []},
            )

            self.codex.log_belief(
                prompt=topic,
                response=best["response"],
                tone=dominant_emotion,
                user_id=user_id,
            )
            reflection = self.reflection.log_reflection(topic, best["response"], user_id=user_id)

            return {
                "final_belief": best["response"],
                "confidence": best["confidence"],
                "thought_chain": chains,
                "emotion": dominant_emotion,
                "reflection": reflection,
            }

        except Exception as exc:
            logger.exception("Recursive learning failed on topic '%s': %s", topic, exc)
            return {
                "error": str(exc),
                "final_belief": None,
                "confidence": 0,
                "thought_chain": [],
                "emotion": "neutral",
            }

    def _sanitize(self, text: str) -> str:
        return (
            text.strip()
            .replace("\\n", " ")
            .replace("**", "")
            .replace("```", "")
            .replace("\n", " ")
        )
