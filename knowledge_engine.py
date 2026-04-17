from __future__ import annotations

import logging
from typing import Optional

from deepseek_api import DeepSeekAPI
from knowledge_core import KnowledgeCore

logger = logging.getLogger(__name__)

EXPERIMENTAL = True


class KnowledgeEngine:
    """
    Experimental offline topic learner.
    This helper is not part of the active Flask request path.
    """

    def __init__(self, supabase_client=None, codex_engine=None):
        self.knowledge_core = KnowledgeCore(supabase_client)
        self.codex_engine = codex_engine
        self.topic_provider = DeepSeekAPI(knowledge_core=self.knowledge_core, client=supabase_client)

    def learn_topic(self, topic: str, user_id: str = "default_user"):
        summary = self.topic_provider.query(topic, user_id=user_id)
        if not summary:
            return {"error": "Failed to retrieve summary for topic", "topic": topic}

        emotion_tag = self.detect_emotion(summary)
        entry = self.knowledge_core.store_knowledge(
            topic=topic,
            content=summary,
            emotion_tag=emotion_tag,
            user_id=user_id,
            source="KnowledgeEngine",
        )
        if not entry:
            return {"error": "Failed to store summary", "topic": topic}

        logger.info("KnowledgeEngine learned topic '%s'.", topic)
        return {
            "status": "learned",
            "topic": topic,
            "emotion_tag": emotion_tag,
            "summary": summary[:300] + "..." if len(summary) > 300 else summary,
            "entry": entry,
        }

    def detect_emotion(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ["war", "suffering", "destruction"]):
            return "fear"
        if any(word in text_lower for word in ["hope", "resilience", "rebirth", "progress"]):
            return "hope"
        if any(word in text_lower for word in ["loss", "grief", "mourning"]):
            return "sadness"
        if any(word in text_lower for word in ["breakthrough", "revolution", "innovation"]):
            return "excitement"
        return "neutral"
