from __future__ import annotations

import logging
from typing import Optional

from gemini_api import GeminiUnavailableError, call_gemini
from knowledge_core import KnowledgeCore, log_knowledge_sync
from local_ollama_api import LocalLLMUnavailableError, call_local_llm
from openrouter_api import OpenRouterUnavailableError, call_mistral
from runtime_config import get_settings, get_supabase_client

logger = logging.getLogger(__name__)


class DeepSeekAPI:
    def __init__(self, knowledge_core: Optional[KnowledgeCore] = None, client=None):
        self.client = client or get_supabase_client()
        self.knowledge_core = knowledge_core or KnowledgeCore(self.client)

    def query(self, topic: str, user_id: str = "default_user") -> str:
        if not topic or not topic.strip():
            raise ValueError("Topic is required.")

        prompt = (
            f"Explain the topic '{topic}' in clear, detailed terms suitable for an advanced AI "
            "learning system to absorb and internalize long-term."
        )
        system_prompt = (
            "You are a knowledgeable and emotionally aware AI teaching another AI how the world works."
        )

        try:
            settings = get_settings()
            if settings.has_local_llm:
                content = call_local_llm(prompt, system_prompt)
            elif settings.has_gemini:
                content = call_gemini(prompt, system_prompt)
            else:
                content = call_mistral(prompt, system_prompt)
        except (LocalLLMUnavailableError, GeminiUnavailableError, OpenRouterUnavailableError) as exc:
            logger.warning("DeepSeekAPI is using local fallback output: %s", exc)
            content = self._fallback_explanation(topic)
        except Exception as exc:
            logger.exception("DeepSeekAPI failed unexpectedly: %s", exc)
            content = self._fallback_explanation(topic)

        log_knowledge_sync(
            user_id=user_id,
            topic=topic,
            content=content,
            source="DeepSeekAPI",
            emotion_tag="curiosity",
            codex_impact=f"Learned about: {topic}",
            client=self.client,
            memory_tree=self.knowledge_core.memory_tree,
        )
        return content

    def _fallback_explanation(self, topic: str) -> str:
        cleaned_topic = " ".join(topic.strip().split()) or "this topic"
        return (
            f"{cleaned_topic} matters because it connects ideas, systems, and consequences. "
            f"In local fallback mode, Mellon can only provide a lightweight overview: start by "
            f"defining the core terms, identify the main mechanisms, and then map how {cleaned_topic} "
            "affects people, technology, or the natural world."
        )
