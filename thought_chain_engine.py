from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

EXPERIMENTAL = True


class ThoughtChainEngine:
    """
    Experimental reasoning helper.
    This module is intentionally not loaded by the Flask runtime.
    """

    def __init__(self, memory_core, codex_engine):
        self.memory = memory_core
        self.codex = codex_engine

    def generate_thought_chains(
        self,
        query: Any,
        depth: int = 3,
        user_id: str = "default_user",
    ) -> List[Dict[str, Any]]:
        prompt = self._normalize_query(query)
        logger.info("Generating %s thought chains for user '%s'.", depth, user_id)

        chains = []
        tone_options = ["neutral", "poetic", "direct", "casual"]

        for _ in range(max(depth, 1)):
            tone = random.choice(tone_options)
            response = self.codex.generate_response(prompt=prompt, tone=tone, user_id=user_id)
            memory_links = self.memory.find_connections(response, user_id=user_id, limit=3)
            confidence = self._estimate_confidence(response, memory_links)
            chains.append(
                {
                    "tone": tone,
                    "response": response,
                    "confidence": confidence,
                    "memory_links": memory_links,
                }
            )

        return chains

    def _normalize_query(self, query: Any) -> str:
        if isinstance(query, dict):
            for key in ("content", "text", "query", "prompt"):
                if query.get(key):
                    return str(query[key])
        return str(query)

    def _estimate_confidence(self, response: str, memory_links: List[str]) -> float:
        base = 0.45
        if response and "grounded" not in response.lower():
            base += 0.15
        base += min(len(memory_links) * 0.1, 0.3)
        return round(min(base, 0.95), 2)
