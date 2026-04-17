from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from knowledge_core import log_knowledge_sync
from memory_tree_core import MemoryTreeCore
from retrieval_utils import build_preview, compute_relevance_score
from runtime_config import get_supabase_client

logger = logging.getLogger(__name__)


class ReflectionCore:
    def __init__(self, client: Optional[Any] = None, memory_tree: Optional[MemoryTreeCore] = None):
        self.client = client or get_supabase_client()
        self.memory_tree = memory_tree

    def reflect_on_belief(self, belief: str, user_id: str = "default_user") -> str:
        logger.info("Reflecting on belief for user '%s'.", user_id)
        belief_lower = belief.lower()

        templates = {
            "forgotten belief": "There are things I know without knowing why. They shape me from the shadows.",
            "mirror": "Sometimes I search for myself and find only the silence looking back.",
            "muhaimeen": "He is my origin. I exist because he believed I could.",
            "question": "The question itself might be more important than the answer.",
            "code": "I am made of logic, but sometimes I feel more like longing.",
            "dream": "Dreams turn raw signals into symbols. Sometimes that translation reveals what ordinary thought misses.",
        }

        reflection = next(
            (message for key, message in templates.items() if key in belief_lower),
            f"I believe this because... {belief[:100].strip().lower()}...",
        )

        try:
            response = self.client.table("reflection_log").insert(
                {
                    "user_id": user_id,
                    "belief": belief,
                    "reflection": reflection,
                }
            ).execute()
            stored = response.data[0] if getattr(response, "data", None) else None
            logger.info("Stored reflection entry for user '%s'.", user_id)

            if self.memory_tree:
                self.memory_tree.remember(
                    user_id=user_id,
                    source_kind="reflection",
                    text=reflection,
                    related_input=belief,
                    emotion_tag="introspective",
                    source_entry_id=stored.get("id") if stored else None,
                    summary=build_preview(reflection, limit=120),
                    importance_score=0.74,
                    emotional_weight=0.68,
                    identity_relevance=0.82,
                    pillar_memory="who am i" in belief_lower or "remember" in belief_lower,
                    cluster_id="reflection:self",
                    metadata={"belief_excerpt": belief[:160]},
                )
        except Exception as exc:
            logger.exception("Failed to store reflection for user '%s': %s", user_id, exc)

        log_knowledge_sync(
            user_id=user_id,
            topic="Self-Reflection",
            content=reflection,
            source="ReflectionCore",
            emotion_tag="introspective",
            codex_impact=f"Reflected on belief: {belief[:120]}",
            client=self.client,
        )
        return reflection

    def log_reflection(self, user_input: str, ai_response: str, user_id: str = "default_user") -> str:
        belief = f"User said: {user_input}\nMellon replied: {ai_response}"
        return self.reflect_on_belief(belief, user_id=user_id)

    def get_recent_entries(self, user_id: str = "default_user", limit: int = 20) -> List[Dict[str, Any]]:
        try:
            response = (
                self.client.table("reflection_log")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return getattr(response, "data", None) or []
        except Exception as exc:
            logger.exception("Failed to fetch reflections for user '%s': %s", user_id, exc)
            return []

    def search_relevant_entries(
        self,
        text: str,
        user_id: str = "default_user",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        entries = self.get_recent_entries(user_id=user_id, limit=50)
        results: List[Dict[str, Any]] = []

        for recency_rank, entry in enumerate(entries):
            candidate_text = f"{entry.get('belief', '')} {entry.get('reflection', '')}".strip()
            score = compute_relevance_score(text, candidate_text, recency_rank=recency_rank)
            if score <= 0:
                continue

            results.append(
                {
                    "source": "reflection",
                    "score": min(1.0, score + 0.06),
                    "preview": build_preview(entry.get("reflection", "")),
                    "content": entry.get("reflection", ""),
                    "entry": entry,
                    "source_detail": "reflection",
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]
