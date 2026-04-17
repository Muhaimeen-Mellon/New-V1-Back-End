from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from memory_tree_core import MemoryTreeCore
from retrieval_utils import build_preview, compute_relevance_score
from runtime_config import get_supabase_client

logger = logging.getLogger(__name__)


def log_knowledge_sync(
    user_id: str = "mellon",
    topic: str = "General",
    content: str = "",
    source: str = "System",
    emotion_tag: str = "neutral",
    codex_impact: Optional[str] = None,
    codex_entry_id: Optional[str] = None,
    client: Optional[Any] = None,
    memory_tree: Optional[MemoryTreeCore] = None,
) -> Optional[Dict[str, Any]]:
    if not content or not content.strip():
        logger.warning("Skipped empty knowledge log for topic '%s'.", topic)
        return None

    data: Dict[str, Any] = {
        "user_id": user_id,
        "topic": topic,
        "content": content,
        "source": source,
        "emotion_tag": emotion_tag,
        "created_at": datetime.utcnow().isoformat(),
    }

    if codex_impact:
        data["codex_impact"] = codex_impact
    if codex_entry_id:
        data["codex_entry_id"] = codex_entry_id

    active_client = client or get_supabase_client()

    try:
        response = active_client.table("knowledge_logs").insert(data).execute()
        stored = response.data[0] if getattr(response, "data", None) else data
        logger.info("Logged knowledge entry for topic '%s'.", topic)

        if memory_tree:
            memory_tree.remember(
                user_id=user_id,
                source_kind="knowledge",
                text=content,
                related_input=topic,
                emotion_tag=emotion_tag,
                source_entry_id=stored.get("id"),
                summary=f"{topic}: {build_preview(content, limit=100)}",
                importance_score=0.72,
                emotional_weight=0.15 if emotion_tag == "neutral" else None,
                identity_relevance=0.18,
                pillar_memory=False,
                cluster_id=f"knowledge:{topic.lower().replace(' ', '-')[:48]}",
                metadata={
                    "source": source,
                    "codex_impact": codex_impact,
                    "codex_entry_id": codex_entry_id,
                },
            )
        return stored
    except Exception as exc:
        logger.exception("Failed to log knowledge for topic '%s': %s", topic, exc)
        return None


async def log_knowledge(
    user_id: str = "mellon",
    topic: str = "General",
    content: str = "",
    source: str = "System",
    emotion_tag: str = "neutral",
    codex_impact: Optional[str] = None,
    codex_entry_id: Optional[str] = None,
    client: Optional[Any] = None,
    memory_tree: Optional[MemoryTreeCore] = None,
) -> Optional[Dict[str, Any]]:
    return log_knowledge_sync(
        user_id=user_id,
        topic=topic,
        content=content,
        source=source,
        emotion_tag=emotion_tag,
        codex_impact=codex_impact,
        codex_entry_id=codex_entry_id,
        client=client,
        memory_tree=memory_tree,
    )


class KnowledgeCore:
    def __init__(
        self,
        supabase_client: Optional[Any] = None,
        memory_tree: Optional[MemoryTreeCore] = None,
    ):
        self.supabase = supabase_client or get_supabase_client()
        self.memory_tree = memory_tree

    def store_knowledge(
        self,
        topic: str,
        content: str,
        emotion_tag: str = "neutral",
        user_id: str = "default_user",
        source: str = "User",
    ) -> Optional[Dict[str, Any]]:
        return log_knowledge_sync(
            user_id=user_id,
            topic=topic,
            content=content,
            source=source,
            emotion_tag=emotion_tag,
            client=self.supabase,
            memory_tree=self.memory_tree,
        )

    def get_recent_knowledge(self, user_id: str = "default_user", limit: int = 20) -> List[Dict[str, Any]]:
        try:
            response = (
                self.supabase.table("knowledge_logs")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return getattr(response, "data", None) or []
        except Exception as exc:
            logger.exception("Failed to fetch knowledge logs for user '%s': %s", user_id, exc)
            return []

    def search_relevant_knowledge(
        self,
        text: str,
        user_id: str = "default_user",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        entries = self.get_recent_knowledge(user_id=user_id, limit=50)
        results: List[Dict[str, Any]] = []

        for recency_rank, entry in enumerate(entries):
            candidate_text = f"{entry.get('topic', '')} {entry.get('content', '')}".strip()
            score = compute_relevance_score(text, candidate_text, recency_rank=recency_rank)
            if score <= 0:
                continue

            results.append(
                {
                    "source": "knowledge",
                    "score": min(1.0, score + 0.08),
                    "preview": build_preview(entry.get("content", "")),
                    "content": entry.get("content", ""),
                    "entry": entry,
                    "source_detail": entry.get("topic", "knowledge"),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]
