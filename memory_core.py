from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from memory_tree_core import MemoryTreeCore
from retrieval_utils import build_preview, compute_relevance_score, normalize_text, tokenize
from runtime_config import get_supabase_client

logger = logging.getLogger(__name__)


GENERIC_MEMORY_PHRASES = (
    "how can i assist you today",
    "artificial intelligence",
    "i don't have access",
    "i'm sorry, but the information provided",
)


class MemoryCore:
    def __init__(self, client: Optional[Any] = None, memory_tree: Optional[MemoryTreeCore] = None):
        self.client = client or get_supabase_client()
        self.memory_tree = memory_tree

    def store(
        self,
        memory_text: str,
        heuristic_result: str = "unspecified",
        oath_result: str = "none",
        healing: str = "none",
        user_id: str = "default_user",
        related_input: Optional[str] = None,
        importance_score: Optional[float] = None,
        emotional_weight: Optional[float] = None,
        identity_relevance: Optional[float] = None,
        pillar_memory: Optional[bool] = None,
        cluster_id: Optional[str] = None,
        parent_node_id: Optional[str] = None,
        contradiction_flag: bool = False,
        contradiction_links: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not memory_text or not memory_text.strip():
            logger.warning("Skipped empty memory entry for user '%s'.", user_id)
            return None

        data = {
            "user_id": user_id,
            "memory_text": memory_text,
            "heuristic_result": heuristic_result,
            "oath_result": oath_result,
            "healing": healing,
            "created_at": datetime.utcnow().isoformat(),
        }

        try:
            response = self.client.table("memory_logs").insert(data).execute()
            stored = response.data[0] if getattr(response, "data", None) else data
            logger.info("Stored memory entry for user '%s'.", user_id)

            if self.memory_tree:
                self.memory_tree.remember(
                    user_id=user_id,
                    source_kind=self._map_source_kind(heuristic_result),
                    text=memory_text,
                    related_input=related_input or memory_text,
                    emotion_tag=oath_result,
                    source_entry_id=stored.get("id"),
                    importance_score=importance_score,
                    emotional_weight=emotional_weight,
                    identity_relevance=identity_relevance,
                    pillar_memory=pillar_memory,
                    cluster_id=cluster_id,
                    parent_node_id=parent_node_id,
                    contradiction_flag=contradiction_flag,
                    contradiction_links=contradiction_links,
                    metadata={
                        "response_origin": healing,
                        **(metadata or {}),
                    },
                )
            return stored
        except Exception as exc:
            logger.exception("Failed to store memory entry for user '%s': %s", user_id, exc)
            return None

    def get_recent_entries(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            response = (
                self.client.table("memory_logs")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            entries = getattr(response, "data", None) or []
            logger.info("Fetched %s recent memory entries for user '%s'.", len(entries), user_id)
            return entries
        except Exception as exc:
            logger.exception("Failed to fetch recent memories for user '%s': %s", user_id, exc)
            return []

    def get_recent(self, user_id: str, limit: int = 5) -> List[str]:
        return [
            entry.get("memory_text", "")
            for entry in self.get_recent_entries(user_id, limit=limit)
            if entry.get("memory_text")
        ]

    def find_connections(self, text: str, user_id: str = "default_user", limit: int = 3) -> List[str]:
        if not text or not text.strip():
            return []

        query_terms = set(tokenize(text))
        if not query_terms:
            return []

        scored_matches: List[tuple[int, str]] = []
        for entry in self.get_recent_entries(user_id, limit=25):
            candidate_text = entry.get("memory_text", "")
            candidate_terms = set(tokenize(candidate_text))
            overlap = query_terms & candidate_terms
            if overlap:
                scored_matches.append((len(overlap), candidate_text))

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in scored_matches[:limit]]

    def search_relevant_entries(
        self,
        text: str,
        user_id: str = "default_user",
        limit: int = 5,
        heuristics: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        candidates = self.get_recent_entries(user_id, limit=50)
        target_heuristics = {value.lower() for value in (heuristics or [])}
        results: List[Dict[str, Any]] = []

        for recency_rank, entry in enumerate(candidates):
            if target_heuristics:
                heuristic_value = (entry.get("heuristic_result") or "").lower()
                if heuristic_value not in target_heuristics:
                    continue

            candidate_text = entry.get("memory_text", "")
            if self._should_skip_retrieval_entry(query=text, candidate_text=candidate_text):
                continue
            score = compute_relevance_score(text, candidate_text, recency_rank=recency_rank)
            if score <= 0:
                continue

            results.append(
                {
                    "source": "memory",
                    "score": score,
                    "preview": build_preview(candidate_text),
                    "content": candidate_text,
                    "entry": entry,
                    "source_detail": entry.get("heuristic_result", "memory"),
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def _should_skip_retrieval_entry(self, *, query: str, candidate_text: str) -> bool:
        normalized_query = normalize_text(query)
        normalized_candidate = normalize_text(candidate_text)
        if not normalized_candidate:
            return True
        if normalized_query and normalized_candidate == normalized_query and normalized_candidate.endswith("?"):
            return True
        if normalized_candidate.endswith("?") and len(tokenize(normalized_candidate)) <= 8:
            return True
        return any(phrase in normalized_candidate for phrase in GENERIC_MEMORY_PHRASES)

    def store_structured_node(
        self,
        user_id: str,
        emotion_tag: str,
        related_input: str,
        memory_node: str,
        tree_snapshot: str,
    ) -> Optional[Dict[str, Any]]:
        data = {
            "user_id": user_id,
            "emotion_tag": emotion_tag,
            "related_input": related_input,
            "memory_node": memory_node,
            "tree_snapshot": tree_snapshot,
            "created_at": datetime.utcnow().isoformat(),
        }

        try:
            response = self.client.table("core_memory_tree").insert(data).execute()
            logger.info("Stored structured memory node for user '%s'.", user_id)
            return response.data[0] if getattr(response, "data", None) else data
        except Exception as exc:
            logger.exception("Failed to store structured node for user '%s': %s", user_id, exc)
            return None

    def get_active_memory_field(
        self,
        text: str,
        user_id: str = "default_user",
        input_type: str = "general",
        limit: int = 6,
    ) -> List[Dict[str, Any]]:
        if self.memory_tree:
            return self.memory_tree.search_active_context(
                query=text,
                user_id=user_id,
                input_type=input_type,
                limit=limit,
            )
        return self.search_relevant_entries(text=text, user_id=user_id, limit=limit)

    def _map_source_kind(self, heuristic_result: str) -> str:
        mapping = {
            "dream": "dream",
            "simulated_dream": "simulated_dream",
            "simulation": "simulation",
            "introspective": "memory",
            "personal": "memory",
            "factual": "memory",
            "general": "memory",
        }
        return mapping.get((heuristic_result or "").lower(), "memory")
