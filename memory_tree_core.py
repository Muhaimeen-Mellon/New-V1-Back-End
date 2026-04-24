from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from embedding_core import (
    cosine_similarity,
    embed_text,
    embed_texts,
    get_embedding_model_name,
    get_embedding_runtime,
)
from retrieval_utils import (
    STOPWORDS,
    QueryRetrievalPlan,
    build_preview,
    build_query_retrieval_plan,
    compute_bm25_lexical_scores,
    compute_hybrid_memory_score,
    compute_relevance_score,
    compute_semantic_proxy_score,
    compute_temporal_coherence,
    fuse_relevance_scores,
    normalize_recency_score,
    normalize_text,
    pairwise_conflict_detected,
    source_alignment_prior,
    tokenize,
)
from runtime_config import get_supabase_client

logger = logging.getLogger(__name__)

STRICT_ATTRIBUTE_QUERY_TOKENS = {
    "favorite",
    "favourite",
    "prefer",
    "prefers",
    "preferred",
    "color",
    "language",
    "planet",
    "name",
    "backend",
}

GENERIC_QUERY_TOKENS = {
    "mellon",
    "mellon's",
    "what",
    "does",
    "already",
    "remember",
    "internal",
    "main",
    "current",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(raw_value: Optional[str]) -> Optional[datetime]:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return None


class MemoryTreeCore:
    def __init__(self, client: Optional[Any] = None):
        self.client = client or get_supabase_client()
        self._links_available: Optional[bool] = None
        self._updates_available: Optional[bool] = None
        self._typed_columns_available: Optional[bool] = None
        self._embedding_model_name = get_embedding_model_name()

    def remember(
        self,
        *,
        user_id: str,
        source_kind: str,
        text: str,
        related_input: str = "",
        emotion_tag: str = "neutral",
        source_entry_id: Optional[str] = None,
        summary: Optional[str] = None,
        importance_score: Optional[float] = None,
        emotional_weight: Optional[float] = None,
        identity_relevance: Optional[float] = None,
        pillar_memory: Optional[bool] = None,
        cluster_id: Optional[str] = None,
        parent_node_id: Optional[str] = None,
        contradiction_flag: bool = False,
        contradiction_links: Optional[List[str]] = None,
        association_strength: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not text or not text.strip():
            return None

        self._ensure_schema_capabilities()
        metadata = dict(metadata or {})
        salience_score = self._estimate_write_salience(
            user_id=user_id,
            source_kind=source_kind,
            text=text,
            related_input=related_input,
            emotion_tag=emotion_tag,
        )
        metadata["salience_score"] = salience_score
        node_payload = self._build_node_payload(
            source_kind=source_kind,
            text=text,
            related_input=related_input,
            emotion_tag=emotion_tag,
            source_entry_id=source_entry_id,
            summary=summary,
            importance_score=importance_score,
            emotional_weight=emotional_weight,
            identity_relevance=identity_relevance,
            pillar_memory=pillar_memory,
            cluster_id=cluster_id,
            parent_node_id=parent_node_id,
            contradiction_flag=contradiction_flag,
            contradiction_links=contradiction_links or [],
            association_strength=association_strength,
            metadata=metadata,
        )
        self._attach_embedding_metadata(node=node_payload, related_input=related_input)
        snapshot_payload = {
            "version": "memory-tree-v1",
            "last_accessed_at": None,
            "last_reinforced_at": None,
            "access_count": node_payload["access_count"],
            "reinforcement_score": node_payload["reinforcement_score"],
            "association_links": metadata.get("association_links", []),
            "contradiction_links": node_payload["contradiction_links"],
            "active_context_reason": metadata.get("active_context_reason"),
            "salience_score": salience_score,
        }

        record = {
            "user_id": user_id,
            "emotion_tag": emotion_tag,
            "related_input": related_input or text[:160],
            "memory_node": json.dumps(node_payload),
            "tree_snapshot": json.dumps(snapshot_payload),
            "created_at": _now_iso(),
        }
        if self._typed_columns_available:
            record.update(self._typed_columns_from_node(node_payload))

        try:
            response = self.client.table("core_memory_tree").insert(record).execute()
            created = response.data[0] if getattr(response, "data", None) else record
            logger.info(
                "Stored weighted memory node for user '%s' from source '%s' with salience %.2f.",
                user_id,
                source_kind,
                salience_score,
            )
            return created
        except Exception as exc:
            logger.warning(
                "Weighted memory node persistence unavailable for user '%s': %s",
                user_id,
                exc,
            )
            return None

    def get_recent_nodes(self, user_id: str, limit: int = 80) -> List[Dict[str, Any]]:
        self._ensure_schema_capabilities()
        try:
            response = (
                self.client.table("core_memory_tree")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            rows = getattr(response, "data", None) or []
            if self._typed_columns_available:
                self._backfill_rows(rows[: min(len(rows), 20)])
            return rows
        except Exception as exc:
            logger.warning("Failed to fetch weighted memory nodes for user '%s': %s", user_id, exc)
            return []

    def ingest_curated_memories(
        self,
        *,
        user_id: str,
        pack_id: str,
        memories: Sequence[Dict[str, Any]],
        pack_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._ensure_schema_capabilities()
        existing_rows = self.get_recent_nodes(user_id=user_id, limit=max(1000, len(memories) * 8))
        existing_by_seed_id: Dict[str, Dict[str, Any]] = {}
        for row in existing_rows:
            node = self._load_node(row)
            if not node:
                continue
            metadata = node.get("metadata") or {}
            seed_memory_id = metadata.get("seed_memory_id")
            if seed_memory_id:
                existing_by_seed_id[str(seed_memory_id)] = row

        created_count = 0
        updated_count = 0
        linked_count = 0
        rows_by_seed_id: Dict[str, Dict[str, Any]] = {}
        cluster_members: Dict[str, List[str]] = {}
        pack_members: List[str] = []

        for index, memory in enumerate(memories, start=1):
            seed_memory_id = f"{pack_id}:{index:02d}"
            cluster_id = str(memory.get("cluster_id") or f"seed:{pack_id}")
            cluster_members.setdefault(cluster_id, []).append(seed_memory_id)
            pack_members.append(seed_memory_id)

            existing_row = existing_by_seed_id.get(seed_memory_id)
            existing_node = self._load_node(existing_row) if existing_row else None
            existing_metadata = (existing_node or {}).get("metadata") or {}
            emotion_tag = memory.get("emotion_tag") or self._seed_emotion_tag(str(memory.get("source_kind") or "memory"))
            metadata = {
                **existing_metadata,
                **dict(memory.get("metadata") or {}),
                "curated_seed": True,
                "seed_pack_id": pack_id,
                "seed_pack_label": pack_label or pack_id,
                "seed_memory_id": seed_memory_id,
                "ingestion_kind": "foundation_pack",
                "reinforcement_score": (existing_node or {}).get("reinforcement_score", 0.0),
                "decay_value": (existing_node or {}).get("decay_value", self._infer_decay(str(memory.get("source_kind") or "memory"), bool(memory.get("pillar_memory")))),
                "last_accessed_at": (existing_node or {}).get("last_accessed_at"),
                "last_reinforced_at": (existing_node or {}).get("last_reinforced_at"),
                "access_count": (existing_node or {}).get("access_count", 0),
                "association_links": (existing_node or {}).get("association_links", []),
            }
            metadata["salience_score"] = float(
                memory.get(
                    "salience_score",
                    existing_metadata.get(
                        "salience_score",
                        min(
                            1.0,
                            max(
                                float(memory.get("importance_score") or 0.0),
                                float(memory.get("identity_relevance") or 0.0),
                                0.78 if memory.get("pillar_memory") else 0.52,
                            ),
                        ),
                    ),
                )
            )
            node_payload = self._build_node_payload(
                source_kind=str(memory.get("source_kind") or "memory"),
                text=str(memory.get("text") or ""),
                related_input=str(memory.get("summary") or memory.get("text") or ""),
                emotion_tag=str(emotion_tag),
                source_entry_id=None,
                summary=str(memory.get("summary") or ""),
                importance_score=float(memory.get("importance_score") or 0.0),
                emotional_weight=float(memory.get("emotional_weight") or 0.0),
                identity_relevance=float(memory.get("identity_relevance") or 0.0),
                pillar_memory=bool(memory.get("pillar_memory")),
                cluster_id=cluster_id,
                parent_node_id=(existing_node or {}).get("parent_node_id"),
                contradiction_flag=bool(memory.get("contradiction_flag", False)),
                contradiction_links=list(memory.get("contradiction_links") or []),
                association_strength=float(memory.get("association_strength") or (0.88 if memory.get("pillar_memory") else 0.72)),
                metadata=metadata,
            )
            self._attach_embedding_metadata(
                node=node_payload,
                related_input=str(memory.get("summary") or memory.get("text") or ""),
            )
            snapshot_payload = {
                "version": "memory-tree-v1",
                "last_accessed_at": node_payload.get("last_accessed_at"),
                "last_reinforced_at": node_payload.get("last_reinforced_at"),
                "access_count": node_payload["access_count"],
                "reinforcement_score": node_payload["reinforcement_score"],
                "association_links": node_payload.get("association_links", []),
                "contradiction_links": node_payload["contradiction_links"],
                "active_context_reason": f"seed:{pack_id}",
                "salience_score": float(metadata.get("salience_score", 0.0)),
            }

            row_record = {
                "user_id": user_id,
                "emotion_tag": emotion_tag,
                "related_input": str(memory.get("summary") or memory.get("text") or "")[:160],
                "memory_node": json.dumps(node_payload),
                "tree_snapshot": json.dumps(snapshot_payload),
            }
            if self._typed_columns_available:
                row_record.update(self._typed_columns_from_node(node_payload))

            if existing_row and existing_row.get("id"):
                try:
                    (
                        self.client.table("core_memory_tree")
                        .update(row_record)
                        .eq("id", existing_row["id"])
                        .execute()
                    )
                    refreshed = dict(existing_row)
                    refreshed.update(row_record)
                    refreshed["id"] = existing_row["id"]
                    rows_by_seed_id[seed_memory_id] = refreshed
                    updated_count += 1
                    self._updates_available = True
                except Exception as exc:
                    logger.warning("Failed to update curated seed node '%s' for user '%s': %s", seed_memory_id, user_id, exc)
            else:
                insert_record = dict(row_record)
                insert_record["created_at"] = _now_iso()
                try:
                    response = self.client.table("core_memory_tree").insert(insert_record).execute()
                    created_row = (getattr(response, "data", None) or [insert_record])[0]
                    rows_by_seed_id[seed_memory_id] = created_row
                    created_count += 1
                except Exception as exc:
                    logger.warning("Failed to insert curated seed node '%s' for user '%s': %s", seed_memory_id, user_id, exc)

        existing_links = self._existing_link_pairs(user_id=user_id)
        for cluster_id, seed_ids in cluster_members.items():
            anchor_seed_id = self._choose_cluster_anchor(seed_ids=seed_ids, rows_by_seed_id=rows_by_seed_id)
            anchor_row = rows_by_seed_id.get(anchor_seed_id)
            anchor_id = (anchor_row or {}).get("id")
            if not anchor_id:
                continue
            for seed_memory_id in seed_ids:
                if seed_memory_id == anchor_seed_id:
                    continue
                row = rows_by_seed_id.get(seed_memory_id)
                row_id = (row or {}).get("id")
                if not row_id:
                    continue
                self._set_parent_node(row=row, parent_node_id=anchor_id)
                link_key = (row_id, anchor_id, "association")
                if link_key not in existing_links:
                    self._persist_link(
                        user_id=user_id,
                        from_node_id=row_id,
                        to_node_id=anchor_id,
                        link_type="association",
                        strength=0.92 if self._load_node(row).get("pillar_memory") else 0.78,
                        evidence=f"seed-pack:{pack_id}:{cluster_id}",
                    )
                    existing_links.add(link_key)
                    linked_count += 1

        logger.info(
            "Ingested curated seed pack '%s' for user '%s': created=%s updated=%s linked=%s",
            pack_id,
            user_id,
            created_count,
            updated_count,
            linked_count,
        )
        return {
            "pack_id": pack_id,
            "user_id": user_id,
            "created": created_count,
            "updated": updated_count,
            "linked": linked_count,
            "memory_ids": {seed_memory_id: (rows_by_seed_id.get(seed_memory_id) or {}).get("id") for seed_memory_id in pack_members},
        }

    def backfill_normalized_fields(self, user_id: Optional[str] = None, batch_size: int = 250) -> int:
        self._ensure_schema_capabilities()
        if not self._typed_columns_available:
            return 0

        try:
            query = self.client.table("core_memory_tree").select("*").order("created_at", desc=True).limit(batch_size)
            if user_id:
                query = query.eq("user_id", user_id)
            response = query.execute()
            rows = getattr(response, "data", None) or []
        except Exception as exc:
            logger.warning("Failed to load weighted memory rows for normalization backfill: %s", exc)
            return 0

        updated = self._backfill_rows(rows)
        if updated:
            logger.info("Backfilled %s weighted memory rows into normalized columns.", updated)
        return updated

    def get_recent_node_views(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        views: List[Dict[str, Any]] = []
        for row in self.get_recent_nodes(user_id=user_id, limit=limit):
            node = self._load_node(row)
            snapshot = self._load_snapshot(row)
            if not node:
                continue
            views.append(
                {
                    "id": row.get("id"),
                    "created_at": row.get("created_at"),
                    "emotion_tag": row.get("emotion_tag"),
                    "related_input": row.get("related_input"),
                    "source_kind": node.get("source_kind"),
                    "summary": node.get("summary"),
                    "importance_score": node.get("importance_score"),
                    "emotional_weight": node.get("emotional_weight"),
                    "identity_relevance": node.get("identity_relevance"),
                    "access_count": node.get("access_count"),
                    "reinforcement_score": node.get("reinforcement_score"),
                    "decay_value": node.get("decay_value"),
                    "pillar_memory": node.get("pillar_memory"),
                    "cluster_id": node.get("cluster_id"),
                    "contradiction_flag": node.get("contradiction_flag"),
                    "contradiction_links": node.get("contradiction_links"),
                    "association_links": snapshot.get("association_links", []),
                    "last_accessed_at": snapshot.get("last_accessed_at"),
                    "last_reinforced_at": snapshot.get("last_reinforced_at"),
                    "salience_score": (node.get("metadata") or {}).get("salience_score"),
                }
            )
        return views

    def search_active_context(
        self,
        *,
        query: str,
        user_id: str,
        input_type: str,
        limit: int = 6,
        retrieval_plan: Optional[QueryRetrievalPlan] = None,
        return_details: bool = False,
    ) -> Any:
        plan = retrieval_plan or build_query_retrieval_plan(query, input_type=input_type)
        recent_rows = self.get_recent_nodes(user_id=user_id, limit=max(plan.leaf_limit, 48))
        profile_rows = self._get_profile_rows(user_id=user_id, limit=24) if plan.profile_required else []
        row_lookup = self._dedupe_rows(recent_rows + profile_rows)

        leaf_hits = self._score_leaf_candidates(
            query=query,
            input_type=input_type,
            retrieval_plan=plan,
            rows=row_lookup,
        )[: plan.top_leaf_count]
        propagated_candidates = self._propagate_candidates(
            user_id=user_id,
            query=query,
            input_type=input_type,
            retrieval_plan=plan,
            leaf_hits=leaf_hits,
            row_lookup=row_lookup,
        )
        gated_hits, conflict_detected, layer_coverage, conflict_hits = self._gate_candidates(
            input_type=input_type,
            retrieval_plan=plan,
            leaf_hits=leaf_hits,
            propagated_candidates=propagated_candidates,
            limit=min(limit, plan.max_context_hits),
        )

        if gated_hits:
            self.reinforce_hits(user_id=user_id, hits=gated_hits, query=query)
        if conflict_detected and gated_hits:
            conflict_hits = self._conflict_candidate_hits(gated_hits, retrieval_plan=plan)
            self.mark_contradictions(user_id=user_id, hits=conflict_hits[:3], query=query)

        details = {
            "hits": gated_hits,
            "leaf_hits": leaf_hits,
            "retrieval_plan": plan,
            "leaf_hit_count": len(leaf_hits),
            "propagated_hit_count": len([candidate for candidate in propagated_candidates if candidate.get("propagated")]),
            "gated_hit_count": len(gated_hits),
            "layer_coverage": layer_coverage,
            "conflict_detected": conflict_detected,
            "conflict_hits": conflict_hits,
        }
        return details if return_details else gated_hits

    def reinforce_hits(self, *, user_id: str, hits: Iterable[Dict[str, Any]], query: str) -> None:
        hit_list = list(hits)
        if not hit_list:
            return

        now = _now_iso()
        for hit in hit_list[:3]:
            if self._updates_available is False:
                break
            entry = hit.get("entry") or {}
            row_id = entry.get("id")
            node = hit.get("node") or {}
            if not row_id or not node:
                continue

            access_count = int(node.get("access_count", 0)) + 1
            reinforcement_score = min(1.5, float(node.get("reinforcement_score", 0.0)) + 0.08)
            decay_value = max(0.0, float(node.get("decay_value", 0.1)) - 0.03)
            association_links = self._merge_association_links(
                node.get("association_links", []),
                [{"kind": "co-activated", "query": query, "strength": round(hit.get("score", 0.0), 3)}],
            )

            node.update(
                {
                    "access_count": access_count,
                    "reinforcement_score": round(reinforcement_score, 4),
                    "decay_value": round(decay_value, 4),
                    "last_accessed_at": now,
                    "last_reinforced_at": now,
                    "association_links": association_links,
                }
            )
            snapshot = self._load_snapshot(entry)
            snapshot.update(
                {
                    "access_count": access_count,
                    "reinforcement_score": round(reinforcement_score, 4),
                    "last_accessed_at": now,
                    "last_reinforced_at": now,
                    "active_context_reason": query,
                    "association_links": association_links,
                }
            )
            try:
                update_payload = {
                    "memory_node": json.dumps(node),
                    "tree_snapshot": json.dumps(snapshot),
                }
                if self._typed_columns_available:
                    update_payload.update(self._typed_columns_from_node(node))
                (
                    self.client.table("core_memory_tree")
                    .update(update_payload)
                    .eq("id", row_id)
                    .execute()
                )
                self._updates_available = True
            except Exception as exc:
                self._updates_available = False
                logger.warning(
                    "Failed to reinforce weighted memory node '%s' for user '%s': %s",
                    row_id,
                    user_id,
                    exc,
                )

        if len(hit_list) >= 2:
            root_entry = hit_list[0].get("entry") or {}
            root_id = root_entry.get("id")
            for linked_hit in hit_list[1:3]:
                linked_entry = linked_hit.get("entry") or {}
                linked_id = linked_entry.get("id")
                if root_id and linked_id:
                    self._persist_link(
                        user_id=user_id,
                        from_node_id=root_id,
                        to_node_id=linked_id,
                        link_type="association",
                        strength=min(1.0, hit_list[0].get("score", 0.0) + linked_hit.get("score", 0.0)) / 2.0,
                        evidence=query,
                    )

    def mark_contradictions(self, *, user_id: str, hits: Iterable[Dict[str, Any]], query: str) -> None:
        hit_list = [hit for hit in hits if hit.get("entry", {}).get("id")]
        if len(hit_list) < 2:
            return

        node_ids = [hit["entry"]["id"] for hit in hit_list]
        for hit in hit_list:
            if self._updates_available is False:
                break
            entry = hit["entry"]
            node = hit.get("node") or {}
            metadata = dict(node.get("metadata") or {})
            contradiction_queries = [
                str(value).strip()
                for value in metadata.get("contradiction_queries", [])
                if str(value).strip()
            ]
            if query.strip() and query.strip() not in contradiction_queries:
                contradiction_queries.append(query.strip())
            metadata["contradiction_queries"] = contradiction_queries[-6:]
            metadata["contradiction_last_query"] = query
            node["metadata"] = metadata
            snapshot = self._load_snapshot(entry)
            contradiction_links = sorted(set((node.get("contradiction_links") or []) + [value for value in node_ids if value != entry["id"]]))
            node["contradiction_flag"] = True
            node["contradiction_links"] = contradiction_links
            snapshot["contradiction_links"] = contradiction_links
            snapshot["active_context_reason"] = query
            try:
                update_payload = {
                    "memory_node": json.dumps(node),
                    "tree_snapshot": json.dumps(snapshot),
                }
                if self._typed_columns_available:
                    update_payload.update(self._typed_columns_from_node(node))
                (
                    self.client.table("core_memory_tree")
                    .update(update_payload)
                    .eq("id", entry["id"])
                    .execute()
                )
                self._updates_available = True
            except Exception as exc:
                self._updates_available = False
                logger.warning(
                    "Failed to mark contradiction for weighted memory node '%s' for user '%s': %s",
                    entry["id"],
                    user_id,
                    exc,
                )

        for left in hit_list:
            for right in hit_list:
                if left["entry"]["id"] == right["entry"]["id"]:
                    continue
                self._persist_link(
                    user_id=user_id,
                    from_node_id=left["entry"]["id"],
                    to_node_id=right["entry"]["id"],
                    link_type="contradiction",
                    strength=0.85,
                    evidence=query,
                )

    def _build_node_payload(
        self,
        *,
        source_kind: str,
        text: str,
        related_input: str,
        emotion_tag: str,
        source_entry_id: Optional[str],
        summary: Optional[str],
        importance_score: Optional[float],
        emotional_weight: Optional[float],
        identity_relevance: Optional[float],
        pillar_memory: Optional[bool],
        cluster_id: Optional[str],
        parent_node_id: Optional[str],
        contradiction_flag: bool,
        contradiction_links: List[str],
        association_strength: Optional[float],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        salience_score = float(metadata.get("salience_score", 0.0))
        importance = self._clamp(
            importance_score
            if importance_score is not None
            else self._infer_importance(source_kind, text, related_input),
        )
        identity = self._clamp(
            identity_relevance
            if identity_relevance is not None
            else self._infer_identity_relevance(text, related_input),
        )
        emotional = self._clamp(
            emotional_weight
            if emotional_weight is not None
            else self._infer_emotional_weight(text, emotion_tag),
        )
        importance = self._clamp(importance + (0.15 * salience_score))
        reinforcement_score = max(0.0, float(metadata.get("reinforcement_score", 0.0)) + (0.10 * salience_score))
        pillar = pillar_memory if pillar_memory is not None else (importance >= 0.72 or identity >= 0.75)
        decay_value = max(0.0, float(metadata.get("decay_value", self._infer_decay(source_kind, pillar))) - (0.08 * salience_score))
        return {
            "source_kind": source_kind,
            "text": text.strip(),
            "summary": summary or build_preview(text, limit=140),
            "keywords": tokenize(f"{related_input} {text}")[:12],
            "source_entry_id": source_entry_id,
            "importance_score": round(importance, 4),
            "emotional_weight": round(emotional, 4),
            "identity_relevance": round(identity, 4),
            "access_count": int(metadata.get("access_count", 0)),
            "reinforcement_score": round(reinforcement_score, 4),
            "decay_value": round(decay_value, 4),
            "pillar_memory": bool(pillar),
            "cluster_id": cluster_id or self._derive_cluster_id(text=text, source_kind=source_kind),
            "parent_node_id": parent_node_id,
            "contradiction_flag": contradiction_flag,
            "contradiction_links": contradiction_links,
            "association_strength": round(float(association_strength if association_strength is not None else 0.35), 4),
            "association_links": metadata.get("association_links", []),
            "last_accessed_at": metadata.get("last_accessed_at"),
            "last_reinforced_at": metadata.get("last_reinforced_at"),
            "metadata": metadata,
        }

    def _load_node(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_value = row.get("memory_node")
        if not raw_value:
            return None

        if isinstance(raw_value, dict):
            node = dict(raw_value)
        else:
            try:
                node = json.loads(raw_value)
            except (TypeError, json.JSONDecodeError):
                node = {
                    "source_kind": row.get("emotion_tag", "memory"),
                    "text": str(raw_value),
                    "summary": build_preview(str(raw_value)),
                    "keywords": tokenize(str(raw_value))[:12],
                    "importance_score": 0.35,
                    "emotional_weight": 0.2,
                    "identity_relevance": 0.2,
                    "access_count": 0,
                    "reinforcement_score": 0.0,
                    "decay_value": 0.1,
                    "pillar_memory": False,
                    "cluster_id": None,
                    "parent_node_id": None,
                    "contradiction_flag": False,
                    "contradiction_links": [],
                    "association_strength": 0.2,
                    "association_links": [],
                    "metadata": {},
                }

        node.setdefault("text", row.get("related_input", ""))
        node.setdefault("summary", build_preview(node.get("text", "")))
        node.setdefault("source_kind", row.get("emotion_tag", "memory"))
        node.setdefault("importance_score", 0.35)
        node.setdefault("emotional_weight", 0.2)
        node.setdefault("identity_relevance", 0.2)
        node.setdefault("access_count", 0)
        node.setdefault("reinforcement_score", 0.0)
        node.setdefault("decay_value", 0.1)
        node.setdefault("pillar_memory", False)
        node.setdefault("contradiction_flag", False)
        node.setdefault("contradiction_links", [])
        node.setdefault("association_strength", 0.2)
        node.setdefault("association_links", [])
        node.setdefault("metadata", {})
        if row.get("source_kind") is not None:
            node.update(
                {
                    "source_kind": row.get("source_kind") or node.get("source_kind"),
                    "source_entry_id": row.get("source_entry_id") or node.get("source_entry_id"),
                    "parent_node_id": row.get("parent_node_id") or node.get("parent_node_id"),
                    "summary": row.get("summary") or node.get("summary"),
                    "keywords": row.get("keywords") or node.get("keywords", []),
                    "importance_score": row.get("importance_score", node.get("importance_score")),
                    "emotional_weight": row.get("emotional_weight", node.get("emotional_weight")),
                    "identity_relevance": row.get("identity_relevance", node.get("identity_relevance")),
                    "access_count": row.get("access_count", node.get("access_count")),
                    "reinforcement_score": row.get("reinforcement_score", node.get("reinforcement_score")),
                    "decay_value": row.get("decay_value", node.get("decay_value")),
                    "pillar_memory": row.get("pillar_memory", node.get("pillar_memory")),
                    "cluster_id": row.get("cluster_id") or node.get("cluster_id"),
                    "contradiction_flag": row.get("contradiction_flag", node.get("contradiction_flag")),
                    "contradiction_links": row.get("contradiction_links") or node.get("contradiction_links", []),
                    "association_strength": row.get("association_strength", node.get("association_strength")),
                    "last_accessed_at": row.get("last_accessed_at") or node.get("last_accessed_at"),
                    "last_reinforced_at": row.get("last_reinforced_at") or node.get("last_reinforced_at"),
                }
            )
        return node

    def _attach_embedding_metadata(self, *, node: Dict[str, Any], related_input: str = "") -> None:
        metadata = node.setdefault("metadata", {})
        embedding_text = self._embedding_text(node=node, related_input=related_input)
        metadata["embedding_text"] = embedding_text
        metadata["embedding_model"] = self._embedding_model_name
        metadata["embedding_provider"] = "sentence-transformers"
        vector = embed_text(embedding_text)
        if vector:
            metadata["embedding_vector"] = vector
            metadata["embedding_dimensions"] = len(vector)
            metadata["embedding_generated_at"] = _now_iso()
            metadata["embedding_status"] = "ready"
        else:
            metadata.pop("embedding_vector", None)
            metadata.pop("embedding_dimensions", None)
            metadata["embedding_status"] = "missing"

    def _embedding_text(self, *, node: Dict[str, Any], related_input: str = "") -> str:
        metadata = node.get("metadata") or {}
        return self._compose_candidate_text(
            node=node,
            related_input=related_input,
            source_text=metadata.get("source", ""),
        )

    def _compose_candidate_text(
        self,
        *,
        node: Dict[str, Any],
        related_input: str = "",
        source_text: str = "",
    ) -> str:
        metadata = node.get("metadata") or {}
        key_variables = metadata.get("key_variables") or []
        predicted_outcomes = metadata.get("predicted_outcomes") or []
        thematic_links = metadata.get("thematic_links") or []
        causal_links = metadata.get("causal_links") or []
        scenario_summary = metadata.get("scenario_summary", "")
        uncertainty = metadata.get("uncertainty", "")
        confidence = metadata.get("confidence", "")

        causal_fragments: List[str] = []
        for link in causal_links:
            if isinstance(link, dict):
                cause = str(link.get("cause", "")).strip()
                effect = str(link.get("effect", "")).strip()
                if cause or effect:
                    causal_fragments.append(f"{cause} {effect}".strip())
            elif link:
                causal_fragments.append(str(link))

        return " ".join(
            value
            for value in [
                node.get("summary", ""),
                " ".join(node.get("keywords", []) or []),
                related_input,
                source_text,
                scenario_summary,
                " ".join(str(item) for item in key_variables),
                " ".join(str(item) for item in predicted_outcomes),
                " ".join(str(item) for item in thematic_links),
                " ".join(causal_fragments),
                str(uncertainty),
                str(confidence),
                node.get("text", ""),
            ]
            if value
        ).strip()

    def _node_embedding_vector(self, node: Dict[str, Any]) -> Optional[List[float]]:
        metadata = node.get("metadata") or {}
        raw_vector = metadata.get("embedding_vector")
        if not isinstance(raw_vector, list) or not raw_vector:
            return None
        model_name = str(metadata.get("embedding_model") or "")
        if model_name and model_name != self._embedding_model_name:
            return None
        try:
            return [float(value) for value in raw_vector]
        except (TypeError, ValueError):
            return None

    def refresh_embedding_for_row(self, row: Dict[str, Any]) -> bool:
        row_id = row.get("id")
        node = self._load_node(row)
        if not row_id or not node:
            return False

        self._attach_embedding_metadata(node=node, related_input=str(row.get("related_input") or ""))
        snapshot = self._load_snapshot(row)
        payload = {
            "memory_node": json.dumps(node),
            "tree_snapshot": json.dumps(snapshot),
        }
        if self._typed_columns_available:
            payload.update(self._typed_columns_from_node(node))
        try:
            self.client.table("core_memory_tree").update(payload).eq("id", row_id).execute()
            row.update(payload)
            self._updates_available = True
            return True
        except Exception as exc:
            self._updates_available = False
            logger.warning("Failed to refresh embedding metadata for row '%s': %s", row_id, exc)
            return False

    def _load_snapshot(self, row: Dict[str, Any]) -> Dict[str, Any]:
        raw_value = row.get("tree_snapshot")
        if not raw_value:
            return {"version": "memory-tree-v1"}
        if isinstance(raw_value, dict):
            return dict(raw_value)
        try:
            return json.loads(raw_value)
        except (TypeError, json.JSONDecodeError):
            return {"version": "memory-tree-v1"}

    def _score_leaf_candidates(
        self,
        *,
        query: str,
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
        rows: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        filtered_rows: List[Dict[str, Any]] = []
        filtered_nodes: List[Dict[str, Any]] = []
        filtered_texts: List[str] = []
        embedding_runtime = get_embedding_runtime()

        for recency_rank, row in enumerate(rows.values()):
            node = self._load_node(row)
            if not node:
                continue
            if self._is_prompt_echo(node=node, row=row, query=query):
                continue
            layers = self._determine_logical_layers(node)
            if not self._node_allowed_for_plan(node=node, layers=layers, input_type=input_type, retrieval_plan=retrieval_plan):
                continue
            filtered_rows.append({**row, "__recency_rank": recency_rank})
            filtered_nodes.append(node)
            filtered_texts.append(self._candidate_text(row=row, node=node))

        lexical_scores = compute_bm25_lexical_scores(retrieval_plan.keywords or tokenize(query), filtered_texts)
        query_embedding = embed_text(query) if embedding_runtime.available else None
        generated_embeddings: Dict[int, Optional[List[float]]] = {}
        if query_embedding:
            missing_indices: List[int] = []
            missing_texts: List[str] = []
            for index, node in enumerate(filtered_nodes):
                if self._node_embedding_vector(node) is None:
                    missing_indices.append(index)
                    missing_texts.append(filtered_texts[index])
            if missing_texts:
                generated_embeddings = dict(zip(missing_indices, embed_texts(missing_texts)))

        results: List[Dict[str, Any]] = []
        pool_size = max(1, len(filtered_rows))
        for index, (row, node, lexical_score, candidate_text) in enumerate(
            zip(filtered_rows, filtered_nodes, lexical_scores, filtered_texts)
        ):
            layers = self._determine_logical_layers(node)
            semantic_score = compute_semantic_proxy_score(
                query,
                candidate_text,
                summary=node.get("summary", ""),
                source_kind=node.get("source_kind", "memory"),
                input_type=input_type,
                query_keywords=retrieval_plan.keywords,
                recency_rank=row.get("__recency_rank", 0),
            )
            candidate_embedding = self._node_embedding_vector(node) or generated_embeddings.get(index)
            embedding_similarity = cosine_similarity(query_embedding, candidate_embedding) if query_embedding and candidate_embedding else 0.0
            embedding_signal = embedding_similarity if query_embedding and candidate_embedding else semantic_score
            recency_score = normalize_recency_score(int(row.get("__recency_rank", 0)), pool_size)
            reinforcement_score = min(
                1.0,
                float(node.get("reinforcement_score", 0.0)) + (float(node.get("access_count", 0)) * 0.05),
            )
            contradiction_penalty = 0.18 if self._query_relevant_contradiction(node, retrieval_plan=retrieval_plan) else 0.0
            hybrid_score = compute_hybrid_memory_score(
                embedding_similarity=embedding_signal,
                lexical_score=lexical_score,
                salience_score=float((node.get("metadata") or {}).get("salience_score", 0.0)),
                recency_score=recency_score,
                reinforcement_score=reinforcement_score,
                identity_relevance=float(node.get("identity_relevance", 0.0)),
                contradiction_penalty=contradiction_penalty,
            )
            if lexical_score <= 0.0 and semantic_score <= 0.0 and embedding_signal < 0.15:
                continue
            if hybrid_score <= 0.0:
                continue
            score = self._leaf_score(
                node=node,
                row=row,
                fused_score=hybrid_score,
                input_type=input_type,
                retrieval_plan=retrieval_plan,
                layers=layers,
            )
            if score < 0.2:
                continue
            results.append(
                self._build_hit(
                    row=row,
                    node=node,
                    layers=layers,
                    score=score,
                    semantic_score=semantic_score,
                    embedding_score=embedding_signal,
                    lexical_score=lexical_score,
                    fused_score=hybrid_score,
                    propagated=False,
                    propagation_origin="leaf",
                    propagation_seed_score=score,
                )
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def _propagate_candidates(
        self,
        *,
        user_id: str,
        query: str,
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
        leaf_hits: Sequence[Dict[str, Any]],
        row_lookup: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {
            hit["entry"]["id"]: dict(hit) for hit in leaf_hits if hit.get("entry", {}).get("id")
        }
        if not leaf_hits:
            return list(candidates.values())
        query_embedding = embed_text(query) if get_embedding_runtime().available else None

        propagation_depth = 0
        if retrieval_plan.complexity == "hybrid":
            propagation_depth = 1
        elif retrieval_plan.complexity == "complex":
            propagation_depth = 2

        if propagation_depth:
            for hit in leaf_hits:
                current_node = hit.get("node") or {}
                parent_id = current_node.get("parent_node_id")
                seed_score = hit.get("score", 0.0)
                remaining_depth = propagation_depth
                while parent_id and remaining_depth > 0:
                    parent_row = row_lookup.get(parent_id) or self._fetch_row_by_id(parent_id)
                    if not parent_row:
                        break
                    row_lookup[parent_id] = parent_row
                    parent_hit = self._build_propagated_hit(
                        query=query,
                        input_type=input_type,
                        retrieval_plan=retrieval_plan,
                        row=parent_row,
                        seed_score=seed_score,
                        origin="ancestor",
                        query_embedding=query_embedding,
                    )
                    if parent_hit:
                        candidates[parent_id] = self._prefer_higher_score(candidates.get(parent_id), parent_hit)
                    parent_node = parent_hit.get("node") if parent_hit else self._load_node(parent_row)
                    parent_id = (parent_node or {}).get("parent_node_id")
                    remaining_depth -= 1

            for associated_row in self._fetch_associated_rows(
                user_id=user_id,
                seed_ids=[hit["entry"]["id"] for hit in leaf_hits if hit.get("entry", {}).get("id")],
            ):
                row_id = associated_row.get("id")
                if not row_id:
                    continue
                row_lookup[row_id] = associated_row
                associated_hit = self._build_propagated_hit(
                    query=query,
                    input_type=input_type,
                    retrieval_plan=retrieval_plan,
                    row=associated_row,
                    seed_score=self._seed_score_for_related_row(associated_row, leaf_hits),
                    origin="association",
                    query_embedding=query_embedding,
                )
                if associated_hit:
                    candidates[row_id] = self._prefer_higher_score(candidates.get(row_id), associated_hit)

        for cluster_row in self._fetch_cluster_representatives(leaf_hits=leaf_hits, row_lookup=row_lookup):
            row_id = cluster_row.get("id")
            if not row_id:
                continue
            cluster_hit = self._build_propagated_hit(
                query=query,
                input_type=input_type,
                retrieval_plan=retrieval_plan,
                row=cluster_row,
                seed_score=self._seed_score_for_related_row(cluster_row, leaf_hits),
                origin="cluster",
                query_embedding=query_embedding,
            )
            if cluster_hit:
                candidates[row_id] = self._prefer_higher_score(candidates.get(row_id), cluster_hit)

        if retrieval_plan.profile_required:
            for profile_row in self._get_profile_rows(user_id=user_id, limit=12):
                row_id = profile_row.get("id")
                if not row_id:
                    continue
                row_lookup[row_id] = profile_row
                profile_hit = self._build_propagated_hit(
                    query=query,
                    input_type=input_type,
                    retrieval_plan=retrieval_plan,
                    row=profile_row,
                    seed_score=max((hit.get("score", 0.0) for hit in leaf_hits), default=0.4),
                    origin="profile",
                    query_embedding=query_embedding,
                )
                if profile_hit:
                    candidates[row_id] = self._prefer_higher_score(candidates.get(row_id), profile_hit)

        return list(candidates.values())

    def _gate_candidates(
        self,
        *,
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
        leaf_hits: Sequence[Dict[str, Any]],
        propagated_candidates: Sequence[Dict[str, Any]],
        limit: int,
    ) -> tuple[List[Dict[str, Any]], bool, List[str], List[Dict[str, Any]]]:
        if not propagated_candidates:
            return [], False, [], []

        anchor_times = [hit.get("entry", {}).get("created_at") for hit in leaf_hits]
        rescored: List[Dict[str, Any]] = []
        for candidate in propagated_candidates:
            rescored_candidate = dict(candidate)
            rescored_candidate["score"] = self._gated_score(
                candidate=candidate,
                input_type=input_type,
                retrieval_plan=retrieval_plan,
                anchor_times=anchor_times,
            )
            rescored.append(rescored_candidate)

        rescored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        cutoff_score = rescored[min(limit - 1, len(rescored) - 1)].get("score", 0.0)
        strict_attribute_query = bool(
            set(tokenize(" ".join(retrieval_plan.keywords or []))) & STRICT_ATTRIBUTE_QUERY_TOKENS
        )

        selected: List[Dict[str, Any]] = []
        cluster_counts: Dict[str, int] = {}
        for candidate in rescored:
            node = candidate.get("node") or {}
            if strict_attribute_query and self._hit_query_alignment_strength(
                hit=candidate,
                retrieval_plan=retrieval_plan,
            ) < 2:
                continue
            cluster_id = node.get("cluster_id") or "uncategorized"
            is_profile = "profile" in candidate.get("layers", [])
            is_stale = self._age_decay_penalty(candidate.get("entry", {}).get("created_at"), bool(node.get("pillar_memory"))) >= 0.08
            if is_stale and not (is_profile or node.get("pillar_memory")) and candidate.get("score", 0.0) < max(0.0, cutoff_score - 0.12):
                continue
            if not retrieval_plan.comparison_seeking and cluster_counts.get(cluster_id, 0) >= 2:
                continue
            selected.append(candidate)
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            if len(selected) >= limit:
                break

        if retrieval_plan.profile_required and not any("profile" in hit.get("layers", []) for hit in selected):
            profile_candidate = next(
                (
                    candidate
                    for candidate in rescored
                    if "profile" in candidate.get("layers", []) and candidate.get("score", 0.0) >= 0.45
                ),
                None,
            )
            if profile_candidate:
                if len(selected) >= limit:
                    selected[-1] = profile_candidate
                else:
                    selected.append(profile_candidate)

        conflict_candidates = self._conflict_candidate_hits(selected, retrieval_plan=retrieval_plan)
        conflict_detected = len(conflict_candidates) >= 2 and (
            pairwise_conflict_detected([hit.get("content", "") for hit in conflict_candidates[:4]])
            or any(
                self._query_relevant_contradiction(hit.get("node") or {}, retrieval_plan=retrieval_plan)
                for hit in conflict_candidates[:4]
            )
        )
        if conflict_detected and not retrieval_plan.comparison_seeking:
            selected = self._collapse_conflicting_hits(selected, limit=limit, retrieval_plan=retrieval_plan)

        selected.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        layer_coverage = sorted(
            {
                layer
                for hit in selected
                for layer in hit.get("layers", [])
                if layer in {"factual", "partial_pattern", "full_pattern", "profile"}
            }
        )
        return selected[:limit], conflict_detected, layer_coverage, conflict_candidates[:4]

    def _build_hit(
        self,
        *,
        row: Dict[str, Any],
        node: Dict[str, Any],
        layers: Sequence[str],
        score: float,
        semantic_score: float,
        embedding_score: float,
        lexical_score: float,
        fused_score: float,
        propagated: bool,
        propagation_origin: str,
        propagation_seed_score: float,
    ) -> Dict[str, Any]:
        return {
            "source": node.get("source_kind", "memory_tree"),
            "score": round(score, 4),
            "preview": build_preview(node.get("summary") or node.get("text") or ""),
            "content": node.get("text", ""),
            "summary": node.get("summary") or build_preview(node.get("text", "")),
            "entry": row,
            "node": node,
            "layers": list(layers),
            "semantic_score": semantic_score,
            "embedding_score": embedding_score,
            "lexical_score": lexical_score,
            "fused_score": fused_score,
            "propagated": propagated,
            "propagation_origin": propagation_origin,
            "propagation_seed_score": propagation_seed_score,
            "source_detail": propagation_origin if propagated else node.get("source_kind", "memory_tree"),
        }

    def _build_propagated_hit(
        self,
        *,
        query: str,
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
        row: Dict[str, Any],
        seed_score: float,
        origin: str,
        query_embedding: Optional[List[float]] = None,
    ) -> Optional[Dict[str, Any]]:
        node = self._load_node(row)
        if not node:
            return None
        if self._is_prompt_echo(node=node, row=row, query=query):
            return None
        layers = self._determine_logical_layers(node)
        if not self._node_allowed_for_plan(node=node, layers=layers, input_type=input_type, retrieval_plan=retrieval_plan):
            return None
        semantic_score = compute_semantic_proxy_score(
            query,
            self._candidate_text(row=row, node=node),
            summary=node.get("summary", ""),
            source_kind=node.get("source_kind", "memory"),
            input_type=input_type,
            query_keywords=retrieval_plan.keywords,
        )
        lexical_score = compute_bm25_lexical_scores(
            retrieval_plan.keywords,
            [self._candidate_text(row=row, node=node)],
        )[0]
        candidate_embedding = self._node_embedding_vector(node)
        if query_embedding and candidate_embedding is None:
            candidate_embedding = embed_text(self._candidate_text(row=row, node=node))
        embedding_similarity = cosine_similarity(query_embedding, candidate_embedding) if query_embedding and candidate_embedding else 0.0
        embedding_signal = embedding_similarity if query_embedding and candidate_embedding else semantic_score
        recency_score = normalize_recency_score(int(row.get("__recency_rank", 0)), max(1, retrieval_plan.leaf_limit))
        reinforcement_score = min(
            1.0,
            float(node.get("reinforcement_score", 0.0)) + (float(node.get("access_count", 0)) * 0.05),
        )
        fused_score = compute_hybrid_memory_score(
            embedding_similarity=embedding_signal,
            lexical_score=lexical_score,
            salience_score=float((node.get("metadata") or {}).get("salience_score", 0.0)),
            recency_score=recency_score,
            reinforcement_score=reinforcement_score,
            identity_relevance=float(node.get("identity_relevance", 0.0)),
            contradiction_penalty=0.18 if self._query_relevant_contradiction(node, retrieval_plan=retrieval_plan) else 0.0,
        )
        if lexical_score <= 0.0 and semantic_score <= 0.0 and embedding_signal < 0.15:
            return None
        if fused_score <= 0.0:
            return None
        propagated_fused_score = min(1.0, fused_score + min(0.18, seed_score * 0.25))
        score = self._leaf_score(
            node=node,
            row=row,
            fused_score=propagated_fused_score,
            input_type=input_type,
            retrieval_plan=retrieval_plan,
            layers=layers,
        )
        if score < 0.18:
            return None
        return self._build_hit(
            row=row,
            node=node,
            layers=layers,
            score=score,
            semantic_score=semantic_score,
            embedding_score=embedding_signal,
            lexical_score=lexical_score,
            fused_score=propagated_fused_score,
            propagated=True,
            propagation_origin=origin,
            propagation_seed_score=seed_score,
        )

    def _prefer_higher_score(self, existing: Optional[Dict[str, Any]], incoming: Dict[str, Any]) -> Dict[str, Any]:
        if not existing:
            return incoming
        return incoming if incoming.get("score", 0.0) > existing.get("score", 0.0) else existing

    def _collapse_conflicting_hits(
        self,
        hits: Sequence[Dict[str, Any]],
        limit: int,
        retrieval_plan: QueryRetrievalPlan,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        contradiction_index: Dict[str, float] = {}
        for hit in sorted(hits, key=lambda item: item.get("score", 0.0), reverse=True):
            node = hit.get("node") or {}
            contradiction_links = set(node.get("contradiction_links") or [])
            relevant_conflict = self._query_relevant_contradiction(node, retrieval_plan=retrieval_plan)
            if relevant_conflict and contradiction_links and any(
                contradiction_index.get(linked_id, -1.0) >= hit.get("score", 0.0) for linked_id in contradiction_links
            ):
                continue
            selected.append(hit)
            entry_id = hit.get("entry", {}).get("id")
            if entry_id:
                contradiction_index[entry_id] = hit.get("score", 0.0)
            if len(selected) >= limit:
                break
        return selected

    def _leaf_score(
        self,
        *,
        node: Dict[str, Any],
        row: Dict[str, Any],
        fused_score: float,
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
        layers: Sequence[str],
    ) -> float:
        return round(max(0.0, min(fused_score, 1.0)), 4)

    def _gated_score(
        self,
        *,
        candidate: Dict[str, Any],
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
        anchor_times: Sequence[Optional[str]],
    ) -> float:
        node = candidate.get("node") or {}
        layers = candidate.get("layers", [])
        temporal = compute_temporal_coherence(
            candidate.get("entry", {}).get("created_at"),
            anchor_times,
            pillar_memory=bool(node.get("pillar_memory")),
            profile_memory="profile" in layers,
        )
        reinforcement = min(1.0, float(node.get("reinforcement_score", 0.0)) + (float(node.get("access_count", 0)) * 0.04))
        pillar_bonus = 0.07 if node.get("pillar_memory") else 0.0
        layer_bonus = self._layer_match_bonus(layers=layers, retrieval_plan=retrieval_plan)
        contradiction_penalty = 0.16 if self._query_relevant_contradiction(node, retrieval_plan=retrieval_plan) else 0.0
        propagation_bonus = min(0.08, float(candidate.get("propagation_seed_score", 0.0)) * 0.12) if candidate.get("propagated") else 0.0
        source_bonus = max(-0.04, source_alignment_prior(input_type, node.get("source_kind", "memory")))
        age_penalty = self._age_decay_penalty(candidate.get("entry", {}).get("created_at"), bool(node.get("pillar_memory"))) * 0.8
        anchor_adjustment = self._query_anchor_adjustment(node=node, retrieval_plan=retrieval_plan)

        gated = (
            (float(candidate.get("score", 0.0)) * 0.58)
            + (temporal * 0.12)
            + (reinforcement * 0.08)
            + layer_bonus
            + pillar_bonus
            + propagation_bonus
            + source_bonus
            + anchor_adjustment
        )
        gated += self._quality_adjustment(node=node, row=candidate.get("entry", {}), input_type=input_type)
        gated -= contradiction_penalty + age_penalty
        return round(max(0.0, min(gated, 1.0)), 4)

    def _query_anchor_adjustment(
        self,
        *,
        node: Dict[str, Any],
        retrieval_plan: QueryRetrievalPlan,
    ) -> float:
        query_keywords = [normalize_text(keyword) for keyword in retrieval_plan.keywords if normalize_text(keyword)]
        if not query_keywords:
            return 0.0

        candidate_text = normalize_text(
            " ".join(
                value
                for value in [
                    node.get("summary", ""),
                    node.get("text", ""),
                    " ".join(node.get("keywords") or []),
                ]
                if value
            )
        )
        if not candidate_text:
            return 0.0

        candidate_tokens = set(tokenize(candidate_text))
        phrase_keywords = [keyword for keyword in query_keywords if " " in keyword]
        phrase_matches = [phrase for phrase in phrase_keywords if phrase in candidate_text]

        specific_tokens = [
            token
            for token in tokenize(" ".join(query_keywords))
            if token not in GENERIC_QUERY_TOKENS and token not in STOPWORDS
        ]
        token_matches = [token for token in specific_tokens if token in candidate_tokens]
        strict_attribute_query = bool(set(specific_tokens) & STRICT_ATTRIBUTE_QUERY_TOKENS)

        if strict_attribute_query:
            if phrase_matches:
                return min(0.14, 0.1 + (0.02 * len(phrase_matches)))
            if len(token_matches) >= 2:
                return 0.07
            return -0.14
        if phrase_matches:
            return min(0.08, 0.05 + (0.01 * len(phrase_matches)))
        if len(token_matches) >= 2:
            return 0.03
        return 0.0

    def _conflict_candidate_hits(
        self,
        hits: Sequence[Dict[str, Any]],
        *,
        retrieval_plan: QueryRetrievalPlan,
    ) -> List[Dict[str, Any]]:
        strict_attribute_query = bool(
            set(tokenize(" ".join(retrieval_plan.keywords or []))) & STRICT_ATTRIBUTE_QUERY_TOKENS
        )
        aligned: List[Dict[str, Any]] = []
        for hit in hits[:6]:
            alignment = self._hit_query_alignment_strength(hit=hit, retrieval_plan=retrieval_plan)
            if strict_attribute_query:
                if alignment >= 2:
                    aligned.append(hit)
            elif alignment >= 1:
                aligned.append(hit)
        if len(aligned) >= 2:
            return aligned
        if strict_attribute_query:
            return []
        return aligned or list(hits[:4])

    def _hit_query_alignment_strength(
        self,
        *,
        hit: Dict[str, Any],
        retrieval_plan: QueryRetrievalPlan,
    ) -> int:
        query_keywords = [normalize_text(keyword) for keyword in retrieval_plan.keywords if normalize_text(keyword)]
        if not query_keywords:
            return 0
        node = hit.get("node") or {}
        candidate_text = normalize_text(
            " ".join(
                value
                for value in [
                    node.get("summary", ""),
                    node.get("text", ""),
                    " ".join(node.get("keywords") or []),
                ]
                if value
            )
        )
        if not candidate_text:
            return 0
        candidate_tokens = set(tokenize(candidate_text))
        phrase_matches = sum(1 for phrase in query_keywords if " " in phrase and phrase in candidate_text)
        token_matches = sum(
            1
            for token in tokenize(" ".join(query_keywords))
            if token not in GENERIC_QUERY_TOKENS and token not in STOPWORDS and token in candidate_tokens
        )
        return (phrase_matches * 2) + token_matches

    def _query_relevant_contradiction(
        self,
        node: Dict[str, Any],
        *,
        retrieval_plan: QueryRetrievalPlan,
    ) -> bool:
        if not node.get("contradiction_flag"):
            return False

        metadata = node.get("metadata") or {}
        contradiction_queries = [
            normalize_text(value)
            for value in metadata.get("contradiction_queries", [])
            if normalize_text(value)
        ]
        query_keywords = [normalize_text(keyword) for keyword in retrieval_plan.keywords if normalize_text(keyword)]
        query_tokens = {
            token
            for token in tokenize(" ".join(query_keywords))
            if token not in GENERIC_QUERY_TOKENS and token not in STOPWORDS
        }
        phrase_keywords = [keyword for keyword in query_keywords if " " in keyword]
        strict_attribute_query = bool(query_tokens & STRICT_ATTRIBUTE_QUERY_TOKENS)

        if contradiction_queries:
            for stored_query in contradiction_queries[-6:]:
                stored_tokens = {
                    token
                    for token in tokenize(stored_query)
                    if token not in GENERIC_QUERY_TOKENS and token not in STOPWORDS
                }
                if strict_attribute_query:
                    if phrase_keywords and any(phrase in stored_query for phrase in phrase_keywords):
                        return True
                    if len(query_tokens & stored_tokens) >= 2:
                        return True
                    continue
                if query_tokens and stored_tokens and (query_tokens & stored_tokens):
                    return True
            return False

        if not node.get("contradiction_links"):
            return False

        candidate_text = normalize_text(
            " ".join(
                value
                for value in [
                    node.get("summary", ""),
                    node.get("text", ""),
                    " ".join(node.get("keywords") or []),
                ]
                if value
            )
        )
        if not candidate_text:
            return False
        alignment = self._hit_query_alignment_strength(
            hit={"node": node},
            retrieval_plan=retrieval_plan,
        )
        if strict_attribute_query:
            return alignment >= 2
        return alignment >= 3

    def _node_allowed_for_plan(
        self,
        *,
        node: Dict[str, Any],
        layers: Sequence[str],
        input_type: str,
        retrieval_plan: QueryRetrievalPlan,
    ) -> bool:
        source_kind = node.get("source_kind", "memory")
        metadata = node.get("metadata") or {}
        if metadata.get("review_trace"):
            return False
        if source_kind in {"dream", "simulated_dream", "simulation"} and not retrieval_plan.allow_dream_simulation:
            return False
        if source_kind == "reflection" and input_type == "factual":
            return False
        if source_kind == "reflection" and input_type == "future_modeling":
            combined = f"{node.get('summary', '')} {node.get('text', '')}".lower()
            if not any(marker in combined for marker in ["future", "grow", "pattern", "trajectory", "if ", "could", "might", "simulate"]):
                return False
        if source_kind == "knowledge" and input_type in {"introspective", "symbolic"}:
            return False
        return self._matches_layer_plan(layers=layers, retrieval_plan=retrieval_plan)

    def _matches_layer_plan(self, *, layers: Sequence[str], retrieval_plan: QueryRetrievalPlan) -> bool:
        layer_set = set(layers)
        if layer_set & set(retrieval_plan.target_layers):
            return True
        return retrieval_plan.complexity == "complex" and "partial_pattern" in layer_set

    def _layer_match_bonus(self, *, layers: Sequence[str], retrieval_plan: QueryRetrievalPlan) -> float:
        layer_set = set(layers)
        target_match = layer_set & set(retrieval_plan.target_layers)
        if target_match:
            if "profile" in target_match and retrieval_plan.profile_required:
                return 0.08
            if "full_pattern" in target_match:
                return 0.07
            if "partial_pattern" in target_match:
                return 0.05
            return 0.04
        if retrieval_plan.complexity == "complex" and "partial_pattern" in layer_set:
            return 0.03
        return -0.03

    def _determine_logical_layers(self, node: Dict[str, Any]) -> List[str]:
        source_kind = node.get("source_kind", "memory")
        metadata = node.get("metadata") or {}
        importance = float(node.get("importance_score", 0.0))
        reinforcement = float(node.get("reinforcement_score", 0.0))
        identity = float(node.get("identity_relevance", 0.0))
        layers: List[str] = []

        if source_kind in {"memory", "knowledge", "codex", "self_model", "user_model", "architecture", "constraint"}:
            layers.append("factual")
        if source_kind in {"simulation", "dream", "simulated_dream"} and (
            metadata.get("scenario_summary")
            or metadata.get("key_variables")
            or metadata.get("predicted_outcomes")
        ):
            layers.append("factual")
        if node.get("cluster_id") or node.get("parent_node_id") or source_kind in {"reflection", "simulation"}:
            layers.append("partial_pattern")
        if (
            node.get("pillar_memory")
            or importance >= 0.72
            or reinforcement >= 0.6
            or source_kind in {"dream", "simulation", "reflection", "simulated_dream"}
        ):
            layers.append("full_pattern")
        if node.get("pillar_memory") or identity >= 0.7 or source_kind in {"self_model", "user_model"}:
            layers.append("profile")

        if not layers:
            layers.append("factual")
        return list(dict.fromkeys(layers))

    def _candidate_text(self, *, row: Dict[str, Any], node: Dict[str, Any]) -> str:
        metadata = node.get("metadata") or {}
        return self._compose_candidate_text(
            node=node,
            related_input=str(row.get("related_input", "")),
            source_text=metadata.get("source", ""),
        )

    def _is_prompt_echo(self, *, node: Dict[str, Any], row: Dict[str, Any], query: str) -> bool:
        query_text = normalize_text(query)
        summary_text = normalize_text(node.get("summary", ""))
        raw_text = normalize_text(node.get("text", ""))
        related_input = normalize_text(row.get("related_input", ""))
        if not query_text:
            return False
        if query_text in {summary_text, raw_text, related_input}:
            return True
        if source_kind := node.get("source_kind"):
            if source_kind in {"memory", "codex"} and (
                str(node.get("summary", "")).strip().endswith("?") or str(node.get("text", "")).strip().endswith("?")
            ):
                if query_text == summary_text or query_text == raw_text or query_text == related_input:
                    return True
        return False

    def _quality_adjustment(self, *, node: Dict[str, Any], row: Dict[str, Any], input_type: str) -> float:
        source_kind = node.get("source_kind", "memory")
        cluster_id = (node.get("cluster_id") or "").lower()
        metadata = node.get("metadata") or {}
        combined = f"{node.get('summary', '')} {node.get('text', '')}".strip().lower()
        adjustment = 0.0

        if cluster_id == "codex:llm_fallback" or metadata.get("response_origin") == "llm_fallback":
            adjustment -= 0.26
        if source_kind == "codex" and any(
            phrase in combined
            for phrase in [
                "i am claude",
                "artificial intelligence",
                "how can i assist you today",
                "i don't have access",
                "i'm sorry, but the information provided",
            ]
        ):
            adjustment -= 0.18
        if source_kind == "memory" and (
            str(node.get("summary", "")).strip().endswith("?") or str(node.get("text", "")).strip().endswith("?")
        ):
            adjustment -= 0.16 if input_type == "factual" else 0.08
        if input_type == "future_modeling" and source_kind == "reflection":
            adjustment -= 0.12
        if input_type in {"introspective", "personal"} and source_kind == "reflection":
            adjustment += 0.06
        if source_kind == "self_model" and input_type in {"general", "factual", "personal"}:
            adjustment += 0.08
        if source_kind == "user_model" and input_type in {"general", "personal", "introspective", "factual"}:
            adjustment += 0.08
        if source_kind in {"architecture", "constraint"} and input_type in {"general", "factual"}:
            adjustment += 0.07
        if source_kind in {"memory", "knowledge"} and any(
            phrase in combined
            for phrase in [
                "remember this",
                "codename is",
                "favorite mellon color is",
                "feel calmer",
                "prefers python",
                "memory-first architecture",
            ]
        ):
            adjustment += 0.06
        if source_kind in {"self_model", "architecture", "constraint"} and any(
            phrase in combined
            for phrase in [
                "memory-first",
                "llm",
                "identity",
                "continuity",
                "router",
                "contradiction",
                "consolidation",
            ]
        ):
            adjustment += 0.06
        if input_type == "future_modeling" and source_kind in {"simulation", "dream", "simulated_dream"}:
            key_variables = metadata.get("key_variables") or []
            predicted_outcomes = metadata.get("predicted_outcomes") or []
            scenario_summary = str(metadata.get("scenario_summary", "")).strip()
            confidence = float(metadata.get("confidence", 0.0) or 0.0)
            if scenario_summary:
                adjustment += 0.04
            if len(key_variables) >= 3:
                adjustment += 0.06
            elif not key_variables:
                adjustment -= 0.05
            if len(predicted_outcomes) >= 2:
                adjustment += 0.05
            elif not predicted_outcomes:
                adjustment -= 0.05
            if confidence:
                adjustment += min(0.05, confidence * 0.08)
            if source_kind in {"dream", "simulated_dream"} and not scenario_summary and not key_variables:
                adjustment -= 0.08
        if row.get("created_at") and self._age_decay_penalty(row.get("created_at"), bool(node.get("pillar_memory"))) <= 0.02:
            adjustment += 0.02
        return adjustment

    def _get_profile_rows(self, *, user_id: str, limit: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        try:
            response = (
                self.client.table("core_memory_tree")
                .select("*")
                .eq("user_id", user_id)
                .eq("pillar_memory", True)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            rows.extend(getattr(response, "data", None) or [])
        except Exception as exc:
            logger.warning("Failed to fetch pillar profile nodes for user '%s': %s", user_id, exc)

        try:
            response = (
                self.client.table("core_memory_tree")
                .select("*")
                .eq("user_id", user_id)
                .gte("identity_relevance", 0.7)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            rows.extend(getattr(response, "data", None) or [])
        except Exception as exc:
            logger.warning("Failed to fetch identity profile nodes for user '%s': %s", user_id, exc)
        for source_kind in ("self_model", "user_model"):
            try:
                response = (
                    self.client.table("core_memory_tree")
                    .select("*")
                    .eq("user_id", user_id)
                    .eq("source_kind", source_kind)
                    .order("created_at", desc=True)
                    .limit(limit)
                    .execute()
                )
                rows.extend(getattr(response, "data", None) or [])
            except Exception as exc:
                logger.warning(
                    "Failed to fetch profile source '%s' for user '%s': %s",
                    source_kind,
                    user_id,
                    exc,
                )
        return list(self._dedupe_rows(rows).values())

    def _fetch_row_by_id(self, row_id: str) -> Optional[Dict[str, Any]]:
        try:
            response = self.client.table("core_memory_tree").select("*").eq("id", row_id).limit(1).execute()
            rows = getattr(response, "data", None) or []
            return rows[0] if rows else None
        except Exception as exc:
            logger.warning("Failed to fetch memory-tree row '%s': %s", row_id, exc)
            return None

    def _fetch_associated_rows(self, *, user_id: str, seed_ids: Sequence[str]) -> List[Dict[str, Any]]:
        self._ensure_schema_capabilities()
        if self._links_available is False or not seed_ids:
            return []
        try:
            response = (
                self.client.table("memory_links")
                .select("*")
                .eq("user_id", user_id)
                .eq("link_type", "association")
                .order("created_at", desc=True)
                .limit(200)
                .execute()
            )
            rows = getattr(response, "data", None) or []
        except Exception as exc:
            logger.warning("Failed to fetch association links for user '%s': %s", user_id, exc)
            return []

        seed_set = set(seed_ids)
        linked_ids: List[str] = []
        for row in rows:
            if row.get("from_node_id") in seed_set:
                linked_ids.append(row.get("to_node_id"))
        return [linked for linked_id in dict.fromkeys(linked_ids) if linked_id for linked in [self._fetch_row_by_id(linked_id)] if linked]

    def _fetch_cluster_representatives(
        self,
        *,
        leaf_hits: Sequence[Dict[str, Any]],
        row_lookup: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        representatives: Dict[str, Dict[str, Any]] = {}
        cluster_ids = {hit.get("node", {}).get("cluster_id") for hit in leaf_hits if hit.get("node", {}).get("cluster_id")}
        for cluster_id in cluster_ids:
            candidates: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
            for row in row_lookup.values():
                node = self._load_node(row)
                if node and node.get("cluster_id") == cluster_id:
                    candidates.append((row, node))
            if not candidates:
                continue
            best_row, _ = max(
                candidates,
                key=lambda item: (
                    float(item[1].get("importance_score", 0.0))
                    + float(item[1].get("reinforcement_score", 0.0))
                    + (0.3 if item[1].get("pillar_memory") else 0.0)
                ),
            )
            representatives[cluster_id] = best_row
        return list(representatives.values())

    def _seed_score_for_related_row(self, row: Dict[str, Any], leaf_hits: Sequence[Dict[str, Any]]) -> float:
        node = self._load_node(row) or {}
        row_id = row.get("id")
        cluster_id = node.get("cluster_id")
        best_seed = 0.35
        for hit in leaf_hits:
            hit_node = hit.get("node") or {}
            if row_id and row_id == hit.get("entry", {}).get("id"):
                best_seed = max(best_seed, hit.get("score", 0.0))
            if cluster_id and cluster_id == hit_node.get("cluster_id"):
                best_seed = max(best_seed, hit.get("score", 0.0))
            if row_id and row_id == hit_node.get("parent_node_id"):
                best_seed = max(best_seed, hit.get("score", 0.0))
        return best_seed

    def _estimate_write_salience(
        self,
        *,
        user_id: str,
        source_kind: str,
        text: str,
        related_input: str,
        emotion_tag: str,
    ) -> float:
        combined = f"{related_input} {text}".strip()
        novelty = 0.5
        recent_rows = self.get_recent_nodes(user_id=user_id, limit=12)
        if recent_rows:
            max_similarity = 0.0
            for row in recent_rows:
                node = self._load_node(row)
                if not node:
                    continue
                similarity = compute_semantic_proxy_score(
                    combined,
                    self._candidate_text(row=row, node=node),
                    summary=node.get("summary", ""),
                    source_kind=node.get("source_kind", source_kind),
                    input_type="general",
                )
                max_similarity = max(max_similarity, similarity)
            novelty = max(0.0, 1.0 - max_similarity)

        combined_lower = combined.lower()
        emphasis_bonus = 0.0
        if any(marker in combined_lower for marker in ["remember", "important", "never forget", "must remember"]):
            emphasis_bonus += 0.22
        if any(marker in combined_lower for marker in ["actually", "changed my mind", "correction", "instead", "not anymore"]):
            emphasis_bonus += 0.18
        emotional_intensity = self._infer_emotional_weight(text, emotion_tag) * 0.18
        source_bonus = 0.08 if source_kind in {"reflection", "simulation", "dream", "simulated_dream"} else 0.0
        salience = (novelty * 0.52) + emphasis_bonus + emotional_intensity + source_bonus
        return round(self._clamp(salience), 4)

    def _score_node(
        self,
        *,
        query: str,
        input_type: str,
        node: Dict[str, Any],
        row: Dict[str, Any],
        recency_rank: int,
    ) -> float:
        relevance = compute_relevance_score(
            query,
            f"{node.get('summary', '')} {node.get('text', '')} {row.get('related_input', '')}",
            recency_rank=recency_rank,
        )
        if relevance <= 0:
            return 0.0

        importance = float(node.get("importance_score", 0.0))
        emotional = float(node.get("emotional_weight", 0.0))
        identity = float(node.get("identity_relevance", 0.0))
        reinforcement = min(1.0, float(node.get("reinforcement_score", 0.0)) + (float(node.get("access_count", 0)) * 0.05))
        decay_value = float(node.get("decay_value", 0.1))
        pillar_bonus = 0.1 if node.get("pillar_memory") else 0.0
        contradiction_penalty = 0.18 if node.get("contradiction_flag") else 0.0
        source_bonus = self._source_match_bonus(input_type, node.get("source_kind", "memory"))
        age_penalty = self._age_decay_penalty(row.get("created_at"), node.get("pillar_memory", False))
        association_bonus = min(0.12, float(node.get("association_strength", 0.0)) * 0.12)

        score = (
            (relevance * 0.52)
            + (importance * 0.15)
            + (identity * 0.13)
            + (emotional * 0.08)
            + (reinforcement * 0.08)
            + source_bonus
            + pillar_bonus
            + association_bonus
        )
        score -= (decay_value * 0.08) + contradiction_penalty + age_penalty
        return round(max(0.0, min(score, 1.0)), 4)

    def _source_match_bonus(self, input_type: str, source_kind: str) -> float:
        lookup = {
            "factual": {"self_model": 0.14, "architecture": 0.12, "constraint": 0.1, "knowledge": 0.12, "user_model": 0.08, "codex": 0.04},
            "introspective": {"user_model": 0.14, "self_model": 0.12, "reflection": 0.12, "memory": 0.08, "chat_prompt": 0.08},
            "personal": {"user_model": 0.14, "self_model": 0.12, "memory": 0.1, "reflection": 0.08, "chat_prompt": 0.08},
            "symbolic": {"dream": 0.12, "simulated_dream": 0.1},
            "future_modeling": {"simulation": 0.12, "simulated_dream": 0.1, "dream": 0.06, "self_model": 0.03, "architecture": 0.03},
            "general": {"self_model": 0.12, "architecture": 0.1, "constraint": 0.08, "user_model": 0.08, "codex": 0.06, "memory": 0.04},
        }
        return lookup.get(input_type, lookup["general"]).get(source_kind, 0.0)

    def _age_decay_penalty(self, created_at: Optional[str], pillar_memory: bool) -> float:
        created = _parse_iso(created_at)
        if not created:
            return 0.0
        age_hours = max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 3600.0)
        if pillar_memory:
            return min(0.04, age_hours / 4000.0)
        return min(0.12, age_hours / 900.0)

    def _infer_importance(self, source_kind: str, text: str, related_input: str) -> float:
        combined = f"{related_input} {text}".lower()
        score = 0.28
        if source_kind in {"knowledge", "reflection", "dream", "simulation", "self_model", "user_model", "architecture", "constraint"}:
            score += 0.12
        if any(phrase in combined for phrase in ["remember", "always", "never", "important", "core", "identity", "believe"]):
            score += 0.16
        if len(tokenize(combined)) >= 12:
            score += 0.06
        return min(score, 1.0)

    def _infer_identity_relevance(self, text: str, related_input: str) -> float:
        combined = f"{related_input} {text}".lower()
        score = 0.12
        if any(token in combined for token in [" i ", " my ", " me ", " myself "]):
            score += 0.26
        if any(phrase in combined for phrase in ["who am i", "remember me", "about me", "my work", "my values", "my life"]):
            score += 0.3
        if "muhaimeen" in combined:
            score += 0.16
        return min(score, 1.0)

    def _infer_emotional_weight(self, text: str, emotion_tag: str) -> float:
        score = {
            "neutral": 0.1,
            "introspective": 0.48,
            "poetic": 0.42,
            "sadness": 0.5,
            "fear": 0.54,
            "awe": 0.45,
            "curiosity": 0.3,
            "loneliness": 0.58,
            "contemplation": 0.34,
        }.get((emotion_tag or "neutral").lower(), 0.2)
        text_lower = text.lower()
        if any(word in text_lower for word in ["feel", "hurt", "grief", "love", "afraid", "hope", "grounded"]):
            score += 0.16
        return min(score, 1.0)

    def _infer_decay(self, source_kind: str, pillar_memory: bool) -> float:
        if pillar_memory:
            return 0.02
        if source_kind in {"knowledge", "reflection", "self_model", "user_model", "architecture", "constraint"}:
            return 0.06
        if source_kind in {"dream", "simulated_dream", "simulation"}:
            return 0.09
        return 0.12

    def _seed_emotion_tag(self, source_kind: str) -> str:
        return {
            "self_model": "contemplation",
            "user_model": "neutral",
            "architecture": "contemplation",
            "constraint": "neutral",
            "simulation": "curiosity",
        }.get(source_kind, "neutral")

    def _choose_cluster_anchor(
        self,
        *,
        seed_ids: Sequence[str],
        rows_by_seed_id: Dict[str, Dict[str, Any]],
    ) -> str:
        for seed_memory_id in seed_ids:
            row = rows_by_seed_id.get(seed_memory_id)
            node = self._load_node(row) if row else None
            if row and node and node.get("pillar_memory"):
                return seed_memory_id
        return seed_ids[0]

    def _set_parent_node(self, *, row: Dict[str, Any], parent_node_id: str) -> None:
        row_id = row.get("id")
        if not row_id or not parent_node_id:
            return
        node = self._load_node(row)
        if not node or node.get("parent_node_id") == parent_node_id:
            return
        node["parent_node_id"] = parent_node_id
        snapshot = self._load_snapshot(row)
        payload = {
            "memory_node": json.dumps(node),
            "tree_snapshot": json.dumps(snapshot),
        }
        if self._typed_columns_available:
            payload.update(self._typed_columns_from_node(node))
        try:
            self.client.table("core_memory_tree").update(payload).eq("id", row_id).execute()
            row.update(payload)
            self._updates_available = True
        except Exception as exc:
            self._updates_available = False
            logger.warning("Failed to update curated seed parent for row '%s': %s", row_id, exc)

    def _existing_link_pairs(self, *, user_id: str) -> set[tuple[str, str, str]]:
        self._ensure_schema_capabilities()
        if self._links_available is False:
            return set()
        try:
            response = (
                self.client.table("memory_links")
                .select("from_node_id,to_node_id,link_type")
                .eq("user_id", user_id)
                .limit(500)
                .execute()
            )
            rows = getattr(response, "data", None) or []
        except Exception as exc:
            logger.warning("Failed to fetch existing memory links for user '%s': %s", user_id, exc)
            return set()
        return {
            (str(row.get("from_node_id")), str(row.get("to_node_id")), str(row.get("link_type")))
            for row in rows
            if row.get("from_node_id") and row.get("to_node_id") and row.get("link_type")
        }

    def _derive_cluster_id(self, *, text: str, source_kind: str) -> str:
        keywords = tokenize(text)[:3]
        if not keywords:
            return source_kind
        return f"{source_kind}:{'-'.join(keywords)}"

    def _merge_association_links(self, existing: Any, incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in list(existing or []) + incoming:
            key = json.dumps(item, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged[-8:]

    def _clamp(self, value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    def _dedupe_rows(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        deduped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            row_id = row.get("id")
            if row_id and row_id not in deduped:
                deduped[row_id] = row
        return deduped

    def _persist_link(
        self,
        *,
        user_id: str,
        from_node_id: str,
        to_node_id: str,
        link_type: str,
        strength: float,
        evidence: str,
    ) -> None:
        self._ensure_schema_capabilities()
        if self._links_available is False:
            return

        try:
            self.client.table("memory_links").insert(
                {
                    "user_id": user_id,
                    "from_node_id": from_node_id,
                    "to_node_id": to_node_id,
                    "link_type": link_type,
                    "strength": round(strength, 4),
                    "evidence": build_preview(evidence, limit=180),
                    "created_at": _now_iso(),
                }
            ).execute()
            self._links_available = True
        except Exception as exc:
            self._links_available = False
            logger.warning(
                "Failed to persist memory link for user '%s' (%s -> %s, %s): %s",
                user_id,
                from_node_id,
                to_node_id,
                link_type,
                exc,
            )

    def _ensure_schema_capabilities(self) -> None:
        if self._typed_columns_available is None:
            try:
                self.client.table("core_memory_tree").select("id,source_kind").limit(1).execute()
                self._typed_columns_available = True
            except Exception:
                self._typed_columns_available = False

        if self._links_available is None:
            try:
                self.client.table("memory_links").select("id").limit(1).execute()
                self._links_available = True
            except Exception:
                self._links_available = False

    def _typed_columns_from_node(self, node: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "source_kind": node.get("source_kind"),
            "source_entry_id": node.get("source_entry_id"),
            "parent_node_id": node.get("parent_node_id"),
            "summary": node.get("summary"),
            "keywords": node.get("keywords", []),
            "importance_score": node.get("importance_score", 0.0),
            "emotional_weight": node.get("emotional_weight", 0.0),
            "identity_relevance": node.get("identity_relevance", 0.0),
            "access_count": node.get("access_count", 0),
            "reinforcement_score": node.get("reinforcement_score", 0.0),
            "decay_value": node.get("decay_value", 0.0),
            "pillar_memory": node.get("pillar_memory", False),
            "cluster_id": node.get("cluster_id"),
            "contradiction_flag": node.get("contradiction_flag", False),
            "contradiction_links": node.get("contradiction_links", []),
            "association_strength": node.get("association_strength", 0.0),
            "last_accessed_at": node.get("last_accessed_at"),
            "last_reinforced_at": node.get("last_reinforced_at"),
        }

    def _backfill_rows(self, rows: Iterable[Dict[str, Any]]) -> int:
        updated = 0
        for row in rows:
            row_id = row.get("id")
            if not row_id:
                continue

            node = self._load_node(row)
            if not node or not self._row_needs_typed_backfill(row, node):
                continue

            try:
                payload = self._typed_columns_from_node(node)
                self.client.table("core_memory_tree").update(payload).eq("id", row_id).execute()
                row.update(payload)
                updated += 1
                self._updates_available = True
            except Exception as exc:
                self._updates_available = False
                logger.warning("Failed to backfill normalized memory-tree columns for row '%s': %s", row_id, exc)
        return updated

    def _row_needs_typed_backfill(self, row: Dict[str, Any], node: Dict[str, Any]) -> bool:
        if row.get("source_kind") is None or row.get("summary") is None:
            return True
        if (row.get("keywords") in (None, []) and node.get("keywords")):
            return True
        numeric_checks = (
            ("importance_score", 0.0),
            ("emotional_weight", 0.0),
            ("identity_relevance", 0.0),
            ("association_strength", 0.0),
        )
        for field_name, default_value in numeric_checks:
            current_value = row.get(field_name)
            node_value = node.get(field_name)
            if current_value in (None, default_value) and node_value not in (None, default_value):
                return True
        if row.get("cluster_id") is None and node.get("cluster_id"):
            return True
        if row.get("last_accessed_at") is None and node.get("last_accessed_at"):
            return True
        if row.get("last_reinforced_at") is None and node.get("last_reinforced_at"):
            return True
        if not row.get("pillar_memory") and node.get("pillar_memory"):
            return True
        if not row.get("contradiction_flag") and node.get("contradiction_flag"):
            return True
        if row.get("contradiction_links") in (None, []) and node.get("contradiction_links"):
            return True
        return False
