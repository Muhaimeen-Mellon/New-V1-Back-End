from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from flask import Flask, jsonify, render_template, request

try:
    from flask_cors import CORS
except Exception:  # pragma: no cover - optional dependency
    def CORS(app, *args, **kwargs):  # type: ignore[override]
        logging.getLogger(__name__).warning(
            "flask_cors is not installed; continuing without CORS middleware."
        )
        return app

from codex import CodexEngine
from core_router import CoreRouter
from deepseek_api import DeepSeekAPI
from dream_core import DreamCore
from knowledge_core import KnowledgeCore, log_knowledge_sync
from memory_core import MemoryCore
from memory_review_engine import MemoryReviewEngine
from memory_tree_core import MemoryTreeCore
from module_registry import ACTIVE_RUNTIME_MODULES, EXPERIMENTAL_MODULES
from reflection_core import ReflectionCore
from runtime_config import (
    configure_logging,
    get_runtime_snapshot,
    get_settings,
    get_supabase_client,
)
from simulation_core import SimulationCore
from trait_graph_engine import TraitGraphEngine
from trait_influence_engine import TraitInfluenceEngine
from tone_core import ToneCore

configure_logging()
logger = logging.getLogger(__name__)

try:
    from learning_firewall import enforce_firewall
except Exception as exc:  # pragma: no cover - safety fallback
    logger.warning("Learning firewall unavailable; continuing in permissive mode. %s", exc)

    def enforce_firewall(user_input: str, user_id: str = "default_user") -> dict:
        del user_input, user_id
        return {"status": "safe", "message": "Firewall unavailable; request allowed."}


@dataclass
class MellonRuntime:
    supabase: Any
    memory_tree: MemoryTreeCore
    trait_graph_engine: TraitGraphEngine
    trait_influence_engine: TraitInfluenceEngine
    memory_core: MemoryCore
    codex_engine: CodexEngine
    core_router: CoreRouter
    dream_core: DreamCore
    reflection_core: ReflectionCore
    knowledge_core: KnowledgeCore
    deepseek_api: DeepSeekAPI
    tone_core: ToneCore
    simulation_core: SimulationCore


def build_runtime() -> MellonRuntime:
    supabase = get_supabase_client()
    memory_tree = MemoryTreeCore(supabase)
    trait_graph_engine = TraitGraphEngine(memory_tree=memory_tree)
    trait_influence_engine = TraitInfluenceEngine()
    memory_tree.trait_graph_engine = trait_graph_engine
    memory_core = MemoryCore(supabase, memory_tree=memory_tree)
    reflection_core = ReflectionCore(supabase, memory_tree=memory_tree)
    codex_engine = CodexEngine(supabase, memory_tree=memory_tree)
    knowledge_core = KnowledgeCore(supabase, memory_tree=memory_tree)
    memory_review_engine = MemoryReviewEngine(memory_tree=memory_tree)
    core_router = CoreRouter(
        memory_tree=memory_tree,
        memory_core=memory_core,
        codex_engine=codex_engine,
        reflection_core=reflection_core,
        knowledge_core=knowledge_core,
        memory_review_engine=memory_review_engine,
    )
    dream_core = DreamCore(
        codex=codex_engine,
        memory_core=memory_core,
        reflection_core=reflection_core,
    )
    deepseek_api = DeepSeekAPI(knowledge_core=knowledge_core, client=supabase)
    tone_core = ToneCore()
    simulation_core = SimulationCore(
        dream_core=dream_core,
        reflection_core=reflection_core,
        codex=codex_engine,
        memory_core=memory_core,
    )

    runtime = MellonRuntime(
        supabase=supabase,
        memory_tree=memory_tree,
        trait_graph_engine=trait_graph_engine,
        trait_influence_engine=trait_influence_engine,
        memory_core=memory_core,
        codex_engine=codex_engine,
        core_router=core_router,
        dream_core=dream_core,
        reflection_core=reflection_core,
        knowledge_core=knowledge_core,
        deepseek_api=deepseek_api,
        tone_core=tone_core,
        simulation_core=simulation_core,
    )
    logger.info("Mellon runtime booted with snapshot: %s", get_runtime_snapshot())
    return runtime


def create_app() -> Flask:
    settings = get_settings()
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": settings.cors_origins}})
    runtime = build_runtime()
    app.config["mellon_runtime"] = runtime

    @app.route("/")
    def home():
        return jsonify(
            {
                "status": "Mellon backend is running",
                "runtime": get_runtime_snapshot(),
            }
        ), 200

    @app.route("/app")
    def mellon_app():
        return render_template("mellon_ui.html")

    @app.route("/status")
    def status():
        snapshot = get_runtime_snapshot()
        return jsonify(
            {
                "codex": True,
                "dreams": True,
                "memory": True,
                "reflection": True,
                "simulation": True,
                "storage_mode": snapshot["storage_mode"],
                "storage": snapshot["storage"],
                "model_mode": snapshot["model_mode"],
                "default_model": snapshot.get("default_model"),
                "active_modules": list(ACTIVE_RUNTIME_MODULES.keys()),
                "experimental_modules": list(EXPERIMENTAL_MODULES.keys()),
            }
        )

    @app.route("/hello", methods=["GET"])
    def say_hello():
        return jsonify({"message": "Hello from Mellon backend. All good here."}), 200

    @app.route("/chat", methods=["POST"])
    def chat():
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 400

        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        user_id = data.get("user_id", "default_user")

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        fw_result = enforce_firewall(user_message, user_id=user_id)
        logger.info("Firewall result for user '%s': %s", user_id, fw_result)
        if fw_result.get("status") == "blocked":
            return jsonify({"error": "Message blocked by firewall", "details": fw_result}), 403

        detected_tone = runtime.tone_core.detect_tone(user_message)
        routing_result = runtime.core_router.route_chat(user_message, user_id=user_id)
        response_payload = runtime.codex_engine.respond_with_memory_strategy(
            prompt=user_message,
            memory_bundle=routing_result.as_dict(),
            tone=detected_tone,
            user_id=user_id,
        )
        final_response = response_payload["response"]

        runtime.core_router.record_review_trace(
            user_id=user_id,
            prompt=user_message,
            routing_result=routing_result,
            response_payload=response_payload,
        )

        runtime.memory_core.store(
            memory_text=user_message,
            heuristic_result=routing_result.input_type,
            oath_result=detected_tone,
            healing=response_payload["response_origin"],
            user_id=user_id,
            related_input=user_message,
            importance_score=0.66 if routing_result.input_type in {"introspective", "personal"} else 0.38,
            emotional_weight=0.55 if routing_result.input_type in {"introspective", "symbolic"} else None,
            identity_relevance=0.8 if routing_result.input_type in {"introspective", "personal"} else 0.28,
            pillar_memory=routing_result.input_type in {"introspective", "personal"} and routing_result.confidence >= 0.7,
            cluster_id=f"{routing_result.input_type}:active",
            contradiction_flag=routing_result.conflict_detected,
            contradiction_links=[
                hit.get("entry", {}).get("id")
                for hit in routing_result.hits[:3]
                if hit.get("entry", {}).get("id")
            ] if routing_result.conflict_detected else None,
                metadata={
                    "modules_consulted": routing_result.modules_consulted,
                    "sources_used": routing_result.sources_used,
                    "response_origin": response_payload["response_origin"],
                    "confidence": routing_result.confidence,
                    "review_state": routing_result.review_state,
                    "review_reason": routing_result.review_reason,
                    "memory_support_strength": routing_result.memory_support_strength,
                    "memory_conflict_detected": routing_result.memory_conflict_detected,
                    "memory_gap_detected": routing_result.memory_gap_detected,
                    "recalled_vs_inferred": routing_result.recalled_vs_inferred,
                    "reflection_bank_used": routing_result.reflection_bank_used,
                    "reflection_ids_used": routing_result.reflection_ids_used,
                    "query_complexity": routing_result.query_complexity,
                    "target_layers": routing_result.target_layers,
                    "layer_coverage": routing_result.layer_coverage,
                },
            )

        if routing_result.input_type in {"introspective", "personal"}:
            runtime.reflection_core.log_reflection(user_message, final_response, user_id=user_id)

        logger.info(
            "Chat response metadata for user '%s': origin=%s review=%s reason=%s llm_attempted=%s llm_called=%s model=%s modules=%s sources=%s fallback=%s confidence=%.2f",
            user_id,
            response_payload["response_origin"],
            routing_result.review_state,
            routing_result.review_reason,
            response_payload["llm_attempted"],
            response_payload["llm_called"],
            response_payload.get("model_used") or "none",
            ",".join(routing_result.modules_consulted) or "none",
            ",".join(routing_result.sources_used) or "none",
            routing_result.fallback_reason or "none",
            routing_result.confidence,
        )
        return jsonify(
            {
                "response": final_response,
                "tone": detected_tone,
                "routing": {
                    "input_type": routing_result.input_type,
                    "sources_used": routing_result.sources_used,
                    "modules_consulted": routing_result.modules_consulted,
                    "memory_sufficient": routing_result.sufficient_memory,
                    "confidence": routing_result.confidence,
                    "review_state": routing_result.review_state,
                    "review_reason": routing_result.review_reason,
                    "memory_support_strength": routing_result.memory_support_strength,
                    "memory_conflict_detected": routing_result.memory_conflict_detected,
                    "memory_gap_detected": routing_result.memory_gap_detected,
                    "recalled_vs_inferred": routing_result.recalled_vs_inferred,
                    "reflection_bank_used": routing_result.reflection_bank_used,
                    "reflection_ids_used": routing_result.reflection_ids_used,
                    "query_complexity": routing_result.query_complexity,
                    "query_keywords": routing_result.query_keywords,
                    "target_layers": routing_result.target_layers,
                    "layer_coverage": routing_result.layer_coverage,
                    "leaf_hit_count": routing_result.leaf_hit_count,
                    "propagated_hit_count": routing_result.propagated_hit_count,
                    "gated_hit_count": routing_result.gated_hit_count,
                    "response_origin": response_payload["response_origin"],
                    "llm_called": response_payload["llm_called"],
                    "llm_attempted": response_payload["llm_attempted"],
                    "model_used": response_payload.get("model_used"),
                    "llm_error": response_payload.get("llm_error"),
                    "fallback_reason": routing_result.fallback_reason,
                    "conflict_detected": routing_result.conflict_detected,
                    "curated_memory_hits": response_payload.get("curated_hits", []),
                    "memory_hits": [
                        {
                            "source": hit.get("source"),
                            "source_detail": hit.get("source_detail"),
                            "score": hit.get("score"),
                            "preview": hit.get("preview"),
                        }
                        for hit in routing_result.hits
                    ],
                },
            }
        )

    @app.route("/dream", methods=["POST"])
    def trigger_dream():
        data = request.get_json(silent=True) or {}
        seed = data.get("seed", "user_triggered")
        tag = data.get("tag", "user")
        user_id = data.get("user_id", "default_user")
        result = runtime.dream_core.seed_dream(seed, tag=tag, user_id=user_id)
        return jsonify(result)

    @app.route("/simulate", methods=["POST"])
    def simulate():
        data = request.get_json(silent=True) or {}
        user_id = data.get("user_id", "default_user")
        result = runtime.simulation_core.simulate_scenario(user_id=user_id)
        return jsonify(result)

    @app.route("/codex", methods=["GET"])
    def get_codex_entries():
        try:
            query = runtime.supabase.table("codex_entries").select("*")
            user_id = request.args.get("user_id")
            if user_id:
                query = query.eq("user_id", user_id)
            result = query.order("created_at", desc=True).limit(20).execute()
            return jsonify(result.data)
        except Exception as exc:
            logger.exception("Failed to fetch codex entries: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/reflections", methods=["GET"])
    def get_reflections():
        try:
            query = runtime.supabase.table("reflection_log").select("*")
            user_id = request.args.get("user_id")
            if user_id:
                query = query.eq("user_id", user_id)
            result = query.order("created_at", desc=True).limit(20).execute()
            return jsonify(result.data)
        except Exception as exc:
            logger.exception("Failed to fetch reflections: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/memory", methods=["GET"])
    def get_memory_entries():
        user_id = request.args.get("user_id", "default_user")
        raw_limit = request.args.get("limit", "20")
        try:
            limit = max(1, min(int(raw_limit), 100))
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400

        try:
            return jsonify(runtime.memory_core.get_recent_entries(user_id=user_id, limit=limit))
        except Exception as exc:
            logger.exception("Failed to fetch memory entries: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/memory/tree", methods=["GET"])
    def get_memory_tree():
        user_id = request.args.get("user_id", "default_user")
        raw_limit = request.args.get("limit", "20")
        try:
            limit = max(1, min(int(raw_limit), 100))
        except ValueError:
            return jsonify({"error": "limit must be an integer"}), 400

        try:
            return jsonify(runtime.memory_tree.get_recent_node_views(user_id=user_id, limit=limit))
        except Exception as exc:
            logger.exception("Failed to fetch memory tree nodes: %s", exc)
            return jsonify({"error": str(exc)}), 500

    @app.route("/knowledge", methods=["GET"])
    def fetch_knowledge():
        user_id = request.args.get("user_id", "default_user")
        return jsonify(runtime.knowledge_core.get_recent_knowledge(user_id=user_id))

    @app.route("/knowledge", methods=["POST"])
    def add_knowledge():
        data = request.get_json(silent=True) or {}
        topic = data.get("topic", "")
        content = data.get("content", "")
        emotion_tag = data.get("emotion_tag", "neutral")
        user_id = data.get("user_id", "default_user")

        if enforce_firewall(content, user_id=user_id).get("status") == "blocked":
            return jsonify({"error": "Blocked content"}), 403

        result = runtime.knowledge_core.store_knowledge(
            topic=topic,
            content=content,
            emotion_tag=emotion_tag,
            user_id=user_id,
        )
        if not result:
            return jsonify({"error": "Knowledge entry was not stored"}), 400
        return jsonify({"status": "success", "entry": result})

    @app.route("/learn", methods=["GET"])
    def learn_knowledge_get():
        topic = request.args.get("topic", "General")
        user_id = request.args.get("user_id", "default_user")
        result = log_knowledge_sync(
            user_id=user_id,
            topic=topic,
            content=f"Auto content for topic: {topic}",
            source="GET request",
            emotion_tag="neutral",
            codex_impact="minimal",
            client=runtime.supabase,
            memory_tree=runtime.memory_tree,
        )
        return jsonify({"status": "logged", "topic": topic, "entry": result})

    @app.route("/learn", methods=["POST"])
    def learn_knowledge():
        if not request.is_json:
            return jsonify({"error": "Invalid content type"}), 400

        data = request.get_json(silent=True) or {}
        result = log_knowledge_sync(
            user_id=data.get("user_id", "default_user"),
            topic=data.get("topic", "General"),
            content=data.get("content", ""),
            source=data.get("source", "System"),
            emotion_tag=data.get("emotion_tag", "neutral"),
            codex_impact=data.get("codex_impact"),
            codex_entry_id=data.get("codex_entry_id"),
            client=runtime.supabase,
            memory_tree=runtime.memory_tree,
        )
        if not result:
            return jsonify({"error": "Knowledge log was not stored"}), 400
        return jsonify({"status": "logged", "entry": result})

    @app.route("/deepseek", methods=["POST"])
    def query_deepseek():
        data = request.get_json(silent=True) or {}
        topic = data.get("topic")
        user_id = data.get("user_id", "default_user")
        if not topic:
            return jsonify({"error": "Missing topic"}), 400

        try:
            result = runtime.deepseek_api.query(topic, user_id=user_id)
            return jsonify({"response": result})
        except Exception as exc:
            logger.exception("DeepSeek route failed: %s", exc)
            return jsonify({"error": str(exc)}), 500

    logger.info("Available routes:")
    for rule in app.url_map.iter_rules():
        logger.info("%s -> %s", rule, rule.endpoint)

    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    app.run(debug=settings.debug, host="0.0.0.0", port=settings.port)
