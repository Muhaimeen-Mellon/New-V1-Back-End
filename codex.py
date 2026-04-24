from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from brevity_core import BrevityCore
from gemini_api import GeminiUnavailableError, call_gemini_with_metadata
from local_ollama_api import LocalLLMUnavailableError, call_local_llm_with_metadata
from memory_tree_core import MemoryTreeCore
from openrouter_api import OpenRouterUnavailableError, call_mistral_with_metadata
from retrieval_utils import build_preview, compute_relevance_score, distinct_texts, normalize_text, tokenize
from runtime_config import get_settings, get_supabase_client
from tone_core import ToneCore

logger = logging.getLogger(__name__)


GENERIC_CODEX_PHRASES = (
    "i am claude",
    "artificial intelligence",
    "how can i assist you today",
    "i don't have access",
    "i'm sorry, but the information provided",
)

GENERIC_ARTICULATION_PREFIXES = (
    "from what i've already learned,",
    "from what i already remember,",
    "from my own stored context,",
    "from my internal patterns and prior symbolic traces,",
    "i have some related internal context, but not enough to answer cleanly without extending it.",
    "dream reflection:",
)

FRAGMENT_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "that",
    "this",
    "it",
    "as",
    "at",
    "from",
    "by",
    "about",
    "i",
    "me",
    "my",
    "you",
    "your",
    "we",
    "our",
    "they",
    "their",
}

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


class CodexEngine:
    def __init__(
        self,
        supabase_client: Optional[Any] = None,
        memory_tree: Optional[MemoryTreeCore] = None,
    ):
        self.supabase = supabase_client or get_supabase_client()
        self.memory_tree = memory_tree
        self.tone_core = ToneCore()
        self.brevity_core = BrevityCore()

    def build_system_prompt(self, tone: str) -> str:
        return (
            "You are Mellon, an emotionally intelligent AI. "
            f"Reply in a {tone} tone with empathy, clarity, and matching verbosity."
        )

    def get_recent_entries(self, user_id: str = "anon", limit: int = 20) -> List[Dict[str, Any]]:
        try:
            response = (
                self.supabase.table("codex_entries")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return getattr(response, "data", None) or []
        except Exception as exc:
            logger.exception("Failed to fetch codex entries for user '%s': %s", user_id, exc)
            return []

    def search_relevant_entries(
        self,
        text: str,
        user_id: str = "anon",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        entries = self.get_recent_entries(user_id=user_id, limit=50)
        results: List[Dict[str, Any]] = []

        for recency_rank, entry in enumerate(entries):
            candidate_text = " ".join(
                value
                for value in [
                    entry.get("prompt", ""),
                    entry.get("response", ""),
                ]
                if value
            ).strip()
            if not candidate_text:
                continue
            if self._should_skip_retrieval_entry(query=text, entry=entry, candidate_text=candidate_text):
                continue

            score = compute_relevance_score(text, candidate_text, recency_rank=recency_rank)
            if score <= 0:
                continue

            results.append(
                {
                    "source": "codex",
                    "score": min(1.0, score + 0.04),
                    "preview": build_preview(entry.get("response") or entry.get("prompt") or ""),
                    "content": entry.get("response") or entry.get("prompt") or "",
                    "entry": entry,
                    "source_detail": "codex",
                }
            )

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def _should_skip_retrieval_entry(self, *, query: str, entry: Dict[str, Any], candidate_text: str) -> bool:
        normalized_query = normalize_text(query)
        prompt_text = normalize_text(entry.get("prompt", ""))
        response_text = normalize_text(entry.get("response", ""))
        candidate_normalized = normalize_text(candidate_text)
        system_version = (entry.get("system_version") or "").lower()

        if not candidate_normalized:
            return True
        if prompt_text and normalized_query == prompt_text and prompt_text.endswith("?"):
            return True
        if response_text.endswith("?") and len(tokenize(response_text)) <= 8:
            return True
        if "memory-first/llm_fallback" in system_version:
            return True
        return any(phrase in candidate_normalized for phrase in GENERIC_CODEX_PHRASES)

    def generate_response(
        self,
        prompt: str,
        tone: Optional[str] = None,
        verbosity: Optional[str] = None,
        user_id: str = "anon",
        system_prompt: Optional[str] = None,
    ) -> str:
        detected_tone = tone or self.tone_core.detect_tone(prompt)
        brevity_level = verbosity or self.brevity_core.detect_verbosity_hint(prompt)
        active_system_prompt = system_prompt or self.build_system_prompt(detected_tone)

        try:
            llm_response = self._call_external_llm(prompt=prompt, system_prompt=active_system_prompt)
            raw_response = llm_response["content"]
        except RuntimeError as exc:
            logger.warning("Codex is using local fallback response generation: %s", exc)
            raw_response = self._fallback_response_from_memory(prompt, {"hits": [], "strategy": "llm_fallback"})
        except Exception as exc:
            logger.exception("Codex generation failed unexpectedly: %s", exc)
            raw_response = self._fallback_response(prompt)

        trimmed = self.brevity_core.apply_brevity(raw_response, brevity_level)
        final_response = self.tone_core.adjust_response(trimmed, detected_tone)
        self.log_belief(prompt=prompt, response=final_response, tone=detected_tone, user_id=user_id)
        return final_response

    def respond_with_memory_strategy(
        self,
        prompt: str,
        memory_bundle: Dict[str, Any],
        tone: Optional[str] = None,
        verbosity: Optional[str] = None,
        user_id: str = "anon",
    ) -> Dict[str, Any]:
        detected_tone = tone or self.tone_core.detect_tone(prompt)
        brevity_level = verbosity or self.brevity_core.detect_verbosity_hint(prompt)
        strategy = memory_bundle.get("strategy", "llm_fallback")
        fallback_reason = memory_bundle.get("fallback_reason")
        context_text = self._build_curated_context_text(memory_bundle)
        llm_called = False
        llm_attempted = False
        llm_error: Optional[str] = None
        model_used: Optional[str] = None
        review_state = memory_bundle.get("review_state")
        review_reason = memory_bundle.get("review_reason")

        if review_state == "conflicting_memory" or fallback_reason in {"conflicting_memory", "conflicting_memory_compare"}:
            raw_response = self._articulate_conflicting_memory(prompt, memory_bundle)
        elif review_state == "insufficient_memory":
            raw_response = self._articulate_insufficient_memory(prompt, memory_bundle)
        elif review_state in {"partial_memory", "reasoning_risk"}:
            raw_response = self._articulate_review_guided_partial(prompt, memory_bundle)
        elif memory_bundle.get("input_type") == "future_modeling" and fallback_reason == "memory_partial":
            raw_response = self._articulate_partial_future(memory_bundle)
        elif strategy == "internal_memory_only":
            raw_response = self.synthesize_from_memory(prompt, memory_bundle)
        else:
            llm_attempted = True
            system_prompt = self.build_memory_first_system_prompt(
                tone=detected_tone,
                memory_bundle=memory_bundle,
            )
            llm_prompt = self.build_memory_augmented_prompt(prompt, context_text=context_text)
            try:
                completion = self._call_external_llm(prompt=llm_prompt, system_prompt=system_prompt)
                raw_response = completion["content"]
                llm_called = True
                model_used = completion["model_used"]
            except RuntimeError as exc:
                llm_error = str(exc)
                logger.warning("Codex LLM path unavailable; falling back locally: %s", exc)
                raw_response = self._fallback_response_from_memory(prompt, memory_bundle)
            except Exception as exc:
                llm_error = str(exc)
                logger.exception("Codex memory-augmented generation failed unexpectedly: %s", exc)
                raw_response = self._fallback_response_from_memory(prompt, memory_bundle)

        trimmed = self.brevity_core.apply_brevity(raw_response, brevity_level)
        final_response = self.tone_core.adjust_response(trimmed, detected_tone)
        self.log_belief(
            prompt=prompt,
            response=final_response,
            tone=detected_tone,
            user_id=user_id,
            system_version=f"memory-first/{strategy}",
        )
        curated_hits = self._serialize_curated_hits(memory_bundle, limit=4)
        return {
            "response": final_response,
            "tone": detected_tone,
            "response_origin": strategy,
            "llm_called": llm_called,
            "llm_attempted": llm_attempted,
            "model_used": model_used,
            "llm_error": llm_error,
            "curated_hits": curated_hits,
        }

    def _articulate_insufficient_memory(self, prompt: str, memory_bundle: Dict[str, Any]) -> str:
        input_type = memory_bundle.get("input_type", "general")
        hits = self._select_synthesis_hits(memory_bundle, limit=2)
        fragments = [
            self._extract_hit_fragment(hit, input_type=input_type)
            for hit in hits
        ]
        fragments = [fragment for fragment in fragments if fragment]
        review_reason = memory_bundle.get("review_reason") or memory_bundle.get("fallback_reason") or "insufficient_internal_memory"

        if input_type == "future_modeling" or review_reason == "vague_future_traces":
            if fragments:
                return (
                    "This is not established strongly enough in my internal future-modeling memory yet. "
                    f"The closest trace I have is {fragments[0].rstrip('.!?')}, but it remains too weak or vague to treat as a grounded forecast."
                )
            return "This is not established strongly enough in my internal future-modeling memory yet."

        if fragments:
            return (
                "This is not established in my internal memory yet. "
                f"The closest trace I have is {fragments[0].rstrip('.!?')}, but it is not strong enough to treat as settled memory."
            )
        return "This is not established in my internal memory yet."

    def _articulate_review_guided_partial(self, prompt: str, memory_bundle: Dict[str, Any]) -> str:
        input_type = memory_bundle.get("input_type", "general")
        review_state = memory_bundle.get("review_state")
        review_reason = memory_bundle.get("review_reason") or "memory_partial"
        hits = self._select_synthesis_hits(memory_bundle, limit=3)
        fragments = [
            self._extract_hit_fragment(hit, input_type=input_type)
            for hit in hits
        ]
        fragments = distinct_texts([fragment for fragment in fragments if fragment], limit=2)
        if not fragments:
            return self._articulate_insufficient_memory(prompt, memory_bundle)

        lead = fragments[0].rstrip(".!?")
        follow_up = fragments[1].rstrip(".!?") if len(fragments) > 1 else ""

        if input_type == "future_modeling" or review_reason == "vague_future_traces":
            response = f"I have a partial future-modeling pattern in memory pointing toward {lead}."
            if follow_up and follow_up.lower() not in lead.lower():
                response = f"{response} A second trace points toward {follow_up}."
            return f"{response} I would treat this as tentative rather than settled because the support is still weak."

        if review_state == "reasoning_risk":
            response = f"I have some relevant internal memory, but the answer is more inferred than directly recalled. The strongest trace says {lead}."
            if follow_up and follow_up.lower() not in lead.lower():
                response = f"{response} A second supporting trace says {follow_up}."
            return f"{response} I would keep the conclusion cautious."

        response = f"I have partial internal memory on this. The strongest trace says {lead}."
        if follow_up and follow_up.lower() not in lead.lower():
            response = f"{response} Another supporting trace says {follow_up}."
        return f"{response} This is a partial recall, not a fully settled memory."

    def _call_external_llm(self, prompt: str, system_prompt: str) -> Dict[str, str]:
        settings = get_settings()
        errors: List[str] = []

        if settings.has_local_llm:
            try:
                completion = call_local_llm_with_metadata(prompt=prompt, system=system_prompt)
                return {"content": completion.content, "model_used": completion.model_used}
            except LocalLLMUnavailableError as exc:
                errors.append(f"local_ollama: {exc}")

        if settings.has_gemini:
            try:
                completion = call_gemini_with_metadata(prompt=prompt, system=system_prompt)
                return {"content": completion.content, "model_used": completion.model_used}
            except GeminiUnavailableError as exc:
                errors.append(f"gemini: {exc}")

        if settings.has_openrouter:
            try:
                completion = call_mistral_with_metadata(prompt=prompt, system=system_prompt)
                return {"content": completion.content, "model_used": completion.model_used}
            except OpenRouterUnavailableError as exc:
                errors.append(f"openrouter: {exc}")

        if errors:
            raise RuntimeError(" | ".join(errors))
        raise RuntimeError("No external LLM provider configured (set GEMINI_API_KEY or OPENROUTER_API_KEY).")

    def build_memory_first_system_prompt(self, tone: str, memory_bundle: Dict[str, Any]) -> str:
        input_type = memory_bundle.get("input_type", "general")
        fallback_reason = memory_bundle.get("fallback_reason")
        guardrail = ""
        if input_type == "future_modeling":
            guardrail = (
                " Use only the strongest future-modeling traces. "
                "Do not let weak or unrelated symbolic fragments steer the answer. "
                "Never claim you lack access to context if retrieved memory is present."
            )
        elif fallback_reason == "memory_partial":
            guardrail = (
                " Stay close to the retrieved memory. "
                "Do not invent facts that are not supported by the supplied context."
            )
        return (
            "You are Mellon. Use Mellon's internal memory as the primary source of truth. "
            "Only extend, clarify, or bridge gaps where the supplied memory is incomplete. "
            "Do not ignore retrieved memory context, and do not overwrite it without saying uncertainty exists. "
            f"Reply in a {tone} tone with empathy, clarity, and matching verbosity. "
            f"Input classification: {input_type}."
            f"{guardrail}"
        )

    def build_memory_augmented_prompt(self, prompt: str, context_text: str) -> str:
        if not context_text:
            return prompt
        return (
            "Retrieved Mellon memory context:\n"
            f"{context_text}\n\n"
            "Use this internal context first. If it is incomplete, extend it carefully. "
            "Do not repeat generic assistant boilerplate. Keep the answer grounded in the supplied memory.\n"
            f"User request:\n{prompt}"
        )

    def synthesize_from_memory(self, prompt: str, memory_bundle: Dict[str, Any]) -> str:
        input_type = memory_bundle.get("input_type", "general")
        hits = memory_bundle.get("hits", [])
        if not hits:
            return self._fallback_response(prompt)

        selected_hits = self._select_synthesis_hits(memory_bundle, limit=3)
        preferred_fragments = [self._extract_hit_fragment(hit, input_type=input_type) for hit in selected_hits]
        selected = self._distinct_fragments_by_overlap(
            [fragment for fragment in preferred_fragments if fragment],
            input_type=input_type,
            limit=2,
        )
        lead = selected[0] if selected else self._fallback_response(prompt)
        follow_up = selected[1] if len(selected) > 1 else ""
        if follow_up and (
            len(follow_up.split()) < 4
            or follow_up.lower() in lead.lower()
            or lead.lower() in follow_up.lower()
        ):
            follow_up = ""

        if input_type == "factual":
            response = f"From what I've already learned, {lead}"
        elif input_type in {"introspective", "personal"}:
            response = f"From what I already remember, {lead}"
        elif input_type in {"symbolic", "future_modeling"}:
            response = f"From my internal future-modeling traces, {lead}" if input_type == "future_modeling" else f"From my internal patterns and prior symbolic traces, {lead}"
        else:
            response = f"From my own stored context, {lead}"

        if follow_up:
            response = f"{response} {follow_up}"
        return response.strip()

    def _fallback_response_from_memory(self, prompt: str, memory_bundle: Dict[str, Any]) -> str:
        fallback_reason = memory_bundle.get("fallback_reason")
        if fallback_reason in {"conflicting_memory", "conflicting_memory_compare"}:
            return self._articulate_conflicting_memory(prompt, memory_bundle)
        if memory_bundle.get("hits"):
            synthesized = self.synthesize_from_memory(prompt, memory_bundle)
            if memory_bundle.get("strategy") == "internal_memory_plus_llm":
                if memory_bundle.get("input_type") == "future_modeling":
                    return f"I have a partial future-modeling pattern in memory, but it is not complete enough to stand on its own. {synthesized}"
                return f"I have some related internal context, but not enough to answer cleanly from memory alone. {synthesized}"
            if memory_bundle.get("strategy") == "llm_fallback":
                return f"I don't have enough internal memory for a confident answer yet. {synthesized}"
            return synthesized
        return self._fallback_response(prompt)

    def _build_curated_context_text(self, memory_bundle: Dict[str, Any]) -> str:
        input_type = memory_bundle.get("input_type", "general")
        lines: List[str] = []
        for hit in self._select_synthesis_hits(memory_bundle, limit=4):
            fragment = self._extract_hit_fragment(hit, input_type=input_type)
            if not fragment:
                continue
            lines.append(
                f"- [{hit.get('source')}] score={hit.get('score', 0.0):.2f} {build_preview(fragment, limit=180)}"
            )
        return "\n".join(lines)

    def _select_synthesis_hits(self, memory_bundle: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
        input_type = memory_bundle.get("input_type", "general")
        hits = list(memory_bundle.get("hits") or [])
        query_keywords = [
            normalize_text(keyword)
            for keyword in (memory_bundle.get("query_keywords") or [])
            if normalize_text(keyword)
        ]
        strict_attribute_query = bool(set(tokenize(" ".join(query_keywords))) & STRICT_ATTRIBUTE_QUERY_TOKENS)
        source_priority = {
            "factual": {"self_model": 7, "architecture": 6, "constraint": 5, "knowledge": 5, "user_model": 4, "memory": 4, "reflection": 1, "codex": 0},
            "introspective": {"user_model": 7, "self_model": 6, "memory": 6, "reflection": 4, "constraint": 2, "knowledge": 1, "codex": 0},
            "personal": {"user_model": 7, "self_model": 6, "memory": 6, "reflection": 4, "constraint": 2, "knowledge": 1, "codex": 0},
            "symbolic": {"dream": 5, "simulated_dream": 5, "simulation": 4, "reflection": 2, "codex": 0},
            "future_modeling": {"simulation": 5, "dream": 4, "simulated_dream": 4, "self_model": 2, "architecture": 2, "reflection": 1, "memory": 1, "codex": 0},
            "general": {"self_model": 6, "architecture": 6, "constraint": 5, "user_model": 5, "memory": 4, "knowledge": 4, "reflection": 3, "codex": 0},
        }
        priority_map = source_priority.get(input_type, source_priority["general"])
        primary_sources = {
            "self_model",
            "user_model",
            "architecture",
            "constraint",
            "memory",
            "reflection",
            "knowledge",
            "dream",
            "simulation",
            "simulated_dream",
        }
        native_future_sources = {"dream", "simulation", "simulated_dream"}
        max_primary_score = max(
            (float(hit.get("score", 0.0)) for hit in hits if hit.get("source") in primary_sources),
            default=0.0,
        )
        strong_primary_present = max_primary_score >= 0.58
        strong_future_native_present = any(
            float(hit.get("score", 0.0)) >= 0.34 and hit.get("source") in native_future_sources
            for hit in hits
        )

        ranked: List[tuple[int, float, Dict[str, Any]]] = []
        for hit in hits:
            source = hit.get("source", "memory")
            fragment = self._extract_hit_fragment(hit, input_type=input_type)
            if not fragment:
                continue
            alignment_score = self._query_alignment_score(fragment=fragment, query_keywords=query_keywords)
            if strict_attribute_query and alignment_score <= 0:
                continue
            if input_type == "future_modeling" and source not in {"simulation", "dream", "simulated_dream"} and hit.get("score", 0.0) < 0.55:
                continue
            if source == "codex":
                if input_type == "future_modeling" and strong_future_native_present:
                    continue
                codex_score = float(hit.get("score", 0.0))
                if strong_primary_present and codex_score <= (max_primary_score + 0.06):
                    continue
                if codex_score < 0.62:
                    continue
            ranked.append((alignment_score, priority_map.get(source, 0), float(hit.get("score", 0.0)), hit))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        selected: List[Dict[str, Any]] = []
        seen_fragments: List[str] = []
        for _, _, _, hit in ranked:
            fragment = normalize_text(self._extract_hit_fragment(hit, input_type=input_type))
            if not fragment:
                continue
            if any(self._is_redundant_fragment(fragment, seen, input_type=input_type) for seen in seen_fragments):
                continue
            seen_fragments.append(fragment)
            selected.append(hit)
            if len(selected) >= limit:
                break
        if selected:
            return selected
        if strict_attribute_query:
            return []
        return hits[:limit]

    def _extract_hit_fragment(self, hit: Dict[str, Any], *, input_type: str) -> str:
        source = hit.get("source", "memory")
        candidate = (
            hit.get("summary")
            or hit.get("content")
            or hit.get("preview")
            or ""
        ).strip()
        if not candidate:
            return ""
        lowered = normalize_text(candidate)
        if source == "codex" and self._is_noisy_codex_fragment(lowered):
            return ""
        if lowered.endswith("?") and source in {"memory", "codex"}:
            return ""
        for prefix in GENERIC_ARTICULATION_PREFIXES:
            if lowered.startswith(prefix):
                candidate = candidate[len(prefix):].strip(" ,:-")
                lowered = normalize_text(candidate)
        if source == "reflection" and candidate.lower().startswith("i believe this because..."):
            candidate = self._strip_reflection_wrapper(candidate)
        lowered = normalize_text(candidate)
        if input_type in {"introspective", "personal"} and lowered.startswith(
            ("what do you", "what is", "who am", "why do", "why am", "do you remember")
        ):
            return ""
        if input_type == "future_modeling" and source in {"dream", "simulated_dream", "simulation", "codex"}:
            candidate = self._simplify_future_fragment(candidate)
        return build_preview(candidate, limit=220).strip()

    def _strip_reflection_wrapper(self, text: str) -> str:
        lowered = text.lower()
        marker = "mellon replied:"
        if marker in lowered:
            idx = lowered.index(marker) + len(marker)
            return text[idx:].strip()
        marker = "user said:"
        if marker in lowered:
            idx = lowered.index(marker) + len(marker)
            fragment = text[idx:].strip()
            if "mellon repl" in fragment.lower():
                fragment = fragment[: fragment.lower().index("mellon repl")].strip()
            return fragment
        return text.replace("I believe this because...", "", 1).strip()

    def _simplify_future_fragment(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.lower().startswith("dream reflection:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        if "and in it Mellon saw" in cleaned:
            cleaned = cleaned.replace("and in it Mellon saw", "where Mellon saw", 1)
        return cleaned

    def _is_noisy_codex_fragment(self, lowered: str) -> bool:
        if any(phrase in lowered for phrase in GENERIC_CODEX_PHRASES):
            return True
        return any(
            phrase in lowered
            for phrase in [
                "i have some related internal context",
                "i don't have enough internal memory",
                "could you provide more details",
            ]
        )

    def _distinct_fragments_by_overlap(self, fragments: List[str], *, input_type: str, limit: int) -> List[str]:
        selected: List[str] = []
        for fragment in fragments:
            normalized = normalize_text(fragment)
            if not normalized:
                continue
            if any(self._is_redundant_fragment(normalized, normalize_text(prev), input_type=input_type) for prev in selected):
                continue
            selected.append(fragment)
            if len(selected) >= limit:
                break
        return selected

    def _is_redundant_fragment(self, fragment: str, previous: str, *, input_type: str) -> bool:
        if not fragment or not previous:
            return False
        if fragment == previous:
            return True
        if fragment in previous or previous in fragment:
            shorter = fragment if len(fragment) <= len(previous) else previous
            return len(tokenize(shorter)) >= 4

        fragment_tokens = self._core_fragment_tokens(fragment)
        previous_tokens = self._core_fragment_tokens(previous)
        if not fragment_tokens or not previous_tokens:
            return False

        intersection = len(fragment_tokens & previous_tokens)
        union = len(fragment_tokens | previous_tokens)
        if union == 0:
            return False

        jaccard = intersection / union
        coverage = intersection / min(len(fragment_tokens), len(previous_tokens))
        if input_type in {"introspective", "personal"}:
            return jaccard >= 0.5 or (coverage >= 0.72 and intersection >= 4)
        return jaccard >= 0.68 or (coverage >= 0.8 and intersection >= 5)

    def _core_fragment_tokens(self, text: str) -> set[str]:
        return {
            token
            for token in tokenize(text)
            if len(token) > 2 and token not in FRAGMENT_STOPWORDS
        }

    def _query_alignment_score(self, *, fragment: str, query_keywords: List[str]) -> int:
        if not fragment or not query_keywords:
            return 0
        normalized_fragment = normalize_text(fragment)
        fragment_tokens = set(tokenize(fragment))
        phrase_matches = sum(1 for keyword in query_keywords if " " in keyword and keyword in normalized_fragment)
        token_matches = sum(
            1
            for token in tokenize(" ".join(query_keywords))
            if token not in GENERIC_QUERY_TOKENS and token not in FRAGMENT_STOPWORDS and token in fragment_tokens
        )
        return (phrase_matches * 2) + token_matches

    def _serialize_curated_hits(self, memory_bundle: Dict[str, Any], limit: int = 4) -> List[Dict[str, Any]]:
        input_type = memory_bundle.get("input_type", "general")
        fallback_reason = memory_bundle.get("fallback_reason")
        selected_hits = self._select_synthesis_hits(memory_bundle, limit=limit)
        has_non_codex = any(hit.get("source") != "codex" for hit in selected_hits)
        curated: List[Dict[str, Any]] = []
        for hit in selected_hits:
            source = hit.get("source")
            if source == "codex" and (
                has_non_codex
                or (input_type == "future_modeling" and fallback_reason == "memory_partial")
            ):
                continue
            fragment = self._extract_hit_fragment(hit, input_type=input_type)
            if not fragment:
                continue
            curated.append(
                {
                    "source": source,
                    "source_detail": hit.get("source_detail"),
                    "score": hit.get("score"),
                    "preview": build_preview(fragment, limit=180),
                }
            )
        return curated

    def _articulate_conflicting_memory(self, prompt: str, memory_bundle: Dict[str, Any]) -> str:
        relevant_hits = self._select_conflict_hits(memory_bundle, limit=2)
        if len(relevant_hits) < 2:
            relevant_hits = self._select_synthesis_hits(memory_bundle, limit=3)
        fragments = [
            self._extract_hit_fragment(hit, input_type=memory_bundle.get("input_type", "general"))
            for hit in relevant_hits
        ]
        fragments = [fragment for fragment in fragments if fragment]
        if len(fragments) >= 2:
            return (
                "My stored memory conflicts on this point. "
                f"One memory says {fragments[0].rstrip('.!?')}. Another says {fragments[1].rstrip('.!?')}. "
                "I can't resolve which one is current from internal memory alone."
            )
        if fragments:
            return (
                "My stored memory appears unstable on this point. "
                f"The strongest surviving trace is: {fragments[0].rstrip('.!?')}. "
                "I can't confirm it cleanly from memory alone."
            )
        return "My stored memory conflicts on this point, and I can't resolve it cleanly from internal memory alone."

    def _select_conflict_hits(self, memory_bundle: Dict[str, Any], limit: int = 2) -> List[Dict[str, Any]]:
        hits = list(memory_bundle.get("conflict_hits") or memory_bundle.get("leaf_hits") or memory_bundle.get("hits") or [])
        hits_by_id = {
            hit.get("entry", {}).get("id"): hit
            for hit in hits
            if hit.get("entry", {}).get("id")
        }
        best_pair: List[Dict[str, Any]] = []
        best_score = -1.0
        for hit in hits:
            node = hit.get("node") or {}
            linked_ids = node.get("contradiction_links") or []
            for linked_id in linked_ids:
                other = hits_by_id.get(linked_id)
                if not other:
                    continue
                pair = [hit, other]
                pair_score = sum(float(item.get("score", 0.0)) for item in pair)
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = pair
        if best_pair:
            best_pair.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            return best_pair[:limit]
        leaf_hits = list(memory_bundle.get("leaf_hits") or [])
        query_keywords = [
            normalize_text(keyword)
            for keyword in (memory_bundle.get("query_keywords") or [])
            if normalize_text(keyword)
        ]
        strict_attribute_query = bool(set(tokenize(" ".join(query_keywords))) & STRICT_ATTRIBUTE_QUERY_TOKENS)
        if strict_attribute_query and leaf_hits:
            ranked_leaf_hits = sorted(
                (
                    hit
                    for hit in leaf_hits
                    if self._query_alignment_score(
                        fragment=self._extract_hit_fragment(hit, input_type=memory_bundle.get("input_type", "general")),
                        query_keywords=query_keywords,
                    ) > 0
                ),
                key=lambda item: float(item.get("score", 0.0)),
                reverse=True,
            )
            if len(ranked_leaf_hits) >= 2:
                return ranked_leaf_hits[:limit]
        return []

    def _articulate_partial_future(self, memory_bundle: Dict[str, Any]) -> str:
        hits = self._select_synthesis_hits(memory_bundle, limit=3)
        fragments = [
            self._extract_hit_fragment(hit, input_type="future_modeling")
            for hit in hits
            if hit.get("source") in {"simulation", "dream", "simulated_dream"}
        ]
        fragments = distinct_texts([fragment for fragment in fragments if fragment], limit=2)
        if not fragments:
            return "I have some future-modeling traces, but they are too weak or scattered to form a clean pattern yet."
        lead = fragments[0].rstrip(".!?")
        follow_up = fragments[1].rstrip(".!?") if len(fragments) > 1 else ""
        response = f"From my internal future-modeling traces, I see a partial pattern of {lead}."
        if follow_up and follow_up.lower() not in lead.lower():
            response = f"{response} A second supporting trace points toward {follow_up}."
        return f"{response} This is a partial pattern, not a settled forecast."

    def log_belief(
        self,
        prompt: str,
        response: str,
        tone: str,
        user_id: str = "anon",
        system_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        entry = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "prompt": prompt,
            "response": response,
            "tone": tone,
            "created_at": datetime.utcnow().isoformat(),
        }
        if system_version:
            entry["system_version"] = system_version

        logger.info("Codex belief generated for user '%s' in tone '%s'.", user_id, tone)

        try:
            result = self.supabase.table("codex_entries").insert(entry).execute()
            stored = result.data[0] if getattr(result, "data", None) else entry
            if self.memory_tree:
                response_origin = system_version.split("/", 1)[1] if system_version and "/" in system_version else "codex"
                self.memory_tree.remember(
                    user_id=user_id,
                    source_kind="codex",
                    text=response,
                    related_input=prompt,
                    emotion_tag=tone,
                    source_entry_id=stored.get("id"),
                    summary=build_preview(response, limit=120),
                    importance_score=0.64 if response_origin == "internal_memory_only" else 0.52,
                    emotional_weight=0.22 if tone == "neutral" else 0.34,
                    identity_relevance=0.45,
                    pillar_memory=response_origin == "internal_memory_only" and "remember" in prompt.lower(),
                    cluster_id=f"codex:{response_origin}",
                    metadata={
                        "prompt_preview": build_preview(prompt, limit=120),
                        "response_origin": response_origin,
                        "system_version": system_version,
                    },
                )
            return stored
        except Exception as exc:
            logger.exception("Failed to log codex belief for user '%s': %s", user_id, exc)
            return None

    def log_codex_belief(
        self,
        prompt: str,
        response: str,
        tone: str,
        user_id: str = "anon",
        system_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.log_belief(
            prompt=prompt,
            response=response,
            tone=tone,
            user_id=user_id,
            system_version=system_version,
        )

    def _fallback_response(self, prompt: str) -> str:
        cleaned = " ".join(prompt.strip().split())
        if not cleaned:
            return "I'm here and listening."
        if cleaned.endswith("?"):
            return f"My grounded take is: {cleaned}"
        if len(cleaned.split()) <= 6:
            return f"You seem focused on '{cleaned}'. Give me one more detail and I can go deeper."
        return f"I hear you. My grounded read is: {cleaned}"
