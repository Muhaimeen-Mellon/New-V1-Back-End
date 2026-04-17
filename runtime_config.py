from __future__ import annotations

import copy
import logging
import os
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):  # type: ignore[override]
        return False


LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
load_dotenv()


def _as_bool(raw_value: Optional[str], default: bool = False) -> bool:
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def configure_logging(level: Optional[str] = None) -> None:
    root_logger = logging.getLogger()
    target_level = getattr(
        logging,
        (level or os.getenv("LOG_LEVEL", "INFO")).upper(),
        logging.INFO,
    )

    if root_logger.handlers:
        root_logger.setLevel(target_level)
        return

    logging.basicConfig(level=target_level, format=LOG_FORMAT)


@dataclass(frozen=True)
class MellonSettings:
    port: int
    debug: bool
    supabase_url: Optional[str]
    supabase_anon_key: Optional[str]
    local_llm_enabled: bool
    local_llm_base_url: str
    local_llm_model: str
    local_llm_timeout_seconds: int
    gemini_api_key: Optional[str]
    gemini_model: str
    gemini_timeout_seconds: int
    openrouter_api_key: Optional[str]
    openrouter_model: str
    openrouter_timeout_seconds: int
    cors_origins: str

    @property
    def has_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_anon_key)

    @property
    def has_local_llm(self) -> bool:
        return self.local_llm_enabled

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)


@lru_cache(maxsize=1)
def get_settings() -> MellonSettings:
    local_timeout_raw = os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "45")
    try:
        local_llm_timeout_seconds = int(local_timeout_raw)
    except ValueError:
        local_llm_timeout_seconds = 45
    local_llm_timeout_seconds = max(5, min(local_llm_timeout_seconds, 300))

    gemini_timeout_raw = os.getenv("GEMINI_TIMEOUT_SECONDS", "30")
    try:
        gemini_timeout_seconds = int(gemini_timeout_raw)
    except ValueError:
        gemini_timeout_seconds = 30
    gemini_timeout_seconds = max(5, min(gemini_timeout_seconds, 120))

    timeout_raw = os.getenv("OPENROUTER_TIMEOUT_SECONDS", "30")
    try:
        openrouter_timeout_seconds = int(timeout_raw)
    except ValueError:
        openrouter_timeout_seconds = 30
    openrouter_timeout_seconds = max(5, min(openrouter_timeout_seconds, 120))

    return MellonSettings(
        port=int(os.getenv("PORT", "5000")),
        debug=_as_bool(os.getenv("FLASK_DEBUG"), default=False),
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_anon_key=os.getenv("SUPABASE_ANON_KEY"),
        local_llm_enabled=_as_bool(os.getenv("LOCAL_LLM_ENABLED"), default=True),
        local_llm_base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434"),
        local_llm_model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:0.5b"),
        local_llm_timeout_seconds=local_llm_timeout_seconds,
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        gemini_timeout_seconds=gemini_timeout_seconds,
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free"),
        openrouter_timeout_seconds=openrouter_timeout_seconds,
        cors_origins=os.getenv("CORS_ORIGINS", "*"),
    )


class LocalQueryResponse:
    def __init__(self, data: Optional[List[Dict[str, Any]]] = None, status_code: int = 200):
        self.data = data or []
        self.status_code = status_code


class LocalSupabaseTable:
    def __init__(self, store: Dict[str, List[Dict[str, Any]]], table_name: str):
        self.store = store
        self.table_name = table_name
        self._action = "select"
        self._payload: Any = None
        self._fields: Optional[str] = "*"
        self._filters: List[tuple[str, Any]] = []
        self._order_by: Optional[str] = None
        self._order_desc = False
        self._limit: Optional[int] = None

    def insert(self, payload: Any) -> "LocalSupabaseTable":
        self._action = "insert"
        self._payload = payload
        return self

    def select(self, fields: str = "*") -> "LocalSupabaseTable":
        self._action = "select"
        self._fields = fields
        return self

    def update(self, payload: Dict[str, Any]) -> "LocalSupabaseTable":
        self._action = "update"
        self._payload = payload or {}
        return self

    def eq(self, field: str, value: Any) -> "LocalSupabaseTable":
        self._filters.append((field, value))
        return self

    def order(self, field: str, desc: bool = False) -> "LocalSupabaseTable":
        self._order_by = field
        self._order_desc = desc
        return self

    def limit(self, value: int) -> "LocalSupabaseTable":
        self._limit = value
        return self

    def execute(self) -> LocalQueryResponse:
        rows = self.store.setdefault(self.table_name, [])

        if self._action == "insert":
            incoming = self._payload if isinstance(self._payload, list) else [self._payload]
            inserted: List[Dict[str, Any]] = []
            for item in incoming:
                record = copy.deepcopy(item or {})
                record.setdefault("id", str(uuid.uuid4()))
                rows.append(record)
                inserted.append(record)
            return LocalQueryResponse(inserted, status_code=201)

        if self._action == "update":
            updated: List[Dict[str, Any]] = []
            for row in rows:
                if all(row.get(field) == value for field, value in self._filters):
                    row.update(copy.deepcopy(self._payload))
                    updated.append(copy.deepcopy(row))
            return LocalQueryResponse(updated, status_code=200)

        results = [copy.deepcopy(row) for row in rows]
        for field, value in self._filters:
            results = [row for row in results if row.get(field) == value]

        if self._order_by:
            results.sort(key=lambda row: row.get(self._order_by) or "", reverse=self._order_desc)

        if self._limit is not None:
            results = results[: self._limit]

        if self._fields and self._fields != "*":
            selected_fields = [field.strip() for field in self._fields.split(",") if field.strip()]
            results = [
                {field: row.get(field) for field in selected_fields}
                for row in results
            ]

        return LocalQueryResponse(results, status_code=200)


class LocalSupabaseClient:
    def __init__(self):
        self._store: Dict[str, List[Dict[str, Any]]] = {}
        self.mode = "in-memory"

    def table(self, table_name: str) -> LocalSupabaseTable:
        return LocalSupabaseTable(self._store, table_name)


@lru_cache(maxsize=1)
def get_supabase_client() -> Any:
    configure_logging()
    logger = logging.getLogger(__name__)
    settings = get_settings()

    if settings.has_supabase:
        try:
            from supabase import create_client

            logger.info("Using configured Supabase backend.")
            return create_client(settings.supabase_url, settings.supabase_anon_key)
        except Exception as exc:
            logger.warning(
                "Supabase initialization failed; falling back to in-memory storage. %s",
                exc,
            )
    else:
        logger.info("Supabase credentials not configured; using in-memory storage.")

    return LocalSupabaseClient()


def get_storage_mode() -> str:
    client = get_supabase_client()
    if isinstance(client, LocalSupabaseClient):
        return "in-memory"
    return "supabase"


def get_storage_status() -> Dict[str, Any]:
    settings = get_settings()
    storage_mode = get_storage_mode()
    return {
        "backend": storage_mode,
        "connected": storage_mode == "supabase",
        "fallback_active": storage_mode != "supabase",
        "supabase_configured": settings.has_supabase,
    }


def get_runtime_snapshot() -> Dict[str, Any]:
    settings = get_settings()
    if settings.has_local_llm:
        model_mode = "local-ollama"
        default_model = settings.local_llm_model
    elif settings.has_gemini:
        model_mode = "gemini"
        default_model = settings.gemini_model
    elif settings.has_openrouter:
        model_mode = "openrouter"
        default_model = settings.openrouter_model
    else:
        model_mode = "local-fallback"
        default_model = None

    return {
        "storage_mode": get_storage_mode(),
        "storage": get_storage_status(),
        "model_mode": model_mode,
        "default_model": default_model,
        "debug": settings.debug,
    }
