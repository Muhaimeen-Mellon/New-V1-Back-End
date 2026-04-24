from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class RestSupabaseError(RuntimeError):
    pass


@dataclass
class RestQueryResponse:
    data: List[Dict[str, Any]]
    status_code: int


class RestSupabaseTable:
    def __init__(self, *, client: "RestSupabaseClient", table_name: str):
        self.client = client
        self.table_name = table_name
        self._action = "select"
        self._payload: Any = None
        self._fields = "*"
        self._filters: List[tuple[str, Any]] = []
        self._order_by: Optional[str] = None
        self._order_desc = False
        self._limit: Optional[int] = None

    def insert(self, payload: Any) -> "RestSupabaseTable":
        self._action = "insert"
        self._payload = payload
        return self

    def select(self, fields: str = "*") -> "RestSupabaseTable":
        self._action = "select"
        self._fields = fields or "*"
        return self

    def update(self, payload: Dict[str, Any]) -> "RestSupabaseTable":
        self._action = "update"
        self._payload = payload or {}
        return self

    def eq(self, field: str, value: Any) -> "RestSupabaseTable":
        self._filters.append((field, "eq", value))
        return self

    def gte(self, field: str, value: Any) -> "RestSupabaseTable":
        self._filters.append((field, "gte", value))
        return self

    def lte(self, field: str, value: Any) -> "RestSupabaseTable":
        self._filters.append((field, "lte", value))
        return self

    def gt(self, field: str, value: Any) -> "RestSupabaseTable":
        self._filters.append((field, "gt", value))
        return self

    def lt(self, field: str, value: Any) -> "RestSupabaseTable":
        self._filters.append((field, "lt", value))
        return self

    def order(self, field: str, desc: bool = False) -> "RestSupabaseTable":
        self._order_by = field
        self._order_desc = desc
        return self

    def limit(self, value: int) -> "RestSupabaseTable":
        self._limit = value
        return self

    def execute(self) -> RestQueryResponse:
        try:
            import requests
        except Exception as exc:  # pragma: no cover - import guard
            raise RestSupabaseError("The 'requests' dependency is required for REST Supabase access.") from exc

        endpoint = f"{self.client.base_url}/rest/v1/{self.table_name}"
        headers = dict(self.client.headers)
        params = self._build_params()

        try:
            if self._action == "insert":
                headers["Prefer"] = "return=representation"
                response = self.client.session.post(
                    endpoint,
                    headers=headers,
                    json=self._payload,
                    timeout=self.client.timeout_seconds,
                )
            elif self._action == "update":
                headers["Prefer"] = "return=representation"
                response = self.client.session.patch(
                    endpoint,
                    headers=headers,
                    params=params,
                    json=self._payload,
                    timeout=self.client.timeout_seconds,
                )
            else:
                response = self.client.session.get(
                    endpoint,
                    headers=headers,
                    params=params,
                    timeout=self.client.timeout_seconds,
                )
        except requests.RequestException as exc:
            raise RestSupabaseError(
                f"Supabase REST request failed for table '{self.table_name}': {exc}"
            ) from exc

        if response.status_code >= 400:
            raise RestSupabaseError(
                f"Supabase REST error for table '{self.table_name}' ({response.status_code}): {response.text[:400]}"
            )

        if not response.text.strip():
            return RestQueryResponse(data=[], status_code=response.status_code)

        try:
            payload = response.json()
        except ValueError as exc:
            raise RestSupabaseError(
                f"Supabase REST returned invalid JSON for table '{self.table_name}'."
            ) from exc

        if isinstance(payload, list):
            data = payload
        elif isinstance(payload, dict):
            data = [payload]
        else:
            data = []
        return RestQueryResponse(data=data, status_code=response.status_code)

    def _build_params(self) -> Dict[str, str]:
        params: Dict[str, str] = {}
        if self._action == "select":
            params["select"] = self._fields
        for field, operator, value in self._filters:
            params[field] = f"{operator}.{self._format_filter_value(value)}"
        if self._order_by:
            direction = "desc" if self._order_desc else "asc"
            params["order"] = f"{self._order_by}.{direction}"
        if self._limit is not None:
            params["limit"] = str(self._limit)
        return params

    @staticmethod
    def _format_filter_value(value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        if value is None:
            return "null"
        return str(value)


class RestSupabaseClient:
    def __init__(self, supabase_url: str, api_key: str, timeout_seconds: int = 30):
        try:
            import requests
        except Exception as exc:  # pragma: no cover - import guard
            raise RestSupabaseError("The 'requests' dependency is required for REST Supabase access.") from exc

        self.base_url = supabase_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.mode = "supabase-rest"
        self.session = requests.Session()
        # Ignore workstation proxy env vars here. In this environment they can point to a
        # dead local proxy and silently break otherwise valid Supabase requests.
        self.session.trust_env = False
        self.headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def table(self, table_name: str) -> RestSupabaseTable:
        return RestSupabaseTable(client=self, table_name=table_name)
