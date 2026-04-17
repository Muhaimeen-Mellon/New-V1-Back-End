from __future__ import annotations

import logging
import os
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from runtime_config import get_supabase_client
from safe_evolution_core import SafeEvolutionCore

logger = logging.getLogger(__name__)


class DeveloperCore:
    def __init__(
        self,
        codex,
        reflection_core,
        dream_core,
        oath_core,
        raven_core,
        memory_core,
        config: Optional[Dict[str, Any]] = None,
        supabase_client: Optional[Any] = None,
    ):
        self._validate_dependencies(codex, reflection_core, dream_core, oath_core, raven_core, memory_core)

        self.config = config or {}
        self.codex = codex
        self.reflection_core = reflection_core
        self.dream_core = dream_core
        self.oath_core = oath_core
        self.raven_core = raven_core
        self.memory_core = memory_core
        self.client = supabase_client or get_supabase_client()

        self.safeguard = SafeEvolutionCore(
            codex=codex,
            reflection_core=reflection_core,
            dream_core=dream_core,
            developer_core=self,
            oath_core=oath_core,
            raven_core=raven_core,
            memory_core=memory_core,
        )

        self._init_file_system()

    def _init_file_system(self) -> None:
        self.base_dir = os.path.abspath(self.config.get("base_dir", "."))
        self.backup_dir = os.path.join(self.base_dir, "backups")
        self.max_backups = self.config.get("max_backups", 5)
        os.makedirs(self.backup_dir, exist_ok=True)

    def _validate_dependencies(self, *args) -> None:
        for module in args:
            if module is None:
                raise ValueError("Missing required core dependency")

    def _get_full_path(self, file: str) -> str:
        return os.path.join(self.base_dir, file)

    def apply_patch(
        self,
        target_file: str,
        function_name: str,
        new_code: str,
        patch_reason: str,
        user_id: str,
    ) -> Dict[str, Any]:
        response: Dict[str, Any] = {"success": False, "steps": {}, "warnings": [], "errors": []}
        backup_path: Optional[str] = None

        try:
            valid, errors = self._validate_patch(target_file, function_name, new_code, patch_reason, user_id)
            if not valid:
                response["errors"] = errors
                return response

            backup_path = self._create_backup(target_file)
            if not backup_path:
                response["errors"].append("backup_failed")
                return response

            applied, apply_errors = self._apply_code_changes(target_file, new_code)
            if not applied:
                response["errors"].extend(apply_errors)
                return response

            self._log_patch(target_file, function_name, patch_reason, user_id)
            self._trigger_post_patch_processes(patch_reason, function_name, user_id)

            response["success"] = True
            response["steps"] = {
                "validated": "yes",
                "backup": backup_path,
                "applied": "success",
                "logged": "codex + drift",
                "dream": "triggered",
            }
            return response

        except Exception:
            logger.error("[CRITICAL PATCH ERROR] %s", traceback.format_exc())
            response["errors"].append("critical_failure")
            self._rollback_operations(target_file, backup_path)
            return response

    def _validate_patch(
        self,
        target_file: str,
        function_name: str,
        code: str,
        reason: str,
        user_id: str,
    ) -> Tuple[bool, list]:
        del function_name, user_id
        errors = []

        if not self._validate_file_path(target_file):
            errors.append("invalid_path")

        oath_ok, oath_issues = self.oath_core.validate_patch(reason)
        if not oath_ok:
            errors.extend([f"oath:{issue}" for issue in oath_issues])

        raven_ok, raven_issues = self.raven_core.evaluate_patch_stability(reason)
        if not raven_ok:
            errors.extend([f"raven:{issue}" for issue in raven_issues])

        if not self.safeguard.guard_patch_integrity(target_file, reason, code):
            errors.append("safeguard:rejected")

        return (len(errors) == 0, errors)

    def _validate_file_path(self, file: str) -> bool:
        return not any(segment in file for segment in ["..", "~", "//"]) and file.endswith(".py")

    def _create_backup(self, file: str) -> Optional[str]:
        try:
            full_path = self._get_full_path(file)
            with open(full_path, "r", encoding="utf-8") as handle:
                content = handle.read()

            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
            backup_name = f"{file}.{timestamp}.bak"
            backup_path = os.path.join(self.backup_dir, backup_name)

            with open(backup_path, "w", encoding="utf-8") as handle:
                handle.write(content)

            self._rotate_backups(file)
            return backup_path
        except Exception as exc:
            logger.error("[BACKUP ERROR] %s", exc)
            return None

    def _rotate_backups(self, file: str) -> None:
        backups = sorted(entry for entry in os.listdir(self.backup_dir) if entry.startswith(file))
        while len(backups) > self.max_backups:
            oldest = os.path.join(self.backup_dir, backups.pop(0))
            os.remove(oldest)

    def _apply_code_changes(self, file: str, code: str) -> Tuple[bool, list]:
        try:
            temp = self._get_full_path(file) + ".tmp"
            final = self._get_full_path(file)
            with open(temp, "w", encoding="utf-8") as handle:
                handle.write(code)
            os.replace(temp, final)
            return (True, [])
        except Exception as exc:
            logger.error("[PATCH APPLY ERROR] %s", exc)
            return (False, ["write_failed"])

    def _log_patch(self, file: str, function_name: str, reason: str, user_id: str) -> None:
        try:
            self.client.table("codex_entries").insert(
                {
                    "user_id": user_id,
                    "target_file": file,
                    "function_name": function_name,
                    "patch_reason": reason,
                    "system_version": os.getenv("SYSTEM_VERSION", "0.0.1"),
                    "created_at": datetime.utcnow().isoformat(),
                }
            ).execute()

            self.client.table("codex_drift").insert(
                {
                    "user_id": user_id,
                    "belief_before": "unknown",
                    "belief_after": f"Patched {function_name} in {file}",
                    "cause": reason,
                    "drift_description": f"System modified: {function_name}",
                    "drift_tags": ["patch", "self_mod"],
                }
            ).execute()
        except Exception as exc:
            logger.warning("[LOGGING ERROR] %s", exc)

    def _trigger_post_patch_processes(self, reason: str, function_name: str, user_id: str) -> None:
        try:
            self.dream_core.seed_dream(
                input_text=reason,
                tag="code_patch",
                user_id=user_id,
                summary=f"Patched function: {function_name}",
                interpretation="Dream triggered by code self-modification.",
            )
        except Exception as exc:
            logger.warning("[DREAM ERROR] %s", exc)

    def _rollback_operations(self, file: str, backup: Optional[str]) -> None:
        if backup and os.path.exists(backup):
            try:
                os.replace(backup, self._get_full_path(file))
                logger.warning("[ROLLBACK] Reverted file from backup.")
            except Exception as exc:
                logger.error("[ROLLBACK FAILED] %s", exc)
