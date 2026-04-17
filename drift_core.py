from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Optional

from runtime_config import get_supabase_client

logger = logging.getLogger(__name__)

EXPERIMENTAL = True


class DriftCore:
    def __init__(self, personality_core=None, supabase_client: Optional[Any] = None):
        self.personality_core = personality_core
        self.supabase = supabase_client or get_supabase_client()
        self.tracked_traits = ["Caretaker", "Challenger", "Reformer"]

    def log_drift(
        self,
        event_type: str,
        user_input: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.personality_core:
            logger.warning("DriftCore is missing a personality_core instance.")
            return

        try:
            traits = self.personality_core.current_state()
            drift_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "event": event_type,
                "input": user_input,
            }

            for trait in self.tracked_traits:
                drift_entry[trait.lower()] = round(traits.get(trait, 0), 3)

            if metadata:
                drift_entry.update(metadata)

            self.supabase.table("personality_drift").insert(drift_entry).execute()
            logger.info("Logged drift event '%s'.", event_type)

        except Exception as exc:
            logger.exception("Failed to log drift event '%s': %s", event_type, exc)

    def log_if_changed(self, new_state: Dict[str, float], threshold: float = 0.05) -> None:
        if not self.personality_core:
            return

        current = self.personality_core.current_state()
        changes = {
            trait: new_state[trait]
            for trait in self.tracked_traits
            if abs(new_state.get(trait, 0) - current.get(trait, 0)) >= threshold
        }

        if changes:
            self.log_drift(
                event_type="trait_shift",
                user_input="Auto: Personality drift detected",
                metadata={"changes": changes},
            )
