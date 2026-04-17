# === brevity_core.py ===

import re
from typing import Literal

class BrevityCore:
    def __init__(self):
        # Configurable keyword banks for tone hints
        self.verbosity_keywords = {
            "short": [
                "keep it short", "tl;dr", "summarize", "in a nutshell", 
                "2 sentences", "one-liner", "quick version", "say it faster"
            ],
            "detailed": [
                "explain deeply", "go deep", "expand", "elaborate", 
                "give more", "long form", "unpack it", "full breakdown"
            ]
        }

    def detect_verbosity_hint(self, message: str) -> Literal["short", "detailed", "default"]:
        """
        Detects the user's preferred response length based on keywords.
        """
        lowered = message.lower()

        for kw in self.verbosity_keywords["short"]:
            if kw in lowered:
                return "short"

        for kw in self.verbosity_keywords["detailed"]:
            if kw in lowered:
                return "detailed"

        return "default"

    def apply_brevity(self, response: str, level: Literal["short", "default", "detailed"]) -> str:
        """
        Trims or expands response based on brevity level.
        """
        if not response:
            return ""

        # Clean and split response into logical sentences
        sentences = re.split(r'(?<=[.!?])\s+', response.strip())
        if not sentences:
            return response.strip()

        if level == "short":
            return " ".join(sentences[:2]).strip()
        elif level == "default":
            return " ".join(sentences[:5]).strip()
        elif level == "detailed":
            return response.strip()

        return response.strip()
