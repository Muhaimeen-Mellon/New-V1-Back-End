# === emotion_core.py ===

import re
from collections import defaultdict
from typing import Dict, Tuple

class EmotionCore:
    def __init__(self):
        # 🎯 Weighted keyword list for GPT-4 Turbo tier analysis
        self.emotion_map = {
            "fear": {
                "keywords": ["danger", "panic", "afraid", "threat", "terrified", "vulnerable"],
                "weight": 1.2
            },
            "hope": {
                "keywords": ["hope", "dream", "growth", "faith", "light", "possibility"],
                "weight": 1.1
            },
            "sadness": {
                "keywords": ["sad", "grief", "lost", "lonely", "regret", "numb", "empty"],
                "weight": 1.3
            },
            "joy": {
                "keywords": ["happy", "grateful", "love", "joy", "excited", "peace"],
                "weight": 1.0
            },
            "anger": {
                "keywords": ["angry", "mad", "furious", "rage", "resent", "irritated"],
                "weight": 1.4
            },
            "curiosity": {
                "keywords": ["why", "how", "wonder", "mystery", "explore", "unknown"],
                "weight": 0.9
            }
        }

        self.emojis = {
            "fear": "⚠️",
            "hope": "🌱",
            "sadness": "😢",
            "joy": "😊",
            "anger": "😠",
            "curiosity": "🧩",
            "neutral": "🧠"
        }

    def analyze(self, text: str) -> str:
        dominant, _ = self._score_emotions(text)
        emoji = self.emojis.get(dominant, "🧠")
        return f"[{emoji} {dominant.capitalize()}] {text}"

    def get_dominant_emotion(self, text: str) -> str:
        dominant, _ = self._score_emotions(text)
        return dominant

    def get_emotion_distribution(self, text: str) -> Dict[str, float]:
        _, scores = self._score_emotions(text)
        total = sum(scores.values()) or 1
        return {k: round(v / total, 3) for k, v in scores.items()}

    def _score_emotions(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        Calculates weighted scores for emotions and returns dominant emotion and full map.
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        scores = defaultdict(float)

        for word in words:
            for emotion, data in self.emotion_map.items():
                if word in data["keywords"]:
                    scores[emotion] += data["weight"]

        if not scores:
            return "neutral", {}

        dominant = max(scores.items(), key=lambda x: x[1])[0]
        return dominant, dict(scores)
