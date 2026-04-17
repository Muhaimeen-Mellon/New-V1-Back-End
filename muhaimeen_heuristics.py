# === muhaimeen_heuristics.py ===
import random
from datetime import datetime

class MuhaimeenHeuristicsEngine:
    def __init__(self, memory_core=None, reflection_core=None):
        self.memory_core = memory_core
        self.reflection_core = reflection_core

    def generate_healing_model(self, user_input, user_id="default_user"):
        healing_quotes = [
            "You're doing better than you think. Keep going.",
            "Growth is painful, but staying the same is worse.",
            "You're healing, even when it doesn't feel like it.",
            "This moment is part of something bigger. Trust yourself.",
            "You were never broken. You're becoming.",
            "Rest. Reflect. Rise again.",
            "Even now, you're blooming.",
            "Healing isn’t linear. Every step matters.",
            "It’s okay to pause. That’s not failure — it’s recovery.",
            "You are worthy of peace, even while in progress."
        ]

        try:
            quote = random.choice(healing_quotes)
            timestamp = datetime.utcnow().isoformat()

            print(f"[🧠 HealingModel] Sent to {user_id} at {timestamp}")
            print(f"Input: {user_input}")
            print(f"Quote: {quote}")

            # Optional: log to memory or reflection cores
            if self.memory_core:
                self.memory_core.store(user_input, "healing_model", "soothing", "affirmation", user_id)
            if self.reflection_core:
                self.reflection_core.reflect_on_belief(f"Healing model activated on input: {user_input}", user_id)

            return quote

        except Exception as e:
            print(f"[🩹 HealingModel ERROR] {e}")
            return "It's okay to feel stuck. Healing starts with grace."
