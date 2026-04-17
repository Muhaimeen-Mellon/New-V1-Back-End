# === tone_core.py ===

import random
import re


class ToneCore:
    def __init__(self):
        self.tone_rules = {
            "genz": {"max_length": 100, "slang_strength": 0.2, "truncate": False},
            "casual": {"max_length": 120, "slang_strength": 0.1, "truncate": True},
            "poetic": {"max_length": 1000, "slang_strength": 0.0, "truncate": False},
            "sarcastic": {"max_length": 60, "slang_strength": 0.0, "truncate": True},
            "direct": {"max_length": 40, "slang_strength": 0.0, "truncate": True},
            "neutral": {"max_length": 200, "slang_strength": 0.0, "truncate": False},
        }

        self.sarcastic_responses = [
            "Oh wow, yeah sure - that's totally new.",
            "Because that's never happened before, right?",
            "Genius. Truly revolutionary.",
            "Okay clown mode activated.",
        ]

    def detect_tone(self, message):
        msg = message.lower()

        if self._contains_any_token(
            msg,
            ["yo", "fr", "bet", "lowkey", "highkey", "vibe", "cap", "lit", "sus", "cooked", "dead", "slay", "valid"],
        ):
            return "genz"
        if "!" in msg or ":)" in msg or self._contains_any_token(msg, ["lol", "bro", "haha", "nah"]):
            return "casual"
        if self._contains_any_token(msg, ["alas", "thou", "beyond", "soul", "eternal", "echo", "whisper"]):
            return "poetic"
        if self._contains_any_phrase(msg, ["yeah right", "of course", "obviously", "as if", "wow great", "sure jan"]):
            return "sarcastic"
        if "?" in msg and len(msg.split()) <= 4:
            return "direct"

        return "neutral"

    def adjust_response(self, response, tone):
        rules = self.tone_rules.get(tone, self.tone_rules["neutral"])
        response = response.strip()

        if rules["truncate"]:
            sentences = re.split(r"(?<=[.!?]) +", response)
            response = " ".join(sentences[:2]).strip()

        if tone == "genz":
            return self._as_genz(response, rules)
        if tone == "casual":
            return f"Real talk - {response}"
        if tone == "poetic":
            return self._as_poetic(response)
        if tone == "sarcastic":
            return random.choice(self.sarcastic_responses)
        if tone == "direct":
            return f"OK: {response.split('.')[0]}." if "." in response else f"OK: {response}"

        return response

    def _as_genz(self, text, rules):
        slang_map = {
            r"\bperhaps\b": "prolly",
            r"\bI am\b": "I'm lowkey",
            r"\bvery\b": "mad",
            r"\bgood\b": "fire",
            r"\btired\b": "cooked",
            r"\binteresting\b": "wild",
            r"\bamazing\b": "goated",
        }

        for pattern, replacement in slang_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return f"Yo fr - {text[:rules['max_length']]}"

    def _as_poetic(self, text):
        poetic = text.replace("you", "thou").replace("You", "Thou")
        poetic = poetic.replace("I ", "Lo, I ")
        return f"Moonlit thought: {poetic.strip()}..."

    def _contains_any_token(self, text, tokens):
        return any(re.search(rf"\b{re.escape(token)}\b", text) for token in tokens)

    def _contains_any_phrase(self, text, phrases):
        return any(phrase in text for phrase in phrases)
