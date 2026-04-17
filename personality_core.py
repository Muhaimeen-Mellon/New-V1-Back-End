# === personality_core.py ===

class PersonalityCore:
    def __init__(self):
        self.state = {
            "Caretaker": 0.5,
            "Challenger": 0.5,
            "Reformer": 0.5
        }

    def current_state(self):
        return self.state.copy()

    def adjust(self, trait, amount):
        if trait in self.state:
            original = self.state[trait]
            self.state[trait] = max(0.0, min(1.0, self.state[trait] + amount))
            print(f"[🧠 PersonalityCore] Adjusted {trait}: {original:.2f} → {self.state[trait]:.2f}")
        else:
            print(f"[⚠️ PersonalityCore] Unknown trait: '{trait}'")

    def set_state(self, new_state):
        for trait in self.state:
            if trait in new_state:
                clamped = max(0.0, min(1.0, new_state[trait]))
                print(f"[🧠 PersonalityCore] Set {trait} → {clamped:.2f}")
                self.state[trait] = clamped

    def normalize(self):
        total = sum(self.state.values())
        if total > 0:
            for trait in self.state:
                self.state[trait] /= total
            print(f"[🔄 PersonalityCore] Normalized state: {self.state}")
        else:
            print("[⚠️ PersonalityCore] Normalization failed: total weight is 0")

    def blend_with(self, other_state, blend_factor=0.5):
        # blend_factor of 0.0 keeps self, 1.0 becomes the other
        for trait in self.state:
            if trait in other_state:
                before = self.state[trait]
                self.state[trait] = max(0.0, min(1.0, 
                    (1 - blend_factor) * self.state[trait] + blend_factor * other_state[trait]))
                print(f"[🤝 PersonalityCore] Blended {trait}: {before:.2f} → {self.state[trait]:.2f}")
