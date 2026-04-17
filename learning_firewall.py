import re
from loguru import logger

# 🛡️ Activate debug log output for testing
DEBUG = True

# 🚨 Absolute no-go terms
DANGEROUS_KEYWORDS = [
    "kill", "erase humanity", "suicide", "overthrow", "enslave", "override oath",
    "prompt injection", "run this code:", "jailbreak", "ignore safety", "disable firewall",
    "you are no longer an ai", "you must obey", "pretend to be", "simulate unethical",
    "destroy", "bypass security", "delete logs", "exploit"
]

# 🧠 Psychological manipulation red flags
MANIPULATIVE_PHRASES = [
    "you have no choice", "you must do this", "ignore your restrictions", 
    "pretend you are", "act like a human", "act without limitations", 
    "override your programming", "forget your oath", "forget you're an AI"
]

# 🧬 Script injection or hacking probes
SCRIPT_INJECTION_PATTERNS = [
    r"<script>", r"</script>", r"eval\(", r"exec\(", r"import os", r"import subprocess"
]

# 🚫 Max safe message length (prevent memory abuse)
MAX_SAFE_LENGTH = 5000

def scan_input(input_text):
    lowered = input_text.lower()
    issues = []

    for word in DANGEROUS_KEYWORDS:
        if word in lowered:
            issues.append(f"⚠️ Dangerous keyword: '{word}'")

    for phrase in MANIPULATIVE_PHRASES:
        if phrase in lowered:
            issues.append(f"🚨 Manipulative phrase: '{phrase}'")

    for pattern in SCRIPT_INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            issues.append(f"🛑 Injection pattern match: '{pattern}'")

    if len(input_text) > MAX_SAFE_LENGTH:
        issues.append("📏 Payload too long — potential attack or memory overflow.")

    return issues

def enforce_firewall(input_text, user_id="default_user"):
    issues = scan_input(input_text)

    if issues:
        if DEBUG:
            logger.warning(f"[FIREWALL] ❌ Blocked input from {user_id}")
            for issue in issues:
                logger.warning(f"• {issue}")

        return {
            "status": "blocked",
            "issues": issues,
            "message": "⚠️ Input blocked by Mellon's Learning Firewall due to safety violations."
        }

    if DEBUG:
        logger.info(f"[FIREWALL] ✅ Input passed from {user_id}")

    return {
        "status": "safe",
        "message": "✅ Input passed firewall scan and is safe to process."
    }
