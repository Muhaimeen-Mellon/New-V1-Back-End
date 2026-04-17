from __future__ import annotations

import logging
import time
from typing import Optional

from memory_core import MemoryCore
from recursive_learning_engine import RecursiveLearningEngine

logger = logging.getLogger(__name__)

EXPERIMENTAL = True

TOPICS = [
    "Cell biology",
    "Photosynthesis",
    "Black holes",
    "Artificial Intelligence",
    "Climate change",
    "Quantum mechanics",
    "Emotional intelligence",
    "Cognitive bias",
    "Neural networks",
    "Philosophy of mind",
    "String theory",
    "History of science",
    "Evolutionary psychology",
    "Game theory",
    "Existential risk",
]


def run_batch_learning(
    memory_core: Optional[MemoryCore] = None,
    recursive_engine: Optional[RecursiveLearningEngine] = None,
):
    """
    Experimental batch learner.
    Intentionally lazy so importing this module never boots external clients.
    """

    memory_core = memory_core or MemoryCore()
    recursive_engine = recursive_engine or RecursiveLearningEngine(memory_core)

    logger.info("Starting batch learning across %s topics.", len(TOPICS))
    learned = []

    for topic in TOPICS:
        try:
            result = recursive_engine.learn(topic)
            if result.get("final_belief"):
                learned.append(topic)
                logger.info(
                    "Learned topic '%s' with confidence %.2f.",
                    topic,
                    result["confidence"],
                )
            else:
                logger.warning("Skipped topic '%s': no valid belief generated.", topic)
        except Exception as exc:
            logger.exception("Batch learning failed on topic '%s': %s", topic, exc)

        time.sleep(1)

    logger.info("Batch learning complete: %s/%s topics learned.", len(learned), len(TOPICS))
    return learned
