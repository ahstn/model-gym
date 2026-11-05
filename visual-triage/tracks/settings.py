"""Track identities keep task semantics independent from experiment versions."""

TRACKS = {
    "quality": {"task": "quality", "arm": "decider", "version": 1, "budget_usd": 2.0},
    "ai_detection": {"task": "ai", "arm": "luna", "version": 1, "budget_usd": 2.0},
    "quality_v2": {"task": "quality", "arm": "decider", "version": 2, "budget_usd": 5.0},
    "ai_detection_v2": {"task": "ai", "arm": "decider", "version": 2, "budget_usd": 5.0},
}


def task_for(track):
    return TRACKS[track]["task"]
