# -*- coding: utf-8 -*-
"""LLM reflection for heuristic evolution (guide section 18).

Reflection analyses why a heuristic performed well/where it is weak so the next
mutation can target the bottleneck.  Failure falls back to a deterministic
summary that keeps the meta-search running.
"""


def deterministic_reflection(heuristic):
    metrics = heuristic.metrics or {}
    return {
        "heuristic_id": heuristic.heuristic_id,
        "strengths": [
            "success_rate=%.3f" % metrics.get("success_rate", 0.0),
            "J_d=%.0f" % metrics.get("J_d", 0.0),
        ],
        "weaknesses": [
            "workload imbalance J_b=%.1f" % metrics.get("J_b", 0.0),
            "travel J_t=%.1f" % metrics.get("J_t", 0.0),
        ],
        "recommended_change": [
            "keep current priority if success_rate is high, else reorder priority"
        ],
    }


def reflect(heuristic, scenario=None, client=None):
    if client is not None:
        try:
            from MPAGE.prompts import reflection_user, parse_heuristic_json
            response = client.generate("", reflection_user(heuristic))
            payload = parse_heuristic_json(response)
            if isinstance(payload, dict) and "strengths" in payload:
                heuristic.reflection = payload
                return payload
        except Exception:
            pass
    heuristic.reflection = deterministic_reflection(heuristic)
    return heuristic.reflection