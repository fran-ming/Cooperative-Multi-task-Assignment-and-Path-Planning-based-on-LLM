# -*- coding: utf-8 -*-
"""Meta-level mutation and crossover operators (guide section 17).

Same-cluster mutation and cross-cluster crossover.  LLM-generated strategies
are parsed and schema-validated; any failure falls back to deterministic
transformations so the experiment never dies (guide section 43).
"""

import copy
import random

from MPAGE.heuristic import CUE_KEYS, Heuristic

_CUES = list(CUE_KEYS)


def _validate_strategy(strategy):
    """Coerce an LLM-provided strategy to the allowed schema."""
    if not isinstance(strategy, dict):
        return None
    priority = strategy.get("priority")
    if isinstance(priority, list):
        priority = [c for c in priority if c in _CUES]
        if not priority:
            return None
    else:
        priority = None
    construction = strategy.get("construction", "greedy")
    if construction != "greedy":
        construction = "greedy"
    repair = strategy.get("repair", "sync_repair")
    weights = strategy.get("weights")
    if not isinstance(weights, dict):
        weights = None
    return {
        "priority": priority,
        "construction": construction,
        "repair": repair,
        "weights": weights,
    }


def _det_mutate_strategy(strategy, rng):
    """Deterministic mutation: rotate or insert a cue into the priority list."""
    new = copy.deepcopy(strategy)
    priority = [c for c in (new.get("priority") or _CUES) if c in _CUES]
    if rng.random() < 0.5 and len(priority) > 1:
        priority = priority[1:] + priority[:1]
    else:
        missing = [c for c in _CUES if c not in priority]
        if missing:
            priority.append(rng.choice(missing))
    new["priority"] = list(_CUES)  # keep order canonical
    # Reorder so the mutated list leads.
    new["priority"] = priority + [c for c in _CUES if c not in priority]
    return new


def _det_crossover_strategy(a_strategy, b_strategy, rng):
    """Deterministic crossover: interleave the two priority orderings."""
    pa = [c for c in (a_strategy.get("priority") or _CUES) if c in _CUES]
    pb = [c for c in (b_strategy.get("priority") or _CUES) if c in _CUES]
    merged = []
    for pair in zip(pa, pb):
        for c in pair:
            if c not in merged:
                merged.append(c)
    for c in _CUES:
        if c not in merged:
            merged.append(c)
    return {
        "priority": merged,
        "construction": a_strategy.get("construction", "greedy"),
        "repair": a_strategy.get("repair", "sync_repair"),
        "weights": b_strategy.get("weights") or a_strategy.get("weights"),
    }


def mutate_heuristic(heuristic, scenario=None, client=None, rng=None):
    rng = rng or random.Random(0)
    if client is not None:
        try:
            from MPAGE.prompts import mutation_user, parse_heuristic_json
            response = client.generate("", mutation_user(heuristic))
            payload = parse_heuristic_json(response)
            strategy = _validate_strategy(payload)
            if strategy:
                return Heuristic(
                    heuristic_id="M%d_%s" % (rng.randrange(100000), heuristic.heuristic_id),
                    name=payload.get("name", "Mut-" + heuristic.name),
                    description=payload.get("description", "LLM-mutated from " + heuristic.name),
                    strategy=strategy,
                    parent_ids=[heuristic.heuristic_id],
                )
        except Exception:
            pass
    strategy = _det_mutate_strategy(heuristic.strategy, rng)
    return Heuristic(
        heuristic_id="M%d_%s" % (rng.randrange(100000), heuristic.heuristic_id),
        name="Mut-" + heuristic.name,
        description="Deterministic mutation of " + heuristic.name,
        strategy=strategy,
        parent_ids=[heuristic.heuristic_id],
    )


def crossover_heuristics(a, b, scenario=None, client=None, rng=None):
    rng = rng or random.Random(0)
    if client is not None:
        try:
            from MPAGE.prompts import crossover_user, parse_heuristic_json
            response = client.generate("", crossover_user(a, b))
            payload = parse_heuristic_json(response)
            strategy = _validate_strategy(payload)
            if strategy:
                return Heuristic(
                    heuristic_id="X%d_%s_%s" % (rng.randrange(100000), a.heuristic_id, b.heuristic_id),
                    name=payload.get("name", "Cross-" + a.name + "-" + b.name),
                    description=payload.get("description", "LLM crossover of %s and %s" % (a.name, b.name)),
                    strategy=strategy,
                    parent_ids=[a.heuristic_id, b.heuristic_id],
                )
        except Exception:
            pass
    strategy = _det_crossover_strategy(a.strategy, b.strategy, rng)
    return Heuristic(
        heuristic_id="X%d_%s_%s" % (rng.randrange(100000), a.heuristic_id, b.heuristic_id),
        name="Cross-" + a.name + "-" + b.name,
        description="Deterministic crossover of %s and %s" % (a.name, b.name),
        strategy=strategy,
        parent_ids=[a.heuristic_id, b.heuristic_id],
    )