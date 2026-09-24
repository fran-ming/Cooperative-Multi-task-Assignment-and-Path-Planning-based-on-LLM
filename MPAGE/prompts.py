# -*- coding: utf-8 -*-
"""MPaGE prompts (guide section 19).

MPaGE only prompts heuristic generation / reflection / clustering / mutation /
crossover.  It reuses ``llm.client.LLMClient`` for the actual call and
``llm.prompts.parse_json_response`` for JSON parsing.
"""

import json

from llm.prompts import parse_json_response as parse_heuristic_json

MPAGE_SYSTEM = """You are designing search heuristics for a constrained cooperative multi-task assignment and path-planning problem (CMAPP).

The CMAPP problem has:
1. fixed task sequence;
2. OT assignment;
3. VUT-triggered time windows;
4. synchronization constraints;
5. road-network travel costs;
6. workload imbalance;
7. four solution-level objectives: Jd (failed tasks), Jm (makespan), Jb (workload imbalance), Jt (total travel).

You do NOT directly solve the CMAPP instance. Your role is to design a search heuristic that generates candidate OT assignments for NSGA-II to refine.

You must not: change the task order, create/remove tasks, create/remove OTs, change task attributes, change objective definitions, or bypass the decoder.

A heuristic is ONLY a structured strategy, for example:
{
  "priority": ["time_window", "synchronization", "load_balance"],
  "construction": "greedy",
  "repair": "sync_repair",
  "local_search": "single_task_ot_reassignment"
}

Allowed priority cues: time_window, synchronization, travel_distance, load_balance.

Return only the requested JSON schema.
"""


def _heuristic_view(h):
    return {
        "heuristic_id": h.heuristic_id,
        "name": h.name,
        "strategy": h.strategy,
        "metrics": h.metrics,
        "reflection": h.reflection,
    }


def generation_user(scenario, count):
    return json.dumps({
        "task": "generate heuristics",
        "count": count,
        "num_tasks": len(scenario.tasks),
        "num_ots": len(scenario.ots),
        "valid_ot_ids": scenario.valid_ot_ids,
    }, ensure_ascii=False)


def reflection_user(heuristic):
    return json.dumps({
        "task": "reflect",
        "heuristic": _heuristic_view(heuristic),
    }, ensure_ascii=False)


def clustering_user(heuristics, n_clusters):
    return json.dumps({
        "task": "cluster",
        "n_clusters": n_clusters,
        "heuristics": [_heuristic_view(h) for h in heuristics],
    }, ensure_ascii=False)


def mutation_user(heuristic):
    return json.dumps({
        "task": "mutate",
        "heuristic": _heuristic_view(heuristic),
    }, ensure_ascii=False)


def crossover_user(a, b):
    return json.dumps({
        "task": "crossover",
        "heuristic_a": _heuristic_view(a),
        "heuristic_b": _heuristic_view(b),
    }, ensure_ascii=False)