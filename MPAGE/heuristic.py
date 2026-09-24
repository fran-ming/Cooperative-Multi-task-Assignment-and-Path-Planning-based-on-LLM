# -*- coding: utf-8 -*-
"""Structured heuristic representation for MPaGE-CMAPP.

The LLM never emits raw Python here.  It emits a small, deterministic strategy
dictionary that is mapped onto a fixed, pre-implemented operator library (guide
sections 6-9):

    {
        "priority": ["time_window", "synchronization", "load_balance"],
        "construction": "greedy",
        "repair": "sync_repair",
        "local_search": "single_task_ot_reassignment",
        "weights": {...},
    }

A heuristic's ``apply`` returns a list of *seed chromosomes* that bias the
initial population of the inner NSGA-II, never a final solution.
"""

import random
from collections import defaultdict

from nsga.decoder import Decoder, INF

# Cue keys understood by the greedy constructor.
CUE_KEYS = ("time_window", "synchronization", "load_balance", "travel_distance")

DEFAULT_PRIORITY = ["time_window", "synchronization", "travel_distance", "load_balance"]

DEFAULT_WEIGHTS = {
    "time_window": 1.0,
    "synchronization": 1.0,
    "travel_distance": 1.0,
    "load_balance": 1.0,
}

# Weight decay applied when a strategy only gives a ``priority`` list.
_PRIORITY_WEIGHTS = [1.0, 0.7, 0.4, 0.2]


class Heuristic:
    """A meta-level individual: an executable search strategy.

    Solution-level metrics are kept strictly separate from heuristic-level
    objectives (guide section 39): a heuristic is evaluated by *its own*
    ``[hypervolume, feasible_rate, runtime]`` triple after its inner NSGA-II
    run completes.
    """

    def __init__(self, heuristic_id, name, description, strategy, parent_ids=None,
                 cluster_id=None):
        self.heuristic_id = heuristic_id
        self.name = name
        self.description = description
        self.strategy = dict(strategy or {})
        self.parent_ids = list(parent_ids or [])
        self.cluster_id = cluster_id

        # Solution-level summary after running the inner NSGA-II.
        self.solution_objectives = None
        self.solution_metrics = {}
        self.best_solution = None

        # Heuristic-level objectives (Pareto quality, feasible rate, runtime).
        self.objectives = None
        self.metrics = {}
        self.pareto_rank = None
        self.crowding_distance = None
        self.grid_cell = None
        self.reflection = None

    def apply(self, scenario, rng, num_seeds=None):
        """Generate seed chromosomes biased by this heuristic.

        Returns ``List[chromosome]`` (index i is the OT for Task i).  NSGA-II
        fills the rest of its population with random chromosomes.
        """
        if num_seeds is None:
            num_seeds = 12

        base = construct_greedy(scenario, self.strategy, rng)
        seeds = [list(base)]

        improved = local_search(scenario, base, rng, budget=150)
        if improved is not None:
            if improved not in seeds:
                seeds.append(improved)

        attempts = 0
        while len(seeds) < num_seeds and attempts < 4 * num_seeds:
            attempts += 1
            candidate = perturb(scenario, base, rng)
            candidate = repair_chromosome(scenario, candidate, rng)
            if candidate not in seeds:
                seeds.append(candidate)
        return seeds

    def __repr__(self):
        return "Heuristic(%s, %s)" % (self.heuristic_id, self.name)


# --------------------------------------------------------------------------- #
# Operator library (deterministic execution of the structured strategy).
# --------------------------------------------------------------------------- #

def _group_tasks(scenario):
    groups = defaultdict(list)
    for task in scenario.tasks:
        groups[task.sync_group].append(task)
    return groups


def _strategies_weights(strategy):
    """Resolve effective cue weights from priority list / weights dict."""
    weights = strategy.get("weights") or {}
    if weights:
        return {k: float(weights.get(k, 0.0)) for k in CUE_KEYS}
    priority = strategy.get("priority") or DEFAULT_PRIORITY
    resolved = {k: 0.0 for k in CUE_KEYS}
    for idx, cue in enumerate(priority):
        if cue in resolved:
            resolved[cue] = _PRIORITY_WEIGHTS[min(idx, len(_PRIORITY_WEIGHTS) - 1)]
    if all(v == 0.0 for v in resolved.values()):
        return dict(DEFAULT_WEIGHTS)
    return resolved


def _travel_time(scenario, ot_id, position, staging):
    distance = scenario.oracle.travel_distance(position, staging)
    if distance >= INF - 1:
        return INF
    return distance / scenario.ots[ot_id - 1].speed


def _compute_cues(task, ot_id, position, available_time, load, scenario):
    """Return per-cue cost (lower is better) for assigning task -> OT."""
    staging = task.staging
    travel_t = _travel_time(scenario, ot_id, position, staging)
    if travel_t >= INF - 1:
        return {k: 1e9 for k in CUE_KEYS}

    estimated_raw = available_time + travel_t
    a, b = task.tw

    if task.sync_group is not None:
        lateness = max(0.0, estimated_raw - a)
        time_window = lateness / max(a, b - a, 1.0)
    else:
        lateness = max(0.0, estimated_raw - b)
        time_window = lateness / max(b, 1.0)

    if task.sync_group is not None:
        gap = max(0.0, estimated_raw - a)
        synchronization = gap / max(a, 1.0)
    else:
        synchronization = 0.0

    distance = scenario.oracle.travel_distance(position, staging)
    min_dist = min(
        (scenario.oracle.travel_distance(ot.initial_position, staging)
         for ot in scenario.ots),
        default=1.0,
    )
    travel_distance = distance / max(1.0, min_dist)

    load_balance = load / max(1, len(scenario.tasks))

    return {
        "time_window": time_window,
        "synchronization": synchronization,
        "travel_distance": travel_distance,
        "load_balance": load_balance,
    }


def construct_greedy(scenario, strategy=None, rng=None):
    """Greedy constructor (guide H1-H5, section 9).

    Processes tasks in decode order and assigns each task the OT with the
    lowest weighted cost.  Sync groups use distinct OTs (paper constraint 16).
    """
    strategy = dict(strategy or {})
    rng = rng if rng is not None else random.Random(0)
    weights = _strategies_weights(strategy)

    num_tasks = len(scenario.tasks)
    valid = scenario.valid_ot_ids
    chromosome = [None] * num_tasks

    used_in_group = defaultdict(set)
    load = {ot_id: 0 for ot_id in valid}
    ot_pos = {ot_id: scenario.ots[ot_id - 1].initial_position for ot_id in valid}
    ot_avail = {ot_id: scenario.ots[ot_id - 1].initial_available_time for ot_id in valid}

    for idx, task in enumerate(scenario.tasks):
        group = task.sync_group
        candidates = []
        for ot_id in valid:
            if group is not None and ot_id in used_in_group[group]:
                continue
            cues = _compute_cues(
                task, ot_id, ot_pos[ot_id], ot_avail[ot_id], load[ot_id], scenario
            )
            score = sum(weights.get(k, 0.0) * cues.get(k, 0.0) for k in CUE_KEYS)
            candidates.append((score, ot_id, cues))
        candidates.sort(key=lambda t: t[0])
        chosen = candidates[0][1] if candidates else rng.choice(valid)

        chromosome[idx] = chosen
        if group is not None:
            used_in_group[group].add(chosen)
        load[chosen] += 1

        travel_t = _travel_time(scenario, chosen, ot_pos[chosen], task.staging)
        if travel_t < INF - 1:
            arrival = max(ot_avail[chosen] + travel_t, task.tw[0])
            ot_pos[chosen] = task.terminal
            ot_avail[chosen] = arrival + task.duration

    return chromosome


def repair_chromosome(scenario, chromosome, rng):
    valid = scenario.valid_ot_ids
    return [c if c in valid else rng.choice(valid) for c in chromosome]


def perturb(scenario, chromosome, rng, flips=2):
    valid = scenario.valid_ot_ids
    child = list(chromosome)
    for _ in range(flips):
        i = rng.randrange(len(child))
        child[i] = rng.choice(valid)
    return child


def local_search(scenario, chromosome, rng, budget=150):
    """Single-task OT-reassignment hill-climb (guide repair operator).

    Minimizes J_d first, then scalar fitness.  The deterministic decoder is the
    ground-truth evaluator (guide section 37).  Returns an improved chromosome
    or None.
    """
    decoder = Decoder(scenario)
    valid = scenario.valid_ot_ids
    n = len(scenario.tasks)

    current = list(chromosome)
    result = decoder.decode(current)
    best_jd = result.metrics["J_d"]
    best_fit = result.metrics["fitness_J"]

    for _ in range(budget):
        i = rng.randrange(n)
        old = current[i]
        new = rng.choice(valid)
        if new == old:
            continue
        current[i] = new
        candidate = decoder.decode(current)
        jd = candidate.metrics["J_d"]
        fit = candidate.metrics["fitness_J"]
        if jd < best_jd or (jd == best_jd and fit < best_fit):
            best_jd, best_fit = jd, fit
        else:
            current[i] = old

    baseline = decoder.decode(chromosome)
    if best_jd < baseline.metrics["J_d"] or best_fit < baseline.metrics["fitness_J"]:
        return current
    return None


# --------------------------------------------------------------------------- #
# Preset heuristic pool (guide section 9: H1-H5).
# --------------------------------------------------------------------------- #

PRESET_HEURISTICS = [
    {
        "heuristic_id": "H1",
        "name": "TimeWindowFirst",
        "description": "Prioritize tasks with the tightest time-window slack and assign "
                       "each task to the OT minimizing deadline violation.",
        "strategy": {
            "priority": ["time_window", "travel_distance"],
            "construction": "greedy",
            "repair": "sync_repair",
            "local_search": "single_task_ot_reassignment",
            "weights": {"time_window": 1.0, "travel_distance": 0.2},
        },
    },
    {
        "heuristic_id": "H2",
        "name": "SynchronizationFirst",
        "description": "Prioritize sync-group consistency: one distinct OT per group "
                       "member, clustered around the group staging area.",
        "strategy": {
            "priority": ["synchronization", "time_window", "travel_distance"],
            "construction": "greedy",
            "repair": "sync_repair",
            "local_search": "single_task_ot_reassignment",
            "weights": {"synchronization": 1.0, "time_window": 0.6, "travel_distance": 0.3},
        },
    },
    {
        "heuristic_id": "H3",
        "name": "LoadBalanceFirst",
        "description": "Assign tasks to the OT with the lowest current workload.",
        "strategy": {
            "priority": ["load_balance", "time_window"],
            "construction": "greedy",
            "repair": "sync_repair",
            "local_search": "single_task_ot_reassignment",
            "weights": {"load_balance": 1.0, "time_window": 0.4},
        },
    },
    {
        "heuristic_id": "H4",
        "name": "TravelDistanceFirst",
        "description": "Assign each task to the OT with the shortest travel to staging.",
        "strategy": {
            "priority": ["travel_distance", "time_window"],
            "construction": "greedy",
            "repair": "sync_repair",
            "local_search": "single_task_ot_reassignment",
            "weights": {"travel_distance": 1.0, "time_window": 0.4},
        },
    },
    {
        "heuristic_id": "H5",
        "name": "Hybrid",
        "description": "Hybrid of time-window, synchronization, travel distance and load balance.",
        "strategy": {
            "priority": ["time_window", "synchronization", "travel_distance", "load_balance"],
            "construction": "greedy",
            "repair": "sync_repair",
            "local_search": "single_task_ot_reassignment",
            "weights": dict(DEFAULT_WEIGHTS),
        },
    },
]


def build_presets():
    """Instantiate the five preset heuristics from the structured specs."""
    pool = []
    for spec in PRESET_HEURISTICS:
        pool.append(Heuristic(
            heuristic_id=spec["heuristic_id"],
            name=spec["name"],
            description=spec["description"],
            strategy=spec["strategy"],
        ))
    return pool