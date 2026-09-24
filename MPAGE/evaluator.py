# -*- coding: utf-8 -*-
"""Heuristic evaluation: run one inner NSGA-II per heuristic (guide section 11).

A heuristic is evaluated by the *heuristic-level* metrics [HV, feasible_rate,
runtime], never directly by a single CMAPP solution (sections 12, 38-40).
"""

import random
import time

from nsga.decoder import Decoder, ObjectiveEvaluator
from nsga.encoding import (
    initialize_population,
    uniform_crossover,
    bitwise_mutation,
)
from nsga.nsgaii import (
    assign_rank_and_crowding,
    environmental_selection,
    evaluate_individual,
    evaluate_population,
    pick_reference,
    tournament_select,
)


def hypervolume(front, reference):
    """2-block hypervolume approximation for a Pareto front (minimization).

    This deliberately small/no-dependency implementation estimates HV by
    summing the dominated hyper-rectangle of each solution against a reference
    point.  Objective vectors are normalized to keep the metric stable.
    """
    if not front:
        return 0.0
    pts = [list(ind.objectives) for ind in front]
    n = len(pts[0])
    area = 0.0
    for p in pts:
        vol = 1.0
        for i in range(n):
            width = max(0.0, reference[i] - p[i])
            vol *= max(1e-9, width)
        area += vol
    return area


def _run_inner_nsga(scenario, population_size, generations, seeds, seed=0):
    rng = random.Random(seed)
    decoder = Decoder(scenario)
    evaluator = ObjectiveEvaluator()

    population = initialize_population(
        population_size, len(scenario.tasks), scenario.valid_ot_ids, rng,
        seed_chromosomes=list(seeds or []),
    )
    evaluate_population(scenario, population, decoder, evaluator)
    assign_rank_and_crowding(population)
    best = pick_reference(population).clone()

    for _ in range(1, generations + 1):
        offspring = []
        while len(offspring) < population_size:
            p1 = tournament_select(population, rng)
            p2 = tournament_select(population, rng)
            if rng.random() < 0.9:
                child = uniform_crossover(p1.chromosome, p2.chromosome, rng)
            else:
                child = list(p1.chromosome)
            if rng.random() < 0.6:
                child = bitwise_mutation(child, scenario.valid_ot_ids, rng)
            offspring.append(evaluate_individual(scenario, child, decoder, evaluator))

        combined = population + offspring
        fronts = assign_rank_and_crowding(combined)
        population = environmental_selection(fronts, population_size)
        assign_rank_and_crowding(population)
        current = pick_reference(population).clone()
        if current.selection_key < best.selection_key:
            best = current

    return best, population


def evaluate_heuristic(heuristic, scenario, config, seed=0):
    """Run the inner NSGA-II biased by ``heuristic`` and fill both metric layers."""
    inner_pop = int(config.get("inner_population_size", 40))
    inner_gen = int(config.get("inner_generations", 10))
    seed_ratio = float(config.get("heuristic_seed_ratio", 0.3))
    num_seeds = max(1, int(inner_pop * seed_ratio))

    rng = random.Random(seed)
    seeds = heuristic.apply(scenario, rng, num_seeds=num_seeds)

    start = time.time()
    best, population = _run_inner_nsga(scenario, inner_pop, inner_gen, seeds, seed=seed)
    runtime = time.time() - start

    # Solution-level metrics (CMAPP Jd/Jm/Jb/Jt, section 38).
    heuristic.best_solution = best
    heuristic.solution_objectives = list(best.objectives)
    heuristic.solution_metrics = dict(best.metrics)

    # Heuristic-level metrics (section 12).
    front = [ind for ind in population if ind.rank == 0]
    num_tasks = len(scenario.tasks)
    ref = [
        num_tasks + 1.0,
        1e9,
        1e9,
        1e9,
    ]
    hv = hypervolume(front, ref)
    feasible_rate = heuristic.solution_metrics.get("success_rate", 0.0)
    if "J_m" in heuristic.solution_metrics and heuristic.solution_metrics.get("J_m", 0) < 1e9:
        # Use real makespan/travel bounds when available for a tighter HV.
        ref = [
            num_tasks + 1.0,
            heuristic.solution_metrics.get("J_m", 1e9) * 1.5 + 1.0,
            heuristic.solution_metrics.get("J_b", 1e9) * 1.5 + 1.0,
            heuristic.solution_metrics.get("J_t", 1e9) * 1.5 + 1.0,
        ]
        hv = hypervolume(front, ref)

    heuristic.metrics = {
        "J_d": heuristic.solution_metrics.get("J_d", float("inf")),
        "J_m": heuristic.solution_metrics.get("J_m", float("inf")),
        "J_b": heuristic.solution_metrics.get("J_b", float("inf")),
        "J_t": heuristic.solution_metrics.get("J_t", float("inf")),
        "success_rate": heuristic.solution_metrics.get("success_rate", 0.0),
        "hypervolume": hv,
        "runtime": runtime,
        "feasible_rate": feasible_rate,
    }
    # Minimization convention (section 12): maximize HV & feasible_rate -> negate.
    heuristic.objectives = [
        -hv,
        -feasible_rate,
        runtime,
    ]
    return heuristic