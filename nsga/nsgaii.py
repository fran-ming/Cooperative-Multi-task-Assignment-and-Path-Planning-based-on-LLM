import random
from collections import defaultdict

from nsga.encoding import (
    Individual,
    initialize_population,
    uniform_crossover,
    bitwise_mutation,
)


def dominates(a, b):
    """Return True when objective vector a dominates b (minimization)."""
    at_least_one = False
    for x, y in zip(a, b):
        if x > y:
            return False
        if x < y:
            at_least_one = True
    return at_least_one


def fast_non_dominated_sort(individuals):
    """Return Pareto fronts in ascending rank order.

    The first front contains every non-dominated individual (not just the
    first one found). Later fronts are built from the dominated sets of the
    previous front.
    """
    if not individuals:
        return []
    domination_count = defaultdict(int)
    dominated_set = defaultdict(list)
    first_front = []
    for p in individuals:
        for q in individuals:
            if p is q:
                continue
            if dominates(p.objectives, q.objectives):
                dominated_set[p].append(q)
            elif dominates(q.objectives, p.objectives):
                domination_count[p] += 1
        if domination_count[p] == 0:
            p.rank = 0
            first_front.append(p)
    fronts = [first_front]
    current = first_front
    while current:
        next_front = []
        for p in current:
            for q in dominated_set[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    q.rank = len(fronts)
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        current = next_front
    return fronts


def crowding_distance(front):
    for ind in front:
        ind.crowding_distance = 0.0
    if len(front) <= 2:
        for ind in front:
            ind.crowding_distance = float("inf")
        return
    n_obj = len(front[0].objectives)
    for m in range(n_obj):
        front.sort(key=lambda ind: ind.objectives[m])
        fmin = front[0].objectives[m]
        fmax = front[-1].objectives[m]
        if fmax - fmin < 1e-12:
            continue
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        for i in range(1, len(front) - 1):
            if front[i].crowding_distance == float("inf"):
                continue
            front[i].crowding_distance += (
                front[i + 1].objectives[m] - front[i - 1].objectives[m]
            ) / (fmax - fmin)


def assign_rank_and_crowding(population):
    fronts = fast_non_dominated_sort(population)
    for i, front in enumerate(fronts):
        for ind in front:
            ind.rank = i
        crowding_distance(front)
    return fronts


def environmental_selection(fronts, size):
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(front)
        else:
            remaining = size - len(selected)
            # Within a Pareto front, scalar J fitness dominates so failed-task
            # minimization is never sacrificed for crowding diversity.
            ordered = sorted(front, key=lambda ind: (
                ind.selection_key,
                -(ind.crowding_distance if ind.crowding_distance is not None else 0.0),
            ))
            selected.extend(ordered[:remaining])
            break
    return selected


def tournament_select(population, rng, k=2):
    candidates = rng.sample(population, min(k, len(population)))
    return min(
        candidates,
        key=lambda ind: (
            ind.selection_key,
            ind.rank if ind.rank is not None else 1e9,
            -(ind.crowding_distance if ind.crowding_distance is not None else 0.0),
        ),
    )


def evaluate_individual(scenario, chromosome, decoder, evaluator):
    decode_result = decoder.decode(chromosome)
    metrics, objectives = evaluator.evaluate(scenario, chromosome, decode_result)
    individual = Individual(list(chromosome), objectives=list(objectives), decode_result=decode_result)
    return individual


def evaluate_population(scenario, population, decoder, evaluator):
    for ind in population:
        decode_result = decoder.decode(ind.chromosome)
        metrics, objectives = evaluator.evaluate(scenario, ind.chromosome, decode_result)
        ind.decode_result = decode_result
        ind.objectives = list(objectives)
    return population


def pick_reference(population):
    return min(population, key=lambda ind: ind.selection_key)


class NSGA2Solver:
    def __init__(self, population_size=100, generations=100, crossover_rate=0.9,
                 mutation_rate=0.6, seed=0):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.rng = random.Random(seed)
        self.decoder = None
        self.evaluator = None
        self.fronts = []

    def solve(self, scenario):
        from nsga.decoder import Decoder, ObjectiveEvaluator
        self.decoder = Decoder(scenario)
        self.evaluator = ObjectiveEvaluator()
        population = initialize_population(
            self.population_size, len(scenario.tasks), scenario.valid_ot_ids, self.rng,
            seed_chromosomes=getattr(scenario, "heuristic_chromosomes", None),
        )
        evaluate_population(scenario, population, self.decoder, self.evaluator)
        self.fronts = assign_rank_and_crowding(population)
        best = pick_reference(population).clone()
        history = [{"gen": 0, "fitness_J": best.fitness}]

        for gen in range(1, self.generations + 1):
            offspring = []
            while len(offspring) < self.population_size:
                p1 = tournament_select(population, self.rng)
                p2 = tournament_select(population, self.rng)
                if self.rng.random() < self.crossover_rate:
                    child_chrom = uniform_crossover(p1.chromosome, p2.chromosome, self.rng)
                else:
                    child_chrom = list(p1.chromosome)
                if self.rng.random() < self.mutation_rate:
                    child_chrom = bitwise_mutation(child_chrom, scenario.valid_ot_ids, self.rng)
                offspring.append(evaluate_individual(scenario, child_chrom, self.decoder, self.evaluator))

            combined = population + offspring
            self.fronts = fast_non_dominated_sort(combined)
            for i, front in enumerate(self.fronts):
                for ind in front:
                    ind.rank = i
                crowding_distance(front)
            population = environmental_selection(self.fronts, self.population_size)
            assign_rank_and_crowding(population)
            cur_best = pick_reference(population).clone()
            history.append({"gen": gen, "fitness_J": cur_best.fitness})
            if cur_best.selection_key < best.selection_key:
                best = cur_best

        return best, history

    def get_pareto_front(self, population=None):
        if population is None:
            population = self.population if hasattr(self, "population") else []
        return [ind for ind in population if ind.rank == 0]
