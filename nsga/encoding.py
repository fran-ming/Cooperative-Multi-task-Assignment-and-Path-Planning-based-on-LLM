import random


class Individual:
    """Single-layer CMAPP chromosome: gene i is the OT assigned to Task i."""

    def __init__(self, chromosome, objectives=None, decode_result=None):
        self.chromosome = [int(v) for v in chromosome]
        self.objectives = objectives
        self.decode_result = decode_result
        self.rank = None
        self.crowding_distance = None
        self.llm_operation = None

    @property
    def metrics(self):
        if self.decode_result is None:
            return {}
        return self.decode_result.metrics

    @property
    def selection_key(self):
        """Lexicographic selection key.

        J_d (failed tasks) is always the first priority, so a solution with
        more finished tasks can never be replaced by one with fewer finished
        tasks merely because its makespan is shorter.
        """
        if self.objectives is not None:
            return tuple(float(x) for x in self.objectives)
        if self.decode_result is not None:
            metrics = self.decode_result.metrics
            return (
                float(metrics.get("J_d", float("inf"))),
                float(metrics.get("J_m", float("inf"))),
                float(metrics.get("J_b", float("inf"))),
                float(metrics.get("J_t", float("inf"))),
            )
        return (float("inf"), float("inf"), float("inf"), float("inf"))

    @property
    def fitness(self):
        # Scalar fitness used by NSGA-II selection. J_d has the dominant
        # penalty, followed by J_m, J_b and J_t.
        if self.decode_result is not None:
            try:
                return float(self.decode_result.metrics.get("fitness_J", float("inf")))
            except (TypeError, ValueError):
                pass
        if self.objectives is None:
            return float("inf")
        obj = [float(x) for x in self.objectives]
        return obj[0] * 1000.0 + obj[1] * 100.0 + obj[2] * 10.0 + obj[3] * 0.1

    def clone(self):
        other = Individual(list(self.chromosome), list(self.objectives) if self.objectives else None)
        other.decode_result = self.decode_result
        other.rank = self.rank
        other.crowding_distance = self.crowding_distance
        other.llm_operation = self.llm_operation
        return other


def validate_chromosome(chromosome, num_tasks, valid_ot_ids):
    if not isinstance(chromosome, (list, tuple)):
        return False
    if len(chromosome) != num_tasks:
        return False
    return all(v in valid_ot_ids for v in chromosome)


def repair_chromosome(chromosome, num_tasks, valid_ot_ids, rng):
    """Repair length/illegal genes deterministically enough for a safe fallback."""
    repaired = []
    for i in range(num_tasks):
        if i < len(chromosome) and chromosome[i] in valid_ot_ids:
            repaired.append(chromosome[i])
        else:
            repaired.append(rng.choice(valid_ot_ids))
    return repaired


def initialize_population(population_size, num_tasks, valid_ot_ids, rng, seed_chromosomes=None):
    population = []
    for chromosome in (seed_chromosomes or []):
        if validate_chromosome(chromosome, num_tasks, valid_ot_ids):
            population.append(Individual([int(v) for v in chromosome]))
    while len(population) < population_size:
        chromosome = [rng.choice(valid_ot_ids) for _ in range(num_tasks)]
        population.append(Individual(chromosome))
    return population


def uniform_crossover(parent_a, parent_b, rng):
    return [a if rng.random() < 0.5 else b for a, b in zip(parent_a, parent_b)]


def one_point_crossover(parent_a, parent_b, rng):
    point = rng.randint(1, len(parent_a) - 1)
    return parent_a[:point] + parent_b[point:]


def bitwise_mutation(chromosome, valid_ot_ids, rng, mutation_prob=None):
    if mutation_prob is None:
        mutation_prob = 1.0 / max(1, len(chromosome))
    child = list(chromosome)
    for i in range(len(child)):
        if rng.random() < mutation_prob:
            child[i] = rng.choice(valid_ot_ids)
    return child
