import random

from data.config import LLM_CONFIG
from nsga.decoder import Decoder, ObjectiveEvaluator
from nsga.encoding import initialize_population
from nsga.nsgaii import evaluate_population, evaluate_individual
from algorithms.llm_nsga import LLMOperator
from llm.client import LLMClient


class LLMPureSolver:
    """LLM-only search baseline: LLM operators produce candidates, decoder evaluates."""

    def __init__(self, population_size=100, generations=100, crossover_rate=0.9,
                 mutation_rate=0.6, seed=0, client=None):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.rng = random.Random(seed)
        self.client = client
        self.operator = None
        self.decoder = None
        self.evaluator = None
        self.history = []

    def solve(self, scenario):
        self.decoder = Decoder(scenario)
        self.evaluator = ObjectiveEvaluator()
        if self.client is None:
            self.client = LLMClient()
        self.operator = LLMOperator(scenario, client=self.client, rng=self.rng)

        population = initialize_population(
            self.population_size, len(scenario.tasks), scenario.valid_ot_ids, self.rng,
            seed_chromosomes=getattr(scenario, "heuristic_chromosomes", None),
        )
        evaluate_population(scenario, population, self.decoder, self.evaluator)
        best = min(population, key=lambda ind: ind.selection_key).clone()
        self.history = [{"gen": 0, "fitness_J": best.fitness}]

        for gen in range(1, self.generations + 1):
            ranked = sorted(population, key=lambda ind: ind.selection_key)[: max(2, self.population_size)]
            pairs = []
            for _ in range(self.population_size):
                a = ranked[self.rng.randrange(len(ranked))]
                b = ranked[self.rng.randrange(len(ranked))]
                pairs.append((a, b))
            mutation_flags = [self.rng.random() < self.mutation_rate for _ in pairs]
            offspring_chrom = self.operator.crossover_mutation_batch(pairs, mutation_flags)

            offspring = [
                evaluate_individual(scenario, chrom, self.decoder, self.evaluator)
                for chrom in offspring_chrom
            ]
            combined = population + offspring
            population = sorted(combined, key=lambda ind: ind.selection_key)[: self.population_size]
            cur_best = min(population, key=lambda ind: ind.selection_key).clone()
            self.history.append({"gen": gen, "fitness_J": cur_best.fitness})
            if cur_best.selection_key < best.selection_key:
                best = cur_best

        return best, self.history

    def get_stats(self):
        return dict(self.client.stats) if self.client else {}
