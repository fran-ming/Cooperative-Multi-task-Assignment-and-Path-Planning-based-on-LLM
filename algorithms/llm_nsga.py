import random

from data.config import LLM_CONFIG
from nsga.decoder import Decoder, ObjectiveEvaluator
from nsga.encoding import (
    Individual,
    initialize_population,
    validate_chromosome,
    repair_chromosome,
    uniform_crossover,
    bitwise_mutation,
)
from nsga.nsgaii import (
    assign_rank_and_crowding,
    environmental_selection,
    evaluate_population,
    evaluate_individual,
    pick_reference,
    tournament_select,
)
from llm.client import LLMClient
from llm.prompts import (
    SELECTION_SYSTEM,
    CROSSOVER_SYSTEM,
    MUTATION_SYSTEM,
    selection_user,
    crossover_batch_user,
    mutation_batch_user,
    parse_json_response,
    extract_selected_ids,
    extract_batch_offspring,
    extract_batch_mutations,
)


class LLMOperator:
    """LLM evolution operators with JSON validation, retry, and fallback."""

    def __init__(self, scenario, client=None, rng=None, candidate_pool=None,
                 batch_size=None):
        self.scenario = scenario
        self.client = client or LLMClient()
        self.rng = rng or random.Random(0)
        self.candidate_pool = candidate_pool or int(LLM_CONFIG.get("candidate_pool", 30))
        self.batch_size = batch_size or int(LLM_CONFIG.get("batch_size", 10))

    def _generate(self, system_prompt, user_prompt):
        try:
            return self.client.generate(system_prompt, user_prompt)
        except Exception as exc:
            self.client.stats["fallback_calls"] += 1
            return None

    def select_parents(self, population, num_parents):
        def _fitness_of(ind):
            # Lexicographic [J_d, J_m, J_b, J_t]: finished-task count is the
            # dominant priority and can never be traded for a shorter makespan.
            return ind.selection_key

        # Prioritize by lexicographic objectives, then Pareto rank and crowding
        # distance as secondary tiebreakers to preserve diversity.
        pool = sorted(
            population,
            key=lambda ind: (
                _fitness_of(ind),
                ind.rank if ind.rank is not None else 1e9,
                -(ind.crowding_distance or 0.0),
            ),
        )[:self.candidate_pool]
        try:
            response = self._generate(SELECTION_SYSTEM, selection_user(pool, num_parents))
            payload = parse_json_response(response) if response else None
            selected_ids = extract_selected_ids(payload) if payload else None
            if selected_ids:
                selected = []
                for sid in selected_ids:
                    try:
                        idx = int(str(sid).strip()) - 1
                    except Exception:
                        continue
                    if 0 <= idx < len(pool):
                        selected.append(pool[idx])
                if selected:
                    while len(selected) < num_parents:
                        selected.append(self.rng.choice(selected) if selected else pool[0])
                    return selected[:num_parents]
        except Exception:
            pass
        return [tournament_select(population, self.rng) for _ in range(num_parents)]

    def crossover_batch(self, parent_pairs):
        if not parent_pairs:
            return []
        batch_size = min(self.batch_size, len(parent_pairs))
        result_chromosomes = []
        for start in range(0, len(parent_pairs), batch_size):
            pairs = parent_pairs[start:start + batch_size]
            fallback = [uniform_crossover(a.chromosome, b.chromosome, self.rng) for a, b in pairs]
            include_task_info = not getattr(self.client, "_initialized", False)
            response = self._generate(CROSSOVER_SYSTEM, crossover_batch_user(pairs, self.scenario, include_task_info=include_task_info))
            payload = parse_json_response(response) if response else None
            offspring = extract_batch_offspring(payload) if payload else None
            if not offspring:
                result_chromosomes.extend(fallback)
                continue
            for i, pair in enumerate(pairs):
                chrom = offspring[i] if i < len(offspring) else None
                if validate_chromosome(chrom, len(self.scenario.tasks), self.scenario.valid_ot_ids):
                    result_chromosomes.append([int(v) for v in chrom])
                else:
                    result_chromosomes.append(fallback[i])
        return result_chromosomes

    def mutation_batch(self, individuals):
        if not individuals:
            return []
        batch_size = min(self.batch_size, len(individuals))
        result_chromosomes = []
        for start in range(0, len(individuals), batch_size):
            batch = individuals[start:start + batch_size]
            fallback = [bitwise_mutation(ind.chromosome, self.scenario.valid_ot_ids, self.rng) for ind in batch]
            include_task_info = not getattr(self.client, "_initialized", False)
            response = self._generate(MUTATION_SYSTEM, mutation_batch_user(batch, self.scenario, include_task_info=include_task_info))
            payload = parse_json_response(response) if response else None
            mutations = extract_batch_mutations(payload) if payload else None
            if not mutations:
                result_chromosomes.extend(fallback)
                continue
            for i, ind in enumerate(batch):
                item = mutations[i] if i < len(mutations) else None
                chrom = item.get("chromosome") if isinstance(item, dict) else None
                if validate_chromosome(chrom, len(self.scenario.tasks), self.scenario.valid_ot_ids):
                    result_chromosomes.append([int(v) for v in chrom])
                else:
                    result_chromosomes.append(fallback[i])
        return result_chromosomes


class LLMNSGA2Solver:
    """LLM-guided NSGA-II: LLM selects/crosses/mutates; NSGA-II controls Pareto selection."""

    def __init__(self, population_size=100, generations=100, crossover_rate=0.9,
                 mutation_rate=0.6, seed=0, client=None,
                 use_llm_selection=True, use_llm_crossover=True, use_llm_mutation=True):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        self.rng = random.Random(seed)
        self.client = client
        self.use_llm_selection = use_llm_selection
        self.use_llm_crossover = use_llm_crossover
        self.use_llm_mutation = use_llm_mutation
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
        assign_rank_and_crowding(population)
        best = pick_reference(population).clone()
        self.history = [{"gen": 0, "fitness_J": best.fitness}]

        for gen in range(1, self.generations + 1):
            if self.use_llm_selection:
                parents = self.operator.select_parents(population, self.population_size)
            else:
                parents = [tournament_select(population, self.rng) for _ in range(self.population_size)]

            pairs = []
            for _ in range(self.population_size):
                a = parents[self.rng.randrange(len(parents))]
                b = parents[self.rng.randrange(len(parents))]
                pairs.append((a, b))

            if self.use_llm_crossover:
                crossed_chromosomes = self.operator.crossover_batch(pairs)
            else:
                crossed_chromosomes = [uniform_crossover(a.chromosome, b.chromosome, self.rng) for a, b in pairs]

            # Apply crossover probability: non-crossover offspring clone parent A.
            for i, (a, b) in enumerate(pairs):
                if self.rng.random() >= self.crossover_rate:
                    crossed_chromosomes[i] = list(a.chromosome)

            mutate_indices = [i for i in range(len(crossed_chromosomes))
                              if self.rng.random() < self.mutation_rate]
            if self.use_llm_mutation:
                mut_inputs = [Individual(crossed_chromosomes[i]) for i in mutate_indices]
                mutated = self.operator.mutation_batch(mut_inputs)
                for pos, idx in enumerate(mutate_indices):
                    crossed_chromosomes[idx] = mutated[pos]
            else:
                for idx in mutate_indices:
                    crossed_chromosomes[idx] = bitwise_mutation(
                        crossed_chromosomes[idx], scenario.valid_ot_ids, self.rng
                    )

            offspring = [
                evaluate_individual(scenario, chrom, self.decoder, self.evaluator)
                for chrom in crossed_chromosomes
            ]

            combined = population + offspring
            fronts = assign_rank_and_crowding(combined)
            population = environmental_selection(fronts, self.population_size)
            assign_rank_and_crowding(population)
            cur_best = pick_reference(population).clone()
            self.history.append({"gen": gen, "fitness_J": cur_best.fitness})
            if cur_best.selection_key < best.selection_key:
                best = cur_best

        return best, self.history

    def get_stats(self):
        return dict(self.client.stats) if self.client else {}
