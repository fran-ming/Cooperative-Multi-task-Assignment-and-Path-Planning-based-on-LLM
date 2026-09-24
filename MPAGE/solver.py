# -*- coding: utf-8 -*-
"""MPaGE-CMAPP meta-level solver (guide sections 22-24).

Orchestrates heuristic-level evolution on top of the unchanged NSGA-II /
Decoder / ObjectiveEvaluator stack.  Every LLM call is guarded so the solver
runs deterministically even with ``enable_llm=False``.
"""

import random

from MPAGE.config import MPAGE_CONFIG
from MPAGE.heuristic import build_presets
from MPAGE.heuristic_pool import HeuristicPool
from MPAGE.evaluator import evaluate_heuristic
from MPAGE.pareto_grid import ParetoGrid
from MPAGE.semantic_cluster import cluster
from MPAGE import operators, reflection
from nsga.nsgaii import dominates


class MPAGESolver:
    def __init__(self, config=None, seed=42):
        self.config = dict(MPAGE_CONFIG if config is None else config)
        self.seed = seed
        self.rng = random.Random(seed)
        self.history = []
        self.heuristic_pool = None
        self.stats = {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "latency_seconds": 0.0,
            "failed_calls": 0, "fallback_calls": 0,
        }

    def _make_client(self):
        if not self.config.get("enable_llm", True):
            return None
        try:
            from llm.client import LLMClient
            return LLMClient()
        except Exception:
            return None

    def _accumulate(self, client):
        if client is None:
            return
        for key in ("calls", "input_tokens", "output_tokens", "total_tokens",
                    "latency_seconds", "failed_calls", "fallback_calls"):
            self.stats[key] = self.stats.get(key, 0) + client.stats.get(key, 0)

    def initialize_heuristics(self, scenario):
        return HeuristicPool(build_presets())

    def _assign_ranks(self, heuristics):
        evaluated = [h for h in heuristics if h.objectives is not None]
        for h in heuristics:
            h.pareto_rank = None
        for h in evaluated:
            h.pareto_rank = sum(
                1 for o in evaluated
                if o is not h and dominates(o.objectives, h.objectives)
            )
        return evaluated

    def evaluate_pool(self, scenario, pool):
        for h in pool:
            evaluate_heuristic(h, scenario, self.config, seed=self.seed)
        self._assign_ranks(pool.heuristics)
        return pool

    def _selection_key(self, h):
        return tuple(h.objectives) if h.objectives is not None else (float("inf"),) * 3

    def solve(self, scenario):
        pool = self.initialize_heuristics(scenario)
        self.evaluate_pool(scenario, pool)
        self.heuristic_pool = pool

        grid = ParetoGrid(int(self.config.get("grid_bins", 5)))
        elite_n = int(self.config.get("elite_heuristics", 4))
        pop_size = int(self.config.get("heuristic_population_size", 6))
        mut_prob = float(self.config.get("mutation_probability", 0.8))
        cross_prob = float(self.config.get("crossover_probability", 0.7))
        meta_gens = int(self.config.get("meta_generations", 10))

        self.history = [self._meta_record(0, pool)]

        for gen in range(1, meta_gens + 1):
            grid.build(pool.heuristics)
            elite = grid.select_elite(pool.heuristics, elite_n)

            client = self._make_client()
            cluster(elite, scenario, client, int(self.config.get("semantic_clusters", 3)))

            offspring = []
            # Same-cluster mutation.
            for h in elite:
                if self.rng.random() < mut_prob:
                    offspring.append(operators.mutate_heuristic(h, scenario, client, self.rng))
            # Cross-cluster crossover.
            cluster_ids = sorted(set(h.cluster_id for h in elite if h.cluster_id is not None))
            for _ in range(max(1, elite_n // 2)):
                if len(cluster_ids) < 2:
                    break
                ca, cb = self.rng.sample(cluster_ids, 2)
                from_ca = [h for h in elite if h.cluster_id == ca]
                from_cb = [h for h in elite if h.cluster_id == cb]
                if not from_ca or not from_cb:
                    continue
                if self.rng.random() < cross_prob:
                    offspring.append(operators.crossover_heuristics(
                        self.rng.choice(from_ca), self.rng.choice(from_cb),
                        scenario, client, self.rng,
                    ))

            # Reflection on the leading heuristic.
            if elite:
                leader = min(elite, key=self._selection_key)
                reflection.reflect(leader, scenario, client)

            self._accumulate(client)

            for h in offspring:
                evaluate_heuristic(h, scenario, self.config,
                                   seed=self.seed + gen * 1000 + len(offspring))

            pool.extend(offspring).remove_duplicates()
            self._assign_ranks(pool.heuristics)
            pool.prune_to(pop_size)
            self.history.append(self._meta_record(gen, pool))

        evaluated = [h for h in pool.heuristics if h.best_solution is not None]
        best_heuristic = min(evaluated, key=lambda h: h.best_solution.selection_key)
        return {
            "best_heuristic": best_heuristic,
            "best_solution": best_heuristic.best_solution,
            "heuristic_pool": pool,
            "history": self.history,
            "stats": dict(self.stats),
        }

    def _meta_record(self, gen, pool):
        evaluated = [h for h in pool.heuristics if h.objectives is not None]
        if not evaluated:
            return {
                "gen": gen, "pool_size": len(pool),
                "best_J_d": None, "best_success_rate": None,
            }
        best = min(evaluated, key=lambda h: h.best_solution.selection_key)
        return {
            "gen": gen,
            "pool_size": len(pool),
            "best_J_d": best.solution_metrics.get("J_d"),
            "best_success_rate": best.solution_metrics.get("success_rate"),
            "best_heuristic": best.heuristic_id,
            "best_name": best.name,
            "hypervolumes": {h.heuristic_id: h.metrics.get("hypervolume") for h in evaluated},
        }

    def get_stats(self):
        return dict(self.stats)