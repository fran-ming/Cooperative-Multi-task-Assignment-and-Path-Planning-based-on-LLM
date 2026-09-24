# -*- coding: utf-8 -*-
"""Meta-level heuristic population container (guide section 10)."""

from nsga.nsgaii import dominates


class HeuristicPool:
    """Maintains the ``H1 H2 H3 ...`` population of search heuristics."""

    def __init__(self, heuristics=None):
        self.heuristics = list(heuristics or [])

    def add(self, heuristic):
        self.heuristics.append(heuristic)
        return heuristic

    def extend(self, heuristics):
        self.heuristics.extend(heuristics)
        return self

    def __iter__(self):
        return iter(self.heuristics)

    def __len__(self):
        return len(self.heuristics)

    # ------------------------------------------------------------------ #
    def _signature(self, h):
        priority = tuple(h.strategy.get("priority") or [])
        return (h.name, priority, h.strategy.get("construction"), h.strategy.get("repair"))

    def remove_duplicates(self):
        seen = {}
        deduped = []
        for h in self.heuristics:
            key = self._signature(h)
            if key in seen:
                continue
            seen[key] = h
            deduped.append(h)
        self.heuristics = deduped
        return self

    def pareto_filter(self):
        """Keep only the non-dominated heuristics by heuristic objectives."""
        evaluated = [h for h in self.heuristics if h.objectives is not None]
        if not evaluated:
            return self
        keep = []
        for h in evaluated:
            dominated = any(
                dominates(o.objectives, h.objectives)
                for o in evaluated if o is not h
            )
            if not dominated:
                keep.append(h)
        self.heuristics = keep
        return self

    def select_elite(self, n):
        """Return the n best heuristics (lower pareto rank, then higher quality)."""
        ranked = [h for h in self.heuristics if h.objectives is not None]
        key = lambda h: (
            h.pareto_rank if h.pareto_rank is not None else 1e9,
            tuple(h.objectives),
        )
        ranked.sort(key=key)
        return ranked[:n]

    def by_cluster(self, cluster_id):
        return [h for h in self.heuristics if h.cluster_id == cluster_id]

    def prune_to(self, n):
        """Trim to at most n heuristics preserving Pareto rank order."""
        if len(self.heuristics) <= n:
            return self
        ranked = [h for h in self.heuristics if h.objectives is not None]
        unevaluated = [h for h in self.heuristics if h.objectives is None]
        rank_key = lambda h: (
            h.pareto_rank if h.pareto_rank is not None else 1e9,
            -(h.crowding_distance if h.crowding_distance is not None else 0.0),
            tuple(h.objectives) if h.objectives is not None else tuple([1e9] * 3),
        )
        ranked.sort(key=rank_key)
        self.heuristics = (ranked + unevaluated)[:n]
        return self