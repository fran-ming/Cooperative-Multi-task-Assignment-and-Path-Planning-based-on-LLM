# -*- coding: utf-8 -*-
"""Pareto Front Grid for heuristic-level objective space (guide section 13-14).

The grid keeps both quality and search-region diversity: sparsely populated
cells are preferred during parent selection so the meta-search does not
collapse onto a single crowded region.
"""

import random


class ParetoGrid:
    def __init__(self, bins=5):
        self.bins = bins
        self.cells = {}
        self.mins = None
        self.maxs = None
        self._located = {}

    def _bounds(self, heuristics):
        objectives = [h.objectives for h in heuristics if h.objectives is not None]
        if not objectives:
            return None, None
        n = len(objectives[0])
        mins = [min(o[i] for o in objectives) for i in range(n)]
        maxs = [max(o[i] for o in objectives) for i in range(n)]
        return mins, maxs

    def normalize(self, values, mins, maxs):
        out = []
        for v, lo, hi in zip(values, mins, maxs):
            if hi - lo < 1e-12:
                out.append(0.0)
            else:
                out.append((v - lo) / (hi - lo))
        return out

    def locate(self, objective_values):
        coords = self.normalize(objective_values, self.mins, self.maxs)
        cell = []
        for c in coords:
            index = int(min(self.bins - 1, max(0, c * self.bins)))
            cell.append(index)
        return tuple(cell)

    def build(self, heuristics):
        self.mins, self.maxs = self._bounds(heuristics)
        self.cells = {}
        self._located = {}
        if self.mins is None:
            return self
        for h in heuristics:
            if h.objectives is None:
                continue
            cell = self.locate(h.objectives)
            h.grid_cell = cell
            self.cells.setdefault(cell, []).append(h)
            self._located[h.heuristic_id] = cell
        return self

    def cell_density(self, cell):
        return len(self.cells.get(cell, []))

    def select_parent(self, heuristics, rng=None):
        """Pick a heuristic preferring sparse cells (diversity + quality)."""
        rng = rng or random.Random(0)
        evaluated = [h for h in heuristics if h.objectives is not None]
        if not evaluated:
            return rng.choice(heuristics) if heuristics else None
        # Lowest-density cells first; within them, best objective tuple first.
        evaluated.sort(key=lambda h: (
            self.cell_density(h.grid_cell),
            h.pareto_rank if h.pareto_rank is not None else 1e9,
            tuple(h.objectives),
        ))
        sparse = evaluated[: max(1, len(evaluated) // 3) + 1]
        return rng.choice(sparse)

    def select_elite(self, heuristics, n):
        """Select n heuristics covering the most promising, least-crowded cells."""
        evaluated = [h for h in heuristics if h.objectives is not None]
        if not evaluated:
            return []
        evaluated.sort(key=lambda h: (
            h.pareto_rank if h.pareto_rank is not None else 1e9,
            self.cell_density(h.grid_cell),
            tuple(h.objectives),
        ))
        return evaluated[:n]