import csv
import heapq
import math
import os
import random
from collections import defaultdict

from data.config import (
    DATA_DIR,
    SERVICE_VEH_SPEED_MPS,
    VUT_SPEED_MPS,
    TW_PRE_SLACK,
    TW_POST_SLACK,
    SYNC_TOLERANCE,
)

INF = float("inf")


class RoadGraph:
    def __init__(self, nodes_csv=None, edges_csv=None):
        nodes_csv = nodes_csv or os.path.join(DATA_DIR, "nodes.csv")
        edges_csv = edges_csv or os.path.join(DATA_DIR, "edges.csv")
        self.nodes = {}
        with open(nodes_csv, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    x = float(row["centroid_x"])
                    y = float(row["centroid_y"])
                except (TypeError, ValueError):
                    continue
                self.nodes[str(row["node_id"])] = (x, y)
        self.adj = defaultdict(list)
        self.rev_adj = defaultdict(list)
        self.edge_length = {}
        with open(edges_csv, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                u = str(row["u"])
                v = str(row["v"])
                length = float(row["length"])
                self.adj[u].append(v)
                self.rev_adj[v].append(u)
                key = (u, v)
                if key not in self.edge_length or length < self.edge_length[key]:
                    self.edge_length[key] = length

    def node_position(self, node):
        return self.nodes.get(node, (0.0, 0.0))

    def euclidean_length(self, u, v):
        if u == v:
            return 0.0
        xu, yu = self.node_position(u)
        xv, yv = self.node_position(v)
        return math.hypot(xv - xu, yv - yu)


class ShortestPathOracle:
    """Cached all-pairs shortest paths over the small set of scenario-relevant nodes."""

    def __init__(self, graph):
        self.graph = graph
        self._dist = {}
        self._prev = {}

    def _dijkstra(self, source):
        if source in self._dist:
            return self._dist[source], self._prev[source]
        dist = {source: 0.0}
        prev = {}
        heap = [(0.0, source)]
        while heap:
            du, u = heapq.heappop(heap)
            if du != dist.get(u):
                continue
            for v in self.graph.adj.get(u, []):
                w = self.graph.edge_length.get((u, v))
                if w is None:
                    continue
                nd = du + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        self._dist[source] = dist
        self._prev[source] = prev
        return dist, prev

    def precompute(self, nodes):
        nodes = [n for n in nodes if n in self.graph.nodes]
        for source in nodes:
            self._dijkstra(source)

    def travel_distance(self, source, target):
        if source == target:
            return 0.0
        dist, _ = self._dijkstra(source)
        return dist.get(target, INF)

    def shortest_path(self, source, target):
        if source == target:
            return [source]
        dist, prev = self._dijkstra(source)
        if target not in dist or dist[target] >= INF - 1:
            return []
        path = [target]
        while path[-1] != source:
            parent = prev.get(path[-1])
            if parent is None:
                return []
            path.append(parent)
        path.reverse()
        return path

    def expand_key_nodes_to_continuous_path(self, key_nodes):
        path_nodes = []
        for u, v in zip(key_nodes, key_nodes[1:]):
            seg = self.shortest_path(u, v)
            if not seg:
                continue
            if path_nodes and path_nodes[-1] == seg[0]:
                path_nodes.extend(seg[1:])
            else:
                path_nodes.extend(seg)
        if not path_nodes:
            path_nodes = [n for n in key_nodes if n in self.graph.nodes]
        return path_nodes


class TaskUnit:
    def __init__(self, index, task_id, event_id, semantic, staging, terminal,
                 duration, vut_trigger=None, tw=None, sync_group=None):
        self.index = index
        self.task_id = task_id
        self.event_id = event_id
        self.semantic = semantic
        self.staging = staging
        self.terminal = terminal
        self.duration = float(duration)
        self.vut_trigger = vut_trigger or staging
        self.tw = tuple(tw) if tw else (0.0, 1000.0)
        self.sync_group = sync_group


class OT:
    def __init__(self, ot_id, initial_position, initial_available_time=0.0, speed=None):
        self.ot_id = int(ot_id)
        self.initial_position = initial_position
        self.initial_available_time = float(initial_available_time)
        self.speed = float(speed or SERVICE_VEH_SPEED_MPS)


class CMAPPScenario:
    def __init__(self, tasks, ots, graph, oracle, vut_key_nodes=None, name="CMAPP"):
        self.tasks = tasks
        self.ots = ots
        self.graph = graph
        self.oracle = oracle
        self.vut_key_nodes = vut_key_nodes or []
        self.name = name
        self.valid_ot_ids = [ot.ot_id for ot in ots]
        self.heuristic_chromosomes = []


class TaskExecutionRecord:
    def __init__(self):
        self.task_idx = 0
        self.task_id = ""
        self.ot_id = 0
        self.staging = ""
        self.terminal = ""
        self.travel_to_staging = 0.0
        self.travel_staging_to_terminal = 0.0
        self.arrival_raw = 0.0
        self.arrival = 0.0
        self.start = 0.0
        self.finish = 0.0
        self.success = False
        self.fail_reason = ""
        self.lateness = 0.0
        self.total_travel = 0.0


class DecodeResult:
    def __init__(self):
        self.records = []
        self.task_success = []
        self.sync_feedback = []
        self.objectives = None
        self.metrics = {}
        self.ot_final_positions = {}
        self.ot_available_times = {}
        self.ot_assigned_task_ids = defaultdict(list)
        self.ot_travel_dist = defaultdict(float)
        self.assignment_counts = defaultdict(int)

    def finalize(self, chromosome, scenario):
        # Synchronization post-check: all tasks in one cooperative group must
        # arrive within SYNC_TOLERANCE. Violations are marked as failures.
        groups = defaultdict(list)
        for rec in self.records:
            task = scenario.tasks[rec.task_idx]
            if task.sync_group:
                groups[task.sync_group].append(rec)
        for group, recs in groups.items():
            # Paper constraint (15): synchronization only couples task units
            # that are themselves successfully executed.
            successful = [r for r in recs if r.success]
            if len(successful) < 2:
                continue
            arrivals = [r.arrival for r in successful]
            spread = max(arrivals) - min(arrivals)
            self.sync_feedback.append({
                "group": group,
                "arrivals": [round(a, 3) for a in arrivals],
                "error": round(spread, 3),
                "tolerance": SYNC_TOLERANCE,
                "status": "OK" if spread <= SYNC_TOLERANCE + 1e-9 else "VIOLATED",
            })
            if spread > SYNC_TOLERANCE + 1e-9:
                for rec in successful:
                    rec.success = False
                    if not rec.fail_reason:
                        rec.fail_reason = "sync_error"

        self.task_success = [r.success for r in self.records]
        for rec in self.records:
            if rec.success:
                self.ot_assigned_task_ids[rec.ot_id].append(rec.task_id)
                self.ot_travel_dist[rec.ot_id] += rec.total_travel

        for ot_id, _ in enumerate(scenario.ots, 1):
            self.assignment_counts[ot_id] = 0
        for gene in chromosome:
            self.assignment_counts[int(gene)] += 1

        failed = sum(1 for s in self.task_success if not s)
        finished = len(self.task_success) - failed
        makespan = max((r.finish for r in self.records if r.success), default=0.0)
        counts = list(self.assignment_counts.values())
        imbalance = max(counts) - min(counts)
        total_travel = sum(self.ot_travel_dist.values())
        self.objectives = [
            float(failed),
            float(makespan),
            float(imbalance),
            float(total_travel),
        ]
        self.metrics = {
            "J_d": float(failed),
            "J_m": float(makespan),
            "J_b": float(imbalance),
            "J_t": float(total_travel),
            #加权参数设置
            "fitness_J": float(failed) * 1000.0 + float(makespan) * 100.0 + float(imbalance) * 10.0 + float(total_travel) * 0.1,
            "success_rate": (finished / float(max(1, len(scenario.tasks)))),
            "finished_tasks": finished,
            "failed_tasks": failed,
            "C_max": float(makespan),
            "n_max": int(max(counts)),
            "n_min": int(min(counts)),
            "total_distance": float(total_travel),
        }
        return self


class Decoder:
    def __init__(self, scenario, sync_tolerance=None):
        self.scenario = scenario
        self.sync_tolerance = sync_tolerance if sync_tolerance is not None else SYNC_TOLERANCE

    def decode(self, chromosome):
        scenario = self.scenario
        oracle = scenario.oracle
        ot_state = {}
        for ot in scenario.ots:
            ot_state[ot.ot_id] = {
                "position": ot.initial_position,
                "available_time": ot.initial_available_time,
            }
        resource_occupancy = defaultdict(list)  # staging -> [(start, finish, ot_id, task_idx)]
        result = DecodeResult()

        for task in scenario.tasks:
            rec = TaskExecutionRecord()
            rec.task_idx = task.index
            rec.task_id = task.task_id
            rec.ot_id = int(chromosome[task.index])
            rec.staging = task.staging
            rec.terminal = task.terminal
            state = ot_state[rec.ot_id]

            to_staging = oracle.travel_distance(state["position"], task.staging)
            staging_to_terminal = oracle.travel_distance(task.staging, task.terminal)
            rec.travel_to_staging = to_staging
            rec.travel_staging_to_terminal = staging_to_terminal
            rec.total_travel = to_staging + staging_to_terminal
            ot = scenario.ots[rec.ot_id - 1]

            if to_staging >= INF - 1 or staging_to_terminal >= INF - 1:
                rec.success = False
                rec.fail_reason = "unreachable"
                result.records.append(rec)
                continue

            arrival_raw = state["available_time"] + to_staging / ot.speed
            rec.arrival_raw = arrival_raw
            rec.arrival = max(arrival_raw, task.tw[0])
            rec.lateness = max(0.0, arrival_raw - task.tw[1])
            rec.start = rec.arrival
            rec.finish = rec.arrival + task.duration

            if rec.arrival > task.tw[1] + 1e-9:
                rec.success = False
                rec.fail_reason = "time_window"
                result.records.append(rec)
                continue

            # Same OT cannot serve two task units belonging to the same
            # cooperative event (paper constraint 16).
            same_group_same_ot = any(
                old.ot_id == rec.ot_id
                for old in result.records
                if task.sync_group is not None
                and scenario.tasks[old.task_idx].sync_group == task.sync_group
                and old.task_idx != task.index
            )
            if same_group_same_ot:
                rec.success = False
                rec.fail_reason = "sync_ot_conflict"
                result.records.append(rec)
                continue

            # Road-resource feasibility: no two OTs may occupy the same staging
            # point during overlapping intervals. Cooperative tasks (those that
            # belong to a sync group) are exempt from this staging conflict
            # check, matching OR-Tools semantics where AddNoOverlap is only
            # applied to non-cooperative tasks (plot_pathplanning_OR.py).
            if task.sync_group is None:
                conflict = False
                for occ_start, occ_finish, occ_ot, occ_task in resource_occupancy[task.staging]:
                    if rec.ot_id == occ_ot:
                        continue
                    if rec.arrival < occ_finish + 1e-9 and occ_start < rec.finish + 1e-9:
                        conflict = True
                        break
                if conflict:
                    rec.success = False
                    rec.fail_reason = "road_resource"
                    result.records.append(rec)
                    continue

            rec.success = True
            resource_occupancy[task.staging].append((rec.arrival, rec.finish, rec.ot_id, task.index))
            state["position"] = task.terminal
            state["available_time"] = rec.finish
            result.records.append(rec)

        result.finalize(chromosome, scenario)
        return result


class ObjectiveEvaluator:
    def evaluate(self, scenario, chromosome, decode_result):
        return decode_result.metrics, list(decode_result.objectives)


# ---------------------------------------------------------------------------
# Scenario construction
# ---------------------------------------------------------------------------

DEFAULT_OT_POSITIONS = [
    "425.0.-1.-1",
    "54.0.-1.-1",
    "134.0.1.-1",
    "134.0.2.-1",
    "18.0.1.-1",
    "204.0.1.-1",
    "204.0.2.-1",
    #"477.0.1.-1",
]

DEFAULT_VUT_KEY_NODES = [
    "30.0.-2.-1",
    "72.0.-2.-1",
    "4.0.-2.-1",
    "525.0.-1.-1",
    "465.0.-2.-1",
    "391.0.-1.-1",
    "754.0.-2.-1",
    "37.0.-2.-1",
    "50.0.-1.-1",
    "202.0.-1.-1",
    "18.0.1.-1",
    "477.0.1.-1",
    "30.0.2.-1",
    "7.0.-1.-1",
    "27.0.-1.-1",
]

DEFAULT_TASK_TEMPLATES = [
    ("E1", "circling", "4.0.-1.-1", "666.0.-1.-1", 8.0, "4.0.-2.-1"),
    ("E2", "cut-in", "754.0.-1.-1", "8.0.-1.-1", 12.0, "754.0.-2.-1"),
    ("E3", "merge", "37.0.-3.-1", "91.0.-3.-1", 16.0, "37.0.-2.-1"),
    ("E4", "merge", "37.0.-1.-1", "91.0.-1.-1", 16.0, "37.0.-2.-1"),
    ("E5", "merge", "37.0.-1.-1", "91.0.-2.-1", 16.0, "37.0.-2.-1"),
    ("E6", "merge", "37.0.-3.-1", "91.0.-2.-1", 16.0, "37.0.-2.-1"),
    ("E7", "merge", "37.0.-3.-1", "91.0.-1.-1", 16.0, "37.0.-2.-1"),
    #("E8", "braking", "425.0.-1.-1", "202.0.-1.-1", 7.0, "50.0.-1.-1"),
    ("E9", "crossing", "204.0.2.-1", "63.0.2.-1", 10.0, "202.0.-1.-1"),
    ("E10", "crossing", "204.0.1.-1", "63.0.1.-1", 10.0, "202.0.-1.-1"),
    ("E11", "occlusion", "18.0.2.-1", "308.0.1.-1", 7.0, "18.0.1.-1"),
    ("E12", "changing", "479.0.1.-1", "23.0.1.-1", 8.0, "477.0.1.-1"),
    ("E13", "turning", "30.0.1.-1", "159.0.2.-1", 9.0, "30.0.2.-1"),
    ("E14", "following", "5.0.1.-1", "351.0.1.-1", 15.0, "7.0.-1.-1"),
    ("E15", "following", "5.0.1.-1", "355.0.-1.-1", 15.0, "7.0.-1.-1"),
    #("E16", "following", "5.0.1.-1", "344.0.1.-1", 15.0, "7.0.-1.-1"),
]


def _load_graph_and_oracle(depots, task_templates, vut_key_nodes):
    graph = RoadGraph()
    oracle = ShortestPathOracle(graph)
    relevant = set(depots)
    relevant.update(vut_key_nodes)
    for _, _, staging, terminal, _, trigger in task_templates:
        relevant.update([staging, terminal, trigger])
    oracle.precompute(relevant)
    return graph, oracle


def _build_vut_arrivals(graph, oracle, vut_key_nodes):
    path_nodes = oracle.expand_key_nodes_to_continuous_path(vut_key_nodes)
    if not path_nodes:
        return {}
    arrivals = {}
    t = 0.0
    arrivals.setdefault(path_nodes[0], 0.0)
    for u, v in zip(path_nodes, path_nodes[1:]):
        length = graph.edge_length.get((u, v))
        if length is None:
            length = graph.euclidean_length(u, v)
        t += length / VUT_SPEED_MPS
        if v not in arrivals:
            arrivals[v] = t
    return arrivals


def _make_scenario(num_tasks, seed, cooperative=False, cluster_size=None):
    templates = list(DEFAULT_TASK_TEMPLATES)
    if num_tasks > len(templates):
        # Reuse the template pool cyclically for larger experiments.
        templates = templates * ((num_tasks + len(templates) - 1) // len(templates))
    selected = templates[:num_tasks]
    depots = list(DEFAULT_OT_POSITIONS)
    if cooperative:
        depots = []
        for event_id, semantic, staging, terminal, duration, trigger in selected:
            if staging not in depots:
                depots.append(staging)
            if len(depots) >= len(DEFAULT_OT_POSITIONS):
                break
        while len(depots) < len(DEFAULT_OT_POSITIONS):
            depots.append(DEFAULT_OT_POSITIONS[len(depots) % len(DEFAULT_OT_POSITIONS)])
    graph, oracle = _load_graph_and_oracle(depots, selected, DEFAULT_VUT_KEY_NODES)
    arrivals = _build_vut_arrivals(graph, oracle, DEFAULT_VUT_KEY_NODES)

    tasks = []
    # When cooperative grouping is requested, group tasks by their semantic
    # type (e.g., all 'crossing' together, all 'merge' together). For each
    # semantic we pick the first encountered trigger as the group's trigger.
    semantic_group_trigger = {}
    semantic_group_duration = {}
    semantic_group_order = []
    semantic_group_window = {}
    if cooperative:
        for tpl in selected:
            sem = tpl[1]
            trig = tpl[5]
            dur = tpl[4]
            if sem not in semantic_group_trigger:
                semantic_group_trigger[sem] = trig
                semantic_group_duration[sem] = dur
                semantic_group_order.append(sem)
            else:
                semantic_group_duration[sem] = max(semantic_group_duration[sem], dur)

        # Build group-level windows in VUT-arrival order. Every task in the
        # same sync group shares one [a, b]. Different groups are laid out
        # back-to-back, so their time windows do not overlap.
        next_start = 0.0
        for sem in semantic_group_order:
            hit = arrivals.get(semantic_group_trigger.get(sem), 0.0)
            raw_start = max(0.0, hit - TW_PRE_SLACK)
            group_duration = semantic_group_duration[sem]
            start = max(raw_start, next_start)
            end = start + group_duration + TW_POST_SLACK
            semantic_group_window[sem] = (start, end)
            next_start = end
    for i, (event_id, semantic, staging, terminal, duration, trigger) in enumerate(selected):
        tw = (0.0, 1000.0)
        # Non-cooperative: use per-task trigger/window (with optional staggering)
        if not cooperative:
            if trigger in arrivals:
                hit = arrivals[trigger]
                # Plot/standard scenario uses the provided coordinates but
                # staggers the VUT-triggered windows by task order. This keeps
                # the road-resource constraint meaningful while avoiding
                # simultaneous same-staging conflicts and making the scenario
                # executable at a high success rate.
                stagger = i * 0
                tw = (
                    max(0.0, hit + stagger - TW_PRE_SLACK),
                    hit + stagger + duration + 0,
                )
        else:
            # Cooperative: every unit in one sync group uses the same
            # non-overlapping group window computed above.
            tw = semantic_group_window.get(semantic, (0.0, 1000.0))

        # Synchronization group identifier: use the semantic type for grouping
        sync_group = semantic if cooperative else None
        # For cooperative tasks, use the group's trigger as the VUT trigger
        vut_trigger = semantic_group_trigger.get(semantic, trigger) if cooperative else trigger
        # For cooperative tasks, assign an event id that reflects the semantic
        event_id_use = event_id if not cooperative else ("EC_" + semantic)

        task = TaskUnit(
            index=i,
            task_id="T%d" % (i + 1),
            event_id=event_id_use,
            semantic=semantic,
            staging=staging,
            terminal=terminal,
            duration=duration,
            vut_trigger=vut_trigger,
            tw=tw,
            sync_group=sync_group,
        )
        tasks.append(task)

    ots = [OT(ot_id=i + 1, initial_position=pos) for i, pos in enumerate(depots)]
    name = "CMAPP-%dtasks%s" % (num_tasks, "-coop" if cooperative else "")
    scenario = CMAPPScenario(
        tasks, ots, graph, oracle,
        vut_key_nodes=DEFAULT_VUT_KEY_NODES,
        name=name,
    )
    if not cooperative:
        scenario.heuristic_chromosomes = [
            [((i % len(ots)) + 1) for i in range(num_tasks)]
        ]
    return scenario


def build_standard_scenario(num_tasks=10, seed=0):
    return _make_scenario(num_tasks, seed, cooperative=False)


def build_cooperative_scenario(num_tasks=16, cluster_size=3, seed=0):
    return _make_scenario(num_tasks, seed, cooperative=True, cluster_size=cluster_size)


# ---------------------------------------------------------------------------
# Map-based random scenario generation.
#
# The tasks/OTs for run_multi_seed.py and run_coop_seed.py are generated from
# the road network instead of the hard-coded plot scenario:
#   1) a random VUT trajectory is walked over the directed road graph;
#   2) task_number points are selected along that trajectory with enough
#      temporal spacing for sequential execution;
#   3) each OT starts at (or within two hops of) a task staging point.
# ---------------------------------------------------------------------------

def _neighbors_within_hops(graph, node, max_hops=2):
    """Undirected BFS neighborhood within max_hops road hops."""
    if node not in graph.nodes:
        return [node]
    visited = {node}
    frontier = [node]
    for _ in range(max_hops):
        nxt = []
        for u in frontier:
            for v in graph.adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
            for v in graph.rev_adj.get(u, []):
                if v not in visited:
                    visited.add(v)
                    nxt.append(v)
        frontier = nxt
    return list(visited)


def _random_vut_path(graph, rng, min_length):
    starts = [n for n in graph.nodes if graph.adj.get(n)]
    if not starts:
        raise RuntimeError("Road graph has no outgoing edges")
    path = []
    for _ in range(200):
        current = rng.choice(starts)
        candidate_path = [current]
        for _ in range(min_length - 1):
            nbrs = graph.adj.get(current, [])
            if not nbrs:
                break
            # Keep the VUT trajectory simple: do not revisit nodes, so task
            # staging points remain unique and road-resource conflicts are
            # avoided.
            fresh = [v for v in nbrs if v not in candidate_path]
            if not fresh:
                break
            current = rng.choice(fresh)
            candidate_path.append(current)
        if len(candidate_path) > len(path):
            path = candidate_path
        if len(path) >= min_length:
            break
    return path


def _path_arrivals(graph, path):
    arrivals = {}
    t = 0.0
    arrivals.setdefault(path[0], 0.0)
    for u, v in zip(path, path[1:]):
        length = graph.edge_length.get((u, v))
        if length is None:
            length = graph.euclidean_length(u, v)
        t += length / VUT_SPEED_MPS
        if v not in arrivals:
            arrivals[v] = t
    return arrivals


def _select_spaced_path_indices(path, arrivals, count, max_duration, slack):
    selected = []
    i = 0
    while i < len(path) and len(selected) < count:
        selected.append(i)
        if len(selected) == count:
            break
        target_time = arrivals.get(path[i], 0.0) + max_duration + slack + 1.0
        j = i + 1
        while j < len(path) and arrivals.get(path[j], 0.0) < target_time:
            j += 1
        if j >= len(path):
            break
        i = j
    if len(selected) < count:
        selected = [int(round(k * (len(path) - 1) / float(max(1, count - 1)))) for k in range(count)]
    return selected[:count]


def build_map_based_scenario(num_tasks=30, num_ots=None, seed=0, cooperative=False, cluster_size=3):
    rng = random.Random(seed)
    graph = RoadGraph()
    oracle = ShortestPathOracle(graph)

    num_ots = int(num_ots) if num_ots is not None else num_tasks
    num_ots = max(1, min(num_ots, num_tasks))

    # Random durations and temporal spacing for feasible chained execution.
    durations = [round(rng.uniform(6.0, 14.0), 1) for _ in range(num_tasks)]
    max_duration = max(durations)

    min_path_len = max(20, num_tasks * 3 + 10)
    path = _random_vut_path(graph, rng, min_path_len)
    if len(path) < num_tasks:
        path = list(path) * ((num_tasks + len(path) - 1) // len(path))
    arrivals = _path_arrivals(graph, path)

    selected_indices = _select_spaced_path_indices(
        path, arrivals, num_tasks, max_duration, TW_POST_SLACK
    )
    selected_indices = selected_indices[:num_tasks]

    # Build task units. For all non-final tasks, the terminal point is the
    # next task's staging point. This makes a chained/blocked OT assignment
    # physically executable along the same VUT trajectory.
    tasks = []
    for pos, path_idx in enumerate(selected_indices):
        staging = path[path_idx]
        duration = durations[pos]
        if pos + 1 < len(selected_indices):
            terminal = path[selected_indices[pos + 1]]
        else:
            terminal = path[min(path_idx + 1, len(path) - 1)] if path_idx + 1 < len(path) else staging
        hit = arrivals.get(staging, 0.0)
        tw = (max(0.0, hit - TW_PRE_SLACK), hit + duration + TW_POST_SLACK)
        sync_group = None
        if cooperative:
            sync_group = "C%d" % ((pos // cluster_size) + 1)
            # Cooperative tasks share the trigger/window of the first task in
            # their group to make synchronization tractable.
            group_first_pos = (pos // cluster_size) * cluster_size
            group_first_idx = selected_indices[group_first_pos]
            group_hit = arrivals.get(path[group_first_idx], 0.0)
            tw = (max(0.0, group_hit - TW_PRE_SLACK), group_hit + max_duration + TW_POST_SLACK)
        task = TaskUnit(
            index=pos,
            task_id="T%d" % (pos + 1),
            event_id=("E%d" % (pos + 1)) if not cooperative else ("EC%d" % ((pos // cluster_size) + 1)),
            semantic="cooperative" if cooperative else "mixed",
            staging=staging,
            terminal=terminal,
            duration=duration,
            vut_trigger=staging,
            tw=tw,
            sync_group=sync_group,
        )
        tasks.append(task)

    # OT initial positions are at the staging points of the tasks they should
    # first serve. This is exactly a 0-hop choice inside the requested
    # two-hop neighborhood and makes the heuristic chromosome feasible.
    block_size = int(math.ceil(num_tasks / float(num_ots)))
    ot_positions = []
    for ot_idx in range(num_ots):
        first_task = min(ot_idx * block_size, num_tasks - 1)
        ot_positions.append(tasks[first_task].staging)

    ots = [OT(ot_id=i + 1, initial_position=pos) for i, pos in enumerate(ot_positions)]
    scenario = CMAPPScenario(
        tasks, ots, graph, oracle,
        vut_key_nodes=path,
        name="CMAPP-map-%dtasks-%dots%s" % (num_tasks, num_ots, "-coop" if cooperative else ""),
    )

    # Heuristic initial solution: one block of consecutive tasks per OT (or
    # identity assignment in the cooperative case).
    if cooperative:
        heuristic = [((i % num_ots) + 1) for i in range(num_tasks)]
    else:
        heuristic = [min((i // block_size) + 1, num_ots) for i in range(num_tasks)]
    scenario.heuristic_chromosomes = [heuristic]
    return scenario


def build_map_standard_scenario(num_tasks=30, num_ots=None, seed=0):
    return build_map_based_scenario(num_tasks=num_tasks, num_ots=num_ots, seed=seed, cooperative=False)


def build_map_cooperative_scenario(num_tasks=12, num_ots=None, cluster_size=3, seed=0):
    return build_map_based_scenario(
        num_tasks=num_tasks, num_ots=num_ots, seed=seed, cooperative=True, cluster_size=cluster_size
    )
