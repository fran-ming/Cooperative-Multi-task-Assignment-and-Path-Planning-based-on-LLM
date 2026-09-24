# -*- coding: utf-8 -*-
"""OR-Tools CP-SAT CMAPP path-planning visualization.

Outputs:
- results/figures/pathplanner/or_nsga_route_10task.png
- results/figures/pathplanner/or_nsga_gantt_10task.png
- results/figures/pathplanner/or_nsga_simulation.gif
- results/figures/pathplanner/or_nsga_schedule_10task.csv
"""
import io
import math
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.config import EXPERIMENT_CONFIG, RESULTS_DIR, SERVICE_VEH_SPEED_MPS, SYNC_TOLERANCE
from nsga.decoder import (
    INF,
    build_map_standard_scenario,
    build_standard_scenario,
    build_cooperative_scenario,
)

try:
    from ortools.sat.python import cp_model
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("OR-Tools is required. Please install it with: python -m pip install ortools") from exc

SAVE_DIR = RESULTS_DIR / "figures" / "pathplanner"

# CP-SAT works with integer variables. Times and distances are scaled to
# milliseconds so VUT-triggered time windows, OT sequencing, and path-cost
# calculations retain their physical meaning while remaining integral.
TIME_SCALE = 1000
BIG_M = 10000000


class _ORScheduleResult:
    def __init__(self, scenario, records):
        self.records = records
        self.task_success = [r.success for r in records]
        self.sync_feedback = []
        self.objectives = None
        self.metrics = {}
        self.ot_final_positions = {}
        self.ot_available_times = {}
        self.ot_assigned_task_ids = defaultdict(list)
        self.ot_travel_dist = defaultdict(float)
        self.assignment_counts = defaultdict(int)

        for rec in records:
            if rec.success:
                self.ot_assigned_task_ids[rec.ot_id].append(rec.task_id)
                self.ot_travel_dist[rec.ot_id] += rec.total_travel

        for ot_id, _ in enumerate(scenario.ots, 1):
            self.assignment_counts[ot_id] = 0

        for rec in records:
            if rec.success and int(rec.ot_id) in self.assignment_counts:
                self.assignment_counts[int(rec.ot_id)] += 1

        failed = sum(1 for s in self.task_success if not s)
        finished = len(self.task_success) - failed
        makespan = max((r.finish for r in records if r.success), default=0.0)
        counts = list(self.assignment_counts.values()) or [0]
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
            "fitness_J": float(failed) * 1000.0 + float(makespan) * 100.0 + float(imbalance) * 10.0 + float(total_travel) * 0.1,
            "success_rate": (finished / float(max(1, len(scenario.tasks)))),
            "finished_tasks": finished,
            "failed_tasks": failed,
            "C_max": float(makespan),
            "n_max": int(max(counts)),
            "n_min": int(min(counts)),
            "total_distance": float(total_travel),
        }


def _distance(graph, oracle, start, end):
    if start == end:
        return 0.0
    d = oracle.travel_distance(start, end)
    if d >= INF - 1:
        return float("inf")
    return d


def _task_candidate_costs(scenario):
    costs = {}
    for task in scenario.tasks:
        costs[task.index] = []
        for ot in scenario.ots:
            to_staging = _distance(scenario.graph, scenario.oracle, ot.initial_position, task.staging)
            staging_to_terminal = _distance(scenario.graph, scenario.oracle, task.staging, task.terminal)
            if math.isinf(to_staging) or math.isinf(staging_to_terminal):
                continue
            travel_total = to_staging + staging_to_terminal
            costs[task.index].append({
                "ot_id": ot.ot_id,
                "travel_total": travel_total,
                "distance_to_staging": to_staging,
                "distance_staging_to_terminal": staging_to_terminal,
            })
    return costs


def _scaled_int(value):
    return int(round(float(value) * TIME_SCALE))


def _build_cmapp_model(scenario):
    """Build the complete integer CMAPP model.

    Decision variables follow the paper notation:
    x[i,k]: task i is assigned to OT k;
    y[i]: task i is executed;
    z[i,j,k]: on OT k, task i precedes task j;
    t[i]: task i start/arrival time (when executed);
    C_k: completion time of OT k;
    C_max: global makespan.

    Returns ``(model, variables)``.
    """
    tasks = scenario.tasks
    ots = scenario.ots
    task_costs = _task_candidate_costs(scenario)

    model = cp_model.CpModel()
    x = {}
    y = {}
    t = {}
    z = {}
    C_k = {}

    duration_scaled = {task.index: _scaled_int(task.duration) for task in tasks}
    time_lb = {task.index: _scaled_int(task.tw[0]) for task in tasks}
    time_ub = {task.index: _scaled_int(task.tw[1]) for task in tasks}

    # Assignment and execution indicators.
    for task in tasks:
        i = task.index
        y[i] = model.NewBoolVar(f"y_{i}")
        choices = []
        for info in task_costs[i]:
            k = info["ot_id"]
            variable = model.NewBoolVar(f"x_{i}_{k}")
            x[(i, k)] = variable
            choices.append(variable)
        model.Add(sum(choices) == y[i])
        
        t[i] = model.NewIntVar(0, BIG_M, f"t_{i}")
        model.Add(t[i] == 0).OnlyEnforceIf(y[i].Not())

    # VUT-triggered time-window constraints: executed tasks must start once
    # during [a_i, b_i]. Waiting is allowed until the earliest side of the
    # window, so this is enforced only for successfully assigned tasks.
    for task in tasks:
        i = task.index
        model.Add(t[i] >= time_lb[i]).OnlyEnforceIf(y[i])
        model.Add(t[i] <= time_ub[i]).OnlyEnforceIf(y[i])

    # Initial-leg travel constraints: an OT cannot arrive at task i before
    # its initial position/dispatch time plus the travel time to staging_i.
    init_time = {}
    init_distance = {}
    exec_distance = {}
    for task in tasks:
        i = task.index
        for info in task_costs[i]:
            k = info["ot_id"]
            ot = ots[k - 1]
            init_time[(i, k)] = _scaled_int(
                ot.initial_available_time + info["distance_to_staging"] / ot.speed
            )
            init_distance[(i, k)] = _scaled_int(info["distance_to_staging"])
            exec_distance[(i, k)] = _scaled_int(info["distance_staging_to_terminal"])
            model.Add(t[i] >= init_time[(i, k)] - BIG_M * (1 - x[(i, k)]))

    # Task-order coupling z[i,j,k]. If i and j are both assigned to OT k,
    # exactly one of the two ordering variables must be active.
    transition_distance = {}
    transition_time = {}
    for i_idx, task_i in enumerate(tasks):
        i = task_i.index
        for j_idx in range(i_idx + 1, len(tasks)):
            task_j = tasks[j_idx]
            j = task_j.index
            for ot in ots:
                k = ot.ot_id
                if (i, k) not in x or (j, k) not in x:
                    continue
                z_ij = model.NewBoolVar(f"z_{i}_{j}_{k}")
                z_ji = model.NewBoolVar(f"z_{j}_{i}_{k}")
                z[(i, j, k)] = z_ij
                z[(j, i, k)] = z_ji
                model.Add(z_ij + z_ji <= 1)
                model.Add(z_ij + z_ji >= x[(i, k)] + x[(j, k)] - 1)

                d_ij = _distance(scenario.graph, scenario.oracle, task_i.terminal, task_j.staging)
                d_ji = _distance(scenario.graph, scenario.oracle, task_j.terminal, task_i.staging)
                tau_ij = _scaled_int(d_ij / ot.speed) if not math.isinf(d_ij) else BIG_M
                tau_ji = _scaled_int(d_ji / ot.speed) if not math.isinf(d_ji) else BIG_M
                model.Add(
                    t[j] >= t[i] + duration_scaled[i] + tau_ij - BIG_M * (1 - z_ij)
                )
                model.Add(
                    t[i] >= t[j] + duration_scaled[j] + tau_ji - BIG_M * (1 - z_ji)
                )
                transition_distance[(i, j, k)] = _scaled_int(d_ij) if not math.isinf(d_ij) else BIG_M
                transition_distance[(j, i, k)] = _scaled_int(d_ji) if not math.isinf(d_ji) else BIG_M

    # OT completion times and global makespan.
    for ot in ots:
        C_k[ot.ot_id] = model.NewIntVar(0, BIG_M, f"C_{ot.ot_id}")
    C_max = model.NewIntVar(0, BIG_M, "C_max")
    for task in tasks:
        i = task.index
        for k in [ot.ot_id for ot in ots]:
            if (i, k) not in x:
                continue
            model.Add(C_k[k] >= t[i] + duration_scaled[i] - BIG_M * (1 - x[(i, k)]))
            model.Add(C_max >= C_k[k])

    # Workload imbalance.
    counts = {}
    for ot in ots:
        k = ot.ot_id
        counts[k] = model.NewIntVar(0, len(tasks), f"count_{k}")
        model.Add(counts[k] == sum(x[(task.index, k)] for task in tasks if (task.index, k) in x))
    max_count = model.NewIntVar(0, len(tasks), "max_count")
    min_count = model.NewIntVar(0, len(tasks), "min_count")
    for ot in ots:
        model.Add(max_count >= counts[ot.ot_id])
        model.Add(min_count <= counts[ot.ot_id])
    imbalance = model.NewIntVar(0, len(tasks), "imbalance")
    model.Add(imbalance == max_count - min_count)

    # Total travel distance. For an executed task i on OT k, its inbound leg
    # starts from the initial depot if it has no predecessor, or from the
    # terminal of its predecessor on the same OT when z[prev,i,k] is active.
    travel_terms = []
    for task in tasks:
        i = task.index
        for info in task_costs[i]:
            k = info["ot_id"]
            travel_terms.append(
                (init_distance[(i, k)] + exec_distance[(i, k)]) * x[(i, k)]
            )
    for (i, j, k), z_var in z.items():
        delta = transition_distance[(i, j, k)] - init_distance[(j, k)]
        travel_terms.append(delta * z_var)
    travel_total = model.NewIntVar(0, 10 * BIG_M, "travel_total")
    model.Add(travel_total == sum(travel_terms))

    failed_count = model.NewIntVar(0, len(tasks), "failed_count")
    model.Add(failed_count == sum(1 - y[task.index] for task in tasks))

    # Staging-resource occupancy constraints: different OTs cannot occupy the
    # same task staging conflict region at the same time.
    staging_intervals = {}
    staging_intervals_noncoop = {}
    staging_intervals_coop = {}
    for task in tasks:
        i = task.index
        group = getattr(task, "sync_group", None)
        staging_intervals.setdefault(task.staging, [])
        if group is None:
            staging_intervals_noncoop.setdefault(task.staging, [])
        for info in task_costs[i]:
            k = info["ot_id"]
            end_var = model.NewIntVar(0, BIG_M, f"end_{i}_{k}")
            model.Add(end_var == t[i] + duration_scaled[i])
            interval = model.NewOptionalIntervalVar(
                t[i], duration_scaled[i], end_var, x[(i, k)], f"iv_{i}_{k}"
            )
            staging_intervals[task.staging].append(interval)
            if group is None:
                staging_intervals_noncoop[task.staging].append(interval)
            else:
                staging_intervals_coop.setdefault(task.staging, []).append(interval)

    # Enforce non-overlap only for non-cooperative tasks (sync_group is None).
    # Cooperative tasks are allowed to overlap at the same staging point.
    for intervals in staging_intervals_noncoop.values():
        if len(intervals) >= 2:
            model.AddNoOverlap(intervals)
    # Prevent overlaps between any non-cooperative interval and any
    # cooperative interval at the same staging point. Cooperative tasks may
    # overlap with each other, but not with non-cooperative ones.
    for staging, noncoop_intervals in staging_intervals_noncoop.items():
        coop_intervals = staging_intervals_coop.get(staging, [])
        if not coop_intervals:
            continue
        for n_iv in noncoop_intervals:
            for c_iv in coop_intervals:
                model.AddNoOverlap([n_iv, c_iv])

    # Cooperative synchronization constraints. The current map-standard
    # scenario has no sync_group, but the model includes the constraint so
    # cooperative scenarios are covered when passed by other experiments.
    groups = {}
    for task in tasks:
        group = getattr(task, "sync_group", None)
        if group:
            groups.setdefault(group, []).append(task)
    for group, group_tasks in groups.items():
        for left, right in combinations([task.index for task in group_tasks], 2):
            model.Add(t[left] - t[right] <= _scaled_int(SYNC_TOLERANCE)).OnlyEnforceIf([y[left], y[right]])
            model.Add(t[right] - t[left] <= _scaled_int(SYNC_TOLERANCE)).OnlyEnforceIf([y[left], y[right]])
        for k in [ot.ot_id for ot in ots]:
            pairs = [(left.index, right.index) for left, right in combinations(group_tasks, 2)]
            for left, right in pairs:
                if (left, k) in x and (right, k) in x:
                    model.Add(x[(left, k)] + x[(right, k)] <= 1)

    # Enforce group time-window exclusivity: any task not in a cooperative
    # group must not start inside the group's time window. Cooperative tasks
    # may overlap with each other, but other tasks must start either before
    # the group's lower bound or at/after the group's upper bound.
    for group, group_tasks in groups.items():
        # compute group's scaled window bounds
        group_lbs = [_scaled_int(task.tw[0]) for task in group_tasks]
        group_ubs = [_scaled_int(task.tw[1]) for task in group_tasks]
        if not group_lbs or not group_ubs:
            continue
        group_lb = min(group_lbs)
        group_ub = max(group_ubs)
        for task in tasks:
            if getattr(task, "sync_group", None) == group:
                continue
            j = task.index
            # boolean: task j starts before group window
            before_var = model.NewBoolVar(f"grp_{group}_before_{j}")
            # boolean: task j starts at/after group window upper bound
            after_var = model.NewBoolVar(f"grp_{group}_after_{j}")
            model.Add(t[j] <= group_lb).OnlyEnforceIf(before_var)
            model.Add(t[j] >= group_ub).OnlyEnforceIf(after_var)
            # If task j is executed (y[j]==1), then it must be either before or after
            model.Add(before_var + after_var >= y[j])

    variables = {
        "x": x,
        "y": y,
        "t": t,
        "z": z,
        "C_k": C_k,
        "C_max": C_max,
        "imbalance": imbalance,
        "travel_total": travel_total,
        "failed_count": failed_count,
    }
    return model, variables


def _solve_stage(model, objective_expr, stage_name):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 3600.0
    solver.parameters.num_search_workers = 16
    solver.parameters.log_search_progress = True
    model.Minimize(objective_expr)
    status = solver.Solve(model)
    print("OR-Tools stage %s status: %s" % (stage_name, solver.StatusName(status)))
    return solver, status


def _solve_cmapp_model(model, variables):
    """Solve the four objectives lexicographically.

    Order: J_d (failed tasks) -> J_m (makespan) -> J_b (imbalance) -> J_t
    (travel distance). Each stage fixes the previous stage's optimum, then
    optimizes the next objective. This preserves the Pareto ordering exactly
    rather than collapsing the four objectives into one arbitrary sum.
    """
    failed = variables["failed_count"]
    makespan = variables["C_max"]
    imbalance = variables["imbalance"]
    travel = variables["travel_total"]

    solver, status = _solve_stage(model, failed, "J_d")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible CMAPP solution found in J_d stage")
    best_failed = int(round(solver.Value(failed)))
    model.Add(failed == best_failed)

    solver, status = _solve_stage(model, makespan, "J_m")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible CMAPP solution found in J_m stage")
    best_makespan = int(round(solver.Value(makespan)))
    model.Add(makespan == best_makespan)

    solver, status = _solve_stage(model, imbalance, "J_b")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible CMAPP solution found in J_b stage")
    best_imbalance = int(round(solver.Value(imbalance)))
    model.Add(imbalance == best_imbalance)

    solver, status = _solve_stage(model, travel, "J_t")
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible CMAPP solution found in J_t stage")
    return solver, status


def _extract_records_from_solver(scenario, solver, variables):
    """Read the CP-SAT solution directly, without greedy re-scheduling."""
    tasks = scenario.tasks
    ots = scenario.ots
    x = variables["x"]
    y = variables["y"]
    t = variables["t"]

    assignment = {}
    for task in tasks:
        i = task.index
        if solver.Value(y[i]) < 0.5:
            assignment[i] = None
            continue
        selected_ot = None
        for ot in ots:
            k = ot.ot_id
            if (i, k) in x and solver.Value(x[(i, k)]) > 0.5:
                selected_ot = k
                break
        assignment[i] = selected_ot or 1

    ot_task_groups = defaultdict(list)
    for task in tasks:
        i = task.index
        k = assignment.get(i)
        if k is None:
            continue
        ot_task_groups[k].append((solver.Value(t[i]) / TIME_SCALE, task.index))
    for k in ot_task_groups:
        ot_task_groups[k].sort()

    ordering_by_ot = {}
    for k, entries in ot_task_groups.items():
        ordering_by_ot[k] = [task_idx for _, task_idx in entries]

    ot_state = {}
    for ot in ots:
        ot_state[ot.ot_id] = {
            "position": ot.initial_position,
            "time": ot.initial_available_time,
        }

    records = []
    for task in tasks:
        i = task.index
        k = assignment.get(i)
        if k is None:
            records.append(SimpleNamespace(
                task_idx=i,
                task_id=task.task_id,
                ot_id=0,
                staging=task.staging,
                terminal=task.terminal,
                travel_to_staging=0.0,
                travel_staging_to_terminal=0.0,
                arrival_raw=0.0,
                arrival=0.0,
                start=0.0,
                finish=0.0,
                success=False,
                fail_reason="unassigned",
                lateness=0.0,
                total_travel=0.0,
            ))
            continue

        ot = ots[k - 1]
        state = ot_state[k]
        to_staging = _distance(scenario.graph, scenario.oracle, state["position"], task.staging)
        staging_to_terminal = _distance(scenario.graph, scenario.oracle, task.staging, task.terminal)
        arrival_raw = state["time"] + (to_staging / ot.speed if not math.isinf(to_staging) else 0.0)
        arrival = solver.Value(t[i]) / TIME_SCALE
        finish = arrival + task.duration
        lateness = max(0.0, arrival - task.tw[1])
        total_travel = (0.0 if math.isinf(to_staging) else to_staging) + (
            0.0 if math.isinf(staging_to_terminal) else staging_to_terminal
        )
        records.append(SimpleNamespace(
            task_idx=i,
            task_id=task.task_id,
            ot_id=k,
            staging=task.staging,
            terminal=task.terminal,
            travel_to_staging=to_staging,
            travel_staging_to_terminal=staging_to_terminal,
            arrival_raw=arrival_raw,
            arrival=arrival,
            start=arrival,
            finish=finish,
            success=True,
            fail_reason="",
            lateness=lateness,
            total_travel=total_travel,
        ))
        state["position"] = task.terminal
        state["time"] = finish

    return records


def run_or_solver(scenario, seed=42, debug_print_llm=False):
    model, variables = _build_cmapp_model(scenario)
    solver, status = _solve_cmapp_model(model, variables)
    records = _extract_records_from_solver(scenario, solver, variables)
    decode_result = _ORScheduleResult(scenario, records)
    metrics = decode_result.metrics
    best = SimpleNamespace(
        objectives=decode_result.objectives,
        metrics=metrics,
        decode_result=SimpleNamespace(
            records=records,
            task_success=decode_result.task_success,
            sync_feedback=decode_result.sync_feedback,
            objectives=decode_result.objectives,
            metrics=metrics,
        ),
    )
    history = [{"objective": decode_result.objectives, "metrics": metrics}]
    stats = {
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "latency_seconds": 0.0,
        "fallback_calls": 0,
        "status": solver.StatusName(status),
    }
    return best, history, stats, "OR-Tools CP-SAT"


def _line_points(scenario, source, target, max_nodes=120):
    path = scenario.oracle.shortest_path(source, target)
    if not path:
        path = [source, target]
    return [(scenario.graph.node_position(n)[0], scenario.graph.node_position(n)[1]) for n in path[:max_nodes]]


def plot_routes_on_map(scenario, best, output_dir):
    fig, ax = plt.subplots(figsize=(14, 12))
    for u, neighbors in scenario.graph.adj.items():
        if u not in scenario.graph.nodes:
            continue
        for v in neighbors:
            if v not in scenario.graph.nodes:
                continue
            x1, y1 = scenario.graph.node_position(u)
            x2, y2 = scenario.graph.node_position(v)
            ax.plot([x1, x2], [y1, y2], color="#e6e6e6", linewidth=0.4, zorder=0)

    records_by_ot = {}
    for rec in best.decode_result.records:
        records_by_ot.setdefault(rec.ot_id, []).append(rec)

    colors = plt.cm.tab10.colors
    for ot in scenario.ots:
        color = colors[(ot.ot_id - 1) % len(colors)]
        x, y = scenario.graph.node_position(ot.initial_position)
        ax.scatter([x], [y], marker="s", s=100, color=color, edgecolors="black", linewidths=0.7, zorder=4)
        ax.annotate("OT%d" % ot.ot_id, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

        current = ot.initial_position
        recs = sorted(records_by_ot.get(ot.ot_id, []), key=lambda r: r.arrival)
        for rec in recs:
            if not rec.success:
                continue
            pts = _line_points(scenario, current, rec.staging)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=1.4, alpha=0.85, zorder=2)
            pts2 = _line_points(scenario, rec.staging, rec.terminal)
            if pts2:
                ax.plot([p[0] for p in pts2], [p[1] for p in pts2], color=color, linewidth=1.4, linestyle="--", alpha=0.85, zorder=2)
            x, y = scenario.graph.node_position(rec.staging)
            ax.scatter([x], [y], marker="o", s=45, color=color, edgecolors="black", linewidths=0.6, zorder=5)
            x2, y2 = scenario.graph.node_position(rec.terminal)
            ax.scatter([x2], [y2], marker="X", s=45, color=color, edgecolors="black", linewidths=0.6, zorder=5)
            ax.annotate(rec.task_id, (x, y), textcoords="offset points", xytext=(4, -8), fontsize=7)
            current = rec.terminal

    try:
        key_nodes = getattr(scenario, "vut_key_nodes", None) or []
        if key_nodes:
            path_nodes = scenario.oracle.expand_key_nodes_to_continuous_path(key_nodes)
            if path_nodes:
                xs = [scenario.graph.node_position(n)[0] for n in path_nodes]
                ys = [scenario.graph.node_position(n)[1] for n in path_nodes]
                ax.plot(xs, ys, color="purple", linewidth=2.0, alpha=0.9, zorder=3, label="VUT route")
                ax.legend()
    except Exception:
        pass
    ax.set_title("OR-Tools CP-SAT CMAPP Route Plan", fontsize=14)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    try:
        plt.show()
    except Exception:
        pass
    path = output_dir / "or_nsga_route_10task.png"
    fig.savefig(path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print("Saved route map: %s" % path)


def plot_gantt(scenario, best, output_dir):
    fig, ax = plt.subplots(figsize=(13, 6))
    records = [r for r in best.decode_result.records if r.success]
    colors = plt.cm.tab10.colors
    for rec in records:
        color = colors[(rec.ot_id - 1) % len(colors)]
        ax.barh(rec.ot_id, rec.finish - rec.arrival, left=rec.arrival, height=0.5, color=color, alpha=0.8)
        ax.text((rec.arrival + rec.finish) / 2.0, rec.ot_id, rec.task_id, ha="center", va="center", fontsize=7)
    for rec in best.decode_result.records:
        if not rec.success:
            ax.scatter([rec.arrival_raw if rec.arrival_raw > 0 else 0], [rec.ot_id], marker="x", color="red", s=70)
    ax.set_yticks([ot.ot_id for ot in scenario.ots])
    ax.set_yticklabels(["OT%d" % ot.ot_id for ot in scenario.ots])
    ax.set_xlabel("Time")
    ax.set_ylabel("Object target")
    ax.set_title("OR-Tools CP-SAT CMAPP Gantt Chart")
    ax.grid(axis="x", alpha=0.2)
    try:
        plt.show()
    except Exception:
        pass
    path = output_dir / "or_nsga_gantt_10task.png"
    fig.savefig(path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print("Saved Gantt chart: %s" % path)


def build_leg_timeline(scenario, best):
    """Return per-OT linear segments: (x1,y1,x2,y2,t1,t2)."""
    timelines = {}
    for ot in scenario.ots:
        segments = []
        cur_node = ot.initial_position
        cur_time = ot.initial_available_time
        recs = [r for r in best.decode_result.records if r.ot_id == ot.ot_id and r.success]
        recs.sort(key=lambda r: r.arrival)
        for rec in recs:
            path = scenario.oracle.shortest_path(cur_node, rec.staging)
            if not path:
                path = [cur_node, rec.staging]
            t0 = cur_time
            for u, v in zip(path, path[1:]):
                d = scenario.graph.edge_length.get((u, v), scenario.graph.euclidean_length(u, v))
                dt = d / ot.speed
                x1, y1 = scenario.graph.node_position(u)
                x2, y2 = scenario.graph.node_position(v)
                segments.append((x1, y1, x2, y2, t0, t0 + dt))
                t0 += dt
            exec_path = scenario.oracle.shortest_path(rec.staging, rec.terminal)
            if not exec_path:
                exec_path = [rec.staging, rec.terminal]
            t0 = rec.arrival
            for u, v in zip(exec_path, exec_path[1:]):
                d = scenario.graph.edge_length.get((u, v), scenario.graph.euclidean_length(u, v))
                dt = d / ot.speed
                segments.append((
                    scenario.graph.node_position(u)[0], scenario.graph.node_position(u)[1],
                    scenario.graph.node_position(v)[0], scenario.graph.node_position(v)[1],
                    t0, min(rec.finish, t0 + dt),
                ))
                t0 += dt
            cur_node = rec.terminal
            cur_time = rec.finish
        timelines[ot.ot_id] = segments
    return timelines


def _pos_at_time(segments, t):
    for x1, y1, x2, y2, t1, t2 in segments:
        if t1 <= t <= t2 + 1e-9:
            if t2 - t1 < 1e-9:
                return x2, y2
            frac = (t - t1) / (t2 - t1)
            return x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac
    if segments:
        x1, y1, x2, y2, t1, t2 = segments[-1]
        return x2, y2
    return None


def _pos_along_path_nodes(graph, path_nodes, fraction):
    if not path_nodes:
        return None
    if len(path_nodes) == 1:
        return graph.node_position(path_nodes[0])
    total_len = 0.0
    segments = []
    for u, v in zip(path_nodes, path_nodes[1:]):
        length = graph.edge_length.get((u, v), graph.euclidean_length(u, v))
        if length <= 0:
            continue
        total_len += length
        segments.append((u, v, length))
    if not segments:
        return graph.node_position(path_nodes[-1])

    target = min(max(fraction, 0.0), 1.0) * total_len
    walked = 0.0
    for u, v, length in segments:
        if walked + length >= target:
            x1, y1 = graph.node_position(u)
            x2, y2 = graph.node_position(v)
            local = (target - walked) / length if length > 0 else 0.0
            return x1 + (x2 - x1) * local, y1 + (y2 - y1) * local
        walked += length
    u, v, _ = segments[-1]
    return graph.node_position(v)


def make_animation(scenario, best, output_dir, frames=30):
    timelines = build_leg_timeline(scenario, best)
    max_t = max((best.metrics.get("C_max", 1.0) or 1.0), 1.0)
    colors = plt.cm.tab10.colors
    vut_path_nodes = []
    try:
        key_nodes = getattr(scenario, "vut_key_nodes", None) or []
        if key_nodes:
            vut_path_nodes = scenario.oracle.expand_key_nodes_to_continuous_path(key_nodes)
    except Exception:
        vut_path_nodes = []

    def render_frame(ax, frame_idx):
        ax.clear()
        t = max_t * frame_idx / max(1, frames - 1)
        for u, neighbors in scenario.graph.adj.items():
            if u not in scenario.graph.nodes:
                continue
            for v in neighbors:
                if v not in scenario.graph.nodes:
                    continue
                x1, y1 = scenario.graph.node_position(u)
                x2, y2 = scenario.graph.node_position(v)
                ax.plot([x1, x2], [y1, y2], color="#eeeeee", linewidth=0.3, zorder=0)

        for task in scenario.tasks:
            x, y = scenario.graph.node_position(task.staging)
            ax.scatter([x], [y], s=55, marker="s", color="#cfeeff", edgecolors="black", linewidths=0.5, zorder=4)

        if vut_path_nodes and len(vut_path_nodes) >= 2:
            xs = [scenario.graph.node_position(n)[0] for n in vut_path_nodes]
            ys = [scenario.graph.node_position(n)[1] for n in vut_path_nodes]
            ax.plot(xs, ys, color="purple", linewidth=2.2, linestyle="--", dashes=(6, 4), alpha=0.9, zorder=3, label="VUT route")
            vut_pos = _pos_along_path_nodes(scenario.graph, vut_path_nodes, (frame_idx + 1) / max(1, frames))
            if vut_pos is not None:
                ax.scatter([vut_pos[0]], [vut_pos[1]], s=90, color="black", edgecolors="black", linewidths=0.3, zorder=6)

        for ot in scenario.ots:
            pos = _pos_at_time(timelines.get(ot.ot_id, []), t)
            if pos is None:
                pos = scenario.graph.node_position(ot.initial_position)
            ax.scatter([pos[0]], [pos[1]], s=110, color=colors[(ot.ot_id - 1) % len(colors)], edgecolors="black", linewidths=0.6, zorder=5)

        ax.set_title("OR-Tools CP-SAT simulation  t=%.1fs" % t)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right") if vut_path_nodes and len(vut_path_nodes) >= 2 else None
        return []

    display_fig, display_ax = plt.subplots(figsize=(9, 8))
    animation_obj = animation.FuncAnimation(
        display_fig,
        lambda i: render_frame(display_ax, i),
        frames=frames,
        interval=120,
        blit=False,
        repeat=True,
    )

    try:
        plt.show()
    except Exception:
        pass

    gif_path = output_dir / "or_nsga_simulation.gif"
    try:
        animation_obj.save(gif_path, writer="pillow", fps=max(1, round(1000 / 120)))
    except Exception:
        images = []
        for frame_idx in range(frames):
            fig, ax = plt.subplots(figsize=(9, 8))
            render_frame(ax, frame_idx)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=1000)
            plt.close(fig)
            buf.seek(0)
            images.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
        if images:
            images[0].save(gif_path, save_all=True, append_images=images[1:], duration=120, loop=0)
    print("Saved simulation GIF: %s" % gif_path)


def write_schedule_csv(scenario, best, output_dir):
    rows = []
    for rec in best.decode_result.records:
        task = scenario.tasks[rec.task_idx]
        rows.append({
            "task_id": rec.task_id,
            "event_id": task.event_id,
            "sync_group": task.sync_group or "",
            "ot_id": rec.ot_id,
            "staging": rec.staging,
            "terminal": rec.terminal,
            "arrival_raw": round(rec.arrival_raw, 4),
            "arrival": round(rec.arrival, 4),
            "start": round(rec.start, 4),
            "finish": round(rec.finish, 4),
            "success": int(rec.success),
            "fail_reason": rec.fail_reason,
            "lateness": round(rec.lateness, 4),
            "travel_distance": round(rec.total_travel, 4),
        })
    path = output_dir / "or_nsga_schedule_10task.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved schedule CSV: %s" % path)


def main():
    output_dir = SAVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use decoder's predefined task/OT positions and VUT key nodes.
    # Set `use_cooperative` to True to build a cooperative scenario.
    use_cooperative = True
    #修改任务参数
    if use_cooperative:
        print("Building 14-task cooperative CMAPP scenario from decoder defaults...")
        scenario = build_cooperative_scenario(num_tasks=15, seed=42)
    else:
        print("Building 10-task CMAPP scenario from decoder defaults...")
        scenario = build_standard_scenario(num_tasks=10, seed=42)
    print("Running OR-Tools CP-SAT...")
    start = time.time()
    best, history, stats, model_name = run_or_solver(scenario, seed=42)
    runtime = time.time() - start
    print("OR-Tools objectives [Jd,Jm,Jb,Jt]:", best.objectives)
    print("OR-Tools stats:", stats)
    plot_routes_on_map(scenario, best, output_dir)
    plot_gantt(scenario, best, output_dir)
    write_schedule_csv(scenario, best, output_dir)
    make_animation(scenario, best, output_dir, frames=30)
    try:
        num_tasks = len(scenario.tasks)
        metrics = best.metrics
        row = {
            #修改seed
            "seed": 15,
            "algorithm": "OR-Tools",
            "num_tasks": num_tasks,
            "runtime": runtime,
            "fitness_J": metrics.get("fitness_J", 0),
            "J_d": metrics.get("J_d", 0),
            "J_m": metrics.get("J_m", 0),
            "J_b": metrics.get("J_b", 0),
            "J_t": metrics.get("J_t", 0),
            "success_rate": metrics.get("success_rate", 0),
            "finished_tasks": metrics.get("finished_tasks", 0),
            "failed_tasks": metrics.get("failed_tasks", 0),
            "C_max": metrics.get("C_max", 0),
            "n_max": metrics.get("n_max", 0),
            "n_min": metrics.get("n_min", 0),
            "total_distance": metrics.get("total_distance", 0),
            "llm_calls": stats.get("calls", 0),
            "llm_input_tokens": stats.get("input_tokens", 0),
            "llm_output_tokens": stats.get("output_tokens", 0),
            "llm_total_tokens": stats.get("total_tokens", 0),
            "llm_latency": stats.get("latency_seconds", 0.0),
            "llm_fallback_calls": stats.get("fallback_calls", 0),
            "model-name": model_name,
        }
        cols = [
            "seed","algorithm","num_tasks","runtime","fitness_J","J_d","J_m","J_b","J_t",
            "success_rate","finished_tasks","failed_tasks","C_max","n_max","n_min","total_distance",
            "llm_calls","llm_input_tokens","llm_output_tokens","llm_total_tokens","llm_latency","llm_fallback_calls",
            "model-name",
        ]
        results_path = output_dir / f"llm_nsga_comparison_single_run_{num_tasks}.csv"
        if results_path.exists():
            existing = pd.read_csv(results_path)
            if "model-name" not in existing.columns:
                existing["model-name"] = ""
            combined = pd.concat([
                existing[cols] if set(cols).issubset(existing.columns) else existing.reindex(columns=cols, fill_value=""),
                pd.DataFrame([row], columns=cols),
            ], ignore_index=True)
            combined.to_csv(results_path, index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame([row], columns=cols).to_csv(results_path, index=False, encoding="utf-8-sig")
        print("Appended single-run result to: %s" % results_path)
    except Exception as e:
        print("Failed to write single-run results CSV:", e)
    print("All visualization files saved to: %s" % output_dir)


if __name__ == "__main__":
    main()
