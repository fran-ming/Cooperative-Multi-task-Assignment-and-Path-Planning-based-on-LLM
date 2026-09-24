# -*- coding: utf-8 -*-
"""MPaGE-CMAPP path-planning visualization (meta-heuristic evolution).

MPaGE (Meta-heuristic Programming with Adaptive Genetic Evolution) evolves
search *heuristics* (not solutions) using the existing NSGA-II / Decoder /
ObjectiveEvaluator stack as its ground-truth feasibility evaluator (project
guide sections 2 and 37).  Each heuristic seeds the inner NSGA-II, which then
searches the solution space; heuristic-level objectives (hypervolume,
feasible-rate, runtime) drive Pareto-grid selection and semantic clustering.

Outputs:
- results/figures/pathplanner/mpage_route_10task.png
- results/figures/pathplanner/mpage_gantt_10task.png
- results/figures/pathplanner/mpage_simulation.gif
- results/figures/pathplanner/mpage_schedule_10task.csv
Appends one row (algorithm="MPaGE-CMAPP") to:
- results/figures/pathplanner/llm_nsga_comparison_single_run_14.csv
"""
import io
import os
import sys
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are saved, never popped up
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.config import RESULTS_DIR
from nsga.decoder import build_standard_scenario, build_cooperative_scenario
from MPAGE.solver import MPAGESolver
from MPAGE.config import MPAGE_CONFIG

SAVE_DIR = RESULTS_DIR / "figures" / "pathplanner"

# MPaGE meta-level configuration.  LLM is disabled by default so the experiment
# runs deterministically without an API dependency (guide section 43); set the
# environment variable MPAGE_ENABLE_LLM=1 to enable LLM-driven heuristic
# reflection / mutation / crossover.
EXPERIMENT_MPAGE_CONFIG = dict(MPAGE_CONFIG)
EXPERIMENT_MPAGE_CONFIG.update({
    "enable_llm": os.getenv("MPAGE_ENABLE_LLM", "0") == "1",
    "heuristic_population_size": 6,
    "meta_generations": 6,
    "inner_generations": 10,
    "inner_population_size": 40,
    "elite_heuristics": 4,
    "grid_bins": 5,
    "semantic_clusters": 3,
    "mutation_probability": 0.8,
    "crossover_probability": 0.7,
    "heuristic_seed_ratio": 0.3,
    "seed": 42,
})


def run_mpage(scenario, seed=42):
    """Run the MPaGE meta-level solver and return its best CMAPP solution."""
    solver = MPAGESolver(config=EXPERIMENT_MPAGE_CONFIG, seed=seed)
    result = solver.solve(scenario)
    best = result["best_solution"]
    history = result["history"]
    stats = solver.get_stats()
    model_name = ("MPaGE-CMAPP" if not EXPERIMENT_MPAGE_CONFIG["enable_llm"]
                  else "MPaGE-CMAPP (LLM)")
    return best, history, stats, model_name


def _line_points(scenario, source, target, max_nodes=120):
    path = scenario.oracle.shortest_path(source, target)
    if not path:
        path = [source, target]
    return [(scenario.graph.node_position(n)[0], scenario.graph.node_position(n)[1])
            for n in path[:max_nodes]]
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
    ax.set_title("MPaGE-CMAPP Route Plan", fontsize=14)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)
    path = output_dir / "mpage_route_10task.png"
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
        ax.text((rec.arrival + rec.finish) / 2.0, rec.ot_id, rec.task_id,
                ha="center", va="center", fontsize=7)
    for rec in best.decode_result.records:
        if not rec.success:
            ax.scatter([rec.arrival_raw if rec.arrival_raw > 0 else 0], [rec.ot_id],
                       marker="x", color="red", s=70)
    ax.set_yticks([ot.ot_id for ot in scenario.ots])
    ax.set_yticklabels(["OT%d" % ot.ot_id for ot in scenario.ots])
    ax.set_xlabel("Time")
    ax.set_ylabel("Object target")
    ax.set_title("MPaGE-CMAPP Gantt Chart")
    ax.grid(axis="x", alpha=0.2)
    path = output_dir / "mpage_gantt_10task.png"
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
            # Travel to staging.
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
            # Execution from staging to terminal.
            exec_path = scenario.oracle.shortest_path(rec.staging, rec.terminal)
            if not exec_path:
                exec_path = [rec.staging, rec.terminal]
            t0 = rec.arrival
            for u, v in zip(exec_path, exec_path[1:]):
                d = scenario.graph.edge_length.get((u, v), scenario.graph.euclidean_length(u, v))
                dt = d / ot.speed
                segments.append((scenario.graph.node_position(u)[0], scenario.graph.node_position(u)[1],
                                 scenario.graph.node_position(v)[0], scenario.graph.node_position(v)[1],
                                 t0, min(rec.finish, t0 + dt)))
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

        # Mark each task start location with a light-blue square.
        for task in scenario.tasks:
            x, y = scenario.graph.node_position(task.staging)
            ax.scatter([x], [y], s=55, marker="s", color="#cfeeff",
                       edgecolors="black", linewidths=0.5, zorder=4)

        # Draw the VUT route in purple dashed line and animate the VUT as a black circle.
        if vut_path_nodes and len(vut_path_nodes) >= 2:
            xs = [scenario.graph.node_position(n)[0] for n in vut_path_nodes]
            ys = [scenario.graph.node_position(n)[1] for n in vut_path_nodes]
            ax.plot(xs, ys, color="purple", linewidth=2.2, linestyle="--",
                    dashes=(6, 4), alpha=0.9, zorder=3, label="VUT route")
            vut_pos = _pos_along_path_nodes(scenario.graph, vut_path_nodes, (frame_idx + 1) / max(1, frames))
            if vut_pos is not None:
                ax.scatter([vut_pos[0]], [vut_pos[1]], s=90, color="black",
                           edgecolors="black", linewidths=0.3, zorder=6)

        for ot in scenario.ots:
            pos = _pos_at_time(timelines.get(ot.ot_id, []), t)
            if pos is None:
                pos = scenario.graph.node_position(ot.initial_position)
            ax.scatter([pos[0]], [pos[1]], s=110, color=colors[(ot.ot_id - 1) % len(colors)],
                       edgecolors="black", linewidths=0.6, zorder=5)

        ax.set_title("MPaGE-CMAPP simulation  t=%.1fs" % t)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.15)
        if vut_path_nodes and len(vut_path_nodes) >= 2:
            ax.legend(loc="upper right")
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

    # Headless-safe: no plt.show(), the GIF is saved straight to disk.
    gif_path = output_dir / "mpage_simulation.gif"
    try:
        animation_obj.save(gif_path, writer="pillow", fps=max(1, round(1000 / 120)))
    except Exception:
        # Fallback: save a GIF from the static frame sequence for environments
        # without a functional animation writer.
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
    path = output_dir / "mpage_schedule_10task.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved schedule CSV: %s" % path)
def main():
    output_dir = SAVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use decoder's predefined task/OT positions and VUT key nodes.  The
    # cooperative scenario groups task units by semantic type so every unit in
    # one sync group shares a single non-overlapping time window (guide section
    # 2).  Kept at 14 tasks to stay consistent with the OR-Tools/LLM-NSGA
    # comparison rows already recorded in *_single_run_14.csv.
    #修改协作参数
    print("Building 14-task cooperative CMAPP scenario from decoder defaults...")
    scenario = build_cooperative_scenario(num_tasks=14, seed=42)
    print("Running MPaGE-CMAPP...")
    start = time.time()
    best, history, stats, model_name = run_mpage(scenario, seed=42)
    runtime = time.time() - start
    print("MPaGE-CMAPP objectives [Jd,Jm,Jb,Jt]:", best.objectives)
    print("MPaGE-CMAPP stats:", stats)
    plot_routes_on_map(scenario, best, output_dir)
    plot_gantt(scenario, best, output_dir)
    write_schedule_csv(scenario, best, output_dir)
    make_animation(scenario, best, output_dir, frames=30)
    # Append a single-run results row using the same columns as the LLM/OR
    # reference files.
    try:
        num_tasks = len(scenario.tasks)
        metrics = best.metrics
        row = {
            "seed": 15,
            "algorithm": "MPaGE-CMAPP",
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