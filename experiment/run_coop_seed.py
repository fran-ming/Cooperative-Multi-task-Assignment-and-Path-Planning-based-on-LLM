# -*- coding: utf-8 -*-
"""Cooperative multi-task CMAPP experiment with multiple repeat runs."""
import os
import sys
import json
import time
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from data.config import EXPERIMENT_CONFIG, RESULTS_DIR
from nsga.decoder import build_map_cooperative_scenario
from nsga.nsgaii import NSGA2Solver
from algorithms.llm_pure import LLMPureSolver
from algorithms.llm_nsga import LLMNSGA2Solver
from llm.client import LLMClient

ALGOS = ["NSGA-II", "LLM", "LLM-NSGA"]


def build_solver(algo, seed, client=None):
    kwargs = dict(
        population_size=EXPERIMENT_CONFIG["population_size"],
        generations=EXPERIMENT_CONFIG["generations"],
        crossover_rate=0.9,
        mutation_rate=0.6,
        seed=seed,
    )
    if algo == "NSGA-II":
        return NSGA2Solver(**kwargs), None
    if algo == "LLM":
        client = client or LLMClient()
        return LLMPureSolver(**kwargs, client=client), client
    if algo == "LLM-NSGA":
        client = client or LLMClient()
        return LLMNSGA2Solver(**kwargs, client=client), client
    raise ValueError("Unknown algorithm: %s" % algo)


def solve_once(algo, run_idx, algo_seed):
    scenario = build_map_cooperative_scenario(
        num_tasks=EXPERIMENT_CONFIG["cooperative_tasks"],
        cluster_size=EXPERIMENT_CONFIG["tasks_per_multi_cluster"],
        seed=run_idx,
    )
    client = LLMClient() if algo != "NSGA-II" else None
    solver, client = build_solver(algo, algo_seed, client=client)
    start_time = time.time()
    best, history = solver.solve(scenario)
    runtime = time.time() - start_time
    stats = solver.get_stats() if hasattr(solver, "get_stats") else {}
    return best, history, runtime, stats


def main():
    results_dir = RESULTS_DIR / "cooperative_tasks"
    history_dir = results_dir / "convergence_history"
    results_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    num_tasks = EXPERIMENT_CONFIG["cooperative_tasks"]
    cluster_size = EXPERIMENT_CONFIG["tasks_per_multi_cluster"]
    generations = EXPERIMENT_CONFIG["generations"]
    repeat_count = EXPERIMENT_CONFIG["repeat_count"]

    all_rows = []
    all_history = {algo: [] for algo in ALGOS}

    for run_idx in range(1, repeat_count + 1):
        print("\n================ Run %d/%d ================" % (run_idx, repeat_count))
        for a_idx, algo in enumerate(ALGOS):
            algo_seed = 1000 * (a_idx + 1) + run_idx
            print("\n--- Running %s | run_idx=%d | algo_seed=%d ---" % (algo, run_idx, algo_seed))
            best, history, runtime, stats = solve_once(algo, run_idx, algo_seed)

            for h in history:
                all_history[algo].append({
                    "run_idx": run_idx,
                    "gen": h["gen"],
                    "fitness_J": h["fitness_J"],
                })

            metrics = best.metrics
            row = {
                "run_idx": run_idx,
                "algorithm": algo,
                "num_tasks": num_tasks,
                "cluster_size": cluster_size,
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
            }
            all_rows.append(row)
            print(json.dumps(row, indent=2, ensure_ascii=False))

    raw_csv = results_dir / f"cooperative_comparison_raw_{num_tasks}_{cluster_size}.csv"
    summary_csv = results_dir / f"cooperative_comparison_summary_{num_tasks}_{cluster_size}.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(raw_csv, index=False, encoding="utf-8-sig")

    metric_cols = [
        "fitness_J", "J_d", "J_m", "J_b", "J_t",
        "success_rate", "finished_tasks", "failed_tasks",
        "C_max", "total_distance", "runtime",
        "llm_calls", "llm_total_tokens", "llm_latency", "llm_fallback_calls",
    ]
    summary_rows = []
    for algo, sub in df.groupby("algorithm"):
        row = {"algorithm": algo}
        for col in metric_cols:
            if col in sub.columns:
                row["%s_mean" % col] = sub[col].mean()
                row["%s_std" % col] = sub[col].std(ddof=1) if len(sub) > 1 else 0.0
                if col in ["success_rate", "finished_tasks", "llm_calls", "llm_total_tokens", "llm_latency", "llm_fallback_calls"]:
                    row["%s_best" % col] = sub[col].max()
                else:
                    row["%s_best" % col] = sub[col].min()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    for algo in ALGOS:
        if all_history[algo]:
            hist_csv = history_dir / f"{algo}_cooperative_history_{num_tasks}_{cluster_size}.csv"
            pd.DataFrame(all_history[algo]).to_csv(hist_csv, index=False, encoding="utf-8-sig")

    print("\nSummary saved to: %s" % summary_csv)
    print("Total rows: %d" % len(df))
    print("\n=== Runtime / LLM budget comparison ===")
    print(df.groupby("algorithm")[["runtime", "llm_calls", "llm_total_tokens"]].mean().round(3))


if __name__ == "__main__":
    main()
