# -*- coding: utf-8 -*-
"""Multi-seed CMAPP experiment: NSGA-II vs LLM-only vs LLM-NSGA.

The three solvers share the same scenario, decoder, objective definitions and
evaluation budget. Results are written to results/ in the same style as the
previous baseline experiment scripts.
"""
import os
import sys
import json
import time
from pathlib import Path
import argparse

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
# Inline overrides: set these variables directly to configure experiments
# Edit below values (integers) to override without using the command line.
# Example: set `task = 16` and `ot = 6` and run the script normally.
task = 20  # set to int to override number of tasks, e.g. task = 16
ot = 10    # set to int to override number of OTs, e.g. ot = 6

# CLI overrides (still supported). Parsed values are written to globals,
# but explicit inline `task` / `ot` variables above take precedence.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--target-tasks", type=int, help="Override number of tasks for standard scenario")
parser.add_argument("--ot-count", type=int, help="Override number of OTs in the scenario")
args, _ = parser.parse_known_args()
if args.target_tasks is not None:
    globals()["_CLI_TARGET_TASKS"] = args.target_tasks
if args.ot_count is not None:
    globals()["_CLI_OT_COUNT"] = args.ot_count

# Inline variables override CLI if set
if task is not None:
    globals()["_CLI_TARGET_TASKS"] = int(task)
if ot is not None:
    globals()["_CLI_OT_COUNT"] = int(ot)

from data.config import EXPERIMENT_CONFIG, RESULTS_DIR
from nsga.decoder import build_map_standard_scenario
from nsga.nsgaii import NSGA2Solver
from algorithms.llm_pure import LLMPureSolver
from algorithms.llm_nsga import LLMNSGA2Solver
from llm.client import LLMClient

ALGOS = ["NSGA-II", "LLM", "LLM-NSGA"]


def save_partial_results(raw_rows, history_by_algo, target_tasks, generations, results_dir, history_dir):
    raw_csv = results_dir / f"llm_nsga_comparison_multi_seed_raw_{target_tasks}.csv"
    summary_csv = results_dir / f"llm_nsga_comparison_multi_seed_summary_{target_tasks}.csv"

    if raw_rows:
        df = pd.DataFrame(raw_rows)
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
        if summary_rows:
            pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    for algo in ALGOS:
        if history_by_algo.get(algo):
            hist_csv = history_dir / f"{algo}_history_{target_tasks}_{generations}.csv"
            pd.DataFrame(history_by_algo[algo]).to_csv(hist_csv, index=False, encoding="utf-8-sig")


def build_solver(algo, seed, client=None):
    pop_size = EXPERIMENT_CONFIG["population_size"]
    generations = EXPERIMENT_CONFIG["generations"]
    kwargs = dict(
        population_size=pop_size,
        generations=generations,
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


def solve_once(algo, instance_seed, algo_seed):
    # Map-based random scenario: tasks are selected from a random VUT
    # trajectory and OTs start within two hops of those task points.
    target_tasks = globals().get("_CLI_TARGET_TASKS") or EXPERIMENT_CONFIG["target_tasks"]
    ot_count = globals().get("_CLI_OT_COUNT")
    scenario = build_map_standard_scenario(
        num_tasks=target_tasks,
        num_ots=ot_count,
        seed=instance_seed,
    )
    client = LLMClient() if algo != "NSGA-II" else None
    solver, client = build_solver(algo, algo_seed, client=client)
    start_time = time.time()
    best, history = solver.solve(scenario)
    runtime = time.time() - start_time
    stats = solver.get_stats() if hasattr(solver, "get_stats") else {}
    return best, history, runtime, stats


def main():
    results_dir = RESULTS_DIR / "multi_seed"
    history_dir = results_dir / "convergence_history"
    results_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    target_tasks = globals().get("_CLI_TARGET_TASKS") or EXPERIMENT_CONFIG["target_tasks"]
    generations = EXPERIMENT_CONFIG["generations"]
    seeds = list(range(EXPERIMENT_CONFIG["seeds"]))

    all_rows = []
    all_history = {algo: [] for algo in ALGOS}

    for s_idx, seed in enumerate(seeds):
        print("\n================ Seed %d (%d/%d) ================" % (seed, s_idx + 1, len(seeds)))
        for a_idx, algo in enumerate(ALGOS):
            algo_seed = seed + 1000 * (a_idx + 1)
            print("\n--- Running %s | instance_seed=%d | algo_seed=%d ---" % (algo, seed, algo_seed))
            best, history, runtime, stats = solve_once(algo, seed, algo_seed)

            for h in history:
                all_history[algo].append({
                    "seed": seed,
                    "gen": h["gen"],
                    "fitness_J": h["fitness_J"],
                })

            metrics = best.metrics
            row = {
                "seed": seed,
                "algorithm": algo,
                "num_tasks": target_tasks,
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
            save_partial_results(all_rows, all_history, target_tasks, generations, results_dir, history_dir)
            print(json.dumps(row, indent=2, ensure_ascii=False))

    raw_csv = results_dir / f"llm_nsga_comparison_multi_seed_raw_{target_tasks}.csv"
    summary_csv = results_dir / f"llm_nsga_comparison_multi_seed_summary_{target_tasks}.csv"
    df = pd.DataFrame(all_rows)
    if not df.empty:
        print("\nSummary saved to: %s" % summary_csv)
        print("Total rows: %d | Seeds: %d" % (len(df), df["seed"].nunique()))
        print("\n=== Runtime / LLM budget comparison ===")
        print(df.groupby("algorithm")[["runtime", "llm_calls", "llm_total_tokens"]].mean().round(3))


if __name__ == "__main__":
    main()
