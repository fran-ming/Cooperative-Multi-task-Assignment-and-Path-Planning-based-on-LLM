"""
prompts.py -- semantic-enriched version

Design change vs the original version:
  1. The chromosome is never shown to the LLM as a bare list [v1,...,vN] alone.
     Every gene is inlined into a task-centric row that also carries the
     task's constraints AND the decoder's evaluation of that gene, so the
     LLM never has to do position-counting / cross-table joins to figure out
     what a gene means or whether it is doing well.
  2. Feasibility is computed by the program (mirroring the mask M_ik used in
     the EAAI/HGAP pipeline) and handed to the LLM as an explicit candidate
     set per task, instead of making the LLM re-derive feasibility from raw
     numbers.
  3. Constraint violations are aggregated across the population into a
     "tension summary" so the LLM can prioritize which relations to fix,
     instead of reading N separate per-individual feedback blocks.
  4. Crossover/mutation no longer ask the LLM to reproduce the whole
     chromosome. They ask for a small list of edits {task_id, new_ot,
     reason}. The program applies edits deterministically onto a copy of
     the parent chromosome (apply_edits). This removes the "LLM mis-copies
     an unrelated gene while rewriting the array" failure mode entirely,
     and makes "task order / gene count / OT set unchanged" a property the
     program guarantees rather than a rule the LLM must remember to obey.
"""

import json
import re


# ---------------------------------------------------------------------------
# 1. Core system prompts
# ---------------------------------------------------------------------------

PROBLEM_DEFINITION = """You are an evolutionary optimization agent for cooperative object-target scheduling in closed-track automated driving testing.

The task sequence is FIXED and MUST NOT be changed.

Chromosome representation:
chi = [v1, v2, ..., vN]
where vi denotes the OT assigned to task Ti.

You will never be shown this raw array by itself. Instead, every task will
be shown together with its own constraints, its currently assigned OT, and
(when available) the decoder's evaluation of that assignment, plus the set
of OTs that are actually feasible for that task. Read assignments from
these per-task rows, not by counting array positions.

Your responsibility is to improve OT assignment patterns.

The four optimization objectives are:
1. Jd: minimize the number of failed task units.
2. Jm: minimize global makespan.
3. Jb: minimize OT workload imbalance.
4. Jt: minimize total travel distance.

The schedule is subject to VUT-triggered execution time-window constraints,
cooperative synchronization constraints, OT availability and sequential
execution constraints, and road topology and road-resource constraints.
These constraints are what determine the FEASIBLE CANDIDATES list given to
you for each task -- a candidate OT that is not in that list is guaranteed
infeasible for that task and must never be proposed.

Objective values and physical feasibility are ultimately confirmed by an
external deterministic decoder and evaluator; the feasible-candidate lists
you receive are a fast pre-filter, not a substitute for that evaluation.

You MUST NOT:
- change task order;
- create or remove tasks;
- create new OTs;
- change task attributes;
- change objective definitions;
- propose an OT for a task that is not in that task's feasible-candidate list;
- directly claim a candidate is feasible without decoder evaluation.

You operate only by proposing EDITS (task_id -> new OT), never by rewriting
the whole chromosome. Unedited genes are left untouched by the program."""


SELECTION_SYSTEM = """You are performing parent selection for an LLM-guided multi-objective evolutionary algorithm.

The task order is fixed. Select individuals that provide a good combination of:
1. Pareto quality;
2. objective performance;
3. population diversity;
4. different OT assignment patterns, especially different resolutions of the
   constraint tensions listed in the CONSTRAINT TENSION SUMMARY;
5. successful execution under testing constraints.

Do not select all individuals from one similar assignment pattern. Individuals
with lower Pareto rank are generally preferred. For individuals on the same
Pareto front, crowding distance should be considered to preserve diversity,
and prefer individuals that resolve DIFFERENT tension hotspots so the next
generation's crossover has genuinely complementary material to combine.

Do not recompute objective values. Return only the selected individual IDs."""


CROSSOVER_SYSTEM = """You are performing crossover for cooperative OT scheduling.

The task sequence is fixed: T1 -> T2 -> ... -> TN.

You will be shown two parents ONLY at the rows where their assignments
differ (rows where both parents agree are omitted -- there is nothing to
recombine there). For each differing task, you will see both parents'
choice of OT, each task's constraints, each candidate's feasibility, and
(when available) how each parent's choice performed under the decoder.

Produce an offspring by selecting, for each differing task, whichever
parent's choice looks better given the task's constraints and decoder
feedback -- or propose a third feasible candidate if neither parent's
choice is good. Consider synchronization groups as a UNIT: if a task
belongs to a sync group, prefer keeping that group's OT choices consistent
with each other rather than picking parents independently per task inside
the same group.

Return ONLY a list of edits relative to Parent A. Do not include tasks
where you keep Parent A's assignment. Each edit must use an OT id that is
in the target task's feasible-candidate list.

Do not calculate objective values yourself. The external decoder will
evaluate the offspring. Return only JSON."""


MUTATION_SYSTEM = """You are performing mutation for cooperative OT scheduling.

The task order is fixed. You will be shown the current individual as a
task-centric view: each task's constraints, its currently assigned OT, the
decoder's evaluation of that assignment (arrival time, success/fail,
lateness), and the feasible-candidate list for that task. You will also be
shown a CONSTRAINT TENSION SUMMARY describing which relations (sync groups,
OT resource conflicts, workload imbalance) are currently under the most
stress across the population.

Propose a small number of targeted edits that address specific, named
tensions -- e.g. "T5 is late by 2.3 under its sync group's tolerance;
reassigning it to OT7 (feasible, currently under-loaded) should reduce
that error." Prefer edits that fix an identified problem over edits with
no stated reason. Each edit's new OT must be in that task's
feasible-candidate list.

Do not change task order. Do not create new tasks. Do not create new OTs.
Do not change task attributes. Return only JSON."""


# ---------------------------------------------------------------------------
# 2. Semantic view builders
#    These are the core of the "semantic understanding" fix: every gene is
#    shown together with its meaning, its evaluation, and its feasible
#    alternatives -- never as a bare number in a bare array.
# ---------------------------------------------------------------------------

def build_ot_information(scenario):
    lines = ["OT INFORMATION", "", "OT ID | Initial Position | Initial Available Time"]
    for ot in scenario.ots:
        lines.append("OT%d | %s | %.1f" % (ot.ot_id, ot.initial_position, ot.initial_available_time))
    return "\n".join(lines)


def compute_feasible_candidates(task, scenario, max_candidates=6):
    """Program-side feasibility pre-filter, mirroring the mask M_ik used in
    the EAAI/HGAP decoder. This is intentionally a cheap, conservative
    filter (time-window reachability only) -- it must never be more
    permissive than the real decoder, only a subset of what the decoder
    would accept. If the EA engine already computes a feasibility mask
    (e.g. individual.decode_result.feasibility_mask), prefer wiring that
    through instead of recomputing it here; this function is a fallback.

    Returns a list of (ot_id, est_arrival) tuples, sorted by estimated
    arrival, truncated to max_candidates so the prompt stays short. Truncation
    is safe as long as the true best candidates are unlikely to be filtered
    out silently -- log/monitor this in the calling code if candidates are
    ever dropped that later matter.
    """
    candidates = []
    for ot in scenario.ots:
        est_arrival = scenario.estimate_arrival(ot, task) if hasattr(scenario, "estimate_arrival") else None
        if est_arrival is None:
            # No cheap estimator available: fall back to "all OTs are
            # candidates" so the LLM at least has a valid list to choose
            # from; feasibility will still be enforced by the decoder later.
            candidates.append((ot.ot_id, None))
            continue
        if task.tw[0] <= est_arrival <= task.tw[1]:
            candidates.append((ot.ot_id, est_arrival))
    candidates.sort(key=lambda x: (x[1] is None, x[1]))
    return candidates[:max_candidates]


def _format_candidates(candidates):
    parts = []
    for ot_id, est_arrival in candidates:
        if est_arrival is None:
            parts.append("OT%d" % ot_id)
        else:
            parts.append("OT%d(~%.1f)" % (ot_id, est_arrival))
    return ", ".join(parts) if parts else "NONE"


def build_assignment_view(scenario, chromosome, decode_result=None, feasibility_fn=None):
    """Task-centric semantic view: one row per task, carrying constraints +
    current assignment + decoder evaluation + feasible candidates, all
    inline. This replaces showing the raw chromosome array and the task
    table as two separate structures the LLM must join by position.
    """
    feasibility_fn = feasibility_fn or compute_feasible_candidates
    records_by_task = {}
    if decode_result is not None:
        records_by_task = {r.task_id: r for r in decode_result.records}

    lines = [
        "CURRENT ASSIGNMENT (one row per task, all fields describe the SAME task)",
        "",
        "Task | Window | Dur | Sync | Assigned OT | Arrival | Status | Lateness | Feasible candidates",
    ]
    for task, ot_id in zip(scenario.tasks, chromosome):
        rec = records_by_task.get(task.task_id)
        arrival = "%.2f" % rec.arrival if rec else "-"
        status = ("OK" if rec.success else "FAIL") if rec else "-"
        lateness = "%.2f" % rec.lateness if rec else "-"
        candidates = feasibility_fn(task, scenario)
        lines.append(
            "%s | [%.1f,%.1f] | %.1f | %s | OT%d | %s | %s | %s | %s" % (
                task.task_id, task.tw[0], task.tw[1], task.duration,
                task.sync_group or "-", ot_id, arrival, status, lateness,
                _format_candidates(candidates),
            )
        )
    return "\n".join(lines)


def build_diff_view(scenario, chromosome_a, chromosome_b, decode_result_a=None, decode_result_b=None,
                     feasibility_fn=None):
    """For crossover: show ONLY the tasks where two parents disagree, each
    with its own evaluation and feasible candidates. Rows where both
    parents agree carry no recombination signal and only dilute the
    context, so they are omitted entirely.
    """
    feasibility_fn = feasibility_fn or compute_feasible_candidates
    records_a = {r.task_id: r for r in decode_result_a.records} if decode_result_a else {}
    records_b = {r.task_id: r for r in decode_result_b.records} if decode_result_b else {}

    lines = [
        "PARENT DIFFERENCES (tasks omitted here are identical in both parents)",
        "",
        "Task | Window | Sync | A:OT(status,lateness) | B:OT(status,lateness) | Feasible candidates",
    ]
    n_diff = 0
    for task, va, vb in zip(scenario.tasks, chromosome_a, chromosome_b):
        if va == vb:
            continue
        n_diff += 1
        ra, rb = records_a.get(task.task_id), records_b.get(task.task_id)
        a_desc = "OT%d(%s,%.2f)" % (va, "OK" if ra and ra.success else "FAIL", ra.lateness) if ra else "OT%d(-,-)" % va
        b_desc = "OT%d(%s,%.2f)" % (vb, "OK" if rb and rb.success else "FAIL", rb.lateness) if rb else "OT%d(-,-)" % vb
        candidates = feasibility_fn(task, scenario)
        lines.append("%s | [%.1f,%.1f] | %s | A:%s | B:%s | %s" % (
            task.task_id, task.tw[0], task.tw[1], task.sync_group or "-",
            a_desc, b_desc, _format_candidates(candidates)))
    if n_diff == 0:
        lines.append("(parents are identical -- no differing tasks to recombine)")
    return "\n".join(lines)


def build_constraint_tension_summary(population, top_k=6):
    """Population-level aggregation of where constraints are currently under
    the most stress. This tells the LLM WHERE to focus edits before it even
    looks at a specific individual, instead of re-deriving hotspots from
    scratch every single call.
    """
    sync_violations = {}   # group -> [count, sum_error]
    resource_conflicts = {}  # (ot_i, ot_j) -> count
    fail_counts = {}       # task_id -> count

    n = 0
    for ind in population:
        if not ind.decode_result:
            continue
        n += 1
        for item in (ind.decode_result.sync_feedback or []):
            if item.get("status") != "ok":
                agg = sync_violations.setdefault(item["group"], [0, 0.0])
                agg[0] += 1
                agg[1] += float(item.get("error", 0.0))
        for rec in ind.decode_result.records:
            if not rec.success:
                fail_counts[rec.task_id] = fail_counts.get(rec.task_id, 0) + 1

    lines = ["CONSTRAINT TENSION SUMMARY (across %d evaluated individuals)" % n, ""]

    if sync_violations:
        lines.append("Sync groups under stress (violated_in / avg_error):")
        ranked = sorted(sync_violations.items(), key=lambda kv: -kv[1][0])[:top_k]
        for group, (count, err_sum) in ranked:
            lines.append("  %s: %d/%d individuals, avg error %.2f" % (group, count, n, err_sum / count))
    else:
        lines.append("Sync groups under stress: none observed")

    if fail_counts:
        lines.append("Tasks failing most often:")
        ranked = sorted(fail_counts.items(), key=lambda kv: -kv[1])[:top_k]
        for task_id, count in ranked:
            lines.append("  %s: failed in %d/%d individuals" % (task_id, count, n))
    else:
        lines.append("Tasks failing most often: none observed")

    return "\n".join(lines)


def _num(value):
    if value is None:
        return "-"
    return "%.3f" % float(value)


def _format_objectives(individual):
    obj = individual.objectives or [None, None, None, None]
    return "[Jd=%s, Jm=%s, Jb=%s, Jt=%s]" % tuple(_num(x) for x in obj)


def _format_chromosome(chromosome):
    return json.dumps([int(v) for v in chromosome])


# ---------------------------------------------------------------------------
# 3. Population presentation for selection
# ---------------------------------------------------------------------------

def build_candidate_table(population, limit=30):
    """Sorted by Pareto rank then crowding distance (least-to-most fitness
    ordering), so improvement direction is visible from the ordering itself
    instead of being buried in an arbitrary population order."""
    ranked = sorted(
        population,
        key=lambda ind: (ind.rank if ind.rank is not None else 1e9, -(ind.crowding_distance or 0.0)),
    )[:limit]

    lines = ["CANDIDATE INDIVIDUALS (sorted best-rank-first)", ""]
    for i, ind in enumerate(ranked, 1):
        lines.append("INDIVIDUAL %03d" % i)
        lines.append("Chromosome: " + _format_chromosome(ind.chromosome))
        lines.append("Objectives: " + _format_objectives(ind))
        lines.append("Pareto Rank: %s | Crowding Distance: %.4f" % (
            ind.rank if ind.rank is not None else "-", ind.crowding_distance or 0.0))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. User prompt builders
# ---------------------------------------------------------------------------

def selection_user(population, num_parents, include_task_info=False, **kwargs):
    """Backward-compatible parent-selection prompt builder.

    The newer prompt layout no longer requires extra task-info flags, but older
    call sites still pass ``include_task_info``. Accepting it here keeps the
    prompt API stable while allowing the caller to remain on the previous
    interface.
    """
    _ = include_task_info  # Accepted for compatibility; prompt content is already task-centric.
    return "SELECT PARENTS\n\nNumber of parents required: %d\n\n%s\n\n%s\n\nReturn exactly:\n{\n  \"selected_ids\": [\"001\", \"007\", ...]\n}" % (
        num_parents,
        build_constraint_tension_summary(population),
        build_candidate_table(population, limit=min(len(population), 50)),
    )


def crossover_user(parent_a, parent_b, scenario, include_task_info=False, **kwargs):
    diff_view = build_diff_view(
        scenario, parent_a.chromosome, parent_b.chromosome,
        decode_result_a=parent_a.decode_result, decode_result_b=parent_b.decode_result,
    )
    return (
        "CROSSOVER\n\n"
        "Parent A objectives: %s\n"
        "Parent B objectives: %s\n\n"
        "%s\n\n"
        "Return edits relative to Parent A. Only list tasks you change:\n"
        "{\n"
        "  \"operation\": \"crossover\",\n"
        "  \"base\": \"A\",\n"
        "  \"edits\": [\n"
        "      {\"task_id\": \"T5\", \"new_ot\": 7, \"reason\": \"...\"},\n"
        "      ...\n"
        "  ]\n"
        "}"
    ) % (_format_objectives(parent_a), _format_objectives(parent_b), diff_view)


def mutation_user(individual, scenario, include_task_info=False, **kwargs):
    _ = include_task_info
    assignment_view = build_assignment_view(scenario, individual.chromosome, individual.decode_result)
    return (
        "MUTATION\n\n"
        "Current individual objectives: %s\n\n"
        "%s\n\n"
        "Propose targeted edits. Return:\n"
        "{\n"
        "  \"operation\": \"mutation\",\n"
        "  \"edits\": [\n"
        "      {\"task_id\": \"T5\", \"new_ot\": 7, \"reason\": \"...\"},\n"
        "      ...\n"
        "  ]\n"
        "}"
    ) % (_format_objectives(individual), assignment_view)


def crossover_batch_user(parent_pairs, scenario, include_task_info=False, **kwargs):
    _ = include_task_info
    lines = ["CROSSOVER BATCH", "", "Generate %d offspring, one edit-list per pair." % len(parent_pairs), ""]
    for i, (pa, pb) in enumerate(parent_pairs, 1):
        lines.append("Pair %d (A objectives %s, B objectives %s):" % (i, _format_objectives(pa), _format_objectives(pb)))
        lines.append(build_diff_view(
            scenario, pa.chromosome, pb.chromosome,
            decode_result_a=pa.decode_result, decode_result_b=pb.decode_result,
        ))
        lines.append("")
    lines.append(
        'Return:\n{\n  "offspring": [\n'
        '      {"base": "A", "edits": [{"task_id": "T5", "new_ot": 7, "reason": "..."}]},\n'
        '      ...\n  ]\n}'
    )
    return "\n".join(lines)


def mutation_batch_user(individuals, scenario, include_task_info=False, **kwargs):
    _ = include_task_info
    lines = ["MUTATION BATCH", "", "Generate targeted edits for %d individuals." % len(individuals), ""]
    for i, ind in enumerate(individuals, 1):
        lines.append("Individual %03d (objectives %s):" % (i, _format_objectives(ind)))
        lines.append(build_assignment_view(scenario, ind.chromosome, ind.decode_result))
        lines.append("")
    lines.append(
        'Return:\n{\n  "mutations": [\n'
        '      {"individual_id": "001", "edits": [{"task_id": "T5", "new_ot": 7, "reason": "..."}]},\n'
        '      ...\n  ]\n}'
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Parsing + deterministic application of edits
#    The LLM never writes into the chromosome directly. It proposes edits;
#    the program validates and applies them onto a COPY of the parent.
# ---------------------------------------------------------------------------

def parse_json_response(text):
    """Parse model output, tolerating markdown code fences and prose around JSON."""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.S | re.I)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            return None
    return None


def extract_edits(payload):
    """Pull out the {task_id, new_ot} edit list from a crossover/mutation response."""
    if not isinstance(payload, dict):
        return None
    edits = payload.get("edits")
    if not isinstance(edits, list):
        return None
    cleaned = []
    for e in edits:
        if isinstance(e, dict) and "task_id" in e and "new_ot" in e:
            cleaned.append({"task_id": e["task_id"], "new_ot": e["new_ot"], "reason": e.get("reason", "")})
    return cleaned


def extract_selected_ids(payload):
    if not isinstance(payload, dict):
        return None
    value = payload.get("selected_ids")
    if not isinstance(value, list):
        return None
    return value


def extract_batch_offspring(payload):
    if not isinstance(payload, dict):
        return None
    value = payload.get("offspring")
    return value if isinstance(value, list) else None


def extract_batch_mutations(payload):
    if not isinstance(payload, dict):
        return None
    value = payload.get("mutations")
    return value if isinstance(value, list) else None


def apply_edits(base_chromosome, edits, task_id_to_index, valid_ot_ids):
    """Deterministically apply a validated edit list onto a COPY of the base
    chromosome. This is what actually "executes" an LLM instruction like
    "cross T5 and T6" -- the LLM only ever names task_id -> new_ot pairs;
    this function is the single place where the chromosome array is
    actually written.

    Returns (child_chromosome, applied, rejected). Rejected edits (unknown
    task_id or infeasible/unknown OT) are reported back so the caller can
    log them or feed them back to the LLM as correction feedback on retry,
    instead of silently dropping or silently corrupting the chromosome.
    """
    child = list(base_chromosome)
    applied, rejected = [], []
    for e in edits or []:
        idx = task_id_to_index.get(e.get("task_id"))
        try:
            new_ot = int(e.get("new_ot"))
        except (TypeError, ValueError):
            new_ot = None
        if idx is None or new_ot is None or new_ot not in valid_ot_ids:
            rejected.append(e)
            continue
        child[idx] = new_ot
        applied.append(e)
    return child, applied, rejected