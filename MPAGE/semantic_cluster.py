# -*- coding: utf-8 -*-
"""Semantic clustering of heuristics (guide sections 15-16).

The first version clusters by the structured strategy signature (a heuristic's
primary priority cue).  No separate embedding model is used; optionally an LLM
may assign themes, but a deterministic fallback always exists.
"""


def _signature(h):
    priority = tuple(h.strategy.get("priority") or [])
    return (priority[0] if priority else "hybrid",
            h.strategy.get("construction"),
            h.strategy.get("repair"))


def cluster_deterministic(heuristics, n_clusters=3):
    """Cluster by the primary priority cue, capping the cluster count."""
    groups = {}
    for h in heuristics:
        key = _signature(h)[0]
        groups.setdefault(key, []).append(h)

    # Map the most frequent signatures to cluster ids 0..n_clusters-1.
    ranked_keys = sorted(groups, key=lambda k: -len(groups[k]))
    id_for_key = {}
    for key in ranked_keys:
        cluster_id = len(id_for_key) % max(1, n_clusters)
        id_for_key[key] = cluster_id

    clusters = {}
    for h in heuristics:
        cid = id_for_key[_signature(h)[0]]
        h.cluster_id = cid
        clusters.setdefault(cid, []).append(h.heuristic_id)
    return clusters


def cluster(heuristics, scenario=None, client=None, n_clusters=3):
    """Cluster with an optional LLM pass, falling back to deterministic logic."""
    if client is not None:
        try:
            from MPAGE.prompts import clustering_user, parse_heuristic_json
            response = client.generate("", clustering_user(heuristics, n_clusters))
            payload = parse_heuristic_json(response)
            clusters = payload.get("clusters") if isinstance(payload, dict) else None
            if isinstance(clusters, list) and clusters:
                applied = False
                for entry in clusters:
                    if not isinstance(entry, dict):
                        continue
                    cid = entry.get("cluster_id")
                    ids = entry.get("heuristic_ids", [])
                    if isinstance(cid, (int, float)) and isinstance(ids, list):
                        for h in heuristics:
                            if h.heuristic_id in ids:
                                h.cluster_id = int(cid)
                        applied = True
                if applied:
                    return clusters
        except Exception:
            pass
    return cluster_deterministic(heuristics, n_clusters=n_clusters)