"""
Pre-execution validation for merged NetworkX plan graphs.
Logs warnings for recoverable issues; critical problems return errors (e.g. cycles).
"""

from __future__ import annotations

import networkx as nx

from memory.context import sanitize_io_keys_list

# Keys commonly satisfied without a prior writer in the DAG (session/bootstrap).
BUILTIN_READ_KEYS = frozenset(
    {
        "original_query",
        "file_manifest",
        "file_profiles",
        "plan_graph",
    }
)


def validate_merged_plan_graph(G: nx.DiGraph) -> tuple[list[str], list[str]]:
    """
    Validate the merged execution graph.

    Returns:
        (errors, warnings) — errors should block or require repair; warnings are informational.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if G.number_of_nodes() == 0:
        errors.append("empty_graph")
        return errors, warnings

    if not nx.is_directed_acyclic_graph(G):
        try:
            cycles = list(nx.simple_cycles(G))[:3]
            errors.append(f"graph_has_cycles: {cycles}")
        except Exception:
            errors.append("graph_has_cycles")

    # Reachability from Query (bootstrap planner node)
    if "Query" in G:
        reachable = set(nx.descendants(G, "Query")) | {"Query"}
        for nid in G.nodes:
            if nid in ("ROOT",):
                continue
            if nid not in reachable:
                warnings.append(f"node_not_reachable_from_Query:{nid}")
    else:
        warnings.append("missing_Query_node")

    # Writes and reads: each step's read keys should be producible by some ancestor or builtin
    for nid in G.nodes:
        if nid in ("ROOT", "Query"):
            continue
        data = G.nodes[nid]
        reads = sanitize_io_keys_list(data.get("reads", []))
        writes = sanitize_io_keys_list(data.get("writes", []))
        ancestors = nx.ancestors(G, nid) if nid in G else set()

        available: set[str] = set(BUILTIN_READ_KEYS)
        for a in ancestors:
            if a in ("ROOT",):
                continue
            aw = sanitize_io_keys_list(G.nodes[a].get("writes", []))
            available.update(aw)

        for rk in reads:
            if rk in available:
                continue
            warnings.append(f"read_may_be_unsatisfied:{nid}:{rk}")

    return errors, warnings
