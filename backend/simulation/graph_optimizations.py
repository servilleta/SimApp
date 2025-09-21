"""Graph optimization utilities for Power Monte Carlo Engine (Phase-2)

Includes:
1. Constant propagation / pruning of formula dictionary.
2. Strongly-connected component (SCC) collapsing.
3. Optional graph partitioning via METIS / nxmetis.
4. Persistence of dependency graph to Apache Arrow IPC file for zero-copy loading.

NOTE: These utilities are self-contained and rely on `networkx` and `pyarrow`. If
`nxmetis` (wrapper for METIS) is not installed the partition step gracefully
falls back to a single partition while logging a warning.
"""

from __future__ import annotations

import logging
import re
import os
from typing import Dict, Tuple, Set, List

import networkx as nx

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pyarrow is required for graph persistence in Phase-2 optimisations."
    ) from e

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. CONSTANT PROPAGATION / FORMULA PRUNING
# ---------------------------------------------------------------------------

_CONST_REGEX = re.compile(r"^=?(\s*[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?)\s*$")


def _is_constant_formula(formula: str) -> bool:
    """Return True if the formula string is a pure numeric literal.

    Accepts forms like `=3.14`, `  -42`, `2.0e-3`, etc.
    """
    if not isinstance(formula, str):
        return False
    return bool(_CONST_REGEX.match(formula.strip()))


def prune_constant_formulas(
    formula_dict: Dict[str, Dict[str, str]]
) -> Tuple[Dict[str, Dict[str, str]], Set[Tuple[str, str]]]:
    """Remove formulas that are numeric constants.

    Returns a new dict (deep-copied) and a set of (sheet, cell) removed constants.
    """
    pruned: Dict[str, Dict[str, str]] = {}
    removed_consts: Set[Tuple[str, str]] = set()

    for sheet, cells in formula_dict.items():
        for cell, formula in cells.items():
            if _is_constant_formula(formula):
                removed_consts.add((sheet, cell))
                continue
            pruned.setdefault(sheet, {})[cell] = formula

    logger.info(
        "[GRAPH_OPT] Pruned %d constant formulas (numeric literals)",
        len(removed_consts),
    )
    return pruned, removed_consts


# ---------------------------------------------------------------------------
# 2. SCC COLLAPSING
# ---------------------------------------------------------------------------

def build_dependency_graph(
    formula_dict: Dict[str, Dict[str, str]]
) -> nx.DiGraph:
    """Very lightweight dependency graph builder using regex cell extraction."""
from simulation.formula_utils import CELL_REFERENCE_REGEX  # delayed import to avoid cycle

    g = nx.DiGraph()

    for sheet_name, cells in formula_dict.items():
        for cell_coord, formula in cells.items():
            node = (sheet_name, cell_coord.upper())
            g.add_node(node)

            # quick parse of dependencies
            deps = CELL_REFERENCE_REGEX.findall(formula)
            for raw_sheet, dep_cell in deps:
                dep_sheet = (
                    raw_sheet[1:-1]  # strip quotes
                    if raw_sheet and raw_sheet.startswith("'") else raw_sheet
                ) or sheet_name
                dep_node = (dep_sheet, dep_cell.upper())
                g.add_edge(dep_node, node)  # edge from dependency -> current

    logger.info("[GRAPH_OPT] Built dependency graph with %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())
    return g


def collapse_sccs(g: nx.DiGraph) -> Tuple[nx.DiGraph, Dict[Tuple[str, str], Tuple[str, str]]]:
    """Collapse strongly-connected components into single representative nodes.

    Returns the collapsed graph and a mapping from original node -> representative.
    """
    sccs = list(nx.strongly_connected_components(g))
    mapping: Dict[Tuple[str, str], Tuple[str, str]] = {}
    collapsed = nx.DiGraph()

    for comp in sccs:
        rep = next(iter(comp))  # deterministic representative
        collapsed.add_node(rep)
        for node in comp:
            mapping[node] = rep

    for u, v in g.edges():
        collapsed.add_edge(mapping[u], mapping[v])

    logger.info(
        "[GRAPH_OPT] Collapsed %d SCCs → %d nodes",
        len(sccs),
        collapsed.number_of_nodes(),
    )
    return collapsed, mapping


# ---------------------------------------------------------------------------
# 3. GRAPH PARTITIONING (METIS)
# ---------------------------------------------------------------------------

try:
    import nxmetis  # type: ignore
except ImportError:  # pragma: no cover
    nxmetis = None  # runtime fallback


def partition_graph(
    g: nx.Graph, num_parts: int = 4
) -> List[Set[Tuple[str, str]]]:
    """Partition graph into *num_parts* parts using METIS if available, else fallback.
    """
    if nxmetis is None:
        logger.warning("[GRAPH_OPT] nxmetis not available → skipping partitioning")
        return [set(g.nodes())]

    edgecuts, parts = nxmetis.partition(g, num_parts)
    logger.info("[GRAPH_OPT] METIS partition complete: edgecuts=%d parts=%d", edgecuts, num_parts)
    return [set(p) for p in parts]


# ---------------------------------------------------------------------------
# 4. ARROW PERSISTENCE
# ---------------------------------------------------------------------------

def persist_graph_arrow(g: nx.DiGraph, file_path: str) -> None:
    """Persist directed graph edges to Apache Arrow IPC file."""
    u_nodes = []
    v_nodes = []
    for u, v in g.edges():
        u_nodes.append(f"{u[0]}!{u[1]}")
        v_nodes.append(f"{v[0]}!{v[1]}")

    table = pa.table({"u": u_nodes, "v": v_nodes})
    with pa_ipc.new_file(file_path, table.schema) as sink:
        sink.write(table)
    logger.info("[GRAPH_OPT] Dependency graph persisted to %s (%d edges)", file_path, g.number_of_edges())


# ---------------------------------------------------------------------------
# 5. TOP-LEVEL OPTIMISATION PIPELINE
# ---------------------------------------------------------------------------

def optimise_formulas(
    formula_dict: Dict[str, Dict[str, str]],
    arrow_out_dir: str | None = None,
    partitions: int | None = None,
) -> Tuple[Dict[str, Dict[str, str]], Dict]:
    """Run Phase-2 optimisations and return an updated formula_dict.

    Returns (new_formula_dict, stats_dict).
    """
    stats: Dict = {}

    # 1. Constant pruning
    formula_dict, removed_consts = prune_constant_formulas(formula_dict)
    stats["constants_pruned"] = len(removed_consts)

    # 2. Build graph and collapse SCCs
    g = build_dependency_graph(formula_dict)
    collapsed, mapping = collapse_sccs(g)
    stats["nodes_after_collapse"] = collapsed.number_of_nodes()

    # 3. Partition (optional)
    parts_out: List[Set[Tuple[str, str]]]
    if partitions and collapsed.number_of_nodes() > 10_000:
        parts_out = partition_graph(collapsed.to_undirected(), num_parts=partitions)
        stats["partitions"] = len(parts_out)
    else:
        parts_out = [set(collapsed.nodes())]
        stats["partitions"] = 1

    # 4. Persist to Arrow (optional)
    if arrow_out_dir:
        os.makedirs(arrow_out_dir, exist_ok=True)
        arrow_path = os.path.join(arrow_out_dir, "power_dag.arrow")
        persist_graph_arrow(collapsed, arrow_path)
        stats["arrow_path"] = arrow_path

    return formula_dict, stats 