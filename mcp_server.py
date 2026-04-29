#!/usr/bin/env python3
"""
mcp_server.py — MCP server wrapping know-thyself-search retrieval.

Exposes the typed-graph retrieval surface as MCP tools so any MCP-aware
client (Claude Code, Claude Desktop, Cursor, etc.) can query a personal
knowledge graph natively without dumping the whole YAML into context.

Tools:
  search_graph     — top-k retrieval with optional type filter + provenance rerank
  get_node         — fetch a single node by id (full statement + metadata)
  list_node_stats  — summary of the loaded index (node counts by type, backend, etc.)

The index is loaded lazily on first call from graph-embeddings.json next
to this script. Set KNOW_THYSELF_INDEX to point elsewhere.

Run:
  python mcp_server.py                 # stdio server (the usual MCP transport)
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from mcp.server.fastmcp import FastMCP

# Reuse vectorizers + tier table from search.py (same directory).
sys.path.insert(0, str(Path(__file__).parent))
from search import (  # noqa: E402
    TENTATIVE_PENALTY,
    TYPE_TIER,
    cosine_query,
    local_vectorize_query,
    openai_vectorize_query,
    tfidf_vectorize_query,
)

INDEX_PATH = Path(
    os.environ.get(
        "KNOW_THYSELF_INDEX",
        str(Path(__file__).parent / "graph-embeddings.json"),
    )
)

mcp = FastMCP("know-thyself-search")

_INDEX: Optional[dict] = None
_NODE_BY_ID: Optional[dict] = None
_MATRIX: Optional[np.ndarray] = None


def _ensure_loaded() -> None:
    global _INDEX, _NODE_BY_ID, _MATRIX
    if _INDEX is not None:
        return
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"index not found at {INDEX_PATH}. "
            f"Run: python embed.py <graph.yaml>"
        )
    _INDEX = json.loads(INDEX_PATH.read_text())
    _NODE_BY_ID = {n["id"]: n for n in _INDEX["nodes"]}
    _MATRIX = np.array(
        [n["vector"] for n in _INDEX["nodes"]], dtype=np.float32
    )


def _vectorize(query: str) -> np.ndarray:
    backend = _INDEX["backend"]
    if backend == "tfidf":
        return tfidf_vectorize_query(query, _INDEX["vocab"])
    if backend == "openai":
        return openai_vectorize_query(query, _INDEX["model"])
    if backend == "local":
        return local_vectorize_query(query, _INDEX["model"])
    raise ValueError(f"unknown backend: {backend}")


@mcp.tool()
def search_graph(
    query: str,
    top_k: int = 5,
    type_filter: Optional[str] = None,
    provenance: bool = False,
) -> list[dict]:
    """Search the typed knowledge graph by semantic similarity.

    Returns top_k node hits ranked by cosine similarity over node
    statements, optionally boosted by Pat-tier provenance reranking
    (References > Overlaps > Observations > Practices ≈ Emergents >
    Novels > Opens, with a tentative-flag penalty).

    Args:
      query: Natural-language query.
      top_k: Max number of results. Default 5.
      type_filter: If set, restrict to nodes of this type
        (observation | reference | overlap | novel | practice |
        emergent | equivalency | open | now). Otherwise no filter.
      provenance: If true, multiply scores by tier(type) and apply
        the tentative penalty. False = pure cosine.

    Returns:
      List of {id, type, name, score, tentative, statement} dicts,
      sorted by score descending. Empty list if the query has no
      in-vocabulary terms (TF-IDF backend) or the type filter is empty.
    """
    _ensure_loaded()
    nodes = _INDEX["nodes"]
    matrix = _MATRIX
    if type_filter:
        keep = [
            (i, n) for i, n in enumerate(nodes)
            if n.get("type") == type_filter
        ]
        if not keep:
            return []
        idxs, sub_nodes = zip(*keep)
        matrix = matrix[list(idxs)]
        nodes = list(sub_nodes)

    qvec = _vectorize(query)
    if np.allclose(qvec, 0):
        return []
    scores = cosine_query(qvec, matrix)

    if provenance:
        adj = scores.copy()
        for i, n in enumerate(nodes):
            tier = TYPE_TIER.get(n.get("type"), 1.0)
            if n.get("tentative"):
                tier *= TENTATIVE_PENALTY
            adj[i] = scores[i] * tier
        scores = adj

    top = np.argsort(-scores)[:top_k]
    return [
        {
            "id": nodes[i]["id"],
            "type": nodes[i].get("type", "?"),
            "name": nodes[i].get("name", ""),
            "score": round(float(scores[i]), 4),
            "tentative": bool(nodes[i].get("tentative")),
            "statement": nodes[i].get("statement", ""),
        }
        for i in top
    ]


@mcp.tool()
def get_node(node_id: str) -> dict:
    """Fetch a single node by id with full statement and metadata.

    Supports short-id lookup: passing "O04" matches the unique node
    whose id starts with "O04-" (e.g. "O04-daughter-grades-recovered").
    If the prefix matches multiple nodes, returns the candidate list.

    Args:
      node_id: Full id ("O04-daughter-grades-recovered", "NOW") or
        unambiguous short id ("O04", "P01").

    Returns:
      {id, type, name, tentative, statement} on success, or
      {error: <message>, matches?: [...]} on failure.
    """
    _ensure_loaded()
    if node_id in _NODE_BY_ID:
        n = _NODE_BY_ID[node_id]
    else:
        prefix = node_id.rstrip("-") + "-"
        cands = [nid for nid in _NODE_BY_ID if nid.startswith(prefix)]
        if len(cands) == 1:
            n = _NODE_BY_ID[cands[0]]
        elif len(cands) > 1:
            return {"error": "ambiguous id", "matches": cands}
        else:
            return {"error": f"no node with id {node_id!r}"}
    return {
        "id": n["id"],
        "type": n.get("type", "?"),
        "name": n.get("name", ""),
        "tentative": bool(n.get("tentative")),
        "statement": n.get("statement", ""),
    }


@mcp.tool()
def list_node_stats() -> dict:
    """Return summary stats about the loaded graph index.

    A session-opener: gives the shape of the graph (counts by type,
    backend, dimensions) without dumping any node contents.

    Returns:
      {backend, model, dim, total_nodes, by_type, index_path}.
    """
    _ensure_loaded()
    types = Counter(n.get("type", "?") for n in _INDEX["nodes"])
    return {
        "backend": _INDEX["backend"],
        "model": _INDEX.get("model"),
        "dim": _INDEX.get("dim"),
        "total_nodes": len(_INDEX["nodes"]),
        "by_type": dict(types.most_common()),
        "index_path": str(INDEX_PATH),
    }


if __name__ == "__main__":
    mcp.run()
