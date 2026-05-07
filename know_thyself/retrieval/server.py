#!/usr/bin/env python3
"""
mcp_server.py — MCP server wrapping know-thyself retrieval.

Exposes the typed-graph retrieval surface as MCP tools so any MCP-aware
client (Claude Code, Claude Desktop, Cursor, etc.) can query a typed
knowledge graph natively without dumping the whole YAML into context.

Design — IDs-only egress by default
-----------------------------------
`search_graph` and `walk_provenance` return id / type / name / score /
tentative — never statement text. Statement text is what your graph
*means*; if the graph is personal, that's the part you want to keep
local. The agent gets relevance signal and can ask follow-up questions
or pull full text out-of-band.

`get_node` returns the full statement and is the one tool that crosses
the egress boundary. It is HIDDEN unless `KNOW_THYSELF_ALLOW_FULL_TEXT=1`
is set in the server's environment. Flip it on only for graphs you've
authorized full-text egress for (the public Alex example graph, a
local-only consumer, etc.).

Tools
-----
  list_node_stats   — counts by type, total, edge counts, backend.
                      Leak-safe; no node ids, no text.
  search_graph      — top-k semantic match. Returns id/type/name/score/
                      tentative + grounded_by_ids + related_to_ids per
                      hit. NO statement text.
  walk_provenance   — walk grounded_by / related_to neighborhood from a
                      node. Returns lightweight neighbor records (no
                      statement). Includes inbound edges by default.
  get_node          — full node including statement. HIDDEN unless
                      KNOW_THYSELF_ALLOW_FULL_TEXT=1.

Configuration via env vars
--------------------------
  KNOW_THYSELF_INDEX            path to the embeddings JSON (default:
                                graph-embeddings.json next to this file)
  KNOW_THYSELF_GRAPH            optional path to the source YAML; when
                                set, the server compares its mtime to
                                the index's and rebuilds in-process if
                                the graph is newer
  KNOW_THYSELF_ALLOW_FULL_TEXT  '1' to expose get_node (default: off)

Run
---
  python mcp_server.py                 # stdio transport (the usual MCP wire)
"""
import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
from mcp.server.fastmcp import FastMCP

from know_thyself.retrieval.search import (
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
GRAPH_PATH = os.environ.get("KNOW_THYSELF_GRAPH")
ALLOW_FULL_TEXT = os.environ.get("KNOW_THYSELF_ALLOW_FULL_TEXT") == "1"

mcp = FastMCP("know-thyself-search")

_INDEX: Optional[dict] = None
_NODE_BY_ID: Optional[dict] = None
_MATRIX: Optional[np.ndarray] = None
_INCOMING: Optional[dict] = None
_INDEX_LOADED_FROM_MTIME: float = 0.0


def _rebuild_index() -> None:
    """Run embed.py logic in-process against KNOW_THYSELF_GRAPH and write
    the result to INDEX_PATH. Only callable when KNOW_THYSELF_GRAPH is set.

    Picks the backend in this order:
      1. KNOW_THYSELF_BACKEND env var, if set
      2. The backend stored in the existing INDEX_PATH (if it exists), so a
         re-embed preserves whatever shape was originally chosen
      3. tfidf (default — no API, no model download)

    Order (2) before (1) would silently override an explicit env var; (1)
    before (2) keeps the env var as the explicit override."""
    if not GRAPH_PATH:
        raise RuntimeError(
            "cannot rebuild index — KNOW_THYSELF_GRAPH is not set"
        )
    from know_thyself.retrieval.embed import build_index

    backend = os.environ.get("KNOW_THYSELF_BACKEND")
    if not backend and INDEX_PATH.exists():
        try:
            existing = json.loads(INDEX_PATH.read_text())
            backend = existing.get("backend")
        except (json.JSONDecodeError, OSError):
            pass
    backend = backend or "tfidf"

    out = build_index(GRAPH_PATH, backend=backend)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(out))


def _ensure_loaded() -> None:
    """Load (or reload) the index. Honors mtime-based auto-rebuild when
    KNOW_THYSELF_GRAPH is set and the graph is newer than the index."""
    global _INDEX, _NODE_BY_ID, _MATRIX, _INCOMING, _INDEX_LOADED_FROM_MTIME

    needs_rebuild = False
    if GRAPH_PATH and Path(GRAPH_PATH).exists():
        graph_mtime = Path(GRAPH_PATH).stat().st_mtime
        if not INDEX_PATH.exists() or INDEX_PATH.stat().st_mtime < graph_mtime:
            needs_rebuild = True

    if needs_rebuild:
        _rebuild_index()
        _INDEX = None  # force reload below

    if _INDEX is not None and INDEX_PATH.stat().st_mtime <= _INDEX_LOADED_FROM_MTIME:
        return

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"index not found at {INDEX_PATH}. "
            f"Run: python embed.py <graph.yaml>  "
            f"(or set KNOW_THYSELF_GRAPH to enable auto-rebuild)"
        )

    _INDEX = json.loads(INDEX_PATH.read_text())
    _NODE_BY_ID = {n["id"]: n for n in _INDEX["nodes"]}
    _MATRIX = np.array(
        [n["vector"] for n in _INDEX["nodes"]], dtype=np.float32
    )
    incoming: dict[str, list[tuple[str, str]]] = {}
    for src in _INDEX["nodes"]:
        for tgt in src.get("grounded_by_ids", []):
            incoming.setdefault(tgt, []).append((src["id"], "grounded_by"))
        for tgt in src.get("related_to_ids", []):
            incoming.setdefault(tgt, []).append((src["id"], "related_to"))
    _INCOMING = incoming
    _INDEX_LOADED_FROM_MTIME = INDEX_PATH.stat().st_mtime


def _light(node: dict) -> dict:
    """Lightweight node view — id, type, name, tentative — no statement."""
    return {
        "id": node["id"],
        "type": node.get("type", "?"),
        "name": node.get("name", ""),
        "tentative": bool(node.get("tentative")),
    }


def _resolve(node_id: str) -> Optional[dict]:
    """Resolve a full or unique-short node id to a node dict, or None."""
    if node_id in _NODE_BY_ID:
        return _NODE_BY_ID[node_id]
    prefix = node_id.rstrip("-") + "-"
    cands = [nid for nid in _NODE_BY_ID if nid.startswith(prefix)]
    if len(cands) == 1:
        return _NODE_BY_ID[cands[0]]
    return None


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
def list_node_stats() -> dict:
    """Return summary stats about the loaded graph index.

    A session-opener: gives the shape of the graph (counts by type,
    backend, dimensions, edge totals) without dumping any node contents.
    Leak-safe — no node ids, no text.

    Returns:
      {backend, model, dim, total_nodes, by_type, edges, index_path}.
    """
    _ensure_loaded()
    types = Counter(n.get("type", "?") for n in _INDEX["nodes"])
    edges_grounded = sum(len(n.get("grounded_by_ids", [])) for n in _INDEX["nodes"])
    edges_related = sum(len(n.get("related_to_ids", [])) for n in _INDEX["nodes"])
    return {
        "backend": _INDEX["backend"],
        "model": _INDEX.get("model"),
        "dim": _INDEX.get("dim"),
        "total_nodes": len(_INDEX["nodes"]),
        "by_type": dict(types.most_common()),
        "edges": {
            "grounded_by": edges_grounded,
            "related_to": edges_related,
            "total": edges_grounded + edges_related,
        },
        "index_path": str(INDEX_PATH),
        "full_text_egress": ALLOW_FULL_TEXT,
    }


@mcp.tool()
def search_graph(
    query: str,
    top_k: int = 5,
    type_filter: Optional[str] = None,
    provenance: bool = False,
) -> list[dict]:
    """Search the typed knowledge graph by semantic similarity.

    Returns top_k node hits with id / type / name / score / tentative,
    plus the structural edges (grounded_by_ids, related_to_ids) so the
    caller can follow provenance without a second query.

    **Statement text is NOT returned by design.** The agent finds which
    nodes are relevant; for full-text retrieval, see get_node (gated
    behind KNOW_THYSELF_ALLOW_FULL_TEXT) or read the source graph
    directly out-of-band.

    Args:
      query: Natural-language query.
      top_k: Max number of results. Default 5.
      type_filter: If set, restrict to nodes of this type (observation,
        reference, overlap, novel, practice, emergent, equivalency,
        open, now, theme, period). Otherwise no filter.
      provenance: If true, multiply scores by tier(type) and apply the
        tentative-flag penalty (References > Overlaps > Observations >
        Practices ≈ Emergents > Novels > Opens, with tentative penalty).
        False = pure cosine similarity.

    Returns:
      List of {id, type, name, score, tentative, grounded_by_ids,
      related_to_ids} dicts, sorted by score descending. Empty list if
      the query has no in-vocab terms (TF-IDF backend) or the type
      filter has no matches.
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
            "grounded_by_ids": nodes[i].get("grounded_by_ids", []),
            "related_to_ids": nodes[i].get("related_to_ids", []),
        }
        for i in top
    ]


@mcp.tool()
def walk_provenance(
    node_id: str,
    depth: int = 1,
    include_incoming: bool = True,
    max_neighbors: int = 50,
) -> dict:
    """Walk the typed-edge neighborhood of a node — returns the
    grounded_by + related_to neighbors (outbound) and optionally the
    nodes that point at this one (inbound).

    Resolves missing references gracefully — if a node references an
    id that isn't in the index (typo, shelved, freshly-renamed), that
    reference is dropped with a note in `unresolved`.

    Use this AFTER search_graph when you've found the right node and
    want its provenance neighborhood without running another semantic
    query for each related id. Lightweight by default — neighbor
    entries carry id/type/name/tentative only, no statement bodies.
    For full text, see get_node (gated).

    Args:
      node_id: Full or unambiguous short node id (same lookup as
        get_node — "O04" matches "O04-daughter-grades-recovered").
      depth: Hops to walk. 1 = immediate neighbors only (default).
        2+ = recursive walk; capped by max_neighbors.
      include_incoming: If True, also return nodes that reference this
        node (inverse edges built at load time). Default True.
      max_neighbors: Hard cap on total nodes returned across the walk.
        Prevents runaway expansion on densely-connected nodes.
        Default 50.

    Returns:
      {
        node: {id, type, name, tentative},        # the queried node
        outgoing: {                               # this node -> ...
          grounded_by: [light_node, ...],
          related_to: [light_node, ...],
        },
        incoming: [                               # ... -> this node
          {edge_type: "grounded_by"|"related_to", source: light_node},
          ...
        ],
        unresolved: [str, ...],                   # ids in edges that
                                                  # didn't resolve
        depth_reached: int,
        truncated: bool,                          # hit max_neighbors?
      }
    """
    _ensure_loaded()
    root = _resolve(node_id)
    if root is None:
        return {"error": f"no node with id {node_id!r}"}

    seen: set[str] = {root["id"]}
    unresolved: list[str] = []
    truncated = False

    def expand(nid: str):
        node = _resolve(nid)
        if node is None:
            return None, [], []
        gb = []
        for tgt_id in node.get("grounded_by_ids", []):
            tgt = _resolve(tgt_id)
            if tgt is None:
                unresolved.append(tgt_id)
            else:
                gb.append(tgt)
        rt = []
        for tgt_id in node.get("related_to_ids", []):
            tgt = _resolve(tgt_id)
            if tgt is None:
                unresolved.append(tgt_id)
            else:
                rt.append(tgt)
        return node, gb, rt

    frontier = [(root["id"], 0)]
    out_grounded: dict[str, list[dict]] = {root["id"]: []}
    out_related: dict[str, list[dict]] = {root["id"]: []}

    while frontier:
        nid, d = frontier.pop(0)
        if d >= depth:
            continue
        node, gb, rt = expand(nid)
        if node is None:
            continue
        out_grounded[nid] = [_light(t) for t in gb]
        out_related[nid] = [_light(t) for t in rt]
        for tgt in gb + rt:
            if len(seen) >= max_neighbors:
                truncated = True
                break
            if tgt["id"] not in seen:
                seen.add(tgt["id"])
                if d + 1 < depth:
                    frontier.append((tgt["id"], d + 1))
        if truncated:
            break

    incoming = []
    if include_incoming:
        for src_id, etype in _INCOMING.get(root["id"], []):
            src = _NODE_BY_ID.get(src_id)
            if src is not None:
                incoming.append({"edge_type": etype, "source": _light(src)})

    return {
        "node": _light(root),
        "outgoing": {
            "grounded_by": out_grounded.get(root["id"], []),
            "related_to": out_related.get(root["id"], []),
        },
        "incoming": incoming,
        "unresolved": sorted(set(unresolved)),
        "depth_reached": depth,
        "truncated": truncated,
    }


if ALLOW_FULL_TEXT:

    @mcp.tool()
    def get_node(node_id: str) -> dict:
        """Fetch a single node by id with full statement and metadata.

        Available because KNOW_THYSELF_ALLOW_FULL_TEXT=1. Use only when
        full-text egress is acceptable for the configured graph.

        Supports short-id lookup: passing "O04" matches the unique node
        whose id starts with "O04-" (e.g. "O04-daughter-grades-recovered").
        If the prefix matches multiple nodes, returns the candidate list.

        Args:
          node_id: Full id ("O04-daughter-grades-recovered", "NOW") or
            unambiguous short id ("O04", "P01").

        Returns:
          {id, type, name, tentative, statement, grounded_by_ids,
          related_to_ids} on success, or {error: <msg>, matches?: [...]}
          on failure.
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
            "grounded_by_ids": n.get("grounded_by_ids", []),
            "related_to_ids": n.get("related_to_ids", []),
        }


if __name__ == "__main__":
    mcp.run()
