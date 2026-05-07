#!/usr/bin/env python3
"""
embed.py — produce a vector index for a typed knowledge-graph YAML file.

Reads a typed-node graph, embeds each node's `statement` text, and writes
a JSON index keyed by node id, carrying the vector + the node metadata
needed for ranking (type, name, tentative flag).

Two backends, picked by --backend:
  tfidf  — hand-rolled TF-IDF over a bag of word-tokens. No deps beyond
           PyYAML + numpy. The classical inverted-index baseline — exactly the shape
           you can build on top of Lucene's inverted index. Good enough for
           a few thousand nodes.
  openai — text-embedding-3-small (1536-dim) via the OpenAI API. Needs
           `pip install openai` and OPENAI_API_KEY in the environment. The
           "modern dense retrieval" substrate.

Both write the same JSON shape so search.py reads either interchangeably —
which is the essay's point: the substrate changes, the shape doesn't.

Usage:
  python embed.py example-graph-extended.yaml
  python embed.py example-graph-extended.yaml --backend openai
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pip install pyyaml")

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: pip install numpy")


# ──────────────────────────────────────────────────────────────────────
# YAML loader — list-of-dicts at top level, each with `id`, `type`,
# `name`, `statement`. Anything without an `id` is skipped (header
# metadata, comments-as-nodes).
# ──────────────────────────────────────────────────────────────────────

_NODE_ID_RE = re.compile(r"(?m)^- id:\s*(\S+)")
_NODE_TYPE_RE = re.compile(r"(?m)^  type:\s*(\S+)")
_NODE_NAME_RE = re.compile(r"(?m)^  name:\s*(?:\"([^\"]*)\"|'([^']*)'|(.+))$")
_NODE_TENTATIVE_RE = re.compile(r"(?m)^  tentative:\s*(true|True|yes|YES)\b")
_NODE_SHELVED_RE = re.compile(r"(?m)^  shelved:\s*(true|True|yes|YES)\b")
_STATEMENT_BLOCK_RE = re.compile(
    r"(?ms)^  statement:\s*\|.*?\n((?:    .*\n|\s*\n)+)"
)
_GROUNDED_BY_BLOCK_RE = re.compile(
    r"(?ms)^  grounded_by:[ \t]*\n((?:[ \t]*-[ \t].*\n)+)"
)
_RELATED_TO_BLOCK_RE = re.compile(
    r"(?ms)^  related_to:[ \t]*\n((?:[ \t]*-[ \t].*\n)+)"
)
# Match a leading node-id token at the start of a list item. Node IDs
# follow TYPE-PREFIX + DIGITS + optional kebab slug ("O04-some-slug",
# "P01-some-slug", "R102", "EQ01-foo"). NOW is the singleton top-of-stack
# node and matches as a literal. Anything that doesn't look like a node
# id (free-text source citations, URLs, file paths) is silently skipped
# — those are provenance evidence, not graph edges.
_LEADING_NODE_ID_RE = re.compile(r"^([A-Z]{1,3}\d+(?:-[a-z0-9-]+)?|NOW)\b")


def _filter_node_id_refs(items):
    """Pull leading node-id tokens out of a list-of-strings edge block.

    Tolerates trailing parenthetical commentary ("R16-foo (parent context)"
    -> "R16-foo") and ignores list items that don't begin with an id."""
    if not items:
        return []
    ids = []
    for item in items:
        if not isinstance(item, str):
            continue
        s = item.strip().strip("`'\"")
        m = _LEADING_NODE_ID_RE.match(s)
        if m:
            ids.append(m.group(1))
    return ids


def _extract_edge_block(block_text):
    """Parse a captured grounded_by/related_to block (regex fallback path).

    Returns a list of leading-node-ids extracted from each list item.
    Continuation lines (wrapped list items) are ignored — we only need
    the leading id, which sits on the line that begins with '- '."""
    if not block_text:
        return []
    items = []
    for line in block_text.splitlines():
        # Match list item start: any leading whitespace, then "- ".
        m = re.match(r"^\s*-\s+(.+)$", line)
        if m:
            items.append(m.group(1))
    return _filter_node_id_refs(items)


def _regex_node(block):
    """Extract a node dict from a per-node text block by regex.

    Used as a fallback when yaml.safe_load can't parse the block (e.g.
    list items with unquoted colons, or other hand-curation quirks).
    Captures id, type, name, statement, tentative, shelved, plus the
    structured-edge node-id references in grounded_by / related_to."""
    m_id = _NODE_ID_RE.search(block)
    if not m_id:
        return None
    n = {"id": m_id.group(1)}
    if m := _NODE_TYPE_RE.search(block):
        n["type"] = m.group(1).strip().strip('"\'')
    if m := _NODE_NAME_RE.search(block):
        n["name"] = (m.group(1) or m.group(2) or m.group(3) or "").strip()
    if _NODE_TENTATIVE_RE.search(block):
        n["tentative"] = True
    if _NODE_SHELVED_RE.search(block):
        n["shelved"] = True
    if m := _STATEMENT_BLOCK_RE.search(block):
        # Strip the 4-space indent off each non-empty line.
        raw = m.group(1)
        lines = [(ln[4:] if ln.startswith("    ") else ln) for ln in raw.splitlines()]
        n["statement"] = "\n".join(lines).rstrip()
    gb = _GROUNDED_BY_BLOCK_RE.search(block)
    n["grounded_by_ids"] = _extract_edge_block(gb.group(1) if gb else "")
    rt = _RELATED_TO_BLOCK_RE.search(block)
    n["related_to_ids"] = _extract_edge_block(rt.group(1) if rt else "")
    return n


# Map richer typed-edge relations onto the 2-bucket model the simplified
# schema uses. The simple shape has two buckets (grounded_by = provenance
# / supporting evidence; related_to = everything else); richer schemas use
# typed edges with relations like "grounds" / "emergent_from" / "informs".
# We collapse the typed form into the 2-bucket form for retrieval —
# provenance-shaped relations land in grounded_by_ids; associative
# relations land in related_to_ids.
_PROVENANCE_RELATIONS = {"grounded_by", "grounds", "derived_from"}


def _bucket_typed_edges(edges, gb_out, rt_out):
    """Process a list of {to, relation} edge dicts (typed-edge shape)
    into the two id buckets, in place."""
    if not edges:
        return
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        tgt = edge.get("to")
        if not isinstance(tgt, str):
            continue
        m = _LEADING_NODE_ID_RE.match(tgt.strip())
        if not m:
            continue
        tgt_id = m.group(1)
        relation = (edge.get("relation") or "").lower().strip()
        if relation in _PROVENANCE_RELATIONS:
            gb_out.append(tgt_id)
        else:
            rt_out.append(tgt_id)


def _annotate_edges(node):
    """For a yaml.safe_load-parsed node, populate grounded_by_ids /
    related_to_ids from any supported edge shape:
      - simple buckets: grounded_by: [id, ...], related_to: [id, ...]
      - typed edges:    edges: [{to: id, relation: <type>}, ...]
    Either or both may be present; the result is the union, with
    leading-id extraction applied to each candidate. Idempotent —
    safe to call on regex-parsed nodes too (already populated)."""
    if "grounded_by_ids" in node and "related_to_ids" in node:
        return node
    gb = list(_filter_node_id_refs(node.get("grounded_by", [])))
    rt = list(_filter_node_id_refs(node.get("related_to", [])))
    _bucket_typed_edges(node.get("edges"), gb, rt)
    node["grounded_by_ids"] = gb
    node["related_to_ids"] = rt
    return node


def load_nodes(path):
    text = Path(path).read_text()

    # Fast path: file is valid YAML (the example-graph shape — pure list
    # of node dicts, or a mapping with a list-shaped value).
    try:
        data = yaml.safe_load(text)
        if not isinstance(data, list):
            for v in data.values() if isinstance(data, dict) else []:
                if isinstance(v, list):
                    data = v
                    break
        if isinstance(data, list):
            return [_annotate_edges(n) for n in data if isinstance(n, dict) and n.get("id")]
    except yaml.YAMLError:
        pass

    # Fallback: file mixes leading mapping keys (e.g. version:, owner:)
    # with the node list at the same level, or has hand-curation quirks
    # that break strict YAML parsing. Split into per-node blocks at
    # lines beginning with "- id:" and parse each block; if a block
    # won't parse, regex-extract the fields we need.
    blocks = re.split(r"(?m)^(?=- id:)", text)
    nodes = []
    for block in blocks:
        if not block.lstrip().startswith("- id:"):
            continue
        try:
            parsed = yaml.safe_load(block)
            if isinstance(parsed, list):
                nodes.extend(_annotate_edges(n) for n in parsed if isinstance(n, dict) and n.get("id"))
                continue
        except yaml.YAMLError:
            pass
        n = _regex_node(block)
        if n is not None:
            nodes.append(n)
    return nodes


# ──────────────────────────────────────────────────────────────────────
# TF-IDF backend (no external embedding service)
#
# This is the same shape Lucene + classical IR use: tokenize, count terms,
# weight by inverse document frequency, store as a vector. At 100 nodes
# brute-force cosine over these vectors is microseconds. The point of
# including it is to make the shape visible: even without neural
# embeddings, the retrieval pattern is identical.
# ──────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the and or but if then else of for to in on at by with as is "
    "are was were be been being have has had do does did this that "
    "these those it its them they their there here from not no yes can "
    "could should would may might will shall just so very also "
    "i you he she we us our your my his her me him".split()
)


def tokenize(text):
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1]


def tfidf_embed(nodes):
    """Return (vocab, vectors) where vectors is (N, V) numpy array."""
    docs = [tokenize(n.get("statement", "") + " " + n.get("name", "")) for n in nodes]

    df = Counter()
    for d in docs:
        for t in set(d):
            df[t] += 1

    # Drop terms that appear in only 1 doc (noise) and terms in >50% of
    # docs (uninformative). Same heuristic Lucene + standard IR use.
    N = len(docs)
    kept = [t for t, c in df.items() if 1 < c < N * 0.5]
    vocab = {t: i for i, t in enumerate(kept)}

    V = len(vocab)
    matrix = np.zeros((N, V), dtype=np.float32)

    for i, d in enumerate(docs):
        if not d:
            continue
        tf = Counter(d)
        for term, count in tf.items():
            j = vocab.get(term)
            if j is None:
                continue
            # Standard log-scaled TF * log IDF
            tf_w = 1.0 + math.log(count)
            idf_w = math.log((N + 1) / (df[term] + 1)) + 1.0
            matrix[i, j] = tf_w * idf_w

    # L2-normalize rows so cosine similarity reduces to dot product.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms
    return vocab, matrix


# ──────────────────────────────────────────────────────────────────────
# OpenAI embeddings backend (modern dense retrieval)
# ──────────────────────────────────────────────────────────────────────

def openai_embed(nodes, model="text-embedding-3-small"):
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai backend requires the openai package; pip install openai"
        ) from e
    client = OpenAI()  # reads OPENAI_API_KEY from env
    texts = [(n.get("statement") or "") + "\n\n" + (n.get("name") or "") for n in nodes]
    print(f"  calling OpenAI {model} for {len(texts)} statements...", file=sys.stderr)
    resp = client.embeddings.create(input=texts, model=model)
    vectors = np.array([d.embedding for d in resp.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return None, vectors  # no vocab — vectors are dense


# ──────────────────────────────────────────────────────────────────────
# Local sentence-transformers backend (dense, free, offline)
#
# Same dense-retrieval shape as the OpenAI backend — every node becomes
# a fixed-dim vector encoding learned semantic features, so cosine
# captures synonymy that TF-IDF can't (mentor ≈ teacher, sober ≈ clean).
# Model runs on CPU; first call downloads weights (~80MB for MiniLM,
# cached at ~/.cache/huggingface/). No API key, no network after that.
# ──────────────────────────────────────────────────────────────────────

def local_embed(nodes, model="sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "local backend requires sentence-transformers; "
            "pip install sentence-transformers"
        ) from e
    print(f"  loading {model} (first run downloads ~80MB to ~/.cache/huggingface/)...", file=sys.stderr)
    encoder = SentenceTransformer(model)
    texts = [(n.get("statement") or "") + "\n\n" + (n.get("name") or "") for n in nodes]
    print(f"  encoding {len(texts)} statements...", file=sys.stderr)
    vectors = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    return None, vectors  # no vocab — vectors are dense


# ──────────────────────────────────────────────────────────────────────

def build_index(
    graph_path,
    backend="tfidf",
    openai_model="text-embedding-3-small",
    local_model="sentence-transformers/all-MiniLM-L6-v2",
    verbose=False,
):
    """Build the index dict for a graph YAML. Single source of truth for
    the index shape — the CLI calls this, the MCP server calls this for
    in-process auto-rebuild.

    Returns the dict ready to be json.dump'd.
    """
    def log(msg):
        if verbose:
            print(msg, file=sys.stderr)

    log(f"loading graph: {graph_path}")
    nodes = load_nodes(graph_path)
    log(f"  {len(nodes)} nodes loaded")

    if backend == "tfidf":
        log("backend: tf-idf (hand-rolled, no deps)")
        vocab, vectors = tfidf_embed(nodes)
        log(f"  vocab: {len(vocab)} terms · vectors: {vectors.shape}")
        model_name = "tfidf"
    elif backend == "openai":
        log(f"backend: openai ({openai_model})")
        vocab, vectors = openai_embed(nodes, model=openai_model)
        log(f"  vectors: {vectors.shape}")
        model_name = openai_model
    elif backend == "local":
        log(f"backend: local sentence-transformers ({local_model})")
        vocab, vectors = local_embed(nodes, model=local_model)
        log(f"  vectors: {vectors.shape}")
        model_name = local_model
    else:
        raise ValueError(f"unknown backend: {backend}")

    return {
        "backend": backend,
        "model": model_name,
        "dim": int(vectors.shape[1]),
        "count": len(nodes),
        "vocab": vocab,  # only present for tfidf
        "nodes": [
            {
                "id": n["id"],
                "type": n.get("type"),
                "name": n.get("name", ""),
                "tentative": bool(n.get("tentative")),
                "statement": n.get("statement", ""),
                "grounded_by_ids": n.get("grounded_by_ids", []),
                "related_to_ids": n.get("related_to_ids", []),
                "vector": vectors[i].tolist(),
            }
            for i, n in enumerate(nodes)
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", help="Path to graph YAML file")
    ap.add_argument("-o", "--output", default="graph-embeddings.json")
    ap.add_argument(
        "-b", "--backend",
        choices=["tfidf", "openai", "local"],
        default="tfidf",
        help="Embedding backend (default: tfidf — no API needed). "
             "'local' uses sentence-transformers offline; "
             "'openai' calls the embeddings API.",
    )
    ap.add_argument(
        "--openai-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model name (used with --backend openai)",
    )
    ap.add_argument(
        "--local-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model name (used with --backend local)",
    )
    args = ap.parse_args()

    out = build_index(
        args.graph,
        backend=args.backend,
        openai_model=args.openai_model,
        local_model=args.local_model,
        verbose=True,
    )
    Path(args.output).write_text(json.dumps(out))
    print(
        f"wrote {args.output}  ({Path(args.output).stat().st_size:,} bytes)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
