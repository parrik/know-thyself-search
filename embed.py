#!/usr/bin/env python3
"""
embed.py — produce a vector index for a Know-Thyself / open-knowledge-graph
YAML file.

Reads a typed-node graph (Pat McCarthy schema), embeds each node's `statement`
text, and writes a JSON index keyed by node id, carrying the vector + the
node metadata needed for ranking (type, name, tentative flag).

Two backends, picked by --backend:
  tfidf  — hand-rolled TF-IDF over a bag of word-tokens. No deps beyond
           PyYAML + numpy. The "Sprinklr 2013 baseline" — exactly the shape
           you can build on top of Lucene's inverted index. Good enough for
           a few thousand nodes.
  openai — text-embedding-3-small (1536-dim) via the OpenAI API. Needs
           `pip install openai` and OPENAI_API_KEY in the environment. The
           "modern dense retrieval" substrate.

Both write the same JSON shape so search.py reads either interchangeably —
which is the essay's point: the substrate changes, the shape doesn't.

Usage:
  python embed.py examples/example-graph-extended.yaml
  python embed.py examples/example-graph-extended.yaml --backend openai
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
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

def load_nodes(path):
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        # Some graphs put metadata as a leading mapping; fall back to
        # extracting any list-shaped value.
        for v in data.values() if isinstance(data, dict) else []:
            if isinstance(v, list):
                data = v
                break
    nodes = [n for n in data if isinstance(n, dict) and n.get("id")]
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
    except ImportError:
        sys.exit("ERROR: pip install openai")
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

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph", help="Path to graph YAML file")
    ap.add_argument("-o", "--output", default="graph-embeddings.json")
    ap.add_argument(
        "-b", "--backend",
        choices=["tfidf", "openai"],
        default="tfidf",
        help="Embedding backend (default: tfidf — no API needed)",
    )
    ap.add_argument(
        "--openai-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model name (used with --backend openai)",
    )
    args = ap.parse_args()

    print(f"loading graph: {args.graph}", file=sys.stderr)
    nodes = load_nodes(args.graph)
    print(f"  {len(nodes)} nodes loaded", file=sys.stderr)

    if args.backend == "tfidf":
        print(f"backend: tf-idf (hand-rolled, no deps)", file=sys.stderr)
        vocab, vectors = tfidf_embed(nodes)
        print(f"  vocab: {len(vocab)} terms · vectors: {vectors.shape}", file=sys.stderr)
    else:
        print(f"backend: openai ({args.openai_model})", file=sys.stderr)
        vocab, vectors = openai_embed(nodes, model=args.openai_model)
        print(f"  vectors: {vectors.shape}", file=sys.stderr)

    out = {
        "backend": args.backend,
        "model": args.openai_model if args.backend == "openai" else "tfidf",
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
                "vector": vectors[i].tolist(),
            }
            for i, n in enumerate(nodes)
        ],
    }
    Path(args.output).write_text(json.dumps(out))
    print(
        f"wrote {args.output}  ({Path(args.output).stat().st_size:,} bytes)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
