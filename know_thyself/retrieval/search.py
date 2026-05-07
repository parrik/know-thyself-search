#!/usr/bin/env python3
"""
search.py — semantic search over an embedded knowledge graph.

Loads a JSON index produced by embed.py, embeds a query the same way,
and returns the top-k matching nodes. Three layers stack on top of each
other; each can be turned on independently:

  1. Pure cosine similarity         (the IR baseline)
  2. + type filter                  (--type observation)
  3. + provenance-aware reranking   (--provenance)

The point of stacking them is to make visible what each layer earns
you. Pure vector retrieval is type-blind: a tentative novel that
happens to share vocabulary with a query will outrank a well-grounded
overlap. Typed nodes give you the type filter for free. Provenance
ranking turns "Attribution ≠ confidence" into a structural property
of retrieval, not just a rule of interpretation.

Usage:
  python search.py "when have I felt isolated"
  python search.py "when have I felt isolated" -k 5 --type observation
  python search.py "when have I felt isolated" --provenance
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: pip install numpy")


# Provenance tier multipliers. Confidence ordering reflects how
# multiply-grounded the node is: References (verifiable facts) >
# Overlaps (multi-grounded patterns) > Observations (single dated
# events) > Practices (operating rules) ≈ Emergents (intersection-
# produced) > Novels (single-derivation, often tentative) > Open
# (unresolved questions). Tentative-flagged nodes get an additional
# penalty.
TYPE_TIER = {
    "reference":   1.10,
    "overlap":     1.08,
    "observation": 1.05,
    "practice":    1.00,
    "emergent":    1.00,
    "equivalency": 1.00,
    "now":         1.00,
    "novel":       0.90,
    "open":        0.92,
}
TENTATIVE_PENALTY = 0.85


def cosine_query(query_vec, matrix):
    """Cosine similarity, assuming both inputs are L2-normalized."""
    qn = query_vec / max(np.linalg.norm(query_vec), 1e-12)
    return matrix @ qn


def tfidf_vectorize_query(query, vocab):
    """Apply the same TF-IDF shape used at index time, but query-flavored
    (no IDF — we don't have collection statistics for the query, just
    weight terms equally with log-scaled TF)."""
    import math
    import re
    from collections import Counter
    token_re = re.compile(r"[a-z0-9]+")
    tokens = [t for t in token_re.findall(query.lower()) if len(t) > 1]
    if not tokens:
        return np.zeros(len(vocab), dtype=np.float32)
    tf = Counter(tokens)
    v = np.zeros(len(vocab), dtype=np.float32)
    for term, count in tf.items():
        idx = vocab.get(term)
        if idx is not None:
            v[idx] = 1.0 + math.log(count)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


def openai_vectorize_query(query, model):
    from openai import OpenAI
    client = OpenAI()
    resp = client.embeddings.create(input=[query], model=model)
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    return v / max(np.linalg.norm(v), 1e-12)


_LOCAL_ENCODER_CACHE = {}


def local_vectorize_query(query, model):
    """Encode a query with a sentence-transformers model. Caches the
    encoder per-process so repeated calls don't reload the weights."""
    encoder = _LOCAL_ENCODER_CACHE.get(model)
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model)
        _LOCAL_ENCODER_CACHE[model] = encoder
    v = encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].astype(np.float32)
    return v


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("query", help="The natural-language query")
    ap.add_argument("-e", "--embeddings", default="graph-embeddings.json")
    ap.add_argument("-k", "--top_k", type=int, default=5)
    ap.add_argument("-t", "--type", help="Filter to nodes of this type only")
    ap.add_argument(
        "--provenance",
        action="store_true",
        help="Re-rank by tier confidence (Reference > Overlap > … > Novel)",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Print full statement text for each hit (default: name only)",
    )
    args = ap.parse_args()

    index = json.loads(Path(args.embeddings).read_text())
    nodes = index["nodes"]
    backend = index["backend"]

    # Optional type filter — applied before scoring so we don't waste cycles
    if args.type:
        nodes = [n for n in nodes if n.get("type") == args.type]
        if not nodes:
            sys.exit(f"no nodes match --type {args.type}")

    matrix = np.array([n["vector"] for n in nodes], dtype=np.float32)

    # Vectorize the query in the same shape the index was built in
    if backend == "tfidf":
        vocab = index["vocab"]
        query_vec = tfidf_vectorize_query(args.query, vocab)
    elif backend == "openai":
        query_vec = openai_vectorize_query(args.query, index["model"])
    elif backend == "local":
        query_vec = local_vectorize_query(args.query, index["model"])
    else:
        sys.exit(f"unknown backend: {backend}")

    if np.allclose(query_vec, 0):
        sys.exit("query produced zero vector — no in-vocab terms (try --backend openai or --backend local)")

    scores = cosine_query(query_vec, matrix)

    # Optional provenance-aware reranking
    if args.provenance:
        adjusted = scores.copy()
        for i, n in enumerate(nodes):
            tier = TYPE_TIER.get(n.get("type"), 1.0)
            if n.get("tentative"):
                tier *= TENTATIVE_PENALTY
            adjusted[i] = scores[i] * tier
        scores = adjusted

    top = np.argsort(-scores)[: args.top_k]

    # Header
    flags = [f"backend={backend}", f"k={args.top_k}"]
    if args.type:
        flags.append(f"type={args.type}")
    if args.provenance:
        flags.append("provenance=on")
    print(f"# query: {args.query!r}")
    print(f"# {' · '.join(flags)}")
    print(f"# searched {len(nodes)} nodes")
    print()

    for rank, idx in enumerate(top, 1):
        n = nodes[idx]
        score = float(scores[idx])
        type_tag = f"[{n.get('type','?')}]"
        tent = " [tentative]" if n.get("tentative") else ""
        print(f"{rank}. {score:6.3f}  {n['id']}  {type_tag}{tent}")
        print(f"           {n.get('name', '')}")
        if args.full and n.get("statement"):
            stmt = n["statement"].strip()
            for line in stmt.splitlines():
                print(f"           │ {line}")
        print()


if __name__ == "__main__":
    main()
