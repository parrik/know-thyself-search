#!/usr/bin/env python3
"""
compare.py — show the same query under three retrieval modes side-by-side.

The point: make visible what each layer of typed-graph retrieval earns
you over flat similarity.

  Mode A  pure cosine        (the IR baseline)
  Mode B  + type filter      (typed nodes know what they ARE)
  Mode C  + provenance rerank (Attribution ≠ Confidence, structurally)

Run it on a query where the answer should be a specific dated event
(e.g. *"when have I felt isolated"*) and watch:
  - Mode A often returns the tentative novel ABOUT isolation, not the
    grounded observation IN which isolation showed up.
  - Mode B (--type observation) returns the actual episode.
  - Mode C demotes a tentative novel that scored high on similarity —
    not because the similarity is wrong, but because the schema knows
    the novel is one-derivation and shouldn't outrank a two-grounded
    overlap.

Usage:
  python compare.py "when have I felt isolated"
  python compare.py "the daughter is doing better" --type-filter observation
"""
import argparse
import json
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: pip install numpy")

# Reuse the same machinery as search.py
from search import (
    TYPE_TIER,
    TENTATIVE_PENALTY,
    cosine_query,
    tfidf_vectorize_query,
    openai_vectorize_query,
)


def rank(query_vec, matrix, nodes, k=5, type_filter=None, provenance=False):
    if type_filter:
        keep_idx = [i for i, n in enumerate(nodes) if n.get("type") == type_filter]
        if not keep_idx:
            return []
        sub = matrix[keep_idx]
        sub_nodes = [nodes[i] for i in keep_idx]
    else:
        sub = matrix
        sub_nodes = nodes

    scores = cosine_query(query_vec, sub)

    if provenance:
        for i, n in enumerate(sub_nodes):
            tier = TYPE_TIER.get(n.get("type"), 1.0)
            if n.get("tentative"):
                tier *= TENTATIVE_PENALTY
            scores[i] = scores[i] * tier

    top = np.argsort(-scores)[:k]
    return [(sub_nodes[i], float(scores[i])) for i in top]


def fmt(hits, label, width=44):
    print(f"━━━ {label} ".ljust(width, "━"))
    if not hits:
        print("    (no hits)")
        return
    for rank_, (n, score) in enumerate(hits, 1):
        type_tag = f"[{n.get('type','?')}]"
        tent = "*" if n.get("tentative") else ""
        print(f"  {rank_}. {score:5.3f}  {n['id']}{tent} {type_tag}")
        # Truncate name to fit
        name = (n.get("name") or "").replace("\n", " ")
        if len(name) > 70:
            name = name[:67] + "..."
        print(f"            {name}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("-e", "--embeddings", default="graph-embeddings.json")
    ap.add_argument("-k", "--top_k", type=int, default=5)
    ap.add_argument(
        "--type-filter",
        default="observation",
        help="Type to use for Mode B (default: observation)",
    )
    args = ap.parse_args()

    index = json.loads(open(args.embeddings).read())
    nodes = index["nodes"]
    backend = index["backend"]
    matrix = np.array([n["vector"] for n in nodes], dtype=np.float32)

    if backend == "tfidf":
        query_vec = tfidf_vectorize_query(args.query, index["vocab"])
    elif backend == "openai":
        query_vec = openai_vectorize_query(args.query, index["model"])
    else:
        sys.exit(f"unknown backend: {backend}")

    print(f"Query: {args.query!r}")
    print(f"Index: {len(nodes)} nodes · backend={backend}")
    print()

    hits_a = rank(query_vec, matrix, nodes, k=args.top_k)
    hits_b = rank(query_vec, matrix, nodes, k=args.top_k, type_filter=args.type_filter)
    hits_c = rank(query_vec, matrix, nodes, k=args.top_k, provenance=True)

    fmt(hits_a, "MODE A — pure cosine")
    fmt(hits_b, f"MODE B — + type filter ({args.type_filter})")
    fmt(hits_c, "MODE C — + provenance rerank")

    print("(* = tentative novel)")


if __name__ == "__main__":
    main()
