"""Vector retrieval over a Know-Thyself graph YAML.

Modules:

  embed   — build a JSON index of node embeddings (TF-IDF, OpenAI, or
            offline sentence-transformers).
  search  — query the index from the command line.
  compare — show top-k under three retrieval modes side by side
            (cosine baseline, type-filtered, provenance-reranked).
  server  — wrap retrieval as an MCP tool surface; IDs-only egress by
            default with a KNOW_THYSELF_ALLOW_FULL_TEXT gate for
            statement text.
"""
