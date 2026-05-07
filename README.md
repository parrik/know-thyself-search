# know-thyself-search

Retrieval, dashboard, and MCP server for [know-thyself](https://github.com/parrik/know-thyself) graphs. Vector embedding + top-k search + IDs-only-by-default privacy gate at the wire. Renderers: interactive fractal dashboard, graphviz, mandala, printable PDF.

## quick start

```bash
pip install -r requirements.txt
python -m know_thyself.retrieval.embed examples/example-graph-extended.yaml
python -m know_thyself.retrieval.search "when did the grades start improving"
```

## features

```bash
# retrieval
python -m know_thyself.retrieval.embed graph.yaml      # build vector index (TF-IDF default; OpenAI / sentence-transformers optional)
python -m know_thyself.retrieval.search "query"        # top-k hits
python -m know_thyself.retrieval.compare "query"       # three ranking modes side-by-side

# renders
python -m know_thyself.render.dashboard graph.yaml     # interactive HTML, NOW node centered
python -m know_thyself.render.graphviz graph.yaml      # static graph + machine-checkable validation (rules 1-6 from SCHEMA.md)
python -m know_thyself.render.mandala graph.yaml       # concentric rings projection
python -m know_thyself.render.printable graph.yaml     # multi-page printable PDF
```

## mcp server

```bash
pip install "mcp[cli]"
claude mcp add know-thyself -s user \
  -e PYTHONPATH=/path/to/know-thyself-search \
  -e KNOW_THYSELF_INDEX=/path/to/graph-embeddings.json \
  -e KNOW_THYSELF_GRAPH=/path/to/graph.yaml \
  -- python -m know_thyself.retrieval.server
```

`KNOW_THYSELF_GRAPH` enables mtime-based auto-rebuild when the YAML is newer than the index.

## privacy

The MCP server is **IDs-only by default**. `search_graph` and `walk_provenance` return `id / type / name / score / tentative` plus structural edges — never statement text. `get_node` returns full statement and is hidden unless `KNOW_THYSELF_ALLOW_FULL_TEXT=1` is set in the server's environment.

Once a graph contains personal content and a cloud-LLM client connects, statement text crossing the wire is the leak. The gate keeps that decision explicit, per graph.

## layout

```
know_thyself/
├── retrieval/    embed / search / compare / server
└── render/       dashboard / graphviz / mandala / printable
examples/
└── example-graph-extended.yaml    87-node case study
```

## ack

[know-thyself](https://github.com/parrik/know-thyself) — schema and elicitation prompt. [Alex Navarro case study](https://parrik.com/alex-case-study.html#spine) — what an 87-node personal graph looks like.

## license

MIT.
