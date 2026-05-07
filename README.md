# know-thyself-search

Retrieval, dashboard, and MCP server for [know-thyself](https://github.com/parrik/know-thyself) graphs. Vector embedding + top-k search + IDs-only-by-default privacy gate at the wire. Renderers: interactive fractal dashboard, graphviz, mandala, printable PDF.

Status: dashboard, mandala, and the MCP server (initialize handshake) are exercised. Graphviz-diagram render and printable PDF need the system `dot` binary. Retrieval works on the default TF-IDF backend; OpenAI / sentence-transformers backends are wired but un-exercised.

## quick start

```bash
pip install -r requirements.txt
python -m know_thyself.render.dashboard examples/example-graph-extended.yaml
# → examples/example-graph-extended.html (interactive viewer, NOW node centered)
```

## features

```bash
# renders
python -m know_thyself.render.dashboard graph.yaml     # ✓ interactive HTML, NOW-centered, fractal/spine/eyes/canaries panels
python -m know_thyself.render.mandala graph.yaml       # ✓ after `pip install matplotlib` — concentric rings + risk-corridor projections
python -m know_thyself.render.graphviz graph.yaml      # ✓ validator (all 10 SCHEMA.md rules → validation.txt; needs `pip install graphviz`); 🚧 diagram render additionally needs the system `dot` binary (`brew install graphviz` / `apt-get install graphviz`)
python -m know_thyself.render.printable graph.yaml     # 🚧 cover renders with `reportlab pypdf`; full + spine pages need the system `dot` binary

# retrieval — TF-IDF backend tested; OpenAI / sentence-transformers wired but un-exercised
python -m know_thyself.retrieval.embed graph.yaml      # ✓ builds index; --backend {openai,local} not exercised end-to-end
python -m know_thyself.retrieval.search "query"        # ✓ top-k hits on TF-IDF
python -m know_thyself.retrieval.compare "query"       # ✓ three ranking modes side-by-side (cosine / type-filtered / provenance-reranked)
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

✓ initialize handshake passes (`protocolVersion 2024-11-05`, advertises `tools/prompts/resources`); end-to-end retrieval-tool calls in a real client untested. `KNOW_THYSELF_GRAPH` enables mtime-based auto-rebuild when the YAML is newer than the index.

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

## todos

- exercise the MCP server against a real client (Claude Code / Claude Desktop) end-to-end on retrieval calls
- exercise OpenAI and sentence-transformers retrieval backends
- benchmark retrieval quality on a 200+ node graph

## ack

[know-thyself](https://github.com/parrik/know-thyself) — schema and elicitation prompt. [Alex Navarro case study](https://parrik.com/alex-case-study.html#tab-spine) — what an 87-node personal graph looks like.

## license

MIT.
