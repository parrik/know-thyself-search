# know-thyself-search

**Retrieval, MCP server, and renderers for [know-thyself](https://github.com/parrik/know-thyself) graphs — built for an AI agent as the primary reader.**

The schema lives at [parrik/know-thyself](https://github.com/parrik/know-thyself). This repo is the tooling: vector embedding, top-k search, three ranking-mode comparison, an MCP server with an IDs-only privacy gate, and four renderers (interactive dashboard, static graphviz, mandala, multi-page printable PDF).

> **Why two repos?** Schema and elicitation prompt are stable and small; tooling iterates faster and brings dependencies. Splitting lets the schema repo stay text-only and the search repo move quickly.

---

## Privacy posture — the reason this lives in a separate repo

The MCP server (`know_thyself.retrieval.server`) is **IDs-only by default**:

- `search_graph` and `walk_provenance` return `id / type / name / score / tentative` plus structural edges. Never statement text.
- `get_node` returns full statement and is **hidden** unless `KNOW_THYSELF_ALLOW_FULL_TEXT=1` is set in the server's environment.

Once a graph contains personal content and a cloud-LLM client connects, statement text crossing the wire is the leak. The gate keeps that decision explicit, per graph. This is the [#4 wedge](https://github.com/parrik/know-thyself#why-this-exists-when-claude-memory-ships-out-of-the-box) against every comparable memory project.

---

## Install

```bash
git clone https://github.com/parrik/know-thyself-search
cd know-thyself-search
pip install -r requirements.txt
```

The package is `know_thyself` (same name as the schema repo's prompt artifact, kept stable so existing MCP setups don't break).

---

## Retrieval

A graph YAML fits in a model's context at 200 nodes. At 2,000 it doesn't, and even when it does the model wastes attention scanning irrelevant parts. The retrieval layer turns the YAML into something an agent can actually query.

```bash
pip install pyyaml numpy
python -m know_thyself.retrieval.embed examples/example-graph-extended.yaml      # → graph-embeddings.json
python -m know_thyself.retrieval.search "when did the grades start improving"
python -m know_thyself.retrieval.compare "when has the user felt isolated"       # three ranking modes side-by-side
```

`know_thyself.retrieval.embed` vectorizes each node's `statement` (TF-IDF default; OpenAI and `sentence-transformers/local` backends optional) and writes a JSON index. `know_thyself.retrieval.search` accepts a query and returns top-k hits. `know_thyself.retrieval.compare` shows the same query under pure cosine, type-filtered, and provenance-reranked modes — what each layer earns is the lesson.

---

## MCP server

```bash
pip install "mcp[cli]"
claude mcp add know-thyself -s user \
  -e PYTHONPATH=/path/to/know-thyself-search \
  -e KNOW_THYSELF_INDEX=/path/to/graph-embeddings.json \
  -e KNOW_THYSELF_GRAPH=/path/to/graph.yaml \
  -- python -m know_thyself.retrieval.server
```

Setting `KNOW_THYSELF_GRAPH` enables mtime-based auto-rebuild of the index whenever the source YAML is newer. `PYTHONPATH` points at this repo's root so the package is importable regardless of the client's working directory.

To expose statement text (off by default), add `-e KNOW_THYSELF_ALLOW_FULL_TEXT=1`. Decide per graph whether you want that.

---

## Renderers

Each module declares its own dependency floor; install only what you need.

```bash
python -m know_thyself.render.dashboard your-graph.yaml   # interactive HTML, NOW node centered
python -m know_thyself.render.graphviz your-graph.yaml    # static graphviz diagram + machine-checkable validation (rules 1–6 from SCHEMA.md)
python -m know_thyself.render.mandala your-graph.yaml     # concentric rings projection
python -m know_thyself.render.printable your-graph.yaml   # multi-page PDF
```

The dashboard is the primary surface; graphviz doubles as a validator (it surfaces SCHEMA.md violations 1–6 as part of rendering); mandala and printable are presentation-only.

---

## Layout

```
know-thyself-search/
├── README.md                  this file
├── LICENSE
├── requirements.txt           dependency floors (most optional per backend)
├── know_thyself/              importable package — name kept stable across the schema/tooling split
│   ├── retrieval/             embed / search / compare / server (MCP)
│   └── render/                dashboard / graphviz / mandala / printable
└── examples/
    └── example-graph-extended.yaml    87-node case study (mirrored from the schema repo for self-contained running)
```

For richer fixtures (rendered PNG/SVG, the `alex-*` companion files, `example-graph.yaml` minimal), see the [schema repo](https://github.com/parrik/know-thyself/tree/main/examples).

---

## Companion

- [know-thyself](https://github.com/parrik/know-thyself) — the schema and elicitation prompt.
- [Alex Navarro case study](https://parrik.com/alex-case-study.html#spine) — what an 87-node personal graph looks like, opened to the spine view.
- [Companion essay](https://parrik.com/essays/know-thyself/) — the full argument.

MIT licensed. See `LICENSE`.
