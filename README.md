# know-thyself-search

Retrieval over a typed, provenance-tagged personal knowledge graph — built with an AI agent as the primary reader.

Companion to **[know-thyself](https://github.com/parrik/know-thyself)** (the schema) and the essay **[*Know Thyself: Search Was Never About Humans*](https://parrik.com/essays/know-thyself-search/)** (the argument).

The know-thyself scaffold gives you a typed YAML graph of yourself: references, observations, overlaps, novels, emergents, equivalencies, opens — each carrying provenance. This scaffold gives you retrieval on top of it, of the shape an AI agent actually needs.

---

## What this is

Three CLIs, ~300 LOC of stdlib + numpy, runnable today against [Alex's example graph](examples/example-graph-extended.yaml) (87 fictional nodes) or your own:

```bash
pip install pyyaml numpy

python embed.py examples/example-graph-extended.yaml
python search.py "when did Mira's grades start improving"
python compare.py "when have I felt isolated"
```

`embed.py` reads the YAML, vectorizes each node's `statement`, and writes `graph-embeddings.json`. `search.py` accepts a query and returns top-k matches. `compare.py` shows the same query under three retrieval modes side-by-side.

## What it teaches

The interesting layer is what each retrieval mode earns you over the previous one:

| Mode | What changes | What it earns |
|---|---|---|
| **A.** Pure cosine | Bag-of-vectors, no schema awareness | The IR baseline. Type-blind. A tentative novel that shares vocabulary with the query can outrank a multi-grounded overlap. |
| **B.** + type filter | `--type observation` returns only dated episodes | The schema's typed nodes pay off. *"When did X happen"* becomes a structured query against episode nodes, not a fuzzy text match. |
| **C.** + provenance rerank | Cosine × tier(type) × tentative-penalty | "Attribution ≠ confidence" becomes a property of retrieval, not just a rule of interpretation. A two-grounded overlap outranks a one-derivation novel even if the novel scores higher on similarity. |

Run `compare.py` on any query to see what changes. The point is not that one mode is best — it's that what you put in the *node* (Pat-shaped: type + provenance) is the difference between vector retrieval and graph retrieval. Same substrate, different shape.

## Two backends

```bash
python embed.py graph.yaml --backend tfidf      # default, no deps
python embed.py graph.yaml --backend openai     # requires OPENAI_API_KEY
```

**TF-IDF** is the Sprinklr-2013 baseline — log-scaled term frequency × inverse document frequency, dropped to a sparse vector, brute-force cosine over a numpy matrix. At 87 nodes it returns in single-digit milliseconds. At 10,000 nodes it still does. *This is part of the lesson:* HNSW is what you reach for when brute-force linear scan stops being free, which for personal-knowledge-graph scale is well past where most people will ever go.

**OpenAI `text-embedding-3-small`** (1536-dim) is the modern dense-retrieval substrate. Same JSON output shape, same `search.py`. Swap-in is one flag. The shape doesn't change.

## Why this exists

A typed personal knowledge graph (Pat McCarthy's [open-knowledge-graph](https://github.com/patdmc/open-knowledge-graph) schema, adapted in [know-thyself](https://github.com/parrik/know-thyself)) sits unread on disk unless something can retrieve from it. Today the standard move is "paste the whole `graph.yaml` into the conversation" — works at 200 nodes, breaks at 2,000. Pat's Paper 1 makes the technical claim explicit: *"the efficient path is not to grow the context window but to grow the encoded knowledge accessible via stored adjacency: filling the graph, not the context window."*

The agent's reader is finite. The graph isn't. Retrieval is the bridge.

## MCP server

`mcp_server.py` exposes the retrieval surface as [MCP](https://modelcontextprotocol.io) tools so any MCP-aware client (Claude Code, Claude Desktop, Cursor, etc.) can query the graph natively — instead of pasting `graph.yaml` into the conversation.

```bash
pip install 'mcp[cli]'                       # the official Python MCP SDK
python embed.py path/to/graph.yaml           # build the index first
python mcp_server.py                         # stdio server (the usual MCP transport)
```

Four tools:

| Tool | What it does |
|---|---|
| `search_graph(query, top_k=5, type_filter=None, provenance=False)` | Top-k retrieval. Each hit carries the node's `grounded_by_ids` + `related_to_ids` so the caller can follow provenance without a second query. Same shape as `search.py`. |
| `get_node(node_id)` | Fetch a single node by full or unambiguous short id (`"O04"` matches `"O04-daughter-grades-recovered"`). |
| `walk_provenance(node_id, depth=1, include_incoming=True)` | Walk the typed-edge neighborhood of a node — `grounded_by` + `related_to` outbound, plus the inverse-edge nodes that point at it. Returns lightweight neighbor entries (id/type/name only); call `get_node` for full text. Unresolved references (typo'd or shelved targets) are reported separately rather than dropped silently. |
| `list_node_stats()` | Index summary: backend, total nodes, counts by type, total edges by relation. Useful as a session-opener. |

### Wire into Claude Code (user scope)

```bash
claude mcp add know-thyself-search -s user -- \
  /path/to/python /path/to/know-thyself-search/mcp_server.py
```

Verify with `claude mcp list`. The server then loads in every Claude Code session; tools appear as `mcp__know-thyself-search__search_graph`, etc.

The index is loaded from `graph-embeddings.json` next to `mcp_server.py` by default. Override with `KNOW_THYSELF_INDEX=/path/to/index.json`.

### Re-embed when the graph changes

The index is a snapshot. After editing `graph.yaml`, re-run `python embed.py path/to/graph.yaml` to refresh `graph-embeddings.json`. The MCP server picks up the new index on next launch.

## What's missing (roadmap)

- **Sub-statement chunking.** Each node currently maps to a single vector regardless of statement length. For long observation nodes that accumulate many dated sub-entries (e.g. a running sobriety log), the whole-statement vector gets averaged across all of them — a query targeting the most-recent sub-entry can fail to surface the parent node because the relevant content is diluted. Paragraph-grain or section-grain chunking with max-pool aggregation is the fix.
- **Recency-aware reranking.** Stale nodes and fresh ones compete on cosine similarity alone. Park et al. (2023) recency × importance × relevance triple, applied to a typed graph. Data is already in the file (dated sub-sections + file mtime); this is one more knob in `search_graph`.
- **Pattern-shaped structural queries.** "Find all nodes where the actor chose narrowness over leverage" isn't a semantic query — it's a typed-edge / typed-tag pattern. Embeddings don't do this. A small `find_pattern_nodes(filters)` tool against schema tags would.
- **HNSW at scale.** Above ~10K nodes brute-force matrix multiply gets uncomfortable. `hnswlib` integration is ~30 LOC; left out because at personal-graph scale you don't need it. The point is *the moment* you'd need it — that's a separable chapter (the `etudes/hnsw-crossover/` benchmark measures it).

## Credit

- Schema and provenance discipline: **Patrick D. McCarthy**, [open-knowledge-graph](https://github.com/patdmc/open-knowledge-graph).
- Personal-graph adaptation: [know-thyself](https://github.com/parrik/know-thyself).
- Adjacent prior work cited in the companion essay: Mem0, Graphiti / Zep, Letta, HippoRAG, A-Mem, Park et al. (2023), Karpathy's LLM Wiki, Anthropic MCP, Will Bryk's Exa "search-for-AI" framing, Lù et al. (2025) "Build the Web for Agents."

## License

MIT. See `LICENSE`.
