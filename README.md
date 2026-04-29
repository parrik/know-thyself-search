# know-thyself-search

**Retrieval CLIs over a typed personal knowledge graph.** Three small tools (~300 LOC of stdlib + numpy) that turn a `graph.yaml` file into something an AI agent can actually query.

Companion to **[know-thyself](https://github.com/parrik/know-thyself)** (the schema) and the essay **[*Know Thyself: Search Was Never About Humans*](https://parrik.com/essays/know-thyself-search/)** (the argument).

## Why this exists

A personal knowledge graph in YAML works fine when you paste the whole file into Claude. At 200 nodes. At 2,000 it doesn't fit, and even when it does, the model wastes attention scanning irrelevant nodes.

**The reader is finite. The graph isn't. Retrieval is the bridge.**

This repo is the smallest thing that makes that bridge real: vectorize the nodes, query by similarity, expose it over MCP so any Claude session can ask the graph directly instead of pasting it.

## What you can do with it

```bash
pip install pyyaml numpy

python embed.py examples/example-graph-extended.yaml
python search.py "when did Mira's grades start improving"
python compare.py "when have I felt isolated"
```

Three CLIs, runnable today against [Alex's example graph](examples/example-graph-extended.yaml) (87 fictional nodes) or your own:

- `embed.py` — reads YAML, vectorizes each node's `statement`, writes `graph-embeddings.json`
- `search.py` — query in, top-k matches out
- `compare.py` — same query under three retrieval modes side-by-side

## What `compare.py` teaches

The interesting layer is what each retrieval mode earns over the previous one:

| Mode | What changes | What it earns |
|---|---|---|
| **A.** Pure cosine | Bag-of-vectors, no schema awareness | The IR baseline. Type-blind — a vocabulary-matching tentative novel can outrank a multi-grounded overlap. |
| **B.** + type filter | `--type observation` returns only dated episodes | The schema's typed nodes pay off. *"When did X happen"* becomes a structured query, not a fuzzy text match. |
| **C.** + provenance rerank | Cosine × tier(type) × tentative-penalty | **Attribution ≠ confidence** becomes a property of retrieval, not just a rule of interpretation. |

Same vectors, three shapes. The lesson: what you put in the *node* — typed, provenance-tagged — is the difference between vector retrieval and graph retrieval.

## Backends

```bash
python embed.py graph.yaml --backend tfidf      # default, no deps
python embed.py graph.yaml --backend local      # offline dense, sentence-transformers
python embed.py graph.yaml --backend openai     # cloud dense, needs OPENAI_API_KEY
```

**TF-IDF** is the classical sparse baseline. Brute-force cosine over a numpy matrix returns in single-digit milliseconds at 87 nodes — and at 10,000. HNSW is what you reach for when linear scan stops being free; for personal-graph scale, that's far past where most people will ever go.

**Local** (`all-MiniLM-L6-v2`, 384-dim) is offline dense retrieval. First run pulls ~80MB to `~/.cache/huggingface/`; thereafter no network, no API key. Use this when the graph is personal and you don't want statements leaving the machine.

**OpenAI** (`text-embedding-3-small`, 1536-dim) is the cloud dense substrate. Same JSON output, same `search.py`. One flag swap.

## MCP server

`mcp_server.py` exposes the retrieval surface as [MCP](https://modelcontextprotocol.io) tools so any MCP-aware client (Claude Code, Claude Desktop, Cursor) can query the graph natively.

```bash
pip install 'mcp[cli]'
python embed.py path/to/graph.yaml     # build the index first
python mcp_server.py                    # stdio server
```

Four tools:

| Tool | What it does |
|---|---|
| `search_graph(query, top_k, type_filter, provenance)` | Top-k retrieval. Each hit carries `grounded_by_ids` + `related_to_ids` so the caller can follow provenance without a second query. |
| `get_node(node_id)` | Fetch by full or unambiguous short id (`"O04"` matches `"O04-daughter-grades-recovered"`). |
| `walk_provenance(node_id, depth, include_incoming)` | Walk the typed-edge neighborhood — outbound `grounded_by` + `related_to` plus inverse edges. Returns lightweight neighbors; call `get_node` for full text. |
| `list_node_stats()` | Index summary: backend, total nodes, counts by type, edges by relation. Good session-opener. |

### Wire into Claude Code

```bash
claude mcp add know-thyself-search -s user -- \
  /path/to/python /path/to/know-thyself-search/mcp_server.py
```

Verify with `claude mcp list`. Tools appear as `mcp__know-thyself-search__search_graph`, etc. The index loads from `graph-embeddings.json` next to `mcp_server.py`; override with `KNOW_THYSELF_INDEX=/path/to/index.json`.

The index is a snapshot — re-run `embed.py` after editing the graph.

## Roadmap

- **Sub-statement chunking.** Long observation nodes that accumulate dated sub-entries get diluted in a single whole-statement vector. Paragraph-grain chunking with max-pool aggregation is the fix.
- **Recency-aware reranking.** Stale and fresh nodes compete on cosine alone. Recency × importance × relevance, applied to the typed graph. Data is already in the file.
- **Pattern-shaped structural queries.** "Find all nodes where the actor chose narrowness over leverage" is a typed-edge pattern, not a semantic query. A small `find_pattern_nodes(filters)` against schema tags would do it.
- **HNSW at scale.** ~30 LOC of `hnswlib`, left out because at personal-graph scale you don't need it. The benchmark in `etudes/hnsw-crossover/` measures the moment you would.

## Credit

Provenance triples come from a long lineage: W3C [RDF](https://www.w3.org/TR/rdf11-concepts/) (2004), [PROV-O](https://www.w3.org/TR/prov-overview/) (2013), and [Anthropic's citations API](https://docs.anthropic.com/en/docs/build-with-claude/citations) for the same triple inside the product surface. [Patrick McCarthy's open-knowledge-graph](https://github.com/patdmc/open-knowledge-graph) supplies formal theorems for the scientific-knowledge-graph case; the personal-graph framing — typed nodes with provenance and tentative flags, the four-scale synthesis, the temporal-validity extension — is this repo's contribution.

Adjacent work cited in the [companion essay](https://parrik.com/essays/know-thyself-search/): Mem0, Graphiti / Zep, Letta, HippoRAG, A-Mem, Park et al. (2023), Karpathy's LLM Wiki, Anthropic MCP, Will Bryk's Exa "search-for-AI" framing, Lù et al. (2025) "Build the Web for Agents."

## License

MIT. See `LICENSE`.
