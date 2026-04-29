# know-thyself-search

**Retrieval over a typed personal knowledge graph — vectors plus provenance.** Three CLIs (~300 LOC) and an MCP server that turn a `graph.yaml` file into something an AI agent can actually query.

> Companion to **[know-thyself](https://github.com/parrik/know-thyself)** (the schema) and the essay **[*Know Thyself: Search Was Never About Humans*](https://parrik.com/essays/know-thyself-search/)** (the full argument).

A YAML graph fits in Claude's context at 200 nodes. At 2,000 it doesn't, and even when it does, the model wastes attention scanning irrelevant ones. **The reader is finite. The graph isn't. Retrieval is the bridge.**

---

## Quickstart

```bash
pip install pyyaml numpy
python embed.py examples/example-graph-extended.yaml
python search.py "when did Mira's grades start improving"
python compare.py "when have I felt isolated"
```

Runs against the 87-node fictional example out of the box, or your own graph.

- `embed.py` — vectorize each node's `statement` → `graph-embeddings.json`
- `search.py` — query in, top-k matches out
- `compare.py` — same query under three retrieval modes side-by-side

---

## The lesson `compare.py` teaches

| Mode | What it earns |
|---|---|
| Pure cosine | The IR baseline. Type-blind — a vocabulary-matching tentative novel can outrank a multi-grounded overlap. |
| + type filter (`--type observation`) | Typed nodes pay off. *"When did X happen"* becomes a structured query. |
| + provenance rerank (cosine × tier × tentative-penalty) | **Attribution ≠ confidence** becomes a property of retrieval, not just a rule of interpretation. |

Same vectors, three shapes. What you put in the *node* — typed, provenance-tagged — is the difference between vector retrieval and graph retrieval.

---

## Backends and MCP

```bash
python embed.py graph.yaml --backend tfidf    # default, no deps
python embed.py graph.yaml --backend local    # offline dense (sentence-transformers)
python embed.py graph.yaml --backend openai   # cloud dense, needs OPENAI_API_KEY
```

Brute-force cosine over a numpy matrix returns in single-digit ms at 87 nodes — and at 10,000. HNSW lives in `etudes/hnsw-crossover/` for when linear scan stops being free; for personal-graph scale, that's far past where most people will ever go.

`mcp_server.py` exposes the retrieval surface as [MCP](https://modelcontextprotocol.io) tools. Wire it into Claude Code:

```bash
claude mcp add know-thyself-search -s user -- \
  /path/to/python /path/to/know-thyself-search/mcp_server.py
```

Tools: `search_graph`, `get_node`, `walk_provenance` (typed-edge neighborhood walk), `list_node_stats`. Each `search_graph` hit carries `grounded_by_ids` + `related_to_ids` so the caller can follow provenance without a second query. Re-run `embed.py` after editing the graph — the index is a snapshot.

---

## Roadmap

- **Sub-statement chunking.** Long observation nodes get diluted in a single whole-statement vector; paragraph-grain chunking with max-pool aggregation is the fix.
- **Recency-aware reranking.** Recency × importance × relevance, applied to the typed graph. Data is already in the file.
- **Pattern-shaped structural queries.** "Find all nodes where the actor chose narrowness over leverage" is a typed-edge pattern, not a semantic query.

---

## Credit

The **personal-graph** adaptation — typed nodes with provenance and tentative flags, the four-scale synthesis, the temporal-validity extension — is this work's contribution. The substrate it sits on is older: W3C [RDF](https://www.w3.org/TR/rdf11-concepts/) (2004) and [PROV-O](https://www.w3.org/TR/prov-overview/) (2013) for provenance triples, and [Patrick McCarthy's open-knowledge-graph](https://github.com/patdmc/open-knowledge-graph) for formal theorems on the **scientific** case.

Adjacent contemporaries (Mem0, Graphiti, Letta, HippoRAG, A-Mem, Park et al., MCP, Exa) are surveyed in the [companion essay](https://parrik.com/essays/know-thyself-search/).

MIT licensed. See `LICENSE`.
