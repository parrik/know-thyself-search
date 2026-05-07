"""know-thyself — typed personal-memory knowledge-graph scaffold.

The package is split into two subpackages:

  retrieval — vector index over a graph YAML, query CLIs, and an
              MCP server that exposes search to MCP-aware clients
              with an IDs-only egress gate by default.

  render    — static / interactive / printable renderers that turn
              a graph YAML into Graphviz, an HTML dashboard, a
              mandala, or a multi-page PDF.

The schema and prose live in ``docs/``; example graphs and the Alex
case-study live in ``examples/``.
"""
