#!/usr/bin/env python3
"""
render_dashboard.py — generate the Alex interactive dashboard HTML.

Apple-inspired tabbed layout (refresh, May 2 2026):

  - Today      curated single-column homepage. Hero (serif italic),
               one invitation card (today's reading entry), three quiet
               doors (Read NOW / Browse practices / Wander the graph),
               aside (counts), quiet footer.
  - Review     sub-nav: Canaries / Eyes / Tentative / Forecasts.
  - Structure  sub-nav: Spine (vis-network, force layout) / Practices
               (operating rules) / Map (concentric mandala, vanilla SVG).
  - Reference  sub-nav: Sources / Vocab.

Reads:
  - example-graph-extended.yaml   (graph: nodes + provenance)
  - alex-vocab.md                 (glossary)
  - alex-actions.md               (open threads cards)
  - alex-needs-eyes.md            (items needing review)
  - README.md, START_HERE.md, SCHEMA.md, SAFETY.md,
    RELATED_FRAMEWORKS.md, skill.md, alex-node-eli5.md (optional)

Writes:
  - example-graph-extended.html   (single self-contained file)

Requires: pyyaml only (vis-network is loaded via CDN at runtime).

Run: python3 render_dashboard.py
Open: example-graph-extended.html in any browser.
"""
import json
import math
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("ERROR: install pyyaml — pip install pyyaml")


# dashboard.py lives at know_thyself/render/dashboard.py; the layout has
# the docs at <root>/docs/ and the examples at <root>/examples/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES  = _REPO_ROOT / "examples"
_DOCS      = _REPO_ROOT / "docs"

# Default to the bundled Alex example; let the user override with a positional
# argument (matches the README's `python -m know_thyself.render.dashboard
# your-graph.yaml` contract). Companion paths (vocab / actions / eyes / eli5)
# resolve in the examples/ directory — those readers gracefully degrade if
# files are missing.
DEFAULT_YAML = _EXAMPLES / "example-graph-extended.yaml"
YAML_PATH    = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else DEFAULT_YAML
VOCAB_PATH   = _EXAMPLES / "alex-vocab.md"
ACTIONS_PATH = _EXAMPLES / "alex-actions.md"
EYES_PATH    = _EXAMPLES / "alex-needs-eyes.md"
ELI5_PATH    = _EXAMPLES / "alex-node-eli5.md"
OUT_PATH     = YAML_PATH.with_suffix(".html")


# ──────────────────────────────────────────────────────────────────────────
#  Schema — rings, type names, colors
# ──────────────────────────────────────────────────────────────────────────

TYPE_RING = {
    "now":         0,
    "practice":    1,
    "emergent":    2,
    "overlap":     3,
    "novel":       4,
    "observation": 5,
    "equivalency": 6,
    "reference":   7,
    "open":        8,
    "open-question": 8,
}

RING_RADIUS = [0, 130, 235, 340, 445, 550, 645, 730, 810]

TYPE_COLOR = {
    "now":         "#fbbf24",
    "practice":    "#ef4444",
    "emergent":    "#8b5cf6",
    "overlap":     "#10b981",
    "novel":       "#f59e0b",
    "observation": "#0ea5e9",
    "equivalency": "#14b8a6",
    "reference":   "#3b82f6",
    "open":        "#6b7280",
    "open-question": "#6b7280",
}

TYPE_LABEL = {
    "now":         "Now",
    "practice":    "Practice",
    "emergent":    "Emergent",
    "overlap":     "Overlap",
    "novel":       "Novel · tentative",
    "observation": "Observation",
    "equivalency": "Equivalency",
    "reference":   "Reference",
    "open":        "Open question",
    "open-question": "Open question",
}


# ──────────────────────────────────────────────────────────────────────────
#  Parsing — markdown helpers
# ──────────────────────────────────────────────────────────────────────────

def parse_vocab(path):
    """Lines like `- **term** — definition`."""
    entries = []
    if not path.exists():
        return entries
    for line in path.read_text().splitlines():
        m = re.match(r'^-\s+\*\*([^*]+?)\*\*\s*[—-]\s*(.+)$', line)
        if m:
            entries.append({"term": m.group(1).strip(), "def": m.group(2).strip()})
    return entries


def parse_action_cards(path):
    """Markdown of the form:
        ## Heading text
        Body text spanning one or more lines.

        ## Next heading
        ...
    Returns [{title, body}, ...] in document order.
    """
    if not path.exists():
        return []
    text = path.read_text()
    cards = []
    current = None
    for line in text.splitlines():
        h = re.match(r'^##\s+(.+)$', line)
        if h:
            if current and (current["body"].strip() or current["title"]):
                cards.append(current)
            current = {"title": h.group(1).strip().rstrip('.'), "body": ""}
        elif current is not None:
            if line.strip():
                current["body"] += (" " if current["body"] else "") + line.strip()
    if current and (current["body"].strip() or current["title"]):
        cards.append(current)
    return cards


def parse_eyes_candidates(path):
    """Parse alex-needs-eyes.md into [{title, body}, ...].

    Each `## Heading` block becomes an entry. Skips meta-style headings
    (Operational note, How to apply, Deferred, Resolved).
    """
    if not path.exists():
        return []
    text = path.read_text()
    items = []
    skip_pat = re.compile(r'^(operational note|how to apply|deferred|resolved)', re.I)
    current = None
    for line in text.splitlines():
        h = re.match(r'^##\s+(.+)$', line)
        if h:
            if current and (current["body"].strip() or current["title"]):
                if not skip_pat.match(current["title"]):
                    items.append(current)
            current = {"title": h.group(1).strip().rstrip('.'), "body": ""}
        elif current is not None:
            if line.strip():
                current["body"] += (" " if current["body"] else "") + line.strip()
    if current and (current["body"].strip() or current["title"]):
        if not skip_pat.match(current["title"]):
            items.append(current)
    return items


def parse_node_eli5(path):
    """Parse alex-node-eli5.md → {node_id: eli5_text}.

    Supports two formats:
      (1) `## <id>` then a paragraph (until next `##` or `---`)
      (2) `## Nodes` section + `### <id>` h3 entries beneath it (also `## Schema words` + `## Practice menu` siblings, which we ignore here — only node entries matter for the drawer)

    Returns {<id>: <eli5_text>}; empty dict if file is missing.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    out = {}
    id_pat = re.compile(r'^([A-Z]+\d+(?:-[A-Za-z0-9-]+)?|NOW)')

    # Format 2: split on h2; for the h2 whose title is the nodes section,
    # iterate its h3 sub-blocks. We also try h2 blocks themselves in case
    # someone authored format 1.
    h2_blocks = re.split(r'(?m)^##\s+', text)
    for block in h2_blocks[1:]:
        lines = block.splitlines()
        if not lines:
            continue
        h2_title = lines[0].strip().lower()
        rest = "\n".join(lines[1:])
        # Format 1: the h2 itself is a node id
        m = id_pat.match(lines[0].strip())
        if m and not re.search(r'(?m)^###\s+', rest):
            body_lines = []
            for ln in lines[1:]:
                if ln.strip() == '---':
                    break
                body_lines.append(ln)
            body = "\n".join(body_lines).strip()
            if body:
                out[m.group(1)] = body
            continue
        # Format 2: this h2 contains h3 sub-entries — only treat as node
        # entries when the section is the nodes section (skip schema/practice).
        if 'node' not in h2_title:
            continue
        for sub in re.split(r'(?m)^###\s+', rest)[1:]:
            slines = sub.splitlines()
            if not slines:
                continue
            stitle = slines[0].strip()
            sm = id_pat.match(stitle)
            if not sm:
                continue
            sbody_lines = []
            for ln in slines[1:]:
                if ln.strip() == '---':
                    break
                sbody_lines.append(ln)
            sbody = "\n".join(sbody_lines).strip()
            if sbody:
                out[sm.group(1)] = sbody
    return out


def extract_now_section(now_text, heading_pattern):
    """From a NOW.statement, extract `## heading_pattern` body until next `## ` or end."""
    if not now_text:
        return ""
    pat = r'^##\s+' + heading_pattern + r'.*?\n(.*?)(?=\n##\s|\Z)'
    m = re.search(pat, now_text, re.S | re.M)
    return m.group(1).strip() if m else ""


def extract_bullets(text):
    items = []
    for line in text.splitlines():
        m = re.match(r'^\s*[•\-\*]\s+(.+)$', line)
        if m:
            items.append(m.group(1).strip())
    return items


# ──────────────────────────────────────────────────────────────────────────
#  Graph parsing + ring layout
# ──────────────────────────────────────────────────────────────────────────

NODE_ID_RE = re.compile(r"\b((?:R|O|N|E|P|PR|OQ|EQ|KT)\d+(?:-[A-Za-z0-9-]+)?)\b")


def normalize_type(t):
    if not t:
        return "reference"
    t = t.strip().lower()
    if t in ("open question", "openquestion"):
        return "open"
    return t


def parse_graph(yaml_path):
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"{yaml_path.name}: must be a top-level YAML list of nodes"
        )

    nodes = []
    edges = set()
    for entry in raw:
        if not isinstance(entry, dict) or "id" not in entry:
            continue
        if entry.get("shelved") is True:
            continue
        nid = str(entry["id"])
        ntype = normalize_type(entry.get("type"))
        name = entry.get("name", nid)
        statement = entry.get("statement", "") or ""
        caveats = entry.get("caveats", "") or ""
        tentative = bool(entry.get("tentative", False))
        horizon = entry.get("horizon")

        prov = entry.get("provenance") or {}
        deriv = (prov.get("derivation") or {}) if isinstance(prov, dict) else {}
        attribution = (prov.get("attribution") or {}) if isinstance(prov, dict) else {}
        evidence = (prov.get("evidence") or {}) if isinstance(prov, dict) else {}

        related_to = entry.get("related_to") or []
        derives_from = deriv.get("from") or []
        evidence_refs = evidence.get("references") or []

        if not isinstance(related_to, list):
            related_to = [related_to]
        if not isinstance(derives_from, list):
            derives_from = [derives_from]
        if not isinstance(evidence_refs, list):
            evidence_refs = [evidence_refs]

        # is_forecast: detect via name containing "forecast" or presence of horizon
        is_forecast = False
        if ntype == "emergent":
            if horizon or "forecast" in str(name).lower():
                is_forecast = True

        # provenance count for "needs second instance" logic on Tentative panel
        prov_count = 0
        if isinstance(prov, dict):
            for k in ("attribution", "evidence", "derivation"):
                v = prov.get(k)
                if isinstance(v, dict) and v:
                    prov_count += 1
                elif isinstance(v, list) and v:
                    prov_count += len(v)

        nodes.append({
            "id": nid,
            "name": str(name),
            "type": ntype,
            "statement": str(statement),
            "caveats": str(caveats),
            "tentative": tentative,
            "horizon": horizon,
            "is_forecast": is_forecast,
            "prov_count": prov_count,
            "attribution": attribution,
            "evidence": evidence,
            "derivation": deriv,
            "related_to": [str(x) for x in related_to],
            "derives_from": [str(x) for x in derives_from],
            "evidence_refs": [str(x) for x in evidence_refs],
        })

        for r in related_to:
            edges.add((nid, str(r)))
        for r in derives_from:
            edges.add((nid, str(r)))
        for r in evidence_refs:
            edges.add((nid, str(r)))
        for m in NODE_ID_RE.finditer(statement + " " + caveats):
            ref = m.group(1)
            if ref != nid:
                edges.add((nid, ref))

    known = {n["id"] for n in nodes}
    edges_list = [{"from": a, "to": b} for (a, b) in edges if a in known and b in known]

    # Ring positions
    by_type = {}
    for n in nodes:
        if n["id"] == "NOW":
            n["x"] = 0
            n["y"] = 0
            continue
        by_type.setdefault(n["type"], []).append(n)

    for ntype, ring_nodes in by_type.items():
        ring = TYPE_RING.get(ntype, len(RING_RADIUS) - 1)
        radius = RING_RADIUS[ring]
        def k(n):
            m = re.search(r'\d+', n["id"])
            return int(m.group(0)) if m else 0
        ring_nodes.sort(key=k)
        count = len(ring_nodes)
        if count == 1:
            ring_nodes[0]["x"] = 0
            ring_nodes[0]["y"] = -radius if radius else 0
            continue
        for i, n in enumerate(ring_nodes):
            angle = -math.pi / 2 + 2 * math.pi * i / count
            n["x"] = round(radius * math.cos(angle), 1)
            n["y"] = round(radius * math.sin(angle), 1)

    # Backlinks (in-degree) — used for spine selection and declutter
    indeg = {}
    for e in edges_list:
        indeg[e["to"]] = indeg.get(e["to"], 0) + 1
    for n in nodes:
        n["backlink_count"] = indeg.get(n["id"], 0)

    # Spine: top-N by in-degree (max(8, min(20, n/10))) plus all
    # Emergent + Equivalency nodes; exclude NOW.
    n_count = len(nodes)
    top_n = max(8, min(20, n_count // 10))
    spine_ids = set(nid for nid, _ in sorted(indeg.items(), key=lambda x: -x[1])[:top_n])
    for n in nodes:
        if n["type"] in ("emergent", "equivalency"):
            spine_ids.add(n["id"])
    spine_ids.discard("NOW")
    for n in nodes:
        n["spine"] = n["id"] in spine_ids

    return {"nodes": nodes, "edges": edges_list}


def extract_now_panels(graph):
    now = next((n for n in graph["nodes"] if n["id"] == "NOW"), None)
    if not now:
        return {"frame": "", "week": [], "month": [], "quarter": [], "rules": [], "canaries": []}
    txt = now["statement"]
    return {
        "frame":    extract_now_section(txt, r'Frame'),
        "week":     extract_bullets(extract_now_section(txt, r'This week')),
        "month":    extract_bullets(extract_now_section(txt, r'This month')),
        "quarter":  extract_bullets(extract_now_section(txt, r'This quarter')),
        "rules":    extract_bullets(extract_now_section(txt, r'Standing rules')),
        "canaries": extract_bullets(extract_now_section(txt, r'Canaries')),
    }


def load_meta_files():
    """Load bundle markdown files into a dict for the drawer-based Sources tab.

    Returns {filename: {"canonical": <text>, "title": <stem>}}. Files that
    don't exist are silently skipped (e.g. alex-node-eli5.md before it lands).
    """
    # Per-file home directories — schema-shaped docs sit in docs/, the
    # Alex case-study sits in examples/, README and skill.md stay at root.
    _meta_paths = [
        _REPO_ROOT / "README.md",
        _DOCS      / "START_HERE.md",
        _DOCS      / "SCHEMA.md",
        _DOCS      / "SAFETY.md",
        _DOCS      / "RELATED_FRAMEWORKS.md",
        _REPO_ROOT / "skill.md",
        _EXAMPLES  / "alex-vocab.md",
        _EXAMPLES  / "alex-actions.md",
        _EXAMPLES  / "alex-needs-eyes.md",
        _EXAMPLES  / "alex-node-eli5.md",
    ]
    out = {}
    for p in _meta_paths:
        if not p.exists():
            continue
        out[p.name] = {
            "canonical": p.read_text(encoding="utf-8"),
            "title": p.stem.replace('-', ' ').replace('_', ' '),
        }
    return out


# Sentence-case display titles for meta files (no .md extension shown).
META_FILE_TITLES = {
    "README.md":            "Readme",
    "START_HERE.md":        "Start here",
    "SCHEMA.md":            "Schema",
    "SAFETY.md":            "Safety",
    "RELATED_FRAMEWORKS.md": "Related frameworks",
    "skill.md":             "Skill",
    "alex-vocab.md":        "Vocab",
    "alex-actions.md":      "Open threads",
    "alex-needs-eyes.md":   "Needs eyes",
    "alex-node-eli5.md":    "Plain-English overlays",
}


# ──────────────────────────────────────────────────────────────────────────
#  Favicon — concentric mandala, blues/teals palette (Alex variant)
# ──────────────────────────────────────────────────────────────────────────

FAVICON_SVG = (
    '<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 64 64%22>'
    '<rect width=%2264%22 height=%2264%22 rx=%2212%22 fill=%22%23eef5fa%22/>'
    '<circle cx=%2232%22 cy=%2232%22 r=%2226%22 fill=%22none%22 stroke=%22%23cde4ee%22 stroke-width=%221.4%22/>'
    '<circle cx=%2232%22 cy=%2232%22 r=%2218%22 fill=%22none%22 stroke=%22%2399c4d8%22 stroke-width=%221.4%22/>'
    '<circle cx=%2232%22 cy=%2232%22 r=%2210%22 fill=%22none%22 stroke=%22%237aaecb%22 stroke-width=%221.4%22/>'
    '<circle cx=%2232%22 cy=%2232%22 r=%224%22 fill=%22%231e6fd9%22/>'
    '</svg>'
)


# ──────────────────────────────────────────────────────────────────────────
#  HTML template
# ──────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alex · know-thyself</title>
<link rel="icon" href="data:image/svg+xml,__FAVICON__">
<meta name="theme-color" content="#fbfaf7">
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  * { box-sizing: border-box; }
  :root {
    --bg: #fbfaf7;
    --bg-elev: #ffffff;
    --text: #1d1d1f;
    --text-2: #525258;
    --text-3: #8e8e93;
    --line: rgba(0,0,0,0.09);
    --line-strong: rgba(0,0,0,0.16);
    --accent: #1e6fd9;
    --warm: #b8873e;
    --serif: "Charter","Iowan Old Style","Georgia",ui-serif,serif;
    --sans: -apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display","Segoe UI",Roboto,sans-serif;
    --mono: "SF Mono","Menlo",monospace;
  }
  html,body { margin:0; padding:0; background:var(--bg); color:var(--text); font-family:var(--sans); -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale; }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }

  /* === Tab bar — segmented control === */
  .tabbar { position:sticky; top:0; z-index:100; background:rgba(251,250,247,0.86); -webkit-backdrop-filter:saturate(180%) blur(14px); backdrop-filter:saturate(180%) blur(14px); border-bottom:1px solid var(--line); padding:10px 24px; height:56px; display:flex; align-items:center; justify-content:space-between; }
  .tabbar .brand { font-size:13px; color:var(--text-2); font-weight:500; }
  .tabbar .seg { display:flex; gap:0; background:rgba(0,0,0,0.05); border-radius:9px; padding:2px; }
  .tabbar button { background:transparent; border:none; cursor:pointer; font-family:var(--sans); font-size:13px; color:var(--text-2); padding:5px 14px; border-radius:7px; font-weight:500; transition:background 0.15s, color 0.15s, box-shadow 0.15s; }
  .tabbar button:hover { color:var(--text); }
  .tabbar button.active { background:var(--bg-elev); color:var(--text); box-shadow:0 1px 2px rgba(0,0,0,0.06); font-weight:590; }
  .tabbar .meta { font-size:11.5px; color:var(--text-3); font-variant:tabular-nums; font-family:var(--mono); }

  /* Sub-nav (lens picker inside a group) */
  .subnav { position:sticky; top:56px; z-index:90; background:rgba(251,250,247,0.94); -webkit-backdrop-filter:saturate(180%) blur(10px); backdrop-filter:saturate(180%) blur(10px); border-bottom:1px solid var(--line); padding:8px 24px; min-height:0; display:flex; gap:18px; align-items:center; overflow-x:auto; overflow-y:hidden; }
  .subnav.empty { display:none; }
  .subnav button { background:transparent; border:none; cursor:pointer; font-family:var(--sans); font-size:12.5px; color:var(--text-3); padding:5px 0; font-weight:500; border-bottom:2px solid transparent; transition:color 0.15s, border-color 0.15s; white-space:nowrap; }
  .subnav button:hover { color:var(--text-2); }
  .subnav button.active { color:var(--text); border-bottom-color:var(--text); font-weight:590; }

  .tab-panel { display:none; }
  .tab-panel.active { display:block; }
  .container { max-width:720px; margin:0 auto; padding:56px 32px 120px; }

  /* === Today (Apple-spaced minimal) === */
  .today-shell { max-width:640px; margin:0 auto; padding:96px 28px 120px; }
  .t-hero { margin-bottom:56px; }
  .t-eyebrow { font-size:11.5px; letter-spacing:0.16em; text-transform:uppercase; color:var(--text-3); font-weight:500; font-variant:tabular-nums; margin-bottom:18px; }
  .t-lead { font-family:var(--serif); font-size:30px; line-height:1.35; color:var(--text); margin:0 0 22px; letter-spacing:-0.012em; font-weight:400; font-style:italic; }
  .t-goals { font-size:12.5px; color:var(--text-3); letter-spacing:0.04em; }
  .t-invitation { padding:36px 0 40px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin-bottom:48px; }
  .t-inv-label { font-size:11px; text-transform:uppercase; letter-spacing:0.18em; color:var(--text-3); font-weight:600; margin-bottom:14px; }
  .t-inv-title { font-family:var(--serif); font-size:24px; line-height:1.3; color:var(--text); font-weight:500; letter-spacing:-0.008em; margin-bottom:14px; }
  .t-inv-body { font-size:15.5px; line-height:1.7; color:var(--text-2); margin-bottom:24px; max-width:56ch; font-family:var(--serif); }
  .t-inv-action { display:inline-block; font-size:14px; color:var(--accent); font-weight:500; cursor:pointer; padding:10px 20px; border:1px solid var(--accent); border-radius:22px; background:transparent; transition:background 0.15s, color 0.15s; text-decoration:none; }
  .t-inv-action:hover { background:var(--accent); color:#fff; text-decoration:none; }
  .t-doors { display:grid; grid-template-columns:repeat(3, 1fr); gap:1px; background:var(--line); border:1px solid var(--line); border-radius:12px; overflow:hidden; margin-bottom:48px; }
  .t-door { display:block; padding:22px 18px; background:var(--bg); cursor:pointer; transition:background 0.15s; text-decoration:none; color:var(--text); }
  .t-door:hover { background:var(--bg-elev); text-decoration:none; }
  .t-door-title { font-size:15px; font-weight:500; letter-spacing:-0.005em; color:var(--text); margin-bottom:4px; }
  .t-door:hover .t-door-title { color:var(--accent); }
  .t-door-sub { font-size:12px; color:var(--text-3); }
  .t-aside { text-align:center; font-size:13px; color:var(--text-3); margin-bottom:48px; line-height:1.7; }
  .t-footer { text-align:center; font-size:12px; color:var(--text-3); letter-spacing:0.02em; line-height:1.7; padding-top:32px; border-top:1px solid var(--line); }
  @media (max-width:640px) {
    .today-shell { padding:56px 18px 80px; }
    .t-lead { font-size:24px; }
    .t-invitation { padding:28px 0 32px; }
    .t-inv-title { font-size:21px; }
    .t-doors { grid-template-columns:1fr; }
  }

  /* === Panel hero (used by Spine, Practices, Eyes, Canaries, Forecasts, Tentative, Sources, Vocab) === */
  .panel-hero h1 { font-size:34px; font-weight:600; letter-spacing:-0.02em; margin:0 0 6px; line-height:1.1; }
  .panel-hero .sub { font-size:14px; color:var(--text-2); line-height:1.55; max-width:60ch; }

  /* === Network panels (Map + Spine, vis-network) === */
  #network-wrap, #network-spine-wrap { position:relative; height:calc(100vh - 96px); overflow:hidden; }
  #network, #network-spine { width:100%; height:100%; background:var(--bg); }
  #fractal-controls, #spine-controls { position:absolute; top:24px; left:24px; background:rgba(255,255,255,0.94); -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px); padding:14px 16px; border-radius:12px; border:1px solid var(--line); font-size:13px; max-width:280px; z-index:5; }
  #fractal-controls .ctitle, #spine-controls .ctitle { font-size:11px; color:var(--text-3); text-transform:uppercase; letter-spacing:0.14em; margin-bottom:8px; font-weight:600; }
  #fractal-controls .legend-item { display:flex; align-items:center; padding:4px 6px; cursor:pointer; border-radius:5px; user-select:none; font-size:13px; color:var(--text); }
  #fractal-controls .legend-item:hover { background:rgba(0,0,0,0.04); }
  #fractal-controls .legend-item.dim { opacity:0.4; }
  #fractal-controls .swatch { width:9px; height:9px; border-radius:50%; margin-right:8px; flex-shrink:0; }
  #fractal-controls input[type=text] { width:100%; padding:7px 10px; border:1px solid var(--line-strong); border-radius:7px; font-size:13px; margin-top:10px; font-family:var(--sans); background:var(--bg-elev); color:var(--text); }
  #fractal-controls .toggle-row { margin-top:12px; padding-top:10px; border-top:1px solid var(--line); display:flex; align-items:center; gap:8px; font-size:12.5px; color:var(--text-2); }
  #fractal-controls .stats, #spine-controls .stats { margin-top:10px; font-size:11px; color:var(--text-3); font-variant:tabular-nums; }
  #spine-controls .spine-blurb { font-size:13px; line-height:1.55; color:var(--text-2); margin-top:4px; max-width:240px; }
  #fractal-zoom, #spine-zoom { position:absolute; top:24px; right:24px; display:flex; flex-direction:column; gap:4px; z-index:5; }
  #fractal-zoom button, #spine-zoom button { width:32px; height:32px; background:rgba(255,255,255,0.94); border:1px solid var(--line); border-radius:7px; cursor:pointer; font-size:17px; color:var(--text); }
  #fractal-zoom button:hover, #spine-zoom button:hover { border-color:var(--accent); color:var(--accent); }

  /* === Map: vanilla SVG mandala (concentric rings) === */
  #network svg.mandala { width:100%; height:100%; cursor:grab; user-select:none; }
  #network svg.mandala:active { cursor:grabbing; }
  .node circle { transition:opacity 0.2s, r 0.15s; }
  .node text { font-family:var(--sans); font-size:11px; fill:var(--text); pointer-events:none; user-select:none; paint-order:stroke; stroke:var(--bg); stroke-width:3px; stroke-linejoin:round; }
  .node.tentative circle { stroke:var(--warm); stroke-width:2; }
  .node.now circle { stroke:#fde68a; stroke-width:3; }
  .node:hover circle { r:14; cursor:pointer; }
  .edge { stroke:rgba(0,0,0,0.15); stroke-width:0.7; fill:none; pointer-events:none; }
  .node.dim { opacity:0.18; }
  .node.dim circle { pointer-events:none; }

  /* === List rows (Practices, Canaries, Forecasts, Tentative, Eyes) === */
  .panel-list { margin-top:18px; }
  .row { display:block; padding:16px 0; border-bottom:1px solid var(--line); cursor:pointer; transition:background 0.12s; }
  .row:hover { background:rgba(0,0,0,0.015); }
  .row .rh { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
  .row .rid { font-family:var(--mono); font-size:12px; color:var(--text-3); font-weight:500; }
  .row .rname { font-size:15.5px; color:var(--text); font-weight:500; letter-spacing:-0.005em; }
  .row:hover .rname { color:var(--accent); }
  .row .rsum { font-size:14px; color:var(--text-2); margin-top:6px; line-height:1.55; }
  .row .reli5 { margin-top:8px; padding:10px 12px; background:rgba(30,111,217,0.05); border-left:2px solid var(--accent); border-radius:0 5px 5px 0; font-size:13px; line-height:1.6; color:var(--text-2); }
  .row .reli5 .velab { font-size:10px; text-transform:uppercase; letter-spacing:0.14em; color:var(--accent); font-weight:600; margin-bottom:4px; display:block; }
  .pill { display:inline-block; font-size:10.5px; padding:2px 8px; border-radius:10px; letter-spacing:0.04em; text-transform:uppercase; font-weight:600; }
  .pill.thin { background:rgba(245,158,11,0.12); color:#b8873e; }
  .pill.fc { background:rgba(251,191,36,0.18); color:#9a6b00; }
  .pill.horizon { background:rgba(184,135,62,0.10); color:var(--warm); font-variant:tabular-nums; }
  .group-head { font-size:11px; text-transform:uppercase; letter-spacing:0.16em; color:var(--text-3); font-weight:600; padding-bottom:10px; border-bottom:1px solid var(--line); margin:32px 0 0; }
  .group-head:first-of-type { margin-top:18px; }
  .empty-list { color:var(--text-3); padding:24px 0; font-style:italic; font-family:var(--serif); }

  /* === Drawer === */
  #drawer-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.16); opacity:0; pointer-events:none; transition:opacity 0.2s; z-index:199; }
  #drawer-overlay.open { opacity:1; pointer-events:auto; }
  #drawer { position:fixed; top:0; right:-620px; width:600px; max-width:96vw; height:100vh; background:var(--bg-elev); border-left:1px solid var(--line); box-shadow:-1px 0 0 var(--line), 0 12px 40px rgba(0,0,0,0.08); transition:right 0.25s cubic-bezier(0.16, 1, 0.3, 1); z-index:200; display:flex; flex-direction:column; }
  #drawer.open { right:0; }
  #drawer .dhead { padding:14px 22px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; flex-shrink:0; }
  #drawer .dtype { font-size:11px; text-transform:uppercase; letter-spacing:0.14em; color:var(--text-3); font-weight:600; }
  #drawer .dhead .controls { display:flex; gap:6px; align-items:center; }
  #drawer .dhead button { background:none; border:1px solid var(--line-strong); border-radius:7px; padding:5px 12px; cursor:pointer; font-size:13px; color:var(--text-2); font-family:var(--sans); }
  #drawer .dhead button:hover { color:var(--text); border-color:var(--text); }
  #drawer .dhead .esc-hint { font-size:11px; color:var(--text-3); font-family:var(--mono); }
  #drawer .dbody { padding:32px 40px 64px; overflow-y:auto; flex:1; }
  #drawer h1 { font-size:30px; margin:0 0 6px; font-weight:600; letter-spacing:-0.015em; line-height:1.2; color:var(--text); font-family:var(--serif); }
  #drawer .dmeta { font-family:var(--mono); font-size:12px; color:var(--text-3); margin-bottom:24px; }
  #drawer .dmeta .tent { color:var(--warm); font-weight:600; margin-right:6px; }
  #drawer .stmt { font-family:var(--serif); font-size:17px; line-height:1.7; color:var(--text); }
  #drawer .stmt p { margin:0 0 14px; }
  #drawer .stmt ul { padding-left:1.2em; margin:0 0 14px; list-style:none; }
  #drawer .stmt li { padding-left:1em; position:relative; margin:6px 0; }
  #drawer .stmt li::before { content:"·"; color:var(--text-3); position:absolute; left:0; font-weight:700; }
  #drawer .stmt strong { color:var(--text); font-weight:600; }
  #drawer .stmt em { color:var(--text); font-style:italic; }
  #drawer .stmt code { font-family:var(--mono); font-size:14px; background:rgba(0,0,0,0.04); color:var(--text); padding:1px 6px; border-radius:4px; }
  #drawer .ref { color:var(--accent); cursor:pointer; font-family:var(--mono); font-size:0.92em; font-style:normal; }
  #drawer .ref:hover { text-decoration:underline; }
  #drawer .caveats { margin:20px 0; padding:14px 16px; background:rgba(184,135,62,0.08); border-left:3px solid var(--warm); border-radius:0 8px 8px 0; font-family:var(--sans); font-size:14px; line-height:1.6; color:#4a4030; }
  #drawer .caveats .clab { font-size:11px; text-transform:uppercase; letter-spacing:0.14em; color:var(--warm); margin-bottom:8px; font-weight:600; }
  #drawer .eli5-block { margin-bottom:24px; padding:14px 18px; background:rgba(30,111,217,0.05); border-left:3px solid var(--accent); border-radius:0 8px 8px 0; font-family:var(--sans); font-size:14px; line-height:1.65; color:var(--text); }
  #drawer .eli5-block .eli5-label { font-size:11px; text-transform:uppercase; letter-spacing:0.14em; color:var(--accent); margin-bottom:8px; font-weight:600; }
  #drawer .related { margin-top:36px; padding-top:24px; border-top:1px solid var(--line); }
  #drawer .relgroup { margin-bottom:16px; }
  #drawer .rlab { font-family:var(--sans); font-size:11px; text-transform:uppercase; letter-spacing:0.14em; color:var(--text-3); margin-bottom:10px; font-weight:600; }
  #drawer .relnode { padding:11px 14px; margin:5px 0; background:var(--bg); border:1px solid var(--line); border-radius:9px; cursor:pointer; transition:all 0.15s; }
  #drawer .relnode:hover { background:rgba(30,111,217,0.04); border-color:var(--accent); }
  #drawer .relnode .rhead { display:flex; align-items:baseline; gap:8px; }
  #drawer .relnode .rid { font-family:var(--mono); font-size:12px; color:var(--accent); font-weight:500; }
  #drawer .relnode .rname { font-size:14px; color:var(--text); font-weight:500; }
  #drawer .relnode .rsum { font-size:13px; color:var(--text-2); margin-top:4px; line-height:1.5; }
  #drawer .provenance { margin-top:20px; padding:12px 14px; background:rgba(0,0,0,0.025); border-radius:9px; font-family:var(--sans); font-size:13px; color:var(--text-2); line-height:1.6; }
  #drawer .provenance .plab { font-size:11px; text-transform:uppercase; letter-spacing:0.14em; color:var(--text-3); margin-bottom:6px; font-weight:600; }
  #drawer .mentioned { margin-top:24px; padding-top:18px; border-top:1px solid var(--line); }
  #drawer .mentioned .ref { margin-right:10px; line-height:1.9; font-size:13px; }

  /* === Sources / vocab === */
  .source-list, .vocab-list { margin-top:24px; }
  .source-item { display:block; padding:18px 0; border-bottom:1px solid var(--line); text-decoration:none; color:var(--text); cursor:pointer; }
  .source-item:hover .stitle { color:var(--accent); }
  .source-item .stitle { font-size:17px; font-weight:500; transition:color 0.15s; letter-spacing:-0.005em; }
  .source-item .sdesc { font-size:14px; color:var(--text-2); margin-top:4px; line-height:1.5; }
  .vocab-row { padding:14px 0; border-bottom:1px solid var(--line); }
  .vocab-row .vterm { font-weight:600; color:var(--text); font-size:15px; }
  .vocab-row .vdef { color:var(--text-2); font-size:14px; line-height:1.6; margin-top:4px; }

  ::-webkit-scrollbar { width:9px; height:9px; }
  ::-webkit-scrollbar-track { background:transparent; }
  ::-webkit-scrollbar-thumb { background:rgba(0,0,0,0.13); border-radius:5px; }
  ::-webkit-scrollbar-thumb:hover { background:rgba(0,0,0,0.25); }

  @media (max-width: 640px) {
    .container { padding:36px 18px 80px; }
    #drawer { width:100vw; }
  }
  @media (min-width: 1100px) {
    .container { max-width:840px; padding:64px 36px 120px; }
  }
</style>
</head>
<body>
<nav class="tabbar">
  <div class="brand">Alex · know-thyself</div>
  <div class="seg" id="group-bar">
    <button data-group="today" class="active">Today</button>
    <button data-group="review">Review</button>
    <button data-group="structure">Structure</button>
    <button data-group="reference">Reference</button>
  </div>
  <div class="meta"><span id="nodecount">0</span> nodes · <span id="edgecount">0</span> edges</div>
</nav>
<nav class="subnav empty" id="subnav-bar"></nav>

<section id="tab-today" class="tab-panel active">
  <div class="today-shell">
    <header class="t-hero">
      <div class="t-eyebrow" id="hero-eyebrow"></div>
      <p class="t-lead" id="hero-lead"></p>
      <div class="t-goals">know thyself · reveal thyself</div>
    </header>

    <article class="t-invitation" id="t-invitation"></article>

    <nav class="t-doors">
      <a class="t-door" onclick="openDrawer('NOW')">
        <div class="t-door-title">Read NOW</div>
        <div class="t-door-sub">today's frame</div>
      </a>
      <a class="t-door" onclick="switchTab('practices')">
        <div class="t-door-title">Browse practices</div>
        <div class="t-door-sub">operating rules</div>
      </a>
      <a class="t-door" onclick="switchTab('fractal')">
        <div class="t-door-title">Wander the graph</div>
        <div class="t-door-sub">the map</div>
      </a>
    </nav>

    <div class="t-aside" id="t-aside"></div>

    <footer class="t-footer">
      <div>kill your darlings &middot; reveal thyself &middot; know thyself</div>
    </footer>
  </div>
</section>

<section id="tab-canaries" class="tab-panel">
  <div class="container">
    <header class="panel-hero">
      <h1>Canaries</h1>
      <div class="sub">Trip-wires Alex set for herself. Each watches a specific failure mode.</div>
    </header>
    <div class="panel-list" id="canary-list"></div>
  </div>
</section>

<section id="tab-eyes" class="tab-panel">
  <div class="container">
    <header class="panel-hero">
      <h1>Eyes</h1>
      <div class="sub">Items flagged for review. Handle deliberately.</div>
    </header>
    <div class="panel-list" id="eyes-list"></div>
  </div>
</section>

<section id="tab-tentative" class="tab-panel">
  <div class="container">
    <header class="panel-hero">
      <h1>Tentative</h1>
      <div class="sub">Claims grounded in only one observation. Watch for second instances.</div>
    </header>
    <div class="panel-list" id="tentative-list"></div>
  </div>
</section>

<section id="tab-forecasts" class="tab-panel">
  <div class="container">
    <header class="panel-hero">
      <h1>Forecasts</h1>
      <div class="sub">What Alex's graph predicts. Ordered by horizon.</div>
    </header>
    <div class="panel-list" id="forecast-list"></div>
  </div>
</section>

<section id="tab-spine" class="tab-panel">
  <div id="network-spine-wrap">
    <div id="network-spine"></div>
    <div id="spine-controls">
      <div class="ctitle">Spine</div>
      <div class="spine-blurb">The load-bearing nodes. Where the graph leans.</div>
      <div class="stats" id="spine-stats"></div>
    </div>
    <div id="spine-zoom">
      <button title="Zoom in" onclick="spineZoomBy(1.25)">+</button>
      <button title="Zoom out" onclick="spineZoomBy(0.8)">−</button>
      <button title="Fit all" onclick="spineNetwork && spineNetwork.fit({animation:{duration:300}})">⟲</button>
    </div>
  </div>
</section>

<section id="tab-practices" class="tab-panel">
  <div class="container">
    <header class="panel-hero">
      <h1>Practices</h1>
      <div class="sub">Operating rules. The shape of Alex's etudes.</div>
    </header>
    <div class="panel-list" id="practice-list"></div>
  </div>
</section>

<section id="tab-fractal" class="tab-panel">
  <div id="network-wrap">
    <div id="network"></div>
    <div id="fractal-controls">
      <div class="ctitle">Filter by type</div>
      <div class="legend-item" data-type="practice"><span class="swatch" style="background:#ef4444;"></span>Practice</div>
      <div class="legend-item" data-type="emergent"><span class="swatch" style="background:#8b5cf6;"></span>Emergent</div>
      <div class="legend-item" data-type="overlap"><span class="swatch" style="background:#10b981;"></span>Overlap</div>
      <div class="legend-item" data-type="novel"><span class="swatch" style="background:#f59e0b;"></span>Novel · tentative</div>
      <div class="legend-item" data-type="observation"><span class="swatch" style="background:#0ea5e9;"></span>Observation</div>
      <div class="legend-item" data-type="reference"><span class="swatch" style="background:#3b82f6;"></span>Reference</div>
      <div class="legend-item" data-type="open"><span class="swatch" style="background:#6b7280;"></span>Open question</div>
      <input type="text" id="search" placeholder="search id or name…" />
      <div class="toggle-row">
        <input type="checkbox" id="declutter" checked>
        <label for="declutter">Hide thin reference nodes (≤1 link in)</label>
      </div>
      <div class="stats" id="fractal-stats"></div>
    </div>
    <div id="fractal-zoom">
      <button title="Zoom in" onclick="zoomBy(1.25)">+</button>
      <button title="Zoom out" onclick="zoomBy(0.8)">−</button>
      <button title="Fit all" onclick="fitMandala()">⟲</button>
    </div>
  </div>
</section>

<section id="tab-sources" class="tab-panel">
  <div class="container">
    <header class="panel-hero">
      <h1>Sources</h1>
      <div class="sub">Files in the bundle. Click any to read.</div>
    </header>
    <div class="source-list" id="source-list"></div>
  </div>
</section>

<section id="tab-vocab" class="tab-panel">
  <div class="container">
    <header class="panel-hero">
      <h1>Vocab</h1>
      <div class="sub" id="vocab-count"></div>
    </header>
    <div class="vocab-list" id="vocab-list"></div>
  </div>
</section>

<div id="drawer-overlay" onclick="closeDrawer()"></div>
<aside id="drawer">
  <div class="dhead">
    <div class="dtype" id="drawer-type"></div>
    <div class="controls">
      <button id="drawer-back" style="display:none;" onclick="drawerBack()">← back</button>
      <span class="esc-hint">esc</span>
      <button onclick="closeDrawer()">Done</button>
    </div>
  </div>
  <div class="dbody">
    <h1 id="drawer-title"></h1>
    <div class="dmeta" id="drawer-meta"></div>
    <div class="stmt" id="drawer-stmt"></div>
  </div>
</aside>

<script>
const GRAPH = __DATA__;
const VOCAB = __VOCAB__;
const THREADS = __THREADS__;
const EYES = __EYES__;
const NOW_PANELS = __NOW_PANELS__;
const TYPE_COLOR = __TYPE_COLOR__;
const TYPE_LABEL = __TYPE_LABEL__;
const NODE_ELI5 = __NODE_ELI5__;
const META_FILES = __META_FILES__;
const META_FILE_TITLES = __META_FILE_TITLES__;

document.getElementById('nodecount').textContent = GRAPH.nodes.length;
document.getElementById('edgecount').textContent = GRAPH.edges.length;

// === Tabs (two-level: groups + sub-nav lenses) ===
const TAB_GROUPS = {
  today:     { label: 'Today',     subs: [], defaultSub: 'today' },
  review:    { label: 'Review',    subs: [
    { id: 'canaries',  label: 'Canaries' },
    { id: 'eyes',      label: 'Eyes' },
    { id: 'tentative', label: 'Tentative' },
    { id: 'forecasts', label: 'Forecasts' },
  ], defaultSub: 'canaries' },
  structure: { label: 'Structure', subs: [
    { id: 'spine',     label: 'Spine' },
    { id: 'practices', label: 'Practices' },
    { id: 'fractal',   label: 'Map' },
  ], defaultSub: 'spine' },
  reference: { label: 'Reference', subs: [
    { id: 'sources',   label: 'Sources' },
    { id: 'vocab',     label: 'Vocab' },
  ], defaultSub: 'sources' },
};
const SUB_TO_GROUP = {};
for (const [g, def] of Object.entries(TAB_GROUPS)) {
  if (def.subs.length === 0) SUB_TO_GROUP[g] = g;
  else def.subs.forEach(s => { SUB_TO_GROUP[s.id] = g; });
}

function switchGroup(group) {
  const def = TAB_GROUPS[group];
  if (!def) return;
  document.querySelectorAll('#group-bar button').forEach(b => b.classList.toggle('active', b.dataset.group === group));
  const sub = document.getElementById('subnav-bar');
  if (def.subs.length === 0) {
    sub.classList.add('empty');
    sub.innerHTML = '';
  } else {
    sub.classList.remove('empty');
    sub.innerHTML = def.subs.map(s => `<button data-sub="${s.id}">${escapeHtml(s.label)}</button>`).join('');
    sub.querySelectorAll('button').forEach(b => {
      b.addEventListener('click', () => switchTab(b.dataset.sub));
    });
  }
  switchTab(def.defaultSub);
}

function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  document.querySelectorAll('#subnav-bar button').forEach(b => b.classList.toggle('active', b.dataset.sub === name));
  const parent = SUB_TO_GROUP[name];
  if (parent) {
    document.querySelectorAll('#group-bar button').forEach(b => b.classList.toggle('active', b.dataset.group === parent));
    // Make sure sub-nav is populated for the parent group when called directly.
    const def = TAB_GROUPS[parent];
    const sub = document.getElementById('subnav-bar');
    if (def && def.subs.length > 0 && sub.children.length === 0) {
      sub.classList.remove('empty');
      sub.innerHTML = def.subs.map(s => `<button data-sub="${s.id}">${escapeHtml(s.label)}</button>`).join('');
      sub.querySelectorAll('button').forEach(b => {
        b.addEventListener('click', () => switchTab(b.dataset.sub));
        if (b.dataset.sub === name) b.classList.add('active');
      });
    } else if (def && def.subs.length === 0) {
      sub.classList.add('empty');
      sub.innerHTML = '';
    }
  }
  if (name === 'fractal') {
    if (!mandalaInited) initMandala();
    setTimeout(fitMandala, 30);
  }
  if (name === 'spine') {
    if (!spineNetwork) initSpineNetwork();
  }
  history.replaceState(null, '', '#' + name);
}

document.querySelectorAll('#group-bar button').forEach(btn => {
  btn.addEventListener('click', () => switchGroup(btn.dataset.group));
});

// === Markdown helpers ===
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
function inlineFormat(s) {
  let t = escapeHtml(s);
  t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  t = t.replace(/\*([^*\s][^*]*?[^*\s]|\S)\*/g, '<em>$1</em>');
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  t = t.replace(/\b((?:R|O|N|E|P|PR|OQ|EQ|KT)\d{1,3}(?:-[A-Za-z0-9-]+)?)\b/g, '<span class="ref" onclick="openDrawer(\'$1\')">$1</span>');
  t = t.replace(/\bNOW\b/g, '<span class="ref" onclick="openDrawer(\'NOW\')">NOW</span>');
  return t;
}
function renderMarkdown(text) {
  if (!text) return '';
  const lines = text.split(/\r?\n/);
  const out = []; let inList = false; let para = [];
  function flushPara() { if (para.length) { out.push('<p>' + inlineFormat(para.join(' ')) + '</p>'); para = []; } }
  function closeList() { if (inList) { out.push('</ul>'); inList = false; } }
  for (let raw of lines) {
    const line = raw.replace(/\s+$/, '');
    if (!line.trim()) { flushPara(); closeList(); continue; }
    const h = line.match(/^#{1,3}\s+(.+)$/);
    if (h) { flushPara(); closeList(); out.push('<h3 style="font-family:var(--sans);font-size:11px;text-transform:uppercase;letter-spacing:0.14em;color:var(--text-3);font-weight:600;margin:18px 0 10px;">' + escapeHtml(h[1]) + '</h3>'); continue; }
    const b = line.match(/^\s*[•\*\-]\s+(.+)$/);
    if (b) { flushPara(); if (!inList) { out.push('<ul>'); inList = true; } out.push('<li>' + inlineFormat(b[1]) + '</li>'); continue; }
    closeList(); para.push(line.trim());
  }
  flushPara(); closeList();
  return out.join('\n');
}
function resolveRef(ref) {
  const tok = (ref || '').match(/^((?:R|O|N|E|P|PR|OQ|EQ|KT)\d{1,3}[a-z]?(?:-[A-Za-z0-9-]+)?|NOW)/);
  if (!tok) return null;
  const k = tok[0];
  return GRAPH.nodes.find(x => x.id === k) || GRAPH.nodes.find(x => x.id.startsWith(k + '-')) || null;
}
function summaryOf(n) {
  if (!n || !n.statement) return '';
  const f = n.statement.split(/\n\s*\n/)[0].trim().replace(/\s+/g, ' ');
  return f.length > 200 ? f.slice(0,200) + '…' : f;
}

// === Today (homepage) — minimal Apple-spaced landing ===
function buildToday() {
  const now = new Date();
  const dStr = now.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' });
  document.getElementById('hero-eyebrow').textContent = `Today · ${dStr}`;

  const framePara = (NOW_PANELS.frame || '').split(/\n\s*\n/)[0].trim();
  document.getElementById('hero-lead').innerHTML = inlineFormat(framePara || 'An entry into Alex’s case study.');

  // Today's reading entry: pick a node from the NOW.Frame that anchors the moment.
  // O05 (Chen book) is the live anchor referenced in the Frame; fall back gracefully.
  const anchorIds = ['O05-workplace-conflict', 'O05', 'PR02-one-sitting-manuscript-read', 'NOW'];
  let anchor = null;
  for (const id of anchorIds) {
    anchor = GRAPH.nodes.find(x => x.id === id) || GRAPH.nodes.find(x => x.id.startsWith(id + '-'));
    if (anchor) break;
  }
  if (!anchor) anchor = GRAPH.nodes.find(x => x.id === 'NOW');
  const inv = document.getElementById('t-invitation');
  if (inv && anchor) {
    const summary = summaryOf(anchor);
    inv.innerHTML = `
      <div class="t-inv-label">Today’s reading entry</div>
      <div class="t-inv-title">${escapeHtml(anchor.name)}</div>
      <div class="t-inv-body">${inlineFormat(summary)}</div>
      <a class="t-inv-action" data-open-drawer="${escapeHtml(anchor.id)}">Open this node →</a>
    `;
  }

  // Aside: counts computed from the actual graph.
  const spineCount = GRAPH.nodes.filter(n => n.spine).length;
  const tentativeCount = GRAPH.nodes.filter(n => n.tentative).length;
  const canaryCount = GRAPH.nodes.filter(n => /^R\d+-canary-/.test(n.id)).length;
  const aside = document.getElementById('t-aside');
  if (aside) {
    aside.innerHTML =
      `${spineCount} load-bearing nodes. ${tentativeCount} tentative. ${canaryCount} canaries to watch.`;
  }
}

// === Map (vanilla SVG mandala) ===
let mandalaInited = false;
let mandalaSvg = null;
let viewBox = { x: -900, y: -900, w: 1800, h: 1800 };
const hiddenTypes = new Set();

function initMandala() {
  if (mandalaInited) return;
  mandalaInited = true;
  const wrap = document.getElementById('network');
  const declutter = document.getElementById('declutter').checked;
  const visible = GRAPH.nodes.filter(n => n.type !== 'now' && !(declutter && n.type === 'reference' && (n.backlink_count || 0) < 2));
  const visibleSet = new Set(visible.map(n => n.id));
  const edges = GRAPH.edges.filter(e => visibleSet.has(e.from) && visibleSet.has(e.to));

  let svg = '<svg class="mandala" xmlns="http://www.w3.org/2000/svg" viewBox="-900 -900 1800 1800" preserveAspectRatio="xMidYMid meet">';
  svg += '<g id="mandala-g">';
  for (const e of edges) {
    const a = GRAPH.nodes.find(x => x.id === e.from);
    const b = GRAPH.nodes.find(x => x.id === e.to);
    if (!a || !b || a.x === undefined || b.x === undefined) continue;
    svg += `<line class="edge" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}"></line>`;
  }
  for (const n of visible) {
    if (n.x === undefined) continue;
    const cls = `node ${n.type}${n.tentative ? ' tentative' : ''}${n.id === 'NOW' ? ' now' : ''}`;
    const color = TYPE_COLOR[n.type] || '#888';
    const slug = n.id.split('-').slice(1).join('-') || n.id;
    svg += `<g class="${cls}" data-id="${escapeHtml(n.id)}" transform="translate(${n.x},${n.y})">`;
    svg += `<title>${escapeHtml(n.name)} (${escapeHtml(n.id)})</title>`;
    svg += `<circle r="10" fill="${color}" stroke="#fff" stroke-width="1"></circle>`;
    svg += `<text dy="22" text-anchor="middle">${escapeHtml(slug.length > 18 ? slug.slice(0,18)+'…' : slug)}</text>`;
    svg += `</g>`;
  }
  svg += '</g></svg>';
  wrap.innerHTML = svg;
  mandalaSvg = wrap.querySelector('svg');

  wrap.querySelectorAll('.node').forEach(node => {
    node.addEventListener('click', e => { e.stopPropagation(); openDrawer(node.dataset.id); });
  });

  let panning = false; let panStart = null; let vbStart = null;
  wrap.addEventListener('mousedown', e => {
    if (e.target.closest('.node')) return;
    panning = true; panStart = { x: e.clientX, y: e.clientY }; vbStart = { ...viewBox };
  });
  window.addEventListener('mousemove', e => {
    if (!panning) return;
    const rect = wrap.getBoundingClientRect();
    const dx = (e.clientX - panStart.x) * (viewBox.w / rect.width);
    const dy = (e.clientY - panStart.y) * (viewBox.h / rect.height);
    viewBox.x = vbStart.x - dx;
    viewBox.y = vbStart.y - dy;
    applyViewBox();
  });
  window.addEventListener('mouseup', () => { panning = false; });
  wrap.addEventListener('wheel', e => { e.preventDefault(); zoomBy(e.deltaY < 0 ? 1.1 : 0.9); }, { passive: false });

  document.getElementById('fractal-stats').textContent = `${visible.length} of ${GRAPH.nodes.length} nodes · ${edges.length} edges`;
  fitMandala();
}
function applyViewBox() {
  if (!mandalaSvg) return;
  mandalaSvg.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`);
}
window.zoomBy = function(f) {
  const cx = viewBox.x + viewBox.w / 2;
  const cy = viewBox.y + viewBox.h / 2;
  viewBox.w /= f;
  viewBox.h /= f;
  viewBox.x = cx - viewBox.w / 2;
  viewBox.y = cy - viewBox.h / 2;
  applyViewBox();
};
window.fitMandala = function() {
  viewBox = { x: -900, y: -900, w: 1800, h: 1800 };
  applyViewBox();
};
document.getElementById('declutter').addEventListener('change', () => {
  mandalaInited = false;
  document.getElementById('network').innerHTML = '';
  initMandala();
});
document.getElementById('search').addEventListener('input', e => {
  if (!mandalaInited) return;
  const q = e.target.value.trim().toLowerCase();
  document.querySelectorAll('#network .node').forEach(node => {
    const id = node.dataset.id.toLowerCase();
    const matches = !q || id.includes(q);
    node.classList.toggle('dim', !matches);
  });
});
document.querySelectorAll('#fractal-controls .legend-item').forEach(item => {
  item.addEventListener('click', () => {
    const t = item.dataset.type;
    if (hiddenTypes.has(t)) { hiddenTypes.delete(t); item.classList.remove('dim'); }
    else { hiddenTypes.add(t); item.classList.add('dim'); }
    if (!mandalaInited) return;
    document.querySelectorAll('#network .node').forEach(node => {
      const id = node.dataset.id;
      const n = GRAPH.nodes.find(x => x.id === id);
      if (!n) return;
      node.classList.toggle('dim', hiddenTypes.has(n.type));
    });
  });
});

// === Spine (load-bearing subset, vis-network force layout) ===
let spineNetwork = null;
function initSpineNetwork() {
  if (typeof vis === 'undefined') {
    document.getElementById('network-spine').innerHTML =
      '<div style="padding:64px 32px;text-align:center;color:var(--text-3);">vis-network failed to load. Check your network connection.</div>';
    return;
  }
  const spineNodes = GRAPH.nodes.filter(n => n.spine);
  const spineIds = new Set(spineNodes.map(n => n.id));
  const nodes = spineNodes.map(n => {
    const color = TYPE_COLOR[n.type] || '#888';
    return {
      id: n.id,
      label: n.id + '\n' + (n.name.length > 36 ? n.name.slice(0, 36) + '…' : n.name),
      title: n.name,
      color: { background: '#ffffff', border: color, highlight: { background: '#f5f5f5', border: color } },
      borderWidth: n.tentative ? 2 : 1.5,
      shape: 'box',
      shapeProperties: { borderDashes: n.tentative ? [4, 3] : false },
      widthConstraint: { maximum: 180 },
      margin: { top: 8, right: 10, bottom: 8, left: 10 },
      font: { color: '#1d1d1f', size: 11, face: '"Charter", "Iowan Old Style", "Georgia", ui-serif, serif', align: 'center', multi: false },
    };
  });
  const edges = GRAPH.edges.filter(e => spineIds.has(e.from) && spineIds.has(e.to)).map((e, i) => ({
    id: i, from: e.from, to: e.to,
    color: { color: 'rgba(0,0,0,0.22)', highlight: '#1d1d1f' },
    arrows: { to: { enabled: true, scaleFactor: 0.35 } },
    smooth: { enabled: true, type: 'continuous', roundness: 0.18 },
    width: 0.9,
  }));

  const ds_nodes = new vis.DataSet(nodes);
  const ds_edges = new vis.DataSet(edges);
  spineNetwork = new vis.Network(document.getElementById('network-spine'), { nodes: ds_nodes, edges: ds_edges }, {
    physics: {
      enabled: true,
      solver: 'forceAtlas2Based',
      forceAtlas2Based: { gravitationalConstant: -55, centralGravity: 0.012, springLength: 110, springConstant: 0.08, damping: 0.5 },
      stabilization: { iterations: 350, fit: true },
    },
    interaction: { hover: true, tooltipDelay: 150, dragNodes: true, dragView: true, zoomView: true, keyboard: false },
  });
  spineNetwork.once('stabilizationIterationsDone', () => {
    spineNetwork.setOptions({ physics: { enabled: false } });
    spineNetwork.fit({ animation: { duration: 400 } });
  });
  spineNetwork.on('click', params => { if (params.nodes.length) openDrawer(params.nodes[0]); });
  document.getElementById('spine-stats').textContent = `${nodes.length} of ${GRAPH.nodes.length} nodes · ${edges.length} edges`;
}
function spineZoomBy(f) { if (!spineNetwork) return; const s = spineNetwork.getScale(); spineNetwork.moveTo({ scale: s * f, animation: { duration: 200 } }); }

// === Practices (P-nodes, sorted by P-index then KT) ===
function renderPractices() {
  const list = document.getElementById('practice-list');
  if (!list) return;
  const ps = GRAPH.nodes.filter(n => n.type === 'practice');
  function pSortKey(id) {
    // PR01 → (0, 1); KT01 → (1, 1); P01 (rare) → (0, 1) too
    if (id.startsWith('KT')) {
      const m = id.match(/^KT(\d+)/);
      return [1, m ? parseInt(m[1]) : 0];
    }
    const m = id.match(/^PR?(\d+)/);
    return [0, m ? parseInt(m[1]) : 0];
  }
  ps.sort((a, b) => {
    const ka = pSortKey(a.id), kb = pSortKey(b.id);
    return (ka[0] - kb[0]) || (ka[1] - kb[1]);
  });
  list.innerHTML = '';
  ps.forEach(n => {
    let summary = (n.statement || '').split(/\n\s*\n/)[0].trim().replace(/\s+/g, ' ');
    if (summary.length > 240) summary = summary.slice(0, 240) + '…';
    const row = document.createElement('div');
    row.className = 'row';
    row.onclick = () => openDrawer(n.id);
    row.innerHTML = `
      <div class="rh">
        <span class="rid">${escapeHtml(n.id)}</span>
        <span class="rname">${escapeHtml(n.name)}</span>
      </div>
      <div class="rsum">${escapeHtml(summary)}</div>
    `;
    list.appendChild(row);
  });
  if (!ps.length) {
    list.innerHTML = '<div class="empty-list">No practice nodes found.</div>';
  }
}

// === Canaries (R-prefix nodes whose id-slug starts with "canary-") ===
function renderCanaries() {
  const clist = document.getElementById('canary-list');
  if (!clist) return;
  const cans = GRAPH.nodes.filter(n => /^R\d+-canary-/.test(n.id));
  const clusters = [
    { key: 'routine',    label: 'Routine',    match: /missed-run|crossword/ },
    { key: 'substance',  label: 'Substance',  match: /drinking/ },
    { key: 'mood',       label: 'Mood',       match: /silent-mornings/ },
    { key: 'relational', label: 'Relational', match: /nadia-uncalled/ },
    { key: 'work',       label: 'Work',       match: /political-budget|old-contracts/ },
    { key: 'other',      label: 'Other',      match: /./ },
  ];
  const buckets = clusters.map(c => ({ ...c, items: [] }));
  cans.forEach(n => {
    for (const b of buckets) {
      if (b.match.test(n.id)) { b.items.push(n); break; }
    }
  });
  function rIndex(id) { const m = id.match(/^R(\d+)/); return m ? parseInt(m[1]) : 0; }
  buckets.forEach(b => b.items.sort((a, c) => rIndex(a.id) - rIndex(c.id)));

  clist.innerHTML = '';
  buckets.forEach(b => {
    if (!b.items.length) return;
    const head = document.createElement('div');
    head.className = 'group-head';
    head.textContent = `${b.label} · ${b.items.length}`;
    clist.appendChild(head);
    b.items.forEach(n => {
      let summary = (n.statement || '').split(/\n\s*\n/)[0].trim().replace(/\s+/g, ' ');
      if (summary.length > 280) summary = summary.slice(0, 280) + '…';
      const eli5 = NODE_ELI5[n.id];
      const row = document.createElement('div');
      row.className = 'row';
      row.onclick = () => openDrawer(n.id);
      row.innerHTML = `
        <div class="rh">
          <span class="rid">${escapeHtml(n.id)}</span>
          <span class="rname">${escapeHtml(n.name)}</span>
        </div>
        <div class="rsum">${escapeHtml(summary)}</div>
        ${eli5 ? `<div class="reli5"><span class="velab">In plain English</span>${inlineFormat(eli5)}</div>` : ''}
      `;
      clist.appendChild(row);
    });
  });
  if (!cans.length) {
    clist.innerHTML = '<div class="empty-list">No canary nodes found.</div>';
  }
}

// === Forecasts (Emergent + horizon, ordered by horizon) ===
function renderForecasts() {
  const flist = document.getElementById('forecast-list');
  if (!flist) return;
  const fcs = GRAPH.nodes.filter(n => n.is_forecast);
  function horizonDays(n) {
    const txt = ((n.horizon || '') + ' ' + n.name + ' ' + (n.statement || '')).toLowerCase();
    const m = txt.match(/(\d+)\s*[- ]?\s*(day|week|month|year|yr|mo)/);
    if (!m) return 99999;
    const v = parseInt(m[1]);
    const u = m[2];
    if (u === 'day') return v;
    if (u === 'week') return v * 7;
    if (u === 'month' || u === 'mo') return v * 30;
    if (u === 'year' || u === 'yr') return v * 365;
    return 99999;
  }
  function horizonLabel(n) {
    if (n.horizon) return String(n.horizon);
    const d = horizonDays(n);
    if (d < 60) return d + 'd';
    if (d < 700) return Math.round(d / 30) + 'mo';
    return Math.round(d / 365) + 'y';
  }
  fcs.forEach(n => { n._h = horizonDays(n); });
  fcs.sort((a, b) => a._h - b._h);
  flist.innerHTML = '';
  fcs.forEach(n => {
    let summary = (n.statement || '').split(/\n\s*\n/)[0].trim().replace(/\s+/g, ' ');
    if (summary.length > 280) summary = summary.slice(0, 280) + '…';
    const tent = n.tentative ? '<span class="pill thin">tentative</span>' : '';
    const row = document.createElement('div');
    row.className = 'row';
    row.onclick = () => openDrawer(n.id);
    row.innerHTML = `
      <div class="rh">
        <span class="pill horizon">${escapeHtml(horizonLabel(n))}</span>
        <span class="rid">${escapeHtml(n.id)}</span>
        <span class="rname">${escapeHtml(n.name)}</span>
        ${tent}
      </div>
      <div class="rsum">${escapeHtml(summary)}</div>
    `;
    flist.appendChild(row);
  });
  if (!fcs.length) {
    flist.innerHTML = '<div class="empty-list">No forecast nodes found.</div>';
  }
}

// === Tentative (filter tentative:true, group by type, "needs second instance" when prov_count==1) ===
function renderTentative() {
  const tlist = document.getElementById('tentative-list');
  if (!tlist) return;
  const tentative = GRAPH.nodes.filter(n => n.tentative && n.id !== 'NOW');
  const order = ['emergent', 'overlap', 'novel', 'practice', 'reference', 'observation', 'open'];
  const labels = {
    emergent:    'Emergent — synthesis claims',
    overlap:     'Overlap — patterns across observations',
    novel:       'Novel — interpretations',
    practice:    'Practice — operating rules',
    reference:   'Reference',
    observation: 'Observation',
    open:        'Open question',
  };
  const groups = {};
  tentative.forEach(n => { (groups[n.type] = groups[n.type] || []).push(n); });
  function key(n) { const m = n.id.match(/^[A-Z]+(\d+)/); return m ? parseInt(m[1]) : 0; }
  tlist.innerHTML = '';
  order.forEach(t => {
    if (!groups[t] || !groups[t].length) return;
    const head = document.createElement('div');
    head.className = 'group-head';
    head.textContent = `${labels[t] || t} · ${groups[t].length}`;
    tlist.appendChild(head);
    groups[t].sort((a, b) => key(a) - key(b)).forEach(n => {
      let summary = (n.statement || '').split(/\n\s*\n/)[0].trim().replace(/\s+/g, ' ');
      if (summary.length > 240) summary = summary.slice(0, 240) + '…';
      const pills = [];
      if ((n.prov_count || 0) <= 1) pills.push('<span class="pill thin">needs second instance</span>');
      if (n.is_forecast) pills.push('<span class="pill fc">forecast</span>');
      const row = document.createElement('div');
      row.className = 'row';
      row.onclick = () => openDrawer(n.id);
      row.innerHTML = `
        <div class="rh">
          <span class="rid">${escapeHtml(n.id)}</span>
          <span class="rname">${escapeHtml(n.name)}</span>
          ${pills.join(' ')}
        </div>
        <div class="rsum">${escapeHtml(summary)}</div>
      `;
      tlist.appendChild(row);
    });
  });
  if (!tentative.length) {
    tlist.innerHTML = '<div class="empty-list">No tentative nodes — graph is fully promoted.</div>';
  }
}

// === Eyes (parsed from alex-needs-eyes.md, h2 candidate sections) ===
function renderEyes() {
  const list = document.getElementById('eyes-list');
  if (!list) return;
  if (!EYES.length) {
    list.innerHTML = '<div class="empty-list">No items in needs-eyes right now.</div>';
    return;
  }
  list.innerHTML = '';
  EYES.forEach(e => {
    // Try to resolve a node id from the leading token of the title (e.g. "N01 — ...").
    const m = e.title.match(/^([A-Z]+\d+(?:-[A-Za-z0-9-]+)?)/);
    const node = m ? (GRAPH.nodes.find(x => x.id === m[1]) || GRAPH.nodes.find(x => x.id.startsWith(m[1] + '-'))) : null;
    const row = document.createElement('div');
    row.className = 'row';
    if (node) row.onclick = () => openDrawer(node.id);
    row.innerHTML = `
      <div class="rh">
        ${node ? `<span class="rid">${escapeHtml(node.id)}</span>` : ''}
        <span class="rname">${escapeHtml(e.title)}</span>
      </div>
      <div class="rsum">${inlineFormat(e.body)}</div>
    `;
    list.appendChild(row);
  });
}

// === Drawer (nodes + meta files via openContentDrawer) ===
const DRAWER_HISTORY = [];
const CONTENT_DRAWERS = {};

function openDrawer(id, opts) {
  opts = opts || {};
  const n = GRAPH.nodes.find(x => x.id === id) || GRAPH.nodes.find(x => x.id.startsWith(id + '-'));
  if (!n) return;
  const cur = document.getElementById('drawer').dataset.cur;
  if (cur && cur !== n.id && !opts.noPush) DRAWER_HISTORY.push(cur);
  document.getElementById('drawer').dataset.cur = n.id;

  document.getElementById('drawer-type').innerHTML =
    (n.tentative ? '<span class="tent">tentative</span>' : '') + escapeHtml(TYPE_LABEL[n.type] || n.type);
  document.getElementById('drawer-title').textContent = n.name;
  document.getElementById('drawer-meta').textContent = n.id + (n.horizon ? ' · horizon ' + n.horizon : '');

  let html = '';
  const eli5 = NODE_ELI5[n.id];
  if (eli5) {
    html += '<div class="eli5-block"><div class="eli5-label">In plain English</div>' + renderMarkdown(eli5) + '</div>';
  }
  html += renderMarkdown(n.statement);
  if (n.caveats && n.caveats.trim()) {
    html += '<div class="caveats"><div class="clab">Caveats</div>' + renderMarkdown(n.caveats) + '</div>';
  }

  if (n.attribution || n.evidence || n.derivation) {
    const att = n.attribution || {};
    const ev = n.evidence || {};
    const dv = n.derivation || {};
    let provHtml = '<div class="provenance"><div class="plab">Provenance</div>';
    if (att.source) provHtml += `<div><strong>Attribution:</strong> ${escapeHtml(att.source)}${att.date ? ' · ' + escapeHtml(att.date) : ''}</div>`;
    if (ev.type) provHtml += `<div><strong>Evidence:</strong> ${escapeHtml(ev.type)}${ev.description ? ' — ' + escapeHtml(ev.description) : ''}</div>`;
    if (dv.method) provHtml += `<div><strong>Derivation:</strong> ${escapeHtml(dv.method)}</div>`;
    provHtml += '</div>';
    html += provHtml;
  }

  const groups = [
    { label: 'Related to (bidirectional)', items: n.related_to || [] },
    { label: 'Derives from (parents)', items: n.derives_from || [] },
    { label: 'Evidence references', items: n.evidence_refs || [] },
  ].filter(g => g.items && g.items.length);
  if (groups.length) {
    html += '<div class="related">';
    groups.forEach(g => {
      html += '<div class="relgroup"><div class="rlab">' + escapeHtml(g.label) + '</div>';
      g.items.forEach(item => {
        const t = resolveRef(item);
        if (t) {
          html += `<div class="relnode" data-open-drawer="${escapeHtml(t.id)}">
            <div class="rhead"><span class="rid">${escapeHtml(t.id)}</span><span class="rname">${escapeHtml(t.name||'')}</span></div>
            <div class="rsum">${escapeHtml(summaryOf(t))}</div></div>`;
        } else {
          html += `<div class="relnode" style="cursor:default;border-style:dashed;background:transparent;"><div class="rsum" style="margin-top:0;">${escapeHtml(item)}</div></div>`;
        }
      });
      html += '</div>';
    });
    html += '</div>';
  }

  const mentioned = mentionedBy(n.id);
  if (mentioned.length) {
    html += '<div class="mentioned"><div class="rlab">Mentioned by — ' + mentioned.length + '</div>';
    html += mentioned.map(m => `<span class="ref" data-open-drawer="${escapeHtml(m.id)}" title="${escapeHtml(m.name||'')}">${escapeHtml(m.id)}</span>`).join('');
    html += '</div>';
  }

  document.getElementById('drawer-stmt').innerHTML = html;
  document.getElementById('drawer-back').style.display = DRAWER_HISTORY.length ? 'inline-block' : 'none';
  document.getElementById('drawer').classList.add('open');
  document.getElementById('drawer-overlay').classList.add('open');
  document.querySelector('#drawer .dbody').scrollTop = 0;
}

function openContentDrawer(payload, opts) {
  // payload: { key, title, type, meta, body, eli5, bodyPrefix }
  opts = opts || {};
  const newId = '_content:' + payload.key;
  CONTENT_DRAWERS[payload.key] = payload;

  const cur = document.getElementById('drawer').dataset.cur;
  if (cur && cur !== newId && !opts.noPush) DRAWER_HISTORY.push(cur);
  document.getElementById('drawer').dataset.cur = newId;

  document.getElementById('drawer-type').innerHTML = escapeHtml(payload.type || '');
  document.getElementById('drawer-title').textContent = payload.title || '';
  document.getElementById('drawer-meta').textContent = payload.meta || '';

  let html = '';
  if (payload.bodyPrefix) html += payload.bodyPrefix;
  if (payload.eli5) {
    html += '<div class="eli5-block"><div class="eli5-label">In plain English</div>' + renderMarkdown(payload.eli5) + '</div>';
  }
  html += renderMarkdown(payload.body || '');

  document.getElementById('drawer-stmt').innerHTML = html;
  document.getElementById('drawer-back').style.display = DRAWER_HISTORY.length ? 'inline-block' : 'none';
  document.getElementById('drawer').classList.add('open');
  document.getElementById('drawer-overlay').classList.add('open');
  document.querySelector('#drawer .dbody').scrollTop = 0;
}

function openMetaFile(filename) {
  const def = META_FILES[filename];
  if (!def) {
    // Graceful fallback: try to open the raw file in a new tab.
    window.open(filename, '_blank');
    return;
  }
  openContentDrawer({
    key: 'meta:' + filename,
    title: META_FILE_TITLES[filename] || def.title,
    type: 'source',
    meta: filename,
    body: def.canonical,
  });
}

function drawerBack() {
  if (!DRAWER_HISTORY.length) return;
  const id = DRAWER_HISTORY.pop();
  if (id && id.startsWith('_content:')) {
    const key = id.slice('_content:'.length);
    const payload = CONTENT_DRAWERS[key];
    if (payload) openContentDrawer(payload, { noPush: true });
  } else {
    openDrawer(id, { noPush: true });
  }
}
function closeDrawer() {
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('drawer-overlay').classList.remove('open');
  document.getElementById('drawer').dataset.cur = '';
  DRAWER_HISTORY.length = 0;
}
function mentionedBy(targetId) {
  const shortM = targetId.match(/^[A-Z]+\d+/);
  const needles = new Set([targetId]);
  if (shortM) needles.add(shortM[0]);
  const hits = [];
  for (const n of GRAPH.nodes) {
    if (n.id === targetId) continue;
    const hay = (n.statement || '') + ' ' + (n.name || '');
    for (const needle of needles) {
      const re = new RegExp('\\b' + needle + '\\b');
      if (re.test(hay)) { hits.push(n); break; }
    }
  }
  return hits.slice(0, 30);
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'Escape') closeDrawer();
  else if (e.key === '/' && document.getElementById('tab-fractal').classList.contains('active')) {
    e.preventDefault(); document.getElementById('search').focus();
  }
  else if ((e.key === '+' || e.key === '=')) {
    if (document.getElementById('tab-spine').classList.contains('active')) spineZoomBy(1.25);
    else if (mandalaInited) zoomBy(1.25);
  }
  else if ((e.key === '-' || e.key === '_')) {
    if (document.getElementById('tab-spine').classList.contains('active')) spineZoomBy(0.8);
    else if (mandalaInited) zoomBy(0.8);
  }
  else if (e.key === '0') {
    if (document.getElementById('tab-spine').classList.contains('active') && spineNetwork) spineNetwork.fit({animation:{duration:300}});
    else if (mandalaInited) fitMandala();
  }
});

// === Sources (drawer-based — openMetaFile) ===
const SOURCE_ENTRIES = [
  { name: 'NOW',                 open: () => openDrawer('NOW'),                       desc: 'Top of the stack — frame, this week, standing rules, canaries.' },
  { name: 'Readme',              open: () => openMetaFile('README.md'),               desc: 'Project overview and how to use the scaffold.' },
  { name: 'Start here',          open: () => openMetaFile('START_HERE.md'),           desc: 'Five-minute orientation for first-time readers.' },
  { name: 'Schema',              open: () => openMetaFile('SCHEMA.md'),               desc: 'Formal spec — node types, edges, provenance triple.' },
  { name: 'Safety',              open: () => openMetaFile('SAFETY.md'),               desc: 'Caveats. Read first.' },
  { name: 'Related frameworks',  open: () => openMetaFile('RELATED_FRAMEWORKS.md'),   desc: 'What this borrows from PROV-O, Toulmin, Zettelkasten, PKG.' },
  { name: 'Skill',               open: () => openMetaFile('skill.md'),                desc: 'Skill-card prompt for the scaffold.' },
  { name: 'Vocab',               open: () => openMetaFile('alex-vocab.md'),           desc: 'Glossary source.' },
  { name: 'Open threads',        open: () => openMetaFile('alex-actions.md'),         desc: 'Live-work cards source.' },
  { name: 'Needs eyes',          open: () => openMetaFile('alex-needs-eyes.md'),      desc: 'Items requiring review source.' },
  { name: 'Plain-English overlays', open: () => openMetaFile('alex-node-eli5.md'),    desc: 'ELI5 layer for spine nodes, canaries, practices.' },
];
function renderSources() {
  const sourceList = document.getElementById('source-list');
  if (!sourceList) return;
  sourceList.innerHTML = '';
  SOURCE_ENTRIES.forEach(s => {
    // Skip meta-file entries whose underlying file wasn't loaded (e.g. eli5 not yet authored).
    if (s.open && s.name !== 'NOW') {
      // Heuristic: read the closure's file by matching against META_FILES keys.
      // We can't introspect the function, so just render and let openMetaFile fall back.
    }
    const a = document.createElement('a');
    a.className = 'source-item';
    a.href = '#';
    a.onclick = (e) => { e.preventDefault(); s.open(); };
    a.innerHTML = `<div class="stitle">${escapeHtml(s.name)}</div><div class="sdesc">${escapeHtml(s.desc)}</div>`;
    sourceList.appendChild(a);
  });
}

// === Vocab ===
function renderVocab() {
  const vlist = document.getElementById('vocab-list');
  if (!vlist) return;
  document.getElementById('vocab-count').textContent = `${VOCAB.length} terms in the working vocabulary.`;
  vlist.innerHTML = '';
  VOCAB.forEach(v => {
    const row = document.createElement('div');
    row.className = 'vocab-row';
    row.innerHTML = `<div class="vterm">${escapeHtml(v.term)}</div><div class="vdef">${escapeHtml(v.def)}</div>`;
    vlist.appendChild(row);
  });
}

// === Drawer-open delegation ===
// All elements with [data-open-drawer="<id>"] open the drawer for that
// node id when clicked. Replaces inline onclick="openDrawer('${id}')"
// patterns that would interpolate raw node ids into a JS string literal
// — XSS surface if an id ever contained a quote or a script-end token.
document.addEventListener('click', e => {
  const t = e.target.closest('[data-open-drawer]');
  if (t) {
    e.preventDefault();
    openDrawer(t.getAttribute('data-open-drawer'));
  }
});

// === Boot ===
buildToday();
renderPractices();
renderCanaries();
renderForecasts();
renderTentative();
renderEyes();
renderSources();
renderVocab();

const initialTab = (location.hash || '').replace('#', '');
const VALID_SUBS = ['today','canaries','eyes','tentative','forecasts','spine','practices','fractal','sources','vocab'];
if (VALID_SUBS.includes(initialTab)) {
  const group = SUB_TO_GROUP[initialTab] || 'today';
  switchGroup(group);
  switchTab(initialTab);
} else {
  switchGroup('today');
}
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    if not YAML_PATH.exists():
        sys.exit(f"ERROR: graph file not found: {YAML_PATH}")

    graph = parse_graph(YAML_PATH)
    vocab = parse_vocab(VOCAB_PATH)
    threads = parse_action_cards(ACTIONS_PATH)
    eyes = parse_eyes_candidates(EYES_PATH)
    now_panels = extract_now_panels(graph)
    node_eli5 = parse_node_eli5(ELI5_PATH)
    meta_files = load_meta_files()

    # safe_json escapes "</" and "<!--" so JSON-encoded payloads (which
    # may contain user-supplied statement text or markdown) can't break
    # out of the surrounding <script> block. Without this, a node statement
    # containing the literal "</script>" would terminate the script tag
    # and execute whatever follows. ensure_ascii=False keeps Unicode
    # readable in source view; the escapes only cover the script-context
    # break sequences.
    def safe_json(data):
        return (
            json.dumps(data, ensure_ascii=False)
            .replace("</", "<\\/")
            .replace("<!--", "<\\!--")
        )

    html = (
        HTML_TEMPLATE
        .replace("__FAVICON__",          FAVICON_SVG)
        .replace("__DATA__",             safe_json(graph))
        .replace("__VOCAB__",            safe_json(vocab))
        .replace("__THREADS__",          safe_json(threads))
        .replace("__EYES__",             safe_json(eyes))
        .replace("__NOW_PANELS__",       safe_json(now_panels))
        .replace("__TYPE_COLOR__",       safe_json(TYPE_COLOR))
        .replace("__TYPE_LABEL__",       safe_json(TYPE_LABEL))
        .replace("__NODE_ELI5__",        safe_json(node_eli5))
        .replace("__META_FILES__",       safe_json(meta_files))
        .replace("__META_FILE_TITLES__", safe_json(META_FILE_TITLES))
    )
    OUT_PATH.write_text(html)
    print(f"Wrote {OUT_PATH}")
    print(f"  nodes: {len(graph['nodes'])}  edges: {len(graph['edges'])}")
    print(f"  vocab: {len(vocab)}  threads: {len(threads)}  eyes: {len(eyes)}")
    print(f"  spine: {sum(1 for n in graph['nodes'] if n.get('spine'))} nodes")
    print(f"  forecasts: {sum(1 for n in graph['nodes'] if n.get('is_forecast'))} nodes")
    print(f"  tentative: {sum(1 for n in graph['nodes'] if n.get('tentative'))} nodes")
    print(f"  meta files loaded: {len(meta_files)}")
    if node_eli5:
        print(f"  node-eli5 entries: {len(node_eli5)}")


if __name__ == "__main__":
    main()
