#!/usr/bin/env python3
"""
render.py — Render any memory graph YAML to a graphviz diagram.

Usage:
    python3 render.py path/to/your-graph.yaml

Produces:
    - your-graph-full.png      (all nodes)
    - your-graph-spine.png     (load-bearing subset)
    - your-graph-validation.txt (schema check)

Requires: PyYAML, graphviz (python package + system dot binary).
    pip install pyyaml graphviz
    apt-get install graphviz   # or brew install graphviz on macOS
"""

import sys
import textwrap
from collections import defaultdict, Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("ERROR: install pyyaml — pip install pyyaml")

try:
    import graphviz
except ImportError:
    sys.exit("ERROR: install graphviz python package — pip install graphviz")


# ─────────────────────────────────────────────────────────────────────────
#  Styling — matches the schema
# ─────────────────────────────────────────────────────────────────────────
TYPE_STYLE = {
    'reference':   {'fillcolor': '#ECECEC', 'color': '#666666',
                    'shape': 'box',           'style': 'filled,rounded'},
    'observation': {'fillcolor': '#C8DCF0', 'color': '#3A5E7A',
                    'shape': 'box',           'style': 'filled,rounded'},
    'overlap':     {'fillcolor': '#B8E0C2', 'color': '#2A6A3A',
                    'shape': 'ellipse',       'style': 'filled'},
    'novel':       {'fillcolor': '#F5E0A0', 'color': '#7A5A0E',
                    'shape': 'ellipse',       'style': 'filled,dashed'},
    'emergent':    {'fillcolor': '#E0C8F0', 'color': '#5A3A7A',
                    'shape': 'diamond',       'style': 'filled'},
    'equivalency': {'fillcolor': '#F5C890', 'color': '#7A4A0E',
                    'shape': 'hexagon',       'style': 'filled'},
    'open':        {'fillcolor': '#F5C8C8', 'color': '#7A2A2A',
                    'shape': 'octagon',       'style': 'filled,dashed'},
    'practice':    {'fillcolor': '#D8E0F5', 'color': '#3A4A7A',
                    'shape': 'parallelogram', 'style': 'filled'},
}

EDGE_STYLE = {
    'grounds':       {'color': '#2A6A3A', 'penwidth': '1.2'},
    'grounded_in':   {'color': '#2A6A3A', 'penwidth': '1.2'},
    'derives_from':  {'color': '#2A4A8A', 'penwidth': '1.0'},
    'instantiates':  {'color': '#8A4A2A', 'penwidth': '1.0'},
    'generalizes':   {'color': '#2A4A8A', 'penwidth': '1.0'},
    'qualifies':     {'color': '#6A2A8A', 'penwidth': '1.0', 'style': 'dashed'},
    'emergent_from': {'color': '#6A4A8A', 'penwidth': '1.0', 'style': 'dotted'},
    'contradicts':   {'color': '#8A2A2A', 'penwidth': '1.2'},
}


# ─────────────────────────────────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────────────────────────────────
def validate(nodes):
    """Run schema-spec rules 1-6 + novel/emergent/overlap sanity checks."""
    issues = []
    ids = [n.get('id') for n in nodes]
    id_set = set(ids)

    # Rule 1: unique IDs
    dupes = [nid for nid, c in Counter(ids).items() if c > 1]
    for d in dupes:
        issues.append(f"DUPLICATE ID: {d}")

    for n in nodes:
        nid = n.get('id', '<unknown>')

        # Rule 2: provenance triple complete
        prov = n.get('provenance') or {}
        if not prov.get('attribution'):
            issues.append(f"{nid}: missing provenance.attribution")
        if not prov.get('evidence'):
            issues.append(f"{nid}: missing provenance.evidence")
        if not prov.get('derivation'):
            issues.append(f"{nid}: missing provenance.derivation")

        # Rule 4: derivation.from references exist
        deriv = prov.get('derivation') or {}
        for src in (deriv.get('from') or []):
            if src not in id_set:
                issues.append(f"{nid}: derivation.from references missing node: {src}")

        # Rule 6: evidence.references exist
        refs = (prov.get('evidence') or {}).get('references') or []
        for src in refs:
            if src not in id_set:
                issues.append(f"{nid}: evidence.references points to missing node: {src}")

        # Rule 5: edge targets exist; rule 3 edge provenance
        for edge in (n.get('edges') or []):
            tgt = edge.get('to')
            if tgt and tgt not in id_set:
                issues.append(f"{nid}: edge -> {tgt} points to missing node")
            if not edge.get('provenance'):
                issues.append(f"{nid}: edge -> {tgt} missing provenance")

        # Rule 7: novel must be tentative with caveats
        if n.get('type') == 'novel':
            if not n.get('tentative'):
                issues.append(f"{nid}: novel node missing tentative: true")
            if not (n.get('caveats') or '').strip():
                issues.append(f"{nid}: novel node missing caveats")

        # Rule 8: emergent needs ≥2 parents
        if n.get('type') == 'emergent':
            parents = deriv.get('from') or []
            if len(set(parents)) < 2:
                issues.append(f"{nid}: emergent node has <2 parents in derivation.from")

        # Rule 9: overlap needs ≥2 independent references
        if n.get('type') == 'overlap':
            refs = (prov.get('evidence') or {}).get('references') or []
            if len(set(refs)) < 2:
                issues.append(f"{nid}: overlap node has <2 references in evidence.references")

        # Rule 10: practice needs descriptive grounding in derivation.from
        if n.get('type') == 'practice':
            parents = deriv.get('from') or []
            if not parents:
                issues.append(f"{nid}: practice node has no derivation.from — a floating rule belongs in goals.md, not the graph")

    return issues


# ─────────────────────────────────────────────────────────────────────────
#  Labels
# ─────────────────────────────────────────────────────────────────────────
def node_label(n, max_width=22, max_chars=48):
    short_id = n['id'].split('-', 1)[0]
    name = n.get('name', '')
    if len(name) > max_chars:
        name = name[:max_chars-1] + '…'
    wrapped = '\n'.join(textwrap.wrap(name, width=max_width))
    return f'{short_id}\n{wrapped}'


# ─────────────────────────────────────────────────────────────────────────
#  In-degree (used for spine view)
# ─────────────────────────────────────────────────────────────────────────
def compute_indegree(nodes):
    indeg = defaultdict(int)
    for n in nodes:
        deriv = n.get('provenance', {}).get('derivation', {}) or {}
        for src in (deriv.get('from') or []):
            indeg[src] += 1
        for edge in (n.get('edges') or []):
            if edge.get('to'):
                indeg[edge['to']] += 1
    return indeg


# ─────────────────────────────────────────────────────────────────────────
#  Render: full graph
# ─────────────────────────────────────────────────────────────────────────
def render_full(nodes, out_path_no_ext, title='Memory graph — full'):
    dot = graphviz.Digraph('full', format='png', engine='dot')
    dot.attr(rankdir='BT', splines='spline', overlap='false',
             nodesep='0.25', ranksep='0.6',
             bgcolor='white', fontname='Helvetica',
             label=title, labelloc='t', fontsize='14')
    dot.attr('node', fontname='Helvetica', fontsize='9', margin='0.08,0.04')
    dot.attr('edge', fontname='Helvetica', fontsize='7', color='#888888')

    id_set = {n['id'] for n in nodes}

    for n in nodes:
        style = TYPE_STYLE.get(n['type'], {})
        dot.node(n['id'], label=node_label(n), **style)

    for n in nodes:
        deriv = n.get('provenance', {}).get('derivation', {}) or {}
        for src in (deriv.get('from') or []):
            if src in id_set and src != n['id']:
                dot.edge(src, n['id'], color='#BBBBBB',
                         penwidth='0.5', arrowsize='0.4')
        for edge in (n.get('edges') or []):
            tgt = edge.get('to')
            rel = edge.get('relation', '')
            if not tgt or tgt not in id_set:
                continue
            style = EDGE_STYLE.get(rel, {'color': '#4A4A4A', 'penwidth': '0.7'})
            dot.edge(n['id'], tgt, **style)

    dot.render(out_path_no_ext, cleanup=True)


# ─────────────────────────────────────────────────────────────────────────
#  Render: spine graph (load-bearing subset)
# ─────────────────────────────────────────────────────────────────────────
def render_spine(nodes, out_path_no_ext, title='Memory graph — load-bearing spine'):
    indeg = compute_indegree(nodes)
    top_n = max(5, min(12, len(nodes) // 3))

    spine_ids = set()
    for nid, _ in sorted(indeg.items(), key=lambda x: -x[1])[:top_n]:
        spine_ids.add(nid)
    for n in nodes:
        if n['type'] in ('emergent', 'equivalency', 'open', 'overlap'):
            spine_ids.add(n['id'])

    spine_nodes = [n for n in nodes if n['id'] in spine_ids]

    dot = graphviz.Digraph('spine', format='png', engine='dot')
    dot.attr(rankdir='BT', splines='spline', overlap='false',
             nodesep='0.3', ranksep='0.8',
             bgcolor='white', fontname='Helvetica',
             label=f'{title}\\n{len(spine_nodes)} of {len(nodes)} nodes shown',
             labelloc='t', fontsize='14')
    dot.attr('node', fontname='Helvetica', fontsize='10', margin='0.1,0.05')
    dot.attr('edge', fontname='Helvetica', fontsize='8', color='#888888')

    for n in spine_nodes:
        style = TYPE_STYLE.get(n['type'], {})
        dot.node(n['id'], label=node_label(n, max_width=20, max_chars=44), **style)

    for n in spine_nodes:
        deriv = n.get('provenance', {}).get('derivation', {}) or {}
        for src in (deriv.get('from') or []):
            if src in spine_ids and src != n['id']:
                dot.edge(src, n['id'], color='#AAAAAA',
                         penwidth='0.6', arrowsize='0.5')
        for edge in (n.get('edges') or []):
            tgt = edge.get('to')
            rel = edge.get('relation', '')
            if not tgt or tgt not in spine_ids:
                continue
            style = EDGE_STYLE.get(rel, {'color': '#4A4A4A', 'penwidth': '1.0'})
            dot.edge(n['id'], tgt, label=rel, **style)

    dot.render(out_path_no_ext, cleanup=True)


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 render.py path/to/graph.yaml")

    yaml_path = Path(sys.argv[1]).resolve()
    if not yaml_path.exists():
        sys.exit(f"Not found: {yaml_path}")

    with open(yaml_path) as f:
        nodes = yaml.safe_load(f)
    if not isinstance(nodes, list):
        sys.exit("YAML must be a list of nodes at the top level.")

    stem = yaml_path.stem
    out_dir = yaml_path.parent

    # Validate
    issues = validate(nodes)
    val_path = out_dir / f'{stem}-validation.txt'
    with open(val_path, 'w') as f:
        if issues:
            f.write(f"Found {len(issues)} issue(s):\n\n")
            for i in issues:
                f.write(f"  - {i}\n")
        else:
            f.write("Validation: clean. No issues found.\n")
    print(f"Validation: {val_path} ({len(issues)} issues)")

    if issues:
        print("WARNING: validation issues detected (see file). Rendering anyway.")

    # Render
    full_path = out_dir / f'{stem}-full'
    render_full(nodes, str(full_path), title=f'{stem} — full ({len(nodes)} nodes)')
    print(f"Rendered: {full_path}.png")

    spine_path = out_dir / f'{stem}-spine'
    render_spine(nodes, str(spine_path), title=f'{stem} — spine')
    print(f"Rendered: {spine_path}.png")

    # Summary
    types = Counter(n['type'] for n in nodes)
    print(f"\nNode totals ({len(nodes)} total):")
    for t in ['reference', 'observation', 'overlap', 'novel',
             'emergent', 'equivalency', 'open', 'practice']:
        if types.get(t):
            print(f"  {t}: {types[t]}")

    indeg = compute_indegree(nodes)
    print("\nTop 5 load-bearing nodes (highest in-degree):")
    for nid, c in sorted(indeg.items(), key=lambda x: -x[1])[:5]:
        name = next((n['name'] for n in nodes if n['id'] == nid), '?')
        print(f"  ({c}x)  {nid}  — {name[:60]}")


if __name__ == '__main__':
    main()
