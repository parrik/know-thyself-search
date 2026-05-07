#!/usr/bin/env python3
"""
printable.py — Generate a multi-page printable PDF from a memory graph YAML.

Usage:
    python3 printable.py path/to/your-graph.yaml

Produces:
    your-graph-printable.pdf  (4 pages: cover + principles + spine + full)

Requires: PyYAML, graphviz, reportlab, pypdf.
    pip install pyyaml graphviz reportlab pypdf
    apt-get install graphviz  # or brew install graphviz on macOS
"""

import sys
import textwrap
from collections import defaultdict, Counter
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

try:
    import yaml
except ImportError:
    sys.exit("ERROR: install pyyaml — pip install pyyaml")

try:
    import graphviz
except ImportError:
    sys.exit("ERROR: install graphviz python package — pip install graphviz")

try:
    from pypdf import PdfReader, PdfWriter
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    )
    from reportlab.lib.enums import TA_LEFT
except ImportError:
    sys.exit("ERROR: install reportlab and pypdf — pip install reportlab pypdf")


TYPE_STYLE = {
    'reference':   {'fillcolor': '#ECECEC', 'color': '#666666', 'shape': 'box',     'style': 'filled,rounded'},
    'observation': {'fillcolor': '#C8DCF0', 'color': '#3A5E7A', 'shape': 'box',     'style': 'filled,rounded'},
    'overlap':     {'fillcolor': '#B8E0C2', 'color': '#2A6A3A', 'shape': 'ellipse', 'style': 'filled'},
    'novel':       {'fillcolor': '#F5E0A0', 'color': '#7A5A0E', 'shape': 'ellipse', 'style': 'filled,dashed'},
    'emergent':    {'fillcolor': '#E0C8F0', 'color': '#5A3A7A', 'shape': 'diamond', 'style': 'filled'},
    'equivalency': {'fillcolor': '#F5C890', 'color': '#7A4A0E', 'shape': 'hexagon', 'style': 'filled'},
    'practice':    {'fillcolor': '#F0D8A0', 'color': '#7A4A2A', 'shape': 'parallelogram', 'style': 'filled'},
    'open':        {'fillcolor': '#F5C8C8', 'color': '#7A2A2A', 'shape': 'octagon', 'style': 'filled,dashed'},
}

EDGE_STYLE = {
    'grounds':       {'color': '#2A6A3A', 'penwidth': '1.2'},
    'grounded_in':   {'color': '#2A6A3A', 'penwidth': '1.2'},
    'derives_from':  {'color': '#2A4A8A', 'penwidth': '1.0'},
    'instantiates':  {'color': '#8A4A2A', 'penwidth': '1.0'},
    'generalizes':   {'color': '#4A6A8A', 'penwidth': '1.0'},
    'qualifies':     {'color': '#6A2A8A', 'penwidth': '1.0', 'style': 'dashed'},
    'emergent_from': {'color': '#6A4A8A', 'penwidth': '1.0', 'style': 'dotted'},
    'contradicts':   {'color': '#8A2A2A', 'penwidth': '1.2'},
}


def node_label(n, max_width=22, max_chars=48):
    short_id = n['id'].split('-', 1)[0]
    name = n.get('name', '')
    if len(name) > max_chars:
        name = name[:max_chars-1] + '…'
    wrapped = '\n'.join(textwrap.wrap(name, width=max_width))
    return f'{short_id}\n{wrapped}'


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


def render_spine_pdf(nodes, out_no_ext, title):
    indeg = compute_indegree(nodes)
    top_n = max(5, min(12, len(nodes) // 3))
    spine_ids = set()
    for nid, _ in sorted(indeg.items(), key=lambda x: -x[1])[:top_n]:
        spine_ids.add(nid)
    for n in nodes:
        if n['type'] in ('emergent', 'equivalency', 'open', 'overlap'):
            spine_ids.add(n['id'])
    spine = [n for n in nodes if n['id'] in spine_ids]

    dot = graphviz.Digraph('spine', format='pdf', engine='dot')
    dot.attr(rankdir='BT', splines='spline', overlap='false',
             nodesep='0.25', ranksep='0.7', size='10.5,8', ratio='fill',
             bgcolor='white', fontname='Helvetica',
             label=f'{title} — spine ({len(spine)} of {len(nodes)} nodes)',
             labelloc='t', fontsize='14')
    dot.attr('node', fontname='Helvetica', fontsize='9', margin='0.1,0.05')
    dot.attr('edge', fontname='Helvetica', fontsize='7', color='#666666')

    for n in spine:
        dot.node(n['id'], label=node_label(n, 20, 44),
                 **TYPE_STYLE.get(n['type'], {}))

    for n in spine:
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
            dot.edge(n['id'], tgt, label=rel,
                     **EDGE_STYLE.get(rel, {'color': '#4A4A4A', 'penwidth': '0.8'}))

    dot.render(out_no_ext, cleanup=True)


def render_full_pdf(nodes, out_no_ext, title):
    dot = graphviz.Digraph('full', format='pdf', engine='dot')
    dot.attr(rankdir='BT', splines='spline', overlap='false',
             nodesep='0.15', ranksep='0.5', size='10.5,8', ratio='fill',
             bgcolor='white', fontname='Helvetica',
             label=f'{title} — full ({len(nodes)} nodes)',
             labelloc='t', fontsize='14')
    dot.attr('node', fontname='Helvetica', fontsize='7', margin='0.06,0.03')
    dot.attr('edge', fontname='Helvetica', fontsize='5', color='#888888')

    id_set = {n['id'] for n in nodes}
    for n in nodes:
        dot.node(n['id'], label=node_label(n, 18, 42),
                 **TYPE_STYLE.get(n['type'], {}))
    for n in nodes:
        deriv = n.get('provenance', {}).get('derivation', {}) or {}
        for src in (deriv.get('from') or []):
            if src in id_set and src != n['id']:
                dot.edge(src, n['id'], color='#BBBBBB',
                         penwidth='0.4', arrowsize='0.4')
        for edge in (n.get('edges') or []):
            tgt = edge.get('to')
            if not tgt or tgt not in id_set:
                continue
            dot.edge(n['id'], tgt,
                     **EDGE_STYLE.get(edge.get('relation', ''),
                                      {'color': '#4A4A4A', 'penwidth': '0.6'}))

    dot.render(out_no_ext, cleanup=True)


def render_cover_pdf(nodes, out_path, title):
    indeg = compute_indegree(nodes)
    type_counts = Counter(n['type'] for n in nodes)
    node_by_id = {n['id']: n for n in nodes}

    doc = SimpleDocTemplate(out_path, pagesize=letter,
                            leftMargin=0.6*inch, rightMargin=0.6*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()

    title_s = ParagraphStyle('T', parent=styles['Title'], fontSize=22,
                              spaceAfter=6, alignment=TA_LEFT,
                              textColor=HexColor('#222'))
    subtitle_s = ParagraphStyle('Sub', parent=styles['Normal'], fontSize=10,
                                 textColor=HexColor('#666'), spaceAfter=14)
    h2_s = ParagraphStyle('H2', parent=styles['Heading2'], fontSize=13,
                           spaceBefore=10, spaceAfter=4,
                           textColor=HexColor('#333'))
    body_s = ParagraphStyle('B', parent=styles['Normal'], fontSize=9.5,
                             leading=13, spaceAfter=6)
    cell_s = ParagraphStyle('C', parent=body_s, fontSize=7.5,
                             leading=10, spaceAfter=0)
    header_s = ParagraphStyle('TH', parent=body_s, fontName='Helvetica-Bold',
                               fontSize=8.5, textColor=HexColor('#222'))

    story = []
    story.append(Paragraph(title, title_s))
    story.append(Paragraph(
        f"{len(nodes)} nodes — schema adapted from Pat McCarthy's "
        "open-knowledge-graph", subtitle_s))

    # Legend
    story.append(Paragraph("Legend", h2_s))
    legend_data = [
        ['Type', 'Shape', 'Meaning', 'Confidence basis', 'Count'],
        ['reference',   'rounded box (gray)',   'biographical fact',                  'single-source, verifiable', type_counts.get('reference', 0)],
        ['observation', 'rounded box (blue)',   'specific episode witnessed',         'direct event',              type_counts.get('observation', 0)],
        ['overlap',     'ellipse (green)',      'pattern across ≥2 episodes',         'multiple groundings',       type_counts.get('overlap', 0)],
        ['novel',       'ellipse dashed',       'single-derivation interpretation',   'tentative — flagged',       type_counts.get('novel', 0)],
        ['emergent',    'diamond (purple)',     'produced by intersection',           'intersection-derived',      type_counts.get('emergent', 0)],
        ['equivalency', 'hexagon (orange)',     'external-theory bridge',             'formal grounding',          type_counts.get('equivalency', 0)],
        ['open',        'octagon dashed (red)', 'unresolved question',                'N/A',                       type_counts.get('open', 0)],
    ]
    legend = Table(legend_data, colWidths=[1.0*inch, 1.5*inch, 2.2*inch, 1.7*inch, 0.6*inch])
    legend.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), 'Helvetica', 8.5),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#E8E8E8')),
        ('BACKGROUND', (0, 1), (-1, 1), HexColor('#ECECEC')),
        ('BACKGROUND', (0, 2), (-1, 2), HexColor('#C8DCF0')),
        ('BACKGROUND', (0, 3), (-1, 3), HexColor('#B8E0C2')),
        ('BACKGROUND', (0, 4), (-1, 4), HexColor('#F5E0A0')),
        ('BACKGROUND', (0, 5), (-1, 5), HexColor('#E0C8F0')),
        ('BACKGROUND', (0, 6), (-1, 6), HexColor('#F5C890')),
        ('BACKGROUND', (0, 7), (-1, 7), HexColor('#F5C8C8')),
        ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, HexColor('#AAA')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, HexColor('#CCC')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(legend)
    story.append(Spacer(1, 10))

    # Load-bearing spine
    story.append(Paragraph("Load-bearing spine", h2_s))
    story.append(Paragraph(
        "Nodes with highest in-degree. If any of these are corrected, "
        "many downstream interpretations have to update.",
        ParagraphStyle('I', parent=body_s, fontName='Helvetica-Oblique',
                       textColor=HexColor('#444'))))

    lb_rows = [[Paragraph('#', header_s),
                Paragraph('Node', header_s),
                Paragraph('Type', header_s)]]
    for rank, (nid, count) in enumerate(sorted(indeg.items(),
                                                key=lambda x: -x[1])[:10], 1):
        n = node_by_id.get(nid)
        if not n:
            continue
        short_id = nid.split('-', 1)[0]
        # xml_escape — node names can contain '&', '<' which would break
        # reportlab's mini-XML parser and abort the PDF build.
        name = xml_escape(n['name'])
        lb_rows.append([
            Paragraph(f'{rank}<br/><font size=6 color="#888">({count})</font>', cell_s),
            Paragraph(f'<b>{short_id}</b>: {name}', cell_s),
            Paragraph(xml_escape(n['type']), cell_s),
        ])
    lb_table = Table(lb_rows, colWidths=[0.45*inch, 5.1*inch, 1.45*inch])
    lb_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#E8E8E8')),
        ('BOX', (0, 0), (-1, -1), 0.5, HexColor('#AAA')),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, HexColor('#CCC')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(lb_table)
    story.append(Spacer(1, 10))

    # How to read
    story.append(Paragraph("How to read the diagram", h2_s))
    story.append(Paragraph(
        "<b>Invariant:</b> every node carries (attribution, evidence, derivation). "
        "A claim without provenance is indistinguishable from noise. "
        "<b>Attribution ≠ confidence</b> — repetition of a claim is NOT confirmation; "
        "multiple independent derivations are.", body_s))
    story.append(Paragraph(
        "<b>Flow direction:</b> arrows go bottom-up. Facts and episodes at the "
        "bottom; patterns above; novel and emergent interpretations above those. "
        "Follow an edge from child to parent to see what a claim rests on.", body_s))
    story.append(Paragraph(
        "<b>Tentative claims:</b> dashed outlines mark novel interpretations and "
        "open questions. Do not propagate these without re-examination.", body_s))

    doc.build(story)


def merge_pdfs(paths, out_path):
    w = PdfWriter()
    for p in paths:
        r = PdfReader(p)
        for page in r.pages:
            w.add_page(page)
    with open(out_path, 'wb') as f:
        w.write(f)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 printable.py path/to/graph.yaml")

    yaml_path = Path(sys.argv[1]).resolve()
    if not yaml_path.exists():
        sys.exit(f"Not found: {yaml_path}")

    with open(yaml_path) as f:
        nodes = yaml.safe_load(f)

    stem = yaml_path.stem
    out_dir = yaml_path.parent
    title = stem.replace('-', ' ').replace('_', ' ').title()

    cover = str(out_dir / f'{stem}-cover.pdf')
    spine = str(out_dir / f'{stem}-spine')
    full = str(out_dir / f'{stem}-full')
    final = str(out_dir / f'{stem}-printable.pdf')

    print("Rendering cover page...")
    render_cover_pdf(nodes, cover, title)
    print("Rendering spine page...")
    render_spine_pdf(nodes, spine, title)
    print("Rendering full page...")
    render_full_pdf(nodes, full, title)
    print("Merging...")
    merge_pdfs([cover, spine + '.pdf', full + '.pdf'], final)

    for p in [cover, spine + '.pdf', full + '.pdf']:
        try:
            Path(p).unlink()
        except FileNotFoundError:
            pass

    print(f"Done: {final}")


if __name__ == '__main__':
    main()
