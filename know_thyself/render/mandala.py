#!/usr/bin/env python3
"""
render_mandala.py — alternate renders for a memory graph YAML.

Produces two kinds of image beyond the graphviz views in render.py:

  1. Mandala view — concentric rings by node type (reference, observation,
     overlap, emergent, novel, equivalency, practice, open). The central ring
     holds the most derived kinds; outer rings hold the raw references. Edges
     are drawn as faint curves from source to target (derivation and edges).

  2. Risk-corridor view — given a pivot node id, highlight every node whose
     derivation chain transitively includes the pivot. Everything else is
     faded. Answers the question: "if this one node is miscoded, how much of
     the graph collapses?"

For each view we also emit a 1200x630 "og" crop with a small baked-in caption
that looks reasonable as a LinkedIn / OG preview.

Usage:
    python3 render_mandala.py path/to/graph.yaml
    python3 render_mandala.py path/to/graph.yaml --pivot O01-first-three-months

Requires: PyYAML, matplotlib. Both are pure-Python installs.
    pip install pyyaml matplotlib
"""

import argparse
import math
import sys
from collections import defaultdict, deque
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("ERROR: install pyyaml — pip install pyyaml")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath
except ImportError:
    sys.exit("ERROR: install matplotlib — pip install matplotlib")


# ─────────────────────────────────────────────────────────────────────────
#  Ring layout — center outward. This ordering follows the construction
#  order of the schema: the innermost rings are the most-derived claims
#  (practices and open questions sit closest to the reader, because they
#  are the operational outputs), and the outer rings are the raw inputs
#  (references are the bedrock at the rim).
# ─────────────────────────────────────────────────────────────────────────
RING_ORDER = [
    "practice",
    "open",
    "equivalency",
    "novel",
    "emergent",
    "overlap",
    "observation",
    "reference",
]

# Fill + stroke per node type — deliberately matches the palette used in
# the graphviz render.py so the two views feel like the same graph.
TYPE_STYLE = {
    "reference":   {"fill": "#ECECEC", "edge": "#666666", "marker": "s"},
    "observation": {"fill": "#C8DCF0", "edge": "#3A5E7A", "marker": "s"},
    "overlap":     {"fill": "#B8E0C2", "edge": "#2A6A3A", "marker": "o"},
    "emergent":    {"fill": "#E0C8F0", "edge": "#5A3A7A", "marker": "D"},
    "novel":       {"fill": "#F5E0A0", "edge": "#7A5A0E", "marker": "o"},
    "equivalency": {"fill": "#F5C890", "edge": "#7A4A0E", "marker": "h"},
    "practice":    {"fill": "#D8E0F5", "edge": "#3A4A7A", "marker": "P"},
    "open":        {"fill": "#F5C8C8", "edge": "#7A2A2A", "marker": "8"},
}

# Highlight colors for the risk-corridor view
RISK_PIVOT_FILL = "#E74C3C"
RISK_PIVOT_EDGE = "#8B1A10"
RISK_DOWNSTREAM_FILL = "#F5B75A"
RISK_DOWNSTREAM_EDGE = "#A0661C"
RISK_FADED_FILL = "#EDEDED"
RISK_FADED_EDGE = "#BEBEBE"


# ─────────────────────────────────────────────────────────────────────────
#  YAML + graph shape
# ─────────────────────────────────────────────────────────────────────────
def load_nodes(yaml_path):
    with open(yaml_path, encoding="utf-8") as f:
        nodes = yaml.safe_load(f)
    if not isinstance(nodes, list):
        raise ValueError("YAML must be a top-level list of nodes.")
    return nodes


def derivation_sources(node):
    """Return the list of node ids this node derives from."""
    deriv = (node.get("provenance") or {}).get("derivation") or {}
    return [s for s in (deriv.get("from") or []) if s]


def outgoing_edges(node):
    """(target_id, relation) for each declared edge that has a target."""
    out = []
    for edge in node.get("edges") or []:
        tgt = edge.get("to")
        if tgt:
            out.append((tgt, edge.get("relation", "")))
    return out


def build_reverse_derivation(nodes):
    """pivot -> set of node ids whose derivation transitively includes pivot."""
    # Forward: child -> parents. Reverse: parent -> children.
    children_of = defaultdict(set)
    for n in nodes:
        for src in derivation_sources(n):
            children_of[src].add(n["id"])
    return children_of


def transitive_descendants(pivot_id, children_of):
    """BFS every descendant reachable via derivation edges."""
    seen = set()
    q = deque([pivot_id])
    while q:
        cur = q.popleft()
        for ch in children_of.get(cur, ()):
            if ch not in seen:
                seen.add(ch)
                q.append(ch)
    return seen


# ─────────────────────────────────────────────────────────────────────────
#  Mandala layout
# ─────────────────────────────────────────────────────────────────────────
def assign_ring_positions(nodes):
    """
    Assign (x, y) to each node on a concentric ring based on its type.
    Returns a dict node_id -> (x, y, radius, ring_index).
    Rings are spaced evenly starting at r = 1.0 for the innermost present ring.
    """
    buckets = defaultdict(list)
    for n in nodes:
        buckets[n.get("type", "")].append(n)

    # Sort each bucket stably by id so parallel edits to the yaml are stable.
    for t in buckets:
        buckets[t].sort(key=lambda n: n.get("id", ""))

    # Compute radii: only assign rings to types that actually have nodes, so
    # the figure doesn't have empty gap-rings when a type is missing.
    present_rings = [t for t in RING_ORDER if buckets.get(t)]
    base_r = 1.0
    step = 1.0
    radii = {t: base_r + step * i for i, t in enumerate(present_rings)}

    positions = {}
    for ring_idx, t in enumerate(present_rings):
        ring_nodes = buckets[t]
        r = radii[t]
        count = len(ring_nodes)
        if count == 1:
            # Single node on a ring: place at top to keep geometry obvious.
            positions[ring_nodes[0]["id"]] = (0.0, -r, r, ring_idx)
            continue
        for i, n in enumerate(ring_nodes):
            # Start at top (-pi/2), sweep clockwise so the eye reads top-first.
            angle = -math.pi / 2 + 2 * math.pi * i / count
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            positions[n["id"]] = (x, y, r, ring_idx)

    return positions, radii


# ─────────────────────────────────────────────────────────────────────────
#  Drawing primitives
# ─────────────────────────────────────────────────────────────────────────
def _curved_path(x0, y0, x1, y1, curvature=0.18):
    """
    Build a quadratic-bezier Path from (x0,y0) to (x1,y1), bulging toward
    the origin by `curvature`. This makes inter-ring edges read as gentle
    arcs rather than straight chords, which is the mandala look.
    """
    mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    # Pull the midpoint toward origin a bit — the closer to the center, the
    # less bulge we want so inner-ring edges stay readable.
    cx = mx * (1.0 - curvature)
    cy = my * (1.0 - curvature)
    verts = [(x0, y0), (cx, cy), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    return MplPath(verts, codes)


def _draw_rings(ax, radii):
    for t, r in radii.items():
        ring = mpatches.Circle(
            (0, 0), r,
            fill=False, edgecolor="#E2E2E2", linewidth=0.6,
            linestyle=(0, (3, 4)), zorder=0,
        )
        ax.add_patch(ring)


def _draw_nodes(ax, nodes, positions, style_for):
    """Draw every positioned node using style_for(node) -> dict."""
    for n in nodes:
        nid = n["id"]
        if nid not in positions:
            continue
        x, y, _, _ = positions[nid]
        s = style_for(n)
        marker = TYPE_STYLE.get(n.get("type", ""), {}).get("marker", "o")
        ax.scatter(
            [x], [y],
            s=s.get("size", 180),
            c=[s["fill"]],
            edgecolors=[s["edge"]],
            linewidths=s.get("linewidth", 0.8),
            marker=marker,
            zorder=s.get("zorder", 3),
            alpha=s.get("alpha", 1.0),
        )
        label = s.get("label")
        if label:
            ax.text(
                x, y - 0.08, label,
                ha="center", va="top",
                fontsize=s.get("fontsize", 6.5),
                color=s.get("label_color", "#2A2A2A"),
                zorder=s.get("zorder", 3) + 1,
                alpha=s.get("alpha", 1.0),
            )


def _draw_edges(ax, nodes, positions, style_for_edge):
    id_set = {n["id"] for n in nodes}
    for n in nodes:
        src = n["id"]
        if src not in positions:
            continue
        x0, y0, _, _ = positions[src]
        # Derivation edges
        for parent in derivation_sources(n):
            if parent not in id_set or parent == src or parent not in positions:
                continue
            x1, y1, _, _ = positions[parent]
            style = style_for_edge(n, parent, relation="derives_from")
            if style is None:
                continue
            patch = mpatches.PathPatch(
                _curved_path(x0, y0, x1, y1, curvature=style.get("curvature", 0.18)),
                facecolor="none",
                edgecolor=style["color"],
                linewidth=style["linewidth"],
                alpha=style.get("alpha", 0.6),
                zorder=style.get("zorder", 1),
            )
            ax.add_patch(patch)
        # Declared edges
        for tgt, relation in outgoing_edges(n):
            if tgt not in id_set or tgt == src or tgt not in positions:
                continue
            x1, y1, _, _ = positions[tgt]
            style = style_for_edge(n, tgt, relation=relation)
            if style is None:
                continue
            patch = mpatches.PathPatch(
                _curved_path(x0, y0, x1, y1, curvature=style.get("curvature", 0.18)),
                facecolor="none",
                edgecolor=style["color"],
                linewidth=style["linewidth"],
                alpha=style.get("alpha", 0.6),
                zorder=style.get("zorder", 1),
            )
            ax.add_patch(patch)


def _short_label(nid):
    """The id prefix before the first hyphen, e.g. O01-foo -> O01."""
    return nid.split("-", 1)[0]


def _setup_square_axes(ax, outer_radius):
    pad = 0.25
    lim = outer_radius + pad
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")


def _legend_patches():
    patches = []
    for t in RING_ORDER:
        if t not in TYPE_STYLE:
            continue
        s = TYPE_STYLE[t]
        patches.append(mpatches.Patch(
            facecolor=s["fill"], edgecolor=s["edge"], label=t,
        ))
    return patches


# ─────────────────────────────────────────────────────────────────────────
#  Mandala render
# ─────────────────────────────────────────────────────────────────────────
def render_mandala(nodes, out_png, out_svg=None, title=None, caption=None,
                   figsize=(10, 10), dpi=160, og=False):
    positions, radii = assign_ring_positions(nodes)
    outer_radius = max(radii.values()) if radii else 1.0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")

    _draw_rings(ax, radii)

    def node_style(n):
        s = TYPE_STYLE.get(n.get("type", ""), TYPE_STYLE["reference"])
        return {
            "fill": s["fill"],
            "edge": s["edge"],
            "size": 220 if og else 260,
            "linewidth": 0.9,
            "label": _short_label(n["id"]) if not og else None,
            "fontsize": 6.0,
            "zorder": 3,
        }

    def edge_style(src_node, dst_id, relation):
        # Derivation edges get a blue; declared edges get a muted gray so the
        # eye follows the derivation structure primarily.
        if relation == "derives_from":
            return {"color": "#8FA8C9", "linewidth": 0.6, "alpha": 0.55,
                    "zorder": 1, "curvature": 0.2}
        return {"color": "#CFCFCF", "linewidth": 0.5, "alpha": 0.5,
                "zorder": 1, "curvature": 0.22}

    _draw_edges(ax, nodes, positions, edge_style)
    _draw_nodes(ax, nodes, positions, node_style)

    _setup_square_axes(ax, outer_radius)

    if title and not og:
        fig.suptitle(title, fontsize=13, color="#1A1A1A", y=0.97)

    # Ring label: print type name just outside each ring at the left, once.
    if not og:
        for t, r in radii.items():
            ax.text(
                -r - 0.02, 0, t,
                ha="right", va="center",
                fontsize=7, color="#8A8A8A",
                rotation=0,
            )

    if caption:
        # Small caption centered near the bottom of the figure.
        fig.text(
            0.5, 0.02 if og else 0.015,
            caption,
            ha="center", va="bottom",
            fontsize=9 if og else 9,
            color="#4A4A4A",
        )

    if not og:
        # Inline legend in the corner
        ax.legend(
            handles=_legend_patches(),
            loc="lower left", frameon=False, fontsize=7,
            bbox_to_anchor=(-0.02, -0.02),
        )

    fig.tight_layout(rect=(0, 0.03, 1, 0.96 if title and not og else 1.0))
    fig.savefig(out_png, facecolor="white", dpi=dpi)
    if out_svg:
        fig.savefig(out_svg, facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
#  Risk-corridor render
# ─────────────────────────────────────────────────────────────────────────
def render_risk_corridor(nodes, pivot_id, out_png, out_svg=None,
                         title=None, caption=None, figsize=(10, 10),
                         dpi=160, og=False):
    id_set = {n["id"] for n in nodes}
    if pivot_id not in id_set:
        # Fall back to any node whose short id matches — tolerates yaml edits
        # that rename the suffix after the dash.
        short = pivot_id.split("-", 1)[0]
        match = next((n["id"] for n in nodes
                      if n["id"].split("-", 1)[0] == short), None)
        if match is None:
            raise ValueError(
                f"pivot node {pivot_id!r} not found in graph "
                f"(and no node id starts with {short!r})"
            )
        pivot_id = match

    children_of = build_reverse_derivation(nodes)
    downstream = transitive_descendants(pivot_id, children_of)

    positions, radii = assign_ring_positions(nodes)
    outer_radius = max(radii.values()) if radii else 1.0

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    _draw_rings(ax, radii)

    def node_style(n):
        nid = n["id"]
        if nid == pivot_id:
            return {
                "fill": RISK_PIVOT_FILL,
                "edge": RISK_PIVOT_EDGE,
                "size": 520 if not og else 400,
                "linewidth": 1.6,
                "label": _short_label(nid) if not og else None,
                "fontsize": 8.0,
                "label_color": "#1A1A1A",
                "zorder": 5,
            }
        if nid in downstream:
            return {
                "fill": RISK_DOWNSTREAM_FILL,
                "edge": RISK_DOWNSTREAM_EDGE,
                "size": 320 if not og else 260,
                "linewidth": 1.1,
                "label": _short_label(nid) if not og else None,
                "fontsize": 6.5,
                "label_color": "#2A2A2A",
                "zorder": 4,
            }
        return {
            "fill": RISK_FADED_FILL,
            "edge": RISK_FADED_EDGE,
            "size": 180 if not og else 150,
            "linewidth": 0.5,
            "label": None,
            "fontsize": 5.5,
            "alpha": 0.55,
            "zorder": 2,
        }

    def edge_style(src_node, dst_id, relation):
        src_id = src_node["id"]
        # Emphasize edges that are part of the risk corridor: either the pivot
        # itself or two downstream nodes, or pivot->direct descendant.
        in_corridor = (
            (src_id == pivot_id or src_id in downstream)
            and (dst_id == pivot_id or dst_id in downstream)
        )
        if in_corridor and relation == "derives_from":
            return {"color": RISK_DOWNSTREAM_EDGE, "linewidth": 1.1,
                    "alpha": 0.85, "zorder": 3, "curvature": 0.2}
        if in_corridor:
            return {"color": RISK_DOWNSTREAM_EDGE, "linewidth": 0.9,
                    "alpha": 0.8, "zorder": 3, "curvature": 0.22}
        # Faded context edges
        return {"color": "#E5E5E5", "linewidth": 0.4, "alpha": 0.5,
                "zorder": 1, "curvature": 0.22}

    _draw_edges(ax, nodes, positions, edge_style)
    _draw_nodes(ax, nodes, positions, node_style)
    _setup_square_axes(ax, outer_radius)

    if title and not og:
        fig.suptitle(title, fontsize=13, color="#1A1A1A", y=0.97)

    if not og:
        # Short corner legend specific to this mode
        legend_patches = [
            mpatches.Patch(facecolor=RISK_PIVOT_FILL,
                           edgecolor=RISK_PIVOT_EDGE, label="pivot (if miscoded)"),
            mpatches.Patch(facecolor=RISK_DOWNSTREAM_FILL,
                           edgecolor=RISK_DOWNSTREAM_EDGE,
                           label=f"downstream — {len(downstream)} nodes"),
            mpatches.Patch(facecolor=RISK_FADED_FILL,
                           edgecolor=RISK_FADED_EDGE, label="unaffected"),
        ]
        ax.legend(
            handles=legend_patches,
            loc="lower left", frameon=False, fontsize=8,
            bbox_to_anchor=(-0.02, -0.02),
        )

    if caption:
        fig.text(
            0.5, 0.02 if og else 0.015,
            caption,
            ha="center", va="bottom",
            fontsize=9,
            color="#4A4A4A",
        )

    fig.tight_layout(rect=(0, 0.03, 1, 0.96 if title and not og else 1.0))
    fig.savefig(out_png, facecolor="white", dpi=dpi)
    if out_svg:
        fig.savefig(out_svg, facecolor="white")
    plt.close(fig)

    return {"pivot": pivot_id, "downstream_count": len(downstream)}


# ─────────────────────────────────────────────────────────────────────────
#  OG sizing helpers
# ─────────────────────────────────────────────────────────────────────────
def og_figsize(width_px=1200, height_px=630, dpi=150):
    return (width_px / dpi, height_px / dpi), dpi


def risk_corridor_og(nodes, pivot_id, out_png, caption):
    """1200x630 risk-corridor crop for LinkedIn previews."""
    fs, dpi = og_figsize()
    return render_risk_corridor(
        nodes, pivot_id, out_png,
        title=None, caption=caption,
        figsize=fs, dpi=dpi, og=True,
    )


def mandala_og(nodes, out_png, caption):
    """1200x630 mandala crop for LinkedIn previews."""
    fs, dpi = og_figsize()
    render_mandala(
        nodes, out_png,
        title=None, caption=caption,
        figsize=fs, dpi=dpi, og=True,
    )


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("yaml_path", help="path to graph YAML")
    parser.add_argument(
        "--pivot", default="O01-first-three-months",
        help="pivot node id for the risk-corridor render "
             "(default: O01-first-three-months)",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml_path).resolve()
    if not yaml_path.exists():
        sys.exit(f"Not found: {yaml_path}")

    nodes = load_nodes(yaml_path)
    stem = yaml_path.stem
    out_dir = yaml_path.parent

    # 1. Mandala full
    mandala_png = out_dir / f"{stem}-mandala.png"
    mandala_svg = out_dir / f"{stem}-mandala.svg"
    render_mandala(
        nodes,
        out_png=str(mandala_png),
        out_svg=str(mandala_svg),
        title=f"{stem} — mandala ({len(nodes)} nodes)",
        caption=None,
    )
    print(f"Rendered: {mandala_png}")

    # 2. Mandala OG
    mandala_og_png = out_dir / f"{stem}-mandala-og.png"
    mandala_og(
        nodes,
        out_png=str(mandala_og_png),
        caption=f"Alex's graph — {len(nodes)} nodes, typed and provenance-tagged",
    )
    print(f"Rendered: {mandala_og_png}")

    # 3. Risk corridor
    risk_png = out_dir / f"{stem}-risk-corridor.png"
    risk_svg = out_dir / f"{stem}-risk-corridor.svg"
    # Desired 1600x900 for the non-og render
    risk_figsize = (1600 / 160, 900 / 160)
    info = render_risk_corridor(
        nodes, args.pivot,
        out_png=str(risk_png),
        out_svg=str(risk_svg),
        title=(f"Risk corridor — if {args.pivot.split('-', 1)[0]} is miscoded, "
               f"downstream derivations need re-checking"),
        caption=None,
        figsize=risk_figsize,
        dpi=160,
    )
    pivot_short = info["pivot"].split("-", 1)[0]
    # Bake the downstream count into the caption for the main image too, by
    # re-rendering with caption (simpler than post-processing). Small cost.
    render_risk_corridor(
        nodes, args.pivot,
        out_png=str(risk_png),
        out_svg=str(risk_svg),
        title=(f"Risk corridor — if {pivot_short} is miscoded, "
               f"{info['downstream_count']} downstream nodes "
               f"need re-derivation"),
        caption=None,
        figsize=risk_figsize,
        dpi=160,
    )
    print(f"Rendered: {risk_png}  (pivot={info['pivot']}, "
          f"downstream={info['downstream_count']})")

    # 4. Risk corridor OG
    risk_og_png = out_dir / f"{stem}-risk-corridor-og.png"
    risk_corridor_og(
        nodes, args.pivot,
        out_png=str(risk_og_png),
        caption=("Risk corridor: if one observation is miscoded, "
                 "these downstream claims collapse."),
    )
    print(f"Rendered: {risk_og_png}")


if __name__ == "__main__":
    main()
