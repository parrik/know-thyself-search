"""Renderers for a Know-Thyself graph YAML.

Modules:

  graphviz  — static Graphviz diagram (``render.py`` historically).
  dashboard — single-file interactive HTML dashboard with the schema
              spine, mandala, today panel, vocab and case-study tabs.
  mandala   — concentric-rings layout (PNG/SVG via matplotlib).
  printable — multi-page PDF combining cover, principles, spine,
              and full graph (reportlab + graphviz + pypdf).
"""
