#!/usr/bin/env python3
"""Create a documented schematic of the CRC cell-type hierarchy.

This is intentionally generated as SVG without plotting dependencies.  The
right-hand labels are the exact values in CRC_cells.csv.  The left-hand side
summarizes the finer Schürch et al. CODEX taxonomy.  A dashed arrow is used
where the exact upstream transformation is not present in this repository.
"""

# Support direct execution from the repository or any working directory.
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from analysis._paths import ANALYSIS_DIR, REPO_ROOT as _REPO_ROOT


from pathlib import Path
import subprocess


OUT = ANALYSIS_DIR / "outputs" / "crc_cell_type_hierarchy"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1800, 1320


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;"))


def rect(x, y, w, h, fill, stroke="#334155", sw=2, rx=16, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>' )


def text(x, y, value, size=25, weight=400, fill="#172033", anchor="start",
         family="Arial, Helvetica, sans-serif", italic=False):
    style = "italic" if italic else "normal"
    return (f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}" '
            f'text-anchor="{anchor}">{esc(value)}</text>')


def multiline(x, y, lines, size=23, line_h=32, weight=400, fill="#172033"):
    parts = [f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
             f'font-size="{size}" font-weight="{weight}" fill="{fill}">']
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else line_h
        parts.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    parts.append('</text>')
    return "".join(parts)


def path(x1, y1, x2, y2, color, dash=None, width=4, marker=True):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arrow)"' if marker else ""
    return (f'<path d="M {x1} {y1} C {(x1+x2)/2} {y1}, {(x1+x2)/2} {y2}, {x2} {y2}" '
            f'fill="none" stroke="{color}" stroke-width="{width}"{d}{m}/>' )


groups = [
    {
        "name": "Lymphoid",
        "color": "#3b82c4", "light": "#e8f2fb", "y": 232, "h": 245,
        "paper": [
            "B cells; plasma cells; CD3⁺ and CD8⁺ T cells",
            "CD4⁺ T; CD4⁺CD45RO⁺; CD4⁺GATA3⁺",
            "Tregs; NK cells",
        ],
        "ours": ["Bcell", "Plasma", "CD3T", "CD4T", "CD8T", "Treg"],
        "note": ["Three CD4 phenotypes → CD4T", "No explicit NK class"],
    },
    {
        "name": "Myeloid",
        "color": "#d97706", "light": "#fff2dc", "y": 500, "h": 245,
        "paper": [
            "CD11c⁺ DCs; CD11b⁺ monocytes",
            "CD11b⁺CD68⁺, CD68⁺, CD163⁺",
            "CD68⁺CD163⁺ macrophage states",
            "Granulocytes",
        ],
        "ours": ["DC", "Macrophage", "Neutrophil"],
        "note": ["Macrophage states are represented", "by one broad label"],
    },
    {
        "name": "Structural",
        "color": "#2f855a", "light": "#e8f7ee", "y": 768, "h": 245,
        "paper": [
            "Vasculature; lymphatics; nerves",
            "Smooth muscle; stroma; adipocytes",
        ],
        "ours": ["Endothelial", "Fibroblast", "Lymphatic", "Neuron", "SMV", "Stroma"],
        "note": ["Broad anatomical classes", "No explicit adipocyte class"],
    },
    {
        "name": "Tumor / mixed / QC",
        "color": "#b54747", "light": "#fcecec", "y": 1036, "h": 185,
        "paper": [
            "Tumor cells; immune; tumor/immune",
            "Immune/vasculature",
            "Uncertain or debris-like annotations",
        ],
        "ours": ["Tumor", "Unidentified"],
        "note": ["Handling of mixed/rare labels", "is not recorded here"],
        "uncertain": True,
    },
]

svg = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
    '<path d="M 0 0 L 10 5 L 0 10 z" fill="#667085"/></marker></defs>',
    '<rect width="100%" height="100%" fill="white"/>',
    text(70, 70, "Why the CRC analysis uses 17 cell types", 42, 700),
    text(70, 112, "The CISM input uses a coarser label vocabulary and a reduced region set", 26, 400, "#475467"),
    rect(65, 145, 480, 58, "#f1f5f9", "#cbd5e1", 2, 10),
    text(305, 183, "Paper taxonomy: 28 fine annotations", 25, 700, anchor="middle"),
    rect(625, 145, 430, 58, "#f8fafc", "#cbd5e1", 2, 10),
    text(840, 183, "Upstream consolidation", 25, 700, anchor="middle"),
    rect(1135, 145, 595, 58, "#dcecff", "#7ba7d6", 2, 10),
    text(1432, 183, "Labels used by CISM: 17", 25, 700, anchor="middle"),
]

for g in groups:
    y, h, c, light = g["y"], g["h"], g["color"], g["light"]
    # broad lineage tree
    svg.append(path(150, 710, 248, y+h/2, c, width=4, marker=False))
    svg.append(rect(240, y, 300, h, light, c, 3, 18))
    svg.append(rect(240, y, 300, 48, c, c, 1, 16))
    svg.append(text(390, y+33, g["name"], 25, 700, "white", "middle"))
    svg.append(multiline(266, y+83, g["paper"], 16, 29, 400, "#25324a"))

    # transformation note
    svg.append(rect(625, y+32, 430, h-64, "#ffffff", "#aab4c3", 2, 14,
                    "8 7" if g.get("uncertain") else None))
    svg.append(multiline(650, y+79, g["note"], 20, 32, 600, "#475467"))

    # outputs
    svg.append(rect(1135, y, 595, h, light, c, 3, 18))
    labels = g["ours"]
    cols = 3 if len(labels) >= 6 else 2
    bw, bh = (165 if cols == 3 else 238), 54
    gapx, gapy = 18, 18
    startx = 1165
    starty = y + 55
    for i, label in enumerate(labels):
        col, row = i % cols, i // cols
        bx, by = startx + col*(bw+gapx), starty + row*(bh+gapy)
        svg.append(rect(bx, by, bw, bh, "white", c, 2, 12))
        svg.append(text(bx+bw/2, by+35, label, 21, 700, c, "middle"))

    svg.append(path(540, y+h/2, 625, y+h/2, c, width=4, marker=True))
    svg.append(path(1055, y+h/2, 1135, y+h/2, c,
                    dash="9 8" if g.get("uncertain") else None, width=4, marker=True))

# root
svg.append(rect(70, 650, 145, 120, "#172033", "#172033", 2, 18))
svg.append(text(142, 696, "CRC", 29, 700, "white", "middle"))
svg.append(text(142, 730, "cells", 29, 700, "white", "middle"))

# explanatory footer
svg.append(rect(70, 1244, 1660, 53, "#f8fafc", "#d0d5dd", 1, 10))
svg.append(text(95, 1277,
    "Solid arrows: defensible lineage-level consolidation.  Dashed arrow: exact upstream rule is absent from the current repository.",
    20, 400, "#475467"))
svg.append('</svg>')

svg_path = OUT / "crc_cell_type_hierarchy.svg"
png_path = OUT / "crc_cell_type_hierarchy.png"
svg_path.write_text("\n".join(svg), encoding="utf-8")
subprocess.run([
    "rsvg-convert", "-w", str(W), "-h", str(H),
    "-o", str(png_path), str(svg_path)
], check=True)
print(svg_path)
print(png_path)
