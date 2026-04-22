"""sheet_viz.py — browser visualization for a BrainSheet.

Generates a self-contained HTML file (no external dependencies) showing
the 2D cortical sheet: regions as labelled rectangles, connections as
colour-coded arrows, with hover highlighting.

Usage
-----
    from sheet import BrainSheet
    from sheet_viz import save_html, render_html

    # From a built brain:
    brain = BrainSheet.from_yaml('brain_configs/my_brain.yaml')
    save_html(brain, 'brain_configs/my_brain.html')

    # From a config file only (no columns instantiated):
    brain = BrainSheet.from_yaml('brain_configs/my_brain.yaml', build=False)
    save_html(brain, 'brain_configs/my_brain.html')
"""
from __future__ import annotations

import math
from typing import NamedTuple

from sheet import BrainSheet, RegionSpec, ConnectionSpec


# ── Style constants ───────────────────────────────────────────────────────────

_REGION_PALETTE = [
    '#D6E8FA', '#D5F5E3', '#FEF9E7', '#FDEDEC',
    '#EAF0FB', '#F9EBF5', '#E8F8F5', '#FDF2F8',
    '#EFF8FB', '#FEF5E7', '#F0F3FF', '#F5EEF8',
]

_KIND_COLOR = {
    'feedforward': '#2471A3',
    'feedback':    '#C0392B',
    'lateral':     '#1E8449',
    'modulatory':  '#D35400',
}

_KIND_DASH = {
    'feedforward': 'none',
    'feedback':    '7,3',
    'lateral':     '3,4',
    'modulatory':  '8,3,2,3',
}

_SVG_W   = 960
_SVG_H   = 680
_PADDING = 60     # px around the sheet content
_REGION_CORNER = 8
_ARROW_SIZE    = 10
_COL_DOT_R     = 2.5
_COL_DOT_MAX   = 200   # don't draw individual dots above this many columns


# ── Coordinate mapping ────────────────────────────────────────────────────────

class _CM:
    """Linear map from sheet units to SVG pixels."""

    def __init__(self, regions: dict) -> None:
        if not regions:
            self.ox = self.oy = _PADDING
            self.sx = self.sy = 1.0
            return
        specs = [r.spec for r in regions.values()]
        min_x = min(s.x for s in specs)
        min_y = min(s.y for s in specs)
        max_x = max(s.x + s.w for s in specs)
        max_y = max(s.y + s.h for s in specs)
        span_x = max_x - min_x or 1
        span_y = max_y - min_y or 1
        avail_w = _SVG_W - 2 * _PADDING
        avail_h = _SVG_H - 2 * _PADDING
        self.sx  = avail_w / span_x
        self.sy  = avail_h / span_y
        self.ox  = _PADDING - min_x * self.sx
        self.oy  = _PADDING - min_y * self.sy

    def px(self, sheet_x: float) -> float: return sheet_x * self.sx + self.ox
    def py(self, sheet_y: float) -> float: return sheet_y * self.sy + self.oy
    def pw(self, sheet_w: float) -> float: return sheet_w * self.sx
    def ph(self, sheet_h: float) -> float: return sheet_h * self.sy

    def center(self, spec: RegionSpec) -> tuple[float, float]:
        return (self.px(spec.x) + self.pw(spec.w) / 2,
                self.py(spec.y) + self.ph(spec.h) / 2)

    def edge_point(self, spec: RegionSpec,
                   nx: float, ny: float) -> tuple[float, float]:
        """Point on the boundary of spec's rectangle in direction (nx, ny)."""
        cx, cy = self.center(spec)
        hw = self.pw(spec.w) / 2
        hh = self.ph(spec.h) / 2
        if nx == 0 and ny == 0:
            return cx, cy
        # Parametric intersection: smallest t > 0 that hits an edge
        ts = []
        if nx != 0:
            ts.append(hw / abs(nx))
        if ny != 0:
            ts.append(hh / abs(ny))
        t = min(ts)
        return cx + nx * t, cy + ny * t


# ── SVG element helpers ───────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return f'{v:.2f}'


def _arrow_defs() -> str:
    """SVG <defs> block: one arrowhead marker per connection kind."""
    markers = []
    for kind, color in _KIND_COLOR.items():
        mid = kind
        markers.append(
            f'<marker id="arr-{mid}" markerWidth="10" markerHeight="7" '
            f'refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
            f'<polygon points="0 0, 10 3.5, 0 7" fill="{color}"/>'
            f'</marker>'
        )
    return '<defs>\n  ' + '\n  '.join(markers) + '\n</defs>'


def _region_svg(region, idx: int, cm: _CM) -> str:
    spec  = region.spec
    color = _REGION_PALETTE[idx % len(_REGION_PALETTE)]
    x  = _fmt(cm.px(spec.x))
    y  = _fmt(cm.py(spec.y))
    w  = _fmt(cm.pw(spec.w))
    h  = _fmt(cm.ph(spec.h))
    cx = _fmt(cm.px(spec.x) + cm.pw(spec.w) / 2)

    # Header label
    label_y = _fmt(cm.py(spec.y) + 18)
    grid_y  = _fmt(cm.py(spec.y) + 34)
    mini_y  = _fmt(cm.py(spec.y) + 48)

    parts = [
        f'<g class="region" id="region-{spec.name}" '
        f'data-name="{spec.name}">',
        f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'rx="{_REGION_CORNER}" ry="{_REGION_CORNER}" '
        f'fill="{color}" stroke="#999" stroke-width="1.5"/>',
        f'  <text x="{cx}" y="{label_y}" text-anchor="middle" '
        f'font-size="13" font-weight="bold" fill="#222">{spec.name}</text>',
        f'  <text x="{cx}" y="{grid_y}" text-anchor="middle" '
        f'font-size="10" fill="#555">{spec.rows}\u00d7{spec.cols} grid</text>',
        f'  <text x="{cx}" y="{mini_y}" text-anchor="middle" '
        f'font-size="10" fill="#555">{spec.n_mini} mini</text>',
    ]

    # Column dots (only when small enough to be legible)
    n = spec.rows * spec.cols
    if n <= _COL_DOT_MAX and cm.pw(spec.w) > 30 and cm.ph(spec.h) > 30:
        for gy in range(spec.rows):
            for gx in range(spec.cols):
                dot_x = _fmt(cm.px(spec.x + spec.w * (gx + 0.5) / spec.cols))
                dot_y = _fmt(cm.py(spec.y + spec.h * (gy + 0.5) / spec.rows))
                parts.append(
                    f'  <circle cx="{dot_x}" cy="{dot_y}" '
                    f'r="{_COL_DOT_R}" fill="#888" opacity="0.6"/>'
                )

    parts.append('</g>')
    return '\n'.join(parts)


def _connection_svg(conn: ConnectionSpec, conn_idx: int,
                    src: object, dst: object, cm: _CM,
                    offset_sign: int = 1) -> str:
    """SVG path + label for one connection."""
    color = _KIND_COLOR.get(conn.kind, '#888')
    dash  = _KIND_DASH.get(conn.kind, 'none')
    mid   = conn.kind

    src_cx, src_cy = cm.center(src.spec)
    dst_cx, dst_cy = cm.center(dst.spec)

    # Unit direction vector src → dst
    dx = dst_cx - src_cx
    dy = dst_cy - src_cy
    length = math.hypot(dx, dy) or 1.0
    nx, ny = dx / length, dy / length

    # Edge exit / entry points
    sx, sy = cm.edge_point(src.spec,  nx,  ny)
    ex, ey = cm.edge_point(dst.spec, -nx, -ny)

    # Perpendicular offset (separates bidirectional pairs)
    perp_x = -ny * 12 * offset_sign
    perp_y =  nx * 12 * offset_sign

    # Quadratic bezier control point at midpoint + perpendicular offset
    mx = (sx + ex) / 2 + perp_x
    my = (sy + ey) / 2 + perp_y

    path = (f'M {_fmt(sx)},{_fmt(sy)} '
            f'Q {_fmt(mx)},{_fmt(my)} '
            f'{_fmt(ex)},{_fmt(ey)}')

    # Label at the control point
    lx = _fmt(mx + perp_x * 0.3)
    ly = _fmt(my + perp_y * 0.3 - 4)

    dash_attr = f'stroke-dasharray="{dash}"' if dash != 'none' else ''

    return (
        f'<g class="connection" id="conn-{conn_idx}" '
        f'data-src="{conn.src}" data-dst="{conn.dst}" data-kind="{conn.kind}">\n'
        f'  <path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
        f'{dash_attr} marker-end="url(#arr-{mid})"/>\n'
        f'  <text x="{lx}" y="{ly}" font-size="9" fill="{color}" '
        f'text-anchor="middle">'
        f'{conn.kind} ({conn.strength})</text>\n'
        f'</g>'
    )


# ── Legend ────────────────────────────────────────────────────────────────────

def _legend_svg() -> str:
    items = list(_KIND_COLOR.items())
    lx = _SVG_W - 170
    ly = _SVG_H - 20 - len(items) * 20
    parts = [
        f'<rect x="{lx-8}" y="{ly-14}" width="162" height="{len(items)*20+8}" '
        f'rx="4" fill="white" stroke="#ccc" stroke-width="1"/>',
        f'<text x="{lx}" y="{ly}" font-size="11" font-weight="bold" fill="#333">'
        f'Connection types</text>',
    ]
    for i, (kind, color) in enumerate(items):
        row_y = ly + 16 + i * 18
        dash = _KIND_DASH[kind]
        dash_attr = f'stroke-dasharray="{dash}"' if dash != 'none' else ''
        parts.append(
            f'<line x1="{lx}" y1="{row_y}" x2="{lx+28}" y2="{row_y}" '
            f'stroke="{color}" stroke-width="2" {dash_attr}/>'
        )
        parts.append(
            f'<polygon points="{lx+24},{row_y-4} {lx+32},{row_y} {lx+24},{row_y+4}" '
            f'fill="{color}"/>'
        )
        parts.append(
            f'<text x="{lx+38}" y="{row_y+4}" font-size="10" fill="#333">'
            f'{kind}</text>'
        )
    return '\n'.join(parts)


# ── JavaScript for hover highlighting ────────────────────────────────────────

_JS = """
document.querySelectorAll('.region').forEach(r => {
  r.addEventListener('mouseenter', () => {
    const name = r.dataset.name;
    document.querySelectorAll('.connection').forEach(c => {
      const active = c.dataset.src === name || c.dataset.dst === name;
      c.style.opacity = active ? '1' : '0.15';
    });
    document.querySelectorAll('.region').forEach(q => {
      const n = q.dataset.name;
      const linked = document.querySelectorAll(
        `.connection[data-src="${name}"][data-dst="${n}"],` +
        `.connection[data-dst="${name}"][data-src="${n}"]`
      ).length > 0;
      q.style.opacity = (n === name || linked) ? '1' : '0.4';
    });
  });
  r.addEventListener('mouseleave', () => {
    document.querySelectorAll('.connection, .region').forEach(e => {
      e.style.opacity = '1';
    });
  });
});
"""


# ── Main render function ──────────────────────────────────────────────────────

def render_html(brain: BrainSheet, title: str = 'Brain Sheet') -> str:
    """Render a BrainSheet to a self-contained HTML string."""
    cm = _CM(brain.regions)
    region_list = list(brain.regions.values())

    # --- SVG content ---------------------------------------------------------
    svg_parts = [_arrow_defs()]

    # Connections (drawn first, behind regions)
    # Count how many connections share the same (src, dst) pair to offset them
    pair_count: dict[tuple, int] = {}
    for conn in brain.connections:
        key = (conn.src, conn.dst)
        pair_count[key] = pair_count.get(key, 0) + 1

    pair_seen: dict[tuple, int] = {}
    for i, conn in enumerate(brain.connections):
        src = brain.regions.get(conn.src)
        dst = brain.regions.get(conn.dst)
        if src is None or dst is None:
            continue
        key = (conn.src, conn.dst)
        rkey = (conn.dst, conn.src)  # reverse direction
        n_reverse = pair_count.get(rkey, 0)
        # If there's a connection in the other direction, offset this one
        offset = 1 if n_reverse > 0 else 0
        svg_parts.append(_connection_svg(conn, i, src, dst, cm, offset_sign=offset))

    # Regions (drawn on top)
    for idx, region in enumerate(region_list):
        svg_parts.append(_region_svg(region, idx, cm))

    # Legend
    svg_parts.append(_legend_svg())

    svg_body = '\n\n'.join(svg_parts)

    # --- HTML assembly -------------------------------------------------------
    region_count  = len(region_list)
    column_count  = sum(r.spec.rows * r.spec.cols for r in region_list)
    connect_count = len(brain.connections)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f0f2f5;
      padding: 20px;
    }}
    h1 {{
      font-size: 20px;
      color: #222;
      margin-bottom: 6px;
    }}
    .meta {{
      font-size: 12px;
      color: #666;
      margin-bottom: 14px;
    }}
    svg {{
      display: block;
      background: white;
      border: 1px solid #ddd;
      border-radius: 10px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      max-width: 100%;
      height: auto;
    }}
    .region {{ cursor: default; transition: opacity 0.15s; }}
    .connection {{ transition: opacity 0.15s; }}
    .region rect {{ transition: stroke-width 0.1s; }}
    .region:hover rect {{ stroke-width: 2.5 !important; stroke: #333 !important; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">
    {region_count} regions &nbsp;|&nbsp;
    {column_count} columns &nbsp;|&nbsp;
    {connect_count} connections
  </div>
  <svg viewBox="0 0 {_SVG_W} {_SVG_H}" width="{_SVG_W}" height="{_SVG_H}">
{svg_body}
  </svg>
  <script>{_JS}</script>
</body>
</html>
"""


def save_html(brain: BrainSheet, path: str,
              title: str = 'Brain Sheet') -> None:
    """Write the visualization to an HTML file."""
    html = render_html(brain, title=title)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved: {path}')
