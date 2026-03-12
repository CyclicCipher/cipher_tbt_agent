"""vision/encoder.py — Foveal patch tokenizer for visual environments.

Implements a two-level foveal pyramid that mirrors the human visual system
at 2 feet from a standard laptop screen.  Resolution is high at the gaze
point (enough to read 12pt text) and degrades with eccentricity.

─── Why foveal, not uniform ──────────────────────────────────────────────────
Uniform 16×16 patches fail for reading: a 10px-wide character is smaller
than one patch and is invisible.  The fovea solves this:

  At 2ft from a 141-PPI screen (e.g. 15.6" 1080p laptop):
    1° visual angle  = 610mm × tan(1°)        ≈ 59 pixels
    human fovea (~5°) = 5 × 59 pixels          ≈ 295 pixels diameter
    ⟹  foveal_radius = 150 px (default)

  Foveal patch size for reading:
    12pt body text = 16px tall, 10px wide
    foveal_patch=2  → character spans 5×8 patches  (clearly distinguishable)
    foveal_patch=4  → character spans 2.5×4 patches (marginal but works)

─── Architecture: two-level pyramid ─────────────────────────────────────────

  Level 0 (fovea):
    Crop of size (2*foveal_radius) × (2*foveal_radius) pixels centred on gaze.
    Tiled with foveal_patch × foveal_patch patches.
    Default: 300×300 px at 2px patches → 150×150 = 22 500 tokens.
    Resolves individual characters.

  Level 1 (context):
    Full frame at coarser resolution.
    Effective patch size: foveal_patch × context_scale (default ×8 → 16px).
    Default: 1920×1080 at 16px patches → 120×67 = 8 040 tokens.
    Guides saccade planning (where to look next).

─── Edge types: foveal_2d() topology ────────────────────────────────────────

  Code  Name        Direction
  ────  ──────────  ─────────────────────────────────────────────────────────
    0   right       adjacent patch to the right  (same level)
    1   left        adjacent patch to the left   (same level)
    2   down        adjacent patch below         (same level)
    3   up          adjacent patch above         (same level)
    4   zoom_in     context patch → fovea patch at same screen location
    5   zoom_out    fovea patch  → context patch at same screen location
    6   prev_frame  same-position patch in the preceding frame
    7   next_frame  same-position patch in the following frame

  8 types total, all fit in uint8.

─── Saccade learning ─────────────────────────────────────────────────────────
  zoom_in edges teach the MorphismGraph:
    "context token X at position (cr, cc) predicts fovea token Y"
  After enough training, EIG on zoom_in edges identifies which peripheral
  patches have the highest uncertainty → those are candidate saccade targets.
  The AgentLoop uses best_action() over zoom_in etypes to plan saccades.

─── Usage ────────────────────────────────────────────────────────────────────
  enc = FovealEncoder()
  enc.set_gaze(960, 540)            # direct foveal attention
  for frame in video:
      for src, etype, tgt in enc.stream_edges(frame):
          mg.observe_edge(src, etype, tgt)
  # Saccade: move gaze to highest-EIG peripheral patch
  new_gx, new_gy = enc.best_saccade_target(mg)
  enc.set_gaze(new_gx, new_gy)

─── Convenience: for_screen() ───────────────────────────────────────────────
  enc = FovealEncoder.for_screen(
      screen_width_px=1920, screen_height_px=1080,
      dpi=141, viewing_distance_cm=61,   # 2 feet = 61 cm
      foveal_degrees=5.0, foveal_patch=2,
  )
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Optional

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from ..core.topology import Topology, EdgeTypeRegistry


# ── Topology factory ───────────────────────────────────────────────────────────

def foveal_2d() -> Topology:
    """2D foveal topology: spatial + cross-level zoom + temporal.

    8 edge types (all fit in uint8):
      0 right       — adjacent patch to the right (same level)
      1 left        — adjacent patch to the left  (same level)
      2 down        — adjacent patch below        (same level)
      3 up          — adjacent patch above        (same level)
      4 zoom_in     — context → fovea (more detail at same screen location)
      5 zoom_out    — fovea  → context (less detail, wider view)
      6 prev_frame  — same patch position in the preceding frame
      7 next_frame  — same patch position in the following frame
    """
    reg = EdgeTypeRegistry()
    for name in ("right", "left", "down", "up"):
        reg.register(name)
    reg.register("zoom_in")       # code 4
    reg.register("zoom_out")      # code 5
    reg.register("prev_frame")    # code 6
    reg.register("next_frame")    # code 7

    class _Foveal2D(Topology):
        def stream_edges(self, grid: Any) -> Iterator[tuple[Any, int, Any]]:
            """Yield spatial edges for a single uniform-patch grid."""
            rows = len(grid)
            if rows == 0:
                return
            cols = len(grid[0])
            r  = self.registry
            rc = r.code("right"); lc = r.code("left")
            dc = r.code("down");  uc = r.code("up")
            for row in range(rows):
                for col in range(cols):
                    v = grid[row][col]
                    if col + 1 < cols:
                        nv = grid[row][col + 1]
                        yield v, rc, nv
                        yield nv, lc, v
                    if row + 1 < rows:
                        bv = grid[row + 1][col]
                        yield v, dc, bv
                        yield bv, uc, v

    return _Foveal2D("foveal_2d", reg)


# Keep video_2d for backwards compatibility with old VisionEncoder tests.
def video_2d() -> Topology:
    """Legacy topology from uniform VisionEncoder.  Prefer foveal_2d()."""
    reg = EdgeTypeRegistry()
    for name in ("right", "left", "down", "up"):
        reg.register(name)
    reg.register("prev_frame")   # code 4
    reg.register("next_frame")   # code 5

    class _Video2D(Topology):
        def stream_edges(self, grid: Any) -> Iterator[tuple[Any, int, Any]]:
            rows = len(grid)
            if rows == 0:
                return
            cols = len(grid[0])
            r  = self.registry
            rc = r.code("right"); lc = r.code("left")
            dc = r.code("down");  uc = r.code("up")
            for row in range(rows):
                for col in range(cols):
                    v = grid[row][col]
                    if col + 1 < cols:
                        nv = grid[row][col + 1]
                        yield v, rc, nv
                        yield nv, lc, v
                    if row + 1 < rows:
                        bv = grid[row + 1][col]
                        yield v, dc, bv
                        yield bv, uc, v

    return _Video2D("video_2d", reg)


# ── Shared patch-encoding helpers ──────────────────────────────────────────────

def _quantize_patches_numpy(arr: Any, patch_size: int, n_levels: int) -> list[list[str]]:
    """Extract non-overlapping patches from a (H,W,3) float32 array."""
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=np.float32)
    else:
        arr = arr.astype(np.float32)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    H, W, _ = arr.shape
    ps = patch_size
    H2 = (H // ps) * ps
    W2 = (W // ps) * ps
    arr = arr[:H2, :W2, :]

    n_rows = H2 // ps
    n_cols = W2 // ps
    patches = arr.reshape(n_rows, ps, n_cols, ps, 3).mean(axis=(1, 3))
    step = 256.0 / n_levels
    q = (patches / step).astype(int).clip(0, n_levels - 1)

    return [[f"c{q[row,col,0]}{q[row,col,1]}{q[row,col,2]}"
             for col in range(n_cols)]
            for row in range(n_rows)]


def _quantize_patches_pure(frame: Any, patch_size: int, n_levels: int) -> list[list[str]]:
    """Pure-Python fallback (no numpy/torch)."""
    ps   = patch_size
    step = 256.0 / n_levels
    H    = len(frame)
    W    = len(frame[0]) if H > 0 else 0
    grid: list[list[str]] = []
    for row in range(H // ps):
        grid_row: list[str] = []
        for col in range(W // ps):
            rs = gs = bs = 0.0
            for dr in range(ps):
                for dc in range(ps):
                    px = frame[row * ps + dr][col * ps + dc]
                    if isinstance(px, (list, tuple)):
                        rs += float(px[0]); gs += float(px[1]); bs += float(px[2])
                    else:
                        v = float(px); rs += v; gs += v; bs += v
            count = ps * ps
            r = min(int(rs / count / step), n_levels - 1)
            g = min(int(gs / count / step), n_levels - 1)
            b = min(int(bs / count / step), n_levels - 1)
            grid_row.append(f"c{r}{g}{b}")
        grid.append(grid_row)
    return grid


def _encode_patches(frame: Any, patch_size: int, n_levels: int) -> list[list[str]]:
    """Dispatch to the fastest available patch encoder."""
    if _HAS_TORCH and isinstance(frame, torch.Tensor):
        t = frame
        if t.dim() == 3 and t.shape[0] == 3:
            t = t.unsqueeze(0).float()
        elif t.dim() == 3 and t.shape[2] == 3:
            t = t.permute(2, 0, 1).unsqueeze(0).float()
        elif t.dim() == 4:
            t = t.float()
        else:
            raise ValueError(f"Unexpected tensor shape {t.shape}")
        patches = F.avg_pool2d(t, kernel_size=patch_size, stride=patch_size)
        patches = patches.squeeze(0).permute(1, 2, 0).cpu()
        step = 256.0 / n_levels
        q = (patches / step).long().clamp(0, n_levels - 1).numpy()
        n_rows, n_cols, _ = q.shape
        return [[f"c{q[row,col,0]}{q[row,col,1]}{q[row,col,2]}"
                 for col in range(n_cols)]
                for row in range(n_rows)]
    if _HAS_NUMPY:
        return _quantize_patches_numpy(frame, patch_size, n_levels)
    return _quantize_patches_pure(frame, patch_size, n_levels)


def _frame_hw(frame: Any) -> tuple[int, int]:
    """Return (H, W) from a frame (numpy, torch, or list-of-lists)."""
    if _HAS_NUMPY and isinstance(frame, np.ndarray):
        return frame.shape[0], frame.shape[1]
    if _HAS_TORCH and isinstance(frame, torch.Tensor):
        if frame.dim() == 3:
            return (frame.shape[1], frame.shape[2]) if frame.shape[0] == 3 else (frame.shape[0], frame.shape[1])
        return frame.shape[2], frame.shape[3]
    return len(frame), len(frame[0]) if frame else 0


def _crop_frame(frame: Any, y0: int, x0: int, y1: int, x1: int) -> Any:
    """Crop frame to [y0:y1, x0:x1]."""
    if _HAS_NUMPY and isinstance(frame, np.ndarray):
        return frame[y0:y1, x0:x1]
    if _HAS_TORCH and isinstance(frame, torch.Tensor):
        if frame.dim() == 3:
            return frame[:, y0:y1, x0:x1] if frame.shape[0] == 3 else frame[y0:y1, x0:x1]
        return frame[:, :, y0:y1, x0:x1]
    return [row[x0:x1] for row in frame[y0:y1]]


def _pad_or_resize(frame: Any, target_h: int, target_w: int) -> Any:
    """Pad a frame that's smaller than target to exactly (target_h, target_w).

    Edge-padding (replicate) so patches near the gaze boundary aren't black.
    """
    H, W = _frame_hw(frame)
    if H == target_h and W == target_w:
        return frame
    if _HAS_NUMPY:
        arr = frame if isinstance(frame, np.ndarray) else np.array(frame, dtype=np.uint8)
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = arr.transpose(1, 2, 0)
        pad_b = max(0, target_h - arr.shape[0])
        pad_r = max(0, target_w - arr.shape[1])
        if arr.ndim == 3:
            arr = np.pad(arr, ((0, pad_b), (0, pad_r), (0, 0)), mode='edge')
        else:
            arr = np.pad(arr, ((0, pad_b), (0, pad_r)), mode='edge')
        return arr[:target_h, :target_w]
    # Pure Python: replicate last row/col
    rows = list(frame)
    while len(rows) < target_h:
        rows.append(rows[-1] if rows else [(0, 0, 0)] * target_w)
    result = []
    for row in rows[:target_h]:
        r = list(row)
        while len(r) < target_w:
            r.append(r[-1] if r else (0, 0, 0))
        result.append(r[:target_w])
    return result


# ── FovealEncoder ──────────────────────────────────────────────────────────────

class FovealEncoder:
    """Two-level foveal encoder matching the human visual system at 2ft from laptop.

    Level 0 — Fovea:
      High-resolution crop centred on the gaze point.
      Default: 300×300 pixel window, 2px patches → 150×150 = 22,500 tokens.
      Resolves 12pt text (16px chars span ~5×8 patches).

    Level 1 — Context:
      Full frame at 8× coarser resolution (16px patches by default).
      Default: 1920×1080 at 16px → 120×67 = 8,040 tokens.
      Guides saccade planning: peripheral tokens with high EIG are candidate
      gaze targets.

    Cross-level zoom_in / zoom_out edges connect each foveal patch to the
    context patch covering the same screen coordinates.  After training, the
    MorphismGraph learns which peripheral contexts predict which foveal details.

    Temporal edges (prev_frame / next_frame) connect same-position patches
    across consecutive frames, enabling motion detection and event tracking.

    Gaze control:
      set_gaze(x, y)  — direct foveal attention to screen position (x, y).
                         Defaults to screen centre if never set.

    GPU acceleration:
      If torch is available: avg_pool2d on specified device.
      If only numpy: reshape + mean (fast for 60fps on CPU).
      Pure-Python fallback: no numpy/torch required.
    """

    def __init__(
        self,
        foveal_radius:  int = 150,  # pixels; ≈5° at 2ft from 141-PPI laptop
        foveal_patch:   int = 2,    # pixels per fovea patch (2px reads 16px chars)
        context_scale:  int = 8,    # context patch = foveal_patch × context_scale
        n_color_levels: int = 4,    # quantization: 4^3 = 64 distinct colour tokens
        n_frames:       int = 3,    # temporal buffer (compare last n_frames)
        device:         str = "cpu",
    ) -> None:
        self.foveal_radius  = foveal_radius
        self.foveal_patch   = foveal_patch
        self.context_patch  = foveal_patch * context_scale
        self.n_color_levels = n_color_levels
        self.n_frames       = n_frames
        self.device         = device
        self.topology       = foveal_2d()

        self._gaze:        Optional[tuple[int, int]] = None
        self._fovea_buf:   list[list[list[str]]]     = []
        self._context_buf: list[list[list[str]]]     = []

    # ── Gaze control ───────────────────────────────────────────────────────────

    def set_gaze(self, x: int, y: int) -> None:
        """Direct foveal attention to screen pixel (x, y).

        Call before stream_edges() to position the foveal crop.
        If never called, defaults to screen centre.
        """
        self._gaze = (x, y)

    def gaze_pixels(self, frame_h: int, frame_w: int) -> tuple[int, int]:
        """Return the current gaze position (gx, gy), defaulting to centre."""
        if self._gaze is not None:
            return self._gaze
        return (frame_w // 2, frame_h // 2)

    # ── Frame encoding ─────────────────────────────────────────────────────────

    def encode_fovea(self, frame: Any) -> list[list[str]]:
        """Extract and encode the foveal crop at foveal_patch resolution.

        Returns grid[row][col] of colour tokens over the 2*foveal_radius square
        centred on gaze.  Pads with edge values if gaze is near the frame border.
        """
        H, W = _frame_hw(frame)
        gx, gy = self.gaze_pixels(H, W)
        r  = self.foveal_radius
        x0 = max(0, gx - r);  x1 = min(W, gx + r)
        y0 = max(0, gy - r);  y1 = min(H, gy + r)
        crop = _crop_frame(frame, y0, x0, y1, x1)
        crop = _pad_or_resize(crop, 2 * r, 2 * r)
        return _encode_patches(crop, self.foveal_patch, self.n_color_levels)

    def encode_context(self, frame: Any) -> list[list[str]]:
        """Encode the full frame at context_patch resolution."""
        return _encode_patches(frame, self.context_patch, self.n_color_levels)

    # ── Edge streaming ─────────────────────────────────────────────────────────

    def stream_edges(self, frame: Any) -> Iterator[tuple[str, int, str]]:
        """Process one frame and yield all spatial + zoom + temporal edges.

        Yields (src_token, etype_code, tgt_token) for MorphismGraph.observe_edge().

        Spatial edges (right/left/down/up):
          within foveal grid and within context grid, separately.

        Zoom edges (zoom_in/zoom_out):
          between each foveal patch and the context patch that covers the same
          screen coordinates.  These encode the multi-scale structure: the agent
          learns that "context patch X at gaze" predicts "foveal pattern Y".

        Temporal edges (prev_frame/next_frame):
          for both fovea and context grids, connecting same-position patches
          across consecutive frames.  Foveal temporal edges detect fine motion
          (character cursor, tooltip appearance); context temporal edges detect
          coarse motion (window appearing, scrolling).

        Call reset() at scene or episode boundaries to discard the buffer.
        """
        H, W = _frame_hw(frame)
        gx, gy = self.gaze_pixels(H, W)

        fovea_grid   = self.encode_fovea(frame)
        context_grid = self.encode_context(frame)

        prev_fovea   = self._fovea_buf[-1]   if self._fovea_buf   else None
        prev_context = self._context_buf[-1] if self._context_buf else None

        self._fovea_buf.append(fovea_grid)
        self._context_buf.append(context_grid)
        if len(self._fovea_buf)   > self.n_frames:  self._fovea_buf.pop(0)
        if len(self._context_buf) > self.n_frames:  self._context_buf.pop(0)

        topo = self.topology

        # Spatial: fovea
        yield from topo.stream_edges(fovea_grid)

        # Spatial: context
        yield from topo.stream_edges(context_grid)

        # Cross-level zoom edges
        yield from self._zoom_edges(fovea_grid, context_grid, gx, gy)

        # Temporal: fovea
        if prev_fovea is not None:
            yield from self._temporal_edges(fovea_grid, prev_fovea)

        # Temporal: context
        if prev_context is not None:
            yield from self._temporal_edges(context_grid, prev_context)

    def _zoom_edges(
        self,
        fovea_grid:   list[list[str]],
        context_grid: list[list[str]],
        gx: int,
        gy: int,
    ) -> Iterator[tuple[str, int, str]]:
        """Yield zoom_in / zoom_out edges between levels at the same screen location.

        Fovea patch (fr, fc) maps to screen position:
          sx = (gx - foveal_radius) + fc * foveal_patch
          sy = (gy - foveal_radius) + fr * foveal_patch

        Context patch covering (sx, sy):
          cc = sx // context_patch
          cr = sy // context_patch
        """
        zi  = self.topology.registry.code("zoom_in")
        zo  = self.topology.registry.code("zoom_out")
        fp  = self.foveal_patch
        cp  = self.context_patch
        r   = self.foveal_radius
        ox  = gx - r   # top-left screen x of the foveal crop
        oy  = gy - r   # top-left screen y of the foveal crop

        ctx_rows = len(context_grid)
        ctx_cols = len(context_grid[0]) if context_grid else 0

        for fr, frow in enumerate(fovea_grid):
            for fc, fovea_tok in enumerate(frow):
                sx = ox + fc * fp
                sy = oy + fr * fp
                cc = sx // cp
                cr = sy // cp
                if 0 <= cr < ctx_rows and 0 <= cc < ctx_cols:
                    ctx_tok = context_grid[cr][cc]
                    yield ctx_tok,   zi, fovea_tok   # context → fovea detail
                    yield fovea_tok, zo, ctx_tok     # fovea  → context overview

    def _temporal_edges(
        self,
        cur_grid:  list[list[str]],
        prev_grid: list[list[str]],
    ) -> Iterator[tuple[str, int, str]]:
        """Yield prev_frame / next_frame edges between consecutive grids."""
        pc = self.topology.registry.code("prev_frame")
        nc = self.topology.registry.code("next_frame")
        rows = min(len(cur_grid), len(prev_grid))
        cols = min(
            len(cur_grid[0])  if cur_grid  else 0,
            len(prev_grid[0]) if prev_grid else 0,
        )
        for row in range(rows):
            for col in range(cols):
                cur  = cur_grid[row][col]
                prev = prev_grid[row][col]
                yield cur,  pc, prev
                yield prev, nc, cur

    # ── Saccade target ─────────────────────────────────────────────────────────

    def best_saccade_target(self, mg) -> Optional[tuple[int, int]]:
        """Return the screen (x, y) of the context patch with the highest EIG.

        Iterates over all context patches, computes expected_info_gain() for
        the zoom_in edge type at each patch's atom ID, and returns the screen
        centre of the patch with the highest EIG.

        Returns None if the MorphismGraph has no data yet.

        Usage in agent loop:
            gx, gy = enc.best_saccade_target(mg) or current_gaze
            enc.set_gaze(gx, gy)
        """
        from ..reasoning.active_inference import expected_info_gain

        zi  = self.topology.registry.code("zoom_in")
        cp  = self.context_patch

        if not self._context_buf:
            return None

        ctx_grid = self._context_buf[-1]
        best_eig = -1.0
        best_pos: Optional[tuple[int, int]] = None

        for cr, crow in enumerate(ctx_grid):
            for cc, ctx_tok in enumerate(crow):
                atom_id = mg.atoms.get(ctx_tok)
                if atom_id is None:
                    continue
                eig = expected_info_gain(mg, atom_id, zi)
                if eig > best_eig:
                    best_eig = eig
                    # Screen centre of this context patch
                    best_pos = (cc * cp + cp // 2, cr * cp + cp // 2)

        return best_pos

    # ── Utility ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all frame buffers (call at scene/episode boundaries)."""
        self._fovea_buf.clear()
        self._context_buf.clear()

    def fovea_grid_shape(self) -> tuple[int, int]:
        """Return (n_rows, n_cols) of the foveal patch grid."""
        r  = self.foveal_radius
        fp = self.foveal_patch
        side = (2 * r) // fp
        return (side, side)

    def context_grid_shape(self, frame_h: int, frame_w: int) -> tuple[int, int]:
        """Return (n_rows, n_cols) of the context patch grid."""
        cp = self.context_patch
        return (frame_h // cp, frame_w // cp)

    def n_tokens(self) -> int:
        """Number of distinct colour tokens: n_color_levels³."""
        return self.n_color_levels ** 3

    @classmethod
    def for_screen(
        cls,
        screen_width_px:     int   = 1920,
        screen_height_px:    int   = 1080,
        dpi:                 float = 141.0,    # 15.6" 1080p laptop
        viewing_distance_cm: float = 61.0,    # 2 feet = 61 cm
        foveal_degrees:      float = 5.0,      # ~5° human fovea diameter
        foveal_patch:        int   = 2,
        **kwargs,
    ) -> "FovealEncoder":
        """Construct a FovealEncoder with the correct foveal_radius for the screen.

        Computes foveal_radius = viewing_distance_cm × (dpi/2.54) × tan(foveal_degrees/2°).

        Examples:
          FovealEncoder.for_screen()                           # 15.6" 1080p at 2ft
          FovealEncoder.for_screen(dpi=227, screen_width_px=2560)  # 13" Retina at 2ft
          FovealEncoder.for_screen(viewing_distance_cm=75)    # 2.5ft
        """
        px_per_cm  = dpi / 2.54
        dist_px    = viewing_distance_cm * px_per_cm
        foveal_radius = int(round(dist_px * math.tan(math.radians(foveal_degrees / 2))))
        return cls(
            foveal_radius=foveal_radius,
            foveal_patch=foveal_patch,
            **kwargs,
        )

    def __repr__(self) -> str:
        fr, fc = self.fovea_grid_shape()
        return (
            f"FovealEncoder("
            f"foveal_radius={self.foveal_radius}px, "
            f"foveal_patch={self.foveal_patch}px, "
            f"context_patch={self.context_patch}px, "
            f"fovea_grid={fr}×{fc}={fr*fc} tokens, "
            f"n_colors={self.n_color_levels}³={self.n_tokens()}, "
            f"gaze={self._gaze!r})"
        )


# ── Legacy VisionEncoder (uniform patches, kept for existing tests) ────────────

class VisionEncoder:
    """Uniform-resolution patch encoder (legacy — prefer FovealEncoder).

    Tiles the entire frame with identical patch_size×patch_size patches.
    No resolution gradient; cannot resolve text at standard font sizes.
    Retained for backwards compatibility with non-reading tasks (e.g. Atari-
    style games with large, clearly separated sprites).

    Use FovealEncoder for any task involving reading or fine-grained detail.
    """

    def __init__(
        self,
        patch_size: int = 16,
        n_levels:   int = 4,
        n_frames:   int = 3,
        device:     str = "cpu",
    ) -> None:
        self.patch_size = patch_size
        self.n_levels   = n_levels
        self.n_frames   = n_frames
        self.device     = device
        self.topology   = video_2d()
        self._frame_buf: list[list[list[str]]] = []

    def encode_frame(self, frame: Any) -> list[list[str]]:
        return _encode_patches(frame, self.patch_size, self.n_levels)

    def stream_edges(self, frame: Any) -> Iterator[tuple[str, int, str]]:
        grid      = self.encode_frame(frame)
        prev_grid = self._frame_buf[-1] if self._frame_buf else None
        self._frame_buf.append(grid)
        if len(self._frame_buf) > self.n_frames:
            self._frame_buf.pop(0)

        yield from self.topology.stream_edges(grid)

        if prev_grid is not None:
            pc = self.topology.registry.code("prev_frame")
            nc = self.topology.registry.code("next_frame")
            rows = min(len(grid), len(prev_grid))
            cols = min(len(grid[0]) if grid else 0, len(prev_grid[0]) if prev_grid else 0)
            for row in range(rows):
                for col in range(cols):
                    cur  = grid[row][col]
                    prev = prev_grid[row][col]
                    yield cur,  pc, prev
                    yield prev, nc, cur

    def stream_tokens_1d(self, frame: Any) -> Iterator[tuple[str, Optional[int]]]:
        from ..core.topology import sequence_1d
        seq_topo  = sequence_1d()
        next_code = seq_topo.registry.code("next")
        grid      = self.encode_frame(frame)
        prev_grid = self._frame_buf[-1] if self._frame_buf else None
        self._frame_buf.append(grid)
        if len(self._frame_buf) > self.n_frames:
            self._frame_buf.pop(0)
        first = True
        for row in range(len(grid)):
            for col in range(len(grid[0]) if grid else 0):
                if prev_grid is not None:
                    prev_tok = prev_grid[row][col]
                    yield prev_tok, (None if first else next_code)
                    first = False
                yield grid[row][col], (None if first else next_code)
                first = False

    def reset(self) -> None:
        self._frame_buf.clear()

    def grid_shape(self, frame_height: int, frame_width: int) -> tuple[int, int]:
        ps = self.patch_size
        return (frame_height // ps, frame_width // ps)

    def n_tokens(self) -> int:
        return self.n_levels ** 3

    def __repr__(self) -> str:
        return (
            f"VisionEncoder(patch_size={self.patch_size}, "
            f"n_levels={self.n_levels}, n_frames={self.n_frames})"
        )
