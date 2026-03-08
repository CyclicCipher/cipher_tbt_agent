"""text_scanner.py -- Phase S.0b: Saccadic layout detection + text reading.
Design principle: THE MODEL MUST NEVER BE DESIGNED AROUND A SPECIFIC TASK.
THE MODEL MUST BE GENERAL.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from modalities.glyph_reader import GlyphReader, GlyphResult
from modalities.visual_symbol import _to_gray_f32


@dataclass
class LayoutRegion:
    """A detected UI region in a captured frame."""
    x0: int
    y0: int
    x1: int
    y1: int
    region_type: str
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def centroid(self) -> Tuple[int, int]:
        return ((self.x0 + self.x1) // 2, (self.y0 + self.y1) // 2)

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1, self.height)


@dataclass
class ReadResult:
    """Text extracted from a single LayoutRegion."""
    text: str
    glyphs: List[GlyphResult] = field(default_factory=list)
    confidence: float = 0.0
    region: Optional[LayoutRegion] = None


@dataclass
class FrameReading:
    """Complete reading of one captured frame."""
    narrative:  str
    buttons:    List[str]
    stats_text: str
    raw_regions: List[LayoutRegion] = field(default_factory=list)
    all_reads:   List[ReadResult]   = field(default_factory=list)


class TextScanner:
    """Saccadic text reader: layout detection + glyph-by-glyph reading."""

    def __init__(
        self,
        glyph_reader:     GlyphReader,
        foveal_size:      int   = 16,
        min_text_energy:  float = 0.02,
        min_region_area:  int   = 400,
        button_max_frac:  float = 0.20,
        button_max_height_frac: float = 0.06,
    ) -> None:
        self._reader               = glyph_reader
        self._foveal_size          = foveal_size
        self._min_text_energy      = min_text_energy
        self._min_region_area      = min_region_area
        self._button_max_frac      = button_max_frac
        self._button_max_height_fr = button_max_height_frac

    def scan_layout(self, frame: np.ndarray) -> List[LayoutRegion]:
        """Detect text/button/status regions in frame."""
        H, W = frame.shape[:2]
        energy = self._gradient_energy(frame)
        mask = (energy > self._min_text_energy).astype(np.uint8)
        regions = self._find_regions(mask, H, W)
        classified: List[LayoutRegion] = []
        for (x0, y0, x1, y1) in regions:
            if (x1-x0)*(y1-y0) < self._min_region_area:
                continue
            rtype = self._classify_region(x0, y0, x1, y1, H, W)
            conf  = float(energy[y0:y1, x0:x1].mean()) if y1>y0 and x1>x0 else 0.0
            classified.append(LayoutRegion(x0, y0, x1, y1, rtype, conf))
        classified.sort(key=lambda r: (r.y0 // max(1, H // 8), r.x0))
        return classified

    def _gradient_energy(self, frame: np.ndarray) -> np.ndarray:
        """Compute gradient magnitude as proxy for text-region energy."""
        gray = _to_gray_f32(frame)
        dy = np.diff(gray, axis=0, prepend=gray[:1])
        dx = np.diff(gray, axis=1, prepend=gray[:, :1])
        energy = (dx**2 + dy**2) ** 0.5
        k = max(1, self._foveal_size // 2)
        return _box_blur(energy, k)

    def _find_regions(self, mask, H, W):
        """Extract bounding boxes from a binary mask via row-scanning."""
        dilated = _dilate_horizontal(mask, ksize=self._foveal_size * 2)
        dilated = _dilate_vertical(dilated, ksize=self._foveal_size)
        row_has = dilated.any(axis=1)
        regions = []
        in_band = False
        band_y0 = 0
        for y in range(H):
            if row_has[y] and not in_band:
                band_y0 = y; in_band = True
            elif not row_has[y] and in_band:
                band = dilated[band_y0:y]
                col_has = band.any(axis=0)
                in_col = False; col_x0 = 0
                for x in range(W):
                    if col_has[x] and not in_col:
                        col_x0 = x; in_col = True
                    elif not col_has[x] and in_col:
                        regions.append((col_x0, band_y0, x, y)); in_col = False
                if in_col: regions.append((col_x0, band_y0, W, y))
                in_band = False
        if in_band:
            band = dilated[band_y0:]
            col_has = band.any(axis=0)
            in_col = False; col_x0 = 0
            for x in range(W):
                if col_has[x] and not in_col:
                    col_x0 = x; in_col = True
                elif not col_has[x] and in_col:
                    regions.append((col_x0, band_y0, x, H)); in_col = False
            if in_col: regions.append((col_x0, band_y0, W, H))
        return regions

    def _classify_region(self, x0, y0, x1, y1, H, W) -> str:
        """Classify a detected region: text, button, or status_bar."""
        w_frac = (x1-x0) / max(1, W)
        h_frac = (y1-y0) / max(1, H)
        if h_frac < 0.06 and w_frac > 0.3:
            return "status_bar"
        if w_frac < self._button_max_frac and h_frac < self._button_max_height_fr:
            return "button"
        return "text"

    def read_region(self, frame: np.ndarray, region: LayoutRegion) -> ReadResult:
        """Read all text within a LayoutRegion via left-to-right saccade."""
        sub = frame[region.y0:region.y1, region.x0:region.x1]
        if sub.size == 0:
            return ReadResult("", region=region)
        gray   = _to_gray_f32(sub)
        H_sub  = region.y1 - region.y0
        W_sub  = region.x1 - region.x0
        stride = self._foveal_size
        glyphs: List[GlyphResult] = []
        for y in range(0, max(1, H_sub - stride + 1), stride):
            for x in range(0, max(1, W_sub - stride + 1), stride):
                patch = gray[y:y+stride, x:x+stride]
                if patch.shape[0] < stride or patch.shape[1] < stride:
                    continue
                result = self._reader.read_patch(
                    patch,
                    x_center=region.x0 + x + stride // 2,
                    y_center=region.y0 + y + stride // 2,
                )
                if result.char and result.confidence > 0.1:
                    glyphs.append(result)
        text = self._assemble_text(glyphs, W_sub)
        conf = sum(g.confidence for g in glyphs) / max(1, len(glyphs))
        return ReadResult(text, glyphs, conf, region)

    def _assemble_text(self, glyphs: List[GlyphResult], region_w: int) -> str:
        """Assemble GlyphResult list into a string."""
        if not glyphs:
            return ""
        row_size = self._foveal_size
        rows: Dict[int, List[GlyphResult]] = {}
        for g in glyphs:
            rows.setdefault(g.y_center // row_size, []).append(g)
        lines = []
        for row_key in sorted(rows.keys()):
            row_glyphs = sorted(rows[row_key], key=lambda g: g.x_center)
            words = []
            cur = [row_glyphs[0].char]
            for i in range(1, len(row_glyphs)):
                gap = row_glyphs[i].x_center - row_glyphs[i-1].x_center
                if gap > row_size * 1.5:
                    words.append("".join(cur))
                    cur = []
                cur.append(row_glyphs[i].char)
            words.append("".join(cur))
            lines.append(" ".join(w for w in words if w.strip()))
        return "\n".join(l for l in lines if l)

    def read_frame(self, frame: np.ndarray) -> FrameReading:
        """Full pipeline: layout -> read each text/button region."""
        regions = self.scan_layout(frame)
        all_reads: List[ReadResult] = []
        narrative_parts: List[str] = []
        button_labels:   List[str] = []
        stats_parts:     List[str] = []
        for region in regions:
            if region.region_type == "background":
                continue
            result = self.read_region(frame, region)
            all_reads.append(result)
            text = result.text.strip()
            if not text:
                continue
            if region.region_type == "button":
                label = text.replace("\n", " ").strip()
                if label:
                    button_labels.append(label)
            elif region.region_type == "status_bar":
                stats_parts.append(text)
            else:
                narrative_parts.append(text)
        return FrameReading(
            narrative   = "\n".join(narrative_parts),
            buttons     = button_labels,
            stats_text  = " | ".join(stats_parts),
            raw_regions = regions,
            all_reads   = all_reads,
        )


def _box_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Simple box blur via cumulative sum."""
    if k <= 1:
        return img
    H, W = img.shape[:2]
    cs = np.cumsum(img, axis=1)
    b = (np.concatenate([cs[:, k:], np.zeros((H, k), dtype=img.dtype)], axis=1)
         - np.concatenate([np.zeros((H, k), dtype=img.dtype), cs[:, :-k]], axis=1)) / (2*k+1)
    cs = np.cumsum(b, axis=0)
    b = (np.concatenate([cs[k:], np.zeros((k, W), dtype=img.dtype)], axis=0)
         - np.concatenate([np.zeros((k, W), dtype=img.dtype), cs[:-k]], axis=0)) / (2*k+1)
    return b.astype(np.float32)


def _dilate_horizontal(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Binary dilation in the horizontal direction."""
    out = mask.copy()
    for i in range(1, ksize + 1):
        out[:, i:] |= mask[:, :-i]
        out[:, :-i] |= mask[:, i:]
    return out


def _dilate_vertical(mask: np.ndarray, ksize: int) -> np.ndarray:
    """Binary dilation in the vertical direction."""
    out = mask.copy()
    for i in range(1, ksize + 1):
        out[i:] |= mask[:-i]
        out[:-i] |= mask[i:]
    return out
