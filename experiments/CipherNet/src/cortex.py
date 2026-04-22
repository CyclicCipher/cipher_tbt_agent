"""Hierarchical cortex — V1 → IT, built from config.

ARCHITECTURE
------------
  Cortex
    └─ Layer  (ordered list, sensor → higher)
         └─ MacroColumn × (grid_h × grid_w)
               └─ MiniColumn × N_MINI

OBSERVATION MODEL (uniform across all layers)
---------------------------------------------
Every layer receives (sdr, location_key) observations via observe_multi().

Sensor layers (V1):
  sdr          = encoder output (Gabor/HOG np.ndarray, or None for blank)
  location_key = column frame.position_key() (centroid-relative retinal pos)

Non-sensor layers (IT, V2 …):
  sdr          = source column's SDR (Gabor when source is a sensor layer)
  location_key = source column's frame.position_key()

There is no sensor vs. non-sensor branching in the forward sweep — every
layer uses the same observe_multi() path.  Different IT columns cover
spatially distinct V1 subsets, so their models are keyed by different
retinal positions → distinct object models → independent votes.

RECEPTIVE FIELDS (auto-computed from grid sizes)
-------------------------------------------------
For upper layer (Uh × Uw) receiving from lower (Lh × Lw):

  centre in lower coords:
    cx = gx * (Lw-1)/(Uw-1)   [or (Lw-1)/2 if Uw==1]
    cy = gy * (Lh-1)/(Uh-1)   [or (Lh-1)/2 if Uh==1]
  radius:
    rf_half_w = Lw / Uw,   rf_half_h = Lh / Uh
  field = lower columns where |vx-cx| <= rf_half_w and |vy-cy| <= rf_half_h
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reference_frames import make_frame, RetinotopicFrame, ReferenceFrame
from column import MacroColumn
from cortical_message import CorticalMessage
from output_cortex import OutputCortex
from modalities.base import SensorModality


CONFIDENCE_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Encoder builders (used only inside from_config to construct VisualModality)
# ---------------------------------------------------------------------------

def _build_encoder(encoder_type: str, params: dict):
    if encoder_type == 'hog':
        from codebook import HOGEncoder
        enc = HOGEncoder(
            patch_size=params.get('patch_size', 5),
            n_bins=params.get('n_bins', 8),
            top_k=params.get('top_k', 3),
        )
        enc.fit(verbose=True)
        return enc
    if encoder_type == 'gabor':
        from codebook import GaborFilterBank
        enc = GaborFilterBank(
            patch_size    =params.get('patch_size',     5),
            n_orientations=params.get('n_orientations', 16),
            n_frequencies =params.get('n_frequencies',  8),
            n_phases      =params.get('n_phases',        2),
            n_positions   =params.get('n_positions',     8),
            top_k         =params.get('top_k',          40),
        )
        enc.fit(verbose=True)
        return enc
    raise ValueError(f"Unknown encoder type '{encoder_type}'")


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    """One cortical layer: a grid of MacroColumns."""

    id:               str
    grid_h:           int
    grid_w:           int
    columns:          list[MacroColumn]
    input_source:     str                  # 'sensor' | lower-layer id
    modality:         SensorModality | None  # non-None only for sensor layers
    supervised:       bool  = False
    chl:              bool  = False
    lateral_bonus:    float = 0.0
    # receptive_fields[i] = flat lower-layer column indices for column i
    receptive_fields: list[list[int]] = field(default_factory=list)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    @property
    def patch_size(self) -> int:
        """Convenience passthrough for diagnostics / config logging."""
        return (self.modality._patch_size
                if self.modality is not None else 0)

    @property
    def stride(self) -> int:
        return (self.modality._stride
                if self.modality is not None else 0)

    def stats(self) -> dict:
        used       = sum(mc.stats()['used_mini']       for mc in self.columns)
        total_locs = sum(mc.stats()['total_locations'] for mc in self.columns)
        return {
            'id':              self.id,
            'n_macrocolumns':  self.n_columns,
            'n_mini':          self.columns[0].N_MINI if self.columns else 0,
            'supervised':      self.supervised,
            'used_mini':       used,
            'total_locations': total_locs,
        }


# ---------------------------------------------------------------------------
# Cortex
# ---------------------------------------------------------------------------

class Cortex:
    """Multi-layer cortex built from a config dict.

    Layers are ordered bottom-up (sensor first, IT last).
    """

    def __init__(self, name: str,
                 layers: dict[str, Layer],
                 ordered_ids: list[str]):
        self.name           = name
        self.layers         = layers
        self._order         = ordered_ids
        self._output_cortex = OutputCortex()

    # ------------------------------------------------------------------
    # Construction from config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict, eye=None) -> 'Cortex':
        """Build a Cortex from a YAML-loaded config dict.

        Args:
            config: Parsed YAML config (see configs/guided.yaml for schema).
            eye:    Eye instance for vision-based sensor layers.  Only used
                    when a layer has input.source == 'sensor' and the encoder
                    type is visual.  Encapsulated into a VisualModality — the
                    Cortex itself never stores or calls the eye directly.
        """
        name = config.get('name', 'cortex')
        layers:      dict[str, Layer] = {}
        ordered_ids: list[str]        = []

        for lcfg in config.get('layers', []):
            lid            = lcfg['id']
            grid_h, grid_w = lcfg['grid']
            frame_cfg      = lcfg['reference_frame']
            frame_type     = frame_cfg['type']
            frame_params   = frame_cfg.get('params', {})
            n_mini         = lcfg.get('n_mini', 10)
            supervised     = lcfg.get('supervised', False)
            chl            = lcfg.get('chl', False)
            lateral_bonus  = lcfg.get('lateral_bonus', 0.0)
            input_cfg      = lcfg.get('input', {})
            input_source   = input_cfg.get('source', 'sensor')
            miss_penalty   = lcfg.get('miss_penalty', 0.0)

            patch_size = frame_params.get('patch_size', 5)
            stride     = frame_params.get('stride', 3)

            # Sensor layers get a SensorModality; non-sensor layers get None.
            modality: SensorModality | None = None
            if input_source == 'sensor':
                enc_params = dict(input_cfg.get('encoder_params', {}))
                enc_params.setdefault('patch_size', patch_size)
                encoder = _build_encoder(
                    input_cfg.get('encoder', 'hog'), enc_params)
                from modalities.vision import VisualModality
                modality = VisualModality(
                    eye, encoder, grid_h, grid_w, patch_size, stride)

            columns: list[MacroColumn] = []
            for gy in range(grid_h):
                for gx in range(grid_w):
                    frame = cls._make_frame(frame_type, frame_params,
                                            gx, gy, eye)
                    columns.append(MacroColumn(frame, n_mini=n_mini,
                                               miss_penalty=miss_penalty))

            rf_indices: list[list[int]] = []
            if input_source != 'sensor' and input_source in layers:
                lower      = layers[input_source]
                rf_indices = cls._compute_rf(
                    lower.grid_h, lower.grid_w, grid_h, grid_w)

            layers[lid] = Layer(
                id=lid, grid_h=grid_h, grid_w=grid_w,
                columns=columns,
                input_source=input_source,
                modality=modality,
                supervised=supervised,
                chl=chl,
                lateral_bonus=lateral_bonus,
                receptive_fields=rf_indices,
            )
            ordered_ids.append(lid)

        return cls(name, layers, ordered_ids)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_frame(frame_type: str, params: dict,
                    gx: int, gy: int, eye=None) -> ReferenceFrame:
        if frame_type == 'retinotopic':
            return RetinotopicFrame(
                grid_pos=(gx, gy),
                stride=params.get('stride', 3),
                patch_size=params.get('patch_size', 5),
                retina_size=params.get('retina_size', 19),
                image_size=(params.get('image_w', 28),
                            params.get('image_h', 28)),
                encoding=params.get('encoding', 'grid'),
            )
        return make_frame(frame_type, params)

    @staticmethod
    def _compute_rf(lower_h: int, lower_w: int,
                    upper_h: int, upper_w: int) -> list[list[int]]:
        """Flat lower-column indices in each upper column's receptive field."""
        rf_half_w = lower_w / upper_w
        rf_half_h = lower_h / upper_h
        result: list[list[int]] = []

        for gy in range(upper_h):
            for gx in range(upper_w):
                cx = (gx * (lower_w - 1) / (upper_w - 1)
                      if upper_w > 1 else (lower_w - 1) / 2.0)
                cy = (gy * (lower_h - 1) / (upper_h - 1)
                      if upper_h > 1 else (lower_h - 1) / 2.0)
                result.append([
                    vy * lower_w + vx
                    for vy in range(lower_h)
                    for vx in range(lower_w)
                    if abs(vx - cx) <= rf_half_w and abs(vy - cy) <= rf_half_h
                ])

        return result

    # ------------------------------------------------------------------
    # Lateral inhibition
    # ------------------------------------------------------------------

    @staticmethod
    def _lateral_pass(layer: Layer) -> None:
        """Apply lateral consistency pressure from grid-adjacent columns."""
        if layer.lateral_bonus <= 0.0:
            return
        h, w = layer.grid_h, layer.grid_w
        tentative = [col.tentative_winner() for col in layer.columns]
        for gy in range(h):
            for gx in range(w):
                idx = gy * w + gx
                neighbor_winners: list[int] = []
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor_winners.append(tentative[ny * w + nx])
                if neighbor_winners:
                    layer.columns[idx].apply_lateral_input(
                        neighbor_winners, layer.lateral_bonus)

    @staticmethod
    def _global_lateral_pass(layer: Layer, bonus: float = 0.05) -> None:
        """Apply full-layer plurality vote as a weak global prior."""
        if not layer.columns:
            return
        winners = [col.tentative_winner() for col in layer.columns]
        if not winners:
            return
        plurality = max(set(winners), key=winners.count)
        for col in layer.columns:
            col.apply_lateral_input([plurality], bonus)

    # ------------------------------------------------------------------
    # Sensor encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_sensor_fixation(layer: Layer,
                                 prepared_input: Any,
                                 fixation: tuple,
                                 ) -> tuple[list, list]:
        """Encode one fixation for a sensor layer.

        Delegates feature extraction to layer.modality.encode(), then
        updates each column's reference frame to compute location keys.

        Returns (features, location_keys):
          features[i]      = SDR np.ndarray or None (blank patch)
          location_keys[i] = frame.position_key() for column i
        """
        features: list = layer.modality.encode(prepared_input, fixation)

        locations: list[tuple] = []
        for col in layer.columns:
            col.frame.set_position(fixation)
            locations.append(col.frame.position_key())

        return features, locations

    # ------------------------------------------------------------------
    # Feedback pass
    # ------------------------------------------------------------------

    def _feedback_pass(self,
                       layer_tentatives: dict[str, list[int]],
                       layer_outputs:    dict[str, tuple[list, list]],
                       feedback_bonus:   float = 0.3) -> None:
        """Top-down prediction: upper winner's stored SDR at each RF slot
        primes the best-matching lower minicolumn for the next fixation.

        With V1 n_mini=1 this is a no-op (no competition to bias).
        """
        for upper_idx in range(len(self._order) - 1, 0, -1):
            upper_id    = self._order[upper_idx]
            lower_id    = self._order[upper_idx - 1]
            upper_layer = self.layers[upper_id]
            lower_layer = self.layers[lower_id]
            upper_winners = layer_tentatives[upper_id]
            _, lower_locs = layer_outputs[lower_id]

            for ui, u_col in enumerate(upper_layer.columns):
                winner_mc = u_col.minicolumns[upper_winners[ui]]
                rf        = upper_layer.receptive_fields[ui]
                for j, lower_col_idx in enumerate(rf):
                    loc           = lower_locs[lower_col_idx]
                    predicted_sdr = winner_mc._model.get(loc)
                    if predicted_sdr is None:
                        continue
                    lower_col = lower_layer.columns[lower_col_idx]
                    if lower_col.N_MINI == 1:
                        continue
                    scores    = [mc.overlap_score(predicted_sdr, loc)
                                 for mc in lower_col.minicolumns]
                    best_mini = int(np.argmax(scores))
                    lower_col.receive_feedback_by_index(best_mini, feedback_bonus)

    # ------------------------------------------------------------------
    # Forward sweep (single fixation)
    # ------------------------------------------------------------------

    def _forward_sweep(self, prepared: dict, fix: tuple,
                       v1_cache: dict | None = None,
                       ) -> tuple[dict, dict]:
        """One fixation: bottom-up observations + lateral + feedback.

        All layers use the same observe_multi() path regardless of whether
        they are sensor or non-sensor layers.

        Args:
            prepared:  Dict mapping sensor layer id -> preprocessed input
                       (output of layer.modality.preprocess()).
            fix:       (x, y) fixation position.
            v1_cache:  Optional pre-computed {fix: (feats, locs)} for the
                       bottom sensor layer (avoids re-encoding during guided
                       fixation selection in classify()).

        Returns:
            layer_outputs[lid]    = (feats, locs)
              feats[i] = np.ndarray SDR or None
              locs[i]  = position key tuple
            layer_tentatives[lid] = list[int] winner indices (after lateral)
        """
        layer_outputs:    dict[str, tuple[list, list]] = {}
        layer_tentatives: dict[str, list[int]]         = {}

        for lid in self._order:
            layer = self.layers[lid]

            if layer.input_source == 'sensor':
                if v1_cache and fix in v1_cache:
                    feats, locs = v1_cache[fix]
                else:
                    feats, locs = self._encode_sensor_fixation(
                        layer, prepared[lid], fix)
                for i, col in enumerate(layer.columns):
                    if feats[i] is not None:
                        col.observe_multi([(feats[i], locs[i])])
                out_feats, out_locs = feats, locs

            else:
                lower_feats, lower_locs = layer_outputs[layer.input_source]
                for i, col in enumerate(layer.columns):
                    rf  = layer.receptive_fields[i]
                    obs = [(lower_feats[j], lower_locs[j])
                           for j in rf if lower_feats[j] is not None]
                    if obs:
                        col.observe_multi(obs)
                # This layer's outputs (one-hots at column positions) are
                # computed after lateral passes below so they reflect WTA.
                out_feats = None   # filled in after lateral
                out_locs  = [col.frame.position_key() for col in layer.columns]

            self._lateral_pass(layer)
            self._global_lateral_pass(layer)

            winners = [col.tentative_winner() for col in layer.columns]
            layer_tentatives[lid] = winners

            if layer.input_source != 'sensor':
                # Build winner one-hot outputs for potential layers above.
                out_feats = []
                for i, col in enumerate(layer.columns):
                    vec = np.zeros(col.N_MINI, dtype=np.int8)
                    vec[winners[i]] = 1
                    out_feats.append(vec)

            layer_outputs[lid] = (out_feats, out_locs)

        self._feedback_pass(layer_tentatives, layer_outputs)
        return layer_outputs, layer_tentatives

    # ------------------------------------------------------------------
    # Guided fixation selection
    # ------------------------------------------------------------------

    def _aggregate_it_evidence(self, it_layer: Layer) -> list[float]:
        """Sum evidence across all IT columns (multi-column voting)."""
        if not it_layer.columns or not it_layer.columns[0]._evidence:
            return []
        n = it_layer.columns[0].N_MINI
        total = [0.0] * n
        for col in it_layer.columns:
            for i, e in enumerate(col._evidence):
                total[i] += e
        return total

    def _guided_fixation(self, it_layer: Layer, v1_layer: Layer,
                         remaining: list[tuple],
                         v1_cache:  dict[tuple, tuple],
                         ) -> tuple:
        """Return the candidate that best discriminates IT leader from runner-up."""
        total_ev = self._aggregate_it_evidence(it_layer)
        if not total_ev:
            return remaining[0]

        sorted_cls = sorted(range(len(total_ev)),
                            key=lambda i: total_ev[i], reverse=True)
        leader = sorted_cls[0]
        runner = sorted_cls[1] if len(sorted_cls) > 1 else sorted_cls[0]

        best_fix  = remaining[0]
        best_disc = -float('inf')

        for fix in remaining:
            if fix not in v1_cache:
                continue
            feats, locs = v1_cache[fix]
            disc = 0.0
            for it_ci, it_col in enumerate(it_layer.columns):
                rf    = it_layer.receptive_fields[it_ci]
                obs   = [(feats[j], locs[j]) for j in rf if feats[j] is not None]
                n_obs = max(len(obs), 1)
                leader_e = sum(
                    it_col.minicolumns[leader].overlap_score(f, l) / n_obs
                    for f, l in obs)
                runner_e = sum(
                    it_col.minicolumns[runner].overlap_score(f, l) / n_obs
                    for f, l in obs)
                disc += leader_e - runner_e
            if disc > best_disc:
                best_disc = disc
                best_fix  = fix

        return best_fix

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, image: np.ndarray, label: int,
              fixations: list[tuple]) -> None:
        """TBT evidence accumulation then WTA commit + OutputCortex update."""
        prepared = {
            lid: self.layers[lid].modality.preprocess(image)
            for lid in self._order
            if self.layers[lid].modality is not None
        }

        for lid in self._order:
            for col in self.layers[lid].columns:
                col.begin_image()

        for fix in fixations:
            self._forward_sweep(prepared, fix)

        for lid in self._order:
            layer = self.layers[lid]
            if layer.chl:
                for col in layer.columns:
                    col.commit_chl(label, write=True)
            elif layer.supervised:
                for col in layer.columns:
                    col.commit_supervised(label, write=True)
            else:
                for col in layer.columns:
                    col.commit(write=True)

        final_layer = self.layers[self._order[-1]]
        for ci, col in enumerate(final_layer.columns):
            self._output_cortex.learn(ci, col.sdr(), label)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, image: np.ndarray,
                 fixations: list[tuple],
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 **_kwargs,
                 ) -> tuple[int, dict]:
        """Classify via TBT evidence accumulation with guided fixation.

        First 2 fixations follow the pre-computed sequence.  Remaining
        fixations are chosen greedily by IT discriminability.  Early stop
        once the leading class accounts for >= confidence_threshold of
        aggregated IT evidence.
        """
        prepared = {
            lid: self.layers[lid].modality.preprocess(image)
            for lid in self._order
            if self.layers[lid].modality is not None
        }

        for lid in self._order:
            for col in self.layers[lid].columns:
                col.begin_image()

        it_layer = self.layers[self._order[-1]]
        v1_lid   = self._order[0]
        v1_layer = self.layers[v1_lid]

        # Pre-cache V1 encodings so guided fixation can score all candidates
        # without re-encoding (expensive), and so the main sweep can reuse them.
        v1_cache: dict[tuple, tuple] = {}
        if v1_layer.modality is not None:
            for fix in fixations:
                v1_cache[fix] = self._encode_sensor_fixation(
                    v1_layer, prepared[v1_lid], fix)

        remaining = list(fixations)

        for k in range(len(fixations)):
            if k < 2 or not remaining:
                fix = remaining.pop(0)
            else:
                fix = self._guided_fixation(
                    it_layer, v1_layer, remaining, v1_cache)
                remaining.remove(fix)

            self._forward_sweep(prepared, fix, v1_cache)

            if k >= 1:
                agg   = self._aggregate_it_evidence(it_layer)
                total = sum(agg)
                if total > 0 and max(agg) / total >= confidence_threshold:
                    active = [
                        (ci, frozenset([col.tentative_winner()]))
                        for ci, col in enumerate(it_layer.columns)
                    ]
                    pred, votes = self._output_cortex.classify(active)
                    if pred != -1:
                        return pred, votes

        active = [
            (ci, frozenset([col.tentative_winner()]))
            for ci, col in enumerate(it_layer.columns)
        ]
        pred, votes = self._output_cortex.classify(active)
        return pred, votes

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def hierarchy_stats(self) -> list[dict]:
        return [self.layers[lid].stats() for lid in self._order]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnose(self, top_locs: int = 3) -> str:
        """Representation report: each layer -> column -> minicolumn model summary."""
        lines: list[str] = []
        bar_width = 20

        for lid in self._order:
            layer = self.layers[lid]
            sup_tag = ' [supervised]' if layer.supervised else ''
            lines.append(
                f"\n{'='*70}\n"
                f"Layer '{lid}'{sup_tag}  "
                f"({layer.grid_h}x{layer.grid_w} columns, "
                f"{layer.columns[0].N_MINI} minicolumns each)"
            )

            for ci, col in enumerate(layer.columns):
                gy, gx = divmod(ci, layer.grid_w)
                lines.append(f"\n  Column ({gy},{gx}):")

                wins_list  = [mc._n_wins for mc in col.minicolumns]
                total_wins = max(sum(wins_list), 1)

                for mi, mc in enumerate(col.minicolumns):
                    wins  = mc._n_wins
                    nlocs = mc.n_locations()
                    frac  = wins / total_wins
                    bar   = '#' * round(frac * bar_width)
                    label = (f"class {mi}" if layer.supervised
                             else f"mini {mi:2d}")

                    if mc._loc_total:
                        sorted_locs = sorted(
                            mc._loc_total.items(),
                            key=lambda kv: kv[1], reverse=True)[:top_locs]
                        loc_strs = []
                        for loc, ltotal in sorted_locs:
                            union   = mc._model[loc]
                            n_bits  = int(union.sum())
                            density = n_bits / max(len(union), 1)
                            loc_strs.append(
                                f"loc{loc}[n={ltotal},"
                                f"bits={n_bits},dens={density:.3f}]")
                        detail = '  '.join(loc_strs)
                    else:
                        detail = '(no model)'

                    lines.append(
                        f"    {label}  {bar:<{bar_width}}  "
                        f"wins={wins:4d}  locs={nlocs:3d}  {detail}"
                    )

        return '\n'.join(lines)
