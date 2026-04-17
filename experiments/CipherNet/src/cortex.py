"""Hierarchical cortex — V1 → V2 → IT, built from config.

ARCHITECTURE
------------
  Cortex
    └─ Layer  (ordered list, sensor → higher)
         └─ MacroColumn × (grid_h × grid_w)
               └─ MiniColumn × N_MINI

OBSERVATION MODEL (per layer type)
------------------------------------
Sensor layers (V1):
  One HOG feature per fixation per column.
    observe(HOG_code, object_relative_loc)

Non-sensor layers (V2, IT):
  ONE INDEPENDENT OBSERVATION PER LOWER-LAYER COLUMN in the RF.
  Feature = str(lower_winner_index).
  Location = local RF position (vy, vx) within the 2D RF grid.
  This gives the column a distribution over features at EACH RF
  slot, mirroring how biology builds V2 complex cells from V1.

  With V2 (2×2 grid, 3×3 RF per column, 9 V1 cols):
    9 observations per fixation per V2 column.
    Model[loc=(ly,lx)] = distribution of V1 winner codes at slot (ly,lx).

  With IT (1×1 grid, all V2 cols in RF):
    4 observations per fixation per IT column.
    Model[loc=(jy,jx)] = distribution of V2 winner codes at slot (jy,jx).
    IT is SUPERVISED: commit_supervised(label) forces correct minicolumn.

WHY INDEPENDENT OBSERVATIONS MATTER
-------------------------------------
A single concatenated feature (str(all_winners_tuple)) requires an EXACT
match of ALL lower-layer winners to produce nonzero overlap.  With 36 V1
columns, even one different winner means zero score → classification fails.

Independent observations let each RF slot contribute partial evidence
independently.  The aggregate is a sum of per-slot probabilities — robust
to variation in any individual slot.  This is the mechanism that made the
original per-label model work at 60%.

RECEPTIVE FIELDS (auto-computed from grid sizes)
-------------------------------------------------
For upper layer (Uh × Uw) receiving from lower (Lh × Lw):

  centre in lower coords:
    cx = gx * (Lw-1)/(Uw-1)   [or (Lw-1)/2 if Uw==1]
    cy = gy * (Lh-1)/(Uh-1)   [or (Lh-1)/2 if Uh==1]
  radius:
    rf_half_w = Lw / Uw,   rf_half_h = Lh / Uh
  field = columns where |vx-cx| ≤ rf_half_w and |vy-cy| ≤ rf_half_h

Local (vy, vx) within the RF is normalised by the minimum indices so
it always starts at (0,0) regardless of which part of the lower grid
this upper column covers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reference_frames import make_frame, RetinotopicFrame, ReferenceFrame
from column import MacroColumn
from cortical_message import CorticalMessage
from output_cortex import OutputCortex


CONFIDENCE_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Encoder builders
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
    input_source:     str              # 'sensor' | lower-layer id
    encoder:          Any | None       # non-None only for sensor layers
    patch_size:       int = 5
    stride:           int = 3
    supervised:       bool  = False
    chl:              bool  = False    # Contrastive Hebbian Learning at this layer
    lateral_bonus:    float = 0.0     # evidence bonus from grid-adjacent column winners
    # receptive_fields[i]  = flat lower-layer column indices for column i
    receptive_fields: list[list[int]]               = field(default_factory=list)
    # rf_local_pos[i][j]   = (vy, vx) local position of RF slot j for column i
    rf_local_pos:     list[list[tuple[int, int]]]   = field(default_factory=list)
    # last_features[i] = HOG SDR (np.ndarray int8) or None (blank patch).
    # Only populated for sensor layers; used by higher layers in direct-HOG mode.
    last_features:    list                          = field(default_factory=list)

    @property
    def n_columns(self) -> int:
        return len(self.columns)

    def stats(self) -> dict:
        used       = sum(mc.stats()['used_mini']       for mc in self.columns)
        total_locs = sum(mc.stats()['total_locations'] for mc in self.columns)
        return {
            'id':             self.id,
            'n_macrocolumns': self.n_columns,
            'n_mini':         self.columns[0].N_MINI if self.columns else 0,
            'supervised':     self.supervised,
            'used_mini':      used,
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
                 ordered_ids: list[str],
                 eye=None):
        self.name           = name
        self.layers         = layers
        self._order         = ordered_ids
        self._eye           = eye
        self._output_cortex = OutputCortex()

    # ------------------------------------------------------------------
    # Construction from config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict, eye=None) -> 'Cortex':
        name = config.get('name', 'cortex')
        layers:      dict[str, Layer] = {}
        ordered_ids: list[str]        = []

        for lcfg in config.get('layers', []):
            lid          = lcfg['id']
            grid_h, grid_w = lcfg['grid']
            frame_cfg    = lcfg['reference_frame']
            frame_type   = frame_cfg['type']
            frame_params = frame_cfg.get('params', {})
            n_mini       = lcfg.get('n_mini', 10)
            supervised    = lcfg.get('supervised', False)
            chl           = lcfg.get('chl', False)
            lateral_bonus = lcfg.get('lateral_bonus', 0.0)
            input_cfg     = lcfg.get('input', {})
            input_source = input_cfg.get('source', 'sensor')

            patch_size   = frame_params.get('patch_size', 5)
            stride       = frame_params.get('stride', 3)

            # Encoder for sensor layers
            encoder = None
            if input_source == 'sensor':
                enc_params = dict(input_cfg.get('encoder_params', {}))
                enc_params.setdefault('patch_size', patch_size)
                encoder = _build_encoder(
                    input_cfg.get('encoder', 'hog'), enc_params)

            miss_penalty = lcfg.get('miss_penalty', 0.0)

            # MacroColumns
            columns: list[MacroColumn] = []
            for gy in range(grid_h):
                for gx in range(grid_w):
                    frame = cls._make_frame(frame_type, frame_params,
                                            gx, gy, eye)
                    columns.append(MacroColumn(frame, n_mini=n_mini,
                                               miss_penalty=miss_penalty))

            # Receptive fields with local positions (non-sensor only)
            rf_indices: list[list[int]]             = []
            rf_lpos:    list[list[tuple[int, int]]] = []
            if input_source != 'sensor' and input_source in layers:
                lower = layers[input_source]
                rf_indices, rf_lpos = cls._compute_rf(
                    lower.grid_h, lower.grid_w, grid_h, grid_w)

            layers[lid] = Layer(
                id=lid, grid_h=grid_h, grid_w=grid_w,
                columns=columns,
                input_source=input_source,
                encoder=encoder,
                patch_size=patch_size, stride=stride,
                supervised=supervised,
                chl=chl,
                lateral_bonus=lateral_bonus,
                receptive_fields=rf_indices,
                rf_local_pos=rf_lpos,
            )
            ordered_ids.append(lid)

        return cls(name, layers, ordered_ids, eye)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_frame(frame_type: str, params: dict,
                    gx: int, gy: int, eye) -> ReferenceFrame:
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
                    upper_h: int, upper_w: int,
                    ) -> tuple[list[list[int]],
                               list[list[tuple[int, int]]]]:
        """Receptive fields and local positions for upper-over-lower layers.

        Returns:
            rf_indices[i]   — flat lower column indices for upper column i
            rf_local_pos[i] — (vy, vx) local position per RF slot (normalised
                              to start at (0,0) within each column's RF)
        """
        rf_half_w = lower_w / upper_w
        rf_half_h = lower_h / upper_h
        all_indices: list[list[int]]             = []
        all_lpos:    list[list[tuple[int, int]]] = []

        for gy in range(upper_h):
            for gx in range(upper_w):
                cx = (gx * (lower_w - 1) / (upper_w - 1)
                      if upper_w > 1 else (lower_w - 1) / 2.0)
                cy = (gy * (lower_h - 1) / (upper_h - 1)
                      if upper_h > 1 else (lower_h - 1) / 2.0)

                flat: list[int]             = []
                apos: list[tuple[int, int]] = []
                for vy in range(lower_h):
                    for vx in range(lower_w):
                        if (abs(vx - cx) <= rf_half_w
                                and abs(vy - cy) <= rf_half_h):
                            flat.append(vy * lower_w + vx)
                            apos.append((vy, vx))

                # Normalise to start at (0,0)
                if apos:
                    min_vy = min(p[0] for p in apos)
                    min_vx = min(p[1] for p in apos)
                    lpos = [(p[0] - min_vy, p[1] - min_vx) for p in apos]
                else:
                    lpos = []

                all_indices.append(flat)
                all_lpos.append(lpos)

        return all_indices, all_lpos

    # ------------------------------------------------------------------
    # Lateral inhibition
    # ------------------------------------------------------------------

    @staticmethod
    def _lateral_pass(layer: Layer) -> None:
        """Apply lateral consistency pressure from grid-adjacent columns.

        After each fixation's observe step, each column receives a small
        evidence bonus (layer.lateral_bonus) for every minicolumn that is
        the current tentative winner of an adjacent column.  This implements
        L2/3-style WTA consistency: if neighbours agree on a class, that
        class gets a nudge in the current column.

        Winners are snapshotted BEFORE applying bonuses so the result is
        order-independent (no column benefits from lateral input it helped
        create in the same pass).
        """
        if layer.lateral_bonus <= 0.0:
            return
        h, w = layer.grid_h, layer.grid_w
        # Snapshot tentative winners before any bonus is applied.
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
    def _global_lateral_pass(layer: Layer,
                              bonus: float = 0.05) -> None:
        """Apply full-layer plurality vote as a weak global prior.

        After the local ±1 lateral pass, every column in the layer receives
        a small bonus for whichever minicolumn index is the most common
        tentative winner across ALL columns in the layer.  This implements
        the long-range horizontal white-matter connections that the ±1
        neighbourhood cannot reach.

        The bonus is intentionally smaller than lateral_bonus so the global
        signal guides without overriding local evidence.
        """
        if not layer.columns:
            return
        winners = [col.tentative_winner() for col in layer.columns]
        if not winners:
            return
        plurality = max(set(winners), key=winners.count)
        for col in layer.columns:
            col.apply_lateral_input([plurality], bonus)

    # ------------------------------------------------------------------
    # Sensor encoding (V1)
    # ------------------------------------------------------------------

    def _feedback_pass(self, layer_tentatives: dict[str, list[int]],
                       feedback_bonus: float = 0.3) -> None:
        """Top-down prediction sweep after each fixation's forward pass.

        Iterates layer pairs upper → lower (IT→V1).  The upper winner
        minicolumn's stored SDR union at each RF slot is the predicted
        feature for the corresponding lower column.  For multi-mini lower
        layers, the lower minicolumn whose model best matches that predicted
        SDR receives an evidence boost — biasing WTA toward the upper
        layer's expectation before the next fixation.

        With the current V1 n_mini=1, this is a no-op (no competition to
        bias).  Retained for future architectures with learned V1 WTA.
        """
        # Feedback path: upper winner's predicted SDR at each RF slot primes
        # the corresponding lower column.  With direct-HOG mode (V1 n_mini=1),
        # lower columns have only one possible mini — feedback is a no-op.
        # This loop is retained for future multi-mini lower layers.
        for upper_idx in range(len(self._order) - 1, 0, -1):
            upper_id    = self._order[upper_idx]
            lower_id    = self._order[upper_idx - 1]
            upper_layer = self.layers[upper_id]
            lower_layer = self.layers[lower_id]
            upper_winners = layer_tentatives[upper_id]

            for ui, u_col in enumerate(upper_layer.columns):
                winner_mc = u_col.minicolumns[upper_winners[ui]]
                rf        = upper_layer.receptive_fields[ui]
                lpos      = upper_layer.rf_local_pos[ui]
                for j, lower_col_idx in enumerate(rf):
                    loc           = lpos[j]
                    predicted_sdr = winner_mc._model.get(loc)
                    if predicted_sdr is None:
                        continue
                    # For multi-mini lower layers: find the mini whose model
                    # best matches the predicted SDR and boost it.
                    lower_col = lower_layer.columns[lower_col_idx]
                    if lower_col.N_MINI == 1:
                        continue   # no competition to bias
                    scores = [mc.overlap_score(predicted_sdr, loc)
                              for mc in lower_col.minicolumns]
                    best_mini = int(np.argmax(scores))
                    lower_col.receive_feedback_by_index(best_mini, feedback_bonus)

    def _encode_sensor_fixation(self, layer: Layer,
                                 dog: np.ndarray,
                                 fixation: tuple,
                                 ) -> tuple[list[np.ndarray | None], list[tuple]]:
        """HOG encode one fixation for a sensor layer.

        Returns features as list[np.ndarray | None].
        None = blank/silent patch — suppressed structurally by the encoder
        (HOGEncoder returns None when mean gradient magnitude is below its
        ACTIVITY_THRESHOLD).  None features are skipped by higher-layer
        observation building so blank background never reaches IT.
        """
        self._eye.fixate(float(fixation[0]), float(fixation[1]))
        retina = self._eye.sample(dog)
        ps, st = layer.patch_size, layer.stride

        patches = np.empty((layer.n_columns, ps, ps), dtype=np.float32)
        k = 0
        for gy in range(layer.grid_h):
            y0 = gy * st
            for gx in range(layer.grid_w):
                x0 = gx * st
                patches[k] = retina[y0:y0 + ps, x0:x0 + ps]
                k += 1

        # Encoder returns list[np.ndarray | None] — None for blank patches.
        features: list[np.ndarray | None] = layer.encoder.encode_batch(patches)
        layer.last_features = features

        locations: list[tuple] = []
        for col in layer.columns:
            col.frame.set_position(fixation)
            locations.append(col.frame.position_key())

        return features, locations

    # ------------------------------------------------------------------
    # Higher-layer observations
    # ------------------------------------------------------------------

    @staticmethod
    def _build_rf_observations(
            lower_winners:    list[int],
            receptive_fields: list[list[int]],
            rf_local_pos:     list[list[tuple[int, int]]],
            col_idx:          int,
            lower_features:   list | None = None,
    ) -> list[tuple[np.ndarray, tuple]]:
        """Build independent (sdr, loc) observations for one upper column.

        If lower_features is provided (lower layer is a sensor), each feature
        is an SDR np.ndarray — the biologically correct input.  None entries
        (blank/silent patches) are filtered out here so observe_multi() never
        sees a zero-energy feature.

        Otherwise (lower layer is a non-sensor with learned WTA winners),
        fall back to winner-index SDRs — a one-hot int8 vector of length
        N_MINI encoding which minicolumn won in the lower layer.
        """
        rf   = receptive_fields[col_idx]
        lpos = rf_local_pos[col_idx]
        if lower_features is not None:
            return [(lower_features[j], lp)
                    for j, lp in zip(rf, lpos)
                    if lower_features[j] is not None]
        # Winner-index fallback: encode as one-hot SDR so MiniColumn.overlap_score
        # still receives an np.ndarray.  This path is used when the lower layer
        # is a non-sensor (learned WTA) — not active in the current V1→IT config.
        return [(np.array([1], dtype=np.int8), lp) for j, lp in zip(rf, lpos)]

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def learn(self, image: np.ndarray, label: int,
              fixations: list[tuple]) -> None:
        """TBT evidence accumulation — fully unsupervised columns.

        Per fixation, bottom-up:
          V1: observe(HOG_feat, retinal_loc)      — single observation
          IT: observe_multi(v1_winner per RF slot) — n_rf obs per column

        After each fixation:
          _feedback_pass   — IT→V1 top-down priming for the next fixation
          _lateral_pass    — ±1 neighbour lateral consistency
          _global_lateral_pass — full-layer plurality bonus

        End of image:
          All layers: commit(write=True)  — unsupervised WTA
          OutputCortex.learn(IT winners, label) — Hebbian label readout
        """
        dog = self._eye.preprocess(image)

        for lid in self._order:
            for col in self.layers[lid].columns:
                col.begin_image()

        for fix in fixations:
            layer_tentative: dict[str, list[int]] = {}

            for lid in self._order:
                layer = self.layers[lid]

                if layer.input_source == 'sensor':
                    feats, locs = self._encode_sensor_fixation(
                        layer, dog, fix)
                    for i, col in enumerate(layer.columns):
                        if feats[i] is not None:   # skip silent/blank patches
                            col.observe(feats[i], locs[i])
                else:
                    lower_id      = layer.input_source
                    lower_layer   = self.layers[lower_id]
                    lower_winners = layer_tentative[lower_id]
                    # Use HOG codes directly when the lower layer is a sensor
                    # (V1 = fixed feature extractor; IT = learning layer).
                    direct_feats = (lower_layer.last_features
                                    if lower_layer.encoder is not None else None)
                    for i, col in enumerate(layer.columns):
                        obs = self._build_rf_observations(
                            lower_winners,
                            layer.receptive_fields,
                            layer.rf_local_pos, i,
                            lower_features=direct_feats)
                        if obs:   # skip if all RF slots were blank
                            col.observe_multi(obs)

                self._lateral_pass(layer)
                self._global_lateral_pass(layer)

                layer_tentative[lid] = [
                    col.tentative_winner() for col in layer.columns
                ]

            # After the full forward sweep, send top-down predictions back
            # (IT→V1): each upper layer's winner primes lower layers for
            # the next fixation.
            self._feedback_pass(layer_tentative)

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

        # Associate the final layer's unsupervised winners with the label
        # via OutputCortex (Hebbian readout, no labels inside columns).
        final_layer = self.layers[self._order[-1]]
        for ci, col in enumerate(final_layer.columns):
            self._output_cortex.learn(ci, col.sdr(), label)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Inference helpers
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
                         v1_cache: dict[tuple, tuple],
                         ) -> tuple:
        """Return the remaining fixation that best discriminates IT leader
        from runner-up, based on expected evidence from estimated V1 winners.
        """
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

            # Estimate V1 tentative winners based on marginal overlap only
            # (approximation — ignores accumulated evidence; fast and unbiased)
            v1_est: list[int] = []
            for ci, col in enumerate(v1_layer.columns):
                if feats[ci] is None:
                    # Blank patch: no signal — default to mini 0 (only one for V1).
                    v1_est.append(0)
                else:
                    scores = [mc.overlap_score(feats[ci], locs[ci]) + mc._boost
                              for mc in col.minicolumns]
                    v1_est.append(int(np.argmax(scores)))

            # Expected discriminability across all IT columns
            disc = 0.0
            for it_ci, it_col in enumerate(it_layer.columns):
                obs = self._build_rf_observations(
                    v1_est, it_layer.receptive_fields,
                    it_layer.rf_local_pos, it_ci)
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

    def classify(self, image: np.ndarray,
                 fixations: list[tuple],
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 **_kwargs,
                 ) -> tuple[int, dict]:
        """Classify via TBT evidence accumulation with guided fixation.

        First 2 fixations follow the pre-computed sequence (centroid + first
        offset).  Remaining fixations are chosen greedily: the candidate that
        best discriminates the current IT leader from the runner-up.

        Multiple IT columns vote by summing their evidence.  Early stopping
        once the leading class accounts for ≥ confidence_threshold of total
        aggregated evidence (after ≥2 fixations).
        """
        from collections import Counter
        dog = self._eye.preprocess(image)

        for lid in self._order:
            for col in self.layers[lid].columns:
                col.begin_image()

        it_layer   = self.layers[self._order[-1]]
        v1_lid     = self._order[0]
        v1_layer   = self.layers[v1_lid]

        # Pre-cache V1 encodings for ALL candidate fixations so guided
        # fixation can score them without re-encoding.
        v1_cache: dict[tuple, tuple] = {}
        if v1_layer.input_source == 'sensor':
            for fix in fixations:
                v1_cache[fix] = self._encode_sensor_fixation(
                    v1_layer, dog, fix)

        remaining = list(fixations)

        for k in range(len(fixations)):
            # Select next fixation
            if k < 2 or not remaining:
                fix = remaining.pop(0)
            else:
                fix = self._guided_fixation(
                    it_layer, v1_layer, remaining, v1_cache)
                remaining.remove(fix)

            # Process bottom-up
            layer_tentative: dict[str, list[int]] = {}
            for lid in self._order:
                layer = self.layers[lid]

                if layer.input_source == 'sensor':
                    feats, locs = v1_cache.get(fix) or \
                        self._encode_sensor_fixation(layer, dog, fix)
                    for i, col in enumerate(layer.columns):
                        if feats[i] is not None:   # skip silent/blank patches
                            col.observe(feats[i], locs[i])
                else:
                    lower_id      = layer.input_source
                    lower_layer   = self.layers[lower_id]
                    lower_winners = layer_tentative[lower_id]
                    direct_feats  = (lower_layer.last_features
                                     if lower_layer.encoder is not None else None)
                    for i, col in enumerate(layer.columns):
                        obs = self._build_rf_observations(
                            lower_winners,
                            layer.receptive_fields,
                            layer.rf_local_pos, i,
                            lower_features=direct_feats)
                        if obs:   # skip if all RF slots were blank
                            col.observe_multi(obs)

                self._lateral_pass(layer)
                self._global_lateral_pass(layer)

                layer_tentative[lid] = [
                    col.tentative_winner() for col in layer.columns
                ]

            # Top-down feedback: IT→V1 priming for the next fixation.
            self._feedback_pass(layer_tentative)

            # Early stopping: dominant IT minicolumn (evidence ratio).
            # Uses OutputCortex to map the unsupervised winner to a label.
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
        return pred, votes  # pred == -1 means no OutputCortex associations yet

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def hierarchy_stats(self) -> list[dict]:
        return [self.layers[lid].stats() for lid in self._order]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnose(self, top_feats: int = 3, top_locs: int = 3) -> str:
        """Full representation report for every layer, column, and minicolumn.

        For each layer → each column → each minicolumn:
          wins   — how many times this minicolumn committed (won WTA)
          locs   — number of unique (feat, loc) location keys learned
          top    — for the top `top_locs` locations: the `top_feats` most
                   common features and their probabilities

        IT (supervised) labels each minicolumn with its digit class.
        V1/V2 minicolumns are labelled by index only (unsupervised).
        """
        lines: list[str] = []
        bar_width = 20

        for lid in self._order:
            layer = self.layers[lid]
            sup_tag = ' [supervised]' if layer.supervised else ''
            lines.append(
                f"\n{'='*70}\n"
                f"Layer '{lid}'{sup_tag}  "
                f"({layer.grid_h}×{layer.grid_w} columns, "
                f"{layer.columns[0].N_MINI} minicolumns each)"
            )

            for ci, col in enumerate(layer.columns):
                gy, gx = divmod(ci, layer.grid_w)
                lines.append(f"\n  Column ({gy},{gx}):")

                wins_list = [mc._n_wins for mc in col.minicolumns]
                total_wins = max(sum(wins_list), 1)

                for mi, mc in enumerate(col.minicolumns):
                    wins  = mc._n_wins
                    nlocs = mc.n_locations()
                    frac  = wins / total_wins
                    bar   = '#' * round(frac * bar_width)
                    label = f"class {mi}" if layer.supervised else f"mini {mi:2d}"

                    # Top locations by observation count
                    if mc._loc_total:
                        sorted_locs = sorted(
                            mc._loc_total.items(), key=lambda kv: kv[1],
                            reverse=True)[:top_locs]
                        loc_strs = []
                        for loc, ltotal in sorted_locs:
                            top = mc._model[loc].most_common(top_feats)
                            feat_str = '  '.join(
                                f"'{f}':{cnt/ltotal:.2f}" for f, cnt in top)
                            loc_strs.append(f"loc{loc}[{feat_str}]")
                        detail = '  '.join(loc_strs)
                    else:
                        detail = '(no model)'

                    lines.append(
                        f"    {label}  {bar:<{bar_width}}  "
                        f"wins={wins:4d}  locs={nlocs:3d}  {detail}"
                    )

        return '\n'.join(lines)
