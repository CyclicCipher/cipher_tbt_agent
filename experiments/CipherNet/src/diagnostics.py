"""CipherNet diagnostic suite — 5 targeted probes for understanding failure modes.

  1. V1 class consistency   — Is CHL producing class-discriminative V1 reps?
  2. IT win distribution    — Is WTA degenerate (rich-get-richer collapse)?
  3. Evidence concentration — Are evidence distributions peaked or flat at test time?
  4. Confusion matrix       — Which classes are confused with which?
  5. Feedback hit rate      — Is IT->V1 feedback accurate or harmful?
"""
from __future__ import annotations

import math
from collections import defaultdict, Counter

import numpy as np


# ---------------------------------------------------------------------------
# Diag 1 — V1 class consistency
# ---------------------------------------------------------------------------

def diag_v1_consistency(cortex, explorer,
                        images: np.ndarray,
                        labels: np.ndarray,
                        n_sample: int = 200) -> None:
    """Free-phase V1 probe: does the tentative winner correlate with the label?

    Runs n_sample images through V1 only (no IT, no feedback).
    For each V1 column, builds a (true_label -> Counter[v1_winner]) table.
    Purity = fraction of images where a column's tentative winner is the
    majority mini for that label.  If purity ~= 0.10 (1/n_mini), CHL is
    doing nothing; higher is better.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1 -- V1 Class Consistency (free-phase probe)")
    print("=" * 60)

    v1_lid   = cortex._order[0]
    v1_layer = cortex.layers[v1_lid]
    n_cols   = v1_layer.n_columns

    # col_label_mini[col_idx][label] = Counter[winner_mini]
    col_label_mini: list[dict[int, Counter]] = [
        defaultdict(Counter) for _ in range(n_cols)
    ]

    n = min(n_sample, len(images))
    for i in range(n):
        img = images[i]
        lbl = int(labels[i])
        fixations = explorer.get_fixations(img)
        prepared  = v1_layer.modality.preprocess(img)

        for col in v1_layer.columns:
            col.begin_image()

        # V1-only observation (no IT, no feedback).
        for fix in fixations:
            feats, locs = cortex._encode_sensor_fixation(v1_layer, prepared, fix)
            for ci, col in enumerate(v1_layer.columns):
                if feats[ci] is not None:
                    col.observe_multi([(feats[ci], locs[ci])])

        for ci, col in enumerate(v1_layer.columns):
            col_label_mini[ci][lbl][col.tentative_winner()] += 1

    # Per-column purity: fraction where the plurality mini wins per label
    purities: list[float] = []
    for ci in range(n_cols):
        total_correct = sum(
            mc.most_common(1)[0][1]
            for mc in col_label_mini[ci].values()
            if mc
        )
        total_all = sum(
            sum(mc.values())
            for mc in col_label_mini[ci].values()
        )
        purities.append(total_correct / max(total_all, 1))

    mean_p  = sum(purities) / len(purities) if purities else 0.0
    best_p  = max(purities) if purities else 0.0
    worst_p = min(purities) if purities else 0.0

    print(f"\n{'Col':>6}  {'Purity':>7}  Bar")
    for ci, p in enumerate(purities):
        gy, gx = divmod(ci, v1_layer.grid_w)
        bar = '#' * round(p * 30)
        print(f"  ({gy},{gx})  {p:.3f}    {bar}")

    distinct_counts = []
    for ci in range(n_cols):
        dominant_minis = {
            col_label_mini[ci][lbl].most_common(1)[0][0]
            for lbl in range(10)
            if col_label_mini[ci].get(lbl)
        }
        distinct_counts.append(len(dominant_minis))
    mean_distinct  = sum(distinct_counts) / len(distinct_counts) if distinct_counts else 0
    worst_distinct = min(distinct_counts) if distinct_counts else 0

    print(f"\nMean purity: {mean_p:.3f}   Best: {best_p:.3f}   Worst: {worst_p:.3f}")
    print(f"Distinct dominant minis per column (out of 10 labels): "
          f"mean={mean_distinct:.1f}  worst={worst_distinct}")

    if mean_p > 0.65 and mean_distinct >= 5:
        verdict = "GOOD -- V1 is class-discriminative; CHL is working"
    elif mean_p > 0.65 and mean_distinct < 3:
        verdict = "BAD -- high purity but WTA monoculture: all labels share same mini"
    elif mean_p > 0.40:
        verdict = "MARGINAL -- partial class discrimination; CHL helping but weak"
    else:
        verdict = "BAD -- near chance; CHL may not be working or n_mini too large"
    print(f"Verdict: {verdict}")

    print("\nLabel -> dominant V1 mini (cols 0..4):")
    for ci in range(min(5, n_cols)):
        gy, gx = divmod(ci, v1_layer.grid_w)
        mapping = {
            lbl: col_label_mini[ci][lbl].most_common(1)[0][0]
            for lbl in range(10)
            if col_label_mini[ci].get(lbl)
        }
        print(f"  Col ({gy},{gx}): {mapping}")


# ---------------------------------------------------------------------------
# Diag 2 — IT win distribution
# ---------------------------------------------------------------------------

def _layer_win_summary(layer, layer_id: str) -> None:
    """Print win-count distribution for one layer."""
    for ci, col in enumerate(layer.columns):
        gy, gx = divmod(ci, layer.grid_w)
        wins   = [mc._n_wins for mc in col.minicolumns]
        total  = max(sum(wins), 1)
        n_mini = col.N_MINI
        n_active = sum(1 for w in wins if w > 0)

        entropy = 0.0
        for w in wins:
            p = w / total
            if p > 0:
                entropy -= p * math.log2(p)
        effective = (2 ** entropy) / n_mini if n_mini > 0 else 0.0

        sorted_wins = sorted(enumerate(wins), key=lambda x: x[1], reverse=True)
        top5 = [(mi, w) for mi, w in sorted_wins[:5] if w > 0]
        top5_str = "  ".join(f"mini{mi}:{w}" for mi, w in top5)

        print(f"  {layer_id} Col ({gy},{gx}):  "
              f"active={n_active}/{n_mini}  "
              f"eff={effective:.3f}  "
              f"top: {top5_str}")

        if effective < 0.1:
            print(f"    *** WTA COLLAPSE -- {n_active} minis absorbing all wins ***")


def diag_it_win_distribution(cortex) -> None:
    """Histogram of _n_wins per IT minicolumn after training.

    Effective utilisation = 2^H / n_mini  (H = entropy of win distribution).
      1.0 = perfectly uniform  |  ~0 = total collapse
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2 -- Win-Count Distribution (all layers)")
    print("=" * 60)

    v1_lid   = cortex._order[0]
    v1_layer = cortex.layers[v1_lid]
    print(f"\nV1 ({v1_layer.grid_h}x{v1_layer.grid_w} columns, "
          f"{v1_layer.columns[0].N_MINI} minis each) -- collapse check:")
    _layer_win_summary(v1_layer, 'V1')

    it_lid   = cortex._order[-1]
    it_layer = cortex.layers[it_lid]

    print(f"\nIT ({it_layer.grid_h}x{it_layer.grid_w} columns, "
          f"{it_layer.columns[0].N_MINI} minis each) -- full histogram:")
    for ci, col in enumerate(it_layer.columns):
        gy, gx = divmod(ci, it_layer.grid_w)
        wins    = [mc._n_wins for mc in col.minicolumns]
        total   = max(sum(wins), 1)
        n_mini  = col.N_MINI

        entropy = 0.0
        for w in wins:
            p = w / total
            if p > 0:
                entropy -= p * math.log2(p)
        effective = (2 ** entropy) / n_mini if n_mini > 0 else 0.0
        n_active  = sum(1 for w in wins if w > 0)

        print(f"\n  IT Column ({gy},{gx}):  "
              f"n_active={n_active}/{n_mini}  "
              f"effective_utilisation={effective:.3f}  "
              f"entropy={entropy:.2f} bits (max {math.log2(n_mini):.2f})")

        sorted_wins = sorted(enumerate(wins), key=lambda x: x[1], reverse=True)
        print(f"  {'mini':>5}  {'wins':>6}  {'%':>6}  bar")
        for mi, w in sorted_wins:
            pct = w / total * 100
            bar = '#' * round(pct / 2)
            print(f"  {mi:>5}  {w:>6}  {pct:>5.1f}%  {bar}")

        if effective > 0.5:
            verdict = "GOOD -- wins are spread; WTA is not collapsing"
        elif effective > 0.25:
            verdict = "MARGINAL -- moderate collapse; consider more training images"
        else:
            verdict = "BAD -- severe WTA collapse; add boosting or soft competition"
        print(f"  Verdict: {verdict}")


# ---------------------------------------------------------------------------
# Diag 3 + 4 — Evidence concentration + confusion matrix (combined pass)
# ---------------------------------------------------------------------------

def diag_evidence_and_confusion(cortex, explorer,
                                 images: np.ndarray,
                                 labels: np.ndarray,
                                 n_sample: int = 200,
                                 confidence: float = 0.6) -> None:
    """Single classify pass collecting two diagnostics:

    Diag 3 — Evidence concentration:
      After each classify(), read IT columns' _evidence vectors.
      Concentration = max(ev) / sum(ev) per column.

    Diag 4 — Confusion matrix.
    """
    it_lid   = cortex._order[-1]
    it_layer = cortex.layers[it_lid]

    n = min(n_sample, len(images))
    concentrations: list[float] = []
    early_stops = 0
    confusion   = np.zeros((10, 10), dtype=int)
    no_assoc    = 0

    for i in range(n):
        img = images[i]
        lbl = int(labels[i])
        fixations = explorer.get_fixations(img)

        pred, votes = cortex.classify(img, fixations,
                                      confidence_threshold=confidence)

        for col in it_layer.columns:
            ev = col._evidence
            if ev:
                s = sum(ev)
                if s > 0:
                    concentrations.append(max(ev) / s)

        total_ev = sum(
            sum(col._evidence) for col in it_layer.columns
            if col._evidence
        )
        max_ev = max(
            (max(col._evidence) for col in it_layer.columns if col._evidence),
            default=0.0
        )
        if total_ev > 0 and max_ev / total_ev >= confidence:
            early_stops += 1

        if pred == -1:
            no_assoc += 1
        else:
            p = max(0, min(9, pred))
            confusion[lbl, p] += 1

    # -------- Print Diag 3 --------
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 3 -- Evidence Concentration at Test Time")
    print("=" * 60)

    if concentrations:
        mean_c  = sum(concentrations) / len(concentrations)
        below20 = sum(1 for c in concentrations if c < 0.2) / len(concentrations)
        above60 = sum(1 for c in concentrations if c >= 0.6) / len(concentrations)

        buckets = [0] * 10
        for c in concentrations:
            b = min(int(c * 10), 9)
            buckets[b] += 1
        total_c = len(concentrations)

        print(f"\nMean concentration: {mean_c:.3f}"
              f"   <0.2 (flat): {below20:.1%}"
              f"   >=0.6 (peaked): {above60:.1%}")
        print(f"Early-stop rate: {early_stops/n:.1%}  ({early_stops}/{n} images)")
        print("\nConcentration histogram (max_ev / sum_ev per IT column):")
        print(f"  {'Range':>12}  {'Count':>6}  bar")
        for b in range(10):
            lo, hi = b * 0.1, (b + 1) * 0.1
            bar = '#' * round(buckets[b] / total_c * 40)
            print(f"  [{lo:.1f}, {hi:.1f})  {buckets[b]:>6}  {bar}")

        if mean_c > 0.5:
            verdict3 = "GOOD -- WTA is producing peaked evidence distributions"
        elif mean_c > 0.3:
            verdict3 = "MARGINAL -- moderate concentration; some WTA ambiguity"
        else:
            verdict3 = "BAD -- flat evidence; WTA is not converging to a winner"
        print(f"Verdict: {verdict3}")
    else:
        print("  (no evidence data collected)")

    # -------- Print Diag 4 --------
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 4 -- Per-Class Accuracy & Confusion Matrix")
    print("=" * 60)

    total_classified = confusion.sum()
    total_all        = total_classified + no_assoc
    overall_acc      = confusion.diagonal().sum() / max(total_all, 1)

    print(f"\nOverall accuracy: {overall_acc:.1%}  "
          f"({confusion.diagonal().sum()}/{total_all})   "
          f"no-assoc: {no_assoc}/{total_all} = {no_assoc/max(total_all,1):.1%}")

    print("\nPer-class recall:")
    print(f"  {'Class':>6}  {'Correct':>8}  {'Total':>6}  {'Recall':>7}  Top confusion")
    for lbl in range(10):
        row       = confusion[lbl]
        row_total = row.sum()
        correct   = confusion[lbl, lbl]
        recall    = correct / max(row_total, 1)
        row_copy  = row.copy()
        row_copy[lbl] = 0
        top_conf_cls  = int(np.argmax(row_copy))
        top_conf_cnt  = row_copy[top_conf_cls]
        conf_str = f"->{top_conf_cls} ({top_conf_cnt})" if top_conf_cnt > 0 else "--"
        bar = '#' * round(recall * 20)
        print(f"  {lbl:>6}  {correct:>8}  {row_total:>6}  {recall:>6.1%}  {conf_str:>12}  {bar}")

    print("\nConfusion matrix (row=true, col=predicted):")
    header = "     " + "".join(f"{c:>5}" for c in range(10))
    print(header)
    for lbl in range(10):
        row_str = "".join(
            f"\033[1m{confusion[lbl,c]:>5}\033[0m" if c == lbl
            else f"{confusion[lbl, c]:>5}"
            for c in range(10)
        )
        print(f"  {lbl}  {row_str}")


# ---------------------------------------------------------------------------
# Diag 5 — Feedback hit rate
# ---------------------------------------------------------------------------

def diag_feedback_hit_rate(cortex, explorer,
                            images: np.ndarray,
                            labels: np.ndarray,
                            n_sample: int = 200) -> None:
    """IT->V1 feedback accuracy probe.

    For each fixation, compares IT winner's stored SDR at each RF slot
    (what IT predicts V1 will see) against V1's actual SDR from that fixation.

    hit      = overlap >= 0.5 between predicted and actual V1 SDR
    miss     = overlap < 0.5
    no_pred  = IT has no model at that location yet

    A hit rate above chance (1/n_mini_v1) means feedback is informative.
    """
    print("\n" + "=" * 60)
    print("DIAGNOSTIC 5 -- IT->V1 Feedback Hit Rate")
    print("=" * 60)

    v1_lid    = cortex._order[0]
    it_lid    = cortex._order[-1]
    v1_layer  = cortex.layers[v1_lid]
    it_layer  = cortex.layers[it_lid]
    n_mini_v1 = v1_layer.columns[0].N_MINI if v1_layer.columns else 10
    chance    = 1.0 / n_mini_v1

    hits     = 0
    misses   = 0
    no_preds = 0

    n = min(n_sample, len(images))
    for i in range(n):
        img = images[i]
        fixations = explorer.get_fixations(img)
        prepared  = {
            lid: cortex.layers[lid].modality.preprocess(img)
            for lid in cortex._order
            if cortex.layers[lid].modality is not None
        }

        for lid in cortex._order:
            for col in cortex.layers[lid].columns:
                col.begin_image()

        for fix in fixations:
            # Full forward pass via the shared helper.
            layer_outputs, layer_tentatives = cortex._forward_sweep(prepared, fix)

            v1_feats  = layer_outputs[v1_lid][0]
            v1_locs   = layer_outputs[v1_lid][1]
            it_winners = layer_tentatives[it_lid]

            # Check IT predictions against actual V1 SDRs.
            for ui, u_col in enumerate(it_layer.columns):
                winner_mc = u_col.minicolumns[it_winners[ui]]
                rf        = it_layer.receptive_fields[ui]
                for j, lower_col_idx in enumerate(rf):
                    loc      = v1_locs[lower_col_idx]
                    pred_sdr = winner_mc._model.get(loc)
                    if pred_sdr is None:
                        no_preds += 1
                        continue
                    actual_sdr = (v1_feats[lower_col_idx]
                                  if lower_col_idx < len(v1_feats) else None)
                    if actual_sdr is None:
                        no_preds += 1
                        continue
                    n_active = int(actual_sdr.sum())
                    if n_active == 0:
                        no_preds += 1
                        continue
                    overlap = (int(np.bitwise_and(pred_sdr, actual_sdr).sum())
                               / n_active)
                    if overlap >= 0.5:
                        hits += 1
                    else:
                        misses += 1

    total_preds = hits + misses
    hit_rate  = hits   / max(total_preds, 1)
    miss_rate = misses / max(total_preds, 1)
    coverage  = total_preds / max(total_preds + no_preds, 1)

    print(f"\nOver {n} images:")
    print(f"  Total feedback slots:    {total_preds + no_preds}")
    print(f"  IT has model (coverage): {coverage:.1%}   "
          f"({total_preds} / {total_preds + no_preds})")
    print(f"  Hit rate:   {hit_rate:.3f}  ({hits})")
    print(f"  Miss rate:  {miss_rate:.3f}  ({misses})")
    print(f"  Chance:     {chance:.3f}  (1 / {n_mini_v1} V1 minis)")

    if total_preds == 0:
        verdict = "INCONCLUSIVE -- IT has no model at feedback slots (too little training)"
    elif hit_rate > chance * 2:
        verdict = "GOOD -- IT predictions are better than chance; feedback is informative"
    elif hit_rate > chance:
        verdict = "MARGINAL -- IT predictions slightly above chance; weak feedback signal"
    else:
        verdict = ("BAD -- hit rate at or below chance; feedback is noise/harmful. "
                   "Consider reducing feedback_bonus or training more images.")
    print(f"Verdict: {verdict}")

    if total_preds > 0:
        net_signal = hit_rate - miss_rate
        print(f"\nNet signal (hit% - miss%): {net_signal:+.3f}")
        if net_signal < 0:
            print("  WARNING: misses outnumber hits -- feedback may be hurting V1.")


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def run_diagnostics(cortex, explorer,
                    train_images: np.ndarray, train_labels: np.ndarray,
                    test_images:  np.ndarray, test_labels:  np.ndarray,
                    n_sample: int = 200,
                    confidence: float = 0.6) -> None:
    """Run all 5 diagnostics and print a structured report."""
    print("\n" + "#" * 60)
    print("  CIPHERNET DIAGNOSTIC REPORT")
    print("#" * 60)

    diag_v1_consistency(
        cortex, explorer,
        train_images, train_labels,
        n_sample=n_sample)

    diag_it_win_distribution(cortex)

    diag_evidence_and_confusion(
        cortex, explorer,
        test_images, test_labels,
        n_sample=n_sample,
        confidence=confidence)

    diag_feedback_hit_rate(
        cortex, explorer,
        train_images, train_labels,
        n_sample=min(n_sample, 100))

    print("\n" + "#" * 60)
    print("  END OF DIAGNOSTIC REPORT")
    print("#" * 60 + "\n")
