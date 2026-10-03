"""Agreement of an ink-prediction image with binary labels, computed exactly from 256-bin histograms.

Scores are uint8 (0..255): a rendered ink prediction, or a probability quantised to 256 levels. Ties are
grouped per score level, as scikit-learn does, so AP and ROC-AUC here equal
`sklearn.metrics.average_precision_score` / `roc_auc_score` on the same pixels (tests check to 1e-9).

Everything is computed on a *domain* mask: the pixels where the surface exists, the same for every arm.
"""

from __future__ import annotations

import numpy as np


def metrics_from_hist(pos: np.ndarray, neg: np.ndarray) -> dict:
    """AP, ROC-AUC and best F1 from per-score-level counts of positive and negative pixels."""
    tp = np.cumsum(np.asarray(pos)[::-1].astype(np.float64))
    fp = np.cumsum(np.asarray(neg)[::-1].astype(np.float64))
    P, N = tp[-1], fp[-1]
    if P == 0 or N == 0:
        return {"ap": float("nan"), "auc": float("nan"), "best_f1": float("nan")}
    keep = (tp + fp) > 0
    tp, fp = tp[keep], fp[keep]
    precision, recall = tp / (tp + fp), tp / P
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    tpr = np.concatenate([[0.0], recall])
    fpr = np.concatenate([[0.0], fp / N])
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"ap": ap, "auc": auc, "best_f1": float(f1.max())}


def block_hists(score: np.ndarray, ink: np.ndarray, dom: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray]:
    """(n_blocks, 256) histograms of `score` over positive and negative domain pixels, per block x block tile.

    `ink` is the boolean label. Accumulated one row of blocks at a time, so a large segment fits in memory.
    """
    if score.dtype != np.uint8:
        raise TypeError(f"score must be uint8, got {score.dtype}")
    if not (score.shape == ink.shape == dom.shape):
        raise ValueError(f"shape mismatch: score {score.shape}, ink {ink.shape}, domain {dom.shape}")
    H, W = score.shape
    nby, nbx = -(-H // block), -(-W // block)
    pos = np.zeros((nby * nbx, 256), np.int64)
    neg = np.zeros((nby * nbx, 256), np.int64)
    cols = np.arange(W) // block
    for by in range(nby):
        sl = slice(by * block, min(H, (by + 1) * block))
        s, k, d = score[sl], ink[sl], dom[sl]
        idx = (by * nbx + np.broadcast_to(cols, s.shape)) * 256 + s.astype(np.int64)
        pos += np.bincount(idx[d & k], minlength=nby * nbx * 256).reshape(-1, 256)
        neg += np.bincount(idx[d & ~k], minlength=nby * nbx * 256).reshape(-1, 256)
    return pos, neg


def paired_bootstrap(
    ha: tuple[np.ndarray, np.ndarray],
    hb: tuple[np.ndarray, np.ndarray],
    n: int,
    rng: np.random.Generator,
) -> dict:
    """95% intervals for AP(B) - AP(A) and AUC(B) - AUC(A), resampling occupied blocks with replacement.

    The same blocks are drawn for both arms (paired), so shared structure cancels.
    """
    (pa, na), (pb, nb_) = ha, hb
    nblocks = pa.shape[0]
    occupied = np.flatnonzero((pa + na).sum(1) > 0)
    d_ap, d_auc = [], []
    for _ in range(n):
        w = np.bincount(rng.choice(occupied, size=len(occupied), replace=True), minlength=nblocks)
        a, b = metrics_from_hist(w @ pa, w @ na), metrics_from_hist(w @ pb, w @ nb_)
        d_ap.append(b["ap"] - a["ap"])
        d_auc.append(b["auc"] - a["auc"])

    def ci(v: list[float]) -> list[float]:
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]

    return {"d_ap_ci": ci(d_ap), "d_auc_ci": ci(d_auc), "n_blocks": int(len(occupied)), "n_boot": n}


def alignment_gate(
    score: np.ndarray,
    ink: np.ndarray,
    dom: np.ndarray,
    shift: int = 2,
    tol: int = 1,
    window: int | None = None,
    min_gain: float = 0.0,
) -> dict:
    """Where does agreement with the labels peak? AUC of `score` against labels shifted by (dy, dx).

    `score[y, x]` is compared with `ink[y + dy, x + dx]` for dy, dx in [-shift, shift]. The peak must lie
    within `tol` px of (0, 0) for the frames to count as aligned. If no shift has a defined AUC (no
    labelled ink, or no surface, in the evaluated region) the status is "undetermined", never
    "misaligned". `window` limits the check to a central window of that size, or the whole domain if None.

    An offset only counts as evidence if it beats (0, 0) by at least `min_gain` in AUC. On a weak or smooth
    signal the AUC surface is flat and its maximum lands anywhere, often on the edge of the search range;
    that is "aligned (flat)", not "misaligned". `at_edge` flags a material peak on the search boundary, where
    the true offset may be larger than the range searched.
    """
    H, W = score.shape
    if window is None:
        h, w = H - 2 * shift, W - 2 * shift
    else:
        h, w = min(H - 2 * shift, window), min(W - 2 * shift, window)
    if h <= 0 or w <= 0:
        return {"status": "undetermined", "passes": False, "reason": "image smaller than the shift range"}
    y0, x0 = (H - h) // 2, (W - w) // 2
    s, d = score[y0 : y0 + h, x0 : x0 + w], dom[y0 : y0 + h, x0 : x0 + w]
    auc = {}
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            k = ink[y0 + dy : y0 + dy + h, x0 + dx : x0 + dx + w]
            p = np.bincount(s[d & k].ravel(), minlength=256)
            n = np.bincount(s[d & ~k].ravel(), minlength=256)
            auc[(dy, dx)] = metrics_from_hist(p, n)["auc"]
    finite = {key: v for key, v in auc.items() if np.isfinite(v)}
    if not finite:
        return {"status": "undetermined", "passes": False, "reason": "no defined AUC at any shift"}
    (py, px), best = max(finite.items(), key=lambda kv: kv[1])
    gain = best - auc[(0, 0)] if np.isfinite(auc[(0, 0)]) else float("inf")
    near = abs(py) <= tol and abs(px) <= tol
    flat = not near and gain < min_gain
    ok = near or flat
    return {
        "status": "aligned (flat)" if flat else "aligned" if near else "misaligned",
        "passes": ok,
        "gain": gain,
        "min_gain": min_gain,
        "at_edge": (not ok) and (abs(py) == shift or abs(px) == shift),
        "peak_dy": py,
        "peak_dx": px,
        "peak_auc": best,
        "auc_at_0": auc[(0, 0)],
        "shift": shift,
        "tol": tol,
        "window": [y0, x0, h, w],
    }
