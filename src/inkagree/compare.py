"""Compare two ink-prediction images ("arms") against labels on a segment's frame.

Order of questions, each of which can stop the answer:

1. **Frame.** Arms and labels are cropped to their common top-left shape; a canvas more than a few pixels
   off the label shape is refused (wrong level or wrong render settings).
2. **Registration.** The alignment gate (on arm A) must peak within `tol` px of (0, 0); "undetermined"
   (no defined AUC anywhere) is reported as such and also stops the comparison.
3. **Agreement.** AP, AUC and best F1 of each arm on the mesh-valid domain, and a paired block-bootstrap
   interval for the B - A difference. The verdict says which arm agrees better only when the AP
   interval excludes zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from .metrics import alignment_gate, block_hists, metrics_from_hist, paired_bootstrap

MAX_CROP = 16  # px: tolerated difference between a canvas and the label raster (edge rounding)


def load_arm(path: Path) -> np.ndarray:
    """A uint8 image: a TIFF (uint8 as-is), or a .npy of uint8, or of probabilities in [0, 1] (x255, rounded)."""
    path = Path(path)
    a = np.load(path) if path.suffix == ".npy" else tifffile.imread(path)
    if a.ndim != 2:
        raise ValueError(f"{path}: expected a 2D image, got shape {a.shape}")
    if a.dtype == np.uint8:
        return a
    if np.issubdtype(a.dtype, np.floating):
        f = a.astype(np.float32)
        if np.nanmin(f) < 0 or np.nanmax(f) > 1:
            raise ValueError(f"{path}: float input must be probabilities in [0, 1]")
        return np.clip(np.rint(np.nan_to_num(f) * 255), 0, 255).astype(np.uint8)
    raise ValueError(f"{path}: unsupported dtype {a.dtype}")


def domain_mask(mesh_x: Path, shape: tuple[int, int]) -> np.ndarray:
    """Canvas pixels whose source grid cell is valid (x != -1), nearest-upsampled to `shape`."""
    gx = tifffile.imread(mesh_x)
    valid = np.isfinite(gx) & (gx != -1)
    gh, gw = valid.shape
    H, W = shape
    ri = np.minimum((np.arange(H) * gh) // H, gh - 1)
    ci = np.minimum((np.arange(W) * gw) // W, gw - 1)
    return valid[np.ix_(ri, ci)]


def compare_arms(
    a: np.ndarray,
    b: np.ndarray,
    ink: np.ndarray,
    dom_full: np.ndarray,
    block: int = 256,
    n_boot: int = 2000,
    seed: int = 20261003,
    gate_shift: int = 2,
    gate_tol: int = 1,
    gate_window: int | None = None,
    gate_min_gain: float = 0.002,
    gate_image: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """`dom_full` is the domain at arm A's full canvas shape (mapping computed before any crop).

    The alignment gate runs on `gate_image` if given (e.g. the raw render, when the arms are a weak scorer's
    outputs whose AUC surface is flat), else on arm A. `gate_min_gain` is the AUC an offset must gain over
    (0, 0) to count as evidence of misalignment (0.002 by default; 0 reproduces a strict argmax).
    """
    if a.shape != dom_full.shape:
        raise ValueError(f"domain {dom_full.shape} must match arm A's canvas {a.shape}")
    H = min(a.shape[0], b.shape[0], ink.shape[0])
    W = min(a.shape[1], b.shape[1], ink.shape[1])
    off = max(abs(s - t) for s, t in ((a.shape[0], ink.shape[0]), (a.shape[1], ink.shape[1]),
                                      (b.shape[0], ink.shape[0]), (b.shape[1], ink.shape[1])))  # fmt: skip
    res: dict = {"shape": [H, W], "arm_a_shape": list(a.shape), "arm_b_shape": list(b.shape),
                 "label_shape": list(ink.shape)}  # fmt: skip
    if off > MAX_CROP:
        res.update(status="frame_mismatch", passes=False,
                   reason=f"canvas and labels differ by {off} px (> {MAX_CROP}); wrong level or render settings?")  # fmt: skip
        return res
    a, b, ink, dom = a[:H, :W], b[:H, :W], ink[:H, :W], dom_full[:H, :W]
    res["domain_px"] = int(dom.sum())
    res["label_ink_frac"] = float(ink[dom].mean()) if dom.any() else float("nan")
    g_img = a if gate_image is None else gate_image[:H, :W]
    if g_img.shape != a.shape:
        raise ValueError(f"gate image {g_img.shape} must cover the compared frame {a.shape}")
    gate = alignment_gate(g_img, ink, dom, shift=gate_shift, tol=gate_tol, window=gate_window, min_gain=gate_min_gain)
    res["gate"] = gate
    if not gate["passes"]:
        res.update(status=gate["status"], passes=False)
        return res
    ha, hb = block_hists(a, ink, dom, block), block_hists(b, ink, dom, block)
    ma = metrics_from_hist(ha[0].sum(0), ha[1].sum(0))
    mb = metrics_from_hist(hb[0].sum(0), hb[1].sum(0))
    boot = paired_bootstrap(ha, hb, n_boot, rng if rng is not None else np.random.default_rng(seed))
    lo, hi = boot["d_ap_ci"]
    verdict = "B agrees better" if lo > 0 else "A agrees better" if hi < 0 else "no resolved difference"
    res.update(status="compared", passes=True, a=ma, b=mb, d_ap=mb["ap"] - ma["ap"],
               d_auc=mb["auc"] - ma["auc"], block=block, seed=seed, verdict=verdict, **boot)  # fmt: skip
    return res
