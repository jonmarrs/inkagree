# inkagree

[![tests](https://github.com/jonmarrs/inkagree/actions/workflows/tests.yml/badge.svg)](https://github.com/jonmarrs/inkagree/actions/workflows/tests.yml)

**Which of two ink renders agrees better with villa's published ink labels?**

villa now publishes binary ink labels on the 2.4 µm PHercParis4 scan's frame for several Scroll-1
segments (`segments/<id>/ink-labels/2.4um-volume-20260411134726/<date>/inklabels.zarr`). inkagree uses
them to answer a practical question about any change to how ink is rendered or scored: does a render
setting, a scorer, or a model agree better with the labels, or only differently?

* **Render** a segment's published 3D ink prediction through its own mesh with `vc_render_tifxyz`. The
  two presets land **exactly** on a label pyramid level, so no registration is needed.
* **Compare** two such images (or any two prediction images on that frame):
  * exact AP / ROC-AUC / best F1 against the labels, on the mesh-valid domain;
  * an alignment gate, so a mis-registered image cannot pass silently;
  * a paired block-bootstrap interval for the difference;
  * a verdict only when that interval excludes zero.

## Install

```bash
pip install git+https://github.com/jonmarrs/inkagree    # numpy, tifffile, tensorstore
```

Rendering also needs `vc_render_tifxyz` (VC3D), either as a local binary (`--binary`) or a Docker image
(`--image`). Comparing does not.

## Use

```bash
inkagree segments                                  # which segments have labels on the 2.4 um frame
inkagree fetch 20231016151002 seg/                 # its 2.4 um mesh -> seg/mesh

# two arms: villa #1818's default vs smooth surface interpolation
inkagree render seg/ linear.tif --image <vc3d image> --cache-home ~/vc3d-cache
inkagree render seg/ smooth.tif --image <vc3d image> --cache-home ~/vc3d-cache -- --surface-interpolation smooth

inkagree compare 20231016151002 seg/ linear.tif smooth.tif --level 2 --json result.json
# verbatim output on this segment's stored renders (vesuvius-autoresearch finding 71):
# 20231016151002: AP A 0.1125  B 0.1124  dAP -0.00010 [-0.00018, -0.00001]  dAUC -0.00026  -> A agrees better
#   alignment peak (0, 0), 87,619,600 domain px, label ink 1.51%
```

A claim about a setting needs several segments. Write each comparison with `--json`, then aggregate under
a rule you declare **before** looking (0.2.0):

```bash
inkagree summarize results/*.json --k 6
# 8/8 compared; dAP > 0 resolved in 0, < 0 in 6 (k = 6); median dAP ...
# VERDICT: A agrees better
```

Segments that were not compared (frame mismatch, misaligned, undetermined) are listed and never counted.

| preset | vc_render_tifxyz settings | label level | used by |
|---|---|---|---|
| `tutorial` (default) | `--group-idx 2 --scale 1`, 16 slices at step 0.5 | 2 (9.6 µm) | villa's 3D-ink tutorial |
| `metric` | `--group-idx 1 --scale 0.25`, 5 slices at step 1 | 3 (19.2 µm) | villa's spiral-fitting ink metric |

Anything after `--` is passed to `vc_render_tifxyz` (e.g. `--surface-interpolation smooth`). An arm can
also be any 2D `.npy` or TIFF on the same frame: uint8 as-is, or probabilities in [0, 1], quantised to
256 levels. To compare scorer outputs, pass `--gate-on` the raw render (see the gate, below).

## Reading the result

* **`dAP` and its interval** are AP(B) − AP(A) on the segment's mesh-valid domain, with a 95% paired block
  bootstrap (256-px blocks, 2000 resamples, fixed seed). "B agrees better" or "A agrees better" only when
  the interval excludes zero. Otherwise: "no resolved difference".
* **The alignment gate** finds where agreement with the labels peaks over ±2 px shifts.
  * An offset counts as misalignment only if it beats (0, 0) by ≥ 0.002 AUC. A flat surface, typical
    of a weak or smooth scorer output, is "aligned (flat)".
  * No labelled ink in the region gives "undetermined", never "misaligned".
  * For weak arms, gate on the raw render with `--gate-on`.
* **Exit codes:** 0 compared; 2 frame mismatch, misaligned or undetermined; 1 error.

## What it has been used for

Measuring villa #1818 (`--surface-interpolation smooth`) on all 8 labelled segments
([vesuvius-autoresearch](https://github.com/jonmarrs/vesuvius-autoresearch), findings 71–73):

* on villa's segment meshes (~20-voxel grid cells), smooth vs linear changes AP by only −0.0001;
* on grids as coarse as spiral-fit surfaces (~80-voxel cells), the raw render agrees slightly better
  under smooth (resolved in 6 of 8, about +1–2% of AP), while villa's ink-count metric moves by −8% to
  +19% per segment.

## Validation

inkagree reproduces the two registered studies it came from, from their stored inputs, **exactly**:
16 of 16 segment results, including every point estimate (to 1e-12) and every bootstrap interval
(`scripts/validate_inkagree.py` and `reports/inkagree_validation.json` in vesuvius-autoresearch).
* The one intended difference is a segment the original gate wrongly excluded as "misaligned"
  (no labelled ink in its window). inkagree reports it "undetermined".
* The offline test suite checks AP/AUC against scikit-learn to 1e-9, the gate on known synthetic
  offsets, and the render plumbing with a stand-in renderer.

## What it cannot do

* **The labels are not ground truth.** villa's README says scroll labels start as hand-annotated
  strokes and are refined by iterative pseudo-labelling. They therefore favour whatever geometry and
  models produced them (default-mode, fine-grid renders). A win for the incumbent is weaker evidence
  than a win against it.
* **One scroll for now.** Only PHercParis4 segments have labels on the scan frame in this layout.
  `inkagree segments` lists them (8 as of 2026-10-03).
* **Rendering streams a lot.** The renderer caches streamed chunks under `HOME`. Whole segments at the
  `metric` preset grew that cache to hundreds of GB; point `--cache-home` at a disk with room.
* **Agreement with labels is not legibility.** A small dAP can matter or not depending on where it falls.

MIT licensed.
