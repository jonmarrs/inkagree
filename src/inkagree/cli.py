"""inkagree: which of two ink renders agrees better with villa's published ink labels?

    inkagree segments                                   list PHercParis4 segments with labels on the 2.4 um frame
    inkagree fetch SEG DIR                              download SEG's 2.4 um mesh into DIR/mesh
    inkagree render DIR OUT.tif [--preset P] (--image IMG | --binary BIN) [-- EXTRA vc_render_tifxyz ARGS]
                                                        render SEG's 3D ink prediction through DIR/mesh, max over slices
    inkagree compare SEG DIR A B --level L [--json OUT] compare arms A and B (TIFF or .npy) against SEG's labels
    inkagree summarize R.json... --k K                aggregate compare results: a verdict only if resolved in >= K segments

Exit codes: 0 compared; 2 frame mismatch, misaligned or undetermined; 1 error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Everything after a bare `--` is passed to vc_render_tifxyz untouched; split it off before argparse,
    # which would otherwise let a positional REMAINDER swallow options such as --image.
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1 :]
    ap = argparse.ArgumentParser(prog="inkagree", description=__doc__.splitlines()[0])
    ap.add_argument("--version", action="version", version=f"inkagree {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("segments", help="list segments with labels on the 2.4 um frame")
    f = sub.add_parser("fetch", help="download a segment's 2.4 um mesh")
    f.add_argument("seg")
    f.add_argument("dir", type=Path)
    r = sub.add_parser("render", help="render the 3D ink prediction through a segment mesh")
    r.add_argument("dir", type=Path, help="segment dir holding mesh/ (from `inkagree fetch`)")
    r.add_argument("out", type=Path, help="output max-over-slices TIFF")
    r.add_argument("--preset", default="tutorial", choices=["tutorial", "metric"])
    r.add_argument("--image", help="Docker image providing vc_render_tifxyz")
    r.add_argument("--binary", help="path to a local vc_render_tifxyz")
    r.add_argument("--volume-url", help="3D ink prediction OME-Zarr (default: PHercParis4 v3-78k-fullsup)")
    r.add_argument(
        "--cache-home", type=Path, help="HOME for the renderer's remote chunk cache (persists streamed chunks)"
    )
    sm = sub.add_parser("summarize", help="aggregate per-segment compare results under a k-of-n rule")
    sm.add_argument("results", nargs="+", type=Path, help="JSON files written by `inkagree compare --json`")
    sm.add_argument(
        "--k", type=int, required=True, help="resolved segments needed for a verdict (declare it in advance)"
    )
    sm.add_argument("--json", type=Path)
    c = sub.add_parser("compare", help="compare two arms against the labels")
    c.add_argument("seg")
    c.add_argument("dir", type=Path, help="segment dir holding mesh/")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument(
        "--level", type=int, required=True, help="label level the arms are rendered at (tutorial 2, metric 3)"
    )
    c.add_argument("--labels-version", help="label release date (default: the latest)")
    c.add_argument("--block", type=int, default=256)
    c.add_argument("--boot", type=int, default=2000)
    c.add_argument("--seed", type=int, default=20261003)
    c.add_argument("--json", type=Path)
    c.add_argument(
        "--all-domain",
        action="store_true",
        help="evaluate on the whole mesh-valid domain, ignoring villa's supervision mask (not recommended: the labels are only annotated inside it)",
    )
    c.add_argument(
        "--gate-on", type=Path, help="run the alignment gate on this image (e.g. the raw render) instead of arm A"
    )
    c.add_argument(
        "--gate-min-gain", type=float, default=0.002, help="AUC an offset must gain over (0,0) to count as misalignment"
    )
    a = ap.parse_args(argv)
    if extra and a.cmd != "render":
        ap.error("arguments after -- are only accepted by `render`")

    if a.cmd == "summarize":
        from .summary import summarize

        res = summarize([json.loads(p.read_text()) for p in a.results], a.k)
        if a.json:
            a.json.write_text(json.dumps(res, indent=2) + "\n")
        print(f"{res['n_compared']}/{res['n_results']} compared; dAP > 0 resolved in {res['resolved_b_better']}, "
              f"< 0 in {res['resolved_a_better']} (k = {res['k']}); median dAP {res['d_ap_median']}")  # fmt: skip
        for seg, st in res["not_compared"].items():
            print(f"  not compared: {seg} ({st})")
        print(f"VERDICT: {res['verdict']}")
        return 0
    if a.cmd == "segments":
        from .labels import labelled_segments

        for s in labelled_segments():
            print(s)
        return 0
    if a.cmd == "fetch":
        from .labels import fetch_mesh

        print(fetch_mesh(a.seg, a.dir / "mesh"))
        return 0
    if a.cmd == "render":
        from .labels import INK3D
        from .render import render

        info = render(a.dir / "mesh", a.out, a.volume_url or INK3D, a.preset, extra,
                      binary=a.binary, image=a.image, cache_home=a.cache_home)  # fmt: skip
        print(f"{a.out}: {info['shape']} (label level {info['level']})")
        return 0

    from .compare import compare_arms, domain_mask, load_arm
    from .labels import fetch_labels

    arm_a, arm_b = load_arm(a.a), load_arm(a.b)
    ink = fetch_labels(a.seg, a.level, a.labels_version)
    sup = None
    if not a.all_domain:
        from .labels import fetch_supervision

        sup = fetch_supervision(a.seg, a.level, a.labels_version)
        if sup is None:
            print(f"{a.seg}: WARNING no supervision mask published; evaluating on the whole mesh-valid domain",
                  file=sys.stderr)  # fmt: skip
    dom = domain_mask(a.dir / "mesh" / "x.tif", arm_a.shape)
    gate_img = load_arm(a.gate_on) if a.gate_on else None
    res = compare_arms(arm_a, arm_b, ink, dom, block=a.block, n_boot=a.boot, seed=a.seed,
                       gate_min_gain=a.gate_min_gain, gate_image=gate_img, supervision=sup)  # fmt: skip
    res.update(segment=a.seg, level=a.level, arm_a=str(a.a), arm_b=str(a.b))
    if a.json:
        a.json.write_text(json.dumps(res, indent=2) + "\n")
    if not res["passes"]:
        print(f"{a.seg}: {res['status'].upper()}: {res.get('reason') or res.get('gate')}")
        return 2
    print(f"{a.seg}: AP A {res['a']['ap']:.4f}  B {res['b']['ap']:.4f}  dAP {res['d_ap']:+.5f} "
          f"[{res['d_ap_ci'][0]:+.5f}, {res['d_ap_ci'][1]:+.5f}]  dAUC {res['d_auc']:+.5f}  -> {res['verdict']}")  # fmt: skip
    print(f"  alignment peak ({res['gate']['peak_dy']}, {res['gate']['peak_dx']}), {res['domain_px']:,} domain px, "
          f"label ink {res['label_ink_frac']:.2%}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
