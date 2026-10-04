"""villa's published ink labels and segment meshes, from the open-data bucket.

villa publishes binary ink labels on the 2.4 um PHercParis4 scan's frame for some segments:

    s3://vesuvius-challenge-open-data/PHercParis4/segments/<seg>/ink-labels/2.4um-volume-20260411134726/<date>/inklabels.zarr

An OME-Zarr v3 pyramid: level k is 2.4 * 2**k um per pixel, values 0/255. The segment's mesh on the same
scan is `segments/<seg>/mesh/<seg>-on-20260411134726-2.4um.tifxyz` (x/y/z.tif + meta.json). Rendering that
mesh with vc_render_tifxyz at output pixel 2.4 * 2**k um lands exactly on label level k (inkagree checks
the shape, and the alignment gate checks the registration).
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import numpy as np

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
SCROLL = "PHercParis4"
SCAN = "20260411134726"
INK3D = (
    BUCKET + "PHercParis4/representations/predictions/ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr"
)


def _list(prefix: str) -> list[str]:
    """Common prefixes (sub-'directories') under `prefix`, following S3 pagination."""
    out, token = [], None
    while True:
        url = f"{BUCKET}?list-type=2&delimiter=/&prefix={prefix}"
        if token:
            url += "&continuation-token=" + urllib.request.quote(token, safe="")
        body = urllib.request.urlopen(url, timeout=60).read().decode()
        out += [p for p in re.findall(r"<Prefix>([^<]+)</Prefix>", body) if p != prefix]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not m:
            return out
        token = m.group(1)


def label_versions(seg: str, scroll: str = SCROLL, scan: str = SCAN) -> list[str]:
    """Dated label releases for `seg` on the scan's frame, oldest first (e.g. ['20260918'])."""
    pre = f"{scroll}/segments/{seg}/ink-labels/2.4um-volume-{scan}/"
    return sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in _list(pre))


def labelled_segments(scroll: str = SCROLL, scan: str = SCAN) -> list[str]:
    """Segments of `scroll` that have ink labels on the scan's frame (walks the bucket; ~1 request each)."""
    segs = [p.rstrip("/").rsplit("/", 1)[-1] for p in _list(f"{scroll}/segments/")]
    return [s for s in segs if label_versions(s, scroll, scan)]


def fetch_labels(
    seg: str, level: int, version: str | None = None, scroll: str = SCROLL, scan: str = SCAN
) -> np.ndarray:
    """The binary label raster at pyramid `level` (2.4 * 2**level um/px) as a bool array."""
    import tensorstore as ts

    version = version or label_versions(seg, scroll, scan)[-1]
    path = f"{scroll}/segments/{seg}/ink-labels/2.4um-volume-{scan}/{version}/inklabels.zarr/{level}/"
    t = ts.open({"driver": "zarr3", "kvstore": {"driver": "http", "base_url": BUCKET, "path": path}}).result()
    a = t.read().result()
    if a.dtype != np.uint8:
        raise ValueError(f"{path}: expected uint8 labels, got {a.dtype}")
    return a > 127


def fetch_supervision(
    seg: str, level: int, version: str | None = None, scroll: str = SCROLL, scan: str = SCAN
) -> np.ndarray | None:
    """The supervision mask beside the labels (where they were annotated), as bool, or None if not published.

    villa's labels are only defined inside this mask: on the 8 PHercParis4 segments it covers 3-13% of the
    canvas and holds 97-100% of the labelled ink. Outside it, "no label" means "not annotated", not "no ink".
    """
    import tensorstore as ts

    try:
        version = version or label_versions(seg, scroll, scan)[-1]
        path = f"{scroll}/segments/{seg}/ink-labels/2.4um-volume-{scan}/{version}/supervision.zarr/{level}/"
        t = ts.open({"driver": "zarr3", "kvstore": {"driver": "http", "base_url": BUCKET, "path": path}}).result()
    except Exception:  # no labels or no supervision published for this segment/version
        return None
    return t.read().result() > 127


def fetch_mesh(seg: str, dest: Path, scroll: str = SCROLL, scan: str = SCAN) -> Path:
    """Download the segment's 2.4 um tifxyz mesh into `dest` (x.tif, y.tif, z.tif, meta.json)."""
    name = f"{seg}-on-{scan}-2.4um.tifxyz"
    base = f"{BUCKET}{scroll}/segments/{seg}/mesh/{name}/"
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("meta.json", "x.tif", "y.tif", "z.tif"):
        with urllib.request.urlopen(base + f, timeout=600) as r, open(dest / f, "wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
    return dest
