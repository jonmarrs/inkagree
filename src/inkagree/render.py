"""Render a segment's 3D ink prediction through its own mesh with vc_render_tifxyz, max over slices.

Two presets, each landing exactly on one label level:

* `tutorial`: villa's 3D-ink tutorial (`--group-idx 2 --scale 1`, 16 slices at step 0.5) -> level 2, 9.6 um/px.
* `metric`: villa's spiral-fitting metric (`--group-idx 1 --scale 0.25`, 5 slices at step 1) -> level 3,
  19.2 um/px.

`--scale-segmentation 1` throughout: the segment meshes are in full-resolution (level-0) voxel coordinates.
vc_render_tifxyz runs either from a local binary or from a Docker image (the scroll volume is streamed
from the bucket into the remote-cache root under HOME, which can grow to hundreds of GB for whole segments).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import tifffile

PRESETS = {
    "tutorial": {"level": 2, "args": ["--group-idx", "2", "--scale", "1", "--num-slices", "16", "--slice-step", "0.5"]},
    "metric": {"level": 3, "args": ["--group-idx", "1", "--scale", "0.25", "--num-slices", "5", "--slice-step", "1"]},
}
SKIP_SIGNATURE = "all slices exist, skipping"


def max_composite(tifdir: Path) -> np.ndarray:
    comp = None
    for f in sorted(tifdir.glob("*.tif")):
        a = tifffile.imread(f)
        comp = a if comp is None else np.maximum(comp, a)
    if comp is None:
        raise FileNotFoundError(f"no slices in {tifdir}")
    return comp


def render(
    mesh: Path,
    out: Path,
    volume_url: str,
    preset: str = "tutorial",
    extra: list[str] | None = None,
    binary: str | None = None,
    image: str | None = None,
    cache_home: Path | None = None,
    keep_slices: bool = False,
) -> dict:
    """Render `mesh` from `volume_url` and write the max-over-slices image to `out` (uint8 TIFF).

    Exactly one of `binary` (path to vc_render_tifxyz) or `image` (Docker image providing it) is used.
    Returns the command, the log path and the image shape. Fails if the renderer skipped sampling.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    if (binary is None) == (image is None):
        raise ValueError("give exactly one of binary= or image=")
    work = out.with_suffix(".slices")
    if work.exists():
        raise FileExistsError(f"{work} exists; vc_render_tifxyz would skip and re-use old slices")
    work.mkdir(parents=True)
    cache = work / "volcache"
    cache.mkdir()
    args = [
        "--volume", str(cache), "--remote-url", volume_url, "--scale-segmentation", "1",
        "--segmentation", str(mesh.resolve()), "--tif-output", str(work.resolve()),
        *PRESETS[preset]["args"], *(extra or []),
    ]  # fmt: skip
    if binary is not None:
        cmd = [binary, *args]
        env = dict(os.environ, **({"HOME": str(cache_home)} if cache_home else {}))
    else:
        mounts = sorted({str(p) for p in (mesh.resolve().parent, work.resolve(), cache_home) if p})
        cmd = ["docker", "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}"]
        for m in mounts:
            cmd += ["-v", f"{m}:{m}"]
        if cache_home:
            cmd += ["-e", f"HOME={cache_home}"]
        cmd += ["--entrypoint", "vc_render_tifxyz", image, *args]
        env = None
    log = out.with_suffix(".render.log")
    with open(log, "w") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT, env=env)
    text = log.read_text(errors="replace")
    if SKIP_SIGNATURE in text:
        raise RuntimeError(f"{log}: the renderer skipped sampling ({SKIP_SIGNATURE!r})")
    comp = max_composite(work)
    tifffile.imwrite(out, comp, compression="zlib")
    if not keep_slices:
        shutil.rmtree(work)
    return {"command": cmd, "log": str(log), "shape": list(comp.shape), "level": PRESETS[preset]["level"]}
