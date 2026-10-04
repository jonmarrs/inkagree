"""inkagree, offline: no network, no real renderer."""

import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from inkagree import cli
from inkagree.compare import compare_arms, domain_mask, load_arm
from inkagree.metrics import alignment_gate, block_hists, metrics_from_hist, paired_bootstrap
from inkagree.render import PRESETS, render


def _blobs(h, w, n, seed):
    rng = np.random.default_rng(seed)
    lab = np.zeros((h, w), bool)
    for _ in range(n):
        y, x = rng.integers(8, h - 8), rng.integers(8, w - 8)
        lab[y - 3 : y + 3, x - 6 : x + 6] = True
    return lab


# ----------------------------------------------------------------------------- metrics


def test_histogram_metrics_equal_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(5):
        y = rng.random(40_000) < 0.07
        s = np.clip(rng.normal(60 + 40 * y, 30), 0, 255).astype(np.uint8)
        m = metrics_from_hist(np.bincount(s[y], minlength=256), np.bincount(s[~y], minlength=256))
        assert m["ap"] == pytest.approx(skm.average_precision_score(y, s), abs=1e-9)
        assert m["auc"] == pytest.approx(skm.roc_auc_score(y, s), abs=1e-9)


def test_metrics_are_undefined_without_both_classes():
    m = metrics_from_hist(np.zeros(256), np.ones(256))
    assert all(np.isnan(v) for v in m.values())


def test_block_hists_sum_to_the_whole_image_and_refuse_floats():
    rng = np.random.default_rng(3)
    s = rng.integers(0, 256, (700, 900), dtype=np.uint8)
    ink, dom = rng.random(s.shape) < 0.1, rng.random(s.shape) < 0.8
    p, n = block_hists(s, ink, dom, 256)
    assert (p.sum(0) == np.bincount(s[dom & ink], minlength=256)).all()
    assert (n.sum(0) == np.bincount(s[dom & ~ink], minlength=256)).all()
    with pytest.raises(TypeError):
        block_hists(s.astype(np.float32), ink, dom, 256)


def test_paired_bootstrap_of_identical_arms_is_exactly_zero():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 256, (600, 600), dtype=np.uint8)
    ink, dom = rng.random(s.shape) < 0.1, np.ones(s.shape, bool)
    h = block_hists(s, ink, dom, 128)
    b = paired_bootstrap(h, h, 200, np.random.default_rng(0))
    assert b["d_ap_ci"] == [0.0, 0.0] and b["d_auc_ci"] == [0.0, 0.0]


# ----------------------------------------------------------------------------- alignment gate


@pytest.mark.parametrize("dy,dx,passes", [(0, 0, True), (1, -1, True), (2, -1, False), (-2, 2, False)])
def test_gate_recovers_a_known_offset(dy, dx, passes):
    H, W = 500, 600
    ink = _blobs(H, W, 250, 1)
    rng = np.random.default_rng(2)
    score = rng.integers(0, 40, (H, W)).astype(np.uint8)
    ys, xs = slice(max(0, -dy), H - max(0, dy)), slice(max(0, -dx), W - max(0, dx))
    yl, xl = slice(max(0, dy), H + min(0, dy)), slice(max(0, dx), W + min(0, dx))
    score[ys, xs] += (ink[yl, xl] * 200).astype(np.uint8)  # score[y, x] mirrors ink[y + dy, x + dx]
    g = alignment_gate(score, ink, np.ones((H, W), bool), shift=2, tol=1)
    assert (g["peak_dy"], g["peak_dx"]) == (dy, dx) and g["passes"] is passes
    assert g["status"] == ("aligned" if passes else "misaligned")


def test_gate_without_labelled_ink_is_undetermined_not_misaligned():
    s = np.random.default_rng(0).integers(0, 255, (300, 300), dtype=np.uint8)
    g = alignment_gate(s, np.zeros(s.shape, bool), np.ones(s.shape, bool))
    assert g["status"] == "undetermined" and g["passes"] is False


# ----------------------------------------------------------------------------- compare


def _arms(H=512, W=640):
    ink = _blobs(H, W, 300, 4)
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 120, (H, W))
    a = np.clip(noise + ink * 40, 0, 255).astype(np.uint8)  # weakly informative
    b = np.clip(noise + ink * 120, 0, 255).astype(np.uint8)  # strongly informative
    return a, b, ink


def test_identical_arms_give_zero_and_no_verdict():
    a, _, ink = _arms()
    r = compare_arms(a, a, ink, np.ones(a.shape, bool), block=128, n_boot=200)
    assert r["passes"] and r["d_ap"] == 0 and r["d_ap_ci"] == [0.0, 0.0]
    assert r["verdict"] == "no resolved difference"


def test_a_clearly_better_arm_is_resolved_both_ways_round():
    a, b, ink = _arms()
    dom = np.ones(a.shape, bool)
    assert compare_arms(a, b, ink, dom, block=128, n_boot=200)["verdict"] == "B agrees better"
    assert compare_arms(b, a, ink, dom, block=128, n_boot=200)["verdict"] == "A agrees better"


def test_small_edge_differences_are_cropped_large_ones_refused():
    a, b, ink = _arms()
    dom = np.ones(a.shape, bool)
    r = compare_arms(a, b[:-5, :-5], ink, dom, block=128, n_boot=50)
    assert r["passes"] and r["shape"] == [a.shape[0] - 5, a.shape[1] - 5]
    big = np.zeros((a.shape[0] + 40, a.shape[1]), np.uint8)
    r = compare_arms(big, b, ink, np.ones(big.shape, bool), block=128, n_boot=50)
    assert r["status"] == "frame_mismatch" and not r["passes"]


def test_load_arm_accepts_uint8_and_probabilities(tmp_path):
    u = np.arange(256, dtype=np.uint8).reshape(16, 16)
    tifffile.imwrite(tmp_path / "u.tif", u)
    assert (load_arm(tmp_path / "u.tif") == u).all()
    np.save(tmp_path / "p.npy", np.array([[0.0, 0.5], [1.0, 0.25]], np.float16))
    assert load_arm(tmp_path / "p.npy").tolist() == [[0, 128], [255, 64]]
    np.save(tmp_path / "bad.npy", np.array([[0.0, 2.0]], np.float32))
    with pytest.raises(ValueError):
        load_arm(tmp_path / "bad.npy")


def test_domain_mask_upsamples_each_grid_cell_exactly(tmp_path):
    g = np.ones((4, 6), np.float32)
    g[1, 2] = -1
    tifffile.imwrite(tmp_path / "x.tif", g)
    d = domain_mask(tmp_path / "x.tif", (20, 30))
    assert d.sum() == 20 * 30 - 25 and not d[5:10, 10:15].any()


# ----------------------------------------------------------------------------- render plumbing (fake renderer)


def _fake_renderer(tmp_path, log_line=""):
    """A stand-in vc_render_tifxyz: writes 3 slices of a known image to --tif-output."""
    script = tmp_path / "fake_vc_render_tifxyz"
    script.write_text(f"""#!{sys.executable}
import sys, numpy as np, tifffile, pathlib
a = sys.argv; out = pathlib.Path(a[a.index('--tif-output') + 1])
for i in range(3):
    tifffile.imwrite(out / f'{{i:02d}}.tif', np.full((8, 10), 10 * (i + 1), np.uint8), compression='lzw')  # like vc_render_tifxyz
print({log_line!r}); print(' '.join(a[1:]))
""")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


def test_render_takes_the_max_over_slices_and_records_the_command(tmp_path):
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    out = tmp_path / "arm.tif"
    info = render(mesh, out, "https://example/ink.zarr", "metric", ["--surface-interpolation", "smooth"],
                  binary=_fake_renderer(tmp_path))  # fmt: skip
    assert (tifffile.imread(out) == 30).all() and info["level"] == PRESETS["metric"]["level"] == 3
    log = Path(info["log"]).read_text()
    assert "--group-idx 1 --scale 0.25" in log and "--surface-interpolation smooth" in log
    assert "--scale-segmentation 1" in log and not out.with_suffix(".slices").exists()


def test_render_refuses_a_skipped_render_and_stale_slices(tmp_path):
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    with pytest.raises(RuntimeError, match="skipped sampling"):
        render(mesh, tmp_path / "a.tif", "u", binary=_fake_renderer(tmp_path, "[tif] all slices exist, skipping."))
    (tmp_path / "b.slices").mkdir()
    with pytest.raises(FileExistsError):
        render(mesh, tmp_path / "b.tif", "u", binary=_fake_renderer(tmp_path))
    with pytest.raises(ValueError):
        render(mesh, tmp_path / "c.tif", "u", binary="x", image="y")


# ----------------------------------------------------------------------------- CLI


def test_cli_compare_end_to_end(tmp_path, monkeypatch, capsys):
    a, b, ink = _arms(400, 500)
    tifffile.imwrite(tmp_path / "a.tif", a)
    tifffile.imwrite(tmp_path / "b.tif", b)
    (tmp_path / "seg" / "mesh").mkdir(parents=True)
    tifffile.imwrite(tmp_path / "seg" / "mesh" / "x.tif", np.ones((80, 100), np.float32))
    monkeypatch.setattr("inkagree.labels.fetch_labels", lambda seg, level, version=None: ink)
    monkeypatch.setattr("inkagree.labels.fetch_supervision", lambda seg, level, version=None: None)
    rc = cli.main(["compare", "S", str(tmp_path / "seg"), str(tmp_path / "a.tif"), str(tmp_path / "b.tif"),
                   "--level", "2", "--boot", "100", "--json", str(tmp_path / "r.json")])  # fmt: skip
    assert rc == 0 and "B agrees better" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json").read_text())["verdict"] == "B agrees better"


def test_cli_help_and_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0 and "inkagree" in capsys.readouterr().out
    assert os.environ.get("INKAGREE_NETWORK") in (None, "", "0", "1")  # network tests are opt-in


def test_a_flat_auc_surface_is_aligned_not_misaligned():
    # weak, spatially smooth score: AUC barely varies with the shift, and argmax lands anywhere
    H, W = 400, 400
    ink = _blobs(H, W, 200, 7)
    rng = np.random.default_rng(8)
    score = np.clip(rng.normal(100, 30, (H, W)) + ink * 2, 0, 255).astype(np.uint8)
    strict = alignment_gate(score, ink, np.ones((H, W), bool), shift=2, tol=0, min_gain=0.0)
    lenient = alignment_gate(score, ink, np.ones((H, W), bool), shift=2, tol=0, min_gain=0.01)
    assert lenient["status"] in ("aligned", "aligned (flat)") and lenient["passes"]
    if strict["status"] == "misaligned":  # the strict argmax may wander; the flat rule must catch it
        assert lenient["status"] == "aligned (flat)" and lenient["gain"] < 0.01


def test_a_material_offset_at_the_search_edge_is_flagged():
    H, W = 500, 600
    ink = _blobs(H, W, 250, 1)
    score = np.random.default_rng(2).integers(0, 40, (H, W)).astype(np.uint8)
    score[:, : W - 2] += (ink[:, 2:] * 200).astype(np.uint8)  # true offset dx = +2 = the search edge
    g = alignment_gate(score, ink, np.ones((H, W), bool), shift=2, tol=1, min_gain=0.002)
    assert g["status"] == "misaligned" and g["at_edge"] and g["gain"] > 0.002


def test_gate_on_uses_the_given_image():
    a, b, ink = _arms()
    dom = np.ones(a.shape, bool)
    shifted = np.zeros_like(b)
    shifted[:, :-2] = b[:, 2:]  # a gate image mis-registered by 2 px
    r = compare_arms(a, b, ink, dom, block=128, n_boot=50, gate_image=shifted, gate_min_gain=0.0)
    assert r["status"] == "misaligned" and (r["gate"]["peak_dy"], r["gate"]["peak_dx"]) == (
        0,
        2,
    )  # gate[:, x] = b[:, x + 2]
    assert compare_arms(a, b, ink, dom, block=128, n_boot=50, gate_image=b)["passes"]


# ----------------------------------------------------------------------------- summarize


def _r(seg, d, lo, hi, status="compared"):
    return {"segment": seg, "status": status, "d_ap": d, "d_ap_ci": [lo, hi]}


def test_summarize_needs_k_resolved_segments_and_never_counts_skipped_ones():
    from inkagree.summary import summarize

    rs = [_r(f"s{i}", 0.01, 0.002, 0.02) for i in range(5)] + [
        _r("x", 0.0, -0.01, 0.01),
        _r("m", 0, 0, 0, "misaligned"),
    ]
    assert summarize(rs, 5)["verdict"] == "B agrees better"
    s = summarize(rs, 6)
    assert (
        s["verdict"] == "no consistent difference" and s["n_compared"] == 6 and s["not_compared"] == {"m": "misaligned"}
    )
    neg = [_r(f"s{i}", -0.01, -0.02, -0.001) for i in range(6)]
    assert summarize(neg, 6)["verdict"] == "A agrees better"


def test_summarize_cli(tmp_path, capsys):
    files = []
    for i in range(3):
        f = tmp_path / f"r{i}.json"
        f.write_text(json.dumps(_r(f"s{i}", 0.01, 0.001, 0.02)))
        files.append(str(f))
    assert cli.main(["summarize", *files, "--k", "3"]) == 0
    assert "VERDICT: B agrees better" in capsys.readouterr().out


def test_extra_args_override_the_preset_instead_of_duplicating_it(tmp_path):
    from inkagree.render import merge_args

    m = merge_args(PRESETS["tutorial"]["args"], ["--slice-step", "1", "--surface-interpolation", "smooth"])
    assert m.count("--slice-step") == 1 and m[m.index("--slice-step") + 1] == "1"
    assert m[m.index("--num-slices") + 1] == "16" and "--surface-interpolation" in m
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    info = render(mesh, tmp_path / "a.tif", "u", "tutorial", ["--slice-step", "2"], binary=_fake_renderer(tmp_path))
    cmd = Path(info["log"]).read_text().splitlines()[0]  # the recorded command
    assert cmd.count("--slice-step") == 1 and "--slice-step 2" in cmd


def test_cli_render_keeps_options_and_passes_everything_after_the_separator(tmp_path, capsys):
    # regression: argparse.REMAINDER used to swallow --binary/--image when given after the positionals
    (tmp_path / "seg" / "mesh").mkdir(parents=True)
    out = tmp_path / "arm.tif"
    rc = cli.main(["render", str(tmp_path / "seg"), str(out), "--binary", _fake_renderer(tmp_path),
                   "--", "--slice-step", "1", "--surface-interpolation", "smooth"])  # fmt: skip
    assert rc == 0 and out.exists()
    cmd = out.with_suffix(".render.log").read_text().splitlines()[0]  # the recorded command
    assert cmd.count("--slice-step") == 1 and "--slice-step 1" in cmd and "--surface-interpolation smooth" in cmd


def test_cli_refuses_a_separator_outside_render(capsys):
    with pytest.raises(SystemExit):
        cli.main(["segments", "--", "--x"])


def test_the_render_log_records_the_command(tmp_path):
    mesh = tmp_path / "mesh"
    mesh.mkdir()
    info = render(mesh, tmp_path / "a.tif", "u", "metric", ["--slice-step", "2"], binary=_fake_renderer(tmp_path))
    first = Path(info["log"]).read_text().splitlines()[0]
    assert first.startswith("# inkagree command: ") and "--slice-step 2" in first and first.count("--slice-step") == 1


def test_lzw_slices_from_the_renderer_can_be_read(tmp_path):
    # vc_render_tifxyz writes LZW TIFFs; reading them needs imagecodecs (a declared dependency since 0.2.2)
    d = tmp_path / "s"
    d.mkdir()
    for i in range(2):
        tifffile.imwrite(d / f"{i:02d}.tif", np.full((4, 4), 7 * (i + 1), np.uint8), compression="lzw")
    from inkagree.render import max_composite

    assert (max_composite(d) == 14).all()


# ----------------------------------------------------------------------------- supervision (0.3.0)


def test_unannotated_ink_penalises_an_arm_unless_evaluation_is_supervised():
    # true ink everywhere; villa-style labels annotate only the left half (the supervised region)
    H, W = 512, 640
    truth = _blobs(H, W, 400, 11)
    sup = np.zeros((H, W), bool)
    sup[:, : W // 2] = True
    labels = truth & sup
    rng = np.random.default_rng(12)
    noise = rng.integers(0, 60, (H, W))
    a = np.clip(noise + (truth & sup) * 150, 0, 255).astype(np.uint8)  # finds ink only where annotated
    b = np.clip(noise + truth * 150, 0, 255).astype(np.uint8)  # finds ink everywhere (equally well on the left)
    dom = np.ones((H, W), bool)
    unsup = compare_arms(a, b, labels, dom, block=128, n_boot=200)
    supv = compare_arms(a, b, labels, dom, block=128, n_boot=200, supervision=sup)
    assert unsup["verdict"] == "A agrees better"  # the flaw: B's real right-half ink counts as false positives
    assert supv["d_ap"] == 0 and supv["verdict"] == "no resolved difference"  # identical where labels exist
    assert supv["domain"] == "mesh-valid AND supervised" and supv["domain_px"] == H * (W // 2)
    assert "not be exhaustive" in unsup["domain"]


def test_cli_uses_supervision_by_default_and_can_opt_out(tmp_path, monkeypatch, capsys):
    a, b, ink = _arms(400, 500)
    tifffile.imwrite(tmp_path / "a.tif", a)
    tifffile.imwrite(tmp_path / "b.tif", b)
    (tmp_path / "seg" / "mesh").mkdir(parents=True)
    tifffile.imwrite(tmp_path / "seg" / "mesh" / "x.tif", np.ones((80, 100), np.float32))
    sup = np.zeros(ink.shape, bool)
    sup[:200] = True
    monkeypatch.setattr("inkagree.labels.fetch_labels", lambda seg, level, version=None: ink)
    monkeypatch.setattr("inkagree.labels.fetch_supervision", lambda seg, level, version=None: sup)
    base = ["compare", "S", str(tmp_path / "seg"), str(tmp_path / "a.tif"), str(tmp_path / "b.tif"), "--level", "2",
            "--boot", "50"]  # fmt: skip
    assert cli.main([*base, "--json", str(tmp_path / "s.json")]) == 0
    assert json.loads((tmp_path / "s.json").read_text())["domain_px"] == 200 * 500
    assert cli.main([*base, "--all-domain", "--json", str(tmp_path / "u.json")]) == 0
    assert json.loads((tmp_path / "u.json").read_text())["domain_px"] == 400 * 500
