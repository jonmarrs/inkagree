"""Aggregate per-segment `inkagree compare --json` results under a pre-declared rule.

A single segment's verdict is one draw; a claim about a setting needs several segments that agree. The
rule: "B agrees better" only if dAP > 0 is resolved (interval excludes zero) in at least `k` of the segments
that were compared; "A agrees better" symmetrically; otherwise "no consistent difference". Segments that
were not compared (frame mismatch, misaligned, undetermined) are listed and never counted for either side.
"""

from __future__ import annotations

import statistics


def summarize(results: list[dict], k: int) -> dict:
    compared = [r for r in results if r.get("status") == "compared"]
    skipped = {r.get("segment", "?"): r.get("status") for r in results if r.get("status") != "compared"}
    pos = sum(r["d_ap_ci"][0] > 0 for r in compared)
    neg = sum(r["d_ap_ci"][1] < 0 for r in compared)
    d_ap = [r["d_ap"] for r in compared]
    if pos >= k:
        verdict = "B agrees better"
    elif neg >= k:
        verdict = "A agrees better"
    else:
        verdict = "no consistent difference"
    return {
        "n_results": len(results),
        "n_compared": len(compared),
        "not_compared": skipped,
        "k": k,
        "resolved_b_better": pos,
        "resolved_a_better": neg,
        "d_ap_positive": sum(v > 0 for v in d_ap),
        "d_ap_negative": sum(v < 0 for v in d_ap),
        "d_ap_median": statistics.median(d_ap) if d_ap else None,
        "d_ap_range": [min(d_ap), max(d_ap)] if d_ap else None,
        "verdict": verdict,
    }
