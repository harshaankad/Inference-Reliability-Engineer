"""Summaries of per-request records and SLO evaluation, shared by the controller and the MCP server.

A record looks like:
  {"request_id", "prompt_id", "t_start", "t_end", "max_tokens", "prompt_tokens",
   "output_tokens", "ttft_ms", "e2e_ms", "status": "ok|error|timeout|cancelled", "error"}
"""
from __future__ import annotations

import math
from typing import Any, Iterable


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return round(s[lo], 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def _lat(records: Iterable[dict[str, Any]], key: str) -> list[float]:
    return [r[key] for r in records if r.get("status") == "ok" and r.get(key) is not None]


def summarize(records: list[dict[str, Any]], duration_s: float | None = None,
              slo: dict[str, Any] | None = None, long_prompt_threshold: int = 2000) -> dict[str, Any]:
    """Latency/throughput/goodput summary. `slo` supplies the per-request goodput thresholds."""
    n = len(records)
    if duration_s is None:
        if records:
            duration_s = max(1.0, max(r["t_start"] for r in records) - min(r["t_start"] for r in records))
        else:
            duration_s = 1.0
    ok = [r for r in records if r.get("status") == "ok"]
    out_tokens = sum(r.get("output_tokens") or 0 for r in ok)
    slo = slo or {}
    e2e_max = (slo.get("p95_e2e_ms") or {}).get("max")
    ttft_max = (slo.get("p95_ttft_ms") or {}).get("max")

    def good(r: dict[str, Any]) -> bool:
        if r.get("status") != "ok":
            return False
        if e2e_max is not None and r["e2e_ms"] > e2e_max:
            return False
        if ttft_max is not None and (r.get("ttft_ms") is None or r["ttft_ms"] > ttft_max):
            return False
        return True

    prompt_tokens = [r["prompt_tokens"] for r in records if r.get("prompt_tokens")]
    long_ = [r for r in records if (r.get("prompt_tokens") or 0) >= long_prompt_threshold]
    short = [r for r in records if 0 < (r.get("prompt_tokens") or 0) < long_prompt_threshold]

    def lat_block(rs: list[dict[str, Any]]) -> dict[str, Any]:
        e2e, ttft = _lat(rs, "e2e_ms"), _lat(rs, "ttft_ms")
        return {"n": len(rs), "p50_ttft_ms": percentile(ttft, 50), "p95_ttft_ms": percentile(ttft, 95),
                "p50_e2e_ms": percentile(e2e, 50), "p95_e2e_ms": percentile(e2e, 95)}

    statuses: dict[str, int] = {}
    for r in records:
        statuses[r.get("status", "?")] = statuses.get(r.get("status", "?"), 0) + 1

    return {
        "requests": n,
        "duration_s": round(duration_s, 1),
        "offered_rps": round(n / duration_s, 3),
        "completed_rps": round(len(ok) / duration_s, 3),
        "output_tokens_per_s": round(out_tokens / duration_s, 1),
        "goodput_rps": round(sum(1 for r in records if good(r)) / duration_s, 3),
        "goodput_ratio": round(sum(1 for r in records if good(r)) / n, 4) if n else None,
        "error_rate": round((n - len(ok)) / n, 4) if n else None,
        "statuses": statuses,
        "latency": lat_block(records),
        "p99_e2e_ms": percentile(_lat(records, "e2e_ms"), 99),
        "by_prompt_length": {
            f"short(<{long_prompt_threshold} tok)": lat_block(short),
            f"long(>={long_prompt_threshold} tok)": lat_block(long_),
        },
        "prompt_tokens": {"p50": percentile(prompt_tokens, 50), "p95": percentile(prompt_tokens, 95),
                          "long_share": round(len(long_) / n, 3) if n else None},
    }


def evaluate_slos(summary: dict[str, Any], slo: dict[str, Any]) -> dict[str, Any]:
    """Compare a summary against policy SLOs. Returns per-SLO verdicts and all_pass."""
    lookups = {
        "p95_e2e_ms": summary["latency"]["p95_e2e_ms"],
        "p95_ttft_ms": summary["latency"]["p95_ttft_ms"],
        "error_rate": summary["error_rate"],
        "goodput_ratio": summary["goodput_ratio"],
    }
    results: dict[str, Any] = {}
    for name, bounds in slo.items():
        value = lookups.get(name)
        if value is None:
            results[name] = {"value": None, "pass": False, "note": "no data"}
            continue
        passed = True
        if "max" in bounds and value > bounds["max"]:
            passed = False
        if "min" in bounds and value < bounds["min"]:
            passed = False
        results[name] = {"value": value, **bounds, "pass": passed}
    return {"slos": results, "all_pass": bool(results) and all(r["pass"] for r in results.values())}
