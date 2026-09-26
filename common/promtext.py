"""Minimal parser for the Prometheus text format served by vLLM's /metrics.

vLLM renamed several metrics across versions, so each value we care about is looked up
through a list of aliases and the first one present wins.
"""
from __future__ import annotations

import re

_LINE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEInfNa]+)')

ALIASES: dict[str, list[str]] = {
    "kv_cache_usage": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
    "running": ["vllm:num_requests_running"],
    "waiting": ["vllm:num_requests_waiting"],
    "preemptions_total": ["vllm:num_preemptions_total", "vllm:num_preemptions"],
    "prefix_hits_total": ["vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"],
    "prefix_queries_total": ["vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total"],
    "prefix_hit_rate": ["vllm:gpu_prefix_cache_hit_rate"],  # older vLLM exposes a gauge instead
    "generation_tokens_total": ["vllm:generation_tokens_total"],
    "prompt_tokens_total": ["vllm:prompt_tokens_total"],
    "request_success_total": ["vllm:request_success_total"],
}

COUNTERS = {"preemptions_total", "prefix_hits_total", "prefix_queries_total",
            "generation_tokens_total", "prompt_tokens_total", "request_success_total"}


def parse(text: str) -> dict[str, float]:
    """Sum each metric over its label sets (one model per server, so this is safe)."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        totals[m.group(1)] = totals.get(m.group(1), 0.0) + value
    return totals


# Newer vLLM (V1 scheduler) often does not preempt under KV pressure; it holds requests back and
# labels why. reason="capacity" = waiting because the KV cache is full.
_WAIT_REASON = re.compile(r'^vllm:num_requests_waiting_by_reason\{[^}]*reason="capacity"[^}]*\}\s+([-+0-9.eE]+)', re.M)


def extract(text: str) -> dict[str, float | None]:
    raw = parse(text)
    out: dict[str, float | None] = {}
    for key, names in ALIASES.items():
        out[key] = next((raw[n] for n in names if n in raw), None)
    cap = [float(v) for v in _WAIT_REASON.findall(text)]
    out["waiting_for_kv_capacity"] = sum(cap) if cap else None
    return out
