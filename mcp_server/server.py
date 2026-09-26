"""inference-ops MCP server: the Firefighter agent's only way to reach the serving fleet.

Three classes of tools, and the line between them is the project's safety story:
  OBSERVE  (readOnlyHint)         telemetry, logs, config, history. Runs freely.
  SHADOW   (not destructive)      experiments on the shadow GPU only. Runs freely; a wrong
                                  hypothesis costs a container restart and a minute.
  PROD     (destructiveHint)      restarts the process serving live traffic. Gated by TrueForge
                                  approval AND by server-side evidence checks.

Run:  python -m mcp_server.server        (serves http://0.0.0.0:8765/mcp, bearer MCP_AUTH_TOKEN)
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import ToolAnnotations

from common import stats
from common import vllm_config as vc
from mcp_server import clients, evidence, prom
from workload.loadtest import engine_summary

POLICY_PATH = Path(os.environ.get("POLICY_PATH", Path(__file__).with_name("policy.yaml")))
DEPLOY_WAIT_S = float(os.environ.get("MCP_DEPLOY_WAIT_S", "480"))
PROD_DEPLOY_WAIT_S = float(os.environ.get("MCP_PROD_DEPLOY_WAIT_S", "900"))

OBSERVE = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
SHADOW = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
PROD = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False)

mcp = MCPServer(
    name="inference-ops",
    instructions=(
        "Tools to diagnose and remediate a vLLM inference service on AWS (Qwen2.5-7B-Instruct on an A10G). "
        "Observe tools read production. Shadow tools experiment on a separate GPU. "
        "apply_production_config and rollback_production restart the live server and need human approval. "
        "Time arguments accept 'now', relative offsets like '-15m', '-2h', '-90s', or ISO-8601."
    ),
)


# --------------------------------------------------------------------------- helpers
def policy() -> dict[str, Any]:
    return yaml.safe_load(POLICY_PATH.read_text())


def parse_time(value: str | float | int) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    v = value.strip().lower()
    if v == "now":
        return time.time()
    m = re.fullmatch(r"-(\d+(?:\.\d+)?)\s*([smhd])", v)
    if m:
        return time.time() - float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ToolError(f"bad time {value!r}; use 'now', '-15m', '-2h' or ISO-8601") from None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def long_threshold() -> int:
    return int(policy().get("analysis", {}).get("long_prompt_threshold_tokens", 2000))


async def _call(coro: Any) -> Any:
    try:
        return await coro
    except clients.ControllerError as e:
        raise ToolError(str(e)) from None


async def engine_samples(start: float, end: float) -> list[dict[str, Any]]:
    """Production engine samples from Prometheus if configured, else the controller's own sampler."""
    if prom.enabled():
        try:
            return await prom.engine_samples(start, end)
        except prom.PromError as e:
            raise ToolError(str(e)) from None
    return await _call(clients.prod().engine_series(start, end))


async def prod_active() -> dict[str, Any]:
    info = await _call(clients.prod().info())
    if not info.get("active_version"):
        raise ToolError("production has no active version yet")
    return info


def window_summary(records: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    s = stats.summarize(records, duration_s=max(1.0, end - start), slo=policy()["slos"],
                        long_prompt_threshold=long_threshold())
    s["slo_evaluation"] = stats.evaluate_slos(s, policy()["slos"])
    return s


def gpu_window(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    mem = [g["memory_used_mib"] for s in samples for g in s.get("gpus", [])]
    util = [g["utilization_pct"] for s in samples for g in s.get("gpus", [])]
    if not mem:
        return None
    return {"memory_used_mib_mean": round(sum(mem) / len(mem)), "utilization_pct_mean": round(sum(util) / len(util), 1),
            "memory_total_mib": samples[-1]["gpus"][0]["memory_total_mib"]}


def pct_change(a: float | None, b: float | None) -> float | None:
    if a in (None, 0) or b is None:
        return None
    return round((b - a) / a * 100, 1)


# --------------------------------------------------------------------------- OBSERVE
@mcp.tool(annotations=OBSERVE)
async def get_slo_status(window_minutes: float = 10) -> dict[str, Any]:
    """Production health right now: latency (p50/p95 TTFT and end-to-end), throughput, goodput and
    error rate over the last `window_minutes`, split by short vs long prompts, each SLO marked
    pass/fail, plus the engine's KV-cache usage, queue depth, preemptions and prefix-cache hit rate.
    Start every investigation here."""
    end = time.time()
    start = end - window_minutes * 60
    p = clients.prod()
    records, samples = await asyncio.gather(_call(p.requests(start, end)), engine_samples(start, end))
    return {"window": {"start": iso(start), "end": iso(end)}, "client_side": window_summary(records, start, end),
            "engine": engine_summary(samples), "policy_slos": policy()["slos"]}


@mcp.tool(annotations=OBSERVE)
async def compare_windows(baseline_start: str, baseline_end: str, incident_start: str,
                          incident_end: str = "now") -> dict[str, Any]:
    """Compare a healthy baseline window with the incident window across everything that could
    explain a regression: offered load (rps), traffic shape (prompt-token distribution, long-prompt
    share), latency by prompt length, throughput, goodput, engine state (KV cache, queue,
    preemptions, prefix-cache hits), GPU memory, and any config deploys in between.
    Use it to decide whether the cause is a config change, more users, different traffic, or a mix."""
    b0, b1, i0, i1 = (parse_time(x) for x in (baseline_start, baseline_end, incident_start, incident_end))
    p = clients.prod()
    rb, ri, eb, ei, gb, gi, hist = await asyncio.gather(
        _call(p.requests(b0, b1)), _call(p.requests(i0, i1)), engine_samples(b0, b1),
        engine_samples(i0, i1), _call(p.gpu(b0, b1)), _call(p.gpu(i0, i1)), _call(p.history()))
    base, inc = window_summary(rb, b0, b1), window_summary(ri, i0, i1)
    eng_b, eng_i = engine_summary(eb), engine_summary(ei)
    deltas = {
        "offered_rps_pct": pct_change(base["offered_rps"], inc["offered_rps"]),
        "p95_ttft_pct": pct_change(base["latency"]["p95_ttft_ms"], inc["latency"]["p95_ttft_ms"]),
        "p95_e2e_pct": pct_change(base["latency"]["p95_e2e_ms"], inc["latency"]["p95_e2e_ms"]),
        "output_tokens_per_s_pct": pct_change(base["output_tokens_per_s"], inc["output_tokens_per_s"]),
        "goodput_rps_pct": pct_change(base["goodput_rps"], inc["goodput_rps"]),
        "prompt_tokens_p50_pct": pct_change(base["prompt_tokens"]["p50"], inc["prompt_tokens"]["p50"]),
        "long_prompt_share": [base["prompt_tokens"]["long_share"], inc["prompt_tokens"]["long_share"]],
        "kv_cache_usage_mean": [eng_b.get("kv_cache_usage_mean"), eng_i.get("kv_cache_usage_mean")],
        "waiting_mean": [eng_b.get("waiting_mean"), eng_i.get("waiting_mean")],
        "preemptions": [eng_b.get("preemptions"), eng_i.get("preemptions")],
    }
    changes = [{k: h.get(k) for k in ("version", "author", "message", "config_hash", "status")} | {"ts": iso(h["ts"])}
               for h in hist if b0 <= h["ts"] <= i1]
    return {
        "baseline": {"window": [iso(b0), iso(b1)], "client_side": base, "engine": eng_b, "gpu": gpu_window(gb.get("samples", []))},
        "incident": {"window": [iso(i0), iso(i1)], "client_side": inc, "engine": eng_i, "gpu": gpu_window(gi.get("samples", []))},
        "deltas": deltas,
        "config_deploys_between_windows": changes,
        "note": "Engine counters (preemptions) are deltas within each window. Compute any further statistics in code.",
    }


@mcp.tool(annotations=OBSERVE)
async def get_request_log(start: str, end: str = "now", limit: int = 5000) -> dict[str, Any]:
    """Raw per-request production records (client-side truth): t_start, prompt_tokens, output_tokens,
    ttft_ms, e2e_ms, status. Large. Intended for analysis in code (Code Mode), e.g. p95 TTFT
    bucketed by prompt length, or latency over time."""
    t0, t1 = parse_time(start), parse_time(end)
    records = await _call(clients.prod().requests(t0, t1, limit))
    return {"window": [iso(t0), iso(t1)], "count": len(records), "records": records}


@mcp.tool(annotations=OBSERVE)
async def get_engine_metrics(start: str, end: str = "now", max_points: int = 120) -> dict[str, Any]:
    """Production vLLM engine time series (from Prometheus when configured, else the node sampler):
    KV-cache usage, running and waiting sequences, preemptions/s, prefix-cache hit rate,
    generation and prompt tokens/s."""
    t0, t1 = parse_time(start), parse_time(end)
    samples = await engine_samples(t0, t1)
    points = []
    for a, b in zip(samples, samples[1:]):
        dt = max(1e-6, b["t"] - a["t"])

        def rate(k: str) -> float | None:
            return None if a.get(k) is None or b.get(k) is None else round((b[k] - a[k]) / dt, 3)

        hq = rate("prefix_queries_total")
        points.append({"t": iso(b["t"]), "kv_cache_usage": b.get("kv_cache_usage"), "running": b.get("running"),
                       "waiting": b.get("waiting"), "preemptions_per_s": rate("preemptions_total"),
                       "prefix_hit_rate": round(rate("prefix_hits_total") / hq, 3) if hq else b.get("prefix_hit_rate"),
                       "generation_tokens_per_s": rate("generation_tokens_total"),
                       "prompt_tokens_per_s": rate("prompt_tokens_total")})
    step = max(1, len(points) // max(1, max_points))
    return {"window": [iso(t0), iso(t1)], "points": points[::step], "summary": engine_summary(samples)}


@mcp.tool(annotations=OBSERVE)
async def query_prometheus(promql: str, start: str = "-30m", end: str = "now", step_s: float | None = None,
                           max_points: int = 200) -> dict[str, Any]:
    """Run any PromQL range query against the monitoring Prometheus. Prod vLLM is job="vllm-prod",
    shadow is job="vllm-shadow", GPUs are ff_gpu_* metrics. Examples:
    rate(vllm:num_preemptions_total{job="vllm-prod"}[1m]),
    histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{job="vllm-prod"}[2m]))).
    Metric names vary by vLLM version; discover them with e.g. {__name__=~"vllm:.*cache.*"}."""
    if not prom.enabled():
        raise ToolError("Prometheus is not configured (PROMETHEUS_URL); use get_engine_metrics instead")
    t0, t1 = parse_time(start), parse_time(end)
    try:
        result = await prom.query_range(promql, t0, t1, step_s)
    except prom.PromError as e:
        raise ToolError(str(e)) from None
    series = []
    for r in result[:20]:
        vals = r["values"]
        stride = max(1, len(vals) // max(1, max_points))
        series.append({"labels": r["metric"], "points": [[iso(float(t)), float(v)] for t, v in vals[::stride]]})
    return {"query": promql, "window": [iso(t0), iso(t1)], "series": series, "series_total": len(result)}


@mcp.tool(annotations=OBSERVE)
async def get_serving_config(target: str = "prod") -> dict[str, Any]:
    """Current serving config for 'prod' or 'shadow': every vLLM knob, the exact launch command
    (the live 'unit'), version, when/why it was deployed, container health and in-flight requests."""
    c = clients.prod() if target == "prod" else clients.shadow() if target == "shadow" else None
    if c is None:
        raise ToolError("target must be 'prod' or 'shadow'")
    info = await _call(c.info())
    if info.get("active_version"):
        info["active_version"]["ts"] = iso(info["active_version"]["ts"])
    info["knobs"] = vc.knob_docs()
    return info


@mcp.tool(annotations=OBSERVE)
async def get_change_history(limit: int = 20) -> dict[str, Any]:
    """Production deploy history: each config version with time, author, message, the knob diff
    against the previous version, and whether it is active, superseded or failed."""
    hist = await _call(clients.prod().history())
    out, prev = [], None
    for h in hist:
        out.append({"version": h["version"], "ts": iso(h["ts"]), "author": h["author"], "message": h["message"],
                    "status": h["status"], "config_hash": h["config_hash"],
                    "diff_vs_previous": vc.diff(prev["config"], h["config"]) if prev else "initial",
                    "config": h["config"]})
        if h["status"] != "failed":
            prev = h
    return {"versions": out[-limit:], "now": iso(time.time())}


@mcp.tool(annotations=OBSERVE)
async def get_logs(target: str = "prod", tail: int = 300, grep: str | None = None) -> dict[str, Any]:
    """vLLM server logs for 'prod' or 'shadow' (engine startup summary incl. KV-cache size/blocks,
    warnings, preemption messages, errors). `grep` is a case-insensitive regex filter."""
    c = clients.prod() if target == "prod" else clients.shadow()
    return await _call(c.logs(min(tail, 2000), grep))


@mcp.tool(annotations=OBSERVE)
async def get_gpu_status(target: str = "prod") -> dict[str, Any]:
    """Live nvidia-smi for the prod or shadow GPU: memory used/total, utilization, temperature, power."""
    c = clients.prod() if target == "prod" else clients.shadow()
    return await _call(c.gpu())


@mcp.tool(annotations=OBSERVE)
def get_policy() -> dict[str, Any]:
    """SLO thresholds, quality gate, evidence rules for production changes, constraints, and the
    allowlisted vLLM knobs with their allowed ranges."""
    return {**policy(), "knobs": vc.knob_docs()}


@mcp.tool(annotations=OBSERVE)
def list_experiments() -> dict[str, Any]:
    """Every shadow experiment so far (compact): id, hypothesis, config diff vs production at the
    time, rate multiplier, key results and SLO verdict. Use it to build the evidence table."""
    rows = []
    for r in evidence.all_("runs"):
        s = r.get("summary") or {}
        rows.append({"run_id": r["id"], "kind": r["kind"], "created_at": iso(r["created_at"]),
                     "hypothesis": r.get("hypothesis"), "diff_vs_prod": r.get("diff_vs_prod"),
                     "config_hash": r["config_hash"], "matches_prod_config": r.get("matches_prod_config"),
                     "workload_id": r.get("workload_id"), "rate_multiplier": r.get("rate_multiplier"),
                     "p95_ttft_ms": (s.get("latency") or {}).get("p95_ttft_ms"),
                     "p95_e2e_ms": (s.get("latency") or {}).get("p95_e2e_ms"),
                     "goodput_ratio": s.get("goodput_ratio"), "error_rate": s.get("error_rate"),
                     "output_tokens_per_s": s.get("output_tokens_per_s"),
                     "preemptions": (r.get("engine") or {}).get("preemptions"),
                     "accuracy": r.get("accuracy"), "all_slos_pass": (r.get("slo_evaluation") or {}).get("all_pass")})
    return {"experiments": rows}


@mcp.tool(annotations=OBSERVE)
def get_experiment(run_id: str, include_records: bool = True) -> dict[str, Any]:
    """Full detail of one experiment, optionally with its per-request records for analysis in code."""
    try:
        run = evidence.load("runs", run_id)
    except KeyError as e:
        raise ToolError(str(e)) from None
    if not include_records:
        run.pop("records", None)
    return run


# --------------------------------------------------------------------------- SHADOW
@mcp.tool(annotations=SHADOW)
async def capture_workload(start: str, end: str = "now", max_requests: int = 600,
                           min_prompt_tokens: int | None = None, max_prompt_tokens: int | None = None
                           ) -> dict[str, Any]:
    """Capture real production requests from a time window as a replayable workload (same prompts,
    output limits and arrival times). Optional prompt-token filters let you isolate e.g. only
    long-context traffic (note: filtering lowers the arrival rate). Returns a workload_id."""
    t0, t1 = parse_time(start), parse_time(end)
    records = await _call(clients.prod().requests(t0, t1))
    rows = [r for r in records if r.get("prompt_id")
            and (min_prompt_tokens is None or (r.get("prompt_tokens") or 0) >= min_prompt_tokens)
            and (max_prompt_tokens is None or (r.get("prompt_tokens") or 0) <= max_prompt_tokens)]
    if not rows:
        raise ToolError("no production requests match that window/filter")
    rows = rows[-max_requests:]  # contiguous tail keeps the real arrival rate
    first = rows[0]["t_start"]
    items = [{"prompt_id": r["prompt_id"], "max_tokens": r["max_tokens"], "offset_s": round(r["t_start"] - first, 3)}
             for r in rows]
    span = max(1.0, items[-1]["offset_s"])
    pt = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
    wl = {"id": evidence.new_id("wl"), "created_at": time.time(), "source": "prod", "window": [t0, t1],
          "filters": {"min_prompt_tokens": min_prompt_tokens, "max_prompt_tokens": max_prompt_tokens},
          "items": items, "n": len(items), "span_s": round(span, 1), "arrival_rps": round(len(items) / span, 3),
          "prompt_tokens": {"p50": stats.percentile(pt, 50), "p95": stats.percentile(pt, 95),
                            "long_share": round(sum(1 for x in pt if x >= long_threshold()) / len(pt), 3) if pt else None}}
    evidence.save("workloads", wl)
    return {k: v for k, v in wl.items() if k != "items"} | {"window": [iso(t0), iso(t1)]}


async def _wait_deploy(c: clients.Controller, dep: dict[str, Any], max_wait_s: float) -> dict[str, Any]:
    deadline = time.time() + max_wait_s
    while dep["status"] in ("queued", "stopping", "starting", "rolling_back") and time.time() < deadline:
        await asyncio.sleep(5)
        dep = await _call(c.deploy_status(dep["id"]))
    return dep


@mcp.tool(annotations=SHADOW)
async def deploy_shadow(changes: dict[str, Any], reason: str, base: str = "prod") -> dict[str, Any]:
    """Launch a candidate vLLM config on the SHADOW GPU (never production). `changes` is a partial
    config applied on top of `base` ('prod' = the current production config, 'shadow' = the
    current shadow config); pass {} to reproduce production exactly. Waits until the server is
    healthy (~1-3 min). If the engine fails to start (e.g. not enough memory) you get the log tail."""
    if base not in ("prod", "shadow"):
        raise ToolError("base must be 'prod' or 'shadow'")
    prod_info = await prod_active()
    base_cfg = prod_info["active_version"]["config"]
    if base == "shadow":
        sh = await _call(clients.shadow().info())
        base_cfg = (sh.get("active_version") or {}).get("config") or base_cfg
    try:
        cfg = vc.apply_patch(base_cfg, changes)
    except vc.ConfigError as e:
        raise ToolError(f"rejected by guardrails: {e}") from None
    c = clients.shadow()
    dep = await _call(c.deploy(cfg, author="inference-firefighter", message=reason, auto_rollback=False))
    dep = await _wait_deploy(c, dep, DEPLOY_WAIT_S)
    return {"deploy_id": dep["id"], "status": dep["status"], "config": cfg,
            "config_hash": dep.get("config_hash"),
            "diff_vs_prod": vc.diff(prod_info["active_version"]["config"], cfg),
            "startup_s": dep.get("downtime_s"), "failure_log_tail": dep.get("failure_log_tail"),
            "hint": "still starting: call wait_for_shadow" if dep["status"] in ("starting", "stopping") else None}


@mcp.tool(annotations=OBSERVE)
async def wait_for_shadow(deploy_id: str, max_wait_s: float = 300) -> dict[str, Any]:
    """Wait for a shadow deploy started by deploy_shadow to finish."""
    c = clients.shadow()
    dep = await _wait_deploy(c, await _call(c.deploy_status(deploy_id)), min(max_wait_s, 600))
    return {k: dep.get(k) for k in ("id", "status", "config_hash", "downtime_s", "failure_log_tail", "error")}


@mcp.tool(annotations=SHADOW)
async def run_load_test(workload_id: str, hypothesis: str, rate_multiplier: float = 1.0,
                        duration_s: float = 60) -> dict[str, Any]:
    """Replay a captured production workload open-loop against the config currently on the SHADOW
    GPU and measure it: client-side latency/throughput/goodput (overall and by prompt length),
    engine KV-cache usage, queue depth, preemptions, prefix-cache hit rate, and a pass/fail per
    SLO. `hypothesis`: state what you expect this config to do and why BEFORE measuring; the
    result is recorded as evidence next to it. rate_multiplier 2.0 = twice the real arrival rate."""
    if not hypothesis.strip():
        raise ToolError("state a hypothesis (what you expect and why) before running an experiment")
    try:
        wl = evidence.load("workloads", workload_id)
    except KeyError as e:
        raise ToolError(str(e)) from None
    prod_info = await prod_active()
    res = await _call(clients.shadow().loadtest({"items": wl["items"], "rate_multiplier": rate_multiplier,
                                                  "duration_s": duration_s}))
    records = res.pop("records")
    summary = stats.summarize(records, duration_s=res["issued_duration_s"], slo=policy()["slos"],
                              long_prompt_threshold=long_threshold())
    slo_eval = stats.evaluate_slos(summary, policy()["slos"])
    prod_cfg = prod_info["active_version"]["config"]
    run = {"id": evidence.new_id("run"), "kind": "load_test", "created_at": time.time(), "hypothesis": hypothesis,
           "target": "shadow", "config": res["config"], "config_hash": res["config_hash"],
           "matches_prod_config": res["config_hash"] == prod_info["active_version"]["config_hash"],
           "diff_vs_prod": vc.diff(prod_cfg, res["config"] or {}), "workload_id": workload_id,
           "workload_captured_at": wl["created_at"], "rate_multiplier": rate_multiplier,
           "duration_s": duration_s, "summary": summary, "slo_evaluation": slo_eval,
           "engine": res["engine"], "records": records}
    evidence.save("runs", run)
    return {k: v for k, v in run.items() if k != "records"} | {
        "note": "Per-request records: get_experiment(run_id). Accept or reject your hypothesis explicitly."}


@mcp.tool(annotations=SHADOW)
async def run_quality_eval(n: int = 40) -> dict[str, Any]:
    """Quality gate on the SHADOW config: golden long-context prompts at temperature 0, each with one
    exact answer hidden in a long document. Returns accuracy and every output. Required before any
    production change to KV-cache precision (e.g. kv_cache_dtype=fp8). Run it on the current
    production config too, to get the baseline."""
    res = await _call(clients.shadow().quality(n))
    prod_info = await prod_active()
    run = {"id": evidence.new_id("qual"), "kind": "quality", "created_at": time.time(), "target": "shadow",
           "config": res["config"], "config_hash": res["config_hash"],
           "matches_prod_config": res["config_hash"] == prod_info["active_version"]["config_hash"],
           "diff_vs_prod": vc.diff(prod_info["active_version"]["config"], res["config"] or {}),
           "accuracy": res["accuracy"], "rows": res["rows"]}
    evidence.save("runs", run)
    return run


# --------------------------------------------------------------------------- PROD
async def _plan(changes: dict[str, Any], evidence_run_ids: list[str]) -> dict[str, Any]:
    pol = policy()
    info = await prod_active()
    cur = info["active_version"]
    try:
        cfg = vc.apply_patch(cur["config"], changes)
    except vc.ConfigError as e:
        raise ToolError(f"rejected by guardrails: {e}") from None
    model_hash = vc.config_hash(cfg, info["model"])
    diff = vc.diff(cur["config"], cfg)
    checks: list[dict[str, Any]] = [{"check": "config differs from production", "pass": bool(diff)}]

    runs = []
    for rid in evidence_run_ids:
        try:
            runs.append(evidence.load("runs", rid))
        except KeyError:
            checks.append({"check": f"evidence {rid} exists", "pass": False})
    ev = pol["evidence"]
    qualifying = [r for r in runs if r["kind"] == "load_test" and r["config_hash"] == model_hash
                  and r["slo_evaluation"]["all_pass"]
                  and r["rate_multiplier"] >= ev["min_rate_multiplier"] and r["duration_s"] >= ev["min_duration_s"]
                  and time.time() - r["workload_captured_at"] <= ev["max_age_minutes"] * 60]
    checks.append({"check": ("a cited shadow load test of this exact config passed every SLO on production traffic "
                             f"captured in the last {ev['max_age_minutes']} min at >= {ev['min_rate_multiplier']}x "
                             f"rate for >= {ev['min_duration_s']}s"),
                   "pass": bool(qualifying), "qualifying_runs": [r["id"] for r in qualifying]})
    if cfg["kv_cache_dtype"] != cur["config"]["kv_cache_dtype"]:
        q = pol["quality"]
        quals = [r for r in runs if r["kind"] == "quality" and r["config_hash"] == model_hash]
        base = [r for r in evidence.all_("runs") if r["kind"] == "quality" and r["config_hash"] == cur["config_hash"]]
        best = max((r["accuracy"] for r in quals), default=None)
        base_acc = max((r["accuracy"] for r in base), default=None)
        ok = best is not None and best >= q["min_accuracy"] and (
            base_acc is None or base_acc - best <= q["max_drop_vs_baseline"])
        checks.append({"check": f"KV-cache precision changes -> cited quality eval with accuracy >= {q['min_accuracy']} "
                                f"and drop vs production baseline <= {q['max_drop_vs_baseline']}",
                       "pass": ok, "candidate_accuracy": best, "prod_baseline_accuracy": base_acc})
    last_downtime = cur.get("downtime_s")
    blast = (f"Restarts the production vLLM container on GPU {info['gpu']} (v{cur['version']} -> new version). "
             f"The endpoint is DOWN until the new engine is healthy: last measured restart took "
             f"{last_downtime}s. ~{int(info.get('requests_running_now') or 0)} in-flight requests will fail and "
             f"new requests error until healthy. If the new engine fails its health check, the controller "
             f"automatically restarts v{cur['version']}; rollback_production can also restore it.")
    return {"current_version": cur["version"], "current_config": cur["config"], "new_config": cfg,
            "new_config_hash": model_hash, "diff": diff, "checks": checks,
            "ready": all(c["pass"] for c in checks), "blast_radius": blast,
            "expected_downtime_s": last_downtime}


@mcp.tool(annotations=OBSERVE)
async def plan_production_change(changes: dict[str, Any], evidence_run_ids: list[str]) -> dict[str, Any]:
    """Dry run of a production change (changes nothing): the exact diff vs the live config, which
    precondition checks pass or fail for the cited evidence, and the blast radius of restarting
    the live server. Call this before apply_production_config and show the result to the human."""
    return await _plan(changes, evidence_run_ids)


@mcp.tool(annotations=PROD)
async def apply_production_config(changes: dict[str, Any], evidence_run_ids: list[str], justification: str,
                                  blast_radius: str, expected_result: str) -> dict[str, Any]:
    """IRREVERSIBLE-ACTION GATE: restarts the production vLLM server with a new config. Requires human
    approval. Only call after plan_production_change reports ready=true. The arguments are what the
    approver reads: `changes` (knob diff), `evidence_run_ids` (the passing shadow runs),
    `justification` (root cause and why this fixes it), `blast_radius` (downtime/in-flight impact
    from the plan), `expected_result` (before -> after numbers). The server re-checks the evidence
    and refuses unproven configs even after approval."""
    plan = await _plan(changes, evidence_run_ids)
    if not plan["ready"]:
        failed = [c for c in plan["checks"] if not c["pass"]]
        raise ToolError(f"refused: preconditions not met: {failed}")
    audit = await clients.github_commit(plan["new_config"], f"firefighter: {justification[:72]}",
                                        {"evidence": ", ".join(evidence_run_ids), "expected": expected_result})
    c = clients.prod()
    dep = await _call(c.deploy(plan["new_config"], author="inference-firefighter (human-approved)",
                               message=f"{justification} | evidence: {', '.join(evidence_run_ids)}", auto_rollback=True))
    dep = await _wait_deploy(c, dep, PROD_DEPLOY_WAIT_S)
    return {"status": dep["status"], "version": dep.get("version"), "downtime_s": dep.get("downtime_s"),
            "diff": plan["diff"], "audit_commit": audit, "rolled_back_to": dep.get("rolled_back_to"),
            "failure_log_tail": dep.get("failure_log_tail"),
            "next": "Verify recovery: wait 2-3 minutes of live traffic, then get_slo_status(window_minutes=3)."}


@mcp.tool(annotations=PROD)
async def rollback_production(to_version: int, reason: str) -> dict[str, Any]:
    """IRREVERSIBLE-ACTION GATE: restart production on a previous known config version (see
    get_change_history). Requires human approval. Use if a production change did not recover SLOs."""
    c = clients.prod()
    hist = await _call(c.history())
    target = next((h for h in hist if h["version"] == to_version and h["status"] != "failed"), None)
    if not target:
        raise ToolError(f"no deployable version {to_version}")
    dep = await _call(c.deploy(target["config"], author="inference-firefighter (human-approved rollback)",
                               message=f"rollback to v{to_version}: {reason}", auto_rollback=True))
    dep = await _wait_deploy(c, dep, PROD_DEPLOY_WAIT_S)
    return {"status": dep["status"], "version": dep.get("version"), "downtime_s": dep.get("downtime_s")}


# --------------------------------------------------------------------------- serve
class BearerAuth:
    """Static bearer token (TrueForge connector 'header auth'). Lifespan scopes pass through."""

    def __init__(self, app: Any, token: str):
        self.app, self.expected = app, f"Bearer {token}".encode()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and self.expected and dict(scope["headers"]).get(b"authorization") != self.expected:
            await send({"type": "http.response.start", "status": 401,
                        "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"unauthorized"})
            return
        await self.app(scope, receive, send)


def build_app() -> Any:
    token = os.environ.get("MCP_AUTH_TOKEN", "")
    if not token and os.environ.get("ALLOW_NO_AUTH") != "1":
        raise SystemExit("MCP_AUTH_TOKEN is required (ALLOW_NO_AUTH=1 only for local dev)")
    app = mcp.streamable_http_app(
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
    return BearerAuth(app, token)


if __name__ == "__main__":
    uvicorn.run(build_app(), host=os.environ.get("MCP_HOST", "0.0.0.0"), port=int(os.environ.get("MCP_PORT", "8765")))
