"""One-image story of an incident from real Prometheus data (queried through Grafana at :3001).

  python -m docs.summary_chart --start 2026-09-26T11:14:00Z --end 2026-09-26T12:10:00Z \
      --phase "chaos: long-context traffic=2026-09-26T11:23:00Z" --phase "agent starts=..." \
      --phase "fix approved + applied=..." --out docs/screenshots/05-summary.png
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import httpx
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

GRAFANA = "http://localhost:3001"
JOB = "vllm-prod"
SERIES = [
    ("p95 time to first token (s)", f'histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{{job="{JOB}"}}[1m])))', 8, "SLO 8 s"),
    ("p95 end-to-end latency (s)", f'histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{{job="{JOB}"}}[1m])))', 40, "SLO 40 s"),
    ("throughput: output tokens/s", f'sum(rate(vllm:generation_tokens_total{{job="{JOB}"}}[1m]))', None, None),
    ("KV-cache usage (%)", f'100 * max(vllm:kv_cache_usage_perc{{job="{JOB}"}})', None, None),
    ("requests running / waiting for KV", [f'sum(vllm:num_requests_running{{job="{JOB}"}})',
                                           f'sum(vllm:num_requests_waiting_by_reason{{job="{JOB}",reason="capacity"}})'], None, None),
    ("p95 prompt length (tokens)", f'histogram_quantile(0.95, sum by (le) (rate(vllm:request_prompt_tokens_bucket{{job="{JOB}"}}[2m])))', None, None),
]


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def query(expr: str, start: float, end: float, step: int = 15) -> tuple[list[datetime], list[float]]:
    body = {"queries": [{"refId": "A", "datasource": {"type": "prometheus", "uid": "prom"}, "expr": expr,
                         "range": True, "intervalMs": step * 1000, "maxDataPoints": 2000}],
            "from": str(int(start * 1000)), "to": str(int(end * 1000))}
    r = httpx.post(f"{GRAFANA}/api/ds/query", json=body, timeout=60).json()
    frames = r["results"]["A"].get("frames") or []
    if not frames:
        return [], []
    t, v = frames[0]["data"]["values"][:2]
    pts = [(datetime.fromtimestamp(a / 1000, timezone.utc), b) for a, b in zip(t, v) if b is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--phase", action="append", default=[], help="label=ISO time")
    ap.add_argument("--out", default="docs/screenshots/05-summary.png")
    a = ap.parse_args()
    start, end = ts(a.start), ts(a.end)
    phases = [(lbl, datetime.fromtimestamp(ts(t), timezone.utc)) for lbl, t in (p.rsplit("=", 1) for p in a.phase)]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(len(SERIES), 1, figsize=(14, 17), sharex=True)
    fig.suptitle("Inference Firefighter on a real NVIDIA L4 (Qwen2.5-7B, vLLM): baseline → chaos → agent fix",
                 fontsize=15, y=0.995)
    colors = ["#73bf69", "#f2cc0c", "#5794f2", "#ff7383", "#b877d9"]
    for ax, (title, expr, slo, slo_lbl) in zip(axes, SERIES):
        exprs = expr if isinstance(expr, list) else [expr]
        labels = ["running", "waiting for KV capacity"] if isinstance(expr, list) else [None]
        for i, (e, lbl) in enumerate(zip(exprs, labels)):
            x, y = query(e, start, end)
            ax.plot(x, y, color=colors[i], lw=1.8, label=lbl)
        if slo:
            ax.axhline(slo, color="#ff5f5f", ls="--", lw=1.2, label=slo_lbl)
        for lbl, t in phases:
            ax.axvline(t, color="#bbbbbb", ls=":", lw=1.2)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(alpha=0.2)
        if slo or isinstance(expr, list):
            ax.legend(loc="upper left", fontsize=9)
    for i, (lbl, t) in enumerate(phases):
        axes[0].annotate(lbl, xy=(t, 1.02 + 0.13 * (i % 2)), xycoords=("data", "axes fraction"), fontsize=9,
                         ha="left", va="bottom", color="#dddddd", annotation_clip=False)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M UTC"))
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(a.out, dpi=130)
    print(a.out)


if __name__ == "__main__":
    main()
