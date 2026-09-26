"""LOCAL PLUMBING TESTS ONLY — a crude stand-in for vLLM so the controller, loadtest and MCP
server can be exercised on a laptop without a GPU.

Its latency model is a toy. NEVER use it for calibration, evidence, or the demo; the hackathon
requires a real system and the whole point of the project is real KV-cache behavior.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="qwen2.5-7b")
ap.add_argument("--config", default="{}")
args = ap.parse_args()
cfg = json.loads(args.config)

KV_CAPACITY = int(80_000 * (cfg.get("gpu_memory_utilization", 0.9) / 0.9)
                  * (2 if str(cfg.get("kv_cache_dtype", "auto")).startswith("fp8") else 1))
MAX_SEQS = int(cfg.get("max_num_seqs", 256))
PREFIX = bool(cfg.get("enable_prefix_caching", True))
state = {"running": 0, "waiting": 0, "kv_tokens": 0, "preemptions": 0, "gen": 0, "prompt": 0,
         "hits": 0, "queries": 0, "success": 0}
slots = asyncio.Semaphore(MAX_SEQS)
app = FastAPI()


def ntok(messages: list[dict[str, str]]) -> tuple[int, int]:
    system = sum(len(m["content"]) for m in messages if m["role"] == "system") // 4
    other = sum(len(m["content"]) for m in messages if m["role"] != "system") // 4
    return system, other


@app.get("/health")
async def health() -> PlainTextResponse:
    return PlainTextResponse("")


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    lines = [
        f'vllm:kv_cache_usage_perc{{model_name="{args.model}"}} {min(1.0, state["kv_tokens"] / KV_CAPACITY):.4f}',
        f'vllm:num_requests_running{{model_name="{args.model}"}} {state["running"]}',
        f'vllm:num_requests_waiting{{model_name="{args.model}"}} {state["waiting"]}',
        f'vllm:num_preemptions_total{{model_name="{args.model}"}} {state["preemptions"]}',
        f'vllm:prefix_cache_hits_total{{model_name="{args.model}"}} {state["hits"]}',
        f'vllm:prefix_cache_queries_total{{model_name="{args.model}"}} {state["queries"]}',
        f'vllm:generation_tokens_total{{model_name="{args.model}"}} {state["gen"]}',
        f'vllm:prompt_tokens_total{{model_name="{args.model}"}} {state["prompt"]}',
        f'vllm:request_success_total{{model_name="{args.model}",finished_reason="stop"}} {state["success"]}',
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/v1/chat/completions")
async def chat(req: Request) -> StreamingResponse:
    body = await req.json()
    system, other = ntok(body["messages"])
    prompt = system + other
    new_tokens = other + (0 if PREFIX else system)
    out = int(body.get("max_tokens", 64))
    if prompt + out > int(cfg.get("max_model_len", 32768)):
        return StreamingResponse(iter([b""]), status_code=400)

    async def gen():  # type: ignore[no-untyped-def]
        state["waiting"] += 1
        async with slots:
            state["waiting"] -= 1
            state["running"] += 1
            state["kv_tokens"] += prompt
            state["queries"] += prompt
            state["hits"] += prompt - new_tokens
            overload = max(0.0, state["kv_tokens"] / KV_CAPACITY - 1.0)
            if overload > 0:
                state["preemptions"] += 1
            await asyncio.sleep(new_tokens / 20000 * (1 + 4 * overload) + 0.02)
            state["prompt"] += prompt
            for i in range(out):
                chunk = {"choices": [{"delta": {"content": "K7-1234 " if i == 0 else "x"}}]}
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                state["gen"] += 1
                await asyncio.sleep(0.004 * (1 + 4 * overload) * (1 + state["running"] / 64))
            state["running"] -= 1
            state["kv_tokens"] -= prompt
            state["success"] += 1
        usage = {"usage": {"prompt_tokens": prompt, "completion_tokens": out}, "choices": []}
        yield f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode()

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    print(f"fake vllm (NOT REAL) port={args.port} cfg={cfg} kv_capacity={KV_CAPACITY}", flush=True)
    time.sleep(1.0)  # pretend to load weights
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
