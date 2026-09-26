"""One streaming chat-completion request against an OpenAI-compatible server, measured client-side."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import httpx


async def send(client: httpx.AsyncClient, base_url: str, model: str, prompt_id: str, prompt: dict[str, Any],
               max_tokens: int | None = None, timeout_s: float = 120.0, temperature: float = 0.7,
               keep_text: bool = False) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "request_id": uuid.uuid4().hex[:12], "prompt_id": prompt_id, "t_start": time.time(), "t_end": None,
        "max_tokens": max_tokens or prompt["max_tokens"], "prompt_tokens": None, "output_tokens": None,
        "ttft_ms": None, "e2e_ms": None, "status": "ok", "error": None,
    }
    body = {
        "model": model, "messages": prompt["messages"], "max_tokens": rec["max_tokens"],
        "temperature": temperature, "stream": True, "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    chunks: list[str] = []
    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=body, timeout=timeout_s) as resp:
            if resp.status_code != 200:
                text = (await resp.aread()).decode(errors="replace")[:300]
                rec.update(status="error", error=f"HTTP {resp.status_code}: {text}")
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    for choice in event.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content")
                        if piece:
                            if rec["ttft_ms"] is None:
                                rec["ttft_ms"] = round((time.perf_counter() - t0) * 1000, 1)
                            if keep_text:
                                chunks.append(piece)
                    usage = event.get("usage")
                    if usage:
                        rec["prompt_tokens"] = usage.get("prompt_tokens")
                        rec["output_tokens"] = usage.get("completion_tokens")
    except httpx.TimeoutException:
        rec.update(status="timeout", error=f"no completion within {timeout_s}s")
    except httpx.HTTPError as e:
        rec.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
    rec["e2e_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    rec["t_end"] = time.time()
    if rec["status"] == "ok" and rec["ttft_ms"] is None:
        rec["ttft_ms"] = rec["e2e_ms"]  # zero-token completion
    if keep_text:
        rec["text"] = "".join(chunks)
    return rec
