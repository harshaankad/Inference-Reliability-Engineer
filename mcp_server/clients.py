"""HTTP clients for the node controllers (prod and shadow) and optional GitHub audit commits."""
from __future__ import annotations

import asyncio
import base64
import os
from datetime import datetime, timezone
from typing import Any

import httpx
import yaml


class ControllerError(RuntimeError):
    pass


class Controller:
    def __init__(self, url: str, token: str, slot: str):
        self.url = url.rstrip("/")
        self.slot = slot
        self.headers = {"Authorization": f"Bearer {token}"}

    async def _req(self, method: str, path: str, timeout: float = 30.0, **kw: Any) -> Any:
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.request(method, f"{self.url}{path}", headers=self.headers, **kw)
        except httpx.HTTPError as e:
            raise ControllerError(f"controller {self.url} unreachable: {type(e).__name__}: {e}") from None
        if r.status_code >= 400:
            raise ControllerError(f"controller {path} -> HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    async def info(self) -> dict[str, Any]:
        return await self._req("GET", f"/slots/{self.slot}")

    async def history(self) -> list[dict[str, Any]]:
        return await self._req("GET", f"/slots/{self.slot}/history")

    async def deploy(self, config: dict[str, Any], author: str, message: str, auto_rollback: bool = True
                     ) -> dict[str, Any]:
        return await self._req("POST", f"/slots/{self.slot}/deploy",
                               json={"config": config, "author": author, "message": message,
                                     "auto_rollback": auto_rollback})

    async def deploy_status(self, deploy_id: str) -> dict[str, Any]:
        return await self._req("GET", f"/deploys/{deploy_id}")

    async def logs(self, tail: int, grep: str | None) -> dict[str, Any]:
        params: dict[str, Any] = {"tail": tail}
        if grep:
            params["grep"] = grep
        return await self._req("GET", f"/slots/{self.slot}/logs", params=params)

    async def engine_series(self, start: float, end: float) -> list[dict[str, Any]]:
        return (await self._req("GET", f"/slots/{self.slot}/engine/series",
                                params={"start": start, "end": end}))["samples"]

    async def gpu(self, start: float | None = None, end: float | None = None) -> dict[str, Any]:
        params = {"start": start, "end": end} if start is not None else None
        return await self._req("GET", "/gpu", params=params)

    async def requests(self, start: float, end: float, limit: int = 20000) -> list[dict[str, Any]]:
        return (await self._req("GET", "/traffic/requests", timeout=60,
                                params={"start": start, "end": end, "limit": limit}))["records"]

    async def _job(self, path: str, body: dict[str, Any], max_wait_s: float = 1200) -> dict[str, Any]:
        """Start a background job on the controller and poll it (every request stays short)."""
        job = await self._req("POST", path, params={"background": "true"}, json=body)
        deadline = asyncio.get_running_loop().time() + max_wait_s
        while job["status"] == "running":
            if asyncio.get_running_loop().time() > deadline:
                raise ControllerError(f"job {job['id']} still running after {max_wait_s}s")
            await asyncio.sleep(5)
            job = await self._req("GET", f"/jobs/{job['id']}")
        if job["status"] != "done":
            raise ControllerError(f"job {job['id']} failed: {job.get('error')}")
        return job["result"]

    async def loadtest(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._job(f"/slots/{self.slot}/loadtest", body)

    async def quality(self, n: int) -> dict[str, Any]:
        return await self._job(f"/slots/{self.slot}/quality", {"n": n})


def prod() -> Controller:
    return Controller(os.environ.get("PROD_CONTROLLER_URL", "http://127.0.0.1:9000"),
                      os.environ.get("CONTROLLER_TOKEN", ""), os.environ.get("PROD_SLOT", "prod"))


def shadow() -> Controller:
    return Controller(os.environ.get("SHADOW_CONTROLLER_URL", "http://127.0.0.1:9001"),
                      os.environ.get("CONTROLLER_TOKEN", ""), os.environ.get("SHADOW_SLOT", "shadow"))


async def github_commit(config: dict[str, Any], message: str, meta: dict[str, Any]) -> dict[str, Any] | None:
    """Optional audit trail: write the new prod config to a GitHub repo. Skipped if not configured."""
    token, repo = os.environ.get("GITHUB_TOKEN"), os.environ.get("GITHUB_REPO")
    path = os.environ.get("GITHUB_CONFIG_PATH", "prod/vllm.yaml")
    if not token or not repo:
        return None
    body_yaml = (f"# Applied by inference-firefighter at {datetime.now(timezone.utc).isoformat()}\n"
                 + "".join(f"# {k}: {v}\n" for k, v in meta.items())
                 + yaml.safe_dump(config, sort_keys=False))
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    async with httpx.AsyncClient(timeout=30) as c:
        cur = await c.get(url, headers=headers)
        payload: dict[str, Any] = {"message": message, "content": base64.b64encode(body_yaml.encode()).decode()}
        if cur.status_code == 200:
            payload["sha"] = cur.json()["sha"]
        r = await c.put(url, headers=headers, json=payload)
    if r.status_code >= 300:
        return {"error": f"GitHub commit failed: HTTP {r.status_code} {r.text[:200]}"}
    commit = r.json()["commit"]
    return {"sha": commit["sha"], "url": commit["html_url"]}
