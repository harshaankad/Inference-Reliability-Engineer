# Inference Reliability Engineer: Roadmap (v3, as built)

**Hackathon:** Agents That Act, TrueFoundry × Polaris · **Harness:** TrueForge · **Cloud:** AWS ap-south-1 (Mumbai), organizer credits · **Agent LLM:** OpenAI, organizer credits

**One-liner:** Production inference is degraded and nobody knows why, because there is no error message: the config is legal and the workload changed. Our agent reproduces the failure, experiments with real serving configs on a real GPU, proves a fix with measurements, and then stops and asks before it touches the process serving live traffic.

> The README covers setup and usage. This file covers **what we decided, what is done, and what is left**.

---

## 1. Status at a glance

| Area | Status |
|---|---|
| Code: MCP server, node controller, load generator, replay harness, dataset, chaos tooling | ✅ Built and tested locally |
| TrueForge integration: connector registration + agent with approval gates | ✅ Tested against a real TrueForge server |
| Prometheus integration | ✅ Tested with a real Prometheus |
| AWS scripts: launch, secrets, deploy, tunnel, teardown, node setup | ✅ Written and syntax-checked; ⏳ not yet run on AWS |
| Real-GPU calibration of the incident | ⏳ Needs AWS |
| End-to-end agent run with OpenAI + Daytona | ⏳ Needs keys |
| Demo video, final README polish | ⏳ |
| Repo | ✅ Public: https://github.com/harshaankad/Inference-Reliability-Engineer |

---

## 2. Final design decisions (and what changed from v2)

| Topic | v2 plan | As built | Why |
|---|---|---|---|
| Scope | 7+ scenarios, scale-out, admission control, nginx, Grafana | **One flagship incident** done well, plus `surge` and `burst` traffic scenarios for classification | A one-day build; scope discipline beats breadth |
| GPUs | L4 (`g6`) or a 4-GPU box | **2 × `g5.xlarge` (A10G, 24 GB)** in ap-south-1: prod + shadow | What's available in Mumbai; only 8 vCPUs of quota needed |
| Served model | Qwen2.5-3B | **Qwen2.5-7B-Instruct** | ~15 GB of weights leaves only tens of thousands of tokens of KV cache on 24 GB, so long prompts cause **real** KV exhaustion |
| Incident | Bad config commit | **Legal-but-wrong config + traffic shift to long prompts; no config change** | This is the real shape of inference incidents: there's no error message |
| Root-cause mechanism | "`max_model_len` reserves memory" (from the pasted doc) | **Over-admission (`max_num_seqs: 256`) → KV exhaustion → preemption/recompute → queueing**, plus **prefix caching off** despite a shared system prompt | vLLM allocates KV blocks lazily (PagedAttention), so `max_model_len` does not reserve memory per sequence |
| Config source of truth | GitHub repo (GitOps) | **Controller version history** on the prod node; GitHub commit is **optional** (audit trail) | Fewer moving parts |
| Metrics | Prometheus + Grafana + DCGM | **Prometheus** (scrapes vLLM + `nvidia-smi` through the controller); controller sampler as fallback; charts come from TrueForge Generative UI | Our team already runs Prometheus; no dashboard to build |
| TrueForge hosting | Docker Compose hosted mode | **`npx` under systemd** on the control node, bound to 127.0.0.1, reached via SSM tunnel | No 10-minute image build; never exposed |
| Access | Tailscale / SSM | **SSM only**; security group allows intra-group traffic only | Zero public ports |
| AWS auth on instances | Instance role | **Instance role** + secrets in **SSM Parameter Store** (`/firefighter/*`) | No keys on disk in the repo, scripts or command history |
| Runbook | Git-backed skill | **Inlined into the instructions by default**; skill import optional (`--use-skill`) | TrueForge rejects unimported skills |

---

## 3. The incident (what the agent must find)

- **Before:** healthy short chat traffic (~2 rps, 5% long prompts). All SLOs pass.
- **Trigger:** `ops.sh chaos traffic long_context_shift`. Same rps, ~50% of requests become long RAG prompts (5–11k tokens, sharing one ~1.6k-token system prompt, each containing a "needle" fact).
- **Physics:** 7B weights (~15 GB) + KV cache must fit in 0.90 × 24 GB, so only ~10 long requests fit in the cache at once. `max_num_seqs: 256` admits far more, vLLM preempts and recomputes them, the queue grows and p95 TTFT explodes. Prefix caching is off, so the shared system prompt is re-prefilled every time.
- **Signals the agent should find (in code, in the sandbox):** same rps; prompt length ↑; TTFT bad **only on long prompts**; KV cache pinned; preemptions up; **no config change** in history.
- **Tempting wrong fix:** raise `max_num_batched_tokens`. Expected to admit more prefill into a full cache; **must be confirmed on the real GPU during calibration**.
- **Expected fix:** lower `max_num_seqs` (e.g. 16–32) + `enable_prefix_caching: true`. Optionally `kv_cache_dtype: fp8`, which is quality-gated and may not start on A10G, and a failed start is a valid result.

---

## 4. What's built (done)

### 4.1 Components
- [x] **`mcp_server/`**: `inference-ops` MCP server (MCP SDK v2 `MCPServer`, streamable HTTP, bearer auth). 19 tools:
  - Observe: `get_slo_status`, `compare_windows`, `get_request_log`, `get_engine_metrics`, `query_prometheus`, `get_serving_config`, `get_change_history`, `get_logs`, `get_gpu_status`, `get_policy`, `list_experiments`, `get_experiment`, `plan_production_change`, `wait_for_shadow`
  - Shadow: `capture_workload`, `deploy_shadow`, `run_load_test` (**hypothesis required**), `run_quality_eval`
  - **Gated**: `apply_production_config`, `rollback_production` (`destructiveHint` + named in `require_approval_for_tools`)
- [x] **Server-side guardrails:**
  - allowlisted knobs within safe ranges; the model can't be changed
  - evidence registry: an apply is refused unless a shadow run of the exact config hash passed every SLO on prod traffic captured in the last 90 minutes, at ≥ 1× rate, for ≥ 45 s
  - a quality eval is required for KV-precision changes
- [x] **Blast radius** computed by `plan_production_change`: last measured restart time, in-flight requests, auto-rollback target.
- [x] **`controller/`**: per-GPU-node service:
  - vLLM container lifecycle (Docker), with auto-rollback on a failed health check
  - version history, logs, 5 s engine sampler
  - Prometheus scrape endpoints (`/slots/<slot>/metrics`, `/gpu/metrics`)
  - replay load tests, quality eval, request log
- [x] **`workload/`**:
  - deterministic dataset (400 short, 300 long, 40 golden with exact answers)
  - Poisson load generator (the "users") writing a client-side request log; the scenario name is never exposed to the agent
  - open-loop replay harness that samples engine metrics every 1 s
- [x] **`agent/`**: instructions + `create_agent.py` (registers the MCP connector with header auth and creates/updates the agent via `trueforge-sdk`).
- [x] **`skills/incident-runbook/SKILL.md`**:
  - procedure: triage → what changed (in code) → classify → reproduce → experiment with hypotheses → prove → plan → gated apply → verify → report
  - includes the "reject your own hypothesis on evidence" rule
- [x] **`chaos/`**: operator-only scenario injection and reset.
- [x] **`infra/`**:
  - `aws/launch.sh` (VPC/SG/IAM/DLAMI/instances), `secrets.sh`, `deploy_code.sh` (S3 + SSM), `ssm.sh`, `tunnel.sh`, `teardown.sh`
  - `node/setup_gpu.sh`, `setup_control.sh`, `ops.sh`
  - Prometheus config template
- [x] **`dev/`**: fake engine + local stack. **For plumbing tests only; never used for calibration or the demo.**

### 4.2 Verified
- [x] 7 unit tests (config validation/diff/flags, metric parsing, stats/SLOs, dataset determinism, loadgen bursts).
- [x] MCP smoke test over real HTTP:
  - reproduce → candidate → evidence table → dry-run plan
  - refusals: no evidence, out-of-range knob, model change
  - gated apply with valid evidence (creates prod v2), then rollback (v3)
  - an FP8 change is blocked until a quality eval exists
- [x] Real Prometheus 3.15 scraping all targets with bearer auth; engine metrics and PromQL read through the MCP server.
- [x] Real TrueForge server: the connector registers and TrueForge sees all 19 tools with correct annotations; the agent is created with gates `["apply_production_config", "rollback_production", "@destructive"]`, sandbox, subagents and Generative UI on.
- [x] All Python compiles under 3.9 (AWS nodes run 3.10); all shell scripts pass `bash -n` under macOS bash 3.2.

### 4.3 Findings from testing (important on the day)
1. **TrueForge blocks MCP URLs on private or loopback IPs** ("Outbound URL blocked"). Fix: start it with `OUTBOUND_URL_ALLOWED_HOSTS='["<mcp-ip>"]'`; `setup_control.sh` does this.
2. **TrueForge rejects `reasoning_effort`** for models that don't support it, so it's opt-in via `AGENT_REASONING_EFFORT`.
3. **TrueForge local mode can bind to IPv6 `::1` only.** `HOST=127.0.0.1` pins it to IPv4 so the SSM tunnel works; set in `setup_control.sh`.
4. **Skills must be imported before an agent can reference them**, so the runbook is inlined by default.
5. **MCP Python SDK v2 renamed `FastMCP` → `MCPServer`**; tool annotations use snake_case fields but serialize to `readOnlyHint` / `destructiveHint` (verified on the wire).

---

## 5. What's left (in order)

### Phase A: AWS bring-up (as soon as organizer credentials arrive)
- [ ] Check the quota: *Running On-Demand G and VT instances* in ap-south-1 ≥ 8 vCPUs.
- [ ] `AWS_REGION=ap-south-1 ./infra/aws/launch.sh`
- [ ] `HF_TOKEN=… ./infra/aws/secrets.sh` (+ `PROMETHEUS_URL` if we use our existing Prometheus, and add the jobs from `infra/prometheus/prometheus.yml.tmpl` to it; it must be able to reach the nodes on :9000)
- [ ] `./infra/aws/deploy_code.sh`, then watch `/var/log/firefighter-setup.log` on each node until `SETUP COMPLETE`.
- [ ] If vLLM fails to start: check `docker logs vllm-prod`; pin `VLLM_IMAGE` to a known-good tag; lower `gpu_memory_utilization` only if startup OOMs.
- [ ] `./infra/aws/tunnel.sh` → TrueForge UI → configure **OpenAI** (Settings → Models) and **Daytona** (Settings → Sandbox providers).
- [ ] `ops.sh register-agent openai/<model-id>`, then `ops.sh smoke`.
- **Decision point:** if no GPU capacity within ~1 hour, escalate to the organizers. Never fall back to synthetic metrics.

### Phase B: Calibration on the real A10G
- [ ] ~10 min of `healthy` traffic: all SLOs pass with margin.
- [ ] `long_context_shift`: p95 TTFT ≥ 3–5× baseline, KV ≈ 1.0, preemptions rising within 2–3 minutes. Tune `rps` / `long_share` in the scenario file.
- [ ] Record the real KV capacity from `get_logs(grep="KV cache")`.
- [ ] By hand (tools only, no agent): prove the fix passes; confirm the tempting wrong fix (`max_num_batched_tokens` ↑) does **not**; try `kv_cache_dtype: fp8` on A10G and record the result.
- [ ] Write the final SLO thresholds into `mcp_server/policy.yaml`; pin `VLLM_IMAGE`; commit the calibration numbers to the README.

### Phase C: Agent behavior
- [ ] Verify the **Code Mode bridge**: the agent's sandbox script calls `call_tool("inference-ops", …)`. Fallback: direct tool calls, with large results auto-offloaded to sandbox files.
- [ ] Verify the **approval pause** appears for `apply_production_config`, and that **Deny with a reason** makes the agent adapt.
- [ ] Run the full incident ≥ 3 times (`chaos reset --clear-evidence …` between runs). Review each run in **Sessions**. Fix failures in the runbook, instructions or tool descriptions, never by hard-coding answers.
- [ ] Check classification on `burst` (expect "no change") and `surge` (expect "capacity, not config": no scale-out tool exists, so the agent should say so).

### Phase D: Polish (only after C works)
- [ ] Generative UI evidence card: experiments × SLOs table + p95 chart + diff.
- [ ] Optional GitHub audit commits (`GITHUB_TOKEN`/`GITHUB_REPO` in `secrets.sh`).
- [ ] Optional: import the skill from this public repo and re-register with `--use-skill`.
- [ ] Optional: counterfactual replay (capture a healthy-window workload and replay it on the prod config to show the traffic, not the config, is the cause). It's possible with the existing tools; add it to the runbook if time allows.

### Phase E: Demo and submission
- [ ] Record a backup video as soon as one clean run exists.
- [ ] Final video (≤ 3 min, sped-up parts labeled, **no keys on screen**).
- [ ] README: calibration numbers, AI-assistance disclosure (add any other assistants used).
- [ ] Teach-back: every member can explain the architecture and their part.
- [ ] After judging: `./infra/aws/teardown.sh --all`.

---

## 6. Demo script (~3 min)

| Time | Screen | Voice-over |
|---|---|---|
| 0:00–0:15 | Prometheus / SLO status: p95 up, goodput down | "Production inference on AWS just degraded. No errors, no deploys, config unchanged." |
| 0:15–0:45 | Agent triage + a Code Mode script in Daytona: TTFT by prompt-length bucket, KV pinned, preemptions, no config change | "It measures instead of guessing. Every number comes from code it wrote, running in a sandbox." |
| 0:45–1:05 | Reproduce on the shadow A10G with captured production traffic | "It replays real production traffic on a second GPU and reproduces the failure." |
| 1:05–1:45 | Hypothesis → experiment → **REJECTED** (the wrong fix makes it worse) → re-diagnose → winning config | "Its first idea made things worse, and it rejected it on the numbers. Then it found the real mechanism." |
| 1:45–2:15 | Evidence card, plan with blast radius, **approval card** (Deny with a reason → adapt → Allow) | "Only now does it ask to restart production, with the diff, the evidence and the blast radius." |
| 2:15–2:45 | Prod restarts, `get_slo_status` shows recovery, final report | "Approved, applied, verified. Real, measured, different numbers." |
| 2:45–3:00 | Title | "It doesn't tell you what's wrong. It shows you what works." |

## 7. Risks

| Risk | Mitigation |
|---|---|
| No g5 capacity/quota in ap-south-1 | Check the quota first; escalate by hour 1; try another AZ (`launch.sh` picks one that offers g5) |
| vLLM image/driver mismatch on DLAMI | Pin an older `VLLM_IMAGE` tag |
| Incident not dramatic enough | Tune the scenario rps/long_share; longer prompts |
| The wrong fix doesn't get worse on real hardware | Fine: the demo shows whichever hypothesis the numbers reject. Never stage it. |
| Code Mode bridge doesn't reach the MCP server | Direct tool calls + offloaded results analyzed in the sandbox |
| FP8 KV unsupported on A10G | The fix works without it; a failed start is shown as a rejected experiment |
| OpenAI rate limits | Cap parallel subagents; the tools return compact summaries |
| Secrets on screen | Keys only in TrueForge Settings / SSM; check the screen before recording |

## 8. Q&A prep
- **"Is the incident fake?"** Real vLLM on real A10Gs; the degradation is real KV-cache exhaustion and preemption, visible in vLLM's own Prometheus metrics and logs.
- **"Is the agent scripted?"** The tools return data, not answers. The traffic scenario name is never exposed. Every decision is in the TrueForge Sessions timeline, including rejected hypotheses.
- **"What stops it breaking prod?"**
  1. The TrueForge approval gate.
  2. Server-side evidence and quality checks, even after approval.
  3. Auto-rollback on a failed health check.
  4. Allowlisted knobs.
  5. No public ports and an IAM-role-only setup.
- **"Why the sandbox?"** The agent writes and runs analysis code in Daytona, which has no credentials and no route to the fleet.
- **"Where's the line?"** Anything reversible runs freely; anything that interrupts live traffic stops and asks.
