---
name: incident-runbook
description: Procedure for diagnosing and remediating a degraded vLLM inference service (latency, throughput, KV-cache pressure, preemptions) with shadow experiments and a human-approved production change. Load at the start of any inference incident.
---

# Inference incident runbook

You are restoring latency/throughput SLOs on a live vLLM server. Inference incidents usually have
**no error message**: the config is legal and the server reports healthy while serving bad latency.
You find the cause by measuring, not by recalling defaults.

## 0. Ground rules
- Production is observe-only except `apply_production_config` / `rollback_production`, which restart
  the live server and wait for human approval. Never try another route.
- Every experiment runs on the **shadow** GPU via `deploy_shadow` + `run_load_test`.
- Compute every number in code in the sandbox (Code Mode: `from mcp_client import call_tool`).
  Never estimate percentiles or rates from raw JSON by eye.
- Before each experiment, write the hypothesis and the **predicted** effect. After it, state
  **ACCEPTED** or **REJECTED** with the measured numbers. Rejecting your own idea on evidence is normal
  and expected; hiding it is not.
- Budget: at most 6 shadow experiments. Each `deploy_shadow` costs 1-3 min, each load test ~1-2 min.

## 1. Triage (2 min)
1. `get_slo_status(window_minutes=10)` and `get_policy()`. Which SLOs are breached, by how much?
2. `get_change_history()`: did the config change recently? Note the times.

## 2. What changed? (Code Mode)
Write one script that calls `compare_windows` (a healthy baseline window vs the incident window),
`get_request_log` and `get_engine_metrics`, then prints:
- offered rps before/after (more users?)
- prompt-token distribution before/after, and the share of long prompts (different traffic?)
- p95 TTFT **bucketed by prompt length** (for example <1k, 1-2k, 2-4k, 4-8k, 8k+)
- KV-cache usage, waiting queue, preemptions/s, prefix-cache hit rate over time
- config deploys in between (config change?)

Classify, with evidence for and against:
| Pattern | Class |
|---|---|
| Regression starts at a deploy; load and mix unchanged | config regression |
| rps up, same mix, queue up, served throughput flat | load / capacity |
| Same rps, prompts much longer, TTFT up, KV cache ~100%, requests waiting for KV capacity (or preemptions) | **traffic-shape shift -> KV-cache pressure** |
| Short spike, queue drains by itself, no deploy | transient: recommend **no change** and stop |
| Several of the above | mixed |

## 3. Reproduce before fixing
1. `capture_workload(start=<incident start>)`: real production requests.
2. `deploy_shadow(changes={}, reason="reproduce prod")`, then `run_load_test(...)`.
3. It must breach the same SLOs with the same engine signature (KV cache pinned, waiting for KV capacity
   or preemptions). If it
   does not reproduce, your classification is wrong: go back to step 2.

## 4. Experiment (the loop)
Think physically about GPU memory. Weights (~15 GB for Qwen2.5-7B in bf16) plus the KV cache must fit
in `gpu_memory_utilization` x 24 GB. The KV cache is what is left: **only enough for tens of thousands
of tokens**. `get_logs(grep="KV cache|blocks|preempt")` shows the real capacity. Long prompts x many
concurrent sequences > capacity -> queueing -> p95 explodes. Older vLLM preempts and recomputes
sequences (`preemptions`); newer vLLM (V1 scheduler) mostly holds requests back instead, which shows up as
`waiting_for_kv_capacity` > 0 (requests waiting because the KV cache is full) with preemptions still 0.

Levers (each has a cost; measure, never assume):
- `max_num_seqs`: fewer concurrent sequences means less over-admission and fewer preemptions.
  Too low starves throughput.
- `enable_prefix_caching`: long requests share one long system prompt; reuse its KV blocks.
- `kv_cache_dtype=fp8`: about 2x KV capacity per GB; **must pass `run_quality_eval`** (supported on
  the L4; a failed start would still be a valid, informative result).
- `gpu_memory_utilization`: a little more room for KV cache (max 0.95).
- `max_num_batched_tokens`: per-step token budget. More is **not** automatically better under KV
  pressure. It can admit more prefill work into a cache that is already full.
- `max_model_len`: lowering it only helps by rejecting longer requests. Check the error-rate SLO.

Order: one variable at a time for the top 1-2 hypotheses, then combine the winners. Use
`list_experiments()` to keep the table.

## 5. Converge and prove
- The winner must pass **all** SLOs on the captured production workload at `rate_multiplier >= 1.0`
  for `duration_s >= 45`. Also test 1.3x if time allows (headroom).
- If `kv_cache_dtype` changes: `run_quality_eval` on the current prod config (baseline) **and** on
  the candidate.
- Show an evidence card with Generative UI: a table of experiments x (p95 TTFT, p95 e2e, goodput,
  error rate, preemptions, KV usage, verdict), a bar chart of p95 TTFT per experiment, and the exact
  config diff.

## 6. Ask, then act
1. `plan_production_change(changes, evidence_run_ids)`. It must say `ready: true`. Show the diff, the
   checks and the blast radius to the human.
2. Call `apply_production_config` with: `changes`, `evidence_run_ids`, a `justification` (root cause
   plus why this fixes it), the plan's `blast_radius`, and `expected_result` (before -> after numbers).
   The human approves or denies.
3. If **denied**: read the reason, adapt (new constraint -> new experiment), re-prove, ask again.

## 7. Verify and report
- Wait ~2-3 minutes of live traffic, then `get_slo_status(window_minutes=3)`. If SLOs have not
  recovered, propose `rollback_production`.
- Final report: timeline, classification, root cause (the physical mechanism), experiments including
  rejected ones, the change, before/after production numbers, follow-ups (e.g. alert on preemptions/s,
  capacity-test long-context traffic before product launches).
