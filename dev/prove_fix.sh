#!/usr/bin/env bash
# Calibration helper (runs on the CONTROL node): capture recent prod traffic and replay it on the
# SHADOW GPU under several candidate configs, printing one summary line per experiment.
#   bash dev/prove_fix.sh '<json changes>' ['<json changes>' ...]      e.g. '{}' '{"max_num_seqs":16}'
set -euo pipefail
cd /opt/firefighter
OPS=infra/node/ops.sh
WL=$($OPS tool capture_workload '{"start":"-4m"}' | jq -r .id)
echo "workload $WL"
for changes in "$@"; do
  dep=$($OPS tool deploy_shadow "$(jq -nc --argjson c "$changes" '{changes: $c, reason: "calibration"}')")
  st=$(echo "$dep" | jq -r .status)
  if [ "$st" != healthy ]; then echo "$changes -> deploy $st: $(echo "$dep" | jq -r '.failure_log_tail' | tail -3)"; continue; fi
  $OPS tool run_load_test "$(jq -nc --arg wl "$WL" --arg c "$changes" '{workload_id: $wl, hypothesis: ("calibration: " + $c), duration_s: 60}')" | jq -c --arg c "$changes" \
    '{changes: $c, pass: .slo_evaluation.all_pass, ttft95: .summary.latency.p95_ttft_ms, e2e95: .summary.latency.p95_e2e_ms,
      goodput: .summary.goodput_ratio, err: .summary.error_rate, n: .summary.requests, out_tok_s: .summary.output_tokens_per_s,
      kv_max: .engine.kv_cache_usage_max, wait_kv_mean: .engine.waiting_for_kv_capacity_mean, running: .engine.running_mean,
      prefix_hit: .engine.prefix_cache_hit_rate, run: .id}'
done
