from __future__ import annotations

import pytest

from common import promtext, stats
from common import vllm_config as vc
from workload import dataset, loadgen

BASE = dict(vc.DEFAULT_CONFIG)


def test_validate_rejects_model_and_unknown_knobs():
    with pytest.raises(vc.ConfigError, match="non-changeable"):
        vc.validate({**BASE, "model": "gpt2"})


def test_validate_ranges_and_types():
    with pytest.raises(vc.ConfigError, match="outside allowed range"):
        vc.apply_patch(BASE, {"gpu_memory_utilization": 0.99})
    with pytest.raises(vc.ConfigError):
        vc.apply_patch(BASE, {"kv_cache_dtype": "int4"})
    assert vc.apply_patch(BASE, {"max_num_seqs": 512, "max_num_batched_tokens": 512})["max_num_seqs"] == 512
    assert vc.apply_patch(BASE, {"enable_prefix_caching": "false"})["enable_prefix_caching"] is False


def test_patch_diff_hash_and_flags():
    new = vc.apply_patch(BASE, {"max_num_seqs": 32, "enable_prefix_caching": False})
    assert vc.diff(BASE, new) == [{"knob": "max_num_seqs", "from": 256, "to": 32},
                                  {"knob": "enable_prefix_caching", "from": True, "to": False}]
    assert vc.config_hash(new, "m") != vc.config_hash(BASE, "m")
    assert vc.config_hash(new, "m") == vc.config_hash(dict(new), "m")
    flags = vc.render_flags(new)
    assert "--no-enable-prefix-caching" in flags
    assert flags[flags.index("--max-num-seqs") + 1] == "32"


def test_promtext_aliases_and_label_sum():
    text = """# HELP x
vllm:gpu_cache_usage_perc{model_name="a"} 0.5
vllm:num_preemptions_total{model_name="a",engine="0"} 3
vllm:num_preemptions_total{model_name="a",engine="1"} 4
vllm:num_requests_waiting{model_name="a"} 7
"""
    m = promtext.extract(text)
    assert m["kv_cache_usage"] == 0.5 and m["preemptions_total"] == 7 and m["waiting"] == 7
    assert m["prefix_hits_total"] is None


def _rec(t, ttft, e2e, pt, status="ok"):
    return {"t_start": t, "ttft_ms": ttft, "e2e_ms": e2e, "prompt_tokens": pt, "output_tokens": 10, "status": status}


def test_summarize_and_slo_eval():
    slo = {"p95_ttft_ms": {"max": 1000}, "p95_e2e_ms": {"max": 3000}, "error_rate": {"max": 0.1},
           "goodput_ratio": {"min": 0.9}}
    recs = [_rec(i, 100, 500, 50) for i in range(18)] + [_rec(18, 4000, 6000, 8000), _rec(19, None, None, None, "timeout")]
    s = stats.summarize(recs, duration_s=20, slo=slo, long_prompt_threshold=2000)
    assert s["requests"] == 20 and s["error_rate"] == 0.05 and s["goodput_ratio"] == 0.9
    assert s["prompt_tokens"]["long_share"] == 0.05
    assert s["by_prompt_length"]["long(>=2000 tok)"]["p95_ttft_ms"] == 4000
    ev = stats.evaluate_slos(s, slo)
    assert ev["slos"]["p95_ttft_ms"]["pass"] and ev["slos"]["goodput_ratio"]["pass"]
    ev2 = stats.evaluate_slos(s, {"goodput_ratio": {"min": 0.95}})
    assert not ev2["all_pass"]


def test_dataset_is_deterministic_and_has_needles():
    a = dataset.build(seed=3, n_short=5, n_long=3, n_golden=2)
    b = dataset.build(seed=3, n_short=5, n_long=3, n_golden=2)
    assert dataset.fingerprint(a) == dataset.fingerprint(b)
    g = a["prompts"]["g-000"]
    assert g["answer"] in g["messages"][1]["content"]
    systems = {p["messages"][0]["content"] for p in a["prompts"].values() if p["kind"] != "short"}
    assert len(systems) == 1  # shared prefix -> prefix caching is a real lever


def test_loadgen_burst_then_base():
    sc = {"rps": 2, "long_share": 0.1, "burst": {"rps": 9, "duration_s": 30}, "started_at": 1000.0}
    assert loadgen.current_rate(sc, 1010) == (9, 0.1)
    assert loadgen.current_rate(sc, 1031) == (2, 0.1)
