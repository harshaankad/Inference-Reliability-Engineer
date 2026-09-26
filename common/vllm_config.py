"""vLLM serving config: the allowlisted knobs, validation, hashing, diffing and flag rendering.

This is the single definition of what the agent is allowed to change. The model itself is
never agent-controllable; it comes from the node's environment.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

KNOBS: dict[str, dict[str, Any]] = {
    "gpu_memory_utilization": {"type": float, "min": 0.50, "max": 0.95, "flag": "--gpu-memory-utilization",
                               "doc": "Fraction of GPU memory vLLM may use (weights + activations + KV cache)."},
    "max_model_len": {"type": int, "min": 2048, "max": 32768, "flag": "--max-model-len",
                      "doc": "Longest request (prompt + output tokens) the server accepts."},
    "max_num_seqs": {"type": int, "min": 1, "max": 512, "flag": "--max-num-seqs",
                     "doc": "Max sequences scheduled concurrently in one batch."},
    "max_num_batched_tokens": {"type": int, "min": 512, "max": 65536, "flag": "--max-num-batched-tokens",
                               "doc": "Token budget per scheduler step (prefill chunks + decode tokens)."},
    "enable_prefix_caching": {"type": bool, "flag": "--enable-prefix-caching", "neg_flag": "--no-enable-prefix-caching",
                              "doc": "Reuse KV blocks for shared prompt prefixes (e.g. a common system prompt)."},
    "kv_cache_dtype": {"type": str, "choices": ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"], "flag": "--kv-cache-dtype",
                       "doc": "KV cache precision. fp8 roughly halves KV memory per token; may cost quality."},
}

DEFAULT_CONFIG: dict[str, Any] = {
    "gpu_memory_utilization": 0.90,
    "max_model_len": 32768,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "enable_prefix_caching": True,
    "kv_cache_dtype": "auto",
}


class ConfigError(ValueError):
    pass


def _coerce(name: str, value: Any) -> Any:
    spec = KNOBS[name]
    t = spec["type"]
    if t is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise ConfigError(f"{name} must be a boolean")
    if t is str:
        value = str(value)
        if value not in spec["choices"]:
            raise ConfigError(f"{name} must be one of {spec['choices']}")
        return value
    try:
        value = t(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a {t.__name__}") from None
    if not (spec["min"] <= value <= spec["max"]):
        raise ConfigError(f"{name}={value} outside allowed range [{spec['min']}, {spec['max']}]")
    return value


def validate(config: dict[str, Any], *, partial: bool = False) -> dict[str, Any]:
    """Return a normalized copy. Unknown keys are rejected (the model is not a knob)."""
    unknown = set(config) - set(KNOBS)
    if unknown:
        raise ConfigError(f"unknown or non-changeable knobs: {sorted(unknown)}; allowed: {sorted(KNOBS)}")
    out = {k: _coerce(k, v) for k, v in config.items()}
    if not partial:
        missing = set(KNOBS) - set(out)
        if missing:
            raise ConfigError(f"missing knobs: {sorted(missing)}")
        if out["max_num_batched_tokens"] < out["max_num_seqs"]:
            raise ConfigError("max_num_batched_tokens must be >= max_num_seqs")
    return out


def apply_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(validate(patch, partial=True))
    return validate(merged)


def config_hash(config: dict[str, Any], model: str) -> str:
    canonical = json.dumps({"model": model, **validate(config)}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def diff(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"knob": k, "from": old.get(k), "to": new.get(k)} for k in KNOBS if old.get(k) != new.get(k)]


def render_flags(config: dict[str, Any]) -> list[str]:
    config = validate(config)
    args: list[str] = []
    for name, spec in KNOBS.items():
        value = config[name]
        if spec["type"] is bool:
            args.append(spec["flag"] if value else spec["neg_flag"])
        else:
            args += [spec["flag"], str(value)]
    return args


def knob_docs() -> dict[str, dict[str, Any]]:
    docs = {}
    for name, spec in KNOBS.items():
        d = {"doc": spec["doc"], "type": spec["type"].__name__}
        if "min" in spec:
            d["range"] = [spec["min"], spec["max"]]
        if "choices" in spec:
            d["choices"] = spec["choices"]
        docs[name] = d
    return docs
