"""Deterministic prompt dataset (no downloads, no real user data).

Three kinds of prompts:
  short  - chat questions, short system prompt, 96-256 output tokens
  long   - RAG-style: one SHARED long system prompt (so prefix caching matters) + a long
           generated document containing a "needle" fact + a question about it
  golden - long prompts with known answers, used for the quality gate (exact-match accuracy)

Every node builds the same file from the same seed, so prompt ids resolve identically on the
prod and shadow hosts:  python -m workload.dataset build --out data/dataset.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

WORDS_PER_TOKEN = 1 / 1.35  # rough for this generated English text with numbers

_DEPTS = ["billing", "logistics", "claims", "onboarding", "fraud review", "payments", "support", "procurement",
          "field service", "returns", "compliance", "vendor management"]
_CITIES = ["Pune", "Chennai", "Hyderabad", "Kolkata", "Jaipur", "Kochi", "Indore", "Nagpur", "Mysuru", "Surat",
           "Lucknow", "Bhopal", "Coimbatore", "Vadodara"]
_ITEMS = ["tickets", "invoices", "shipments", "claims", "applications", "refund requests", "work orders",
          "purchase orders", "chargebacks", "service calls"]
_ADJ = ["steady", "seasonal", "unexpected", "gradual", "sharp", "modest", "sustained", "temporary"]
_CAUSES = ["a vendor outage", "a festival sale", "a policy change", "a new product launch", "monsoon delays",
           "a backlog from the previous quarter", "a system migration", "staff reallocation"]
_TOPICS = ["time management", "learning Python", "writing clear emails", "preparing for interviews",
           "saving money", "improving sleep", "public speaking", "code review", "running a standup",
           "remote work", "negotiating a salary", "reading research papers", "debugging production issues"]
_SHORT_TEMPLATES = [
    "Give me {n} practical tips for {topic}.",
    "Explain {topic} to a new graduate in a short paragraph.",
    "What are the most common mistakes people make with {topic}? List {n}.",
    "Write a short checklist for {topic}.",
    "Summarize the key ideas of {topic} in {n} bullet points.",
]


def _sentence(rng: random.Random) -> str:
    kind = rng.randrange(4)
    if kind == 0:
        return (f"In Q{rng.randint(1, 4)} {rng.randint(2019, 2025)}, the {rng.choice(_DEPTS)} team in "
                f"{rng.choice(_CITIES)} processed {rng.randint(120, 98000):,} {rng.choice(_ITEMS)} with an average "
                f"turnaround of {rng.randint(1, 30)} days.")
    if kind == 1:
        return (f"The {rng.choice(_ADJ)} change in {rng.choice(_ITEMS)} volume was attributed to "
                f"{rng.choice(_CAUSES)}, which raised handling cost by {rng.randint(2, 40)} percent.")
    if kind == 2:
        return (f"Audit reference {rng.randint(1000, 9999)}-{rng.choice('ABCDEFGH')} notes that the "
                f"{rng.choice(_DEPTS)} desk escalated {rng.randint(3, 900)} cases to the regional office in "
                f"{rng.choice(_CITIES)}.")
    return (f"Management approved a {rng.choice(_ADJ)} staffing plan for {rng.choice(_DEPTS)} covering "
            f"{rng.randint(2, 60)} analysts and a budget of Rs {rng.randint(5, 900)} lakh.")


def _document(rng: random.Random, target_tokens: int) -> str:
    target_words = int(target_tokens * WORDS_PER_TOKEN)
    paras, words = [], 0
    while words < target_words:
        para = " ".join(_sentence(rng) for _ in range(rng.randint(5, 9)))
        paras.append(para)
        words += len(para.split())
    return "\n\n".join(paras)


def _system_prompt() -> str:
    rng = random.Random(1234)  # fixed: identical for every long request -> shareable prefix
    rules = [f"Rule {i + 1}: " + _sentence(rng) for i in range(60)]
    return ("You are the internal operations analyst assistant for Acme Services India. Answer strictly from "
            "the provided document. If the answer is not in the document, say you do not know. Keep answers "
            "short.\n\nBackground reference material (do not quote unless asked):\n" + "\n".join(rules))


def _needle(rng: random.Random) -> tuple[str, str, str]:
    case_id = f"{rng.choice('KLMNPQRSTV')}{rng.randint(100, 999)}"
    code = f"{rng.choice('ABCDEFGHJK')}{rng.randint(1, 9)}-{rng.randint(1000, 9999)}"
    return case_id, code, f"Reference note: the escalation code for case {case_id} is {code}."


def _long_prompt(rng: random.Random, system: str, target_tokens: int) -> dict[str, Any]:
    doc = _document(rng, target_tokens)
    case_id, code, needle = _needle(rng)
    paras = doc.split("\n\n")
    paras.insert(rng.randint(0, len(paras)), needle)
    question = f"According to the document, what is the escalation code for case {case_id}? Reply with the code only."
    return {
        "kind": "long",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": "Document:\n" + "\n\n".join(paras) + "\n\nQuestion: " + question}],
        "max_tokens": 48,
        "answer": code,
        "approx_prompt_tokens": target_tokens + 1600,
    }


def build(seed: int = 7, n_short: int = 400, n_long: int = 300, n_golden: int = 40,
          long_min_tokens: int = 2500, long_max_tokens: int = 8000) -> dict[str, Any]:
    rng = random.Random(seed)
    system = _system_prompt()
    prompts: dict[str, dict[str, Any]] = {}
    for i in range(n_short):
        text = rng.choice(_SHORT_TEMPLATES).format(n=rng.randint(3, 7), topic=rng.choice(_TOPICS))
        prompts[f"s-{i:04d}"] = {
            "kind": "short",
            "messages": [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": text}],
            "max_tokens": rng.choice([96, 128, 192, 256]),
            "approx_prompt_tokens": 40,
        }
    for i in range(n_long):
        prompts[f"l-{i:04d}"] = _long_prompt(rng, system, rng.randint(long_min_tokens, long_max_tokens))
    for i in range(n_golden):
        p = _long_prompt(rng, system, rng.randint(long_min_tokens, long_max_tokens))
        p["kind"] = "golden"
        prompts[f"g-{i:03d}"] = p
    return {"seed": seed, "prompts": prompts}


def fingerprint(dataset: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()[:12]


def load(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default="data/dataset.json")
    b.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    ds = build(seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(ds, f)
    kinds: dict[str, int] = {}
    for p in ds["prompts"].values():
        kinds[p["kind"]] = kinds.get(p["kind"], 0) + 1
    print(f"wrote {args.out}: {kinds} fingerprint={fingerprint(ds)}")


if __name__ == "__main__":
    main()
