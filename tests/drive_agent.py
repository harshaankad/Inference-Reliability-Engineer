"""Drive the Firefighter agent through the TrueForge SDK: stream a turn, log every step, and stop at
approval pauses so a HUMAN decides. Useful for rehearsals and wiring tests; the demo uses the UI.

  TRUEFORGE_BASE_URL=http://localhost:8791 python -m tests.drive_agent --message "..."
  python -m tests.drive_agent --deny "reason"     # resolve the pending approval(s) of the last run
  python -m tests.drive_agent --allow
State (session id, pending approvals) is kept in $DRIVE_STATE (default /tmp/ff-drive.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from trueforge_sdk import TrueForge

STATE = os.environ.get("DRIVE_STATE", "/tmp/ff-drive.json")


def short(text: Any, n: int = 400) -> str:
    s = text if isinstance(text, str) else json.dumps(text, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + " …"


def content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return " ".join(getattr(c, "text", "") or "" for c in content)


def run_turn(client: TrueForge, session_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    messages: dict[str, Any] = {}
    pending: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    t0 = time.time()
    status = "?"
    for ev in client.sessions.create_turn_stream(session_id=session_id, input=items):
        et = getattr(ev, "type", "")
        tag = f"[{time.time() - t0:6.1f}s]"
        thread = getattr(ev, "thread_id", None)
        who = "" if thread in (None, "main") else f"(sub {thread[:6]}) "
        if et == "model.message":
            messages[ev.id] = ev
            text = content_text(ev.content)
            if text.strip():
                print(f"{tag} {who}AGENT: {short(text, 1200)}")
            for tc in ev.tool_calls or []:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", None) or getattr(getattr(tc, "tool_info", None), "name", "?")
                names[tc.id] = name
                print(f"{tag} {who}CALL {name}({short(getattr(fn, 'arguments', ''), 300)})")
        elif et == "tool.response":
            print(f"{tag} {who}  -> {names.get(ev.tool_call_id, ev.tool_call_id)}: {short(ev.content, 350)}")
        elif et == "thread.created":
            print(f"{tag} SUBAGENT started: {getattr(ev, 'title', '')}")
        elif et == "sandbox.created":
            print(f"{tag} SANDBOX created")
        elif et == "tool.approval_required":
            for ref in ev.tool_calls:
                msg = messages.get(ref.source_event_id)
                call = next((c for c in (msg.tool_calls or []) if c.id == ref.id), None) if msg else None
                fn = getattr(call, "function", None)
                pending.append({"thread_id": ev.thread_id, "tool_call_id": ref.id,
                                "name": getattr(fn, "name", "?"), "arguments": getattr(fn, "arguments", "")})
                print(f"{tag} *** APPROVAL REQUIRED: {getattr(fn, 'name', '?')}\n    args: {short(getattr(fn, 'arguments', ''), 2000)}")
        elif et == "turn.done":
            status = getattr(ev.state, "status", "?")
            print(f"{tag} TURN DONE: {status}")
    return {"session_id": session_id, "pending": pending, "status": status}


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--message")
    g.add_argument("--allow", action="store_true")
    g.add_argument("--deny", metavar="REASON")
    ap.add_argument("--agent", default="inference-firefighter")
    ap.add_argument("--new-session", action="store_true")
    args = ap.parse_args()
    client = TrueForge(base_url=os.environ.get("TRUEFORGE_BASE_URL", "http://localhost:8791"), timeout=1800)
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}

    if args.message:
        if args.new_session or not state.get("session_id"):
            state["session_id"] = client.sessions.create(agent={"name": args.agent}).data.id
            print(f"session {state['session_id']}")
        items = [{"type": "user.message", "content": args.message}]
    else:
        if not state.get("pending"):
            sys.exit("no pending approvals")
        decision = {"status": "allow"} if args.allow else {"status": "deny", "reason": args.deny}
        items = [{"type": "user.tool_approval", "thread_id": p["thread_id"], "tool_call_id": p["tool_call_id"],
                  "approval": decision} for p in state["pending"]]
        print(f"resolving {len(items)} approval(s): {decision}")
    state.update(run_turn(client, state["session_id"], items))
    json.dump(state, open(STATE, "w"))


if __name__ == "__main__":
    main()
