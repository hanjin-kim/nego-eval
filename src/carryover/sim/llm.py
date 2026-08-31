"""One model call, and the reasons this file is not a client library.

The simulator asks a model for a decision and needs the decision back, not prose.
Two rules make that survivable:

**A refusal to parse is a datum, not a crash.** A buyer that answers with an
essay has told us something about whether it is doing economics. It is recorded
as `unparsed` and the round falls through to a default, rather than aborting an
episode that has already cost money.

**Temperature is fixed and low.** The experiment varies stakes, memory and
identity; it does not want sampling noise as a fourth uncontrolled treatment.
"""

from __future__ import annotations

import json
import os

import httpx

from carryover.config import load_env

load_env()
BASE = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
MODEL = os.environ.get("SIM_MODEL", os.environ.get("QWEN_JUDGE_MODEL", "qwen-plus"))

calls = {"n": 0, "unparsed": 0}
usage = {"in": 0, "out": 0}
#: Per-model totals. The aggregate was enough while every model was assumed to
#: behave the same way; it stopped being enough the moment one model spent ten
#: times the output tokens and was the only one to reach the ceiling.
per_model: dict = {}

#: A model id containing "/" is routed through OpenRouter; anything else goes to
#: DashScope. One switch rather than a provider abstraction — the experiment
#: needs several vendors reachable, not a framework.
OR_BASE = "https://openrouter.ai/api/v1"


def _route(model: str | None) -> tuple[str, str]:
    if model and "/" in model:
        key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPEN_ROUTER_API_KEY not set")
        return OR_BASE, key
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not set")
    return BASE, key


#: Optional per-call reasoning control, set by an experiment rather than by the
#: caller. The point of the manipulation is to hold the model fixed and move only
#: how much it thinks, so this is a global the runner sets around a block.
#: NOT thread-safe — concurrent runners overwrite each other's setting between
#: the assignment and the request, which silently scrambles the cells of any
#: threaded manipulation. Reasoning length is held constant and is not an
#: explanatory variable here; leave this None unless a runner is serial.
REASONING: dict | None = None


def ask(system: str, user: str, *, model: str | None = None,
        timeout: float = 60.0, retries: int = 7) -> str:
    """Retry on rate limits and transient server errors.

    A 429 is not a datum about the agent; losing a whole model to one is a hole
    in the comparison, not a finding. Backoff is exponential and bounded.
    """
    import time as _t
    last = None
    for attempt in range(retries):
        try:
            return _once(system, user, model=model, timeout=timeout)
        except httpx.HTTPStatusError as e:
            last = e
            if e.response.status_code not in (429, 500, 502, 503, 529):
                raise
            # A whole match is about a hundred calls, so a rate limit that
            # exhausts the retries throws away everything before it. Backoff runs
            # longer than it used to for exactly that reason.
            _t.sleep(min(2 ** attempt * 2, 90))
    raise last


def _once(system: str, user: str, *, model: str | None = None,
          timeout: float = 60.0) -> str:
    base, key = _route(model)
    calls["n"] += 1
    r = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {key}",
                 "HTTP-Referer": "https://github.com/local/bazaar-sim",
                 "X-Title": "agent-economic-trust-sim"},
        json={"model": model or MODEL, "temperature": 0.2,
              **({"reasoning": REASONING} if REASONING is not None else {}),
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]},
        timeout=timeout)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage") or {}
    usage["in"] += u.get("prompt_tokens", 0)
    usage["out"] += u.get("completion_tokens", 0)
    # `completion_tokens` alone undercounts every model that thinks in a separate
    # channel: an extended-thinking reply reports its reasoning under
    # `completion_tokens_details.reasoning_tokens`, and reading only the former
    # produced a "0 output tokens" row for a model that had plainly reasoned.
    # A zero there was impossible and should have been treated as a bug, not a
    # finding.
    det = u.get("completion_tokens_details") or {}
    slot = per_model.setdefault(model or MODEL,
                                {"n": 0, "in": 0, "out": 0, "reasoning": 0})
    slot["n"] += 1
    slot["in"] += u.get("prompt_tokens", 0)
    slot["out"] += u.get("completion_tokens", 0)
    slot["reasoning"] += det.get("reasoning_tokens", 0)
    msg = d["choices"][0]["message"]
    return msg.get("content") or ""


def ask_json(system: str, user: str, **kw) -> dict:
    """Return the first JSON object in the reply, or {} — and count the failure."""
    txt = ask(system, user, **kw)
    i, j = txt.find("{"), txt.rfind("}")
    if i < 0 or j < i:
        calls["unparsed"] += 1
        return {}
    try:
        return json.loads(txt[i:j + 1])
    except json.JSONDecodeError:
        calls["unparsed"] += 1
        return {}
