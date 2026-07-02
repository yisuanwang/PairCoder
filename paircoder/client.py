"""OpenAI-compatible LLM client for PairCoder.

PairCoder talks to any OpenAI-compatible chat-completions endpoint (OpenAI,
vLLM, LM Studio, OpenRouter, Volcano ARK / doubao / deepseek, ...). All
credentials come from environment variables — nothing is hard-coded.

Required:
    PAIRCODER_API_BASE   e.g. https://api.openai.com/v1
    PAIRCODER_API_KEY    your API key

Optional (Volcano ARK, for doubao-* / deepseek-* models — routed automatically):
    ARK_API_BASE         default https://ark.cn-beijing.volces.com/api/v3
    ARK_API_KEY          your ARK key

Optional tuning:
    PAIRCODER_MAX_CONC     max concurrent API calls (default 8)
    PAIRCODER_HTTP_TIMEOUT per-request timeout seconds (default 75)
    PAIRCODER_TOKLOG       path to dump a JSON token tally at exit
"""
import os
import itertools
import threading

import httpx
from openai import OpenAI

# ---- endpoint configuration (env only — no secrets in source) --------------
API_BASE = os.environ.get("PAIRCODER_API_BASE", "https://api.openai.com/v1")
API_KEY = os.environ.get("PAIRCODER_API_KEY", "")

# Volcano ARK is reached by model-name prefix (doubao-* / deepseek-*), not by
# the blind endpoint rotation. Configure it only if you use those models.
ARK_BASE = os.environ.get("ARK_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
ARK_KEY = os.environ.get("ARK_API_KEY", "")

ENDPOINTS = [(API_BASE, API_KEY)]


def _mk(base, key):
    # trust_env=False bypasses any local proxy (relays are often not on the
    # proxy's allow-list); the curl UA dodges WAFs that block the SDK default UA.
    return OpenAI(
        base_url=base,
        api_key=key or "EMPTY",
        http_client=httpx.Client(
            trust_env=False,
            timeout=int(os.environ.get("PAIRCODER_HTTP_TIMEOUT", "75")),
        ),
        default_headers={"User-Agent": "curl/8.4.0"},
    )


_CLIENTS = [_mk(b, k) for b, k in ENDPOINTS]
_ARK_CLIENT = _mk(ARK_BASE, ARK_KEY) if ARK_KEY else None


def _is_ark_model(m):
    return isinstance(m, str) and (m.startswith("doubao") or m.startswith("deepseek"))


def make_client():
    """Return the primary client (PairCoder calls funnel through guarded_create)."""
    return _CLIENTS[0]


API_SEM = threading.Semaphore(int(os.environ.get("PAIRCODER_MAX_CONC", "8")))
_RR = itertools.count()
_RETRY = ("429", "rate", "blocked", "overload", "timeout", "502", "503", "500",
          "insufficient_balance", "unavailable", "bad gateway", "internal",
          "unknown", "not found", "does not exist", "invalid model", "404")

TOKENS = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}


def guarded_create(client, **kwargs):
    """Single entry point for every API call.

    - bounds global concurrency (PAIRCODER_MAX_CONC),
    - retries transient/balance errors with exponential backoff,
    - routes doubao-*/deepseek-* to ARK and translates reasoning_effort:none
      into ARK's thinking={"type": "disabled"} (so "thinking off" works there),
    - tallies token usage.
    """
    import time as _t
    r = None
    last = None
    n = max(len(_CLIENTS), 1)
    ark = _is_ark_model(kwargs.get("model"))
    if ark:
        eb = dict(kwargs.get("extra_body") or {})
        eff = eb.pop("reasoning_effort", "none")
        eb["thinking"] = {"type": "disabled" if eff in (None, "none") else "enabled"}
        kwargs = {**kwargs, "extra_body": eb}
    for att in range(6):
        if ark:
            if _ARK_CLIENT is None:
                raise RuntimeError("ARK model requested but ARK_API_KEY is not set.")
            cl = _ARK_CLIENT
        else:
            cl = _CLIENTS[next(_RR) % n]
        try:
            with API_SEM:
                r = cl.chat.completions.create(**kwargs)
            break
        except Exception as e:  # noqa: BLE001
            last = e
            m = str(e).lower()
            if any(x in m for x in _RETRY):
                _t.sleep(min(2 ** att, 20))
                continue
            raise
    if r is None:
        raise last
    try:
        u = r.usage
        if u:
            TOKENS["prompt"] += u.prompt_tokens or 0
            TOKENS["completion"] += u.completion_tokens or 0
            TOKENS["total"] += u.total_tokens or 0
            TOKENS["calls"] += 1
    except Exception:
        pass
    return r


import atexit as _atexit
import json as _json


@_atexit.register
def _dump_tokens():
    p = os.environ.get("PAIRCODER_TOKLOG")
    if p:
        try:
            with open(p, "w") as f:
                _json.dump(TOKENS, f)
        except Exception:
            pass
