"""PairCoder: a verification-grounded pair-programming framework for LLM code generation.

Two agents, one keyboard:
    Driver    writes / revises the code.
    Navigator reviews it against verification evidence (compile / execute /
              render results) and returns [NOERROR] or a concrete REVISE.
Roles switch on each error; the loop stops when the Navigator accepts or the
round budget is exhausted.

Quick use:
    from paircoder import paper_solve, single_baseline

    baseline = single_baseline(question, model="gpt-5.4-mini")           # 1 shot
    code, info = paper_solve(question, model="gpt-5.4-mini", check=my_check)

See README.md for a full walkthrough, including parametric-3D generation.
"""
from .loop import (paper_solve, single_baseline, Agent, TOK_SPLIT, USAGE,
                   DRIVER_PROMPT, NAVIGATOR_PROMPT)
from .client import make_client, guarded_create, TOKENS
from . import client

__all__ = ["paper_solve", "single_baseline", "Agent", "make_client",
           "guarded_create", "TOK_SPLIT", "USAGE", "TOKENS", "client",
           "DRIVER_PROMPT", "NAVIGATOR_PROMPT"]
__version__ = "1.0.0"
