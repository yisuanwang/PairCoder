#!/usr/bin/env python3
"""PairCoder in 30 lines — the smallest useful example.

It generates a Python function two ways and compares them:
    1. single-model baseline (one shot)
    2. PairCoder (Driver/Navigator loop, grounded on "does it run + pass a check")

Run:
    export PAIRCODER_API_BASE="https://api.openai.com/v1"
    export PAIRCODER_API_KEY="sk-..."
    python examples/quickstart.py
"""
import re
import textwrap

from paircoder import paper_solve, single_baseline

MODEL = "gpt-5.4-mini"  # any OpenAI-compatible chat model

QUESTION = textwrap.dedent("""\
    Write a Python function `is_palindrome(s: str) -> bool` that returns True iff
    `s` reads the same forwards and backwards, ignoring case, spaces and
    punctuation. Return ONLY one ```python``` code block.
""")


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text or "", re.DOTALL)
    return (m[0] if m else (text or "")).strip()


def check(code: str):
    """The verification signal the Navigator sees: does the code run + pass a spot test?"""
    ns = {}
    try:
        exec(code, ns)
        f = ns["is_palindrome"]
        ok = f("A man, a plan, a canal: Panama") is True and f("hello") is False
        return (ok, "spot tests passed" if ok else "spot tests failed", 1.0 if ok else 0.0)
    except Exception as e:  # noqa: BLE001
        return (False, f"{type(e).__name__}: {e}", 0.0)


if __name__ == "__main__":
    base = extract_code(single_baseline(QUESTION, MODEL))
    pc, info = paper_solve(QUESTION, MODEL, max_iters=4, check=check, extract=extract_code)
    pc = extract_code(pc)

    print("=== single-model baseline ===\n", base)
    print("\n=== PairCoder (accepted=%s, iters=%d) ===\n" % (info["accepted"], info["iters"]), pc)
    print("\nbaseline passes check:", check(base)[0])
    print("PairCoder passes check:", check(pc)[0])
