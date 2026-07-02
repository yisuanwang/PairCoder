#!/usr/bin/env python3
"""
LiveCodeBench (code_generation_lite) loader + executor.

Two problem types:
  - functional: starter_code defines `class Solution: def method(self, ...)`.
                each test `input` is newline-separated JSON args; call method(*args),
                compare to json-parsed `output`.
  - stdin:      run the program as a script, feed `input` to stdin, compare stdout
                (line-by-line, trailing whitespace stripped) to `output`.

Grading: a solution PASSES a problem iff it passes ALL provided test cases.
"""
import os, json, base64, zlib, pickle, re, subprocess, tempfile, sys
import multiprocessing as mp
from huggingface_hub import hf_hub_download

LCB_FILE = {"v1": "test.jsonl", "v2": "test2.jsonl", "v3": "test3.jsonl",
            "v4": "test4.jsonl", "v5": "test5.jsonl", "v6": "test6.jsonl"}


def _decode_tests(s):
    """Decode public_test_cases / private_test_cases into a list of dicts."""
    if not s:
        return []
    # try plain (possibly double-encoded) JSON first
    try:
        v = json.loads(s)
        if isinstance(v, str):
            v = json.loads(v)
        return v
    except Exception:
        pass
    raw = zlib.decompress(base64.b64decode(s))
    try:
        v = json.loads(raw)
    except Exception:
        v = pickle.loads(raw)
    if isinstance(v, str):
        v = json.loads(v)
    return v


def load_lcb(version="v1"):
    path = hf_hub_download("livecodebench/code_generation_lite",
                           LCB_FILE[version], repo_type="dataset")
    probs = []
    for line in open(path):
        r = json.loads(line)
        pub = _decode_tests(r["public_test_cases"])
        priv = _decode_tests(r["private_test_cases"])
        functional = bool(r["starter_code"].strip())
        probs.append({
            "task_id": r["question_id"],
            "title": r["question_title"],
            "platform": r["platform"],
            "difficulty": r["difficulty"],
            "question": r["question_content"],
            "starter_code": r["starter_code"],
            "functional": functional,
            "public_tests": pub,
            "all_tests": pub + priv,
        })
    return probs


# ---------------- prompt ----------------
def build_prompt(prob):
    q = prob["question"]
    if prob["functional"]:
        instr = (
            "Write a Python solution. Complete the provided class/method exactly "
            "(keep the signature). Return ONLY one ```python``` block with the full, "
            "self-contained solution (include any imports such as `from typing import *`).\n\n"
            f"```python\n{prob['starter_code']}\n```")
    else:
        instr = (
            "Write a complete Python program that reads from standard input and writes "
            "the answer to standard output, matching the exact output format. Return ONLY "
            "one ```python``` block with the full runnable script.")
    return f"{q}\n\n{instr}"


def extract_code(text):
    if not text:
        return ""
    f = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if f:
        return max(f, key=len).strip()
    m = re.search(r"(^|\n)(from |import |class |def )", text)
    return text[m.start():].strip() if m else text.strip()


# ---------------- execution ----------------
_FUNC_HARNESS = r"""
import sys, json
from typing import *
import collections, math, heapq, bisect, itertools, functools, re, string
{CODE}

def _run():
    data = json.loads(sys.stdin.read())
    args_lines = data["input"].split("\n")
    # drop a single trailing empty line if present
    if args_lines and args_lines[-1] == "":
        args_lines = args_lines[:-1]
    args = []
    for ln in args_lines:
        try:
            args.append(json.loads(ln))
        except Exception:
            args.append(ln)
    sol = Solution()
    # find the (single) public method
    meths = [m for m in dir(sol) if not m.startswith("_")]
    res = getattr(sol, meths[0])(*args)
    print(json.dumps(res, default=str))

_run()
"""


def _norm_out(s):
    return "\n".join(line.rstrip() for line in s.strip("\n").splitlines()).strip()


def _check_one(prob_functional, code, test, timeout):
    """Return True/False for a single test case. Runs code in a subprocess."""
    if prob_functional:
        src = _FUNC_HARNESS.replace("{CODE}", code)
        try:
            p = subprocess.run([sys.executable, "-c", src],
                               input=json.dumps(test), capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        if p.returncode != 0:
            return False
        try:
            got = json.loads(p.stdout.strip())
        except Exception:
            return _norm_out(p.stdout) == _norm_out(str(test["output"]))
        try:
            exp = json.loads(test["output"])
        except Exception:
            exp = test["output"]
        return got == exp
    else:
        try:
            p = subprocess.run([sys.executable, "-c", code],
                               input=test["input"], capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        if p.returncode != 0:
            return False
        return _norm_out(p.stdout) == _norm_out(test["output"])


def check_solution(prob, code, tests, timeout=8, max_tests=None):
    """True iff `code` passes ALL `tests`. Short-circuits on first failure."""
    if not code or not code.strip():
        return False
    if prob["functional"] and not re.search(r"\b(def|class)\b", code):
        return False
    use = tests if max_tests is None else tests[:max_tests]
    for t in use:
        if not _check_one(prob["functional"], code, t, timeout):
            return False
    return True
