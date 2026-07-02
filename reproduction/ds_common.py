#!/usr/bin/env python3
"""DS-1000 (data-science code generation) loader + executor.

Each problem: a prompt asking for a snippet that assigns `result`; a `code_context` that defines
`test_execution(solution)` (inserts the snippet, runs it on a generated test input, asserts the
result equals the reference). We grade with the official `test_execution`.

PairCoder selection signal = EXECUTION CONSENSUS: run each candidate on the context's test input,
capture repr(result), majority-vote across candidates (self-consistency; the gold answer is never
used for selection). Runnable libraries only (Pandas/Numpy/Scipy/Sklearn) — torch/TF not installed.
"""
import os, re, sys, json, subprocess, tempfile
from huggingface_hub import hf_hub_download

RUNNABLE_LIBS = {"Pandas", "Numpy", "Scipy", "Sklearn"}
_PATH = None


def _file():
    global _PATH
    if _PATH is None:
        _PATH = hf_hub_download("xlangai/DS-1000", "test.jsonl", repo_type="dataset")
    return _PATH


def load_ds(libs=None):
    libs = libs or RUNNABLE_LIBS
    out = []
    for l in open(_file()):
        r = json.loads(l)
        if r["metadata"]["library"] in libs:
            out.append({"task_id": f"DS/{r['metadata']['problem_id']}",
                        "prompt": r["prompt"], "code_context": r["code_context"],
                        "library": r["metadata"]["library"]})
    return out


def build_prompt(prob):
    return (prob["prompt"].rstrip() +
            "\n\nReturn ONLY one ```python``` block with the solution code that assigns the "
            "required `result` (no surrounding function, no prints, no test harness).")


def extract_code(text):
    if not text:
        return ""
    f = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if f:
        return max(f, key=len).strip()
    return text.strip()


# ---- official grading ----
_GRADE = r"""
import sys
{CTX}
_sol = {SOL!r}
try:
    test_execution(_sol)
    print("PASS")
except Exception as e:
    print("FAIL:" + type(e).__name__)
"""


def run_test(prob, code, timeout=30):
    if not code or "result" not in code:
        return False
    src = _GRADE.replace("{CTX}", prob["code_context"]).replace("{SOL!r}", repr(code))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout, cwd=tempfile.gettempdir())
        return p.stdout.strip().endswith("PASS")
    except subprocess.TimeoutExpired:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass


# ---- consensus signature: run candidate on the context's test input, repr(result) ----
_SIG = r"""
import sys
{CTX}
_sol = {SOL!r}
try:
    test_input, _ = generate_test_case(1)
    env = {"test_input": test_input}
    code = exec_context.replace("[insert]", _sol)
    exec(code, env)
    print("SIG:" + repr(env.get("result"))[:300])
except Exception as e:
    print("EXC:" + type(e).__name__)
"""


def signature(prob, code, timeout=25):
    if not code:
        return None
    src = _SIG.replace("{CTX}", prob["code_context"]).replace("{SOL!r}", repr(code))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout, cwd=tempfile.gettempdir())
        for line in p.stdout.splitlines():
            if line.startswith("SIG:") or line.startswith("EXC:"):
                return line
        return None
    except subprocess.TimeoutExpired:
        return None
    finally:
        try: os.unlink(path)
        except Exception: pass
