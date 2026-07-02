#!/usr/bin/env python3
"""
BigCodeBench (software-engineering-flavored, library-heavy) loader + executor.

Each problem: complete_prompt (imports+signature+docstring), entry_point=task_func,
and a hidden unittest `TestCases` class. Grading runs the full unittest suite.

NOTE: official BigCodeBench targets Python 3.10 with pinned libs; here we run on the
available interpreter with latest wheels. Absolute numbers may differ from the leaderboard,
but baseline and PairCoder face IDENTICAL conditions, so the comparison is valid. Problems
whose required libraries are not importable are skipped (reported as `skipped`).
"""
import os, re, sys, json, ast, doctest, subprocess, tempfile, importlib
from huggingface_hub import hf_hub_download
import pandas as pd

_LIB_CACHE = {}
# map dataset lib name -> importable module name
_IMPORT_NAME = {
    "sklearn": "sklearn", "bs4": "bs4", "PIL": "PIL", "cv2": "cv2", "yaml": "yaml",
    "Levenshtein": "Levenshtein", "dateutil": "dateutil", "Crypto": "Crypto",
}


def load_bcb(version="v0.1.4"):
    path = hf_hub_download("bigcode/bigcodebench",
                           f"data/{version}-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    probs = []
    for _, r in df.iterrows():
        raw = r["libs"]
        if isinstance(raw, str):
            try:
                libs = list(ast.literal_eval(raw))
            except Exception:
                libs = []
        elif raw is not None:
            libs = list(raw)
        else:
            libs = []
        probs.append({
            "task_id": r["task_id"],
            "entry_point": r["entry_point"],
            "prompt": r["complete_prompt"],
            "code_prompt": r["code_prompt"],
            "test": r["test"],
            "libs": libs,
        })
    return probs


def libs_available(libs):
    for lib in libs:
        mod = _IMPORT_NAME.get(lib, lib)
        if mod in _LIB_CACHE:
            ok = _LIB_CACHE[mod]
        else:
            try:
                importlib.import_module(mod); ok = True
            except Exception:
                ok = False
            _LIB_CACHE[mod] = ok
        if not ok:
            return False
    return True


def build_prompt(prob):
    return (f"Complete the following Python function. Return ONLY one ```python``` block "
            f"with the FULL self-contained solution (keep all imports and the exact "
            f"`{prob['entry_point']}` signature; implement the body).\n\n"
            f"```python\n{prob['prompt']}\n```")


def extract_code(text):
    if not text:
        return ""
    f = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if f:
        return max(f, key=len).strip()
    m = re.search(r"(^|\n)(from |import |def )", text)
    return text[m.start():].strip() if m else text.strip()


def has_entry(code, entry):
    return bool(re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", code or ""))


# ---- weak public grounding: docstring doctests (often type/format checks) ----
def doctest_examples(prompt):
    try:
        return doctest.DocTestParser().get_examples(prompt)
    except Exception:
        return []


_DOCTEST_RUNNER = r"""
import doctest, sys
{CODE}
_p = {PROMPT!r}
import io
ex = doctest.DocTestParser().get_examples(_p)
if not ex:
    print("NONE"); sys.exit(0)
g = dict(globals())
t = doctest.DocTest(ex, g, "v", None, None, None)
r = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE)
buf = io.StringIO(); r.run(t, out=buf.write)
print("PASS" if r.failures == 0 else "FAIL")
"""


def public_ok(prob, code, timeout=15):
    """True if code passes the docstring doctests (weak signal). NONE counts as pass."""
    if not has_entry(code, prob["entry_point"]):
        return False
    if not doctest_examples(prob["prompt"]):
        return True  # no examples -> nothing to fail on
    src = _DOCTEST_RUNNER.replace("{CODE}", code).replace("{PROMPT!r}", repr(prob["prompt"]))
    try:
        p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return p.stdout.strip().endswith("PASS") or p.stdout.strip().endswith("NONE")


# ---- full grading: run the hidden unittest suite ----
_UNITTEST_RUNNER = r"""
import unittest, sys
{SOLUTION}

{TEST}

if __name__ == "__main__":
    loader = unittest.TestLoader()
    try:
        suite = loader.loadTestsFromTestCase(TestCases)
    except Exception as e:
        print("LOAD_ERROR", e); sys.exit(2)
    res = unittest.TextTestRunner(verbosity=0).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)
"""


def run_unittest(prob, code, timeout=40):
    """True iff the full hidden unittest suite passes."""
    if not has_entry(code, prob["entry_point"]):
        return False
    src = _UNITTEST_RUNNER.replace("{SOLUTION}", code).replace("{TEST}", prob["test"])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout, cwd=tempfile.gettempdir())
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        try: os.unlink(path)
        except Exception: pass


def run_nav_test(code, test_code, timeout=30):
    """Execute solution + NAVIGATOR-authored test script in one process (TDD-style review).
    Returns (ok, evidence). ok = ran and no 'NAV_FAIL' / traceback."""
    if not (code or "").strip() or not (test_code or "").strip():
        return False, "navigator test missing"
    src = code + "\n\n# ---- navigator test ----\n" + test_code + "\n"
    try:
        p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "nav-test timeout"
    o = (p.stdout + p.stderr)
    ok = (p.returncode == 0) and ("NAV_FAIL" not in o) and ("Traceback" not in o)
    return ok, o[-400:]
