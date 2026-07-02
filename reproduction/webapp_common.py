#!/usr/bin/env python3
"""WebApp1K-Duo-React: TDD web (React) code generation benchmark.

Each task: a scenario with 2 features; 4 Jest tests (2 success + 2 failure) define the spec.
The model writes a single React `App` component (default export). We run the 4 tests with Jest
(jsdom env, no browser) and pass iff all 4 pass.

This is a TDD setting: the tests are the deliverable spec and are shown to the model. Baseline =
one-shot; PairCoder = best-of-N + repair using the Jest failure output (a strong executable
signal). We report base@1/3/5 (oracle over candidates) and PairCoder@1.
"""
import os, re, json, subprocess, tempfile, shutil
from huggingface_hub import hf_hub_download
import pandas as pd

WEBBENCH = os.path.join(os.path.dirname(__file__), "webbench")
_DF = None

TEST_HEADER = """import React from 'react';
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MemoryRouter } from 'react-router-dom';
import fetchMock from 'fetch-mock';
import App from './App';

afterEach(() => { fetchMock.reset(); fetchMock.restore(); });

"""


def load_webapp(n=None):
    global _DF
    if _DF is None:
        p = hf_hub_download("onekq-ai/WebApp1K-Duo-React", "dataset.parquet", repo_type="dataset")
        _DF = pd.read_parquet(p)
    rows = _DF if n is None else _DF.head(n)
    tasks = []
    for i, r in rows.iterrows():
        tests = [r["Success Case 1"], r["Failure Case 1"], r["Success Case 2"], r["Failure Case 2"]]
        tasks.append({"task_id": f"{r['Category']}/{r['Scenario']}",
                      "category": r["Category"], "scenario": r["Scenario"],
                      "tests": [t for t in tests if isinstance(t, str) and t.strip()]})
    return tasks


def build_prompt(task):
    tests = "\n\n".join(task["tests"])
    return (
        "Implement a single React component `App` (default export) so that ALL of the following "
        "Jest tests pass. The tests run under @testing-library/react with jsdom; routing uses "
        "react-router-dom's MemoryRouter; network calls use `fetch` and are mocked by fetch-mock "
        "(so call the exact endpoints the tests expect). Handle both success and failure (error) "
        "cases shown.\n\n"
        f"```javascript\n{tests}\n```\n\n"
        "Return ONLY one ```jsx``` code block with the complete App.js (include `import React"
        "` and any hooks; default-export `App`). No tests, no explanation.")


def extract_jsx(text):
    if not text:
        return ""
    f = re.findall(r"```(?:jsx|javascript|js|tsx)?\s*\n(.*?)```", text, re.DOTALL)
    code = max(f, key=len).strip() if f else text.strip()
    return code


def has_app(code):
    return bool(code) and "App" in code and ("export default" in code or "export {" in code or "module.exports" in code)


def run_tests(task, app_code, timeout=90):
    """Run the 4 Jest tests against the generated App. Returns (passed_all, fail_output)."""
    if not has_app(app_code):
        return False, "no App export"
    work = tempfile.mkdtemp(dir=WEBBENCH, prefix="t_")
    sub = os.path.basename(work)
    try:
        open(os.path.join(work, "App.js"), "w").write(app_code)
        open(os.path.join(work, "a.test.js"), "w").write(TEST_HEADER + "\n\n".join(task["tests"]))
        try:
            # run from WEBBENCH (rootDir) so babel.config.js + node_modules resolve; select this test
            p = subprocess.run(
                ["npx", "jest", f"{sub}/a.test.js", "--silent", "--ci"],
                cwd=WEBBENCH, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "jest timeout"
        out = p.stdout + p.stderr
        return (p.returncode == 0 and "Tests:" in out and "failed" not in
                out.split("Tests:")[-1].split("\n")[0]), out[-1500:]
    finally:
        shutil.rmtree(work, ignore_errors=True)
