#!/usr/bin/env python3
"""HumanEval-X multilingual loader + executor (C++, Java, JavaScript).

Per problem: declaration (imports + signature), canonical_solution, test (a runnable harness
with asserts), example_test (a few public asserts for grounding). The model returns a COMPLETE
solution in the target language; we compile/run solution + test.

Pass: C++/Java -> process exits 0 (assert failure aborts / throws AssertionError);
      JS -> no "Assertion failed" in stderr and exit 0.
Public grounding signal (regression-safe): the solution compiles (and passes example_test if present).
"""
import os, re, json, subprocess, tempfile, shutil
from huggingface_hub import hf_hub_download

LANGS = ["cpp", "java", "js"]
GPP = shutil.which("g++") or "g++"
JAVAC = shutil.which("javac") or "javac"
JAVA = shutil.which("java") or "java"
NODE = shutil.which("node") or "node"


def load_hex(lang):
    path = hf_hub_download("THUDM/humaneval-x", f"data/{lang}/data/humaneval.jsonl", repo_type="dataset")
    probs = []
    for l in open(path):
        r = json.loads(l)
        probs.append({
            "task_id": r["task_id"],
            "declaration": r.get("declaration", ""),
            "prompt": r["prompt"],
            "test": r["test"],
            "example_test": r.get("example_test", "") or "",
            "lang": lang,
        })
    return probs


_FENCE = {"cpp": r"(?:cpp|c\+\+|c)", "java": r"java", "js": r"(?:javascript|js)"}


def extract_code(text, lang):
    if not text:
        return ""
    f = re.findall(rf"```(?:{_FENCE[lang]})?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if f:
        return max(f, key=len).strip()
    return text.strip()


def build_prompt(prob, lang):
    names = {"cpp": "C++", "java": "Java", "js": "JavaScript"}
    return (f"Complete the following {names[lang]} solution. Return ONLY one ```{lang}``` block "
            f"with the FULL self-contained implementation (include all necessary imports/headers "
            f"and the required function/class, but DO NOT write a main function or tests).\n\n"
            f"```{lang}\n{prob['declaration'] or prob['prompt']}\n```")


# ---- execution ----
def _run_cpp(code, test, timeout):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.cpp"); open(src, "w").write(code + "\n\n" + test)
        exe = os.path.join(d, "a.out")
        try:
            c = subprocess.run([GPP, "-std=c++17", "-O0", "-w", "-o", exe, src],
                               capture_output=True, text=True, timeout=timeout)
            if c.returncode != 0:
                return False
            r = subprocess.run([exe], capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def _run_java(code, test, timeout):
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "Main.java"), "w").write(code + "\n\n" + test)
        try:
            c = subprocess.run([JAVAC, "Main.java"], cwd=d, capture_output=True, text=True, timeout=timeout)
            if c.returncode != 0:
                return False
            r = subprocess.run([JAVA, "Main"], cwd=d, capture_output=True, text=True, timeout=timeout)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def _run_js(code, test, timeout):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "p.js"); open(src, "w").write(code + "\n\n" + test)
        try:
            r = subprocess.run([NODE, src], capture_output=True, text=True, timeout=timeout)
            if r.returncode != 0:
                return False
            return "Assertion failed" not in (r.stderr + r.stdout)
        except subprocess.TimeoutExpired:
            return False


_RUN = {"cpp": _run_cpp, "java": _run_java, "js": _run_js}


def has_code(code, lang):
    if not code or not code.strip():
        return False
    if lang == "java":
        return "class" in code
    return True


def run_full(prob, code, timeout=30):
    if not has_code(code, prob["lang"]):
        return False
    return _RUN[prob["lang"]](code, prob["test"], timeout)


def public_ok(prob, code, timeout=30):
    """Regression-safe public signal: passes the example_test if present, else just compiles/runs."""
    if not has_code(code, prob["lang"]):
        return False
    if prob["example_test"].strip():
        return _RUN[prob["lang"]](code, prob["example_test"], timeout)
    # no example test -> compile-only check by running an empty harness
    empty = {"cpp": "int main(){return 0;}", "java": "public class Main{public static void main(String[] a){}}",
             "js": ""}[prob["lang"]]
    return _RUN[prob["lang"]](code, empty, timeout)
