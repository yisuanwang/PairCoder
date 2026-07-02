#!/usr/bin/env python3
"""RTLLM (design spec -> Verilog) loader + iverilog executor. 50 real-world designs, each with a
design_description.txt (prompt incl. module name) and testbench.v. Pass = generated module compiles
with the testbench and the run prints 'Passed'. Public signal for PairCoder = compiles (iverilog)."""
import os, re, glob, subprocess, tempfile, shutil
ROOT = os.path.join(os.path.dirname(__file__), "RTLLM")
IV = shutil.which("iverilog") or "iverilog"
VVP = shutil.which("vvp") or "vvp"


def load_rtllm():
    probs = []
    for desc in glob.glob(os.path.join(ROOT, "**", "design_description.txt"), recursive=True):
        d = os.path.dirname(desc)
        tb = os.path.join(d, "testbench.v")
        if not os.path.exists(tb):
            continue
        text = open(desc).read()
        m = re.search(r"[Mm]odule name:\s*\n?\s*([A-Za-z_]\w*)", text)
        name = m.group(1) if m else os.path.basename(d)
        probs.append({"task_id": os.path.basename(d), "desc": text,
                      "module": name, "testbench": open(tb).read()})
    return probs


def build_prompt(prob):
    return prob["desc"].strip() + "\n\nReturn ONLY one ```verilog``` code block with the complete module."


def extract_code(text):
    if not text:
        return ""
    f = re.findall(r"```(?:verilog|systemverilog|sv|v)?\s*\n(.*?)```", text, re.DOTALL)
    if f:
        return max(f, key=len).strip()
    m = re.search(r"\bmodule\b", text)
    return text[m.start():].strip() if m else text.strip()


def has_module(code, name):
    return bool(re.search(rf"\bmodule\s+{re.escape(name)}\b", code or ""))


def compiles(code, name, timeout=25):
    """iverilog elaboration of the generated module alone (syntax/elaboration check)."""
    if not has_module(code, name):
        return False
    d = tempfile.mkdtemp()
    try:
        src = os.path.join(d, "m.v"); open(src, "w").write(code)
        p = subprocess.run([IV, "-g2012", "-o", os.path.join(d, "a.out"), src],
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def compile_err(code, name, timeout=25):
    if not has_module(code, name):
        return f"no module {name}"
    d = tempfile.mkdtemp()
    try:
        src = os.path.join(d, "m.v"); open(src, "w").write(code)
        p = subprocess.run([IV, "-g2012", "-o", os.path.join(d, "a.out"), src],
                           capture_output=True, text=True, timeout=timeout)
        return "" if p.returncode == 0 else (p.stderr or p.stdout)[:500]
    except subprocess.TimeoutExpired:
        return "timeout"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_test(prob, code, timeout=40):
    """Compile module + testbench, run, pass iff output contains 'passed'."""
    if not has_module(code, prob["module"]):
        return False
    d = tempfile.mkdtemp()
    try:
        m = os.path.join(d, "m.v"); open(m, "w").write(code)
        tb = os.path.join(d, "tb.v"); open(tb, "w").write(prob["testbench"])
        out = os.path.join(d, "a.out")
        c = subprocess.run([IV, "-g2012", "-o", out, tb, m], capture_output=True, text=True, timeout=timeout)
        if c.returncode != 0:
            return False
        r = subprocess.run([VVP, out], capture_output=True, text=True, timeout=timeout)
        return "passed" in (r.stdout + r.stderr).lower()
    except subprocess.TimeoutExpired:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_nav_test(code, tb_text, timeout=40):
    """Run the module against a NAVIGATOR-authored self-checking testbench (TDD-style review).
    Returns (ok, evidence). ok = compiled & ran & no 'NAV_FAIL' in output."""
    tb = tb_text if "module" in (tb_text or "") else ""
    if not tb or not code:
        return False, "navigator test missing/invalid"
    d = tempfile.mkdtemp()
    try:
        m = os.path.join(d, "m.v"); open(m, "w").write(code)
        t = os.path.join(d, "t.v"); open(t, "w").write(tb)
        out = os.path.join(d, "a.out")
        c = subprocess.run([IV, "-g2012", "-o", out, t, m], capture_output=True, text=True, timeout=timeout)
        if c.returncode != 0:
            return False, "nav-test compile error: " + (c.stderr or c.stdout)[:300]
        r = subprocess.run([VVP, out], capture_output=True, text=True, timeout=timeout)
        o = (r.stdout + r.stderr)
        ok = ("NAV_FAIL" not in o) and ("error" not in o.lower() or "0 error" in o.lower())
        return ok, o[-400:]
    except subprocess.TimeoutExpired:
        return False, "nav-test timeout"
    finally:
        shutil.rmtree(d, ignore_errors=True)
