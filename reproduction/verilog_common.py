#!/usr/bin/env python3
"""VerilogEval (spec-to-rtl) loader + iverilog executor.

Each problem: a natural-language spec (prompt.txt) describing a module `TopModule`, a golden
`RefModule` (ref.sv), and a testbench (test.sv) that instantiates both and prints
"Mismatches: N in M samples". Pass iff N==0 (and it compiles & finishes).

Public grounding signal for PairCoder = COMPILATION (the generated module elaborates under
iverilog -g2012). This is regression-safe (compiling is necessary for correctness) and lets
the loop repair the common LLM failure mode of syntactically/elaboration-broken Verilog.
"""
import os, re, glob, subprocess, tempfile, shutil

DATA = os.path.join(os.path.dirname(__file__), "verilog-eval", "dataset_spec-to-rtl")
IVERILOG = shutil.which("iverilog") or "iverilog"
VVP = shutil.which("vvp") or "vvp"


def load_verilog():
    probs = []
    for pf in sorted(glob.glob(os.path.join(DATA, "*_prompt.txt"))):
        base = pf[:-len("_prompt.txt")]
        ref, test = base + "_ref.sv", base + "_test.sv"
        if not (os.path.exists(ref) and os.path.exists(test)):
            continue
        probs.append({
            "task_id": os.path.basename(base),
            "prompt": open(pf).read(),
            "ref": open(ref).read(),
            "test": open(test).read(),
        })
    return probs


def build_prompt(prob):
    return (f"{prob['prompt']}\n\n"
            "Write the complete Verilog/SystemVerilog module named `TopModule` implementing the "
            "above specification. Return ONLY one ```verilog``` code block with the full module "
            "(module ... endmodule). Do not include a testbench.")


def extract_code(text):
    if not text:
        return ""
    f = re.findall(r"```(?:verilog|systemverilog|sv)?\s*\n(.*?)```", text, re.DOTALL)
    if f:
        # prefer a block containing 'module TopModule'
        for b in f:
            if "module TopModule" in b or "module  TopModule" in b:
                return b.strip()
        return max(f, key=len).strip()
    m = re.search(r"\bmodule\b", text)
    return text[m.start():].strip() if m else text.strip()


def has_topmodule(code):
    return bool(re.search(r"\bmodule\s+TopModule\b", code or ""))


def compiles(code, timeout=20):
    """True if the generated module elaborates standalone under iverilog -g2012."""
    if not has_topmodule(code):
        return False
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "dut.sv")
        open(src, "w").write(code)
        try:
            p = subprocess.run([IVERILOG, "-g2012", "-o", os.path.join(d, "a.out"), src],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return p.returncode == 0


def run_test(prob, code, timeout=40):
    """True iff generated TopModule passes the hidden testbench (Mismatches: 0)."""
    if not has_topmodule(code):
        return False
    with tempfile.TemporaryDirectory() as d:
        dut = os.path.join(d, "dut.sv"); ref = os.path.join(d, "ref.sv"); test = os.path.join(d, "test.sv")
        open(dut, "w").write(code)
        open(ref, "w").write(prob["ref"])
        open(test, "w").write(prob["test"])
        out = os.path.join(d, "sim.out")
        try:
            c = subprocess.run([IVERILOG, "-g2012", "-o", out, ref, dut, test],
                               capture_output=True, text=True, timeout=timeout)
            if c.returncode != 0:
                return False
            r = subprocess.run([VVP, out], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        txt = r.stdout + r.stderr
        if "TIMEOUT" in txt:
            return False
        m = re.search(r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples", txt)
        if m:
            return int(m.group(1)) == 0 and int(m.group(2)) > 0
        return False
