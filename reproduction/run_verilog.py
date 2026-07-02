#!/usr/bin/env python3
"""PairCoder on VerilogEval (spec-to-rtl) — hardware RTL generation (non-Python).
baseline = Driver's direct candidate[0]; paircoder = robust best-of-N over the same pool with
COMPILATION as the public grounding signal (keep candidate[0] if it compiles, else a compiling
sibling, else repair on the iverilog error). Grading runs the hidden testbench. Dumps all
candidates for pass@k."""
import os, sys, json, time, argparse, threading, subprocess, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from verilog_common import (load_verilog, build_prompt, extract_code, compiles,
                            has_topmodule, IVERILOG)

API_BASE = os.environ.get("PAIRCODER_API_BASE", "https://api.openai.com/v1")
API_KEY  = os.environ.get("PAIRCODER_API_KEY",  "")
_lock = threading.Lock(); USAGE = {"calls": 0}

DRIVER_SYS = (
    "You are the Driver in a pair-programming team and an expert hardware (Verilog/SystemVerilog) "
    "engineer. Implement the specified synthesizable module named TopModule. Return ONLY one "
    "```verilog``` block with the complete module. No prose, no testbench.")
ANGLES = [
    "Write the most direct correct implementation.",
    "Carefully match the exact port names, widths, and directions in the interface.",
    "Think about sequential vs combinational logic, reset behaviour, and edge cases.",
    "Re-read the specification and implement precisely the required behaviour.",
    "Use clean idiomatic synthesizable Verilog and ensure it elaborates without errors.",
]


def llm(model, system, user, retries=4):
    from paircoder.client import make_client, guarded_create
    client = make_client()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for a in range(retries):
        try:
            r = guarded_create(client, model=model, messages=msgs,
                                               extra_body={"reasoning_effort": os.environ.get("PAIRCODER_EFFORT","none")})
            if r.usage:
                with _lock: USAGE["calls"] += 1
            return r.choices[0].message.content or ""
        except Exception as e:
            if a == retries - 1:
                print(f"[api error] {e}", file=sys.stderr); return ""
            time.sleep(2 * (a + 1))


def _compile_err(code, timeout=20):
    if not has_topmodule(code):
        return "no module TopModule found"
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "dut.sv"); open(src, "w").write(code)
        try:
            p = subprocess.run([IVERILOG, "-g2012", "-o", os.path.join(d, "a.out"), src],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "compile timeout"
        return "" if p.returncode == 0 else (p.stderr or p.stdout)[:600]


AUTHOR_TB = ("Write a SELF-CHECKING Verilog testbench module named nav_tb for the TopModule "
             "described in the task (instantiate TopModule with its exact ports). Drive "
             "representative and edge-case stimuli, compare outputs against the spec, "
             "$display(\"NAV_FAIL ...\") on any mismatch, and $display(\"NAV_DONE\") then $finish "
             "at the end. Return ONLY one ```verilog``` block with the complete testbench.")


def _run_nav_tb(code_text, tb_text, timeout=40):
    code = extract_code(code_text); tb = extract_code(tb_text)
    if not has_topmodule(code):
        return False, "no module TopModule"
    if "module" not in (tb or ""):
        return False, "navigator test missing"
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "dut.sv"), "w").write(code)
        open(os.path.join(d, "tb.sv"), "w").write(tb)
        try:
            p = subprocess.run([IVERILOG, "-g2012", "-o", os.path.join(d, "a.out"),
                                os.path.join(d, "tb.sv"), os.path.join(d, "dut.sv")],
                               capture_output=True, text=True, timeout=timeout)
            if p.returncode != 0:
                return False, "nav-tb compile error: " + (p.stderr or p.stdout)[:300]
            r = subprocess.run(["vvp", os.path.join(d, "a.out")], capture_output=True, text=True, timeout=timeout)
            o = r.stdout + r.stderr
            return ("NAV_FAIL" not in o), o[-400:]
        except subprocess.TimeoutExpired:
            return False, "nav-tb timeout"


def solve_paired(prob, model, n_cand, max_iters):
    from paircoder import paper_solve, single_baseline
    q = build_prompt(prob)

    def chk(text):
        c = extract_code(text)
        if not has_topmodule(c):
            return False, "no module TopModule found"
        err = _compile_err(c)
        return (True, "") if not err else (False, err[:300])
    baseline = extract_code(single_baseline(q, model))
    pc_raw, _tel = paper_solve(q, model, max_iters=max_iters, check=chk,
                               author_test=AUTHOR_TB, run_authored_test=_run_nav_tb)
    pc = extract_code(pc_raw)
    return baseline, pc, [baseline]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="all")
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--outdir", default="results_verilog")
    args = ap.parse_args()

    probs = load_verilog()
    if args.n != "all":
        probs = probs[: int(args.n)]
    os.makedirs(args.outdir, exist_ok=True)
    base_out = os.path.join(args.outdir, f"samples_baseline_verilog_{args.model}.jsonl")
    pc_out = os.path.join(args.outdir, f"samples_paircoder_verilog_{args.model}.jsonl")
    cand_out = os.path.join(args.outdir, f"samples_candidates_verilog_{args.model}.jsonl")

    def existing(path):
        s = set()
        if os.path.exists(path):
            for l in open(path):
                try: s.add(json.loads(l)["task_id"])
                except Exception: pass
        return s
    have = existing(base_out) & existing(pc_out) & existing(cand_out)
    todo = [p for p in probs if p["task_id"] not in have]
    print(f"\n=== PAIRED | VerilogEval | {args.model} | {len(probs)} probs "
          f"({len(have)} cached, {len(todo)} to do) | cands={args.cands} ===", flush=True)
    t0 = time.time()
    bf, pf, cf = open(base_out, "a"), open(pc_out, "a"), open(cand_out, "a")
    lk = threading.Lock()

    def work(prob):
        b, p, cands = solve_paired(prob, args.model, args.cands, args.max_iters)
        return prob["task_id"], b, p, cands

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, p): p for p in todo}
        for f in as_completed(futs):
            try: tid, b, p, cands = f.result()
            except Exception: continue
            with lk:
                bf.write(json.dumps({"task_id": tid, "solution": b}) + "\n"); bf.flush()
                pf.write(json.dumps({"task_id": tid, "solution": p}) + "\n"); pf.flush()
                cf.write(json.dumps({"task_id": tid, "candidates": cands}) + "\n"); cf.flush()
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}]", flush=True)
    bf.close(); pf.close(); cf.close()
    print(f"  -> {base_out}\n  -> {pc_out}\n  -> {cand_out}\n  ({time.time()-t0:.0f}s, {USAGE['calls']} calls)", flush=True)


if __name__ == "__main__":
    main()
