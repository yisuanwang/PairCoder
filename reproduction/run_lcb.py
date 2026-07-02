#!/usr/bin/env python3
"""
PairCoder on LiveCodeBench (competitive programming; real headroom for strong models).

baseline : a single direct generation (the Driver's first attempt).
paircoder: robust best-of-N + public-test grounding (pair-programming loop):
   1. Driver writes the direct candidate (== baseline distribution) + N-1 diverse candidates.
   2. Each candidate is checked against the PUBLIC sample tests (given in the problem).
   3. Selection (monotone — never worse than the direct candidate on the public signal):
        - if the direct candidate passes all public tests -> keep it;
        - elif any sibling passes all public tests       -> take the first such sibling;
        - else Navigator feeds the concrete failing public test back to the Driver and it
          repairs, up to max_iters; keep the first repair that passes public, else best effort.

Grading (grade_lcb.py) runs the chosen solution against ALL tests (public + private):
a problem counts as solved iff every test passes. This is identical for both modes.
"""
import os, re, sys, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from lcb_common import load_lcb, build_prompt, extract_code, check_solution

API_BASE = os.environ.get("PAIRCODER_API_BASE", "https://api.openai.com/v1")
API_KEY  = os.environ.get("PAIRCODER_API_KEY",  "")

_lock = threading.Lock()
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

DRIVER_SYS = (
    "You are the Driver in a pair-programming team and an expert competitive programmer. "
    "Read the problem carefully, handle all constraints and edge cases, and return ONLY one "
    "```python``` block with a complete, efficient, self-contained solution. No prose.")
ANGLES = [
    "Write the most direct correct solution.",
    "Pay close attention to constraints and time complexity; choose an efficient algorithm.",
    "Carefully handle edge cases: empty input, minimum/maximum sizes, ties, and boundaries.",
    "Re-read the statement and the sample I/O, derive the exact required behaviour, then implement it.",
    "Consider a cleaner or different algorithmic approach and make the I/O format exact.",
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
                with _lock:
                    USAGE["calls"] += 1
                    USAGE["prompt_tokens"] += r.usage.prompt_tokens or 0
                    USAGE["completion_tokens"] += r.usage.completion_tokens or 0
            return r.choices[0].message.content or ""
        except Exception as e:
            if a == retries - 1:
                print(f"[api error] {e}", file=sys.stderr); return ""
            time.sleep(2 * (a + 1))


def _pub_ok(prob, code):
    return bool(code) and check_solution(prob, code, prob["public_tests"], timeout=6)


def solve_paired(prob, model, n_cand, max_iters):
    """Paper-faithful PairCoder: baseline = single direct generation; PairCoder = the
    Driver/Navigator loop (pc_paper) with the PUBLIC sample tests as psi evidence."""
    from paircoder import paper_solve, single_baseline
    q = build_prompt(prob)

    def chk(text):
        c = extract_code(text)
        if not c:
            return False, "no python code found"
        for t in prob["public_tests"]:
            if not check_solution(prob, c, [t], timeout=6):
                return False, (f"failed sample test. INPUT:\n{str(t.get('input'))[:300]}\n"
                               f"EXPECTED OUTPUT:\n{str(t.get('output'))[:300]}")
        return True, ""
    baseline = extract_code(single_baseline(q, model))
    pc_raw, _tel = paper_solve(q, model, max_iters=max_iters, check=chk)
    pc = extract_code(pc_raw)
    return baseline, pc, [baseline]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="all")
    ap.add_argument("--mode", choices=["paircoder", "baseline", "both"], default="both")
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--outdir", default="results_lcb")
    args = ap.parse_args()

    probs = load_lcb(args.version)
    if args.n != "all":
        probs = probs[: int(args.n)]
    os.makedirs(args.outdir, exist_ok=True)

    base_out = os.path.join(args.outdir, f"samples_baseline_lcb{args.version}_{args.model}.jsonl")
    pc_out = os.path.join(args.outdir, f"samples_paircoder_lcb{args.version}_{args.model}.jsonl")
    cand_out = os.path.join(args.outdir, f"samples_candidates_lcb{args.version}_{args.model}.jsonl")

    # resume: skip task_ids already present in BOTH output files
    def existing(path):
        d = {}
        if os.path.exists(path):
            for l in open(path):
                try:
                    r = json.loads(l); d[r["task_id"]] = r
                except Exception:
                    pass
        return d
    have = set(existing(base_out)) & set(existing(pc_out)) & set(existing(cand_out))
    todo = [p for p in probs if p["task_id"] not in have]
    print(f"\n=== PAIRED | LCB {args.version} | {args.model} | {len(probs)} probs "
          f"({len(have)} cached, {len(todo)} to do) | cands={args.cands} ===", flush=True)
    t0 = time.time()

    # incremental append (checkpoint): a key failure mid-run keeps everything finished so far
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
            if done % 20 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}]", flush=True)
    bf.close(); pf.close(); cf.close()
    print(f"  -> {base_out}\n  -> {pc_out}\n  -> {cand_out}\n  ({time.time()-t0:.0f}s, {USAGE['calls']} calls)", flush=True)


if __name__ == "__main__":
    main()
