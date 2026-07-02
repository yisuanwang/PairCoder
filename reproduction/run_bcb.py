#!/usr/bin/env python3
"""PairCoder on BigCodeBench (paired design; doctest grounding).
baseline = Driver's direct candidate[0]; paircoder = robust selection over a best-of-N pool
(keep candidate[0] if it passes the docstring examples, else a passing sibling, else repair).
Grading (grade_bcb.py) runs the hidden unittest suite — identical for both modes."""
import os, sys, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from bcb_common import load_bcb, libs_available, build_prompt, extract_code, public_ok, has_entry

API_BASE = os.environ.get("PAIRCODER_API_BASE", "https://api.openai.com/v1")
API_KEY  = os.environ.get("PAIRCODER_API_KEY",  "")

_lock = threading.Lock()
USAGE = {"calls": 0}

DRIVER_SYS = (
    "You are the Driver in a pair-programming team and an expert Python engineer. Implement "
    "the requested function fully and correctly, honoring the docstring, requirements, and "
    "return format. Return ONLY one ```python``` block with the complete self-contained "
    "solution (all imports + the function). No prose.")
ANGLES = [
    "Write the most direct correct implementation.",
    "Follow the docstring's Requirements and Returns exactly; match types and output format.",
    "Handle edge cases and invalid inputs as the docstring implies.",
    "Re-read the spec and examples; implement precisely what is asked.",
    "Use the standard idioms of the required libraries correctly.",
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


def solve_paired(prob, model, n_cand, max_iters):
    """Paper-faithful PairCoder: baseline = single direct generation; PairCoder = Driver/Navigator
    loop with doctest psi + a NAVIGATOR-authored unit test (TDD review) as evidence."""
    from paircoder import paper_solve, single_baseline
    from bcb_common import run_nav_test
    q = build_prompt(prob); ep = prob["entry_point"]
    AUTHOR = (f"Write a plain Python test SCRIPT (no unittest/pytest framework) for the function "
              f"`{ep}` described in the task. The function will already be defined above your "
              f"script. Derive several concrete input/expected-output cases STRICTLY from the "
              f"docstring (including edge cases), call the function, and print('NAV_FAIL', detail) "
              f"on any mismatch; print('NAV_DONE') at the end. Return ONLY one ```python``` block.")

    def chk(text):
        c = extract_code(text)
        if not has_entry(c, ep):
            return False, f"no function {ep} found"
        return (True, "") if public_ok(prob, c) else (False, "fails the docstring example checks")

    def run_tb(code_text, tb_text):
        return run_nav_test(extract_code(code_text), extract_code(tb_text))
    baseline = extract_code(single_baseline(q, model))
    pc_raw, _tel = paper_solve(q, model, max_iters=max_iters, check=chk,
                               author_test=AUTHOR, run_authored_test=run_tb)
    pc = extract_code(pc_raw)
    return baseline, pc, [baseline]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v0.1.4")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="all")
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--outdir", default="results_bcb")
    ap.add_argument("--require-libs", action="store_true", help="only keep problems whose libs import")
    args = ap.parse_args()

    probs = load_bcb(args.version)
    if args.require_libs:
        probs = [p for p in probs if libs_available(p["libs"])]
        print(f"after lib filter: {len(probs)} problems", flush=True)
    if args.n != "all":
        probs = probs[: int(args.n)]
    os.makedirs(args.outdir, exist_ok=True)

    base_out = os.path.join(args.outdir, f"samples_baseline_bcb_{args.model}.jsonl")
    pc_out = os.path.join(args.outdir, f"samples_paircoder_bcb_{args.model}.jsonl")
    cand_out = os.path.join(args.outdir, f"samples_candidates_bcb_{args.model}.jsonl")

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
    print(f"\n=== PAIRED | BCB {args.version} | {args.model} | {len(probs)} probs "
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
            if done % 25 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}]", flush=True)
    bf.close(); pf.close(); cf.close()
    print(f"  -> {base_out}\n  -> {pc_out}\n  -> {cand_out}\n  ({time.time()-t0:.0f}s, {USAGE['calls']} calls)", flush=True)


if __name__ == "__main__":
    main()
