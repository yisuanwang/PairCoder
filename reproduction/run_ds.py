#!/usr/bin/env python3
"""PairCoder on DS-1000 (data-science code gen). Paired best-of-N with EXECUTION-CONSENSUS
selection (run candidates on the context's test input, majority-vote the result; gold never used).
Dumps candidates for pass@k."""
import os, sys, json, time, argparse, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from ds_common import load_ds, build_prompt, extract_code, signature

_lock = threading.Lock(); USAGE = {"calls": 0}
DRIVER_SYS = ("You are the Driver in a pair-programming team and an expert data scientist. "
              "Write the solution snippet that computes `result` as specified. Return ONLY one "
              "```python``` block with the snippet (no function wrapper, no prints, no tests).")
ANGLES = ["Write the most direct correct solution.",
          "Match the exact output type/shape/format requested.",
          "Handle edge cases and dtypes carefully.",
          "Re-read the problem and reproduce exactly the required transformation.",
          "Use idiomatic pandas/numpy/scipy/sklearn correctly."]


def llm(model, system, user, retries=4):
    from paircoder.client import make_client, guarded_create
    client = make_client()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for a in range(retries):
        try:
            r = guarded_create(client, model=model, messages=msgs, extra_body={"reasoning_effort": os.environ.get("PAIRCODER_EFFORT","none")})
            if r.usage:
                with _lock: USAGE["calls"] += 1
            return r.choices[0].message.content or ""
        except Exception as e:
            if a == retries - 1:
                print(f"[api error] {e}", file=sys.stderr); return ""
            time.sleep(2 * (a + 1))


def solve_paired(prob, model, n_cand):
    """Paper-faithful PairCoder: baseline = single direct generation; PairCoder = Driver/Navigator
    loop with execution-on-context (signature) as psi evidence."""
    from paircoder import paper_solve, single_baseline
    q = build_prompt(prob)

    def chk(text):
        c = extract_code(text)
        if not c:
            return False, "no python code found"
        s_ = signature(prob, c)
        ok = bool(s_) and s_.startswith("SIG:")
        return (True, "") if ok else (False, ("execution failed: " + str(s_)[:300]))
    baseline = extract_code(single_baseline(q, model))
    pc_raw, _tel = paper_solve(q, model, max_iters=3, check=chk)
    pc = extract_code(pc_raw)
    return baseline, pc, [baseline]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="all")
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--outdir", default="results_ds")
    args = ap.parse_args()
    probs = load_ds()
    if args.n != "all":
        probs = probs[: int(args.n)]
    os.makedirs(args.outdir, exist_ok=True)
    base_out = os.path.join(args.outdir, f"samples_baseline_ds_{args.model}.jsonl")
    pc_out = os.path.join(args.outdir, f"samples_paircoder_ds_{args.model}.jsonl")
    cand_out = os.path.join(args.outdir, f"samples_candidates_ds_{args.model}.jsonl")

    def ids(p):
        s = set()
        if os.path.exists(p):
            for l in open(p):
                try: s.add(json.loads(l)["task_id"])
                except Exception: pass
        return s
    have = ids(base_out) & ids(pc_out) & ids(cand_out)
    todo = [p for p in probs if p["task_id"] not in have]
    print(f"\n=== PAIRED | DS-1000 | {args.model} | {len(probs)} probs ({len(have)} cached, {len(todo)} to do) ===", flush=True)
    t0 = time.time()
    bf, pf, cf = open(base_out, "a"), open(pc_out, "a"), open(cand_out, "a")
    lk = threading.Lock()

    def work(prob):
        b, p, cands = solve_paired(prob, args.model, args.cands)
        return prob["task_id"], b, p, cands

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(work, p) for p in todo]):
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
    print(f"  -> {pc_out}  ({time.time()-t0:.0f}s, {USAGE['calls']} calls)", flush=True)


if __name__ == "__main__":
    main()
