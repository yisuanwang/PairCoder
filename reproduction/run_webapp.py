#!/usr/bin/env python3
"""PairCoder on WebApp1K-Duo-React (TDD web). Generates N React App candidates, runs the 4 Jest
tests on each (executable signal = the spec). baseline = candidate[0]; PairCoder keeps a candidate
that passes all tests, else repairs candidate[0] on the Jest failure output. Records per-candidate
pass for pass@k. Writes results jsonl with per-candidate booleans + paircoder pass."""
import os, sys, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from webapp_common import load_webapp, build_prompt, extract_jsx, has_app, run_tests

_lock = threading.Lock(); USAGE = {"calls": 0}
DRIVER_SYS = ("You are the Driver in a pair-programming team and an expert React engineer. "
              "Write a single complete React App component (default export) that makes the given "
              "Jest tests pass. Return ONLY one ```jsx``` code block. No tests, no explanation.")
ANGLES = ["Implement it directly and correctly.",
          "Carefully match every endpoint, label, button text, and message the tests expect.",
          "Handle both success and error/failure paths the tests check.",
          "Re-read each test assertion and implement exactly what it expects.",
          "Use React hooks (useState/useEffect) and fetch correctly; mind routing and async act()."]


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
            time.sleep(3 * (a + 1))


def solve(task, model, n_cand, max_iters):
    """Paper-faithful PairCoder: baseline = single direct generation; PairCoder = Driver/Navigator
    loop with the GIVEN Jest tests (TDD bench) as psi evidence."""
    from paircoder import paper_solve, single_baseline
    q = build_prompt(task)

    def chk(text):
        c = extract_jsx(text)
        if not has_app(c):
            return False, "no App component/export found"
        ok, out = run_tests(task, c)
        return (True, "") if ok else (False, "Jest failures:\n" + out[-600:])
    base = extract_jsx(single_baseline(q, model))
    base_ok, _ = run_tests(task, base) if has_app(base) else (False, "")
    pc_raw, tel = paper_solve(q, model, max_iters=max_iters, check=chk)
    pc = extract_jsx(pc_raw)
    pc_ok, _ = run_tests(task, pc) if has_app(pc) else (False, "")
    return {"task_id": task["task_id"], "cand_pass": [base_ok], "paircoder_pass": pc_ok,
            "iters": tel["iters"], "accepted": tel["accepted"]}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="200")
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--outdir", default="results_webapp")
    args = ap.parse_args()
    tasks = load_webapp(None if args.n == "all" else int(args.n))
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"results_webapp_{args.model}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["task_id"])
            except Exception: pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"\n=== WebApp1K | {args.model} | {len(tasks)} tasks ({len(done)} cached, {len(todo)} to do) ===", flush=True)
    fh = open(out, "a"); lk = threading.Lock(); t0 = time.time(); n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(solve, t, args.model, args.cands, args.max_iters) for t in todo]):
            try: r = f.result()
            except Exception: r = None
            if r is None: continue
            with lk:
                fh.write(json.dumps(r) + "\n"); fh.flush()
            n += 1
            if n % 10 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}]", flush=True)
    fh.close()
    print(f"  -> {out}  ({time.time()-t0:.0f}s, {USAGE['calls']} calls)", flush=True)


if __name__ == "__main__":
    main()
