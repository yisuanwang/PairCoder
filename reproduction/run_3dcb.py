#!/usr/bin/env python3
"""PairCoder on 3DCodeBench (Blender procedural 3D). Metric = execution pass rate (script runs
in Blender and produces mesh geometry). baseline = candidate[0]; PairCoder keeps a candidate that
runs+produces geometry, else repairs candidate[0] on the Blender error. Dumps candidates for pass@k."""
import os, sys, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from blender_common import load_3dcb, build_prompt, extract_code, run_in_blender

_lock = threading.Lock(); USAGE = {"calls": 0}
DRIVER_SYS = ("You are the Driver in a pair-programming team and an expert Blender/bpy technical "
              "artist. Write a Blender 4.2 Python script that procedurally builds the described "
              "object as real mesh geometry. Return ONLY one ```python``` block. No explanation.")
ANGLES = ["Build it directly with bpy primitives and mesh operations.",
          "Decompose the object into parts and create each as mesh geometry.",
          "Use bpy.ops mesh primitives + transforms/modifiers; ensure real vertices are created.",
          "Re-read the description and model each named component.",
          "Keep the bpy API usage correct for Blender 4.2 so it runs headless without errors."]


def llm(model, system, user, retries=4):
    from paircoder.client import make_client, guarded_create
    client = make_client()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for a in range(retries):
        try:
            r = guarded_create(client, model=model, messages=msgs, extra_body={"reasoning_effort": "none"})
            if r.usage:
                with _lock: USAGE["calls"] += 1
            return r.choices[0].message.content or ""
        except Exception as e:
            if a == retries - 1:
                print(f"[api error] {e}", file=sys.stderr); return ""
            time.sleep(3 * (a + 1))


def solve(task, model, n_cand, max_iters):
    """Paper-faithful PairCoder: baseline = single direct generation; PairCoder = Driver/Navigator
    loop with Blender headless execution as psi evidence."""
    from paircoder import paper_solve, single_baseline
    q = build_prompt(task)

    def chk(text):
        c = extract_code(text)
        if not c:
            return False, "no python code found"
        ok, nv, err = run_in_blender(c)
        return (True, "") if ok else (False, ("Blender error: " + str(err)[:400]))
    base = extract_code(single_baseline(q, model))
    base_ok, _, _ = run_in_blender(base) if base else (False, 0, "empty")
    pc_raw, tel = paper_solve(q, model, max_iters=max_iters, check=chk)
    pc = extract_code(pc_raw)
    pc_ok, _, _ = run_in_blender(pc) if pc else (False, 0, "empty")
    return {"task_id": task["task_id"], "cand_ok": [base_ok], "baseline_ok": base_ok,
            "paircoder_ok": pc_ok, "iters": tel["iters"], "accepted": tel["accepted"],
            "base_code": base, "pc_code": pc}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="80")
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="results_3dcb")
    args = ap.parse_args()
    tasks = load_3dcb(None if args.n == "all" else int(args.n))
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, f"results_3dcb_{args.model}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["task_id"])
            except Exception: pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"\n=== 3DCodeBench | {args.model} | {len(tasks)} tasks ({len(done)} cached, {len(todo)} to do) ===", flush=True)
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
