#!/usr/bin/env python3
"""PairCoder on DaTikZ (caption -> TikZ/LaTeX). Metric = compile-rate (generated LaTeX compiles
to a non-empty image via pdflatex). baseline = candidate[0]; PairCoder keeps a candidate that
compiles, else repairs candidate[0] on the pdflatex error. Records tokens (pc_client.TOKENS)."""
import os, sys, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datikz_common import load_datikz, build_prompt, extract_latex, compile_tikz, latex_err

_lock = threading.Lock(); USAGE = {"calls": 0}
SYS = ("You are the Driver, an expert in LaTeX/TikZ. Produce a COMPLETE compilable standalone "
       "LaTeX document (pdflatex) that draws the described figure. Return ONLY one ```latex``` block.")
ANGLES = ["Draw it directly and make sure it compiles.",
          "Use standard tikz libraries; keep the document self-contained.",
          "Reproduce the described structure (nodes, edges, shapes, labels).",
          "Prefer simple robust TikZ that compiles with a base texlive.",
          "Match the figure's layout and elements; ensure valid LaTeX syntax."]


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
            if a == retries - 1: return ""
            time.sleep(3 * (a + 1))


def solve(task, model, n_cand, max_iters, art):
    from paircoder import paper_solve, single_baseline
    tid = task["task_id"]; q = build_prompt(task)

    def chk(text):
        c = extract_latex(text)
        png = os.path.join(art, f"{tid}_chk.png")
        good = compile_tikz(c, png)
        return (True, "") if good else (False, latex_err(c)[:300])
    base = extract_latex(single_baseline(q, model))
    base_ok = compile_tikz(base, os.path.join(art, f"{tid}_base.png"))
    pc_raw, tel = paper_solve(q, model, max_iters=max_iters, check=chk)
    pc = extract_latex(pc_raw)
    pc_ok = compile_tikz(pc, os.path.join(art, f"{tid}_pc.png"))
    return {"task_id": tid, "base_ok": base_ok, "pc_ok": pc_ok,
            "iters": tel["iters"], "accepted": tel["accepted"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="60"); ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--outdir", default="results_datikz")
    args = ap.parse_args()
    tasks = load_datikz(None if args.n == "all" else int(args.n))
    art = os.path.join(args.outdir, "art"); os.makedirs(art, exist_ok=True)
    out = os.path.join(args.outdir, f"results_datikz_{args.model}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["task_id"])
            except Exception: pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"=== DaTikZ caption->TikZ | {args.model} | {len(tasks)} ({len(done)} cached) ===", flush=True)
    fh = open(out, "a"); lk = threading.Lock(); n = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(solve, t, args.model, args.cands, args.max_iters, art) for t in todo]):
            try: r = f.result()
            except Exception: r = None
            if r is None: continue
            with lk: fh.write(json.dumps(r) + "\n"); fh.flush()
            n += 1
            if n % 10 == 0 or n == len(todo): print(f"  [{n}/{len(todo)}]", flush=True)
    fh.close()
    from paircoder import client as pc_client
    print(f"  -> {out} ({time.time()-t0:.0f}s) TOKENS={pc_client.TOKENS}", flush=True)


if __name__ == "__main__":
    main()
