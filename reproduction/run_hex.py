#!/usr/bin/env python3
"""PairCoder on HumanEval-X multilingual (C++, Java, JavaScript). Paired best-of-N with
compile/example-test grounding; dumps candidates for pass@k. Grading = full hidden test."""
import os, sys, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from hex_common import load_hex, build_prompt, extract_code, public_ok, has_code

API_BASE = os.environ.get("PAIRCODER_API_BASE", "https://api.openai.com/v1")
API_KEY  = os.environ.get("PAIRCODER_API_KEY",  "")
_lock = threading.Lock(); USAGE = {"calls": 0}

ANGLES = ["Write the most direct correct implementation.",
          "Match the required signature and types exactly.",
          "Handle edge cases and boundaries carefully.",
          "Re-read the spec and implement precisely what is asked.",
          "Use clean idiomatic code for this language."]


def driver_sys(lang):
    names = {"cpp": "C++", "java": "Java", "js": "JavaScript"}
    return (f"You are the Driver in a pair-programming team and an expert {names[lang]} engineer. "
            f"Return ONLY one ```{lang}``` block with the complete self-contained solution "
            f"(imports/headers + required function/class, NO main, NO tests). No prose.")


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
    from paircoder import paper_solve, single_baseline
    lang = prob["lang"]; q = build_prompt(prob, lang)

    def chk(text):
        c = extract_code(text, lang)
        if not has_code(c, lang):
            return False, f"no {lang} code found"
        return (True, "") if public_ok(prob, c) else (False, "fails to compile or fails the example checks from the docstring")
    baseline = extract_code(single_baseline(q, model), lang)
    pc_raw, _tel = paper_solve(q, model, max_iters=max_iters, check=chk)
    pc = extract_code(pc_raw, lang)
    return baseline, pc, [baseline]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["cpp", "java", "js"])
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="all")
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--outdir", default="results_hex")
    args = ap.parse_args()

    probs = load_hex(args.lang)
    if args.n != "all":
        probs = probs[: int(args.n)]
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"{args.lang}_{args.model}"
    base_out = os.path.join(args.outdir, f"samples_baseline_hex{tag}.jsonl")
    pc_out = os.path.join(args.outdir, f"samples_paircoder_hex{tag}.jsonl")
    cand_out = os.path.join(args.outdir, f"samples_candidates_hex{tag}.jsonl")

    def ex_ids(path):
        s = set()
        if os.path.exists(path):
            for l in open(path):
                try: s.add(json.loads(l)["task_id"])
                except Exception: pass
        return s
    have = ex_ids(base_out) & ex_ids(pc_out) & ex_ids(cand_out)
    todo = [p for p in probs if p["task_id"] not in have]
    print(f"\n=== PAIRED | HumanEval-X {args.lang} | {args.model} | {len(probs)} probs "
          f"({len(have)} cached, {len(todo)} to do) ===", flush=True)
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
            if done % 20 == 0 or done == len(todo):
                print(f"  [{done}/{len(todo)}]", flush=True)
    bf.close(); pf.close(); cf.close()
    print(f"  -> {pc_out}  ({time.time()-t0:.0f}s, {USAGE['calls']} calls)", flush=True)


if __name__ == "__main__":
    main()
