#!/usr/bin/env python3
"""LiveCodeBench pass@k grader.

Baseline pass@k = oracle best-of-k over the k diverse Driver candidates (candidate[0] is the
direct attempt = pass@1). Since the model is deterministic (temperature ignored by the proxy),
diversity comes from the angle-prompted candidates, so we report ORDERED best-of-k.
PairCoder reports pass@1 (a single selected solution).

Usage: python grade_lcb_passk.py <candidates.jsonl> <paircoder.jsonl> [version] [workers] [timeout]
"""
import sys, json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from lcb_common import load_lcb, check_solution

KS = [1, 3, 5]


def main():
    cand_file = sys.argv[1]
    pc_file = sys.argv[2]
    version = sys.argv[3] if len(sys.argv) > 3 else "v1"
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 48
    timeout = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0

    probs = {p["task_id"]: p for p in load_lcb(version)}
    cands = {json.loads(l)["task_id"]: json.loads(l)["candidates"] for l in open(cand_file)}
    pcs = {json.loads(l)["task_id"]: json.loads(l)["solution"] for l in open(pc_file)}

    # grade each candidate of each problem + the paircoder solution
    jobs = []
    for tid, cl in cands.items():
        if tid not in probs:
            continue
        for i, c in enumerate(cl):
            jobs.append((tid, i, c))
        jobs.append((tid, "pc", pcs.get(tid, "")))

    results = defaultdict(dict)  # tid -> {idx: bool}

    def grade(job):
        tid, idx, code = job
        return tid, idx, check_solution(probs[tid], code, probs[tid]["all_tests"], timeout=timeout)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(grade, j) for j in jobs]
        d = 0
        for f in as_completed(futs):
            tid, idx, ok = f.result(); results[tid][idx] = ok; d += 1
            if d % 200 == 0:
                print(f"  graded {d}/{len(jobs)}", file=sys.stderr, flush=True)

    # aggregate
    diffs = {tid: probs[tid]["difficulty"] for tid in results}
    def rate(pred):
        sel = [tid for tid in results if pred(diffs[tid])]
        if not sel:
            return None
        out = {}
        for k in KS:
            c = sum(1 for tid in sel if any(results[tid].get(i, False) for i in range(k)))
            out[f"base@{k}"] = c / len(sel)
        out["pc@1"] = sum(1 for tid in sel if results[tid].get("pc", False)) / len(sel)
        out["n"] = len(sel)
        return out

    print(f"\n==== LCB {version} pass@k :: {cand_file} ====")
    for label, pred in [("easy", lambda d: d == "easy"), ("medium", lambda d: d == "medium"),
                        ("hard", lambda d: d == "hard"), ("OVERALL", lambda d: True)]:
        r = rate(pred)
        if r:
            print(f"  {label:8s} (n={r['n']:3d}): "
                  f"base@1={r['base@1']:.3f}  base@3={r['base@3']:.3f}  base@5={r['base@5']:.3f}  "
                  f"|  PairCoder@1={r['pc@1']:.3f}")


if __name__ == "__main__":
    main()
