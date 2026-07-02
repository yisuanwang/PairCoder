#!/usr/bin/env python3
"""DS-1000 pass@k grader. Usage: python grade_ds.py <candidates.jsonl> <paircoder.jsonl> [workers] [timeout]"""
import sys, json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from ds_common import load_ds, run_test

KS = [1, 3, 5]


def main():
    cand_file, pc_file = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    timeout = float(sys.argv[4]) if len(sys.argv) > 4 else 30.0
    probs = {p["task_id"]: p for p in load_ds()}
    cands = {json.loads(l)["task_id"]: json.loads(l)["candidates"] for l in open(cand_file)}
    pcs = {json.loads(l)["task_id"]: json.loads(l)["solution"] for l in open(pc_file)}
    jobs = []
    for tid, cl in cands.items():
        if tid not in probs: continue
        for i, c in enumerate(cl): jobs.append((tid, i, c))
        jobs.append((tid, "pc", pcs.get(tid, "")))
    res = defaultdict(dict)
    def grade(j):
        tid, idx, code = j
        return tid, idx, run_test(probs[tid], code, timeout=timeout)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        d = 0
        for f in as_completed([ex.submit(grade, j) for j in jobs]):
            tid, idx, ok = f.result(); res[tid][idx] = ok; d += 1
            if d % 200 == 0: print(f"  graded {d}/{len(jobs)}", file=sys.stderr, flush=True)
    n = len(res)
    print(f"\n==== DS-1000 pass@k :: {cand_file} (n={n}) ====")
    line = "  "
    for k in KS:
        c = sum(1 for tid in res if any(res[tid].get(i, False) for i in range(k)))
        line += f"base@{k}={c/n:.3f}  "
    pc = sum(1 for tid in res if res[tid].get("pc", False)) / n
    print(line + f"|  PairCoder@1={pc:.3f}")


if __name__ == "__main__":
    main()
