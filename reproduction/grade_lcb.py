#!/usr/bin/env python3
"""Grade LiveCodeBench samples: a problem is solved iff the solution passes ALL tests.
Usage: python grade_lcb.py <samples.jsonl> [version] [workers] [per_test_timeout]"""
import sys, json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from lcb_common import load_lcb, check_solution


def main():
    sample_file = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else "v1"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 48
    timeout = float(sys.argv[4]) if len(sys.argv) > 4 else 8.0

    probs = {p["task_id"]: p for p in load_lcb(version)}
    samples = {json.loads(l)["task_id"]: json.loads(l)["solution"] for l in open(sample_file)}

    def grade(tid):
        p = probs[tid]
        ok = check_solution(p, samples.get(tid, ""), p["all_tests"], timeout=timeout)
        return tid, p["difficulty"], ok

    by_diff = defaultdict(lambda: [0, 0])
    total = [0, 0]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(grade, tid) for tid in samples if tid in probs]
        done = 0
        for f in as_completed(futs):
            tid, diff, ok = f.result()
            by_diff[diff][1] += 1; by_diff[diff][0] += int(ok)
            total[1] += 1; total[0] += int(ok)
            done += 1
            if done % 50 == 0:
                print(f"  graded {done}/{len(futs)}", file=sys.stderr, flush=True)

    print(f"\n==== {sample_file} (LCB {version}) ====")
    for d in ["easy", "medium", "hard"]:
        if d in by_diff:
            c, n = by_diff[d]
            print(f"  {d:7s}: {c/n:.3f}  ({c}/{n})")
    print(f"  OVERALL: {total[0]/total[1]:.3f}  ({total[0]}/{total[1]})")


if __name__ == "__main__":
    main()
