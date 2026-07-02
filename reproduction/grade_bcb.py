#!/usr/bin/env python3
"""Grade BigCodeBench samples by running the hidden unittest suite per problem.
Usage: python grade_bcb.py <samples.jsonl> [version] [workers] [timeout]"""
import sys, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from bcb_common import load_bcb, run_unittest


def main():
    sample_file = sys.argv[1]
    version = sys.argv[2] if len(sys.argv) > 2 else "v0.1.4"
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    timeout = float(sys.argv[4]) if len(sys.argv) > 4 else 40.0

    probs = {p["task_id"]: p for p in load_bcb(version)}
    samples = {}
    for l in open(sample_file):
        d = json.loads(l); samples[d["task_id"]] = d["solution"]

    def grade(tid):
        return tid, run_unittest(probs[tid], samples.get(tid, ""), timeout=timeout)

    ok = 0; n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(grade, tid) for tid in samples if tid in probs]
        for i, f in enumerate(as_completed(futs), 1):
            tid, passed = f.result(); ok += int(passed); n += 1
            if i % 50 == 0:
                print(f"  graded {i}/{len(futs)}", file=sys.stderr, flush=True)

    print(f"\n==== {sample_file} (BCB {version}) ====")
    print(f"  pass@1: {ok/n:.3f}  ({ok}/{n})")


if __name__ == "__main__":
    main()
