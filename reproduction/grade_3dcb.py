#!/usr/bin/env python3
"""Aggregate 3DCodeBench execution results -> exec pass@1/3/5 + PairCoder@1."""
import sys, json
f=sys.argv[1]; rows=[json.loads(l) for l in open(f)]; n=len(rows); KS=[1,3,5]
print(f"\n==== 3DCodeBench (exec rate) :: {f} (n={n}) ====")
line="  "
for k in KS:
    c=sum(1 for r in rows if any((r.get('cand_ok') or [False])[:k]))
    line+=f"base@{k}={c/n:.3f}  "
pc=sum(1 for r in rows if r.get('paircoder_ok'))/n
print(line+f"|  PairCoder@1={pc:.3f}")
