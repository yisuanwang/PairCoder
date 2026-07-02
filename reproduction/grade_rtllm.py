"""Aggregate RTLLM results from a JSONL: reports baseline vs PairCoder pass@1/3/5
(a design passes iff its generated Verilog compiles and passes the testbench). Usage: python grade_rtllm.py <results.jsonl>"""
import sys,json,numpy as np
rows=[json.loads(l) for l in open(sys.argv[1])]; n=len(rows); KS=[1,3,5]
print(f"\n==== RTLLM Verilog (n={n}) ====")
line="  "
for k in KS:
    c=sum(1 for r in rows if any((r.get('cand_pass') or [False])[:k]))
    line+=f"base@{k}={c/n:.3f}  "
pc=sum(1 for r in rows if r.get('paircoder_pass'))/n
print(line+f"|  PairCoder@1={pc:.3f}")
