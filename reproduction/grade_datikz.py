"""Grade DaTikZ compile-rate from a results JSONL: reports baseline vs PairCoder
fraction of captions whose generated TikZ/LaTeX compiled. Usage: python grade_datikz.py <results.jsonl>"""
import sys, json, numpy as np
rows=[json.loads(l) for l in open(sys.argv[1])]; n=len(rows)
b=np.mean([r['base_ok'] for r in rows]); p=np.mean([r['pc_ok'] for r in rows])
print(f"\n==== DaTikZ caption->TikZ compile-rate (n={n}) ====")
print(f"  compile-rate↑: baseline={b:.3f}  PairCoder={p:.3f}")
