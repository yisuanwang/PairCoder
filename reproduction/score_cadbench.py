#!/usr/bin/env python3
"""Score GenCAD image->CadQuery: execution rate + Chamfer (gen STL vs ref STL). venv python (trimesh).
Usage: python score_cadbench.py <results.jsonl> <art_dir>"""
import sys, os, json, numpy as np, trimesh
from scipy.spatial import cKDTree
res, art = sys.argv[1], sys.argv[2]
PEN = 1.0
def norm_pts(stl, n=4000):
    try:
        m = trimesh.load(stl, force="mesh")
        e = m.extents.max() or 1.0
        m.apply_translation(-m.centroid); m.apply_scale(1.0/e)
        return m.sample(n)
    except Exception: return None
def chamfer(a, b):
    pa, pb = norm_pts(a), norm_pts(b)
    if pa is None or pb is None: return None
    return float(cKDTree(pb).query(pa)[0].mean() + cKDTree(pa).query(pb)[0].mean())
rows = [json.loads(l) for l in open(res)]
rows = [r for r in rows if r.get("ref_ok")]
n = len(rows)
out = {"base": {"ok": [], "ch_c": [], "ch_a": []}, "pc": {"ok": [], "ch_c": [], "ch_a": []}}
for r in rows:
    tid = r["task_id"]; refstl = os.path.join(art, f"{tid}_ref.stl")
    for side in ["base", "pc"]:
        ok = r[f"{side}_ok"]; out[side]["ok"].append(1.0 if ok else 0.0)
        stl = os.path.join(art, f"{tid}_{side}.stl")
        c = chamfer(stl, refstl) if (ok and os.path.exists(stl)) else None
        if c is not None: out[side]["ch_c"].append(c); out[side]["ch_a"].append(c)
        else: out[side]["ch_a"].append(PEN)
m = lambda x: round(float(np.mean(x)), 4) if x else None
print(f"\n==== GenCAD image->CadQuery (n={n}) — baseline vs PairCoder ====")
print(f"  Execution-rate↑: base={m(out['base']['ok'])}  pc={m(out['pc']['ok'])}")
print(f"  Chamfer↓ (cond): base={m(out['base']['ch_c'])}  pc={m(out['pc']['ch_c'])}")
print(f"  Chamfer↓ (agg) : base={m(out['base']['ch_a'])}  pc={m(out['pc']['ch_a'])}")
