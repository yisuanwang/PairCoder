#!/usr/bin/env python3
"""Score 3DCodeBench full metrics from a manifest (run with CONDA python: has torch).
Reports baseline vs PairCoder on Executability, Chamfer (cond + aggregate), SigLIP-2, DINO.
Usage: <conda-python> tdcb_score_run.py <manifest.jsonl>"""
import sys, json, numpy as np
from tdcb_score import chamfer, siglip_sim, dino_sim

CHAMFER_PENALTY = 6.0   # worst-case: larger than any executing-shape Chamfer (~2-3)


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    rows = [r for r in rows if r.get("ref_obj") and r.get("ref_png")]   # need a reference
    n = len(rows)
    base_ok = [r["base_ok"] for r in rows]; pc_ok = [r["pc_ok"] for r in rows]

    def metrics(side):
        ch_c, sg_c, dn_c = [], [], []      # conditional (only when this side executed)
        ch_a, sg_a, dn_a = [], [], []      # aggregate (failed -> worst)
        for r in rows:
            ok = r[f"{side}_ok"]
            if ok and r[f"{side}_obj"]:
                c = chamfer(r[f"{side}_obj"], r["ref_obj"])
                s = siglip_sim(r[f"{side}_png"], r["ref_png"])
                d = dino_sim(r[f"{side}_png"], r["ref_png"])
                if c is not None: ch_c.append(c); ch_a.append(c)
                else: ch_a.append(CHAMFER_PENALTY)
                if s is not None: sg_c.append(s); sg_a.append(s)
                else: sg_a.append(0.0)
                if d is not None: dn_c.append(d); dn_a.append(d)
                else: dn_a.append(0.0)
            else:
                ch_a.append(CHAMFER_PENALTY); sg_a.append(0.0); dn_a.append(0.0)
        m = lambda x: round(float(np.mean(x)), 4) if x else None
        return {"chamfer_cond": m(ch_c), "siglip_cond": m(sg_c), "dino_cond": m(dn_c),
                "chamfer_agg": m(ch_a), "siglip_agg": m(sg_a), "dino_agg": m(dn_a)}

    b = metrics("base"); p = metrics("pc")
    print(f"\n==== 3DCodeBench FULL metrics (n={n}, ref available) ====")
    print(f"  Executability:        baseline={np.mean(base_ok):.3f}   PairCoder={np.mean(pc_ok):.3f}   (↑)")
    print(f"  -- conditional on executing --")
    print(f"  Chamfer↓ :  baseline={b['chamfer_cond']}   PairCoder={p['chamfer_cond']}")
    print(f"  SigLIP-2↑:  baseline={b['siglip_cond']}   PairCoder={p['siglip_cond']}")
    print(f"  DINO↑    :  baseline={b['dino_cond']}   PairCoder={p['dino_cond']}")
    print(f"  -- aggregate (post-loop; non-exec = worst) --")
    print(f"  Chamfer↓ :  baseline={b['chamfer_agg']}   PairCoder={p['chamfer_agg']}")
    print(f"  SigLIP-2↑:  baseline={b['siglip_agg']}   PairCoder={p['siglip_agg']}")
    print(f"  DINO↑    :  baseline={b['dino_agg']}   PairCoder={p['dino_agg']}")


if __name__ == "__main__":
    main()
