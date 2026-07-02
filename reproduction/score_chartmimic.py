#!/usr/bin/env python3
"""Score ChartMimic: exec-rate + SSIM + CLIP between rendered chart and target image (conda python).
Usage: <conda-python> score_chartmimic.py <results.jsonl> <art_dir>"""
import sys, os, json, numpy as np
import torch, open_clip
from PIL import Image
from skimage.metrics import structural_similarity as ssim
res, art = sys.argv[1], sys.argv[2]
dev="cuda"; m,_,pp=open_clip.create_model_and_transforms("ViT-B-32",pretrained="openai"); m=m.to(dev).eval()
def clipf(png):
    x=pp(Image.open(png).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad(): f=m.encode_image(x)
    return (f/f.norm(dim=-1,keepdim=True)).cpu().numpy().reshape(-1)
def ssim_p(a,b):
    A=np.asarray(Image.open(a).convert("RGB").resize((256,256))); B=np.asarray(Image.open(b).convert("RGB").resize((256,256)))
    return float(ssim(A,B,channel_axis=2))
rows=[json.loads(l) for l in open(res)]; n=len(rows)
agg={"base":{"ok":[],"ssim":[],"clip":[]},"pc":{"ok":[],"ssim":[],"clip":[]}}
for r in rows:
    tid=r["task_id"]; gt=os.path.join(art,f"{tid}_gt.png")
    if not os.path.exists(gt): continue
    gf=clipf(gt)
    for side in ["base","pc"]:
        p=os.path.join(art,f"{tid}_{side}.png"); ok=os.path.exists(p)
        agg[side]["ok"].append(1.0 if ok else 0.0)
        agg[side]["ssim"].append(ssim_p(p,gt) if ok else 0.0)
        agg[side]["clip"].append(float(clipf(p)@gf) if ok else 0.0)
M=lambda x: round(float(np.mean(x)),4) if x else None
print(f"\n==== ChartMimic image->matplotlib (n={len(agg['base']['ok'])}) — baseline vs PairCoder ====")
print(f"  Exec-rate↑: base={M(agg['base']['ok'])}  pc={M(agg['pc']['ok'])}")
print(f"  SSIM↑     : base={M(agg['base']['ssim'])}  pc={M(agg['pc']['ssim'])}")
print(f"  CLIP↑     : base={M(agg['base']['clip'])}  pc={M(agg['pc']['clip'])}")
