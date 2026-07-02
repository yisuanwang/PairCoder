#!/usr/bin/env python3
"""DaTikZ visual metrics vs reference figure: SSIM + CLIP + DINO (base/pc arms).
Usage: score_datikz_vis.py <results jsonl> <art dir>   (conda python: torch+open_clip)"""
import sys, os, json
import numpy as np
from PIL import Image
import torch, open_clip
from transformers import AutoModel, AutoImageProcessor
from skimage.metrics import structural_similarity as ssim
res_file, art = sys.argv[1], sys.argv[2]
dev = "cuda" if torch.cuda.is_available() else "cpu"
clip_m, _, clip_pp = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_m = clip_m.to(dev).eval()
dino_m = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
dino_pp = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
def im(p): return Image.open(p).convert("RGB").resize((224,224))
def feat_clip(p):
    with torch.no_grad():
        f = clip_m.encode_image(clip_pp(im(p)).unsqueeze(0).to(dev)); return (f/f.norm(dim=-1,keepdim=True)).cpu().numpy()[0]
def feat_dino(p):
    d = {k:v.to(dev) for k,v in dino_pp(images=im(p), return_tensors="pt").items()}
    with torch.no_grad():
        f = dino_m(**d).pooler_output; return (f/f.norm(dim=-1,keepdim=True)).cpu().numpy()[0]
def arr(p): return np.asarray(im(p))
agg={"base":{"ssim":[],"clip":[],"dino":[],"ok":[]},"pc":{"ssim":[],"clip":[],"dino":[],"ok":[]}}
for l in open(res_file):
    r=json.loads(l); tid=r["task_id"]
    ref=os.path.join(art,f"{tid}_ref.png")
    if not os.path.exists(ref): continue
    rc,rd,ra=feat_clip(ref),feat_dino(ref),arr(ref)
    for armk,okk in (("base","base_ok"),("pc","pc_ok")):
        p=os.path.join(art,f"{tid}_{armk}.png")
        ok=r.get(okk) and os.path.exists(p)
        if ok:
            try: im(p)
            except Exception: ok=False
        agg[armk]["ok"].append(1.0 if ok else 0.0)
        if ok:
            agg[armk]["ssim"].append(float(ssim(arr(p),ra,channel_axis=2)))
            agg[armk]["clip"].append(float(np.dot(feat_clip(p),rc)))
            agg[armk]["dino"].append(float(np.dot(feat_dino(p),rd)))
        else:  # aggregate: failures count as 0
            agg[armk]["ssim"].append(0.0); agg[armk]["clip"].append(0.0); agg[armk]["dino"].append(0.0)
n=len(agg['base']['ok'])
print(f"==== DaTikZ visual metrics vs reference (n={n}, aggregate: fail=0) ====")
for m in ("ok","ssim","clip","dino"):
    print(f"  {m:5s}: base={np.mean(agg['base'][m]):.4f}  pc={np.mean(agg['pc'][m]):.4f}")
