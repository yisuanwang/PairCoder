#!/usr/bin/env python3
"""Score StarVector image->SVG: SSIM (from results) + CLIP-image-sim + DINO-sim between the
generated render and the reference render. Run with CONDA python (torch). baseline vs PairCoder.
Usage: <conda-python> score_svgbench.py <results.jsonl> <art_dir>"""
import sys, os, json, numpy as np
import torch, open_clip
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

res_file, art = sys.argv[1], sys.argv[2]
dev = "cuda"
clip_m, _, clip_pp = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
clip_m = clip_m.to(dev).eval()
dino_m = AutoModel.from_pretrained("facebook/dinov2-base").to(dev).eval()
dino_pp = AutoImageProcessor.from_pretrained("facebook/dinov2-base")


def feat_clip(png):
    img = clip_pp(Image.open(png).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        f = clip_m.encode_image(img)
    return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy().reshape(-1)


def feat_dino(png):
    img = Image.open(png).convert("RGB")
    p = {k: v.to(dev) for k, v in dino_pp(images=img, return_tensors="pt").items()}
    with torch.no_grad():
        o = dino_m(**p); f = o.pooler_output
    return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy().reshape(-1)


def sim(a, b):
    try:
        return float(a @ b)
    except Exception:
        return None


rows = [json.loads(l) for l in open(res_file)]
agg = {"base": {"ssim": [], "clip": [], "dino": [], "render": []},
       "pc": {"ssim": [], "clip": [], "dino": [], "render": []}}
for r in rows:
    tid = r["task_id"]; refp = os.path.join(art, f"{tid}_ref.png")
    if not os.path.exists(refp):
        continue
    rf_c, rf_d = feat_clip(refp), feat_dino(refp)
    for side in ["base", "pc"]:
        genp = os.path.join(art, f"{tid}_{side}.png")
        rendered = os.path.exists(genp)
        agg[side]["render"].append(1.0 if rendered else 0.0)
        # aggregate: non-rendering -> worst (ssim 0, clip/dino 0)
        agg[side]["ssim"].append(max(r.get(f"{side}_ssim", -1.0), 0.0))
        if rendered:
            agg[side]["clip"].append(sim(feat_clip(genp), rf_c))
            agg[side]["dino"].append(sim(feat_dino(genp), rf_d))
        else:
            agg[side]["clip"].append(0.0); agg[side]["dino"].append(0.0)

m = lambda x: round(float(np.mean([v for v in x if v is not None])), 4) if x else None
n = len(agg["base"]["ssim"])
print(f"\n==== StarVector image->SVG (n={n}) — baseline vs PairCoder ====")
print(f"  render-rate: base={m(agg['base']['render'])}  pc={m(agg['pc']['render'])}  (↑)")
print(f"  SSIM↑      : base={m(agg['base']['ssim'])}  pc={m(agg['pc']['ssim'])}")
print(f"  CLIP-img↑  : base={m(agg['base']['clip'])}  pc={m(agg['pc']['clip'])}")
print(f"  DINO↑      : base={m(agg['base']['dino'])}  pc={m(agg['pc']['dino'])}")
