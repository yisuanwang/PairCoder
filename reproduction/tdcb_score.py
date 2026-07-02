#!/usr/bin/env python3
"""3DCodeBench geometric+visual metrics (run with the CONDA python that has torch).
Chamfer (lower better), SigLIP-2 cosine (higher better), DINO cosine (higher better)
between a generated mesh/render and the reference.
"""
import numpy as np, trimesh
from scipy.spatial import cKDTree

_SIG = {}; _DINO = {}


def _load_siglip():
    if not _SIG:
        import torch
        from transformers import AutoModel, AutoProcessor
        mid = "google/siglip2-base-patch16-224"
        _SIG["m"] = AutoModel.from_pretrained(mid).to("cuda").eval()
        _SIG["p"] = AutoProcessor.from_pretrained(mid)
        _SIG["torch"] = torch
    return _SIG


def _load_dino():
    if not _DINO:
        import torch
        from transformers import AutoModel, AutoImageProcessor
        for mid in ["facebook/dinov3-vitb16-pretrain-lvd1689m", "facebook/dinov2-base"]:
            try:
                _DINO["m"] = AutoModel.from_pretrained(mid).to("cuda").eval()
                _DINO["p"] = AutoImageProcessor.from_pretrained(mid)
                _DINO["id"] = mid; _DINO["torch"] = torch; break
            except Exception:
                continue
    return _DINO


def chamfer(obj_a, obj_b, n=4000):
    """Symmetric Chamfer distance between two OBJ meshes (both normalized to unit size)."""
    try:
        a = trimesh.load(obj_a, force="mesh"); b = trimesh.load(obj_b, force="mesh")
        pa = a.sample(n); pb = b.sample(n)
    except Exception:
        return None
    da, _ = cKDTree(pb).query(pa); db, _ = cKDTree(pa).query(pb)
    return float(da.mean() + db.mean())


def _img_feat(loader, png):
    from PIL import Image
    M = loader(); torch = M["torch"]
    img = Image.open(png).convert("RGB")
    proc = M["p"](images=img, return_tensors="pt")
    proc = {k: v.to("cuda") for k, v in proc.items()}
    with torch.no_grad():
        m = M["m"]
        if hasattr(m, "get_image_features"):
            f = m.get_image_features(**proc)
        else:
            o = m(**proc)
            f = o.pooler_output if getattr(o, "pooler_output", None) is not None else o.last_hidden_state.mean(1)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().numpy().reshape(-1)


def siglip_sim(png_a, png_b):
    try:
        return float(_img_feat(_load_siglip, png_a) @ _img_feat(_load_siglip, png_b))
    except Exception:
        return None


def dino_sim(png_a, png_b):
    try:
        return float(_img_feat(_load_dino, png_a) @ _img_feat(_load_dino, png_b))
    except Exception:
        return None


if __name__ == "__main__":
    import sys
    base = "/tmp/tdcb"
    print("chamfer monkey-monkey2:", round(chamfer(f"{base}/monkey.obj", f"{base}/monkey2.obj"), 4))
    print("chamfer monkey-cube:   ", round(chamfer(f"{base}/monkey.obj", f"{base}/cube.obj"), 4))
    print("siglip monkey-monkey2: ", round(siglip_sim(f"{base}/monkey.png", f"{base}/monkey2.png"), 4))
    print("siglip monkey-cube:    ", round(siglip_sim(f"{base}/monkey.png", f"{base}/cube.png"), 4))
    print("dino monkey-monkey2:   ", round(dino_sim(f"{base}/monkey.png", f"{base}/monkey2.png"), 4))
    print("dino monkey-cube:      ", round(dino_sim(f"{base}/monkey.png", f"{base}/cube.png"), 4))
    print("dino model:", _DINO.get("id"))
