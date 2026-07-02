#!/usr/bin/env python3
"""StarVector image->SVG benchmark (standard SVG code-gen). The model SEES a rendered target
image (multimodal) and must reproduce it as SVG code. baseline = candidate[0]; PairCoder picks
the candidate whose render has the highest SSIM to the GIVEN input image (legitimate: the target
image is the input, not a hidden oracle), repairing non-rendering candidates. Saves baseline/pc
SVGs + their renders + the reference render for CLIP/DINO/SSIM scoring (score_svgbench.py)."""
import os, re, io, sys, json, time, base64, argparse, threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import cairosvg
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from huggingface_hub import hf_hub_download
import pandas as pd

_lock = threading.Lock(); USAGE = {"calls": 0}
SIZE = 224
SYS = ("You are the Driver, an expert at SVG vector graphics. Reproduce the given image as a "
       "single self-contained <svg>...</svg>. Return ONLY one ```svg``` code block.")
ANGLES = ["Reproduce it as faithfully as possible.",
          "Match shapes, colors, positions and proportions to the image.",
          "Use clean paths/primitives; keep the same layout and palette.",
          "Pay attention to fine details and overall silhouette.",
          "Capture the dominant shapes and colors accurately."]


def render(svg, size=SIZE):
    if not svg or "<svg" not in svg:
        return None
    try:
        png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size, background_color="white")
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB").resize((size, size)))
    except Exception:
        return None


def extract_svg(text):
    if not text: return ""
    m = re.search(r"<svg\b.*?</svg>", text, re.DOTALL | re.IGNORECASE)
    return m.group(0) if m else ""


def _ssim(a, b):
    return float(ssim(a, b, channel_axis=2)) if (a is not None and b is not None) else -1.0


def llm_mm(model, sys_msg, text, img_b64, retries=4):
    from paircoder.client import make_client, guarded_create
    client = make_client()
    content = [{"type": "text", "text": text},
               {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": content}]
    for a in range(retries):
        try:
            r = guarded_create(client, model=model, messages=msgs, extra_body={"reasoning_effort": os.environ.get("PAIRCODER_EFFORT","none")})
            if r.usage:
                with _lock: USAGE["calls"] += 1
            return r.choices[0].message.content or ""
        except Exception as e:
            if a == retries - 1: return ""
            time.sleep(3 * (a + 1))


def solve(task, model, n_cand, max_iters, art):
    """Paper-faithful PairCoder: baseline = single direct generation; PairCoder = Driver/Navigator
    loop. psi = renders; SSIM-to-target reported as evidence (the target image IS the input)."""
    from paircoder import paper_solve, single_baseline
    ref_img = render(task["ref_svg"])
    if ref_img is None:
        return None
    buf = io.BytesIO(); Image.fromarray(ref_img).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    q = ("Reproduce this exact image as a single self-contained <svg>...</svg>. "
         "Return ONLY one ```svg``` code block.")

    def chk(text):
        c = extract_svg(text)
        im = render(c)
        if im is None:
            return False, "SVG does not render (invalid markup)", -1.0
        s_ = _ssim(im, ref_img)
        return True, f"renders OK; structural similarity to the target image = {s_:.3f} (1.0 = identical)", s_
    def rc(text):
        im = render(extract_svg(text))
        if im is None:
            return None
        bb = io.BytesIO(); Image.fromarray(im).save(bb, format="PNG")
        return base64.b64encode(bb.getvalue()).decode()
    baseline = extract_svg(single_baseline(q, model, image_b64=b64))
    pc_raw, tel = paper_solve(q, model, max_iters=max_iters, check=chk, image_b64=b64, render_current=rc)
    pc = extract_svg(pc_raw)
    tid = task["task_id"]
    Image.fromarray(ref_img).save(os.path.join(art, f"{tid}_ref.png"))
    for tag, svg in [("base", baseline), ("pc", pc)]:
        im = render(svg)
        if im is not None:
            Image.fromarray(im).save(os.path.join(art, f"{tid}_{tag}.png"))
    return {"task_id": tid, "base_ssim": max([_ssim(render(baseline), ref_img), -1.0]),
            "pc_ssim": max([_ssim(render(pc), ref_img), -1.0]),
            "base_render": render(baseline) is not None, "pc_render": render(pc) is not None,
            "iters": tel["iters"], "accepted": tel["accepted"]}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="60")
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--repo", default="starvector/svg-icons")
    ap.add_argument("--outdir", default="results_svgbench")
    args = ap.parse_args()
    df = pd.read_parquet(hf_hub_download(args.repo, "data/test-00000-of-00001.parquet", repo_type="dataset"))
    df = df.head(int(args.n)) if args.n != "all" else df
    tasks = [{"task_id": r["Filename"].replace(".svg", ""), "ref_svg": r["Svg"]} for _, r in df.iterrows()]
    art = os.path.join(args.outdir, "art"); os.makedirs(art, exist_ok=True)
    out = os.path.join(args.outdir, f"results_svgbench_{args.model}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["task_id"])
            except Exception: pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"=== StarVector image->SVG | {args.model} | {len(tasks)} ({len(done)} cached) ===", flush=True)
    fh = open(out, "a"); lk = threading.Lock(); n = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(solve, t, args.model, args.cands, args.max_iters, art) for t in todo]):
            try: r = f.result()
            except Exception: r = None
            if r:
                if r is None: continue
            with lk: fh.write(json.dumps(r) + "\n"); fh.flush()
            n += 1
            if n % 10 == 0 or n == len(todo): print(f"  [{n}/{len(todo)}]", flush=True)
    fh.close()
    print(f"  -> {out} ({time.time()-t0:.0f}s, {USAGE['calls']} calls)", flush=True)


if __name__ == "__main__":
    main()
