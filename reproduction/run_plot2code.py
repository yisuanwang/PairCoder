#!/usr/bin/env python3
"""Plot2Code (TencentARC, matplotlib test, n=132): chart image + instruction -> matplotlib code.
Paper-faithful PairCoder (pc_paper loop): psi = code runs headless (mpl_exec); SSIM-to-target as
evidence; two-image visual review. baseline = single direct generation. Metrics: exec + SSIM/CLIP
(score_chartmimic.py-compatible art layout: {tid}_gt/_base/_pc.png)."""
import os, re, sys, json, time, base64, argparse, threading, subprocess, tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from huggingface_hub import hf_hub_download

PYEXE = sys.executable
MPLX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpl_exec.py")


def render_code(code, png, timeout=60):
    if not code or ("matplotlib" not in code and "plt" not in code):
        return False
    d = tempfile.mkdtemp()
    cf = os.path.join(d, "c.py"); open(cf, "w").write(code)
    try:
        p = subprocess.run([PYEXE, MPLX, cf, png], capture_output=True, text=True, timeout=timeout)
        return "MPL_OK" in (p.stdout + p.stderr) and os.path.exists(png)
    except subprocess.TimeoutExpired:
        return False
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)


def extract_code(t):
    if not t: return ""
    f = re.findall(r"```(?:python|py)?\s*\n(.*?)```", t, re.DOTALL)
    return (max(f, key=len).strip() if f else t.strip())


def to_arr(p):
    return np.asarray(Image.open(p).convert("RGB").resize((256, 256)))


def _ssim_png(png, ref_arr):
    try:
        return float(ssim(to_arr(png), ref_arr, channel_axis=2))
    except Exception:
        return -1.0


def load_tasks(n=None):
    meta = hf_hub_download("TencentARC/Plot2Code", "data/python_matplotlib/test/metadata.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(meta)]
    if n: rows = rows[:n]
    tasks = []
    for i, r in enumerate(rows):
        img = hf_hub_download("TencentARC/Plot2Code", f"data/python_matplotlib/test/{r['file_name']}", repo_type="dataset")
        tasks.append({"task_id": f"p2c_{i}", "instruction": r["instruction"], "gt_png": img})
    return tasks


def solve(task, model, max_iters, art):
    from paircoder import paper_solve, single_baseline
    tid = task["task_id"]
    ref_arr = to_arr(task["gt_png"])
    b64 = base64.b64encode(open(task["gt_png"], "rb").read()).decode()
    q = ("Reproduce this chart as closely as possible with matplotlib code (the image is the "
         "target). Description of the figure:\n" + task["instruction"][:1500] +
         "\nReturn ONLY one ```python``` code block (runs headless, produces the figure).")
    chk_png = os.path.join(art, f"{tid}_chk.png")

    def chk(text):
        c = extract_code(text)
        if not render_code(c, chk_png):
            return False, "matplotlib code failed to run headless", -1.0
        s_ = _ssim_png(chk_png, ref_arr)
        return True, f"runs OK; structural similarity to the target chart = {s_:.3f} (1.0 = identical)", s_

    def rc(text):
        c = extract_code(text)
        rp = os.path.join(art, f"{tid}_rc.png")
        if not render_code(c, rp):
            return None
        return base64.b64encode(open(rp, "rb").read()).decode()
    base = extract_code(single_baseline(q, model, image_b64=b64))
    b_png = os.path.join(art, f"{tid}_base.png"); base_ok = render_code(base, b_png)
    pc_raw, tel = paper_solve(q, model, max_iters=max_iters, check=chk, image_b64=b64, render_current=rc)
    pc = extract_code(pc_raw)
    p_png = os.path.join(art, f"{tid}_pc.png"); pc_ok = render_code(pc, p_png)
    Image.open(task["gt_png"]).convert("RGB").save(os.path.join(art, f"{tid}_gt.png"))
    return {"task_id": tid, "base_ok": base_ok, "pc_ok": pc_ok,
            "iters": tel["iters"], "accepted": tel["accepted"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="all"); ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default="results_plot2code")
    args = ap.parse_args()
    tasks = load_tasks(None if args.n == "all" else int(args.n))
    art = os.path.join(args.outdir, "art"); os.makedirs(art, exist_ok=True)
    out = os.path.join(args.outdir, f"results_plot2code_{args.model}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["task_id"])
            except Exception: pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"=== Plot2Code (mpl) | {args.model} | {len(tasks)} ({len(done)} cached) ===", flush=True)
    fh = open(out, "a"); lk = threading.Lock(); n = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(solve, t, args.model, args.max_iters, art) for t in todo]):
            try: r = f.result()
            except Exception: r = None
            if r is None: continue
            with lk: fh.write(json.dumps(r) + "\n"); fh.flush()
            n += 1
            if n % 10 == 0 or n == len(todo): print(f"  [{n}/{len(todo)}]", flush=True)
    fh.close()
    from paircoder import client as pc_client
    print(f"  -> {out} ({time.time()-t0:.0f}s) TOKENS={pc_client.TOKENS}", flush=True)


if __name__ == "__main__":
    main()
