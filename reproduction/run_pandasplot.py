#!/usr/bin/env python3
"""PandasPlotBench (JetBrains-Research, n=175): pandas DataFrame + plot description/style ->
matplotlib code. Paper-faithful PairCoder: psi = code runs on the df (mpl_exec with data.csv in
cwd); SSIM-to-GT as evidence; two-image visual review. baseline = single direct generation.
Metrics: exec + SSIM/CLIP (art layout {tid}_gt/_base/_pc.png, score_chartmimic-compatible)."""
import os, re, sys, json, time, base64, argparse, threading, subprocess, tempfile, io
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from huggingface_hub import hf_hub_download
import pandas as pd

PYEXE = sys.executable
MPLX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpl_exec.py")
LOADER = 'import pandas as pd\ndf = pd.read_csv("data.csv")\n'


def render_code(code, data_csv, png, timeout=60):
    """Run df-loader + model plotting code in a temp cwd containing data.csv."""
    if not code:
        return False, "empty code"
    d = tempfile.mkdtemp()
    try:
        open(os.path.join(d, "data.csv"), "w").write(data_csv)
        cf = os.path.join(d, "c.py"); open(cf, "w").write(LOADER + code)
        png = os.path.abspath(png)
        p = subprocess.run([PYEXE, MPLX, cf, png], capture_output=True, text=True,
                           timeout=timeout, cwd=d)
        o = p.stdout + p.stderr
        ok = "MPL_OK" in o and os.path.exists(png)
        return ok, ("" if ok else o[-300:])
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)


def extract_code(t):
    if not t: return ""
    f = re.findall(r"```(?:python|py)?\s*\n(.*?)```", t, re.DOTALL)
    return (max(f, key=len).strip() if f else t.strip())


def to_arr_b(img_bytes):
    return np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((256, 256)))


def _ssim_png(png, ref_arr):
    try:
        return float(ssim(np.asarray(Image.open(png).convert("RGB").resize((256, 256))), ref_arr, channel_axis=2))
    except Exception:
        return -1.0


def load_tasks(n=None):
    p = hf_hub_download("JetBrains-Research/PandasPlotBench", "data/test-00000-of-00001.parquet", repo_type="dataset")
    df = pd.read_parquet(p)
    if n: df = df.head(n)
    tasks = []
    for _, r in df.iterrows():
        gt = r["plots_gt"][0] if len(r["plots_gt"]) else None
        if gt is None: continue
        tasks.append({"task_id": f"ppb_{r['id']}", "data_csv": r["data_csv"],
                      "desc": str(r["task__plot_description"]), "style": str(r["task__plot_style"]),
                      "gt_b64": gt})
    return tasks


def solve(task, model, max_iters, art):
    from paircoder import paper_solve, single_baseline
    tid = task["task_id"]
    gt_bytes = base64.b64decode(task["gt_b64"])
    ref_arr = to_arr_b(gt_bytes)
    head = "\n".join(task["data_csv"].splitlines()[:12])
    q = ("A pandas DataFrame `df` is ALREADY loaded (from data.csv; do not load it yourself). "
         "Write matplotlib code that uses `df` to draw the requested plot.\n"
         f"Data preview (first rows of data.csv):\n{head}\n\n{task['desc'][:1200]}\n{task['style'][:800]}\n"
         "Return ONLY one ```python``` code block (headless, produces the figure).")
    chk_png = os.path.join(art, f"{tid}_chk.png")

    def chk(text):
        c = extract_code(text)
        ok, err = render_code(c, task["data_csv"], chk_png)
        if not ok:
            return False, f"code failed: {err}", -1.0
        s_ = _ssim_png(chk_png, ref_arr)
        return True, f"runs OK; structural similarity to the target plot = {s_:.3f} (1.0 = identical)", s_

    def rc(text):
        c = extract_code(text)
        rp = os.path.join(art, f"{tid}_rc.png")
        ok, _ = render_code(c, task["data_csv"], rp)
        if not ok:
            return None
        return base64.b64encode(open(rp, "rb").read()).decode()
    gt64 = base64.b64encode(gt_bytes).decode()
    base = extract_code(single_baseline(q, model, image_b64=gt64))
    b_png = os.path.join(art, f"{tid}_base.png"); base_ok, _ = render_code(base, task["data_csv"], b_png)
    pc_raw, tel = paper_solve(q, model, max_iters=max_iters, check=chk, image_b64=gt64, render_current=rc)
    pc = extract_code(pc_raw)
    p_png = os.path.join(art, f"{tid}_pc.png"); pc_ok, _ = render_code(pc, task["data_csv"], p_png)
    Image.open(io.BytesIO(gt_bytes)).convert("RGB").save(os.path.join(art, f"{tid}_gt.png"))
    return {"task_id": tid, "base_ok": base_ok, "pc_ok": pc_ok,
            "iters": tel["iters"], "accepted": tel["accepted"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="all"); ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default="results_pandasplot")
    args = ap.parse_args()
    tasks = load_tasks(None if args.n == "all" else int(args.n))
    art = os.path.join(args.outdir, "art"); os.makedirs(art, exist_ok=True)
    out = os.path.join(args.outdir, f"results_pandasplot_{args.model}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["task_id"])
            except Exception: pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"=== PandasPlotBench | {args.model} | {len(tasks)} ({len(done)} cached) ===", flush=True)
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
