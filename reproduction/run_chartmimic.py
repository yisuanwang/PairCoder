#!/usr/bin/env python3
"""ChartMimic (image->matplotlib code). Multimodal: model sees a target chart image + instruction,
writes matplotlib code. baseline = candidate[0]; PairCoder picks the candidate whose rendered chart
has highest SSIM to the GIVEN target image, repairing execution errors. Saves baseline/pc/target
PNGs for CLIP/SSIM scoring (score_chartmimic.py)."""
import os, re, io, sys, json, time, base64, argparse, threading, subprocess, tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from huggingface_hub import hf_hub_download
import pandas as pd

_lock = threading.Lock(); USAGE = {"calls": 0}
PYEXE = sys.executable
MPLX = os.path.join(os.path.dirname(__file__), "mpl_exec.py")
SYS = ("You are an expert Python developer specializing in matplotlib. Reproduce the given chart "
       "image as closely as possible with matplotlib code. Return ONLY one ```python``` code block "
       "(it must run headless and produce the figure).")
ANGLES = ["Reproduce the chart faithfully.", "Match chart type, data, colors, labels, layout.",
          "Pay attention to axes, legends, ticks, and annotations.",
          "Reproduce subplot structure and styling precisely.",
          "Capture the overall appearance and all visible elements."]


def to_img(b):
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((256, 256)))


def render_code(code, png, timeout=60):
    if not code or "matplotlib" not in code and "plt" not in code:
        return False
    d = tempfile.mkdtemp(); cf = os.path.join(d, "c.py"); open(cf, "w").write(code)
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


def _ssim_png(png, ref_arr):
    try:
        a = np.asarray(Image.open(png).convert("RGB").resize((256, 256)))
        return float(ssim(a, ref_arr, channel_axis=2))
    except Exception:
        return -1.0


def llm_mm(model, text, img_b64, retries=4):
    from paircoder.client import make_client, guarded_create
    client = make_client()
    content = [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": content}]
    for a in range(retries):
        try:
            r = guarded_create(client, model=model, messages=msgs, extra_body={"reasoning_effort": os.environ.get("PAIRCODER_EFFORT","none")})
            if r.usage:
                with _lock: USAGE["calls"] += 1
            return r.choices[0].message.content or ""
        except Exception:
            if a == retries - 1: return ""
            time.sleep(3 * (a + 1))


def solve(task, model, n_cand, max_iters, art):
    """Paper-faithful PairCoder: baseline = single direct generation; PairCoder = Driver/Navigator
    loop. psi = matplotlib executes; SSIM-to-target as evidence (target image IS the input)."""
    from paircoder import paper_solve, single_baseline
    tid = task["task_id"]; b64 = base64.b64encode(task["img"]).decode()
    ref_arr = to_img(task["gt"])
    q = (task["instruction"][:1500] + "\nReturn ONLY one ```python``` code block "
         "(matplotlib, runs headless, produces the figure).")
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
        import base64 as _b64
        return _b64.b64encode(open(rp, "rb").read()).decode()
    base = extract_code(single_baseline(q, model, image_b64=b64))
    b_png = os.path.join(art, f"{tid}_base.png"); base_ok = render_code(base, b_png)
    pc_raw, tel = paper_solve(q, model, max_iters=max_iters, check=chk, image_b64=b64, render_current=rc)
    pc = extract_code(pc_raw)
    p_png = os.path.join(art, f"{tid}_pc.png"); pc_ok = render_code(pc, p_png)
    Image.fromarray(ref_arr).save(os.path.join(art, f"{tid}_gt.png"))
    return {"task_id": tid, "base_ok": base_ok, "pc_ok": pc_ok,
            "base_ssim": _ssim_png(b_png, ref_arr) if base_ok else -1.0,
            "iters": tel["iters"], "accepted": tel["accepted"]}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="60"); ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="results_chartmimic")
    args = ap.parse_args()
    shards = [hf_hub_download("ChartMimic/ChartMimic", f"preview/test-0000{i}-of-00010.parquet", repo_type="dataset") for i in range(2)]
    df = pd.concat([pd.read_parquet(s) for s in shards]).reset_index(drop=True)
    df = df.head(int(args.n)) if args.n != "all" else df
    tasks = [{"task_id": f"{r['Task']}_{r['ExampleID']}".replace("/", "_"), "instruction": r["Instruction"],
              "img": r["InputFigurePreview"]["bytes"], "gt": r["GroundTruthFigurePreview"]["bytes"]} for _, r in df.iterrows()]
    art = os.path.join(args.outdir, "art"); os.makedirs(art, exist_ok=True)
    out = os.path.join(args.outdir, f"results_chartmimic_{args.model}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["task_id"])
            except Exception: pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"=== ChartMimic image->matplotlib | {args.model} | {len(tasks)} ({len(done)} cached) ===", flush=True)
    fh = open(out, "a"); lk = threading.Lock(); n = 0; t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(solve, t, args.model, args.cands, args.max_iters, art) for t in todo]):
            try: r = f.result()
            except Exception: r = None
            if r is None: continue
            with lk: fh.write(json.dumps(r) + "\n"); fh.flush()
            n += 1
            if n % 10 == 0 or n == len(todo): print(f"  [{n}/{len(todo)}]", flush=True)
    fh.close()
    print(f"  -> {out} ({time.time()-t0:.0f}s, {USAGE['calls']} calls)", flush=True)


if __name__ == "__main__":
    main()
