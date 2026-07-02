#!/usr/bin/env python3
"""GenCAD-Code image->CadQuery benchmark (standard CAD code-gen). Multimodal: model sees the CAD
image and writes CadQuery code. baseline = candidate[0]; PairCoder keeps a candidate that EXECUTES
to a valid solid, else repairs candidate[0] on the CadQuery error. Saves baseline/pc/ref STLs for
Chamfer scoring. Metrics: execution rate (here) + Chamfer (score_cadbench.py)."""
import os, re, io, sys, json, time, base64, argparse, threading, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
import pandas as pd

_lock = threading.Lock(); USAGE = {"calls": 0}
PYEXE = sys.executable
CADX = os.path.join(os.path.dirname(__file__), "cad_exec.py")
SYS = ("You are the Driver, an expert in parametric CAD with the CadQuery Python library. Given a "
       "rendered image of a 3D part, write CadQuery code that reconstructs it. Assign the final "
       "solid to a variable named `result`. Return ONLY one ```python``` code block.")
ANGLES = ["Reconstruct the part directly with CadQuery primitives and operations.",
          "Identify the base shape and features (holes, fillets, extrusions) from the image.",
          "Match overall proportions and the main geometric features.",
          "Use Workplane operations; ensure the code executes to a valid solid named result.",
          "Keep CadQuery API usage correct so the code runs without errors."]


def llm_mm(model, text, img_b64, retries=4):
    from paircoder.client import make_client, guarded_create
    client = make_client()
    content = [{"type": "text", "text": text},
               {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": content}]
    for a in range(retries):
        try:
            r = guarded_create(client, model=model, messages=msgs, extra_body={"reasoning_effort": os.environ.get("PAIRCODER_EFFORT","none")})
            if r.usage:
                with _lock: USAGE["calls"] += 1
            return r.choices[0].message.content or ""
        except Exception as e:
            if a == retries - 1: return ""
            time.sleep(3 * (a + 1))


def extract_code(text):
    if not text: return ""
    f = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return (max(f, key=len).strip() if f else text.strip())


def cad_exec(code, stl, timeout=60):
    """Run CadQuery code -> STL. Return (ok, error)."""
    if not code or "cadquery" not in code and "cq." not in code:
        return False, "no cadquery"
    d = tempfile.mkdtemp(); cf = os.path.join(d, "c.py"); open(cf, "w").write(code)
    try:
        p = subprocess.run([PYEXE, CADX, cf, stl], capture_output=True, text=True, timeout=timeout)
        out = p.stdout + p.stderr
        if "CAD_OK" in out:
            return True, ""
        m = re.search(r"CAD_ERR (.*)", out)
        return False, (m.group(1)[:200] if m else "crash")
    except subprocess.TimeoutExpired:
        return False, "timeout"


def solve(task, model, n_cand, max_iters, art):
    """Paper-faithful PairCoder: baseline = single direct generation; PairCoder = Driver/Navigator
    loop with CadQuery execution (valid solid) as psi evidence."""
    from paircoder import paper_solve, single_baseline
    tid = task["task_id"].replace("/", "_")
    b64 = base64.b64encode(task["image"]).decode()
    q = ("Reconstruct this 3D part as CadQuery code (assign the solid to `result`). "
         "Return ONLY one ```python``` code block.")
    pc_stl = os.path.join(art, f"{tid}_pc.stl")

    def chk(text):
        c = extract_code(text)
        ok, err = cad_exec(c, pc_stl)
        return (True, "executes to a valid solid") if ok else (False, "CadQuery error: " + str(err)[:300])
    base = extract_code(single_baseline(q, model, image_b64=b64))
    b_stl = os.path.join(art, f"{tid}_base.stl"); base_ok, _ = cad_exec(base, b_stl)
    pc_raw, tel = paper_solve(q, model, max_iters=max_iters, check=chk, image_b64=b64)
    pc = extract_code(pc_raw)
    pc_ok, _ = cad_exec(pc, pc_stl)
    r_stl = os.path.join(art, f"{tid}_ref.stl")
    if not os.path.exists(r_stl):
        cad_exec(task["ref_cadquery"], r_stl, timeout=90)
    return {"task_id": tid, "base_ok": base_ok, "pc_ok": pc_ok,
            "ref_ok": os.path.exists(r_stl), "iters": tel["iters"], "accepted": tel["accepted"]}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="60")
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default="results_cadbench")
    args = ap.parse_args()
    df = pd.read_parquet(hf_hub_download("CADCODER/GenCAD-Code", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    df = df.head(int(args.n)) if args.n != "all" else df
    tasks = [{"task_id": r["deepcad_id"], "image": r["image"]["bytes"], "ref_cadquery": r["cadquery"]} for _, r in df.iterrows()]
    art = os.path.join(args.outdir, "art"); os.makedirs(art, exist_ok=True)
    out = os.path.join(args.outdir, f"results_cadbench_{args.model}.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["task_id"])
            except Exception: pass
    todo = [t for t in tasks if t["task_id"].replace("/", "_") not in done]
    print(f"=== GenCAD image->CadQuery | {args.model} | {len(tasks)} ({len(done)} cached) ===", flush=True)
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
