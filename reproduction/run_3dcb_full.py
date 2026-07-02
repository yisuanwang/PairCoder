#!/usr/bin/env python3
"""3DCodeBench FULL metrics: generate baseline + PairCoder Blender code, capture OBJ+PNG for each
and for the reference, save a manifest. Geometric/visual scoring (Chamfer/SigLIP/DINO) is done
separately by tdcb_score_run.py (conda python w/ torch). Executability is recorded here."""
import os, sys, json, time, argparse, threading, subprocess, tempfile, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
from blender_common import load_3dcb, build_prompt, extract_code

BL = os.path.join(os.path.dirname(__file__), "tools", "blender-4.2.3-linux-x64", "blender")
CAP = os.path.join(os.path.dirname(__file__), "bl_capture.py")
_lock = threading.Lock(); USAGE = {"calls": 0}
DRIVER_SYS = ("You are the Driver, an expert Blender/bpy technical artist. Write a Blender 4.2 "
              "Python script that procedurally builds the described object as real mesh geometry. "
              "Return ONLY one ```python``` block. No explanation.")
ANGLES = ["Build it directly with bpy primitives and mesh operations.",
          "Decompose the object into parts; create each as mesh geometry.",
          "Use bpy.ops mesh primitives + transforms/modifiers; ensure real vertices.",
          "Re-read the description and model each named component.",
          "Keep bpy API correct for Blender 4.2 so it runs headless without errors."]


def llm(model, system, user, retries=4):
    from paircoder.client import make_client, guarded_create
    client = make_client()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for a in range(retries):
        try:
            r = guarded_create(client, model=model, messages=msgs, extra_body={"reasoning_effort": "none"})
            if r.usage:
                with _lock: USAGE["calls"] += 1
            return r.choices[0].message.content or ""
        except Exception as e:
            if a == retries - 1:
                return ""
            time.sleep(3 * (a + 1))


def capture(code, obj_path, png_path, timeout=150):
    """Run code in Blender, export OBJ + render PNG. Return True if geometry produced."""
    if not code or "bpy" not in code:
        return False
    d = tempfile.mkdtemp(); cf = os.path.join(d, "c.py"); open(cf, "w").write(code)
    try:
        p = subprocess.run([BL, "--background", "--python", CAP, "--", cf, obj_path, png_path],
                           capture_output=True, text=True, timeout=timeout)
        ok = "CAP_OK" in (p.stdout + p.stderr)
        return ok and os.path.exists(obj_path) and os.path.exists(png_path)
    except subprocess.TimeoutExpired:
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def blender_err(code, timeout=120):
    from blender_common import run_in_blender
    return run_in_blender(code, timeout)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--n", default="24")
    ap.add_argument("--cands", type=int, default=5)
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--outdir", default="results_3dcb_full")
    args = ap.parse_args()
    tasks = load_3dcb(None if args.n == "all" else int(args.n))
    art = os.path.join(args.outdir, "art"); os.makedirs(art, exist_ok=True)
    manifest = os.path.join(args.outdir, f"manifest_{args.model}.jsonl")
    done = set()
    if os.path.exists(manifest):
        for l in open(manifest):
            try: done.add(json.loads(l)["task_id"])
            except Exception: pass
    todo = [t for t in tasks if t["task_id"] not in done]
    print(f"=== 3DCB-full | {args.model} | {len(tasks)} ({len(done)} cached, {len(todo)} to do) ===", flush=True)
    fh = open(manifest, "a"); lk = threading.Lock(); t0 = time.time(); n = 0

    def solve(task):
        tid = task["task_id"]; base_user = build_prompt(task)

        def gen(i):
            return extract_code(llm(args.model, DRIVER_SYS, f"{base_user}\n\nApproach: {ANGLES[i%len(ANGLES)]}"))
        with ThreadPoolExecutor(max_workers=args.cands) as ex:
            cands = [c for c in ex.map(gen, range(args.cands)) if c] or [""]
        # baseline = cand0
        b_obj = os.path.join(art, f"{tid}_base.obj"); b_png = os.path.join(art, f"{tid}_base.png")
        base_ok = capture(cands[0], b_obj, b_png)
        # PairCoder: first capturing candidate, else repair cand0 on the blender error
        pc_obj = os.path.join(art, f"{tid}_pc.obj"); pc_png = os.path.join(art, f"{tid}_pc.png")
        pc_ok = False; chosen = cands[0]
        for c in cands:
            if c == cands[0] and base_ok:
                shutil.copy(b_obj, pc_obj); shutil.copy(b_png, pc_png); pc_ok = True; chosen = c; break
            if capture(c, pc_obj, pc_png):
                pc_ok = True; chosen = c; break
        if not pc_ok:
            cand = cands[0]
            for _ in range(args.max_iters):
                err = blender_err(cand)
                fix = (f"{base_user}\n\nYour current script:\n```python\n{cand[:4000]}\n```\n\n"
                       f"It failed in Blender 4.2 with:\n{err}\n\nReturn a corrected complete "
                       "```python``` Blender script that runs headless and creates mesh geometry.")
                new = extract_code(llm(args.model, DRIVER_SYS, fix))
                if new and capture(new, pc_obj, pc_png):
                    pc_ok = True; break
                cand = new or cand
        # reference
        r_obj = os.path.join(art, f"{tid}_ref.obj"); r_png = os.path.join(art, f"{tid}_ref.png")
        if not os.path.exists(r_obj):
            try:
                refcode = open(hf_hub_download("YipengGao/3DCode", f"3DCodeBench/{tid}/{tid}.py", repo_type="dataset")).read()
                capture(refcode, r_obj, r_png, timeout=200)
            except Exception:
                pass
        return {"task_id": tid, "base_ok": base_ok, "pc_ok": pc_ok,
                "base_obj": b_obj if base_ok else None, "base_png": b_png if base_ok else None,
                "pc_obj": pc_obj if pc_ok else None, "pc_png": pc_png if pc_ok else None,
                "ref_obj": r_obj if os.path.exists(r_obj) else None,
                "ref_png": r_png if os.path.exists(r_png) else None}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(solve, t) for t in todo]):
            try: r = f.result()
            except Exception: r = None
            if r is None: continue
            with lk:
                fh.write(json.dumps(r) + "\n"); fh.flush()
            n += 1; print(f"  [{n}/{len(todo)}] {r['task_id']} base={r['base_ok']} pc={r['pc_ok']}", flush=True)
    fh.close()
    print(f"  -> {manifest} ({time.time()-t0:.0f}s, {USAGE['calls']} calls)", flush=True)


if __name__ == "__main__":
    main()
