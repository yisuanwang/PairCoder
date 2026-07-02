#!/usr/bin/env python3
"""Capture OBJ+PNG for the paper-method 3DCodeBench rerun (base_code/pc_code in jsonl) and the
reference scripts; write a manifest compatible with tdcb_score_run.py."""
import os, json, subprocess, tempfile
BL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "blender-4.2.3-linux-x64", "blender")
CAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bl_capture.py")
import sys
SRC = sys.argv[1]
ART = os.path.join(os.path.dirname(SRC), "art"); os.makedirs(ART, exist_ok=True)
MAN = os.path.join(os.path.dirname(SRC), "manifest.jsonl")

def cap(code, obj, png, timeout=180):
    if not code: return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); cp = f.name
    try:
        p = subprocess.run([BL, "--background", "--python", CAP, "--", cp, obj, png],
                           capture_output=True, text=True, timeout=timeout)
        return "CAP_OK" in (p.stdout + p.stderr) and os.path.exists(obj)
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(cp)

rows = [json.loads(l) for l in open(SRC)]
mf = open(MAN, "w")
for i, r in enumerate(rows):
    tid = r["task_id"]
    out = {"task_id": tid, "base_ok": False, "pc_ok": False,
           "base_obj": None, "base_png": None, "pc_obj": None, "pc_png": None,
           "ref_obj": None, "ref_png": None}
    for arm in ("base", "pc"):
        code = r.get(f"{arm}_code") or ""
        obj, png = f"{ART}/{tid}_{arm}.obj", f"{ART}/{tid}_{arm}.png"
        if cap(code, obj, png):
            out[f"{arm}_ok"] = True; out[f"{arm}_obj"] = obj; out[f"{arm}_png"] = png
    # reuse references already captured by the old full pipeline
    ro, rp = f"results_3dcb_full/art/{tid}_ref.obj", f"results_3dcb_full/art/{tid}_ref.png"
    if os.path.exists(ro):
        out["ref_obj"] = ro
        out["ref_png"] = rp if os.path.exists(rp) else None
    mf.write(json.dumps(out) + "\n"); mf.flush()
    print(f"[{i+1}/{len(rows)}] {tid} base={out['base_ok']} pc={out['pc_ok']} ref={bool(out['ref_obj'])}", flush=True)
mf.close()
print("MANIFEST_DONE", MAN)
