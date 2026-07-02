#!/usr/bin/env python3
"""3DCodeBench loader + Blender executor (procedural 3D via Blender Python).

Task: a natural-language description -> generate a Blender Python script that builds the object.
Oracle (executable, no vision models): run the script headless in portable Blender; SUCCESS =
it runs without an uncaught exception AND produces >=1 mesh with vertices (this is 3DCodeBench's
primary 'Blender pass rate' metric). PairCoder repairs runtime errors in generated Blender code.
"""
import os, re, json, subprocess, tempfile, shutil
from huggingface_hub import hf_hub_download, HfApi

BLENDER = os.path.join(os.path.dirname(__file__), "tools", "blender-4.2.3-linux-x64", "blender")
REPO = "YipengGao/3DCode"
_TASKLIST = None


def _list_tasks():
    global _TASKLIST
    if _TASKLIST is None:
        files = [s.rfilename for s in HfApi().dataset_info(REPO).siblings]
        names = sorted({f.split("/")[1] for f in files
                        if f.startswith("3DCodeBench/") and len(f.split("/")) == 3})
        _TASKLIST = names
    return _TASKLIST


def load_3dcb(n=None):
    names = _list_tasks()
    if n is not None:
        names = names[:n]
    tasks = []
    for nm in names:
        try:
            desc = open(hf_hub_download(REPO, f"3DCodeBench/{nm}/prompt_description.txt", repo_type="dataset")).read()
        except Exception:
            continue
        tasks.append({"task_id": nm, "desc": desc})
    return tasks


def build_prompt(task):
    return (f"Write a Blender Python script (using the `bpy` API, Blender 4.2) that procedurally "
            f"builds the following 3D object as mesh geometry in the scene:\n\n{task['desc']}\n\n"
            "The script must run headless (`blender --background --python script.py`) and create "
            "one or more MESH objects with real geometry. Return ONLY one ```python``` code block.")


def extract_code(text):
    if not text:
        return ""
    f = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if f:
        return max(f, key=len).strip()
    m = re.search(r"(^|\n)(import bpy|import )", text)
    return text[m.start():].strip() if m else text.strip()


_WRAP = r"""
import bpy
for o in list(bpy.data.objects):
    try: bpy.data.objects.remove(o, do_unlink=True)
    except Exception: pass
_SRC = open({SRCPATH!r}).read()
try:
    exec(compile(_SRC, 'gen', 'exec'), {'__name__': '__main__', 'bpy': bpy})
    _tot = sum(len(o.data.vertices) for o in bpy.data.objects if o.type == 'MESH' and o.data)
    print('RESULT_OK', _tot)
except Exception as e:
    print('RESULT_ERR', repr(e)[:400])
"""


def run_in_blender(code, timeout=150):
    """Return (success, nverts, error). success = ran without exception AND nverts>0."""
    if not code or "bpy" not in code:
        return False, 0, "no bpy"
    d = tempfile.mkdtemp()
    try:
        src = os.path.join(d, "gen.py"); open(src, "w").write(code)
        wrap = os.path.join(d, "wrap.py"); open(wrap, "w").write(_WRAP.replace("{SRCPATH!r}", repr(src)))
        try:
            p = subprocess.run([BLENDER, "--background", "--python", wrap],
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, 0, "timeout"
        out = p.stdout + p.stderr
        m = re.search(r"RESULT_OK (\d+)", out)
        if m:
            nv = int(m.group(1))
            return nv > 0, nv, ""
        me = re.search(r"RESULT_ERR (.*)", out)
        return False, 0, (me.group(1)[:200] if me else "no result / crashed")
    finally:
        shutil.rmtree(d, ignore_errors=True)
