#!/usr/bin/env python3
"""Materialize P3D-Bench text-to-3d GT meshes WITHOUT p3dbench prepare.

prepare couples GT-mesh generation with an OCC preview render that crashes
(stack smashing) on this box, marking every case incomplete. We only need GT
geometry (STEP+STL) for the geometry/topology metrics, so we compile each GT
minimal-JSON directly with the same shared interpreter prepare uses, and write a
data/full manifest. Run from the P3D-Bench repo root.
"""
import sys, json, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))
from huggingface_hub import hf_hub_download
from p3dbench.compile.exporter import compile_code

FULL = Path("data/full")
SRC = Path("srcroot/text2cad/minimal_json")


def main():
    up = hf_hub_download("SpatiaOS/P3D-Bench", "data/text_to_3d/uids.jsonl", repo_type="dataset")
    ap = hf_hub_download("SpatiaOS/P3D-Bench", "data/text_to_3d/annotations.jsonl", repo_type="dataset")
    uids = [json.loads(l)["uid"] for l in open(up)]
    anns = {json.loads(l)["uid"]: json.loads(l) for l in open(ap)}

    (FULL / "targets/mesh").mkdir(parents=True, exist_ok=True)
    (FULL / "targets/step").mkdir(parents=True, exist_ok=True)
    (FULL / "targets/minimal-json").mkdir(parents=True, exist_ok=True)

    def build(i_uid):
        i, uid = i_uid
        b, f = uid.split("/")
        mj = SRC / b / f / "minimal_json" / f"{f}.json"
        if not mj.exists():
            return ("missing_src", uid, None)
        cid = f"p3d_text-to-3d_{i:06d}"
        code = mj.read_text(encoding="utf-8")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            try:
                cr = compile_code(code, "minimal-json", Path(td))
            except Exception as e:
                return ("compile_exc", uid, None)
            if not (cr.valid and cr.stl):
                return ("gt_invalid", uid, None)
            step_rel = f"targets/step/{cid}.step"
            mesh_rel = f"targets/mesh/{cid}.stl"
            code_rel = f"targets/minimal-json/{cid}.json"
            shutil.copy(cr.stl, FULL / mesh_rel)
            if cr.step:
                shutil.copy(cr.step, FULL / step_rel)
            (FULL / code_rel).write_text(code, encoding="utf-8")
            a = anns.get(uid, {})
            row = {
                "id": cid, "task": "text-to-3d", "split": "full",
                "input": {"text": a.get("text_param", ""), "image_paths": [], "part_annotations": []},
                "target": {"format": "minimal-json", "code_path": code_rel,
                           "step_path": (step_rel if cr.step else None), "mesh_path": mesh_rel,
                           "render_paths": [], "part_paths": [], "qa_bank_path": None},
                "metadata": {"source": "text2cad-v1.1", "source_id": uid,
                             "summary": a.get("summary"), "text_desc": a.get("text_desc")},
            }
            return ("ok", uid, row)

    rows, stats = [], {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for k, (status, uid, row) in enumerate(ex.map(build, list(enumerate(uids)))):
            stats[status] = stats.get(status, 0) + 1
            if row:
                rows.append(row)
            if (k + 1) % 50 == 0:
                print(f"  [{k+1}/{len(uids)}] {stats}", flush=True)

    rows.sort(key=lambda r: r["id"])
    mdir = Path("data/manifests"); mdir.mkdir(parents=True, exist_ok=True)
    out = mdir / "text_to_3d_full.jsonl"
    with open(out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nGT built: {stats}")
    print(f"manifest: {out} ({len(rows)} rows)")
    print(f"meshes: {len(list((FULL/'targets/mesh').glob('*.stl')))}")


if __name__ == "__main__":
    main()
