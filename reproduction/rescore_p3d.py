#!/usr/bin/env python3
"""Re-score existing P3D-Bench predictions with geometry + topology.

The full-split runs were inferred before GT geometry existed (synthesized cases
with empty targets). Now that `p3dbench prepare` has materialized GT meshes under
data/full, we patch each compiled prediction's case.target with the matching GT
(keyed by Text2CAD uid) and run P3D-Bench's geometry + topology buckets. No
re-inference and no re-compilation — we reuse the already-compiled model STLs.

Run from the P3D-Bench repo root:
  python /…/repro/rescore_p3d.py --models doubao deepseek gpt-5.4 --metric geometry,topology
"""
import sys, json, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))
from p3dbench import pipeline
from p3dbench.utils import read_jsonl, write_jsonl
from p3dbench.metrics.base import bucket_score_for_case, SCORE_BUCKETS, ScoreContext
from p3dbench.registry import get_metric_bucket
from p3dbench.data.loader import ResolvedCase, data_root
from p3dbench.data.schema import Case

WORKERS = 48


def gt_by_uid(manifest):
    m = {}
    for r in read_jsonl(manifest):
        uid = (r.get("metadata") or {}).get("source_id")
        if uid:
            m[uid] = r["target"]
    return m


def patch(compiled_path, gtmap, out):
    rows = list(read_jsonl(compiled_path))
    patched, n = [], 0
    for r in rows:
        uid = (r.get("case", {}).get("metadata") or {}).get("uid")
        tgt = gtmap.get(uid)
        if tgt:
            r["case"]["target"] = tgt
            r["split"] = "full"          # -> data_root = data/full where GT lives
            n += 1
        patched.append(r)
    write_jsonl(out, patched)
    return patched, n


def _score_one(row, buckets, work_root):
    """Replicate pipeline.score's per-case work for one row (thread-safe)."""
    rc = ResolvedCase(Case.from_dict(row["case"]), data_root(row["split"]))
    ctx = ScoreContext(
        task=row["task"], fmt=row["format"], case=rc,
        compiled=row.get("compile", {}), work_dir=Path(work_root) / row["id"].replace("/", "_"),
        judge_client=None, decompose_client=None,
        shared={"stage1_code": row.get("code"), "text_mode": row.get("text_mode", "parametric")},
    )
    raw = {}
    for b in buckets:
        try:
            raw.update(get_metric_bucket(b).score(ctx) or {})
        except Exception as e:
            raw[f"_{b}_error"] = f"{type(e).__name__}: {e}"
    return {"id": row["id"], "task": row["task"], "format": row["format"],
            "model": row["model"], "split": row["split"], "valid": bool(row.get("valid")),
            "raw_metrics": raw}


def score_buckets(compiled_gt, buckets, outdir, tag, model):
    rows_in = list(read_jsonl(compiled_gt))
    work_root = Path("results_p3d") / "rescore_work"
    out = [None] * len(rows_in)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_score_one, r, buckets, work_root): i for i, r in enumerate(rows_in)}
        done = 0
        for fut in futs:
            pass
        from concurrent.futures import as_completed
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
            done += 1
            if done % 100 == 0:
                print(f"     scored {done}/{len(rows_in)} [{tag}]", flush=True)
    rows = [r for r in out if r]
    write_jsonl(Path(outdir) / f"metrics_gt_{tag}.jsonl", rows)
    return rows


def aggregate(rows, only_ids=None):
    """Mean of each raw geometry/topology sub-metric over gradeable cases."""
    keys = ["chamfer", "chamfer_distance", "cd", "fscore_0.05", "fscore_0.01", "f_score",
            "normal_consistency", "iou", "topology", "topo"]
    acc = {}
    used = [r for r in rows if (only_ids is None or r["id"] in only_ids)]
    for r in used:
        rm = r.get("raw_metrics", {})
        for k, v in rm.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                acc.setdefault(k, []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items() if v}, len(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["doubao", "deepseek", "gpt-5.4"])
    ap.add_argument("--metric", default="geometry,topology")
    ap.add_argument("--source", default="full")
    ap.add_argument("--manifest", default="data/manifests/text_to_3d_full.jsonl")
    ap.add_argument("--outdir", default="results_p3d")
    args = ap.parse_args()

    buckets = [b.strip() for b in args.metric.split(",") if b.strip()]
    gtmap = gt_by_uid(args.manifest)
    print(f"GT available for {len(gtmap)} uids\n")

    for model in args.models:
        print(f"===== {model} =====")
        res = {}
        gradeable_ids = None
        for tag in ["baseline", "paircoder"]:
            cpath = Path(args.outdir) / f"compiled_{tag}_{model}_{args.source}.jsonl"
            if not cpath.exists():
                print(f"  [{tag}] missing {cpath}; skip"); continue
            cgt = Path(args.outdir) / f"compiled_gt_{tag}_{model}.jsonl"
            patched, n = patch(cpath, gtmap, cgt)
            rows = score_buckets(cgt, buckets, args.outdir, f"{tag}_{model}", model)
            res[tag] = rows
            # gradeable = cases valid in BOTH arms with GT (fair common set computed after)
        # fair common set: ids that are valid in both arms (so geometry is defined for both)
        if "baseline" in res and "paircoder" in res:
            valid_b = {r["id"] for r in res["baseline"] if r.get("valid")}
            valid_p = {r["id"] for r in res["paircoder"] if r.get("valid")}
            common = valid_b & valid_p
            for tag in ["baseline", "paircoder"]:
                allm, n_all = aggregate(res[tag])
                comm, n_c = aggregate(res[tag], common)
                vr = sum(1 for r in res[tag] if r.get("valid")) / max(len(res[tag]), 1)
                print(f"  [{tag}] valid_rate={vr:.3f}  n={len(res[tag])}")
                print(f"     all-valid means : { {k: round(v,4) for k,v in allm.items()} }")
                print(f"     common({len(common)}) means: { {k: round(v,4) for k,v in comm.items()} }")
        print()


if __name__ == "__main__":
    main()
