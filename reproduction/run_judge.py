#!/usr/bin/env python3
"""Score the P3D-Bench text-to-3D `judge` bucket (QA-S / QA-P) on existing predictions.

Patches each compiled prediction's case with the manifest target (which now carries
qa_bank_path + GT), then runs P3D-Bench's official judge: render the predicted mesh
(pyrender/EGL on GPU) and have the configured vision VLM answer the shipped 12-question
MCQ bank. Aggregates QA-S (semantic) and QA-P (parametric) accuracy.

Run from the P3D-Bench repo root with ITOO_KEY set (the judge VLM).
"""
import os, sys, json, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

REPRO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPRO))
from p3dbench import pipeline
from p3dbench.utils import read_jsonl, write_jsonl
from p3dbench.metrics.base import ScoreContext
from p3dbench.registry import get_metric_bucket
from p3dbench.data.loader import ResolvedCase, data_root
from p3dbench.data.schema import Case
from p3dbench.models import get_client
from p3dbench.config import load_judge_config

WORKERS = 4


def gt_by_uid(manifest):
    return {(json.loads(l)["metadata"]["source_id"]): json.loads(l)["target"] for l in open(manifest)}


def patch(compiled_path, gtmap, out, limit):
    rows = list(read_jsonl(compiled_path))
    patched = []
    for r in rows:
        uid = (r.get("case", {}).get("metadata") or {}).get("uid")
        tgt = gtmap.get(uid)
        if tgt and r.get("valid"):        # judge only makes sense on a valid mesh
            r["case"]["target"] = tgt
            r["split"] = "full"
            patched.append(r)
        if limit and len(patched) >= limit:
            break
    write_jsonl(out, patched)
    return len(patched)


def agg_qa(metrics_path):
    rows = list(read_jsonl(metrics_path))
    qs, qp = [], []
    for r in rows:
        m = r.get("raw_metrics", {})
        for k, v in m.items():
            if isinstance(v, (int, float)):
                if k in ("qa_semantic", "qa_s", "QA-S", "qa_semantic_acc"):
                    qs.append(v)
                elif k in ("qa_param", "qa_p", "QA-P", "qa_param_acc"):
                    qp.append(v)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--manifest", default="data/manifests/text_to_3d_full.jsonl")
    ap.add_argument("--outdir", default="results_p3d")
    args = ap.parse_args()
    gtmap = gt_by_uid(args.manifest)
    jc = load_judge_config(Path("configs"))
    judge_client = get_client(jc.judge_model, Path("configs"))
    bucket = get_metric_bucket("judge")
    work_root = Path(args.outdir) / "judge_work"

    def score_one(row):
        rc = ResolvedCase(Case.from_dict(row["case"]), data_root(row["split"]))
        ctx = ScoreContext(task=row["task"], fmt=row["format"], case=rc,
                           compiled=row.get("compile", {}), work_dir=work_root / row["id"].replace("/", "_"),
                           judge_client=judge_client, decompose_client=None,
                           shared={"stage1_code": row.get("code"), "text_mode": row.get("text_mode", "parametric")})
        try:
            return {"id": row["id"], "raw_metrics": bucket.score(ctx) or {}}
        except Exception as e:  # noqa: BLE001
            return {"id": row["id"], "raw_metrics": {"_judge_error": str(e)[:120]}}

    for model in args.models:
        print(f"===== judge {model} =====", flush=True)
        for tag in ["baseline", "paircoder"]:
            comp = Path(args.outdir) / f"compiled_{tag}_{model}_full.jsonl"
            if not comp.exists():
                print(f"  [{tag}] missing {comp}"); continue
            cgt = Path(args.outdir) / f"compiled_judge_{tag}_{model}.jsonl"
            n = patch(comp, gtmap, cgt, args.limit)
            rows_in = list(read_jsonl(cgt))
            out = []
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(score_one, r) for r in rows_in]
                done = 0
                for fut in as_completed(futs):
                    out.append(fut.result()); done += 1
                    if done % 50 == 0:
                        print(f"    judged {done}/{len(rows_in)} [{tag}]", flush=True)
            write_jsonl(Path(args.outdir) / f"metrics_judge_{tag}_{model}.jsonl", out)
            acc = {}
            for r in out:
                for k, v in (r.get("raw_metrics", {}) or {}).items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        acc.setdefault(k, []).append(v)
            means = {k: round(sum(v) / len(v), 4) for k, v in acc.items() if v}
            print(f"  [{tag}] n={n} judge means: {means}", flush=True)


if __name__ == "__main__":
    main()
