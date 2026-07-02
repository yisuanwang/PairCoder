#!/usr/bin/env python3
"""PairCoder on P3D-Bench (parametric 3D code generation).

We reuse P3D-Bench's *official* prompts, compiler, and metrics and only swap the
inference: any OpenAI-compatible model via paircoder.client (which disables
thinking), with the regression-safe paired design.

Per case we generate N diverse candidates. The public grounding signal is
P3D-Bench's own compile() ("the code compiled to a valid STL"):
  baseline  = candidate[0]                 (the Driver's direct attempt)
  PairCoder = candidate[0] if it compiles, else the first sibling that compiles,
              else candidate[0] repaired on its concrete compile error (<=K iters),
              else candidate[0]             (never worse than baseline)

We then emit two predictions.jsonl in P3D-Bench's schema and grade both with
P3D-Bench's compile -> score -> summarize stages.

Run from the P3D-Bench repo root (configs/ + data/ are resolved relatively):
  python /path/to/examples/run_p3dbench.py --model gpt-5.4 --source full --limit 400 --metric valid
  python /path/to/examples/run_p3dbench.py --model gpt-5.4 --source demo --metric valid,geometry
"""
import os, sys, json, time, argparse, tempfile, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# P3D-Bench data/config resolve relative to its repo root (your CWD).
# make the top-level `paircoder` package importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p3dbench.registry import resolve_task, resolve_format
from p3dbench.data.loader import load_cases
from p3dbench.data.schema import Case
from p3dbench import pipeline

from paircoder.client import make_client, guarded_create

_lock = threading.Lock()
USAGE = {"calls": 0, "prompt": 0, "completion": 0}

# Diversity nudges appended to the user prompt for sibling candidates (cand 0 = plain).
ANGLES = [
    "",
    "Be meticulous about exact coordinates, radii, extrusion depths, and translation vectors stated in the spec.",
    "Re-read the spec and reproduce every numeric dimension and operation exactly; keep the profile loops closed and ordered.",
    "Favor the simplest construction that still matches every stated feature; double-check units and signs.",
    "Pay special attention to part ordering, join/cut operations, and that each profile forms a valid closed loop.",
]


def llm(model, system, user, images, temperature, retries=4):
    msgs = [{"role": "system", "content": system}]
    if images:
        content = [{"type": "text", "text": user}]
        import base64
        for p in images:
            b = base64.b64encode(Path(p).read_bytes()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b}"}})
        msgs.append({"role": "user", "content": content})
    else:
        msgs.append({"role": "user", "content": user})
    for a in range(retries):
        try:
            r = guarded_create(make_client(), model=model, messages=msgs,
                               temperature=temperature,
                               extra_body={"reasoning_effort": os.environ.get("PAIRCODER_EFFORT", "none")})
            if r.usage:
                with _lock:
                    USAGE["calls"] += 1
                    USAGE["prompt"] += r.usage.prompt_tokens or 0
                    USAGE["completion"] += r.usage.completion_tokens or 0
            return r.choices[0].message.content or ""
        except Exception as e:
            if a == retries - 1:
                print(f"[api error] {e}", file=sys.stderr)
                return ""
            time.sleep(2 * (a + 1))


def compiles(fmt_obj, code):
    """P3D-Bench's own validity signal: code -> a non-empty STL."""
    if not code or not code.strip():
        return False, "empty code"
    try:
        with tempfile.TemporaryDirectory(prefix="p3d_ground_") as td:
            cr = fmt_obj.compile(code, Path(td))
            return bool(cr.valid), ("" if cr.valid else "; ".join(map(str, cr.errors[:2])))
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def solve_paired(task_obj, fmt_obj, rc, model, n_cand, max_iters, text_mode):
    bundle = task_obj.build_prompt(fmt_obj, rc.case, [str(p) for p in rc.image_paths], text_mode=text_mode)
    imgs = bundle.images

    def gen(i):
        angle = ANGLES[i % len(ANGLES)]
        user = bundle.user + (f"\n\n{angle}" if angle else "")
        temp = 0.0 if i == 0 else 0.8
        return fmt_obj.extract_code(llm(model, bundle.system, user, imgs, temp))

    with ThreadPoolExecutor(max_workers=n_cand) as ex:
        cands = list(ex.map(gen, range(n_cand)))
    cands = [c for c in cands if c and c.strip()] or [""]
    baseline = cands[0]

    ok0, _ = compiles(fmt_obj, baseline)
    if ok0:
        return bundle, baseline, baseline, cands, "cand0_valid"
    for c in cands[1:]:
        ok, _ = compiles(fmt_obj, c)
        if ok:
            return bundle, baseline, c, cands, "sibling_valid"
    # repair candidate[0] on its concrete compile error
    cand = baseline
    for _ in range(max_iters):
        _, err = compiles(fmt_obj, cand)
        lang = {"minimal-json": "json", "openscad": "scad", "cadquery": "python", "threejs": "javascript"}.get(fmt_obj.slug, "")
        fix_user = (f"{bundle.user}\n\n---\nYour previous {fmt_obj.display_name} code failed to "
                    f"compile/export:\n\n```{lang}\n{cand}\n```\n\nThe error was:\n```\n{err}\n```\n"
                    "Fix the error and output ONLY the corrected complete code.")
        new = fmt_obj.extract_code(llm(model, bundle.system, fix_user, imgs, 0.3))
        if new and new.strip():
            cand = new
            ok, _ = compiles(fmt_obj, cand)
            if ok:
                return bundle, baseline, cand, cands, "repaired"
    return bundle, baseline, cand, cands, "no_valid"


def row_for(rc, task, fmt, model, split, text_mode, bundle, code):
    return {
        "id": rc.id, "task": task, "format": fmt, "model": model, "split": split,
        "text_mode": text_mode, "case": rc.case.to_dict(),
        "prompt": {"system": bundle.system, "user": bundle.user, "images": bundle.images},
        "raw_text": None, "code": code, "usage": {}, "error": None if code and code.strip() else "empty",
    }


def synth_full_text_cases(limit):
    """Build text-to-3d cases from the HF annotations (text_param). No GT geometry
    (HF ships none), so only the `valid` metric is meaningful for these."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("SpatiaOS/P3D-Bench", "data/text_to_3d/annotations.jsonl", repo_type="dataset")
    from p3dbench.data.loader import ResolvedCase
    out = []
    for i, line in enumerate(open(p)):
        if limit and i >= limit:
            break
        a = json.loads(line)
        cid = f"p3d_text-to-3d_full_{i:04d}"
        case = Case.from_dict({
            "id": cid, "task": "text-to-3d", "split": "full",
            "input": {"text": a["text_param"], "image_paths": [], "part_annotations": []},
            "target": {"format": "minimal-json", "code_path": None, "step_path": None,
                       "mesh_path": None, "render_paths": [], "part_paths": [], "qa_bank_path": None},
            "metadata": {"source": "text2cad-v1.1", "text_desc": a.get("text_desc", ""),
                         "summary": a.get("summary", ""), "uid": a.get("uid", "")},
        })
        out.append(ResolvedCase(case, Path("data/full")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ARK model id (doubao-… / deepseek-…) or alias")
    ap.add_argument("--task", default="text-to-3d")
    ap.add_argument("--format", default="minimal-json")
    ap.add_argument("--source", choices=["demo", "full"], default="demo",
                    help="demo = in-repo 3 cases w/ GT geometry; full = 400 HF text specs (valid only)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cands", type=int, default=4)
    ap.add_argument("--max-iters", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--text-mode", default="parametric")
    ap.add_argument("--metric", default="valid")
    ap.add_argument("--outdir", default="results_p3d")
    args = ap.parse_args()

    ALIAS = {"doubao": "doubao-seed-2-0-mini-260428", "deepseek": "deepseek-v3-2-251201",
             "doubao-seed-2.0-mini": "doubao-seed-2-0-mini-260428",
             "doubao-1.5-lite": "doubao-1-5-lite-32k-250115",
             "deepseek-v3.2": "deepseek-v3-2-251201",
             "deepseek-v4-flash": "deepseek-v4-flash-260425"}
    model = ALIAS.get(args.model, args.model)
    model_tag = args.model

    task_obj = resolve_task(args.task)
    fmt_obj = resolve_format(args.format)
    task_obj.check_format(fmt_obj)

    if args.source == "demo":
        cases = load_cases(args.task, "demo", limit=(args.limit or None))
        split = "demo"
    else:
        cases = synth_full_text_cases(args.limit or None)
        split = "full"

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    base_path = outdir / f"pred_baseline_{args.task}_{args.format}_{model_tag}_{args.source}.jsonl"
    pc_path = outdir / f"pred_paircoder_{args.task}_{args.format}_{model_tag}_{args.source}.jsonl"

    print(f"=== P3D-Bench | {args.task}/{args.format} | {model_tag} | {args.source} | "
          f"{len(cases)} cases | N={args.cands} ===", flush=True)

    base_rows, pc_rows, decisions = [], [], {}
    done = 0
    sem = threading.Semaphore(args.workers)
    rows_lock = threading.Lock()

    def work(rc):
        nonlocal done
        with sem:
            bundle, base_code, pc_code, cands, why = solve_paired(
                task_obj, fmt_obj, rc, model, args.cands, args.max_iters, args.text_mode)
        with rows_lock:
            base_rows.append(row_for(rc, args.task, args.format, f"{model_tag}-baseline", split, args.text_mode, bundle, base_code))
            pc_rows.append(row_for(rc, args.task, args.format, f"{model_tag}-paircoder", split, args.text_mode, bundle, pc_code))
            decisions[why] = decisions.get(why, 0) + 1
            done += 1
            if done % 10 == 0 or done == len(cases):
                print(f"  [{done}/{len(cases)}] decisions={decisions}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, cases))

    from p3dbench.utils import write_jsonl
    write_jsonl(base_path, base_rows)
    write_jsonl(pc_path, pc_rows)
    print(f"\nwrote {base_path} ({len(base_rows)}) and {pc_path} ({len(pc_rows)})")
    print(f"PairCoder decisions: {decisions}")
    print(f"tokens: {USAGE}")

    # Grade both with P3D-Bench's own compile -> score -> summarize.
    from p3dbench.utils import read_jsonl
    buckets = [b.strip() for b in args.metric.split(",") if b.strip()]
    for tag, pred in [("baseline", base_path), ("paircoder", pc_path)]:
        wd = outdir / f"work_{tag}_{model_tag}_{args.source}"
        comp = outdir / f"compiled_{tag}_{model_tag}_{args.source}.jsonl"
        met = outdir / f"metrics_{tag}_{model_tag}_{args.source}.jsonl"
        summ = outdir / f"summary_{tag}_{model_tag}_{args.source}.json"
        pipeline.compile_predictions(pred, out=comp, work_dir=wd)
        # Run each requested bucket, merging raw_metrics per case into one file.
        merged = {}
        for b in buckets:
            tmp = outdir / f"_m_{b}_{tag}.jsonl"
            pipeline.score(comp, b, out=tmp, work_dir=wd)
            for r in read_jsonl(tmp):
                m = merged.setdefault(r["id"], r)
                m["raw_metrics"] = {**m.get("raw_metrics", {}), **r.get("raw_metrics", {})}
                m["buckets"] = sorted(set(m.get("buckets", [])) | set(r.get("buckets", [])))
            tmp.unlink(missing_ok=True)
        write_jsonl(met, list(merged.values()))
        pipeline.summarize(met, out=summ)
        s = json.loads(summ.read_text())
        for g in s["groups"]:
            print(f"  [{tag}] valid_rate={g['valid_rate']:.3f}  score={g['score']}  buckets={g['buckets']}")


if __name__ == "__main__":
    main()
