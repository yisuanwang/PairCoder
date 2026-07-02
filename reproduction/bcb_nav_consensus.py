#!/usr/bin/env python3
"""BigCodeBench PairCoder via Navigator-generated-input execution consensus.

Reuses the already-saved 5 candidates per problem (NO regeneration). For each problem the
Navigator (1 LLM call) writes several DISCRIMINATING test-input expressions that call task_func
with realistic/diverse/edge arguments (constructing dataframes/strings/etc. inline). We run each
candidate on those inputs, form a behavior signature, and pick a representative of the LARGEST
agreeing cluster (self-consistency / MBR-exec — no correctness oracle needed). Monotone bias:
keep candidate[0] when it is in the majority cluster, or when there is no usable agreement, so
PairCoder never regresses below baseline.

Output: a paircoder solution file. Grade it with grade_bcb.py vs the baseline.
Usage: python bcb_nav_consensus.py <candidates.jsonl> <out.jsonl> [model] [version] [workers]
"""
import os, re, sys, json, time, threading, subprocess, tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from bcb_common import load_bcb

API_BASE = os.environ.get("PAIRCODER_API_BASE", "https://api.openai.com/v1")
API_KEY  = os.environ.get("PAIRCODER_API_KEY",  "")
_lock = threading.Lock(); USAGE = {"calls": 0}

NAV_SYS = (
    "You are the Navigator in a pair-programming team. Given a Python function specification, "
    "produce diverse, discriminating TEST INPUTS that would expose a wrong implementation "
    "(typical cases, edge cases, boundaries). Output ONLY a ```python``` block defining a list "
    "`probes` of strings, each a complete expression that CALLS `task_func(...)` with concrete "
    "arguments (construct any needed objects inline, e.g. pandas DataFrames, lists, strings). "
    "6-10 probes. Do NOT include expected outputs.")


def nav_inputs(model, prob, retries=3):
    user = f"Function specification:\n```python\n{prob['prompt']}\n```\nReturn the `probes` list."
    from paircoder.client import make_client, guarded_create
    client = make_client()
    for a in range(retries):
        try:
            r = guarded_create(client, 
                model=model, messages=[{"role": "system", "content": NAV_SYS},
                                       {"role": "user", "content": user}],
                extra_body={"reasoning_effort": "none"})
            with _lock: USAGE["calls"] += 1
            txt = r.choices[0].message.content or ""
            m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", txt, re.DOTALL)
            code = m[0] if m else txt
            g = {}
            exec(code, g)
            probes = g.get("probes")
            if isinstance(probes, list):
                return [str(p) for p in probes if "task_func" in str(p)][:10]
        except Exception:
            if a == retries - 1:
                return []
            time.sleep(2)
    return []


_PROBE = r"""
import sys, io, contextlib
{CODE}
_probes = {PROBES!r}
_sig = []
for _c in _probes:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _v = eval(_c)
        _sig.append("OK:" + repr(_v)[:160])
    except Exception as e:
        _sig.append("EXC:" + type(e).__name__)
print("SIG\t" + "\t".join(_sig))
"""


def behavior(code, probes, timeout=25):
    if not code or "task_func" not in code or not probes:
        return None
    src = _PROBE.replace("{CODE}", code).replace("{PROBES!r}", repr(probes))
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True,
                           timeout=timeout, cwd=tempfile.gettempdir())
        for line in p.stdout.splitlines():
            if line.startswith("SIG\t"):
                return line[4:]
        return None
    except subprocess.TimeoutExpired:
        return None
    finally:
        try: os.unlink(path)
        except Exception: pass


def select(model, prob, cands):
    if not cands:
        return ""
    base = cands[0]
    probes = nav_inputs(model, prob)
    if not probes:
        return base
    sigs = [behavior(c, probes) for c in cands]
    clusters = defaultdict(list)
    for c, s in zip(cands, sigs):
        if s is None:
            continue
        # require at least one non-exception result so we don't cluster on shared crashes
        if all(part.startswith("EXC") for part in s.split("\t")):
            continue
        clusters[s].append(c)
    if not clusters:
        return base
    best = max(clusters.values(), key=len)
    if len(best) < 2:               # no agreement -> trust baseline
        return base
    return base if base in best else min(best, key=len)


def main():
    cand_file, out_file = sys.argv[1], sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-5.4-mini"
    version = sys.argv[4] if len(sys.argv) > 4 else "v0.1.4"
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 15
    probs = {p["task_id"]: p for p in load_bcb(version)}
    cands = {json.loads(l)["task_id"]: json.loads(l)["candidates"] for l in open(cand_file)}

    # resume
    done = {}
    if os.path.exists(out_file):
        for l in open(out_file):
            try: d = json.loads(l); done[d["task_id"]] = d["solution"]
            except Exception: pass
    todo = [t for t in cands if t not in done and t in probs]
    print(f"nav-consensus: {len(cands)} probs, {len(done)} cached, {len(todo)} to do", flush=True)

    fh = open(out_file, "a"); lk = threading.Lock()

    def work(tid):
        return tid, select(model, probs[tid], cands[tid])

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(work, tid) for tid in todo]):
            tid, sol = f.result()
            with lk:
                fh.write(json.dumps({"task_id": tid, "solution": sol}) + "\n"); fh.flush()
            n += 1
            if n % 50 == 0: print(f"  {n}/{len(todo)} ({USAGE['calls']} nav calls)", flush=True)
    fh.close()
    print(f"done -> {out_file} ({USAGE['calls']} nav calls)", flush=True)


if __name__ == "__main__":
    main()
