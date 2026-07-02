#!/usr/bin/env python3
"""BigCodeBench consensus re-selection (no new API calls).

Reuses the already-saved 5 candidates per problem. For each problem we probe the candidates
by EXECUTING the docstring example call(s) (the `>>> ... task_func(...)` lines) and recording
each candidate's behavior signature (repr of the return, or the exception raised). We then pick
a representative of the LARGEST agreeing cluster — self-consistency / MBR-exec, which needs no
correctness oracle. Monotone bias: if candidate[0] is in the majority cluster (or no usable
probe / all crash), keep candidate[0] so we never regress below baseline.

Writes a new paircoder solution file: samples_paircoder_consensus_bcb_<model>.jsonl
Usage: python bcb_consensus.py <candidates.jsonl> <out.jsonl> [version] [workers]
"""
import os, re, sys, json, subprocess, tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from bcb_common import load_bcb

CALL_RE = re.compile(r"task_func\s*\(")


def extract_probe_calls(prompt):
    """Return up to 3 distinct `task_func(...)` call expressions from the docstring examples."""
    calls = []
    for line in prompt.splitlines():
        s = line.strip()
        if s.startswith(">>>") or s.startswith("..."):
            s = s.lstrip(">. ").strip()
            m = CALL_RE.search(s)
            if m:
                # take from task_func( to the matching close paren (best effort: to end / last ')')
                expr = s[m.start():]
                # if it's an assignment RHS already stripped; trim trailing comparison ops
                expr = re.split(r"\s==\s|\s!=\s", expr)[0].strip()
                if expr.count("(") <= expr.count(")"):
                    calls.append(expr)
    # dedupe, keep order
    seen = set(); out = []
    for c in calls:
        if c not in seen:
            seen.add(c); out.append(c)
    return out[:3]


_PROBE = r"""
import sys, io, contextlib
{CODE}
_calls = {CALLS!r}
_sig = []
for _c in _calls:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _v = eval(_c)
        _sig.append("OK:" + repr(_v)[:200])
    except Exception as e:
        _sig.append("EXC:" + type(e).__name__)
print("SIG\t" + "\t".join(_sig))
"""


def behavior(code, calls, timeout=20):
    if not code or "task_func" not in code or not calls:
        return None
    src = _PROBE.replace("{CODE}", code).replace("{CALLS!r}", repr(calls))
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


def select(prob, cands):
    """Return the consensus-selected candidate (monotone bias to candidate[0])."""
    if not cands:
        return ""
    base = cands[0]
    calls = extract_probe_calls(prob["prompt"])
    if not calls:
        return base
    sigs = [behavior(c, calls) for c in cands]
    # cluster by signature, ignoring None (crash/timeout) and all-exception sigs
    clusters = defaultdict(list)
    for c, s in zip(cands, sigs):
        if s is None or all(part.startswith("EXC") for part in s.split("\t")):
            continue
        clusters[s].append(c)
    if not clusters:
        return base
    # largest cluster; if candidate[0] is in it, keep candidate[0] (monotone)
    best = max(clusters.values(), key=len)
    if len(best) == 1:               # no agreement -> don't trust; keep baseline
        return base
    return base if base in best else min(best, key=len)


def main():
    cand_file, out_file = sys.argv[1], sys.argv[2]
    version = sys.argv[3] if len(sys.argv) > 3 else "v0.1.4"
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    probs = {p["task_id"]: p for p in load_bcb(version)}
    cands = {json.loads(l)["task_id"]: json.loads(l)["candidates"] for l in open(cand_file)}

    def work(tid):
        return tid, select(probs[tid], cands[tid]) if tid in probs else ""

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        d = 0
        for f in as_completed([ex.submit(work, tid) for tid in cands]):
            tid, sol = f.result(); results[tid] = sol; d += 1
            if d % 100 == 0: print(f"  selected {d}/{len(cands)}", file=sys.stderr, flush=True)

    with open(out_file, "w") as fh:
        for tid in cands:
            fh.write(json.dumps({"task_id": tid, "solution": results[tid]}) + "\n")
    print(f"  -> {out_file}")


if __name__ == "__main__":
    main()
