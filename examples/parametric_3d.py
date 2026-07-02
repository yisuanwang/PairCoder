#!/usr/bin/env python3
"""PairCoder on a PARAMETRIC 3D generation task (self-contained).

Parametric generation = the model writes a *program* that builds an artifact
(here a 3D solid via `trimesh`), and we verify the artifact by *running* the
program. That executable check is exactly what PairCoder's Navigator grounds its
review on — so the loop fixes code that fails to build a valid mesh instead of
trusting the model's first guess.

This mirrors how we evaluate PairCoder on parametric-3D benchmarks such as
P3D-Bench (text -> CAD program). The only difference there is the program target
(Text2CAD minimal-JSON / OpenSCAD / CadQuery) and an official geometry metric;
the PairCoder usage pattern is identical to what you see below.

Requires: pip install trimesh
Run:
    export PAIRCODER_API_BASE="https://api.openai.com/v1"
    export PAIRCODER_API_KEY="sk-..."
    python examples/parametric_3d.py
"""
import os
import re
import sys
import json
import textwrap
import tempfile
import subprocess

from paircoder import paper_solve, single_baseline

MODEL = os.environ.get("PAIRCODER_MODEL", "gpt-5.4-mini")

# A parametric spec with exact dimensions — the kind of task where a single shot
# often produces code that does not run, but the geometry is fully checkable.
SPEC = textwrap.dedent("""\
    Write Python code using the `trimesh` library that builds the following solid:
    an open cup (mug body) — a solid cylinder of radius 3 and height 8, with a
    coaxial cylindrical cavity of radius 2.4 and depth 7 carved out from the top
    (leaving a 1-unit-thick bottom and 0.6-thick walls).

    Define a function `build()` that returns a single `trimesh.Trimesh`.
    Use trimesh.creation primitives, transforms, and boolean ops as needed.
    Return ONLY one ```python``` code block.
""")

# What the model must produce, plus the program we run to verify it.
_RUNNER = """
import json, sys
import numpy as np
{CODE}
m = build()
print("PROPS " + json.dumps({"volume": float(abs(m.volume)),
                             "watertight": bool(m.is_watertight)}))
"""


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.DOTALL)
    return (max(m, key=len) if m else (text or "")).strip()


def build_props(code: str, timeout=30):
    """Run the generated program in a subprocess; return its mesh properties or None."""
    if "def build" not in code:
        return None
    src = _RUNNER.replace("{CODE}", code)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        for line in p.stdout.splitlines():
            if line.startswith("PROPS "):
                return json.loads(line[6:])
        return None
    except subprocess.TimeoutExpired:
        return None
    finally:
        os.unlink(path)


def check(code: str):
    """Navigator's grounding signal psi: does build() run and return a real solid?

    Returns (ok, evidence, score) — the score lets PairCoder pick the best
    candidate if no round is fully accepted.
    """
    props = build_props(extract_code(code))
    if props is None:
        return (False, "build() failed to run or returned no mesh", 0.0)
    if props["volume"] <= 1e-6:
        return (False, "build() returned an empty / degenerate mesh", 0.0)
    ev = f"volume={props['volume']:.2f}, watertight={props['watertight']}"
    return (True, ev, 1.0)


if __name__ == "__main__":
    print(f"model={MODEL}\n")

    base = extract_code(single_baseline(SPEC, MODEL))
    base_ok = check(base)

    pc_text, info = paper_solve(SPEC, MODEL, max_iters=4, check=check, extract=extract_code)
    pc = extract_code(pc_text)
    pc_ok = check(pc)

    print("=== single-model baseline ===")
    print("  builds a valid solid:", base_ok[0], "|", base_ok[1])
    print(f"=== PairCoder (accepted={info['accepted']}, iters={info['iters']}) ===")
    print("  builds a valid solid:", pc_ok[0], "|", pc_ok[1])
    print()
    print("PairCoder recovered a failing baseline." if (pc_ok[0] and not base_ok[0])
          else "Both arms produced a valid solid (no headroom on this task).")
