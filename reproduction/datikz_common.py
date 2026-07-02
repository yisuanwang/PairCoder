#!/usr/bin/env python3
"""DaTikZ (caption -> TikZ/LaTeX) benchmark. Model writes a standalone LaTeX TikZ document from a
caption. Compile with pdflatex -> PDF -> PNG (ImageMagick convert). Public signal = COMPILES to a
non-empty image (PairCoder repairs LaTeX errors). Metric: compile-rate + image SSIM/CLIP vs reference."""
import os, re, io, subprocess, tempfile, shutil
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import pandas as pd

_DF = None


def load_datikz(n=None):
    global _DF
    if _DF is None:
        _DF = pd.read_parquet(hf_hub_download("nllg/datikz-v3", "data/test-00000-of-00001.parquet", repo_type="dataset"))
    df = _DF if n is None else _DF.head(n)
    out = []
    for i, r in df.iterrows():
        cap = r["caption"]
        if not isinstance(cap, str) or len(cap.strip()) < 10:
            continue
        out.append({"task_id": f"tikz_{i}", "caption": cap.strip()[:1200],
                    "ref_code": r["code"], "ref_img": r["image"]["bytes"]})
    return out


def build_prompt(task):
    return (f"Write a COMPLETE, compilable standalone LaTeX document using TikZ that draws the "
            f"figure described below. It must compile with pdflatex. Return ONLY one ```latex``` "
            f"code block (\\documentclass ... \\end{{document}}).\n\nFigure description:\n{task['caption']}")


def extract_latex(text):
    if not text:
        return ""
    f = re.findall(r"```(?:latex|tex)?\s*\n(.*?)```", text, re.DOTALL)
    code = max(f, key=len).strip() if f else text.strip()
    if "\\documentclass" not in code:  # wrap a bare tikzpicture
        if "\\begin{tikzpicture}" in code:
            code = ("\\documentclass[tikz,border=2pt]{standalone}\n\\usepackage{tikz}\n"
                    "\\usetikzlibrary{arrows,arrows.meta,positioning,shapes,calc,patterns,decorations.pathmorphing}\n"
                    "\\begin{document}\n" + code + "\n\\end{document}")
    return code


def compile_tikz(code, png, timeout=40):
    """Compile LaTeX -> PDF -> PNG. Return True if a non-empty image is produced."""
    if not code or "\\" not in code:
        return False
    d = tempfile.mkdtemp()
    try:
        tex = os.path.join(d, "f.tex"); open(tex, "w").write(code)
        try:
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"],
                           cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        pdf = os.path.join(d, "f.pdf")
        if not os.path.exists(pdf) or os.path.getsize(pdf) < 800:
            return False
        try:  # ghostscript (ImageMagick `convert` is blocked by PDF security policy)
            subprocess.run(["gs", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r120",
                            "-dFirstPage=1", "-dLastPage=1", f"-sOutputFile={png}", pdf],
                           capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return os.path.exists(png) and os.path.getsize(png) > 1000
    finally:
        shutil.rmtree(d, ignore_errors=True)


def latex_err(code, timeout=40):
    if "\\" not in (code or ""):
        return "empty/invalid"
    d = tempfile.mkdtemp()
    try:
        open(os.path.join(d, "f.tex"), "w").write(code)
        p = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"],
                           cwd=d, capture_output=True, text=True, timeout=timeout)
        errs = [l for l in (p.stdout + p.stderr).splitlines() if l.startswith("!")]
        return " | ".join(errs[:4])[:300] or "compile failed"
    except subprocess.TimeoutExpired:
        return "timeout"
    finally:
        shutil.rmtree(d, ignore_errors=True)
