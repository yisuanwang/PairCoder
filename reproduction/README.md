# Reproducing the PairCoder++ experiments

This directory contains the exact runners, graders, and scorers used to produce
the PairCoder++ results (17 public benchmarks × 7 models × 3 vendors, thinking
off). Every benchmark follows the same three steps:

1. **generate** — a `run_*.py` runner queries the model twice per case: a
   single-model baseline and PairCoder (`paircoder.paper_solve`), grounding the
   Navigator on that benchmark's own toolchain (compile / execute / render).
2. **grade / score** — a `grade_*.py` or `score_*.py` computes the official
   metric(s) for both arms from the saved predictions.
3. **read** — each script prints `single -> PairCoder` and writes a `results*/`
   JSONL you can aggregate.

All model access is through the `paircoder` package, which is **env-configured
and carries no secrets** (see the top-level README). Nothing here hard-codes an
API key.

## 0. One-time setup

```bash
# from the repository root
pip install -e .                       # exposes the `paircoder` package
pip install -r reproduction/requirements-repro.txt   # per-benchmark extras

export PAIRCODER_API_BASE="https://api.openai.com/v1"
export PAIRCODER_API_KEY="sk-..."      # your key
export PAIRCODER_EFFORT=none           # thinking OFF (as in the paper)
# for doubao-* / deepseek-* models (Volcano ARK), also:
export ARK_API_BASE="https://ark.cn-beijing.volces.com/api/v3"
export ARK_API_KEY="ark-..."
```

Run the scripts from the repository root (so `import paircoder` resolves), e.g.
`python reproduction/run_lcb.py --model gpt-5.4-mini`.

### Datasets, benchmark clones, and toolchains

Benchmarks pull their data from Hugging Face at run time; a few need a local
clone or a system toolchain. Place clones/tools where each `*_common.py`
expects them (relative to `reproduction/`) or point the documented env var at
your copy:

| Benchmark | Data / clone | System toolchain |
|---|---|---|
| LiveCodeBench | HF `livecodebench/code_generation_lite` | — |
| BigCodeBench | HF `bigcode/bigcodebench` | — |
| DS-1000 | HF `xlangai/DS-1000` | — |
| HumanEval-X (C++/Java/JS) | HF `THUDM/humaneval-x` | `g++`, `java`, `node` |
| WebApp1K | HF `onekq-ai/WebApp1K-Duo-React` + `webbench/` harness | Node.js + `jest` |
| VerilogEval | `verilog-eval` dataset | `iverilog`, `vvp` |
| RTLLM | `RTLLM/` clone | `iverilog`, `vvp` |
| DaTikZ | HF `nllg/datikz-v3` | `pdflatex`, ImageMagick `convert` |
| Plot2Code | HF `TencentARC/Plot2Code` | matplotlib |
| PandasPlotBench | HF `JetBrains-Research/PandasPlotBench` | matplotlib |
| ChartMimic | HF `ChartMimic/ChartMimic` | matplotlib |
| StarVector | HF `starvector/svg-icons` | cairosvg |
| GenCAD-Code | HF `CADCODER/GenCAD-Code` | `cadquery` |
| 3DCodeBench | HF `YipengGao/3DCode` | Blender 4.2.3 (headless) |
| P3D-Bench | HF `SpatiaOS/P3D-Bench` + the `p3dbench` package | (see below) |

The visual scorers (`score_datikz_vis.py`, `score_chartmimic.py`,
`score_svgbench.py`, `score_cadbench.py`, `tdcb_score*.py`) need a GPU Python
with `torch`, `open_clip`, `transformers` (DINO/SigLIP-2), `scikit-image`,
`trimesh`, `scipy`. Keep them in a separate environment from the lightweight
runner env if you like — they only read the saved artifacts.

## 1. Program synthesis, multilingual, web & hardware

```bash
# LiveCodeBench
python reproduction/run_lcb.py   --model gpt-5.4-mini
python reproduction/grade_lcb.py results_lcb/<run>.jsonl          # pass@1
python reproduction/grade_lcb_passk.py ...                        # pass@k

# BigCodeBench
python reproduction/run_bcb.py   --model gpt-5.4-mini
python reproduction/grade_bcb.py results_bcb/<run>.jsonl

# DS-1000
python reproduction/run_ds.py    --model gpt-5.4-mini
python reproduction/grade_ds.py  results_ds/<run>.jsonl

# HumanEval-X  (lang in {cpp, java, js})
python reproduction/run_hex.py   --model gpt-5.4-mini --lang cpp
python reproduction/grade_hex.py results_hex/<run>.jsonl

# WebApp1K  (needs the webbench/ Node harness)
python reproduction/run_webapp.py    --model gpt-5.4-mini
python reproduction/grade_webapp.py  results_webapp/<run>.jsonl

# VerilogEval / RTLLM  (need iverilog + vvp)
python reproduction/run_verilog.py --model gpt-5.4-mini
python reproduction/grade_verilog.py results_verilog/<run>.jsonl
python reproduction/run_rtllm.py   --model gpt-5.4-mini
python reproduction/grade_rtllm.py results_rtllm/<run>.jsonl
```

## 2. Code-driven artifacts (charts, figures, SVG, CAD, 3D)

```bash
# DaTikZ  (compile rate + visual similarity)
python reproduction/run_datikz.py       --model gpt-5.4-mini
python reproduction/grade_datikz.py     results_datikz/<run>.jsonl   # compile rate
python reproduction/score_datikz_vis.py results_datikz/<run>.jsonl   # SSIM/CLIP/DINO (GPU env)

# Plot2Code / PandasPlotBench / ChartMimic  (matplotlib; scored by score_chartmimic.py)
python reproduction/run_plot2code.py    --model gpt-5.4-mini
python reproduction/run_pandasplot.py   --model gpt-5.4-mini
python reproduction/run_chartmimic.py   --model gpt-5.4-mini
python reproduction/score_chartmimic.py results_<bench>/<run>.jsonl   # exec + SSIM + CLIP (GPU env)

# StarVector (SVG)
python reproduction/run_svgbench.py     --model gpt-5.4-mini --repo starvector/svg-icons
python reproduction/score_svgbench.py   results_svg*/<run>.jsonl       # SSIM/CLIP/DINO (GPU env)

# GenCAD-Code  (image -> CadQuery; needs cadquery)
python reproduction/run_cadbench.py     --model gpt-5.4-mini
python reproduction/score_cadbench.py   results_cadbench/<run>.jsonl   # exec + Chamfer

# 3DCodeBench  (text -> Blender; needs Blender 4.2.3)
python reproduction/run_3dcb_full.py    --model gpt-5.4-mini           # full-metric capture
python reproduction/tdcb_capture_any.py results_3dcb_full/<run>        # join/export via Blender
python reproduction/tdcb_score_run.py   results_3dcb_full/art/manifest.jsonl   # Chamfer/SigLIP-2/DINO
python reproduction/grade_3dcb.py       results_3dcb_full/<run>.jsonl  # executability
```

## 3. P3D-Bench (parametric 3D)

P3D-Bench ships its own prompts, compiler, and metrics as the `p3dbench`
package. Install it and run these from the **P3D-Bench repo root** (so its
`configs/` and `data/` resolve):

```bash
python /path/to/reproduction/run_p3d.py   --model gpt-5.4 --task text_to_3d --format minimal_json
python /path/to/reproduction/rescore_p3d.py results_p3d/                 # geometry + topology
python /path/to/reproduction/run_judge.py   results_p3d/                 # QA-S / QA-P VLM judge
python /path/to/reproduction/gen_gt_p3d.py                               # materialize GT meshes
```

`examples/run_p3dbench.py` (top level) is a self-contained reference that wires
PairCoder into P3D-Bench's `compile -> score -> summarize` pipeline end to end.

## 4. Ablations

No extra scripts — both ablations are environment switches on the same runners:

```bash
# Role-switch policy (Sec. 4.5): err<eta> | fixed<k> | none
PAIRCODER_SWITCH=err1  python reproduction/run_datikz.py --model gpt-5.4-mini   # default
PAIRCODER_SWITCH=fixed2 ...
PAIRCODER_SWITCH=none   ...

# Self-refinement control (single agent, matched budget, no role switch)
PAIRCODER_MODE=selfrefine python reproduction/run_bcb.py --model gpt-5.4-mini
```

The BigCodeBench consensus ablations are `bcb_consensus.py` (re-select over saved
candidates, no API) and `bcb_nav_consensus.py` (Navigator-input consensus).

## Token accounting

Every run tallies tokens per arm. Set `PAIRCODER_TOKLOG=tok.json` to dump the
baseline-vs-PairCoder split at process exit (used for the cost-vs-benefit
figure).
