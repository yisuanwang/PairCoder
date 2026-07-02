#!/usr/bin/env python3
"""PairCoder (paper-faithful Driver/Navigator loop) on RTLLM (design spec -> Verilog).
baseline = single direct generation; PairCoder = pc_paper.paper_solve (Navigator review with
[NOERROR] protocol + iverilog-compile psi evidence + error-triggered role switching)."""
import os, sys, json, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from rtllm_common import load_rtllm, build_prompt, extract_code, compiles, compile_err, run_test, has_module, run_nav_test
from paircoder import paper_solve, single_baseline, USAGE
_lock=threading.Lock()
AUTHOR_TB=("Write a SELF-CHECKING Verilog testbench module named nav_tb for the module described "
           "in the task (instantiate it by its exact name/ports). Drive several representative and "
           "edge-case stimuli, compare outputs against the spec, $display(\"NAV_FAIL ...\") on any "
           "mismatch, and $display(\"NAV_DONE\") then $finish at the end. Return ONLY one "
           "```verilog``` block with the complete testbench.")
def solve(prob,model,n_cand,max_iters):
    nm=prob["module"]; q=build_prompt(prob)
    def chk(text):
        c=extract_code(text)
        if not has_module(c,nm): return False, f"no module named {nm} found"
        return (True,"") if compiles(c,nm) else (False, compile_err(c,nm)[:300])
    def run_tb(code_text, tb_text):
        return run_nav_test(extract_code(code_text), extract_code(tb_text))
    base=extract_code(single_baseline(q,model))
    base_pass=run_test(prob,base)
    pc_raw,tel=paper_solve(q,model,max_iters=max_iters,check=chk,
                           author_test=AUTHOR_TB,run_authored_test=run_tb)
    pc=extract_code(pc_raw)
    pc_pass=run_test(prob,pc)
    return {"task_id":prob["task_id"],"cand_pass":[base_pass],"base_pass":base_pass,
            "paircoder_pass":pc_pass,"iters":tel["iters"],"accepted":tel["accepted"]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="gpt-5.4-mini"); ap.add_argument("--cands",type=int,default=5)
    ap.add_argument("--max-iters",type=int,default=3); ap.add_argument("--workers",type=int,default=10)
    ap.add_argument("--outdir",default="results_rtllm")
    args=ap.parse_args(); probs=load_rtllm(); os.makedirs(args.outdir,exist_ok=True)
    out=os.path.join(args.outdir,f"results_rtllm_{args.model}.jsonl"); done=set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l)["task_id"])
            except: pass
    todo=[p for p in probs if p["task_id"] not in done]
    print(f"=== RTLLM | {args.model} | {len(probs)} ({len(done)} cached) ===",flush=True)
    fh=open(out,"a"); lk=threading.Lock(); n=0; t0=time.time()
    import concurrent.futures as _cf
    ex=ThreadPoolExecutor(max_workers=args.workers)
    futs={ex.submit(solve,p,args.model,args.cands,args.max_iters):p for p in todo}
    pending=set(futs); PER=300  # max 5 min of no-progress before abandoning hung designs
    while pending:
        done_set,pending=_cf.wait(pending,timeout=PER,return_when=_cf.FIRST_COMPLETED)
        if not done_set:
            print(f"  [no-progress {PER}s -> abandoning {len(pending)} stuck design(s)]",flush=True); break
        for f in done_set:
            try: r=f.result()
            except Exception: r=None
            if r:
                with lk: fh.write(json.dumps(r)+"\n"); fh.flush()
            n+=1
            if n%5==0: print(f"  [{n}/{len(todo)}]",flush=True)
    for f,p in futs.items():
        if not f.done():
            with lk: fh.write(json.dumps({"task_id":p["task_id"],"cand_pass":[False],"base_pass":False,"paircoder_pass":False})+"\n"); fh.flush()
    fh.close(); from paircoder import client as pc_client
    print(f"  -> {out} ({time.time()-t0:.0f}s) TOKENS={pc_client.TOKENS}",flush=True)
    os._exit(0)  # hung worker threads can't be joined; hard-exit cleanly
if __name__=="__main__": main()
