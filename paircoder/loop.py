#!/usr/bin/env python3
"""Faithful PairCoder loop from the paper (Algorithm 1), generic over
benchmarks. Two LLM agents with conversation memory:
  Driver    — writes/revises code               C_t = F_driver(I_dri, Q, M_{t-1})
  Navigator — reviews, returns [NOERROR]/REVISE R_t = F_navigator(I_nav, Q, C_t, M_{t-1})
Loop until ACCEPT ([NOERROR]) or max_iters; roles SWITCH each fix+review round (error-triggered:
a REVISE is an error signal — matches the original repro and the paper's switching policy).
Self-Mirror = role prompts re-asserted on every switch. Optional `check` = the paper's
verification predicates psi_i (parse/compile/execute evidence fed into the Navigator's review).

Baseline for comparison = ONE direct generation by the same model (single-model).
"""
import os, time, threading
from .client import make_client, guarded_create

_lock = threading.Lock()
USAGE = {"calls": 0}
# Per-arm token accounting: baseline vs paper-loop (dumped to tok_split.json at process exit)
TOK_SPLIT = {"base": {"prompt": 0, "completion": 0, "total": 0, "calls": 0},
             "pc":   {"prompt": 0, "completion": 0, "total": 0, "calls": 0}}

DRIVER_PROMPT = ("You are now the Driver in pair programming and your task is to write code. "
                 "Please follow the instructions below to generate the code, and only return the "
                 "full code content, not the extra text.")
NAVIGATOR_PROMPT = ("You are now the Navigator in pair programming and your task is to review the "
                    "code and provide feedback. Please review the code below to indicate if there "
                    "is an error, or just return [NOERROR] if there is no error.")


class Agent:
    """One LLM agent with its own conversation memory (M_t lives in both agents' histories)."""

    def __init__(self, model, effort=None, bucket="pc"):
        self.model = model
        self.effort = effort or os.environ.get("PAIRCODER_EFFORT", "none")
        self.client = make_client()
        self.history = []  # list of {role, content}
        self.bucket = bucket

    def request(self, content, retries=4):
        self.history.append({"role": "user", "content": content})
        for a in range(retries):
            try:
                r = guarded_create(self.client, model=self.model, messages=self.history,
                                   extra_body={"reasoning_effort": self.effort})
                with _lock:
                    USAGE["calls"] += 1
                    try:
                        u = r.usage
                        if u:
                            b = TOK_SPLIT[self.bucket]
                            b["prompt"] += u.prompt_tokens or 0
                            b["completion"] += u.completion_tokens or 0
                            b["total"] += u.total_tokens or 0
                            b["calls"] += 1
                    except Exception:
                        pass
                out = r.choices[0].message.content or ""
                if not out.strip() and a < retries - 1:
                    # empty response (provider hiccup) — retry without polluting history
                    self.history.pop()
                    time.sleep(2 * (a + 1))
                    self.history.append({"role": "user", "content": content})
                    continue
                self.history.append({"role": "assistant", "content": out})
                return out
            except Exception:
                if a == retries - 1:
                    self.history.pop()  # keep history consistent on failure
                    return ""
                time.sleep(3 * (a + 1))


def _mm2(content, extra_b64=None):
    """Append a second image to an existing multimodal content list."""
    if not extra_b64:
        return content
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    return content + [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{extra_b64}"}}]


def _content(text, image_b64=None):
    """Build a message content: plain text, or text+image for multimodal benches."""
    if not image_b64:
        return text
    return [{"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]


def single_baseline(question, model, sys_hint="", image_b64=None):
    """Single-model baseline: one direct generation (no pairing)."""
    a = Agent(model, bucket="base")
    if sys_hint:
        a.history.append({"role": "system", "content": sys_hint})
    return a.request(_content(question, image_b64))


def paper_solve(question, model, max_iters=4, check=None, sys_hint="", extract=None,
                author_test=None, run_authored_test=None, image_b64=None, render_current=None):
    """The paper's Driver/Navigator loop. Returns (final_code_text, telemetry).

    question : the task prompt Q (bench-specific, includes formatting instructions)
    check    : optional callable(code_text) -> (ok: bool, evidence: str). Implements the paper's
               verification predicates psi_1/psi_2 (parse/compile); shown to the Navigator.
    sys_hint : optional bench-specific system context appended to both role prompts.
    extract  : optional callable(text)->code used ONLY for the check (dialogue keeps raw text).
    author_test / run_authored_test : optional TDD-style review (paper's psi_3
               `satisfy(c, spec(q))` + the TDD strategy the paper cites). `author_test` is the
               instruction telling the NAVIGATOR to write a test for the spec; the test is
               authored ONCE before reviewing. `run_authored_test(code_text, test_text)` executes
               the Driver's code against the Navigator's test and returns (ok, evidence). The
               Navigator weighs this evidence in its review; [NOERROR] remains its decision.
    """
    if os.environ.get("PAIRCODER_MODE") == "selfrefine":
        # Ablation / reviewer control: SINGLE agent, same psi evidence, same round budget,
        # NO second persona and NO role switch (Reflexion / Self-Debug style self-refinement).
        # Isolates "pair programming" (second persona + role switch) from "execution feedback
        # + more tokens". Matched budget: same max_iters rounds, same evidence (incl. render).
        return _self_refine_solve(question, model, max_iters, check, sys_hint, extract,
                                  image_b64, render_current)
    drv = Agent(model)
    nav = Agent(model)
    # Self-Mirror (paper: SelfMirror = concat(RolePrompt(rho), Q, M)) — identity-prefixed prompts
    # injected as system messages (prompt construction, not an extra dialogue turn).
    drv.history.append({"role": "system", "content": DRIVER_PROMPT + (("\n" + sys_hint) if sys_hint else "")})
    nav.history.append({"role": "system", "content": NAVIGATOR_PROMPT + (("\n" + sys_hint) if sys_hint else "")})

    # Navigator authors its test first (TDD: tests before judging the implementation)
    authored = None
    if author_test and run_authored_test:
        authored = nav.request(_content(
            f"As the Navigator, before reviewing any code, write a test for this task so you can "
            f"verify the Driver's code objectively.\n{author_test}\nThe task is:\n{question[:4000]}", image_b64))

    code = drv.request(_content(question, image_b64))  # C_1
    iters = 0
    accepted = False
    _consec = 0
    history = []  # M_t: [(C_t, psi_ok, R_t)] — for Algorithm-1's argmax-Quality fallback
    while iters < max_iters:
        # --- Navigator phase: R_t = F_nav(I_nav, Q, C_t, M_{t-1}) (+ psi evidence if available)
        psi_ok = None; psi_score = None; evidence = ""
        if check is not None:
            try:
                _r = check(extract(code) if extract else code)
                psi_ok, ev = _r[0], _r[1]
                psi_score = _r[2] if len(_r) > 2 else None
                evidence = ("\nExecution/verification result: " +
                            ("PASSED (no errors detected by tools)." if psi_ok else f"FAILED: {ev}"))
                if psi_ok and ev:
                    evidence += f" ({ev})"
            except Exception as e:
                evidence = f"\nExecution/verification result: tool error: {str(e)[:200]}"
        if authored is not None:
            try:
                t_ok, t_ev = run_authored_test(extract(code) if extract else code, authored)
                evidence += ("\nYour own test's result on this code: " +
                             ("ALL PASSED." if t_ok else f"FAILED: {t_ev[:400]}") +
                             "\n(If your test itself looks wrong, judge by the spec instead.)")
                psi_ok = bool(psi_ok if psi_ok is not None else True) and t_ok
            except Exception as e:
                evidence += f"\nYour own test could not run: {str(e)[:200]}"
        cur_b64 = None
        if render_current is not None:
            try:
                cur_b64 = render_current(code)
            except Exception:
                cur_b64 = None
        _rev_imgs = ("\nThe FIRST image is the TARGET; the SECOND image is what the current code "
                     "renders. Compare them visually and point out concrete differences to fix."
                     if (image_b64 and cur_b64) else "")
        _rev_content_extra = cur_b64
        review = nav.request(_mm2(_content(
            f"Review the code below to indicate if there is an error, and only return [NOERROR] "
            f"if there is no error. Review like a rigorous pair-programming Navigator: "
            f"(1) re-read EVERY requirement in the question; (2) construct 2-3 concrete test "
            f"inputs (including edge cases) and TRACE the code's behavior on them step by step, "
            f"comparing against the expected outputs; (3) check syntax and interfaces (names, "
            f"signatures, formats) exactly match the question. Do NOT return [NOERROR] unless "
            f"your traces all match the expectations. HOWEVER: if the verification evidence "
            f"PASSED and you cannot identify a CONCRETE, DEFINITE error (quote the exact code "
            f"line and the requirement it violates), you MUST return [NOERROR] — never revise "
            f"working code on vague suspicion or style preference. If there is a definite error, "
            f"describe it concretely and state the specific fix. "
            f"The question is [{question[:4000]}]. "
            f"The code you need to check is [{code[:8000]}]{evidence}{_rev_imgs}", image_b64), _rev_content_extra))
        history.append((code, psi_ok, psi_score, review or ""))
        if "NOERROR" in (review or ""):
            accepted = True
            break
        # --- Role switching policy (Sec. 4.5 ablation). PAIRCODER_SWITCH:
        #   'err<eta>'  : error-triggered, switch after eta consecutive REVISE signals (default eta=1)
        #   'fixed<k>'  : fixed-interval, switch every k rounds regardless of signal
        #   'none'      : never switch (single Driver keeps the keyboard, Navigator always reviews)
        _consec += 1
        _pol = os.environ.get("PAIRCODER_SWITCH", "err1")
        _do_switch = True
        if _pol.startswith("err"):
            eta = int(_pol[3:] or "1"); _do_switch = (_consec >= eta)
            if _do_switch: _consec = 0
        elif _pol.startswith("fixed"):
            k = int(_pol[5:] or "1"); _do_switch = ((iters + 1) % k == 0)
        elif _pol == "none":
            _do_switch = False
        # --- Error-triggered role switch FIRST: the Navigator (who found the issue) takes the
        # keyboard; the fixer and the reviewer of that fix are always different agents.
        if _do_switch:
            drv, nav = nav, drv
            drv.history.append({"role": "system", "content": "ROLE SWITCH. " + DRIVER_PROMPT})
            nav.history.append({"role": "system", "content": "ROLE SWITCH. " + NAVIGATOR_PROMPT})
        # --- Driver phase: new Driver fixes per the review it just issued.
        # Shared environment (paper Sec. Components): the Driver sees the concrete tool
        # evidence too, not only the Navigator's prose.
        code = drv.request(_content(
            f"Follow the instructions below to fix errors in the code. Your answer only needs to "
            f"return the full code content without any extra text. The code to fix is "
            f"[{code[:8000]}]. The reviewer's feedback is [{(review or '')[:3000]}]."
            f"{evidence[:1500]}\nAnalyze the root cause of the failures above, then fix the code "
            f"accordingly.", image_b64)) or code
        iters += 1
    if not accepted and history:
        # Algorithm 1 line 19: return argmax_{C_i in M_tau} Quality(C_i).
        # Quality = (psi passes, continuous score if the bench provides one, recency).
        cands = list(history)
        final_ok = None; final_score = None
        if check is not None:
            try:
                _r = check(extract(code) if extract else code)
                final_ok = _r[0]; final_score = _r[2] if len(_r) > 2 else None
            except Exception:
                final_ok = None
        cands.append((code, final_ok, final_score, ""))
        def quality(item):
            i, (c, ok, sc, _) = item
            return (1 if ok else 0, sc if sc is not None else 0.0, i)
        code = max(enumerate(cands), key=quality)[1][0]
    return code, {"iters": iters, "accepted": accepted}


def _self_refine_solve(question, model, max_iters, check, sys_hint, extract,
                       image_b64, render_current):
    """Single-agent self-refinement at matched budget (no Navigator, no role switch)."""
    ag = Agent(model)
    ag.history.append({"role": "system", "content": DRIVER_PROMPT + (("\n" + sys_hint) if sys_hint else "")})
    code = ag.request(_content(question, image_b64))
    iters = 0; accepted = False; history = []
    while iters < max_iters:
        psi_ok = None; psi_score = None; evidence = ""
        if check is not None:
            try:
                _r = check(extract(code) if extract else code)
                psi_ok, ev = _r[0], _r[1]
                psi_score = _r[2] if len(_r) > 2 else None
                evidence = ("\nExecution/verification result: " +
                            ("PASSED (no errors detected by tools)." if psi_ok else f"FAILED: {ev}"))
                if psi_ok and ev:
                    evidence += f" ({ev})"
            except Exception as e:
                evidence = f"\nExecution/verification result: tool error: {str(e)[:200]}"
        cur_b64 = None
        if render_current is not None:
            try: cur_b64 = render_current(code)
            except Exception: cur_b64 = None
        _rev_imgs = ("\nThe FIRST image is the TARGET; the SECOND image is what your current code "
                     "renders. Compare them and fix concrete differences." if (image_b64 and cur_b64) else "")
        # The SAME agent reflects on the evidence and either accepts or fixes its own code.
        msg = (f"Here is the verification result for your current code.{evidence}{_rev_imgs}\n"
               f"If the code fully satisfies the task with no error, reply exactly [NOERROR]. "
               f"Otherwise, analyze the root cause and return the corrected full code only. "
               f"The task is [{question[:4000]}]. Your current code is [{code[:8000]}].")
        out = ag.request(_mm2(_content(msg, image_b64), cur_b64)) or ""
        history.append((code, psi_ok, psi_score, ""))
        if "NOERROR" in out:
            accepted = True; break
        code = (extract_codeblock(out) if False else out) or code
        iters += 1
    if not accepted and history:
        cands = list(history)
        final_ok = None; final_score = None
        if check is not None:
            try:
                _r = check(extract(code) if extract else code)
                final_ok = _r[0]; final_score = _r[2] if len(_r) > 2 else None
            except Exception: final_ok = None
        cands.append((code, final_ok, final_score, ""))
        def quality(item):
            i, (c, ok, sc, _) = item
            return (1 if ok else 0, sc if sc is not None else 0.0, i)
        code = max(enumerate(cands), key=quality)[1][0]
    return code, {"iters": iters, "accepted": accepted}


import atexit as _atexit2, json as _json2
@_atexit2.register
def _dump_tok_split():
    p = os.environ.get("PAIRCODER_TOKLOG")
    if p:
        try:
            with open(p.replace("tok.json", "tok_split.json"), "w") as f:
                _json2.dump(TOK_SPLIT, f)
        except Exception:
            pass
