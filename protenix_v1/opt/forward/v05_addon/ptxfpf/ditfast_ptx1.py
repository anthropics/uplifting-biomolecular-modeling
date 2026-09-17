"""ditfast_ptx1.py — the protenix 1.1.0 adapter of the fused diffusion-sampler package lib/protenix_fpf_ditfast (lever words `cond_dedupe`,
`dit_fused`, `dit_lowp`, `atom_fused`) and of the shared core's attention-with-pair-bias face for `atom_attn_exact` (opt_core.kernels.apb by the
`exact` tier word), of the kit arm (levers_ptx1.apply: `<trimul>[+lever...]`). It sets the env words the package reads (ENV) in-process from the
arm words (never asked of the caller), applies the dependency / card / slot rules BY NAME, and keeps the account levers_ptx1.describe() reads.

  cond_dedupe      DiffusionConditioning evaluated ONCE per denoiser step: protenix 1.1.0's sampler (model/generator.py sample_diffusion) hands the
                   denoiser the step's noise level as one scalar `.reshape(...,1).expand(..., N_sample)` — stride 0 over the samples (the graphed
                   step `sg` replays the same layout) — so stock computes the identical [N, c_s] conditioning once per sample and every consumer
                   downstream once per sample too. The package runs DiffusionConditioning.forward on t_hat[..., :1] when the sample dimension is
                   an expanded scalar (decided from strides, no device read) and returns s with a sample dimension of 1 that every stock consumer
                   broadcasts; a t_hat that is not an expanded scalar takes the stock rows (counted `stock_path`, never refused). TOLERANCE class
                   (the same values through GEMMs of another M: cuBLAS may pick another kernel / split), fast + big.
  dit_fused        the sampler's 24-block token DiffusionTransformer (diffusion_module.diffusion_transformer: n_heads 16, c_token 768, c_s 384,
                   transition n=2) as ONE fused forward: per block an AdaLN row kernel, ONE q|k|v|g GEMM, the fused pair-bias attention (= lever
                   ditattn's kernel and cell — REQUIRED: dit_fused without ditattn engaged installs nothing and is named), o GEMM, [gate*x + fp32
                   residual + next AdaLN] row kernel, ONE a1|a2 GEMM, SwiGLU row kernel, b GEMM, [gate + residual + the next block's AdaLN]
                   kernel; the conditioning side of all blocks as two GEMMs per call; the pair bias of all 24 blocks produced once per item into
                   the DiT hoist's slots `tok.fastbias<i>.float32` (recorded at the eager step, replayed inside the sampler graph) when `hoist` is
                   on, per call otherwise. TOLERANCE class (merged GEMMs, another reduction grouping), fast + big.
  dit_lowp         PRECISION lever riding dit_fused (word fp16; the package's PTX_DIT_LOWP): the a-path activations / GEMM operands and the
                   attention operands of the fused token stack in fp16 with fp32 accumulation, LayerNorm / softmax statistics and the residual
                   stream fp32. Without dit_fused it installs nothing and is named (rides_dit_fused).
  atom_fused       both 3-block atom transformers of the sampler (atom_attention_encoder / decoder .atom_transformer: c_atom 128, 4 heads x 32,
                   32x128 local windows) as fused stacks — entry AdaLN kernel, ONE q|g and ONE k|v GEMM per block, the fused local attention
                   (= lever atomattn's kernel — REQUIRED, as dit_fused requires ditattn), o GEMM, row kernels, ONE a1|a2 GEMM, SwiGLU, b GEMM;
                   every step-invariant operand (the per-block conditioning with sigmoid applied, the local pair bias from p) produced ONCE per
                   item through the hoist's slots. TOLERANCE class (TF32-order differences), fast + big.
  atom_attn_exact  EXACT class, exact tier only: the atom transformer's 32-query x 128-key local attention (AttentionPairBias.attention on the
                   3 + 3 atom blocks: 4 heads x 32, fp32) through the shared core's face by the `exact` TIER WORD — opt_core.kernels.apb
                   .select(cc, fp32, atom_h4d32w32x128, N, word="exact") names the row: a kernel row the core vouches byte-exact on this stack is
                   served through apb.atom_attention at those modules (q/k/v/g from the module's own Linears, the windowed pair bias read in
                   place, re-selected per call at the call's size); when the word names the stock statement (a row of class stock) nothing is
                   registered and the upstream op serves BY NAME (`exact_word:<row>`: bytes are --mode off's by construction). No kit cell table,
                   no card list: the face decides per card. Asked beside atomattn / atom_fused it steps aside BY NAME (slot_served_by).

Cards: the fused-stack package's cells are its CELLS.json rows; SERVED_CC is the served compute capabilities per word — a card outside it steps
aside BY NAME (`card_off`: the stock statement by design, exit 0). atom_attn_exact has no card list.
Order (levers_ptx1.apply): after the sampler graphs + hoist (`sg`, `hoist`) and after apb_ptx1 (ditattn / atomattn): the fused stacks strip the
hoist's per-module overrides on the modules they own and produce their step-invariant operands through the hoist's own slot protocol; inside
the graphed denoiser step they are capture-safe (no allocation-dependent host sync, no device read on the host: cond_dedupe's guard reads
strides; the stacks' per-call work is kernels + GEMMs on the step's tensors).
Census (describe()): per word `state` on | skipped | off, `reason`, `served` (Python-level calls the lever answered — inside the sampler graph a
forward is entered at the eager record step and at capture, replays do not re-enter Python), `gated` (the stock statement by the lever's own
rule: cond_dedupe's non-expanded t_hat, atom_attn_exact's envelope misses), `fallback` (an install error, or a dependency the caller withheld:
dit_fused without ditattn, dit_lowp without dit_fused, atom_fused without atomattn — evidence of a partial arm, report.partial_of), `card` /
`aside` (stepped aside by name: exit 0).
"""
from __future__ import annotations

import os
import sys
import time

LEVERS = ("cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact")     # the arm words this adapter answers (levers_ptx1.LEVER_NAMES carries them)
CLASS = {"cond_dedupe": "tolerance", "dit_fused": "tolerance", "dit_lowp": "precision", "atom_fused": "tolerance", "atom_attn_exact": "exact"}
SERVED_CC = {                                                   # per word: the compute capabilities ("major.minor") with a cell (the package's CELLS.json rows: sm_90)
    "cond_dedupe": ("9.0",), "dit_fused": ("9.0",), "dit_lowp": ("9.0",), "atom_fused": ("9.0",),
}                                                               # atom_attn_exact: no card list — opt_core.kernels.apb's exact word decides per card (ATOM_FACE)
REQUIRES = {"dit_fused": "ditattn", "dit_lowp": "dit_fused", "atom_fused": "atomattn"}   # rider -> the lever whose kernel / stack it rides (named, installs nothing without it)
SLOT_OWNERS = ("atomattn", "atom_fused")                        # the fast tiers' owners of primitives._local_attention: atom_attn_exact beside them steps aside by name
LOWP_WORD = "fp16"                                              # dit_lowp's precision word (the package reads PTX_DIT_LOWP = off | bf16 | fp16; the kit ships fp16)
ENV = {"dit_attn": "PTX_DIT_ATTN", "dit_attn_fp16": "PTX_DIT_ATTN_FP16", "dit_fast": "PTX_DIT_FAST", "dit_lowp": "PTX_DIT_LOWP",
       "atom_attn": "PTX_ATOM_ATTN", "atom_fast": "PTX_ATOM_FAST"}   # the package's env words, set in-process from the arm words
ATOM_FACE = {"cell": ("atom", 4, 32), "dtype": "fp32", "word": "exact", "samples": 5, "probe_tokens": 400}   # atom_attn_exact: opt_core.kernels.apb cell_word("atom", heads=4,
                                                                # head_dim=32) = atom_h4d32w32x128, fp32 operands, the EXACT tier word, 5 diffusion samples; probe_tokens = the N of the
                                                                # install-time census probe only (an exact row is vouched per stack, not per size; each call re-selects at its own size)
ATOM_COUNTS = {}                                                # atom_attn_exact's per-call census: kernel:<row> (served through the face), stock:<row> (the exact word named the
                                                                # stock statement at that size: the upstream op ran), original:<reason> (a call outside the windowed form)


class LeverRefused(RuntimeError):
    """A word that steps aside BY NAME (the stock statement serves by design; accounted, exit 0) — _install reads the class name."""


def _blank():
    return {"requested": False, "engaged": False, "card": None, "named": None, "dep": None, "error": None, "seconds": None, "report": {}}


STATE = {w: _blank() for w in LEVERS}
CARD = {"cc": None, "name": None}


def card_key():
    """("9.0", "sm_90") of device 0, (None, None) without CUDA."""
    if CARD["cc"] is None:
        try:
            import torch
            if torch.cuda.is_available():
                mj, mn = torch.cuda.get_device_capability()
                CARD.update(cc="%d.%d" % (mj, mn), name="sm_%d%d" % (mj, mn))
        except Exception:
            pass
    return CARD["cc"], CARD["name"]


def _say(msg):
    print("[ditfast_ptx1] " + msg, file=sys.stderr, flush=True)


def _apb_state():
    """apb_ptx1's account of ditattn / atomattn (engaged = installed on >= 1 module; fp16 = ditattnfp16 engaged) — {} when the adapter is absent."""
    m = sys.modules.get("apb_ptx1")
    if m is None:
        return {}
    st = getattr(m, "STATE", {}) or {}
    d, a = st.get("dit") or {}, st.get("atom") or {}
    return {"ditattn": bool(d.get("installed_on")), "ditattnfp16": bool(d.get("installed_on")) and bool(d.get("fp16")), "atomattn": bool(a.get("installed_on"))}


APB_ALIAS = {}                                                  # {"protenix_fpf_apb": "<module the name was registered on>"} once _alias_apb ran (describe() reports it)


def _alias_apb():
    """dit_fused / atom_fused import the attention levers' kernels by the package name `protenix_fpf_apb`. In this kit those
    kernels are the shared core's carried copy — opt_core.kernels.apb.fpf_apb, the package apb_ptx1 binds ditattn / ditattnfp16 / atomattn to
    (apb_ptx1.PACKAGE) — so the name is registered on THAT module object: the fused stacks then read the very
    STATE / CELLS the attention levers installed with (one package, two names), never a second copy. A `protenix_fpf_apb` importable on its
    own is left as it is. Returns the module or None (the words then refuse by name: requires lever dit_attn)."""
    if "protenix_fpf_apb" in sys.modules:
        return sys.modules["protenix_fpf_apb"]
    import importlib
    try:
        return importlib.import_module("protenix_fpf_apb")
    except ImportError:
        pass
    name = getattr(sys.modules.get("apb_ptx1"), "PACKAGE", None) or "opt_core.kernels.apb.fpf_apb"
    try:
        core = importlib.import_module(name)
    except ImportError:
        return None
    sys.modules["protenix_fpf_apb"] = core
    for sub in ("install", "apb_triton", "atom_triton", "pf_triton", "loadcheck"):
        try:
            sys.modules["protenix_fpf_apb." + sub] = importlib.import_module(name + "." + sub)
        except ImportError:
            pass
    APB_ALIAS["protenix_fpf_apb"] = name
    return core


def apb_face():
    """The shared core's attention-with-pair-bias face (opt_core.kernels.apb: select / atom_attention / describe / STOCK_ROWS) or None when this
    opt_core has no such face (then atom_attn_exact steps aside by name: no_apb_face)."""
    try:
        from opt_core.kernels import apb as APB
    except Exception:                                            # noqa: BLE001
        return None
    return APB if all(hasattr(APB, n) for n in ("select", "atom_attention", "describe", "STOCK_ROWS", "cell_word")) else None


def atom_exact_selection(APB, cc, n_tokens, capture=True):
    """The face's answer for the EXACT tier word at the atom-attention cell on this card at N tokens: (Selection, one-token description)."""
    kind, heads, hd = ATOM_FACE["cell"]
    try:
        stack = APB.stack_word()
    except Exception:                                            # noqa: BLE001  (no torch / no CUDA in this process: the table's reference stack)
        stack = None
    sel = APB.select(cc, ATOM_FACE["dtype"], APB.cell_word(kind, heads=heads, head_dim=hd), int(n_tokens), word=ATOM_FACE["word"], samples=ATOM_FACE["samples"],
                     capture=bool(capture), stack=stack, heads=heads, head_dim=hd)
    return sel, APB.describe(sel)


def _is_stock(APB, sel):
    return getattr(sel, "cls", None) == "stock" or getattr(sel, "row", None) in getattr(APB, "STOCK_ROWS", ())


def _wrap_atom_sites(model, APB, cc):
    """Serve the exact word's KERNEL row at the atom-attention modules (the 3 + 3 atom transformer blocks' AttentionPairBias.attention): per call the
    face re-selects at the call's own size (N ~ n_atoms / 8, the face's own estimate); a size at which the word names the stock statement runs the
    UPSTREAM op itself (never the face's reference — bytes stay --mode off's); a call outside the windowed form (no local bias, other dtype /
    heads) runs the upstream op, counted by reason (apb_ptx1.atom_route's words). Returns the number of modules wrapped."""
    import math
    import apb_ptx1 as AP                                         # the kit's atom-attention site knowledge (module finder + call-form router), shared with lever atomattn
    sites, why = AP._site_modules(model, "atom")
    if not sites:
        raise LeverRefused("no_atom_sites:%s" % (why or "none found"))
    cache = {}

    def make(att, orig):
        H = int(att.num_heads)

        def forward(q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
            miss = AP.atom_route(tuple(q_x.shape), tuple(kv_x.shape), None if attn_bias is None else tuple(attn_bias.shape),
                                 None if trunked_attn_bias is None else tuple(trunked_attn_bias.shape), H, n_queries, n_keys,
                                 dtype=str(q_x.dtype).rsplit(".", 1)[-1], is_cuda=bool(q_x.is_cuda))
            if miss is None:
                N = int(q_x.shape[-2]); S = max(1, int(q_x.numel() // (N * q_x.shape[-1])))
                sel = cache.get((N, S))
                if sel is None:
                    sel = cache[(N, S)] = APB.select(cc, ATOM_FACE["dtype"], APB.cell_word(*ATOM_FACE["cell"][:1], heads=H, head_dim=int(att.linear_q.out_features) // H),
                                                     max(1, N // 8), word=ATOM_FACE["word"], samples=S, capture=True, stack=APB.stack_word(q_x.device), heads=H,
                                                     head_dim=int(att.linear_q.out_features) // H)
                if _is_stock(APB, sel):
                    miss = "stock:%s" % sel.row
            if miss is not None:                                 # the upstream statement itself: outside the windowed form, or the exact word names stock at this size
                ATOM_COUNTS[miss if miss.startswith("stock:") else "original:%s" % miss] = ATOM_COUNTS.get(miss if miss.startswith("stock:") else "original:%s" % miss, 0) + 1
                return orig(q_x, kv_x, attn_bias=attn_bias, trunked_attn_bias=trunked_attn_bias, n_queries=n_queries, n_keys=n_keys, inf=inf,
                            inplace_safe=inplace_safe, chunk_size=chunk_size)
            q = att.linear_q(q_x); k = att.linear_k(kv_x); v = att.linear_v(kv_x)
            g = att.linear_g(q_x) if att.gating else None
            C = q.shape[-1]; D = C // H; lead = q.shape[:-2]; N = q.shape[-2]
            q4, k4, v4 = (t.reshape(-1, N, H, D) for t in (q, k, v))
            g4 = g.reshape(-1, N, H, D) if g is not None else None
            b = trunked_attn_bias
            while b.dim() > 4:
                b = b[0]
            if b.stride(-1) != 1:
                b = b.contiguous()
            o, _ = APB.atom_attention(q4, k4, v4, b, g4, word=ATOM_FACE["word"], selection=sel, n_queries=int(n_queries), n_keys=int(n_keys),
                                      scale=1.0 / math.sqrt(D), capture=True, out_dtype=q.dtype)
            ATOM_COUNTS["kernel:%s" % sel.row] = ATOM_COUNTS.get("kernel:%s" % sel.row, 0) + 1
            return att.linear_o(o.view(*lead, N, C))

        forward.__wrapped__ = orig; forward._ditfast_ptx1 = "atom_attn_exact"
        return forward

    n = 0
    for _name, att in sites:
        cur = att.__dict__.get("forward")
        if getattr(cur, "_ditfast_ptx1", None) == "atom_attn_exact":
            n += 1; continue
        if cur is not None and (getattr(cur, "_apb_ptx1", None) or getattr(cur, "__wrapped__", None)):
            raise LeverRefused("slot_served_by:%s" % (getattr(cur, "_apb_ptx1", None) or "another_patch"))   # another lever owns the module's forward: never stacked
        att.forward = make(att, att.forward); n += 1
    return n



def install_atom_exact_by_word(model):
    """atom_attn_exact: ask the shared face for the EXACT tier word's row at the atom-attention cell on this card (install-time probe at
    ATOM_FACE["probe_tokens"]); a stock-class answer installs nothing — the word steps aside BY NAME `exact_word:<row>` and the upstream op serves
    (bytes = --mode off by construction); a kernel row (vouched byte-exact on this stack by the core) is served through apb.atom_attention at the
    atom-attention modules, re-selected per call at the call's size. -> the install report (installed, row, apb, modules)."""
    st = STATE["atom_attn_exact"]
    APB = apb_face()
    cc, _ccname = card_key()
    sel, token = (None, None)
    if APB is not None:                                          # (1) the face's EXACT tier word at the atom cell: a byte-vouched kernel row is served through the face
        sel, token = atom_exact_selection(APB, cc, ATOM_FACE["probe_tokens"])
        st["apb"] = token; st["row"] = getattr(sel, "row", None)
        if not _is_stock(APB, sel):
            n = _wrap_atom_sites(model, APB, cc)
            return {"installed": bool(n), "source": "opt_core.kernels.apb", "row": sel.row, "apb": token, "modules": n}
    if APB is None:                                              # (2) nothing else: the upstream op serves (bytes = --mode off by construction)
        raise LeverRefused("no_apb_face(opt_core.kernels.apb)")
    raise LeverRefused("exact_word:%s" % sel.row)


def _install(word, fn, model):
    """Run a package install; LeverRefused -> named (the lever steps aside by name), any other exception -> error (fallback evidence)."""
    st = STATE[word]
    t0 = time.time()
    try:
        rep = fn(model)
        st["report"] = dict(rep or {})
        st["engaged"] = bool((rep or {}).get("installed"))
        if not st["engaged"] and not st["named"]:
            st["named"] = "install_returned_not_installed"
    except Exception as e:                                   # noqa: BLE001
        refused = type(e).__name__ == "LeverRefused"
        if refused:
            st["named"] = str(e)[:300]
            _say("%s stepped aside by name: %s" % (word, st["named"]))
        else:
            st["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
            _say("%s install error (the stock statement serves; evidence of a partial arm): %s" % (word, st["error"]))
    st["seconds"] = round(time.time() - t0, 2)
    return st


def apply(model, cond_dedupe=False, dit_fused=False, dit_lowp=False, atom_fused=False, atom_attn_exact=False) -> dict:
    """Install the requested words on `model` (the runner's Protenix module) in LEVERS order: cond_dedupe, dit_fused (+ dit_lowp,
    its precision word), atom_fused, atom_attn_exact. Idempotent per process (a word already engaged is left as it is)."""
    want = {"cond_dedupe": bool(cond_dedupe), "dit_fused": bool(dit_fused), "dit_lowp": bool(dit_lowp), "atom_fused": bool(atom_fused),
            "atom_attn_exact": bool(atom_attn_exact)}
    for w, on in want.items():
        STATE[w]["requested"] = on
    if not any(want.values()):
        return describe()
    cc, ccname = card_key()
    apb = _apb_state()
    for w in LEVERS:
        st = STATE[w]
        if not st["requested"] or st["engaged"]:
            continue
        if cc is None:
            st["named"] = "no_cuda"; continue
        if w in SERVED_CC and cc not in SERVED_CC[w]:           # no cell for this card: the lever steps aside BY NAME (the stock statement by design)
            st["card"] = "%s: no %s cell (cells: %s)" % (ccname, w, ",".join("sm_" + c.replace(".", "") for c in SERVED_CC[w])); continue
        base = REQUIRES.get(w)
        if base in ("ditattn", "atomattn") and not apb.get(base):
            st["dep"] = "requires:%s(%s)" % (base, "not engaged" if sys.modules.get("apb_ptx1") else "not in the arm"); continue
        if w == "dit_lowp":                                      # rides dit_fused (LEVERS order: decided after it): engaged there when the package took the word
            fused = STATE["dit_fused"]
            if not want["dit_fused"]:
                st["dep"] = "rides_dit_fused(not in the arm)"
            elif not fused["engaged"] and not (st["dep"] or st["error"]):
                st["dep"] = "rides_dit_fused(%s)" % ("stepped aside" if fused["named"] else "not engaged")
            continue
        if w == "atom_attn_exact":
            owners = [o for o in SLOT_OWNERS if (o == "atomattn" and apb.get("atomattn")) or (o == "atom_fused" and want["atom_fused"])]
            if owners:
                st["named"] = "slot_served_by:%s" % owners[0]; continue
            _install(w, install_atom_exact_by_word, model)      # the shared face's exact tier word at the atom-attention call form (below)
        elif w == "cond_dedupe":
            from protenix_fpf_ditfast import install_cond_dedupe as fn
            _install(w, fn, model)
        elif w == "dit_fused":
            _alias_apb()                                         # the attention kernels' package under the name the vendored stack imports (see _alias_apb)
            os.environ[ENV["dit_attn"]] = "1"
            if apb.get("ditattnfp16"):
                os.environ[ENV["dit_attn_fp16"]] = "1"          # the fused stack's attention takes ditattn's fp16 cell when ditattnfp16 is engaged (fpf_apb.install._cell)
            os.environ[ENV["dit_fast"]] = "1"
            os.environ[ENV["dit_lowp"]] = LOWP_WORD if want["dit_lowp"] else "off"
            from protenix_fpf_ditfast import install_dit_fast as fn
            _install(w, fn, model)
            lp = STATE["dit_lowp"]
            if want["dit_lowp"] and st["engaged"]:
                if str(st["report"].get("lowp")) == LOWP_WORD:
                    lp["engaged"] = True; lp["report"] = {"word": LOWP_WORD, "act": st["report"].get("act")}
                else:
                    lp["error"] = "dit_fused engaged with lowp=%s" % st["report"].get("lowp")
        elif w == "atom_fused":
            _alias_apb()
            os.environ[ENV["atom_attn"]] = "1"
            os.environ[ENV["atom_fast"]] = "1"
            from protenix_fpf_ditfast import install_atom_fast as fn
            _install(w, fn, model)
    on = [w for w in LEVERS if STATE[w]["engaged"]]
    off = ["%s(%s)" % (w, STATE[w]["card"] or STATE[w]["named"] or STATE[w]["dep"] or STATE[w]["error"]) for w in LEVERS if STATE[w]["requested"] and not STATE[w]["engaged"]]
    _say("card=%s engaged=%s not_engaged=%s" % (ccname, ",".join(on) or "-", ";".join(off) or "-"))
    return describe()


def _exits():
    try:
        from protenix_fpf_ditfast import report
        return report() or {}
    except Exception:
        return {}


def word_account(w) -> dict:
    """One word's census: state on|skipped|off, reason, served, gated, fallback, facts, card / aside (report.kit_evidence's shapes). A word
    engaged but idle BY DESIGN on the run's inputs — `cond_dedupe` when every denoiser call carried one sample row (N_sample == 1: nothing to
    de-duplicate) — reads skipped / aside=n_sample:1 (accounted, exit 0), like a package refusal by name; served 0 otherwise stays served 0."""
    st = STATE[w]
    out = {"requested": st["requested"], "engaged": st["engaged"], "class": CLASS[w], "served": 0, "gated": {}, "fallback": {}, "facts": {},
           "card": None, "aside": None, "state": "off", "reason": None, "seconds": st["seconds"]}
    if not st["requested"]:
        return out
    if st["card"]:
        out.update(state="skipped", reason="card_off", card=st["card"]); return out
    if st["named"] and not st["engaged"]:
        out.update(state="skipped", reason="aside", aside=st["named"])
        if w == "atom_attn_exact" and st.get("apb"):                # the face's answer behind the aside (row / cell / class), for the LEVER line
            out["facts"] = {"row": _fact_token(st.get("row")), "apb": _fact_token(st.get("apb"))}
        return out
    if st["dep"]:
        out.update(state="skipped", reason=st["dep"].split("(")[0], fallback={"requires": st["dep"]}); return out
    if st["error"]:
        out.update(state="skipped", reason="fallback", fallback={"install": st["error"]}); return out
    if not st["engaged"]:
        out.update(state="skipped", reason="not_installed", fallback={"install": "not installed"}); return out
    ex = _exits()
    rep = st["report"]
    idle = None                                                  # a by-design idle state decided from the run's own census (below): the `aside` word, or None
    if w == "cond_dedupe":
        c = dict((rep.get("census") or {}))
        try:
            from protenix_fpf_ditfast import cond_dedupe as C
            c = dict(C.REPORT.get("census") or c)
        except Exception:
            pass
        out["served"] = int(c.get("dedupe_calls") or 0)
        sp, rows_in = int(c.get("stock_path_calls") or 0), int(c.get("rows_in") or 0)
        if sp:
            out["gated"] = {"stock_path": sp}
        out["facts"] = {"rows_in": rows_in, "rows_computed": int(c.get("rows_computed") or 0), "guard": "stride0"}
        if not out["served"] and sp and rows_in == sp:          # every call carried ONE sample row (N_sample == 1: `--sample 1`, a reach probe): there is nothing to
            idle = "n_sample:1"                                  # de-duplicate BY DESIGN — the word steps aside by name for this run (see the end of this function)
    elif w == "dit_fused":
        e = ex.get("dit_fused") or {}
        out["served"] = int(e.get("stack_calls") or 0)
        out["facts"] = {"blocks": rep.get("blocks"), "act": str(rep.get("act")), "lowp": rep.get("lowp"), "attention": rep.get("attention"),
                        "bias_slots": e.get("bias_slots") or rep.get("bias_slots"), "packed_mb": rep.get("packed_param_MB")}
    elif w == "dit_lowp":
        e = ex.get("dit_fused") or {}
        out["served"] = int(e.get("stack_calls") or 0)
        out["facts"] = {"word": LOWP_WORD, "act": str(rep.get("act")), "rides": "dit_fused"}
    elif w == "atom_fused":
        e = ex.get("atom_fused") or {}
        calls = e.get("stack_calls") or {}
        out["served"] = int(sum(int(v) for v in calls.values())) if isinstance(calls, dict) else int(calls or 0)
        stacks = rep.get("stacks") or {}
        out["facts"] = {"enc_blocks": (stacks.get("enc") or {}).get("blocks"), "dec_blocks": (stacks.get("dec") or {}).get("blocks"),
                        "act": str(rep.get("act")), "attention": rep.get("attention"), "cond_slots": rep.get("cond_slots")}
    elif w == "atom_attn_exact":                                 # served = the calls the kernel answered (the face's row, or the moved op at the upstream site); gated = the
        served, gated = 0, {}                                    # upstream op by rule: stock:<row> (the exact word named stock at that size), original:<reason> (outside
        for k2, v2 in sorted(ATOM_COUNTS.items()):
            if k2.startswith("kernel:"):
                served += int(v2)
            else:
                gated[k2] = int(v2)
        out["facts"] = {"source": "apb:%s" % st.get("row"), "row": st.get("row"), "apb": st.get("apb"), "modules": rep.get("modules")}
        out["served"] = served
        if gated:
            out["gated"] = gated
    out["facts"] = {k: _fact_token(v) for k, v in out["facts"].items() if v is not None}
    if idle:                                                     # engaged, yet idle by design on this run's inputs (cond_dedupe at N_sample == 1): accounted, exit 0 —
        out.update(state="skipped", reason="aside", aside=idle)  # report.partial_of / lever_lines read `aside` as stepped aside by name, never as a partial activation;
        return out                                               # served 0 with multi-sample rows on the stock path (rows_in > calls) stays served 0: a real failure to engage
    out["state"] = "on"
    return out


def _fact_token(v):
    """A LEVER-line value: dicts as k:v,... and sequences as a,b (no blanks; report._token replaces any that remain)."""
    if isinstance(v, dict):
        return ",".join("%s:%s" % (k, _fact_token(x)) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))) or "-"
    if isinstance(v, (list, tuple, set)):
        return ",".join(str(_fact_token(x)) for x in v) or "-"
    return str(v).replace(" ", "") if not isinstance(v, (int, float, bool)) else v


def describe() -> dict:
    cc, ccname = card_key()
    vers = {}
    for pkg in ("protenix_fpf_ditfast",):
        m = sys.modules.get(pkg)
        vers[pkg] = getattr(m, "__version__", None) if m else None
    APB = apb_face()
    vers["opt_core.kernels.apb"] = "present" if APB is not None else None
    return {"card": ccname, "versions": vers, "served_cc": {w: list(v) for w, v in SERVED_CC.items()}, "apb_package": APB_ALIAS.get("protenix_fpf_apb"), "words": {w: word_account(w) for w in LEVERS}}
