# odde_arm_t (the ARM-T add-on, levers/ARMT) -- trunk arms for OpenDDE: ARM U (triangle attention through the core provider + the TriMul site through the core provider +
# Pairformer/MSA pair stacks under bf16 autocast; tier 2, numerics-changing) and ARM Z (the transpose-free pair update alone; exact). Run-time
# monkeypatches composed with the kit's other levers by the served-levers hook; nothing under site-packages is edited.
"""odde_arm_t -- OpenDDE trunk arms: U (tier 2, run-to-run deterministic; NEVER 'EXACT') and Z (exact).

Switches (all default OFF; read once at import of this module):
  ODDE_ARM_U=1                 install ARM U (the fast line's arm): triangle attention through the core provider (ODDE_TRIATTN, the sibling unit odde_triattn_bind) + the TriMul site bound (see ODDE_ARM_U2_TRIMUL / ODDE_TRIMUL)
                               + model.pairformer_stack and every MSA-module pair_stack run inside
                               torch.autocast('cuda', bfloat16) (bf16 trunk numerics): attention/transition/block GEMMs bf16 with fp32 accumulation,
                               outputs cast back to fp32 for every consumer. TIER-2b bf16 trunk.
  ODDE_ARM_U2_TRIMUL=1         (U2) with ODDE_ARM_U=1: the arm's TriMul route — every admitted TriangleMultiplicativeUpdate call (design gate, scope, rows > 100)
                               is handed to the core TriMul provider by the tier word ODDE_TRIMUL (sibling unit odde_trimul_bind: the measured row per
                               (cc, precision, c_z, c_hidden, N bucket, direction) cell); a call the provider hands back (measured stock winner, refusal by
                               name) and every call without the word run the STOCK TriMul by name, counted. The arm carries no TriMul kernels of its own.
  ODDE_ARM_T_MIN_TOKENS=300    design-level size gate (N_token of the prediction, recorded at get_pairformer_output entry): below it every kernel call is stock, so
                               ARM U's bf16 autocast region stays active below the gate (kernels stock).
  ODDE_ARM_T_SCOPE=trunk       trunk (default): the arm binds model.pairformer_stack (48 blocks), the MSA-module pair stacks and the template embedder's pairformer:
                               kernels, and under U the bf16 region / transpose-free block / U3 prologue+transition, act there only; structural_token_refiner (4-block
                               PairformerStack over the structural tokens) and the confidence head's 4-block stack keep the stock kernels and dtype.
                               trunk+refiner | trunk+confidence | all (= trunk+refiner+confidence): the same binds extended to model.structural_token_refiner and/or
                               model.confidence_head.pairformer_stack (a named module that is absent raises at bind: never a silent narrower scope); under `all` the
                               kernels also serve any pair-track module outside the binds.
  ODDE_ARM_T_TRIMUL=fast|stock (stock = the TriMul site left unbound)
  ODDE_ARM_T_VERBOSE=1 prints refusals.
  ODDE_TRIATTN=fast|exact|big|<row>  the attention site served through the core's triangle-attention provider (opt_core.kernels.triattn) by the sibling
                               unit odde_triattn_bind (levers/ARMT/odde_triattn_bind.py): under either arm the wrapper below is installed and every call it
                               admits (design gate, scope, dtype, rank) is the provider's -- the tier word straight to its select(), the row per cell its
                               decision, a refusal its named fallback, a stock cell the stock op by name. Unset = the attention site keeps the stock op
                               under both arms (attention=stock); the arm carries no attention kernel of its own.

HAZARD: OpenDDE runs predict() under torch.autocast('cuda', dtype=<--dtype>) even for fp32, i.e. autocast is ENABLED with
dtype float32. Any rule of the form 'if autocast is enabled, use the autocast dtype' resolves to fp32 there. This module treats autocast-fp32 as 'no low-precision
autocast'; the bf16 boundary is taken only inside an explicit bf16 region (ARM U). The census key att_dtypes records what actually ran
('float32->float32->float32' = fp32 boundary outside a bf16 region; 'bfloat16->bfloat16->bfloat16' = inside U).

What is replaced (only on the cuEq code path, only for designs at/above the gate, only in scope, only on cross-checked cells; everything else -> stock, counted):
  (a) opendde.model.triangular.layers.cuequivariance_triangular_attn -> the core provider's row for the call's cell (ODDE_TRIATTN=<tier word>; see above):
      deterministic rows, fp32 softmax statistics and accumulation, output in the caller's dtype; the exact tier's rows are bitwise the stock op.
  (b) opendde.model.triangular.triangular.TriangleMultiplicativeUpdate.forward -> the core TriMul provider (opt_core.kernels.trimul) by tier word through
      odde_trimul_bind.serve (U2); the stock forward by name for every call the provider does not serve.
  (c) ARM U: forward hooks put model.pairformer_stack and MSABlock.pair_stack inside torch.autocast('cuda', bfloat16) and cast their outputs back to float32.
Census: odde_arm_t.COUNTS (calls served per kernel / refused per reason / dtypes seen / designs seen), printed at exit as '[odde_arm_t] COUNTS {...}'.
"""
from __future__ import annotations
from opt_core.oom import is_oom                      # an out-of-memory error is re-raised before any reroute below (the core's one classifier)
import atexit, json, os, sys, threading, time

__version__ = "0.3"
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # levers/ARMT/ (the add-on root: sitecustomize.py, the bind units, third_party/)
_TP = os.path.join(_ROOT, "third_party")


def _env():
    return dict(
        MIN_TOKENS=int(os.environ.get("ODDE_ARM_T_MIN_TOKENS", "300") or 0),
        TRIMUL_MODE=os.environ.get("ODDE_ARM_T_TRIMUL", "fast"),
        SCOPE=os.environ.get("ODDE_ARM_T_SCOPE", "trunk"),                # trunk | trunk+refiner | trunk+confidence | all  (parsed by _scope_ext)
        U2_TRIMUL=os.environ.get("ODDE_ARM_U2_TRIMUL", "0") not in ("", "0"),   # U2: the TriMul site through the core provider by the tier word (odde_trimul_bind); 0 = the site bound, every call the stock forward by name
        ZT=os.environ.get("ODDE_ARM_ZT", "1") not in ("", "0"),      # transpose-free PairformerBlock pair update, EXACT (same arithmetic as the base arm), ON inside every installed arm; ODDE_ARM_ZT=0 opts out
        Z_ALONE=os.environ.get("ODDE_ARM_Z", "0") not in ("", "0"),   # ODDE_ARM_Z=1 with no U flag = arm Z = base arm + transpose-free update only (EXACT tier)
        U3=os.environ.get("ODDE_ARM_U3", "1") not in ("", "0"),      # ON under U (U3: cast-once tri-attention prologue + the transition binding under ODDE_TRANSITION; byte-identical to U without it); ODDE_ARM_U3=0 = U without U3
        U_TRANSITION=os.environ.get("ODDE_U_TRANSITION", "1") not in ("", "0"),   # under U3: the pair-transition binding may apply (it does only with ODDE_TRANSITION live: lever transition_core | transition_exact); =0 keeps the engine's module under U
        VERBOSE=os.environ.get("ODDE_ARM_T_VERBOSE", "0") not in ("", "0"),
    )


CFG = _env()
_read_cfg = _env      # public alias: re-read the flag defaults from the environment (used by the test driver to test the shipped env path)
SCOPE = CFG["SCOPE"]
U2_TRIMUL = CFG["U2_TRIMUL"]
ZT = CFG["ZT"]



def arch_key(device=None):
    """'sm_<major><minor>' of the CUDA device (None without CUDA)."""
    import torch
    if not torch.cuda.is_available():
        return None
    mj, mn = torch.cuda.get_device_capability(0 if device is None else device)
    return "sm_%d%d" % (mj, mn)


# design-level gate state: N_token of the CURRENT prediction, recorded at OpenDDE.get_pairformer_output entry (class-level wrap installed by install()).
# The size gate compares THIS number (not the per-call row count) with MIN_TOKENS, so a design below the gate is served 100 % by stock kernels,
# including the structural-token refiner whose pair tensor is ~7x larger than N_token.
STATE = {"n_token": None, "in_scope": 0, "in_conf": 0}      # in_conf: inside model.confidence_head.pairformer_stack (lever triattn_conf: odde_triattn_bind.CONF; att_conf_calls / att_conf_prov_calls count that stack's calls)
COUNTS = {"version": __version__, "installed": False, "arm": None, "min_tokens": CFG["MIN_TOKENS"], "att_mode": None, "trimul_mode": CFG["TRIMUL_MODE"], "scope": CFG["SCOPE"], "u2_trimul": CFG["U2_TRIMUL"], "u2_mode": "provider", "zt": CFG["ZT"], "u3": CFG["U3"], "trimul_bf16_calls": 0, "zt_block_calls": 0, "zt_block_reached": 0, "castonce_calls": 0, "castonce_stock_calls": 0, "fpf_transition": None,
          "att_prov_calls": 0, "att_conf_calls": 0, "att_conf_prov_calls": 0, "trimul_prov_calls": 0, "trimul_prov_asides": {}, "att_stock_calls": 0, "att_stock_reasons": {}, "att_dtypes": {}, "trimul_fast_calls": 0, "trimul_stock_calls": 0,
          "trimul_stock_reasons": {}, "trimul_admission": {"checked": 0, "refused": 0, "first_refusal": None, "max_needed_gib": 0.0},
          "bf16_stack_calls": 0, "designs_seen": {}, "errors": {}, "device": None}
_LOCK = threading.Lock()
_ORIG = {}            # key -> (owner object, attr, original value)
_ZT_IDS = set()       # ids of in-scope PairformerBlocks served by the transpose-free forward (U2c / arm Z)
_U3_TR = []           # pair_transition modules bound to fpf_transition_odde (U3); restored by remove()
_TB = {"mod": False}
_TMB = {"mod": False}              # odde_trimul_bind (the TriMul provider binding) once imported: module | None


def _trimul_bind():
    """The sibling unit odde_trimul_bind when its word (ODDE_TRIMUL) is set, else None: the TriMul site's provider binding. Imported once."""
    if _TMB["mod"] is False:
        _TMB["mod"] = None
        if os.environ.get("ODDE_TRIMUL", "").strip().lower() not in ("", "0", "off", "none", "stock"):
            import importlib
            m = importlib.import_module("odde_trimul_bind")       # levers/ARMT is this package's own directory on the path
            _TMB["mod"] = m if m.active() else None
            COUNTS["trimul_word"] = getattr(m, "WORD", None)
    return _TMB["mod"]


def _triattn_bind():
    """The sibling unit odde_triattn_bind when its word (ODDE_TRIATTN) is set, else None: the attention site's provider binding. Imported once."""
    if _TB["mod"] is False:
        _TB["mod"] = None
        if os.environ.get("ODDE_TRIATTN", "").strip().lower() not in ("", "0", "off", "none", "stock"):
            import importlib
            m = importlib.import_module("odde_triattn_bind")      # levers/ARMT is this package's own directory on the path
            _TB["mod"] = m if m.active() else None
            COUNTS["triattn_word"] = getattr(m, "WORD", None)
    return _TB["mod"]


def _log(msg):
    print(f"[odde_arm_t] {msg}", file=sys.stderr, flush=True)


def _bump(key, reason=None, n=1):
    with _LOCK:
        if reason is None:
            COUNTS[key] += n
        else:
            d = COUNTS[key]; d[reason] = d.get(reason, 0) + n


# ---- TriMul stock-call reasons that are NOT the lever's declared domain (the declared ones: design gate, rows <= 100, torch path, scope, the provider's named
# asides): a kernel exception, plus the two admission words kept as census vocabulary for the LEVER line's readers (nothing in this module sizes operand planes,
# so nothing here emits them). Counted per call under trimul_stock_reasons[<reason>] like every stock call; opendde_opt reads these tuples (see the two below).
DEGRADED_TRIMUL_REASONS = ("exception", "planes_exceed_free_memory", "free_memory_probe_unavailable")
ADMISSION_TRIMUL_REASONS = ("planes_exceed_free_memory", "free_memory_probe_unavailable")   # the U2 admission gate's refusals: that call runs the stock TriMul BY RULE, named and
                                                                                             # counted (trimul_admission=<reason>:<n> on the arm's LEVER line) — a step-aside, the run complete
FALLBACK_TRIMUL_REASONS = ("exception",)                                                      # a kernel exception: a per-call FALLBACK by name — the run is PARTIAL
TESTED_CC = {(9, 0): "H100 80GB HBM3", (8, 0): "A100-SXM4-80GB"}   # devices (compute capability) with a test record of this add-on; install() warns by name on any other



def warn_trimul_exception_once(e):
    """One stderr line at the FIRST TriMul kernel exception of the process (a defect signal: every occurrence is still counted under
    trimul_stock_reasons.exception and that call runs the stock TriMul)."""
    with _LOCK:
        if STATE.get("trimul_exception_warned"):
            return False
        STATE["trimul_exception_warned"] = True
    _log(trimul_exception_line(e))
    return True


def trimul_exception_line(e):
    msg = " ".join(str(e).split())[:200]
    return (f"TRIMUL-EXCEPTION {type(e).__name__}: {msg} "
            "(printed once; every kernel exception is counted under trimul_stock_reasons.exception and that call runs the stock TriMul - census at exit)")


def degraded_trimul():
    """{reason: n} over DEGRADED_TRIMUL_REASONS for this process (empty = no degraded TriMul call)."""
    with _LOCK:
        d = dict(COUNTS.get("trimul_stock_reasons") or {})
    return {r: int(d[r]) for r in DEGRADED_TRIMUL_REASONS if d.get(r)}


def admission_asides():
    """{reason: n} over ADMISSION_TRIMUL_REASONS for this process (empty = the admission gate refused no call): named step-asides, not fallbacks."""
    with _LOCK:
        d = dict(COUNTS.get("trimul_stock_reasons") or {})
    return {r: int(d[r]) for r in ADMISSION_TRIMUL_REASONS if d.get(r)}


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------- third-party loading


# ----------------------------------------------------------------------------------------------------------------- (a) triangle attention
def make_attn(orig, min_tokens=None, att_mode=None, gate_design=True):
    """Build the replacement for layers.cuequivariance_triangular_attn(q, k, v, bias, mask, scale) -> tuple (caller takes [0]).
    gate_design=True (in-model install): the provider serves only when the current design's N_token >= min_tokens AND the calling module is in scope.
    gate_design=False (op-level driver): serve every call of a design.  Without the provider word (ODDE_TRIATTN unset) every call is the stock op's, counted."""
    import torch
    TB = _triattn_bind()                                                       # the core provider binding (ODDE_TRIATTN set) or None (attention=stock: install() does not bind the site then)
    min_tokens = CFG["MIN_TOKENS"] if min_tokens is None else int(min_tokens)

    def _stock5(q_, k_, v_, b_, mask=None, scale=None):                       # the stock op on 5-D tensors (cuequivariance signature) for the provider rows that call it per head
        r = orig(q_, k_, v_, b_, mask, scale)
        return r[0] if isinstance(r, tuple) else r
    verbose = CFG["VERBOSE"]

    def _stock(reason, q, k, v, bias, mask, scale):
        _bump("att_stock_calls"); _bump("att_stock_reasons", reason)
        return orig(q, k, v, bias, mask, scale)

    def cuequivariance_triangular_attn(q, k, v, bias, mask, scale):
        # q/k/v: [*, H, S, D]; bias: [*, 1, H, S, S] fp32; mask: bool [*, S, 1, 1, S] (True = keep); returns a 1-tuple like the stock wrapper's [0] use
        try:
            SQ, D = int(q.shape[-2]), int(q.shape[-1])
            nt = STATE["n_token"]
            if gate_design and (nt is None or nt < min_tokens):
                return _stock(f"design_below_gate_{min_tokens}" if nt is not None else "no_design_context", q, k, v, bias, mask, scale)
            if STATE["in_conf"] > 0 and TB is not None and TB.CONF == "stock":   # lever triattn_conf left out: the confidence head's stack is the stock op's,
                _bump("att_conf_calls"); TB.note_aside("conf_stock")              # by name, before any of the wrapper's own work (its prep costs the fp32
                return _stock("conf_stock", q, k, v, bias, mask, scale)           # confidence pass at 1200 rows more than the rows save there)
            if gate_design and SCOPE != "all" and STATE["in_scope"] <= 0:
                return _stock("out_of_scope_module", q, k, v, bias, mask, scale)
            if TB is None:                                            # no provider word: the stock op (install() binds the site only with the word; the op-level driver may not)
                return _stock("no_provider_word", q, k, v, bias, mask, scale)
            if D not in (16, 32, 64, 128) or not q.is_cuda:
                return _stock(f"cell_D{D}_cuda{int(q.is_cuda)}", q, k, v, bias, mask, scale)
            if q.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                return _stock(f"dtype_{q.dtype}", q, k, v, bias, mask, scale)
            if q.dim() == 4:
                q5, k5, v5 = q[None], k[None], v[None]
                b5 = bias.reshape((1,) * (5 - bias.dim()) + tuple(bias.shape)) if bias.dim() < 5 else bias
                m5 = None if mask is None else (mask.reshape((1,) * (5 - mask.dim()) + tuple(mask.shape)) if mask.dim() < 5 else mask)
            elif q.dim() == 5:
                q5, k5, v5, b5, m5 = q, k, v, bias, mask
            else:
                return _stock(f"rank_{q.dim()}", q, k, v, bias, mask, scale)
            if b5.dim() != 5 or (m5 is not None and m5.dim() != 5):
                return _stock("bias_or_mask_rank", q, k, v, bias, mask, scale)
            in_dt = q5.dtype
            ac_dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else None
            if ac_dt is not None and ac_dt != torch.float32:         # ARM U region / --dtype bf16: behave exactly like cuEq under autocast (cast to the autocast dtype)
                cdt = ac_dt; odt = cdt
            else:                                                     # outside a bf16 region: OpenDDE predict() runs under autocast(dtype=float32) ->
                cdt = in_dt; odt = in_dt                              # fp32 inputs, tl.dot input_precision tf32 (same TF32 class as stock cuEq)
            _bump("att_dtypes", f"{str(in_dt).replace('torch.', '')}->{str(cdt).replace('torch.', '')}->{str(odt).replace('torch.', '')}")
            if q5.dtype != cdt:
                q5 = q5.to(cdt); k5 = k5.to(cdt); v5 = v5.to(cdt)
            if b5.dtype != torch.float32:
                b5 = b5.float()
            try:                                                      # ODDE_TRIATTN: the provider's row for this cell (or the stock op by name: Aside)
                with torch.autocast("cuda", enabled=False):
                    in_conf = STATE["in_conf"] > 0
                    if in_conf:
                        _bump("att_conf_calls")
                    o = TB.serve(q5, k5, v5, b5, m5, scale, stock5=_stock5)
                    if in_conf:
                        _bump("att_conf_prov_calls")
            except TB.Aside as a:
                TB.note_aside(a.reason)
                return _stock(f"prov_{a.reason}", q, k, v, bias, mask, scale)
            if q.dim() == 4:
                o = o[0]
            _bump("att_prov_calls")
            return (o,)
        except Exception as e:  # noqa: BLE001 -- never fail the model: count and serve stock
            if is_oom(e): raise
            COUNTS["errors"]["att_last"] = repr(e)[:300]
            if verbose: _log("attention exception -> stock: " + repr(e)[:300])
            return _stock("exception", q, k, v, bias, mask, scale)

    cuequivariance_triangular_attn._odde_arm_t = True
    cuequivariance_triangular_attn._orig = orig
    return cuequivariance_triangular_attn


# ----------------------------------------------------------------------------------------------------------------- (b) triangle multiplication



def make_trimul_forward(orig_forward, min_tokens=None, gate_design=True, u2=None):
    """TriangleMultiplicativeUpdate.forward under the arm: the design gate, the scope, upstream's small-N regime, then (U2) the core TriMul
    provider by the tier word through the sibling unit odde_trimul_bind -- every admitted call at every row count; a call the provider hands
    back (Aside: measured stock winner, refusal by name, excluded row) and every call without U2 / without the word run the STOCK forward BY
    NAME, counted under trimul_stock_reasons[<reason>]. The arm carries no TriMul construction of its own."""
    u2_fixed = u2                      # None -> read CFG["U2_TRIMUL"] at call time (so the test driver can switch variants between arms)
    min_tokens = CFG["MIN_TOKENS"] if min_tokens is None else int(min_tokens)
    verbose = CFG["VERBOSE"]
    TB = _trimul_bind()                # the core TriMul provider binding (ODDE_TRIMUL word) or None: every admitted call the stock forward by name
    COUNTS["trimul_bind"] = None if TB is None else TB.WORD

    def forward(self, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch"):
        def stock(reason):
            _bump("trimul_stock_calls"); _bump("trimul_stock_reasons", reason)
            return orig_forward(self, z, mask=mask, inplace_safe=inplace_safe, _add_with_inplace=_add_with_inplace,
                                _inplace_chunk_size=_inplace_chunk_size, triangle_multiplicative=triangle_multiplicative)
        try:
            N = int(z.shape[-2])
            if triangle_multiplicative != "cuequivariance" or self.c_z != self.c_hidden:
                return stock("torch_path_or_c_ne_c_hidden")
            nt = STATE["n_token"]
            if gate_design and (nt is None or nt < min_tokens):
                return stock(f"design_below_gate_{min_tokens}" if nt is not None else "no_design_context")
            if gate_design and SCOPE != "all" and STATE["in_scope"] <= 0:
                return stock("out_of_scope_module")
            if N <= 100:
                return stock("rows_le_100_cueq_torch_regime")
            u2 = CFG["U2_TRIMUL"] if u2_fixed is None else u2_fixed
            if not u2:
                return stock("u2_off")
            if TB is None:                                              # ODDE_TRIMUL unset (lever trimul_core left out): the stock TriMul by name
                return stock("no_provider_word")
            try:                                                        # ODDE_TRIMUL=<word>: the core provider's row for this cell (odde_trimul_bind.serve), every row count
                out = TB.serve(self, z, mask, residual=bool(inplace_safe is True and _add_with_inplace), site="arm_u2")
            except TB.Aside as _a:                                      # measured stock winner / refusal by name / excluded row: the stock forward BY NAME, counted
                TB.note_aside(_a.reason); _bump("trimul_prov_asides", _a.reason)
                return stock("provider_aside:" + str(_a.reason)[:80])
            _bump("trimul_prov_calls"); _bump("trimul_bf16_calls" if z.dtype != __import__("torch").float32 or __import__("torch").is_autocast_enabled() else "trimul_fast_calls")
            return out
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise
            COUNTS["errors"]["trimul_last"] = repr(e)[:300]
            if not warn_trimul_exception_once(e) and verbose: _log("trimul exception -> stock: " + repr(e)[:300])
            return stock("exception")

    forward._odde_arm_t = True
    forward._orig = orig_forward
    return forward



# ----------------------------------------------------------------------------------------------------------------- (d) U2c: transpose-free pair update
def make_zt_block_forward(orig_forward, scope_ids, source=False):
    """PairformerBlock.forward without the two `z = z.transpose(-2, -3).contiguous()` copies around tri_att_end (inference / inplace_safe branch only).
    On opendde 1.1.1 the stack reaches its blocks through PairformerBlock.forward_source (pairformer.py:602-632, partial(b.forward_source, ...));
    PairformerBlock.forward is the Fold-CP hook + a delegate to forward_source (:273-323), whose body is the 1.0.0 forward byte for byte (:325-415). Bound as
    forward_source (source=True) the replacement IS reached, and leaves the Fold-CP hook to .forward (it is not repeated here).
    Stock: z^T (copy) ; z^T += att_end(z^T) ; z = (z^T)^T (copy).  Here: z += att_end(z.transpose(-2,-3)).transpose(-2,-3)  -- the SAME module applied to a
    strided view (LayerNorm / Linear read strided rows; the attention kernels make q/k/v contiguous themselves exactly as before), and the update added through the
    transposed view. Arithmetic per element is identical (same kernels, same operands); only the memory layout of the LN/linear inputs differs, which can change
    cuBLAS' kernel choice -> byte equality with the base arm is established under the deterministic recipe, never assumed."""
    import torch

    def forward(self, s, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None, extra_attn_bias=None):
        _bump("zt_block_reached")                                  # every entry of the bound statement (served or gated to stock): a class-level bind the engine never reaches stays 0
        if not inplace_safe or id(self) not in scope_ids or (STATE["n_token"] is not None and STATE["n_token"] < CFG["MIN_TOKENS"] and CFG.get("ZT_GATED", True)):
            return orig_forward(self, s, z, pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size, extra_attn_bias=extra_attn_bias)
        if not source:                                             # bound as .forward (opendde 1.0.0): the Fold-CP hook first, as the stock forward; as .forward_source the caller ran it
            fr = self._maybe_forward_foldcp_pair_only(s, z, pair_mask, chunk_size)
            if fr is not None:
                return fr
        z = self.tri_mul_out(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
        z = self.tri_mul_in(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
        z += self.tri_att_start(z, mask=pair_mask, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        zt = z.transpose(-2, -3)                                   # view, no copy
        upd = self.tri_att_end(zt, mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        zt += upd                                                  # in-place add through the transposed view == z += upd^T
        del upd, zt
        z += self.pair_transition(z)
        _bump("zt_block_calls")
        if self.c_s > 0:
            s = s + self.attention_pair_bias(a=s, s=None, z=z, extra_attn_bias=extra_attn_bias)
            s = s + self.single_transition(s)
        return s, z
    forward._odde_arm_t = True; forward._orig = orig_forward
    return forward


# ----------------------------------------------------------------------------------------------------------------- (e) U3: cast-once tri-attention prologue (EXACT vs U)
def _transition_bind():
    """The kit's transition provider binding (levers/ARMT/odde_transition_bind) when its word is live; None otherwise (the engine's module serves)."""
    try:
        import odde_transition_bind as _tb
    except Exception as e:  # noqa: BLE001
        COUNTS["errors"]["transition_bind_import"] = repr(e)[:200]; return None
    return _tb if _tb.active() else None


def _load_fpf_transition():
    """The kit's transition composition (third_party/fpf_transition_odde: sep16); returns the module or None (counted)."""
    d = _TP
    if d not in sys.path: sys.path.insert(0, d)
    try:
        import importlib
        return importlib.import_module("fpf_transition_odde")
    except Exception as e:  # noqa: BLE001
        if is_oom(e): raise
        COUNTS["errors"]["fpf_transition_import"] = repr(e)[:200]; return None


def make_castonce_ta_forward(orig_forward):
    """TriangleAttention.forward for the ARM-U bf16 region with the LayerNorm output cast to bf16 ONCE.
    Stock under autocast: x32 = LN(x) (fp32; LN is fp32-listed / disables autocast) ; then linear (bias, H out), linear_q, linear_k, linear_v, linear_g each receive x32
    and autocast casts it to bf16 separately (5 identical [I,J,384] fp32->bf16 casts per node).  Here: x16 = x32.to(bfloat16) once; every Linear then receives a bf16
    input -> upstream's Linear bf16 branch runs F.linear(x16, W.to(bf16)) with autocast disabled == exactly the call autocast makes (same cast op on the same tensors, same
    cuBLAS GEMM arguments) => byte-identical outputs by construction. Outside a bf16 autocast region (fp32) -> stock forward."""
    import torch
    from opendde.model.utils import permute_final_dims

    def forward(self, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
        if not (torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.bfloat16 and x.is_cuda and x.dtype == torch.float32
                and getattr(self.linear, "precision", None) is None and STATE["in_scope"] > 0):
            _bump("castonce_stock_calls"); return orig_forward(self, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
        if mask is None:
            mask = x.new_ones(x.shape[:-1])
        if not self.starting:
            x = x.transpose(-2, -3); mask = mask.transpose(-1, -2)
        x = self.layer_norm(x)                       # fp32 (stock)
        x = x.to(torch.bfloat16)                     # THE cast autocast applies to each Linear input -- done once
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1)).unsqueeze(-4)
        biases = [mask_bias, triangle_bias]
        if chunk_size is not None:
            x = self._chunk(x, biases, chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
        else:
            x = self.mha(q_x=x, kv_x=x, biases=biases, triangle_attention=triangle_attention)
        if not self.starting:
            x = x.transpose(-2, -3)
        _bump("castonce_calls")
        return x
    forward._odde_arm_t = True; forward._orig = orig_forward
    return forward

# ----------------------------------------------------------------------------------------------------------------- install / remove
_SCOPE_EXT = {"trunk": (), "all": ("refiner", "confidence"), "trunk+refiner": ("refiner",), "trunk+confidence": ("confidence",),
              "trunk+refiner+confidence": ("refiner", "confidence")}


def _scope_ext(scope=None):
    """the modules ODDE_ARM_T_SCOPE adds to the trunk binds: () | ('refiner',) | ('confidence',) | ('refiner', 'confidence'); an unknown spelling raises."""
    s = (SCOPE if scope is None else scope).strip().lower()
    if s not in _SCOPE_EXT:
        raise ValueError("[odde_arm_t] ODDE_ARM_T_SCOPE=%r is not one of %s" % (s, "|".join(_SCOPE_EXT)))
    return _SCOPE_EXT[s]


def _scope_modules(model):
    """modules the arm binds (kernels; under U the bf16 region, the transpose-free block, the U3 prologue/transition): the 48-block trunk stack, every
    MSA-module pair stack, the template embedder's pairformer (only when it has blocks: n_blocks > 0); plus, when ODDE_ARM_T_SCOPE names them,
    model.structural_token_refiner (the 4-block PairformerStack over the structural tokens) and model.confidence_head.pairformer_stack (4 blocks, per sample).
    A named extension whose module is absent raises (the scope asked for is the scope bound, or the bind fails by name)."""
    mods = []
    ps = getattr(model, "pairformer_stack", None)
    if ps is not None: mods.append(("pairformer_stack", ps))
    mm = getattr(model, "msa_module", None)
    if mm is not None:
        for i, b in enumerate(getattr(mm, "blocks", [])):
            if hasattr(b, "pair_stack"): mods.append((f"msa_module.blocks.{i}.pair_stack", b.pair_stack))
    te = getattr(model, "template_embedder", None)
    if te is not None and getattr(te, "n_blocks", 0) > 0:
        for name in ("pairformer_stack", "pairformer"):
            if hasattr(te, name): mods.append((f"template_embedder.{name}", getattr(te, name)))
    ext = _scope_ext()
    if "refiner" in ext:
        r = getattr(model, "structural_token_refiner", None)
        if r is None or not hasattr(r, "blocks"):
            raise RuntimeError("[odde_arm_t] ODDE_ARM_T_SCOPE=%s names the structural refiner but model.structural_token_refiner is absent or has no blocks" % SCOPE)
        mods.append(("structural_token_refiner", r))
    if "confidence" in ext:
        ch = getattr(model, "confidence_head", None)
        ps = getattr(ch, "pairformer_stack", None) if ch is not None else None
        if ps is None or not hasattr(ps, "blocks"):
            raise RuntimeError("[odde_arm_t] ODDE_ARM_T_SCOPE=%s names the confidence head but model.confidence_head.pairformer_stack is absent or has no blocks" % SCOPE)
        mods.append(("confidence_head.pairformer_stack", ps))
    return mods


_HOOKS = []           # removable hook handles (instance-level)
_MODELS = []          # models with instance hooks installed


def _enter_conf(mod, args, kwargs=None):
    STATE["in_conf"] += 1
def _exit_conf(mod, args, output):
    STATE["in_conf"] -= 1


def _enter_scope(mod, args, kwargs=None):
    STATE["in_scope"] += 1

def _exit_scope(mod, args, output):
    STATE["in_scope"] -= 1


def bind(model, arm=None):
    """Instance-level part of install(): scope hooks and the bf16-autocast hooks (U) on THIS model's trunk pair stacks. Called by
    install(model=...) or automatically from the get_pairformer_output wrapper the first time a model predicts."""
    import torch
    arm = (arm or COUNTS.get("arm") or "U").upper()
    if any(m is model for m in _MODELS):
        return
    # HAZARD: a caller that monkeypatched model.get_pairformer_output on the INSTANCE before install() shadows the class-level
    # size-gate wrapper -> STATE['n_token'] is never set -> every kernel call is refused with reason 'no_design_context'. Detect it loudly.
    shadow = [n for n in ("get_pairformer_output",) if n in getattr(model, "__dict__", {}) and not getattr(model.__dict__[n], "_resolves_class_attr_at_call", False)]
    if shadow:
        msg = (f"install/bind: the model INSTANCE has its own attribute(s) {shadow} shadowing the class-level wrapper that odde_arm_t installs; "
               f"the size gate would never see a design (att/trimul calls refused with 'no_design_context'). Patch the CLASS, or make your instance wrapper "
               f"resolve type(model).get_pairformer_output at CALL time and mark it `wrapper._resolves_class_attr_at_call = True`, or set ODDE_ARM_T_ALLOW_SHADOW=1 to proceed anyway "
               f"(then check the census: att_prov_calls must be > 0 above the gate).")
        COUNTS["errors"]["instance_shadow"] = shadow
        if os.environ.get("ODDE_ARM_T_ALLOW_SHADOW", "0") in ("", "0"):
            raise RuntimeError("[odde_arm_t] " + msg)
        _log("WARNING " + msg)
    _cps = getattr(getattr(model, "confidence_head", None), "pairformer_stack", None)   # the confidence head's pair stack: marked for the attention binding's
    if _cps is not None and _triattn_bind() is not None:                                  # lever triattn_conf (left out: ODDE_TRIATTN_CONF=stock, its calls are the stock op's by name)
        _HOOKS.append(_cps.register_forward_pre_hook(_enter_conf))
        _HOOKS.append(_cps.register_forward_hook(_exit_conf, always_call=True))
        try:                                                                              # + the mark one level down, on that stack's triangle-attention modules: the paths that never
            from . import odde_conf_chunk as _CC                                                # enter the stack's forward (the offload unit's conf stage: block.tri_att_*.mha per host-streamed
            COUNTS["conf_chunk"] = _CC.bind(model)                                        # row block) are served by the word under the same rule, with their own census (odde_conf_chunk)
        except Exception as e:  # noqa: BLE001
            COUNTS["errors"]["conf_chunk_bind"] = repr(e)[:200]
    for name, m in _scope_modules(model):
        _HOOKS.append(m.register_forward_pre_hook(_enter_scope))
        _HOOKS.append(m.register_forward_hook(_exit_scope, always_call=True))
        if arm == "U":
            def _pre(mod, args, kwargs):
                outer = torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16   # the caller already runs bf16 autocast (upstream --dtype bf16): the stock's outputs are bf16 there
                ctx = torch.autocast("cuda", dtype=torch.bfloat16); ctx.__enter__(); mod._odde_arm_u_ctx = getattr(mod, "_odde_arm_u_ctx", []) + [(ctx, outer)]
            def _post(mod, args, output):
                ctxs = getattr(mod, "_odde_arm_u_ctx", [])
                outer = False
                if ctxs:
                    ctx, outer = ctxs.pop(); ctx.__exit__(None, None, None)
                _bump("bf16_stack_calls")
                if outer:                                        # bf16 base: hand back the stock's own dtypes (an fp32 upcast here breaks upstream's bf16 index_put downstream)
                    return output
                if isinstance(output, tuple):
                    return tuple(o.float() if torch.is_tensor(o) and o.is_floating_point() and o.dtype != torch.float32 else o for o in output)
                if torch.is_tensor(output) and output.is_floating_point() and output.dtype != torch.float32:
                    return output.float()
                return output
            _HOOKS.append(m.register_forward_pre_hook(_pre, with_kwargs=True))
            _HOOKS.append(m.register_forward_hook(_post, always_call=True))
    if arm == "U" and CFG["U3"]:
        import opendde.model.triangular.triangular as TT
        cls = TT.TriangleAttention
        if not getattr(cls.forward, "_odde_arm_t", False):
            _ORIG["u3@TriangleAttention"] = (cls, "forward", cls.forward)
            cls.forward = make_castonce_ta_forward(cls.forward)
    TB = _transition_bind()                                          # lever transition_core | transition_exact: the pair-transition sites under the kit's word
    if TB is not None and (arm == "Z" or (CFG["U3"] and CFG["U_TRANSITION"])):   # (ODDE_TRANSITION; both arms) -- the provider's cell decision per call, a stock-row cell
        FT = _load_fpf_transition()                                  # served by the kit's single-engine composition sep16 BY NAME (third_party/fpf_transition_odde)
        if FT is None:
            _log("fpf_transition_odde not importable -> the pair transition stays the engine's module by name (" + COUNTS["errors"].get("fpf_transition_import", "") + ")")
        else:
            n = 0
            for name, m in _scope_modules(model):
                for sub in m.modules():
                    pt = getattr(sub, "pair_transition", None)
                    if pt is not None and not getattr(pt, "_fpf_odde_applied", False):
                        FT.apply(pt); _U3_TR.append(pt); n += 1
            COUNTS["fpf_transition"] = {"version": getattr(FT, "__version__", "?"), "applied_modules": n, "variant": getattr(FT, "VARIANT", "?"), "word": TB.WORD}
            COUNTS["transition_bind"] = TB.WORD
    if CFG["ZT"] or arm == "Z":
        import opendde.model.modules.pairformer as PF
        ids = set()
        for name, m in _scope_modules(model):
            for sub in m.modules():
                if isinstance(sub, PF.PairformerBlock): ids.add(id(sub))
        _ZT_IDS.update(ids)
        cls = PF.PairformerBlock
        attr = "forward_source" if hasattr(cls, "forward_source") else "forward"   # opendde 1.1.1's stack calls b.forward_source (pairformer.py:602-632); 1.0.0's calls b.forward
        if not getattr(getattr(cls, attr), "_odde_arm_t", False):
            _ORIG["zt@PairformerBlock"] = (cls, attr, getattr(cls, attr))
            setattr(cls, attr, make_zt_block_forward(getattr(cls, attr), _ZT_IDS, source=(attr == "forward_source")))
        COUNTS["zt_blocks"] = len(_ZT_IDS); COUNTS["zt_bind"] = attr                 # zt_block_calls counts every served call: installed with zero calls after a prediction = the bind was not reached
    _MODELS.append(model)
    COUNTS["bound_modules"] = [n for n, _ in _scope_modules(model)]


def stock_knobs(environ=None):
    """The kit package's per-run fact ``ODDE_STOCK_KNOBS=<flag>=<value>[,...]``: upstream's own --triatt_kernel / --trimul_kernel STATED other than
    `auto` for this run. The arm's attention site (triatt_kernel) / TriMul routes (trimul_kernel) then stay on the stock op BY NAME —
    upstream's kernel of the caller's choice serves; the rest of the arm is unchanged. {} when absent."""
    env = os.environ if environ is None else environ
    out = {}
    for tok in (env.get("ODDE_STOCK_KNOBS") or "").split(","):
        if "=" in tok:
            k, v = tok.split("=", 1)
            if k.strip() and v.strip():
                out[k.strip()] = v.strip()
    return out


def install(arm=None, verbose=True, min_tokens=None, att_mode=None, trimul_mode=None, model=None):
    """Install ARM 'U' or 'Z' (default: U when ODDE_ARM_U is set, else Z). Idempotent per attribute; call remove() first to switch arms.
    Returns COUNTS. model= binds the instance hooks immediately; otherwise they are bound at the first get_pairformer_output call."""
    try:                                                                     # the core's import trampoline, once at the arm's install: the libraries the provider
        import opt_core as _oc                                               # faces reach (cuequivariance & co.) imported here, not inside the first served call of a pass; the faces
        COUNTS["warm_imports"] = dict(_oc.warm_imports(libraries=("torch", "cuequivariance_ops_torch", "cuequivariance_torch"), origin="opendde:odde_arm_t") or {})   # torch FIRST (the cuEq ops library resolves NVRTC through it); the faces auto-warm on their first call too — this only moves the cost to start-up; no bytes change
    except Exception as _e:  # noqa: BLE001 -- an older core or a library that cannot import: the rows refuse by name at their own call
        COUNTS["warm_imports"] = {"error": repr(_e)[:120]}
    import torch
    arm = (arm or ("U" if os.environ.get("ODDE_ARM_U", "0") not in ("", "0") else "Z")).upper()
    assert arm in ("U", "Z"), arm
    att_mode = att_mode or "stock"; trimul_mode = trimul_mode or CFG["TRIMUL_MODE"]   # attention: the core provider under the word (below), else the stock op under both arms
    if arm == "Z":                       # arm Z = base arm + transpose-free pair update ONLY (the EXACT tier's arm: same arithmetic as the base arm)
        att_mode = "stock"; trimul_mode = "stock"
    if _triattn_bind() is not None:      # ODDE_TRIATTN=<word>: the attention site through the core provider under either arm (arm Z: its exact word, bitwise rows only)
        att_mode = "core_" + str(_triattn_bind().WORD)
    kn = stock_knobs()                   # upstream's own triangle-kernel flags stated for this run (the kit package's fact): those sites stay on the stock op by name
    COUNTS["stock_knobs"] = dict(kn)
    if kn.get("triatt_kernel"):
        att_mode = "stock:stock_knob"; COUNTS["att_aside"] = f"stock_knob:triatt_kernel={kn['triatt_kernel']}"
    if kn.get("trimul_kernel"):
        trimul_mode = "stock:stock_knob"; COUNTS["trimul_aside"] = f"stock_knob:trimul_kernel={kn['trimul_kernel']}"
        CFG["U2_TRIMUL"] = False; COUNTS["u2_trimul"] = False   # the U2 route rides the TriMul bind: off with it, named
    min_tokens = CFG["MIN_TOKENS"] if min_tokens is None else int(min_tokens)
    if COUNTS["installed"]:
        if COUNTS["arm"] == arm:
            return COUNTS
        remove()
    import opendde.model.triangular.layers as L
    import opendde.model.triangular.triangular as T
    import opendde.model.modules.pairformer as PF
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cc = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else (0, 0)
    COUNTS.update(device=dev, cc=list(cc), min_tokens=min_tokens, att_mode=att_mode, trimul_mode=trimul_mode)
    if cc[0] < 8:
        _log(f"device {dev} cc {cc}: not an sm80+ CUDA device -> NOT installing (stock path)"); return COUNTS
    if cc not in TESTED_CC:
        _log(f"WARNING device {dev} cc {cc}: only {', '.join('sm%d%d (%s)' % (k[0], k[1], v) for k, v in sorted(TESTED_CC.items()))} carry this add-on's test record; the arm runs UNTESTED here")
    if att_mode.startswith("core_"):
        orig = L.cuequivariance_triangular_attn
        if getattr(orig, "_odde_arm_t", False):
            orig = orig._orig
        new = make_attn(orig, min_tokens=min_tokens, att_mode=att_mode)
        _ORIG["att"] = (L, "cuequivariance_triangular_attn", orig)
        L.cuequivariance_triangular_attn = new          # looked up as a module global inside layers.Attention.forward at call time
        for mn, m in list(sys.modules.items()):        # defensive: any opendde module that imported the symbol by value
            if mn and mn.startswith("opendde") and m is not L and getattr(m, "cuequivariance_triangular_attn", None) is orig:
                setattr(m, "cuequivariance_triangular_attn", new); _ORIG[f"att@{mn}"] = (m, "cuequivariance_triangular_attn", orig)
    if trimul_mode == "fast":
        cls = T.TriangleMultiplicativeUpdate
        orig_f = cls.forward
        if getattr(orig_f, "_odde_arm_t", False):
            orig_f = orig_f._orig
        _ORIG["trimul"] = (cls, "forward", orig_f)
        cls.forward = make_trimul_forward(orig_f, min_tokens=min_tokens)
    # design-level gate + lazy instance binding: wrap OpenDDE.get_pairformer_output (class level) to record N_token of the current prediction
    import opendde.model.opendde as OM
    cls = OM.OpenDDE
    og = cls.get_pairformer_output
    if getattr(og, "_odde_arm_t", False):
        og = og._orig
    def get_pairformer_output(self, *a, _og=og, **kw):
        ifd = kw.get("input_feature_dict", a[0] if a else None)
        try:
            nt = None
            if isinstance(ifd, dict):
                for key in ("residue_index", "token_index", "asym_id"):
                    if key in ifd and hasattr(ifd[key], "shape"):
                        nt = int(ifd[key].shape[-1]); break
            STATE["n_token"] = nt
        except Exception:  # noqa: BLE001
            STATE["n_token"] = None
        _bump("designs_seen", f"n_token={STATE['n_token']}")
        bind(self, COUNTS.get("arm"))
        return _og(self, *a, **kw)
    get_pairformer_output._odde_arm_t = True; get_pairformer_output._orig = og
    _ORIG["gate@OpenDDE.get_pairformer_output"] = (cls, "get_pairformer_output", og)
    cls.get_pairformer_output = get_pairformer_output
    COUNTS["arm"] = arm
    if model is not None:
        bind(model, arm)
    COUNTS["installed"] = True; COUNTS["arm"] = arm; COUNTS["installed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if verbose:
        comp = []
        if arm == "U": comp.append(("triattn:" + att_mode[5:] if att_mode.startswith("core_") else "attention=stock") + ("+TriMul:" + str(COUNTS.get("trimul_bind")) if COUNTS.get("trimul_bind") else "+TriMul:stock"))
        if arm == "Z" and att_mode.startswith("core_"): comp.append("triattn:" + att_mode[5:] + " (attention through the core provider's exact rows)")
        if arm == "U": comp += ["bf16 pair stacks"] + (["cast-once+sep16-transition(U3)"] if CFG["U3"] else ["(U3 off = v0.2 U)"]) + (["TriMul through the core provider (U2)"] if CFG["U2_TRIMUL"] else [])
        if CFG["ZT"] or arm == "Z": comp.append("Z transpose-free update (EXACT)")
        lab = {"U": "TIER-2b: bf16 trunk (Protenix-default numerics applied to OpenDDE), r2r-deterministic",
               "Z": "EXACT tier: transpose-free pair update only (DET byte-identical to the base arm)"}[arm] + "; composition=" + "+".join(comp)
        _log(f"ACTIVE v{__version__} ARM {arm} ({lab}{'' if arm == 'Z' else '; never EXACT'}): attention={att_mode} trimul={trimul_mode} gate={min_tokens} tok "
             f"(stock kernels below) device='{dev}' cc={cc}")
    return COUNTS


def remove(verbose=True):
    """Restore every patched attribute and remove every instance hook (back to the base arm exactly as launched). Prints the census line first
    (COUNTS accumulated since the last reset_counts()); the atexit COUNTS line of a process that called remove() reflects only calls made after it."""
    n = 0
    if verbose and COUNTS.get("installed"):
        _report(prefix="COUNTS@remove ")
    for key, (owner, attr, orig) in list(_ORIG.items()):
        setattr(owner, attr, orig); _ORIG.pop(key, None); n += 1
    for h in _HOOKS:
        try: h.remove(); n += 1
        except Exception: pass  # noqa: BLE001
    _HOOKS.clear(); _MODELS.clear(); STATE["in_scope"] = 0
    _ZT_IDS.clear()
    for pt in _U3_TR:
        try:
            if getattr(pt, "_fpf_odde_applied", False):
                orig = pt.__dict__.pop("_fpf_odde_orig_forward", None)
                pt.__dict__.pop("forward", None)            # drop the instance-bound forward -> class forward again
                pt._fpf_odde_applied = False; n += 1
        except Exception: pass  # noqa: BLE001
    _U3_TR.clear()
    try:                                   # arm-switch hygiene: drop the TriMul provider binding's per-word row caches (packed weights / workspaces)
        tb = _TMB["mod"]
        if tb:
            tb._PROV.clear()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except Exception: pass  # noqa: BLE001
    was = COUNTS.get("arm")
    COUNTS["installed"] = False; COUNTS["arm"] = None
    if verbose and n:
        _log(f"REMOVED arm {was}: {n} attributes restored to stock")
    return COUNTS


def stats():
    out = json.loads(json.dumps(COUNTS, default=str))
    if _TMB["mod"]:
        try:
            out["trimul_prov"] = json.loads(json.dumps(_TMB["mod"].describe(), default=str))
        except Exception as e:  # noqa: BLE001
            out["trimul_prov"] = {"error": repr(e)[:200]}
    if _TB["mod"]:
        try:
            out["triattn"] = json.loads(json.dumps(_TB["mod"].describe(), default=str))
        except Exception as e:  # noqa: BLE001
            out["triattn"] = {"error": repr(e)[:200]}
    try:
        if "fpf_transition_odde" in sys.modules:
            out["fpf_transition_stats"] = json.loads(json.dumps(sys.modules["fpf_transition_odde"].describe(), default=str))
    except Exception: pass  # noqa: BLE001
    return out


def reset_counts():
    for k in ("att_prov_calls", "att_stock_calls", "trimul_fast_calls", "trimul_stock_calls", "bf16_stack_calls"):
        COUNTS[k] = 0
    if _TB["mod"]:
        _TB["mod"].reset_counts()
    for k in ("att_stock_reasons", "trimul_stock_reasons", "att_dtypes", "errors", "designs_seen"):
        COUNTS[k] = {}
    COUNTS["trimul_admission"] = {"checked": 0, "refused": 0, "first_refusal": None, "max_needed_gib": 0.0}


def _report(prefix="COUNTS "):
    try:
        line = json.dumps(stats(), default=str, sort_keys=True)
        _log(prefix + line)
    except Exception:  # noqa: BLE001
        pass


atexit.register(lambda: _report("COUNTS@exit "))


def install_from_env(verbose=True):
    """ODDE_ARM_U=1 -> ARM U (provider triangle attention + the TriMul site through the provider (U2) + bf16 pair stacks + cast-once prologue + the transition binding (U3) + Z);
    ODDE_ARM_Z=1 alone -> arm Z (EXACT tier: transpose-free pair update only);  nothing set -> inert (stock/base path, nothing patched)."""
    if os.environ.get("ODDE_ARM_U", "0") not in ("", "0"):
        return install("U", verbose=verbose)
    if CFG["Z_ALONE"]:
        return install("Z", verbose=verbose)
    return COUNTS
