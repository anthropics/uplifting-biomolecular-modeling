"""levers_ptx1 — the kit's levers on the pip-installed protenix 1.1.0 model, installed as run-time patches over
protenix.model.triangular.triangular (TriangleMultiplicativeUpdate, TriangleAttention), modules.primitives.Transition and, through the
adapters beside this file, the diffusion sampler, the MSA module and the template embedder (the 1.1.0 checkpoint has c_z = 128).

Arm grammar: "<trimul>[+lever[+lever...]]" — `LEVER_NAMES` is the lever list; trimul is stock | fast | exact: the shared core's
triangle-multiplication provider (opt_core.kernels.trimul) by TIER word per key — `exact` binds rows bit-equal to stock only, `fast` the
tolerance-class row (`big` under --mode big, kit_mode()); a key the provider refuses BY NAME keeps upstream's own statement, counted
`stock:trimul:<refusal>`. What each lever replaces and its numerics class: the kit's CHANGES.md; the census words each route counts are the
ones `describe()` returns for the LEVER lines.
  gblock / gflash (+ triexact / tricuda) the shared core's fused triangle-attention block (opt_core.attn.pair_fused) — exact / tolerance class;
                                         gblock keeps the stock attention core, or takes it from the triangle-attention provider's `exact`
                                         tier word (opt_core.kernels.triattn) under triexact; gflash takes its core from the provider's `fast`
                                         tier word under tricuda; gblock and gflash are exclusive, each core word rides its own block only
  xtr / ttr                              the pair / MSA transitions by the transition provider's `exact` / `fast` tier word
                                         (opt_core.kernels.transition); exclusive
  sg, hoist, sampler_prep                CUDA-graph capture of the diffusion denoiser step, the DiT step-invariant hoist, the graphed loop's
                                         host path (lib/kit112_src: infopt_graphs, dit_hoist) — exact class; sampler_prep rides `sg`
  keep_pool, summary_hostidx, lazy_init  lib/ptx1_keep_pool.py, lib/ptx1_summary_host.py, lib/ptx1_lazy_init.py — exact class
  ditattn, ditattnfp16, atomattn         ptxfpf/apb_ptx1.py (opt_core.kernels.apb rows fpf_apb / fpf_atom) — tolerance class
  dit_attn_exact                         opt_core.kernels.apb row dit_exact (prebuilt sm_90 CUDA kernel for the DiT pair-bias attention,
                                         bit-identical to the SDPA kernel it replaces; env word DIT_ATTN_EXACT_ENV set in-process) — exact class
  pfattn, opm_fused, pwa_fused           ptxfpf/trunk2_ptx1.py (opt_core.kernels.apb + lib/protenix_fpf_msa) — tolerance class
  cond_dedupe, dit_fused, dit_lowp,      ptxfpf/ditfast_ptx1.py (lib/protenix_fpf_ditfast; atom_attn_exact: opt_core.kernels.apb by the
  atom_fused, atom_attn_exact            `exact` tier word — exact class, the others tolerance class)
  template_dedupe, tmpl_*                ptxfpf/ptx1_templ.py (the template embedder's c = 64 Pairformer)
Install order is fixed in `apply()`: dit_attn_exact before the sampler graph (its route must be on primitives._attention when the step is
captured); keep_pool after the sampler levers (it wraps whatever empty_cache is current, the graph guard included); the apb / trunk2 /
ditfast adapters after the sampler graph and hoist (instance-level forwards the graphed step reaches through Module.__call__); lazy_init is
installed by protenix_v1_opt.stack BEFORE the runner is built and only checked here. dit_hoist has no uninstall.
One arm per process: `bind_model(model)` then `apply(arm)` once, on the runner the stock CLI built.
"""
import os, sys, math, time, json
import torch
import torch.nn.functional as Fn
from opt_core.oom import is_oom                     # the one out-of-memory classifier: every rerouting handler below re-raises an OOM first
import ptx1_templ as _TEMPL                          # the template-embedder levers (template_dedupe, tmpl_triatt, tmpl_trimul, tmpl_xtr, tmpl_pairfused): ptxfpf/ptx1_templ.py

CFG = {"trimul": "stock", "gblock": False, "gflash": False, "tricuda": False, "triexact": False, "xtr": False, "ttr": False, "sg": False, "hoist": False, "keep_pool": False, "summary_hostidx": False,
       "ditattn": False, "ditattnfp16": False, "atomattn": False, "dit_attn_exact": False, "lazy_init": False, "template_dedupe": False, "tmpl_triatt": False,
       "tmpl_trimul": False, "sampler_prep": False, "pfattn": False, "opm_fused": False, "pwa_fused": False, "cond_dedupe": False, "dit_fused": False, "dit_lowp": False, "atom_fused": False, "atom_attn_exact": False, "tmpl_xtr": False, "tmpl_pairfused": False, "tmpl_trimul_exact": False}
LEVER_NAMES = ("gblock", "gflash", "tricuda", "triexact", "xtr", "ttr", "sg", "hoist", "keep_pool", "summary_hostidx", "ditattn", "ditattnfp16", "atomattn", "dit_attn_exact", "lazy_init",
               "template_dedupe", "tmpl_triatt", "tmpl_trimul", "sampler_prep", "pfattn", "opm_fused", "pwa_fused", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "atom_attn_exact", "tmpl_xtr", "tmpl_pairfused", "tmpl_trimul_exact")   # the grammar's lever words (kit.lever_grammar reads this tuple)
DIT_ATTN_EXACT_ENV = "PTX_DIT_ATTN_EXACT"                      # the vendored package's own switch word (read by its apply()); set in-process from the arm word, never asked of the caller
TRIMUL_TIER = {"exact": "exact", "fast": "fast"}                    # the TriMul words of the arm grammar -> the shared core's TriMul provider TIER word served per key (opt_core.kernels.trimul:
                                                                   # select(word=<tier>) names the row, triangle_multiplication(selection=...) serves it); under --mode big the arm composes
                                                                   # the `fast` word and the provider's `big` tier is asked (kit_mode()).  No row word, no size floor, no kit cell table and
                                                                   # no kit-side kernel statement: a key the provider refuses BY NAME keeps upstream's own TriangleMultiplicativeUpdate
                                                                   # statement, counted `stock:trimul:<refusal>`; the LEVER line names the rows served (row= / served_by= / selections= / bind=tier:<word>)


def kit_mode():
    """The kit's active mode word (exact | fast | big) when the kit package activated these levers, else None: the fast TriMul word binds the provider's
    `big` tier word under --mode big and its `fast` tier word otherwise (the package is read from sys.modules; nothing is imported here)."""
    st = sys.modules.get("protenix_v1_opt.stack")
    try:
        m = st.status().get("mode") if st is not None else None
    except Exception:
        m = None
    return m if m in ("exact", "fast", "big") else None


TRANSITION_GATE_ROWS = 4096                                       # the template Pairformer's c=64 x 128 transition route (tmpl_xtr) serves a call of >= 4096 rows; below it the stock statement, by name
                                                                  # (`stock:C=64`). The c=128 / c=64 x 256 transitions have no kit gate: the provider's cells and exact vouch decide by size (xtr | ttr).
TRIATTN_CEILING_TOKENS = 2048                                     # structural: above 2048 tokens upstream ROW-CHUNKS the triangle attention (chunk_size set — its memory statement; the block computes the whole [N,N] bias/logit frame at once) and the stock statement answers, by name (`stock:chunk`); the provider's cells also end at N<=2048
# The two levers' block CONSTRUCTIONS (opt_core.attn.pair_fused `impl=`: which prologue / epilogue statement the fused block is built from — a structural
# selector of the lever's class, not an attention-kernel choice; the attention core inside either construction is the provider's tier word, below):
GBLOCK_PROLOGUE = "fpf"                                                           # gblock = the exact-class construction: the module's own LayerNorm output projected by the FlashPairformer prologue/epilogue (the bit-equality argument is on these bytes)
GFLASH_PROLOGUE = "lnl"                                                           # gflash = the tolerance-class construction: the LayerNorm fused into the prologue (LN+bias kernel, one cuBLAS q|k|v|g GEMM, the gate kernel)
TRIATTN_GATE_TOKENS = 16                                                          # upstream's own small-input route: an item of <= 16 tokens never reaches the kernel statement the block replaces (stock:gate) — structural, not a speed floor
# The exact-class pair levers (gblock: the fused triangle-attention block; xtr: the fused pair transition) compute their projections in ONE
# summation order: the order the stock Linear layers' cuBLAS bf16 GEMMs use at every row count on compute capability 9.0, so their outputs equal
# stock's bit for bit there. On a card whose cuBLAS switches one of those GEMM shapes to another summation order at some row counts, the lever
# cannot equal stock at those counts whatever its cells: the shared core refuses such a call BY NAME before any launch — opt_core.attn.pair_fused
# "exact_rows", the exact construction's served-row rule per compute capability (on 8.0 a floor per piece and,
# from 2**20 - 32 rows on, the cuBLAS piece-remainder rule — served when 161,424 <= rows mod 2**20 <= 2**20 - 8,192, the core's table;
# opt_core.kernels.transition's `exact` tier honours the same statement) — and
# the stock forward answers it, counted `stock:below_min_rows` / `stock:above_max_rows`; the LEVER line carries the rule's numbers
# (`min_rows=<n> max_rows=<n>`, row_range_facts()). Rows = the flattened call's leading dimensions, what the GEMMs see: a pair call [B, N, N, C]
# is B·N·N rows (an N-token input runs its pairformer / MSA-module pair calls at N·N rows). A card without an entry in the core's rule has one
# order at every row count: every supported call is served and nothing is printed (9.0). The Tier-2 levers (gflash, ttr) have none anywhere: a
# summation order is inside their tolerance.
RANGED_LEVERS = ("gblock", "xtr")                                                # the exact-class levers the core's rule may refuse by row count (row_range_facts reports these)
RANGED_PIECES = {"gblock": "prologue", "xtr": "transition"}                      # lever -> the rule's piece word in opt_core.attn.pair_fused (the block: its prologue + epilogue projections)
BELOW_MIN_ROWS = "stock:below_min_rows"                                          # the census word of a call under the served rows (a stock route by name, `stock:*`)
ABOVE_MAX_ROWS = "stock:above_max_rows"                                          # ... and of a call at or over the top whose remainder piece is not served
COUNTS = {}
_ORIG = {}
_CC = {}                                                                          # device index -> "M.m" (one probe per device)


def device_cc(dev):
    """'M.m' of a CUDA device (cached per index); None off CUDA."""
    if dev.type != "cuda":
        return None
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    if i not in _CC:
        _CC[i] = "%d.%d" % torch.cuda.get_device_capability(i)
    return _CC[i]


def exact_rows_word(lever, x):
    """The census word (BELOW_MIN_ROWS | ABOVE_MAX_ROWS) when the core's served-row rule refuses an exact-construction call of `lever` on the tensor `x`
    ([..., C]) on this card, else None (served, or the card has no entry): opt_core.attn.pair_fused.exact_rows_word — the refusal the core raises by
    name (Unsupported 'exact-rows:<word>') at plan time, read before the block's LayerNorm is computed."""
    w = _PF().exact_rows_word(RANGED_PIECES[lever], device_cc(x.device), x.numel() // int(x.shape[-1]))
    return None if w is None else _rows_census_word(w)


def _rows_census_word(w):
    """The core's refusal word ('below_min_rows' | 'above_max_rows', bare or inside a provider Refusal kind / an Unsupported reason) -> the census word."""
    w = str(w)
    return BELOW_MIN_ROWS if "below_min_rows" in w else (ABOVE_MAX_ROWS if "above_max_rows" in w else None)


def row_range_facts():
    """{lever: {cc, min_rows, max_rows, below, above, piece, piece_min, piece_top_margin}} for every ranged lever the arm carries that the core's rule
    covers on this process's card (opt_core.attn.pair_fused.exact_rows_facts; the LEVER line's / report's facts; below / above = the two census words
    the report counts as outside the served rows); {} off CUDA, when no carried lever has an entry on this card (nothing printed), or on a core
    without the rule."""
    if not torch.cuda.is_available():
        return {}
    cc = device_cc(torch.device("cuda", torch.cuda.current_device()))
    out = {}
    try:
        facts = _PF().exact_rows_facts
    except Exception:
        return {}
    for lever in RANGED_LEVERS:
        f = facts(cc, RANGED_PIECES[lever]) if CFG.get(lever) else None
        if f:
            out[lever] = {"cc": f["cc"], "min_rows": f["min_rows"], "max_rows": f["max_rows"], "below": BELOW_MIN_ROWS, "above": ABOVE_MAX_ROWS,
                          "piece": f["piece"], "piece_min": f["piece_min"], "piece_top_margin": f["piece_top_margin"]}
    return out


def _c(kind, key, n=1):
    d = COUNTS.setdefault(kind, {})
    d[key] = d.get(key, 0) + n


def _cache(mod):
    c = getattr(mod, "_fpf_cache", None)
    if c is None:
        c = mod._fpf_cache = {}
    return c


# =====================================================================================  TriMul
TRIMUL_CACHE = {}                                                  # the ONE provider cache shared by all TriMul modules (triangle_multiplication(cache=...): weight packs keyed by the weights'
                                                                   # identity, scratch arenas keyed by (row, C, N, dtype, device) — pairformer / MSA / confidence modules of one width share them)
TRIMUL_SEL = {}                                                    # (tier, cc, prec, C, N, direction) -> the provider's Selection | CoreAside(kind, fallback): decided once per key
_TRIMUL_STACK = []                                                 # the provider's stack word of this process (T.stack_word), read once


def _TM():
    from opt_core.kernels import trimul as T           # the shared core's TriMul provider (the kit's opt_core pin)
    return T


def trimul_tier():
    """The provider tier word this process's TriMul lever asks: `big` under --mode big (the arm composes the `fast` word), else the arm's word (`exact` | `fast`)."""
    w = TRIMUL_TIER.get(CFG["trimul"])
    return "big" if (w == "fast" and kit_mode() == "big") else w


def _trimul_w10(m):
    """The provider's ten canonical weight tensors of this module, made ONCE (the provider keys its packs on their identity)."""
    c = _cache(m)
    w = c.get("ptx1_w10")
    if w is None:
        w = c["ptx1_w10"] = dict(ln_in_w=m.layer_norm_in.weight.detach(), ln_in_b=m.layer_norm_in.bias.detach(),
                                 w_ag=m.linear_a_g.weight.detach(), w_ap=m.linear_a_p.weight.detach(),
                                 w_bg=m.linear_b_g.weight.detach(), w_bp=m.linear_b_p.weight.detach(),
                                 ln_out_w=m.layer_norm_out.weight.detach(), ln_out_b=m.layer_norm_out.bias.detach(),
                                 w_o=m.linear_z.weight.detach(), w_og=m.linear_g.weight.detach())
    return w


def trimul_select(tier, z, N, direction):
    """The provider's Selection for the tier word at this call's key (cc, precision word, C, N, direction), or CoreAside(kind, fallback) when it refuses the key BY NAME —
    decided once per (tier, key) and mirrored into COUNTS["coresel:trimul_<word>"] as one `<tier>:<prec>:C<c>:N<n>:<dir>=><row>@<cell>,x<x_stock>` | `...=>refused:<kind>` entry
    per key for the LEVER line's `selections=`."""
    T = _TM()
    zs4 = z if z.dim() == 4 else z[None]
    prec, _ = T.call_precision(zs4)
    dt = prec.replace("f32z_", "") if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec)
    cc = torch.cuda.get_device_capability(z.device)
    key = (tier, cc, prec, int(z.shape[-1]), int(N), direction)
    hit = TRIMUL_SEL.get(key)
    if hit is not None:
        return hit
    if not _TRIMUL_STACK:
        _TRIMUL_STACK.append(T.stack_word(zs4))
    note_key = "%s:%s:C%d:N%d:%s" % (tier, prec, int(z.shape[-1]), int(N), direction[:3])
    try:
        sel = T.select(cc, dt, int(z.shape[-1]), int(z.shape[-1]), int(N), direction, word=tier, residency=("fp32" if prec.startswith("f32z_") else None),
                       tf32=(prec == "tf32"), has_cueq=True, stack=_TRIMUL_STACK[0])
        note = "%s@%s,x%s%s" % (sel.row, sel.cell, sel.x_stock, "" if sel.size_measured else ",unmeasured")
    except T.Refusal as r:
        sel = CoreAside(str(r.kind), r.fallback or "stock")
        note = "refused:%s" % sel.kind
    TRIMUL_SEL[key] = sel
    COUNTS.setdefault("coresel:trimul_" + CFG["trimul"], {})["%s=>%s" % (note_key, note)] = 1
    return sel


def _provider_trimul(self, z, mask, direction, add, N, cdt):
    """Serve one TriMul call through the provider by the tier word (trimul_tier()): the row it selects for the key, served by triangle_multiplication(selection=...) with the
    shared cache.  None = upstream's own statement answers this call: the key was refused BY NAME at selection or at the call (counted `stock:trimul:<kind>` under
    COUNTS["trimul"]), or the row erred (counted `error:<Type>`, printed twice: read as partial).  Served rows are counted under COUNTS["core:trimul_<word>"] by row name."""
    T = _TM()
    tier = trimul_tier()
    sel = trimul_select(tier, z, N, direction)
    if isinstance(sel, CoreAside):
        _c("trimul", "stock:trimul:%s" % sel.kind); return None
    try:
        out = T.triangle_multiplication(z, mask, direction=direction, weights=_trimul_w10(self), selection=sel, residual=add, cache=TRIMUL_CACHE, pad=16)
    except T.Refusal as r:                                                         # a call-time refusal by name (a form the row does not take): upstream's statement
        _c("trimul", "stock:trimul:%s" % r.kind); return None
    _c("core:trimul_" + CFG["trimul"], sel.row)
    return out if out.dtype == cdt else out.to(cdt)



def _trimul_forward(self, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch"):
    mode = CFG["trimul"]
    orig = _ORIG["trimul"]
    if mode == "stock" or not (triangle_multiplicative == "cuequivariance" and self.c_z == self.c_hidden) or z.dim() not in (3, 4) or not z.is_cuda:
        if self.c_z != self.c_hidden and z.dim() in (3, 4) and z.is_cuda:      # the template embedder's TriMul (c_z 64, c_hidden 128: the stock takes its torch path): `tmpl_trimul` routes it
            out = _TEMPL.trimul_route(self, z, mask, inplace_safe, _add_with_inplace)   # through the core's provider (ptx1_templ.trimul_route: None = the stock statement answers, counted there when the lever is on)
            if out is not None:
                _c("trimul", "tmpl_trimul"); return out
        _c("trimul", "stock:%s" % ("mode" if mode == "stock" else "path"))
        return orig(self, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)
    N = z.shape[-2]
    add = bool(inplace_safe is True and _add_with_inplace)
    direction = "outgoing" if self._outgoing else "incoming"
    cdt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else self.linear_a_p.weight.dtype
    try:
        out = _provider_trimul(self, z, mask, direction, add, int(N), cdt)          # the provider's row for the tier word at this key; None = refused by name -> upstream's statement
        if out is not None:
            _c("trimul", mode); return out
    except Exception as e:
        if is_oom(e): raise                                   # an out-of-memory is the caller's to see: never rerouted (opt_core.oom)
        _c("trimul", "error:" + type(e).__name__)
        if COUNTS["trimul"].get("error:" + type(e).__name__, 0) <= 2:
            print("[levers_ptx1] trimul %s provider error -> stock for this call: %r" % (mode, e), flush=True)
    return orig(self, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative)



# =====================================================================================  triangle attention: the core's fused block (opt_core.attn.pair_fused)
# One block = prologue (LayerNorm -> q|k|v|g|pair-bias in one kernel) -> attention core -> epilogue (sigmoid gate * o @ W_o^T in one kernel), served by
# the shared core for the c=128 TriangleAttention instances (trunk pairformer, MSA-module pair stack, confidence pairformer: same class, 4 heads x 32);
# the template embedder's c=64 pairformer keeps the stock statement by name (`stock:c=64`: the core's cell table covers the fpf block at c=128).
#   gblock : impl 'fpf', ln='stock' (the module's own LayerNorm output is projected), core = the STOCK cuEquivariance kernel by name -> the exact-class
#            construction (every GEMM one ascending fp32 chain over K<=256 == cuBLAS bf16 bits on cc 9.0); bit-equal to stock on the pinned stack. Under `triexact`
#            the core is the provider's `exact` tier word per key (gblock_core): a row bit-identical to the library op where the core vouches one, the library op by name elsewhere.
#   gflash : impl 'lnl', ln='fused' (the core's lnl_fused LN+bias kernel writing LN(z) in the attention frame, one cuBLAS q|k|v|g GEMM, the lnl gate kernel + W_o GEMM),
#            core = the provider's fast tier word per key under `tricuda` (gflash_core), the fused block's own default core without it -> tier 2. No kit size floor.
# A shape outside the core's cell table is refused BEFORE any launch (pair_fused.Unsupported) and counted `fallback:<reason>`: the stock forward answers that
# call and the package reads the count as a fallback (partial activation), never silently.
def _PF():
    from opt_core.attn import pair_fused as PF          # the shared core's fused pair block (triangle attention, transition)
    return PF


def _tri_pf_weights(m, PF):
    c = _cache(m)
    if "pf_triattn" not in c:
        mha = m.mha
        for lin in (mha.linear_q, mha.linear_k, mha.linear_v, mha.linear_o, m.linear):
            assert lin.bias is None
        assert mha.linear_g is not None and mha.linear_g.bias is None
        ln = m.layer_norm
        c["pf_triattn"] = PF.pack_triattn_weights(ln_w=ln.weight, ln_b=ln.bias, w_q=mha.linear_q.weight, w_k=mha.linear_k.weight, w_v=mha.linear_v.weight,
                                                  w_g=mha.linear_g.weight, w_b=m.linear.weight, w_o=mha.linear_o.weight, n_heads=mha.no_heads, head_dim=mha.c_hidden,
                                                  eps=float(getattr(ln, "eps", 1e-5)))
    return c["pf_triattn"]


def _cueq_core(q, k, v, bias, mask5, scale):
    """The stock attention core between the core's prologue and epilogue: layers.Attention.forward's cuequivariance statement, verbatim semantics
    (q,k,v [B,I,H,J,D] unscaled; bias fp32 [B,1,H,I,J]; key mask bool [B,I,1,1,J], all-True where stock's mask is the default ones)."""
    import protenix.model.triangular.layers as TL
    if mask5 is None:
        mask5 = torch.ones(q.shape[0], q.shape[1], 1, 1, q.shape[3], dtype=torch.bool, device=q.device)
    return TL.cuequivariance_triangular_attn(q, k, v, bias, mask5, scale)[0]


# ---- the attention CORE through the shared core's triangle-attention provider (opt_core.kernels.triattn), bound by TIER WORD per key.
#   tricuda (rides gflash): the provider's `fast` tier word (`big` under --mode big) at the kit's call form
#           (TRIATTN_FORM) — whatever row the cell names on this card and stack, at every row length.
#   triexact (rides gblock): the provider's `exact` tier word in every mode — the exact-class row `triattn_exact` (bit-identical to the library op)
#           on a (card | torch | library) stack the core's cell table vouches for, the library op itself (row `cueq`, the block's own stock statement
#           handed in as `stock=`) by name everywhere else; on this kit's stack the table vouches the row on H100 (cc 9.0) inside the row's proven
#           shapes (`row=triattn_exact`), `cueq` at every other key. The exact member's own per-call census rides the LEVER line (core_member_census).
# Selection is static per (word, tier, card, dtype, head_dim, heads, row length, direction): decided ONCE per key through the provider (the card, the
# dtype under autocast, the stack's prebuilt ABI, the exact vouch key, the cell) and cached; an admitted key binds the named row with that
# Selection; a key the tier word refuses BY NAME binds the stock cuEquivariance statement for that key — counted `cueq:<refusal>` per call, the refusal
# printed once. A refusal the served row raises at call time on the tensors themselves answers that call with the stock statement, counted the same
# way; any other exception is counted `error:<Type>` (a fallback the report reads as partial) and answered by the stock statement.
CORE_WORDS = {"tricuda": {"host": "gflash", "tier": "fast"},                       # word -> the lever it rides and the provider TIER word it binds (core_tier: `big` for tricuda under --mode big;
              "triexact": {"host": "gblock", "tier": "exact"}}                     # triexact binds `exact` in every mode)
LIBRARY_OP = "cueq"                                                                # the census name of the stock cuEquivariance statement a refused key / call steps aside to BY NAME (`cueq:<refusal>`) — the library op, never a row the kit asks for
TRIATTN_FORM = "mask_bias"                                         # the trunk call form the kit hands the core: the pair bias [1,H,I,J] + a per-row boolean key mask [B,I,1,1,J] built from the pair mask (opt_core.attn.pair_fused _key_mask) — the provider's `mask_bias` form
CORE_SEL = {}                                                                      # (word, tier, cc, dtype, D, H, S, direction) -> the provider's Selection | CoreAside(kind, fallback)
CORE_NOTES = {}                                                                    # word -> {key: describe(selection) | "refused:<kind>-><fallback>"}; mirrored as COUNTS["coresel:<word>"]["<key>=><note>"] = 1 (the account the report reads carries `counts`)


class CoreAside(object):
    """A key the tier word refused BY NAME at selection: `kind` = the provider's refusal word, `fallback` = what answers instead (LIBRARY_OP: the stock statement)."""
    __slots__ = ("kind", "fallback")

    def __init__(self, kind, fallback):
        self.kind, self.fallback = str(kind), str(fallback)

    def __repr__(self):
        return "CoreAside(%r -> %r)" % (self.kind, self.fallback)


def _T():
    from opt_core.kernels import triattn as T          # the shared core's triangle-attention provider
    return T


def core_tier(word):
    """The provider tier word `word` binds in this process: tricuda -> `fast` under --mode fast, `big` under --mode big
    (CORE_WORDS[word]['tier'] outside a kit mode); triexact -> `exact` in every mode."""
    if word == "tricuda" and kit_mode() == "big":
        return "big"
    return CORE_WORDS[word]["tier"]


def _core_key(x, n_heads, head_dim, n_tokens):
    """(cc 'M.m', dtype word, head_dim, heads, row length, direction) of the block's core calls for this forward — the provider's selection key.
    The dtype is the autocast dtype (the block is reached under autocast only: the prologue hands the core q|k|v in it); direction is fwd unless
    a gradient is live."""
    dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else x.dtype
    dtype = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}.get(dt, str(dt))
    direction = "fwdbwd" if torch.is_grad_enabled() and x.requires_grad else "fwd"
    return device_cc(x.device), dtype, int(head_dim), int(n_heads), int(n_tokens), direction


def _stock_triattn(q, k, v, bias, mask=None, scale=None):
    """The kit's stock callable in the provider's calling convention (`stock=`): the stock cuEquivariance statement (_cueq_core). A stock row the
    tier word names (cueq), the exact_headsplit row (the stock op once per head) and a call the exact member `triattn_exact` refuses by name are
    served through it."""
    return _cueq_core(q, k, v, bias, mask, scale)


def _tier_select(T, tier, cc, dtype, head_dim, heads, n_tokens, direction):
    """The provider's Selection for the TIER word at this key — through opt_core.attn.pair_fused.resolve_tier_core where the core carries it (the
    per-card prebuilt ABI key of the CUDA rows and the exact tier's vouch key are the core's own readings), else the same reading spelled here."""
    cc_t = tuple(T.norm_cc(cc))
    res = getattr(_PF(), "resolve_tier_core", None)
    if res is not None:
        return res(tier, cc_t, dtype, int(head_dim), int(heads), int(n_tokens), form=TRIATTN_FORM, direction=direction)
    stack = None
    try:
        if cc_t == (9, 0):
            from opt_core.kernels.triattn import cuda_sm90a as _C
            stack = _C.stack_key()
        elif cc_t[0] == 8:
            from opt_core.kernels.triattn import triattn_native as _CC
            stack = _CC.stack_key()
    except ImportError:                                                       # a core without the row module: the provider checks any built tree itself
        stack = None
    kw = {}
    if tier == "exact" and hasattr(T, "exact_stack_key"):                     # the exact tier: this process's (card | torch | library) vouch key decides between an exact-class row and the library op
        kw["exact_stack"] = T.exact_stack_key(cc_t)
    return T.select(cc_t, dtype, int(head_dim), int(heads), int(n_tokens), direction, word=tier, stack=stack, form=TRIATTN_FORM, **kw)


def core_select(word, cc, dtype, head_dim, heads, n_tokens, direction="fwd"):
    """The provider's Selection for `word`'s TIER word at this key, or CoreAside(<refusal>, cueq) — decided once per key (CORE_SEL) and described
    once per key for the LEVER line (CORE_NOTES / COUNTS coresel:, corefact:)."""
    tier = core_tier(word)
    key = (word, tier, cc, dtype, int(head_dim), int(heads), int(n_tokens), direction)
    hit = CORE_SEL.get(key)
    if hit is not None:
        return hit
    T = _T()
    note_key = "%s:D%d:H%d:N%d%s" % (dtype, int(head_dim), int(heads), int(n_tokens), "" if direction == "fwd" else ":" + direction)
    try:
        sel = _tier_select(T, tier, cc, dtype, head_dim, heads, n_tokens, direction)
        note = T.describe(sel)                                              # row= word= class= cell= measured= x_stock= (the provider's one line)
    except T.Refusal as r:                                                  # the tier word cannot serve this key on this card / stack / class: the stock statement, by name
        sel = CoreAside(r.kind, LIBRARY_OP)
        note = "refused:%s->%s" % (sel.kind, sel.fallback)
        print("[levers_ptx1] tri-attention core %s (tier %s) refused by name at cc %s %s D%d H%d N%d (%s) -> the stock cuEquivariance statement for this key" % (word, tier, cc, dtype, int(head_dim), int(heads), int(n_tokens), r.kind), flush=True)
    CORE_SEL[key] = sel
    CORE_NOTES.setdefault(word, {})[note_key] = note
    COUNTS.setdefault("coresel:" + word, {})["%s|%s=>%s" % (cc, note_key.replace(":", "|"), CORE_NOTES[word][note_key])] = 1
    facts = COUNTS.setdefault("corefact:" + word, {})
    facts["tier=%s" % tier] = 1                                             # the tier word bound (report: `tier=` on the LEVER line)
    if word == "tricuda":
        facts["form=%s" % TRIATTN_FORM] = 1                                 # the call form the cell was asked for
        try:
            if tuple(T.norm_cc(cc)) == (9, 0):                              # the sm_90a CUDA row's prebuilt for this stack (a fact of the box, whichever row the cell named)
                from opt_core.kernels.triattn import cuda_sm90a as _C
                sk = _C.stack_key()
                facts["prebuilt=%s:%s" % (sk, "built" if sk in T.stacks_built() else "missing")] = 1
        except Exception:
            pass
    elif word == "triexact":
        try:                                                                # the exact tier's vouch key of this process (card | torch | library): an exact-class row serves only on a key
            facts["exact_stack=%s" % T.exact_stack_key(tuple(T.norm_cc(cc)))] = 1   # the core's cell table records; the library op (row cueq) by name on every other key
        except Exception:
            facts["exact_stack=unknown"] = 1
    return sel


def _count_core(word, token):
    d = COUNTS.setdefault("core:" + word, {})
    d[token] = d.get(token, 0) + 1


def _provider_core(word, sel):
    """The block's core callable serving the provider row of `sel` through the provider face (word = the tier word, selection = the cached decision,
    stock = the kit's stock statement); a per-call refusal by name -> the stock statement for that call, counted `cueq:<kind>`; any other error ->
    counted `error:<Type>`, the stock statement."""
    tier = sel.word if getattr(sel, "word", None) else core_tier(word)

    def core(q, k, v, bias, mask5, scale):
        T = _T()
        try:
            o = T.triangle_attention(q, k, v, bias, mask5, scale, word=tier, selection=sel, form=TRIATTN_FORM, stock=_stock_triattn)
        except T.Refusal as r:
            tok = "%s:%s" % (LIBRARY_OP, r.kind)
            _count_core(word, tok)
            if COUNTS["core:" + word].get(tok, 0) <= 1:
                print("[levers_ptx1] tri-attention core %s: row %s refused this call by name (%s) -> the stock cuEquivariance statement" % (word, sel.row, r.kind), flush=True)
            return _cueq_core(q, k, v, bias, mask5, scale)
        except Exception as e:
            if is_oom(e):
                raise
            _count_core(word, "error:" + type(e).__name__)
            if COUNTS["core:" + word].get("error:" + type(e).__name__, 0) <= 2:
                print("[levers_ptx1] tri-attention core %s: row %s error -> the stock cuEquivariance statement for this call: %r" % (word, sel.row, e), flush=True)
            return _cueq_core(q, k, v, bias, mask5, scale)
        _count_core(word, sel.row)
        return o
    return core


def gblock_core(x, n_heads, head_dim, n_tokens):
    """gblock's attention core for this forward: the stock cuEquivariance statement (the block replaces only the projections around it); under
    `triexact` the provider's `exact` tier word for the key, served through the provider — the exact-class row `triattn_exact` where the core's
    cell table vouches it for this stack, the same stock statement (row cueq, handed in as stock=) by name where it does not; a key the word
    refuses by name binds the stock statement directly, counted cueq:<refusal>."""
    if not CFG["triexact"]:
        return _cueq_core
    cc, dtype, D, H, S, direction = _core_key(x, n_heads, head_dim, n_tokens)
    sel = core_select("triexact", cc, dtype, D, H, S, direction)
    if isinstance(sel, CoreAside):
        _count_core("triexact", "%s:%s" % (sel.fallback, sel.kind))
        return _cueq_core
    return _provider_core("triexact", sel)


def core_member_census(word):
    """The exact member's own per-call census behind `word` (opt_core.kernels.triattn.exact_member: `served` by the member's kernel, `calls` handed
    to it, `refused:<reason>` answered by the stock statement it was handed), mirrored as COUNTS['coremember:<word>'] for the LEVER line
    (report.core_evidence: member= / member_refused=); {} when the word is off, binds no exact tier, or the core carries no member."""
    if not CFG.get(word) or CORE_WORDS.get(word, {}).get("tier") != "exact":
        return {}
    try:
        from opt_core.kernels.triattn import exact_member as EM
        c = EM.counts()
    except Exception:                                                       # an older core: no member module, nothing to mirror
        return {}
    d = {"served": int(c.get("served") or 0), "calls": int(c.get("calls") or 0)}
    d.update({"refused:%s" % w: int(n) for w, n in (c.get("refused") or {}).items() if n})
    COUNTS["coremember:" + word] = d
    return dict(d)


def gflash_core(x, n_heads, head_dim, n_tokens):
    """gflash's attention core for this forward: under `tricuda` the provider's fast-class tier word for the key (core_tier: `fast` in --mode fast,
    `big` in --mode big) — the row its cell names, served through the provider; the stock cuEquivariance statement inside
    the block where the tier word refuses the key by name; the fused block's own default core (opt_core.attn.pair_fused DEFAULT_CORE: the shared
    core's choice, no kit word) where the word is off. COUNTS['gflash_core'] names the core per call for the LEVER line's `cores=`."""
    d = COUNTS.setdefault("gflash_core", {})
    if not CFG["tricuda"]:
        default = getattr(_PF(), "DEFAULT_CORE", "default")
        d[default] = d.get(default, 0) + 1
        return default
    cc, dtype, D, H, S, direction = _core_key(x, n_heads, head_dim, n_tokens)
    sel = core_select("tricuda", cc, dtype, D, H, S, direction)
    if isinstance(sel, CoreAside):
        tok = "%s:%s" % (sel.fallback, sel.kind)
        d[tok] = d.get(tok, 0) + 1
        _count_core("tricuda", tok)
        return _cueq_core
    d["tricuda:" + sel.row] = d.get("tricuda:" + sel.row, 0) + 1
    return _provider_core("tricuda", sel)


def describe_cores():
    """The provider words' account for the LEVER lines: per word {requested, host, host_on, tier, fallback, counts, selections, stacks_built? (tricuda),
    member? (triexact: the exact member's census, core_member_census)}."""
    out = {}
    for word, spec in CORE_WORDS.items():
        try:
            tier = core_tier(word)
        except Exception:                                                   # the core not importable in this process: the word's default tier
            tier = spec["tier"]
        d = {"requested": bool(CFG.get(word)), "host": spec["host"], "host_on": bool(CFG.get(spec["host"])), "tier": tier, "fallback": LIBRARY_OP,
             "counts": dict(COUNTS.get("core:" + word) or {}), "selections": dict(CORE_NOTES.get(word) or {})}
        if word == "tricuda" and CFG.get(word):
            try:
                d["stacks_built"] = list(_T().stacks_built())
            except Exception as e:
                d["stacks_built"] = "error:%s" % type(e).__name__
        elif spec["tier"] == "exact" and CFG.get(word):
            d["member"] = core_member_census(word)
        out[word] = d
    return out


def _tri_forward(self, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
    orig = _ORIG["triattn"]
    lever = "gflash" if CFG["gflash"] else ("gblock" if CFG["gblock"] else None)
    if lever is None:
        return orig(self, x, mask, chunk_size, triangle_attention, inplace_safe)
    N = x.shape[-2]
    if triangle_attention != "cuequivariance":
        _c("triattn", "stock:path"); return orig(self, x, mask, chunk_size, triangle_attention, inplace_safe)
    if chunk_size is not None and N > TRIATTN_CEILING_TOKENS:       # stock row-chunks above 1024 tokens for the torch path's N^2 x H logits; the fused cores hold no
        _c("triattn", "stock:chunk"); return orig(self, x, mask, chunk_size, triangle_attention, inplace_safe)   # logits, so the block serves unchunked up to 2048 (rows are independent: same arithmetic per row)
    if not x.is_cuda or not torch.is_autocast_enabled() or x.dim() not in (3, 4) or x.shape[-3] != N or N <= TRIATTN_GATE_TOKENS:   # N<=16: stock itself leaves cuEquivariance for torch
        _c("triattn", "stock:gate"); return orig(self, x, mask, chunk_size, triangle_attention, inplace_safe)
    adt = torch.get_autocast_dtype("cuda")
    if adt != torch.bfloat16:                                       # --dtype fp32 | fp16: autocast on at another compute dtype — the block's cells (trunk and template) are bf16:
        _c("triattn", "stock:dtype:%s" % str(adt).rsplit(".", 1)[-1])   # the stock statement BY NAME, before any cell lookup or launch (never a per-call refusal / fallback word)
        return orig(self, x, mask, chunk_size, triangle_attention, inplace_safe)
    if x.shape[-1] != 128:                                          # the template embedder's pairformer (c = 64, 2 blocks): `tmpl_triatt` sends it through the same block when the core's cells serve
        word = _TEMPL.triatt_gate(lever, self, x)                   # its shape on this card (ptx1_templ.triatt_gate: None = serve; else the by-name word — `stock:c=64` with the lever off)
        if word is not None:
            _c("triattn", word); return _TEMPL.stock_forward(orig, self, x, mask, chunk_size, triangle_attention, inplace_safe)   # the stock statement, routed by name
    if lever == "gblock":                                           # the exact-class block's served rows on this card (the core's exact_rows rule): outside them the stock forward, by name
        word = exact_rows_word("gblock", x)                                    # the core's served-row rule for the exact block on this card (opt_core.attn.pair_fused exact_rows)
        if word is not None:
            _c("triattn", word); return orig(self, x, mask, chunk_size, triangle_attention, inplace_safe)
    PF = _PF()
    W = _tri_pf_weights(self, PF)
    ending = not self.starting
    if lever == "gflash":                                           # every row length the block serves: the provider's tier word per key under `tricuda`, the block's own default core without it
        impl, core, ln, tag = GFLASH_PROLOGUE, gflash_core(x, self.mha.no_heads, self.mha.c_hidden, N), "fused", "gflash"
    else:                                                           # the exact-class construction; core = the stock cuEquivariance statement, the provider's `exact` tier word per key under `triexact`
        impl, core, ln, tag = GBLOCK_PROLOGUE, gblock_core(x, self.mha.no_heads, self.mha.c_hidden, N), "stock", "gblock"
    x_ln = None
    if ln == "stock":                                               # the stock statement: LayerNorm of the (transposed, for the ending node) pair tensor
        x_ln = self.layer_norm(x.transpose(-2, -3) if ending else x)
        if not x_ln.is_contiguous():
            x_ln = x_ln.contiguous()
    key_mask = mask
    if key_mask is not None and key_mask.dtype != torch.bool:
        key_mask = (self.inf * (key_mask - 1)) == 0                 # stock's key-mask statement ((inf * (mask - 1)) == 0), in z's frame; the core transposes it for the ending node
    refused = None
    try:
        if x.shape[-1] != 128:                                      # a template call tmpl_triatt routes: `tmpl_pairfused` may choose the fpf prologue / epilogue with the LayerNorm fused
            impl, ln, x_ln = _TEMPL.triatt_impl(lever, impl, ln, x_ln, int(x.shape[-1]))   # (ptx1_templ.triatt_impl; the trunk lever's choice otherwise)
        u = PF.tri_attn_block(x, W, key_mask, ending=ending, residual=False, impl=impl, core=core, ln=ln, x_ln=x_ln, scale=1.0 / math.sqrt(self.mha.c_hidden))
    except PF.Unsupported as e:                                     # refused before any launch: keep the WORD (a bound exception would keep this frame's tensors alive until GC)
        refused = e.reason
    if refused is not None:                                         # the stock statement answers this call, counted as a fallback (the package reads it as partial)
        _c("triattn", "fallback:%s:%s" % (lever, refused))
        if COUNTS["triattn"].get("fallback:%s:%s" % (lever, refused), 0) <= 2:
            print("[levers_ptx1] tri-attention %s refused by the core (%s) -> stock for this call" % (lever, refused), flush=True)
        return orig(self, x, mask, chunk_size, triangle_attention, inplace_safe)
    _c("triattn", tag)
    return u


# =====================================================================================  transition: the shared core's transition PROVIDER by TIER word (opt_core.kernels.transition)
#   xtr : --mode exact's word. The pair transitions (c=128, hidden 512: trunk, MSA-module pair stack, confidence pairformer) and the MSA transition
#         (c=64, hidden 256) ask select('exact') with the module's own LayerNorm output handed in (the exact construction): the provider's
#         bitwise-vouched kernel row for this stack, size and row count serves; a cell whose exact tier IS the stock statement (the MSA transition's,
#         every card in the table), a stack / size without a vouch, or a row count outside the card's served rows (opt_core.attn.pair_fused exact_rows:
#         cc 8.0) is answered by the stock module BY NAME — bytes equal to --mode off either way.
#   ttr : --mode fast's word (`fast`; `big` under --mode big, transition_tier()): the cell table's fast-class row for the same calls (tolerance class).
# A provider Refusal is never substituted by a kit statement: the stock module answers the call, counted by name. Both leave the single-representation
# transitions (c_s=384, fp32: N rows, not N*N), any call outside bf16 autocast, and shapes without a cell (counted `stock:<lever>:<refusal>`; the template
# Pairformer's c=64 x 128 transition is `tmpl_xtr`'s — ptx1_templ.transition_route) to the stock statement by name.
TRANSITION = {"selections": {}, "refusals": {}}                 # "<lever>:<family>:c<C>x<H>" -> the provider's Selection line served (LEVER line `selections=`); refusal kind -> count
TRANSITION_FAMILY = {128: "pair", 64: "rows"}                   # channel width -> the provider's cell family: the pair transitions' [.., N, N, 128] rows are `pair` cells keyed by N; the MSA
                                                                # transition's [S, N, 64] rows the `rows` cells (N*N rows of width 64) keyed by the equivalent N = sqrt(rows)


def _TR():
    from opt_core.kernels import transition as T                # the shared core's transition provider: one face over every carried row + TRANSITION_CELLS.json
    return T


def transition_tier(lever=None):
    """The provider tier word the transition lever asks: xtr -> `exact`; ttr -> `big` under --mode big (the arm composes the fast words), else `fast`."""
    lever = lever or ("xtr" if CFG["xtr"] else "ttr")
    if lever == "xtr":
        return "exact"
    return "big" if kit_mode() == "big" else "fast"


def _provider_weights(m, T):
    c = _cache(m)
    if "tr_provider" not in c:
        ln = m.layernorm1
        for lin in (m.linear_no_bias_a, m.linear_no_bias_b, m.linear_no_bias):
            assert getattr(lin, "bias", None) is None
        c["tr_provider"] = T.pack(w_o=m.linear_no_bias.weight, w_a=m.linear_no_bias_a.weight, w_b=m.linear_no_bias_b.weight, ln_w=ln.weight, ln_b=getattr(ln, "bias", None),
                                  eps=float(getattr(ln, "eps", 1e-5)), device=m.linear_no_bias.weight.device)
    return c["tr_provider"]


def _transition_refused(lever, kind, detail=""):
    """Count a provider refusal BY NAME (the stock module answers the call): the card's served-row rule keeps its two census words
    (stock:below_min_rows / stock:above_max_rows); any other kind is `stock:<lever>:<kind>`. Printed once per distinct word."""
    word = _rows_census_word(kind) if str(kind).startswith("exact_rows_") else None
    word = word or "stock:%s:%s" % (lever, str(kind)[:96])
    _c("transition", word)
    TRANSITION["refusals"][str(kind)] = TRANSITION["refusals"].get(str(kind), 0) + 1
    if COUNTS["transition"].get(word, 0) <= 1 and not str(kind).startswith("exact_rows_"):
        print("[levers_ptx1] transition %s (tier word %s) refused by the provider BY NAME (%s%s) -> the stock module for such calls" % (
            lever, transition_tier(lever), kind, (": " + detail[:160]) if detail else ""), flush=True)
    return word


def _transition_forward(self, x):
    orig = _ORIG["transition"]
    lever = "xtr" if CFG["xtr"] else ("ttr" if CFG["ttr"] else None)
    if lever is None:
        return orig(self, x)
    C = int(x.shape[-1]); HID = int(self.linear_no_bias_a.weight.shape[0])
    if not x.is_cuda or not torch.is_autocast_enabled() or x.dtype != torch.bfloat16 or C not in TRANSITION_FAMILY:
        _c("transition", "stock:C=%d" % C); return orig(self, x)     # the single-representation transitions (c_s=384, fp32) and any call outside bf16 autocast: the stock statement
    rows = x.numel() // C
    if C == 64 and HID == 128:                                      # the template Pairformer's transition (c 64, n 2): `tmpl_xtr` serves it in the exact construction at that width
        if rows < TRANSITION_GATE_ROWS:                             # (ptx1_templ.transition_route: None = the stock statement answers, counted there when the lever is on); its
            _c("transition", "stock:C=%d" % C); return orig(self, x) # small-row envelope (>= 4096 rows) is that route's contract
        y = _TEMPL.transition_route(self, x)
        if y is not None:
            _c("transition", "tmpl_xtr"); return y
        return orig(self, x)
    try:
        T = _TR()
    except Exception as e:                                          # a core without the provider: named, the stock module (the kit's core gate refuses such a core before this)
        _c("transition", "stock:%s:no_provider" % lever)
        if COUNTS["transition"].get("stock:%s:no_provider" % lever, 0) <= 1:
            print("[levers_ptx1] transition %s: the core has no kernels.transition provider (%s: %s) -> the stock module" % (lever, type(e).__name__, e), flush=True)
        return orig(self, x)
    word = transition_tier(lever)
    family = TRANSITION_FAMILY[C]
    n_eq = int(x.shape[-2]) if (x.dim() >= 3 and x.shape[-2] == x.shape[-3]) else int(round(math.sqrt(rows)))
    xs = x if x.stride(-1) == 1 else x.contiguous()
    try:
        sel = T.select(word, c=C, hidden=HID, n_tokens=n_eq, family=family, timing="eager", device=xs.device, rows_count=rows, ln_given=(lever == "xtr"))
    except T.Refusal as r:                                          # no cell for this shape / card, no vouch on this stack or size, the card's served-row rule, a missing binary: by name
        _transition_refused(lever, r.kind, getattr(r, "detail", "") or ""); return orig(self, x)
    if sel.row in T.STOCK_ROWS:                                     # the tier's answer for this cell IS the stock statement: the module's own forward, by name
        _c("transition", "stock:%s:row=%s" % (lever, sel.row)); return orig(self, x)
    refused = None
    try:
        y, sel = T.transition(xs, _provider_weights(self, T), word=word, residual=False, x_ln=(self.layernorm1(xs) if lever == "xtr" else None),
                              n_tokens=n_eq, family=family, timing="eager")
    except T.Refusal as r:                                          # keep the word, never the exception (see _tri_forward)
        refused = (r.kind, getattr(r, "detail", "") or "")
    except Exception as e:
        if is_oom(e): raise
        w = "fallback:%s:error:%s" % (lever, type(e).__name__)
        _c("transition", w)
        if COUNTS["transition"].get(w, 0) <= 2:
            print("[levers_ptx1] transition %s: provider error %r -> the stock module for this call" % (lever, e), flush=True)
        return orig(self, x)
    if refused is not None:
        _transition_refused(lever, refused[0], refused[1]); return orig(self, x)
    _c("transition", "%s:C=%d" % (lever, C))
    key = "%s:%s:c%dx%d" % (lever, family, C, HID)
    if key not in TRANSITION["selections"]:
        TRANSITION["selections"][key] = sel.line()
    return y


# =====================================================================================  diffusion sampler: graphed denoiser step (+ DiT hoist)
# lib/kit112_src: `infopt_graphs.protenix` (GraphedDenoiseLoop: the denoiser step captured in a CUDA graph per (N_atom, N_token, N_sample, dtype);
# every random draw made OUTSIDE the graph in stock order) + `dit_hoist` (step-invariant hoist of the pair-bias chain / conditioning terms).
# The graphed loop reads `protenix.tfg.parse_tfg_config` (training-free guidance, a protenix 2.x module absent in 1.1.0): _ensure_tfg_shim
# registers a module that reports guidance disabled.
# Precondition: Protenix's fast LayerNorm extension launches on the legacy default stream (not capturable) -> the stream-correct rebuild
# (fastln_prebuilt .so when torch/CUDA/source hashes match, else fastln_stream source rebuild with nvcc, once per machine and stack) is installed
# first; it is kernel-level bit-equal to the original (checked in-process by the installer).  dit_hoist has no uninstall: once a 'hoist' arm ran,
# later arms in the same process carry its (mode 'off') wrappers -> callers order hoist arms last.
_MODEL = None
SG = {"graphs": False, "hoist": None, "fastln": None, "handles": None, "errors": [], "prep": False, "prep_aside": None}


def bind_model(model):
    global _MODEL
    _MODEL = model


def _ensure_tfg_shim():
    try:
        import protenix.tfg  # noqa: F401  (2.x)
    except Exception:
        import types, protenix
        mod = types.ModuleType("protenix.tfg")

        class _Cfg:
            enable = False

        mod.parse_tfg_config = lambda cfg=None: _Cfg()
        mod.TFGEngine = None
        sys.modules["protenix.tfg"] = mod; protenix.tfg = mod


def ensure_stream_ln():
    if SG["fastln"] is not None:
        return SG["fastln"]
    if os.environ.get("LAYERNORM_TYPE", "fast_layernorm") != "fast_layernorm":   # unset reads as the upstream reads it (protenix 1.1.0 triangular/layers.py:33, runner/inference.py:121: the fast LayerNorm) — the stream-correct build is needed then too; only an explicit other value (e.g. `torch`) makes it unneeded
        SG["fastln"] = {"installed": True, "how": "not-needed(LAYERNORM_TYPE=%s)" % os.environ.get("LAYERNORM_TYPE")}; return SG["fastln"]
    import protenix.model.layer_norm.layer_norm as LN
    rep = {"installed": False}
    t0 = time.time()
    try:
        import fastln_prebuilt as FPB
        pre_dir = os.path.join(os.path.dirname(os.path.abspath(FPB.__file__)), "fastln_prebuilt")
        r = FPB.install_prebuilt_fastln(pre_dir, require_bitwise=True)
        rep = {"installed": bool(r.get("installed")), "how": "prebuilt", "reason": r.get("reason"), "bitwise": r.get("bitwise", {}).get("all_bitwise_equal") if isinstance(r.get("bitwise"), dict) else r.get("bitwise")}
    except Exception as e:
        if is_oom(e): raise                                   # an out-of-memory is the caller's to see: never rerouted (opt_core.oom)
        rep = {"installed": False, "how": "prebuilt", "reason": repr(e)[:300]}
    if not rep["installed"] or not getattr(LN, "_infopt_stream_patched", False):
        try:
            from infopt_graphs.protenix.fastln_stream import install_stream_correct_fastln
            r = install_stream_correct_fastln()
            rep = {"installed": bool(getattr(LN, "_infopt_stream_patched", False)), "how": "source-rebuild", "prebuilt_reason": rep.get("reason"),
                   "bitwise": (r.get("bitwise", {}) or {}).get("all_bitwise_equal") if isinstance(r, dict) else None, "prepare_s": (r or {}).get("prepare_s") if isinstance(r, dict) else None}
        except Exception as e:
            if is_oom(e): raise                                   # an out-of-memory is the caller's to see: never rerouted (opt_core.oom)
            rep = {"installed": False, "how": "source-rebuild", "reason": repr(e)[:300], "prebuilt_reason": rep.get("reason")}
    rep["s"] = round(time.time() - t0, 1)
    SG["fastln"] = rep
    print("[levers_ptx1] stream-correct fast-LN:", rep, flush=True)
    return rep


def set_sampler(graphs, hoist, prep=False):
    """prep = lever sampler_prep: the graphed loop is built with its host-path parts on (infopt_graphs.protenix.sampler_prep); it rides the
    sampler graph — without `sg` (ablated, or above the graph cap) it steps aside by name (SG["prep_aside"])."""
    import infopt_graphs.protenix as IG
    SG["prep"] = bool(prep and graphs); SG["prep_aside"] = "no_sampler_graph" if (prep and not graphs) else None
    if graphs and not SG["graphs"]:
        assert _MODEL is not None, "bind_model(model) first"
        ln = ensure_stream_ln()
        if not ln.get("installed"):
            SG["errors"].append("fastln:" + str(ln.get("reason"))); raise RuntimeError("stream-correct fast-LN unavailable -> refusing sampler graphs: %s" % (ln,))
        _ensure_tfg_shim()
        h = IG.install(_MODEL, sampler=True, trunk=False, pool="private", fastln_stream_fix=False, sampler_prep=bool(prep))   # lever sampler_prep: the loop's host path
        SG["handles"] = h; SG["graphs"] = True
    elif not graphs and SG["graphs"]:
        IG.uninstall(_MODEL); SG["graphs"] = False; SG["handles"] = None
    if hoist and SG["hoist"] is None:
        import dit_hoist
        sys.modules.setdefault("biascache_static", dit_hoist)
        loop = (SG["handles"] or {}).get("sampler") if graphs else None
        SG["hoist"] = dit_hoist.install(_MODEL, sampler=loop)
    elif hoist and SG["hoist"] is not None and graphs:
        loop = (SG["handles"] or {}).get("sampler")
        if loop is not None and getattr(loop, "biascache", None) is None:
            loop.biascache = SG["hoist"]
    if not hoist and SG["hoist"] is not None:
        try:
            SG["hoist"].set_mode("off")
        except Exception as e:
            SG["errors"].append("hoist-off:" + repr(e)[:100])


def set_keep_pool(on):
    """The keep_pool lever (lib/ptx1_keep_pool.py): install over the empty_cache callable current NOW (after set_sampler: the sampler-graph
    guard, when `sg` is on, sits underneath and keeps relaying); off = uninstall. One arm per process."""
    import ptx1_keep_pool as KP
    if on:
        KP.install()
    else:
        KP.uninstall()
    return KP.report()


def set_lazy_init(on):
    """The lazy_init lever is installed by the package (stack._wrap_runner -> ptx1_lazy_init.install) BEFORE the stock runner constructs the
    model; by the time apply() runs the construction is done, so the arm word is only checked against that install: without it the word is
    left unset in CFG and the activation names the lever as not applied (partial, by name) — never a silent stock initialisation claimed as lazy."""
    if not on:
        return
    try:
        import ptx1_lazy_init as LZ
        ok = bool(LZ.report().get("installed")) and int(LZ.report().get("constructs") or 0) > 0
    except Exception:                                      # noqa: BLE001
        ok = False
    if not ok:
        CFG["lazy_init"] = False


def describe_lazy_init():
    try:
        import ptx1_lazy_init as LZ
        return LZ.report()
    except Exception as e:                                 # noqa: BLE001
        return {"installed": False, "error": repr(e)[:200]}


def set_summary_host(on):
    """The summary_hostidx lever (lib/ptx1_summary_host.py): install over protenix.model.sample_confidence.compute_full_data_and_summary (a
    signature drift raises by name; a function already replaced by the row-sharded line = step aside by name); off = uninstall."""
    import ptx1_summary_host as SH
    if on:
        SUMHOST["mark"] = SH.install()
    else:
        SH.uninstall(); SUMHOST["mark"] = None
    return SH.report()


SUMHOST = {"mark": None}
DITX = {"mark": None, "card_off": None, "error": None}


def _dit_card_off(reason):
    """True when the package's refusal names the card / stack (no prebuilt for this stack key, cc outside the prebuilt's arch list, no CUDA):
    the lever steps aside by name there (another card's row never refuses); any other refusal (digest / load-check / ABI) is a failed install."""
    r = str(reason)
    return any(t in r for t in ("no prebuilt for stack", "not in the prebuilt arch list", "no CUDA device"))


def set_dit_attn_exact(on):
    """The dit_attn_exact lever (opt_core.kernels.apb row dit_exact): its apply() loads the prebuilt for
    this stack (manifest / sha256 / 3-case bit check vs torch SDPA in this process) and installs the route on primitives._attention. The
    package reads its own switch word (PTX_DIT_ATTN_EXACT); the arm word sets it for this process. No prebuilt for the card = card_off (named)."""
    if not on:
        return describe_dit_attn_exact()
    os.environ[DIT_ATTN_EXACT_ENV] = "1"
    from opt_core.kernels import apb as _APB
    DX = _APB.carried_module("dit_exact")                                              # the shared core's carried dit_exact package, by row word
    try:
        DITX["mark"] = DX.apply()
    except RuntimeError as e:                           # the package refuses by name: card/stack without a prebuilt -> step aside by name; anything else -> failed install (partial)
        if _dit_card_off(e):
            DITX["card_off"] = str(e)
        else:
            DITX["error"] = str(e)[:300]
    return describe_dit_attn_exact()


def describe_dit_attn_exact():
    out = {"installed": False, "mark": DITX["mark"], "card_off": DITX["card_off"], "error": DITX["error"]}
    try:
        from opt_core.kernels import apb as _APB
        DX = _APB.carried_module("dit_exact")
        rep = DX.report()
        out.update(installed=bool(rep.get("installed")), calls=int(rep.get("calls") or 0), routes=dict(rep.get("routes") or {}), loadcheck=rep.get("loadcheck"))
    except Exception as e:
        out["error"] = out["error"] or repr(e)[:200]
    return out


def describe_summary_host():
    try:
        import ptx1_summary_host as SH
        return dict(SH.report(), mark=SUMHOST["mark"])
    except Exception as e:
        return {"installed": False, "error": repr(e)[:200]}


def describe_keep_pool():
    try:
        import ptx1_keep_pool as KP
        return KP.report()
    except Exception as e:                              # never imported (lever off) or unavailable: the account says so
        return {"installed": False, "error": repr(e)[:200]}


def describe_sampler():
    out = {"graphs": SG["graphs"], "hoist_installed": SG["hoist"] is not None, "fastln": SG["fastln"], "errors": SG["errors"][-3:],
           "prep": {"on": False, "aside": SG.get("prep_aside")}}                       # lever sampler_prep: the loop's prep_report() below when the graph loop is installed
    try:
        h = (SG["handles"] or {}).get("sampler")
        if h is not None and hasattr(h, "summary"):
            s = h.summary(); out["sampler"] = {k: v for k, v in (s.items() if isinstance(s, dict) else []) if k not in ("events",) and isinstance(v, (int, float, str, bool))}
        if h is not None and hasattr(h, "prep_report"):
            out["prep"] = dict(h.prep_report(), aside=SG.get("prep_aside"))
    except Exception as e:
        out["sampler_err"] = repr(e)[:120]
    try:
        if SG["hoist"] is not None:
            import dit_hoist
            out["hoist"] = {k: v for k, v in getattr(dit_hoist, "STATS", {}).items() if isinstance(v, (int, float, str, bool))}
    except Exception as e:
        out["hoist_err"] = repr(e)[:120]
    return out


# =====================================================================================  install / apply
def _install():
    import protenix.model.triangular.triangular as TT
    import protenix.model.modules.primitives as PR
    import protenix.model.modules.transformer as TF
    if _ORIG:
        return
    _ORIG["trimul"] = TT.TriangleMultiplicativeUpdate.forward
    _ORIG["triattn"] = TT.TriangleAttention.forward
    _ORIG["transition"] = PR.Transition.forward
    TT.TriangleMultiplicativeUpdate.forward = _trimul_forward
    TT.TriangleAttention.forward = _tri_forward
    PR.Transition.forward = _transition_forward


def apply(arm):
    """arm: 'stock' | '<trimul>[+lever...]' (levers: LEVER_NAMES).  Returns the parsed CFG.  'stock' leaves the wrappers installed but every one of them
    takes the saved original path (counted).  One arm per process."""
    _install()
    parts = [p for p in arm.replace(",", "+").split("+") if p]
    tri = parts[0] if parts and parts[0] in ("stock", "fast", "exact") else "stock"
    lev = set(parts[1:] if parts and parts[0] in ("stock", "fast", "exact") else parts)
    unknown = lev - set(LEVER_NAMES)
    if unknown:
        raise ValueError("unknown levers %s in arm %r" % (sorted(unknown), arm))
    CFG.update(trimul=tri, **{name: (name in lev) for name in LEVER_NAMES})
    if CFG["gblock"] or CFG["gflash"] or CFG["xtr"] or CFG["ttr"] or CFG["tmpl_triatt"]:
        try:
            _PF()
        except ImportError as e:                       # a shared core without attn.pair_fused: the lever cannot attach -> refused by name, never a silent stock run
            raise RuntimeError("levers %s refused: opt_core.attn.pair_fused unavailable (%r)" % (sorted(l for l in ("gblock", "gflash", "xtr", "ttr") if CFG[l]), e))
    if CFG["dit_attn_exact"]:
        set_dit_attn_exact(True)                        # before the sampler levers: the route is on primitives._attention when the denoiser step is first captured
    set_lazy_init(CFG["lazy_init"])                     # installed ahead of the model by the package; checked here
    set_sampler(CFG["sg"], CFG["hoist"], CFG["sampler_prep"])
    set_keep_pool(CFG["keep_pool"])                     # after the sampler levers: wraps whatever empty_cache is current (the graph guard included)
    _TEMPL.apply(CFG)                                   # the template-embedder levers: template_dedupe (the TemplateEmbedder class patch), tmpl_triatt (the c=64 route of _tri_forward), tmpl_trimul (the c_z!=c_hidden route of _trimul_forward)
    set_summary_host(CFG["summary_hostidx"])
    set_apb(CFG["ditattn"], CFG["ditattnfp16"], CFG["atomattn"])
    set_trunk2(CFG["pfattn"], CFG["opm_fused"], CFG["pwa_fused"])   # after the sampler levers, like set_apb: instance-level statements on the bound model (ptxfpf/trunk2_ptx1.py)
    set_ditfast(CFG["cond_dedupe"], CFG["dit_fused"], CFG["dit_lowp"], CFG["atom_fused"], CFG["atom_attn_exact"])   # after the apb levers: dit_fused / atom_fused ride ditattn / atomattn's kernels (ptxfpf/ditfast_ptx1.py)
    return dict(CFG)


def set_apb(ditattn, ditattnfp16, atomattn):
    """The sampler attention-with-pair-bias levers (ptxfpf/apb_ptx1.py over opt_core.kernels.apb rows fpf_apb / fpf_atom): installed AFTER the sampler graphs / hoist on the
    bound model's DiffusionModule attention modules; their census words live in COUNTS["dit"] / COUNTS["atom"] (the adapter's own dicts)."""
    if not (ditattn or ditattnfp16 or atomattn) and "apb_ptx1" not in sys.modules:
        return
    import apb_ptx1
    if (ditattn or ditattnfp16 or atomattn) and _MODEL is None:
        raise RuntimeError("levers %s refused: bind_model(model) first" % sorted(w for w in apb_ptx1.LEVERS if CFG.get(w)))
    apb_ptx1.apply(_MODEL, ditattn=bool(ditattn), ditattnfp16=bool(ditattnfp16), atomattn=bool(atomattn))
    COUNTS["dit"] = apb_ptx1.COUNTS["dit"]; COUNTS["atom"] = apb_ptx1.COUNTS["atom"]


def set_trunk2(pfattn, opm_fused, pwa_fused):
    """The trunk-side fused-kernel levers (ptxfpf/trunk2_ptx1.py over opt_core.kernels.apb + lib/protenix_fpf_msa): the pairformer
    single attention with pair bias (`pfattn`) and the MSA module's OuterProductMean / MSAPairWeightedAveraging (`opm_fused` / `pwa_fused`) on the
    bound model; their census words live in COUNTS["pf"] / COUNTS["opm"] / COUNTS["pwa"] (the adapter's own dicts)."""
    if not (pfattn or opm_fused or pwa_fused) and "trunk2_ptx1" not in sys.modules:
        return
    import trunk2_ptx1
    if (pfattn or opm_fused or pwa_fused) and _MODEL is None:
        raise RuntimeError("levers %s refused: bind_model(model) first" % sorted(w for w in trunk2_ptx1.LEVERS if CFG.get(w)))
    trunk2_ptx1.apply(_MODEL, pfattn=bool(pfattn), opm_fused=bool(opm_fused), pwa_fused=bool(pwa_fused))
    for k in ("pf", "opm", "pwa"):
        COUNTS[k] = trunk2_ptx1.COUNTS[k]


def set_ditfast(cond_dedupe, dit_fused, dit_lowp, atom_fused, atom_attn_exact):
    """The fused diffusion-sampler levers (ptxfpf/ditfast_ptx1.py over lib/protenix_fpf_ditfast and, for atom_attn_exact, opt_core.kernels.apb's exact tier word):
    installed AFTER the sampler graphs / hoist and after the apb levers on the bound model (dit_fused / atom_fused REQUIRE ditattn / atomattn engaged;
    dit_lowp rides dit_fused; atom_attn_exact steps aside by name beside atomattn / atom_fused); their account is ditfast_ptx1.describe()."""
    words = dict(cond_dedupe=bool(cond_dedupe), dit_fused=bool(dit_fused), dit_lowp=bool(dit_lowp), atom_fused=bool(atom_fused), atom_attn_exact=bool(atom_attn_exact))
    if not any(words.values()) and "ditfast_ptx1" not in sys.modules:
        return
    import ditfast_ptx1
    if any(words.values()) and _MODEL is None:
        raise RuntimeError("levers %s refused: bind_model(model) first" % sorted(w for w, on in words.items() if on))
    ditfast_ptx1.apply(_MODEL, **words)


def describe():
    for word in CORE_WORDS:                                     # the exact core word's member census (COUNTS coremember:<word>) read fresh before the counts are copied
        core_member_census(word)
    out = {"cfg": dict(CFG), "counts": json.loads(json.dumps(COUNTS)), "card_rows": row_range_facts(), "gates": {"transition_rows": TRANSITION_GATE_ROWS, "triattn_ceiling_tokens": TRIATTN_CEILING_TOKENS, "triattn_gate_tokens": TRIATTN_GATE_TOKENS}, "sampler": describe_sampler(),
           "keep_pool": describe_keep_pool() if CFG.get("keep_pool") else {"installed": False},
           "templ": _TEMPL.describe(),                    # ptx1_templ: template_dedupe / tmpl_triatt / tmpl_trimul accounts (report.kit_evidence reads them)
           "summary_hostidx": describe_summary_host() if CFG.get("summary_hostidx") else {"installed": False},
           "dit_attn_exact": describe_dit_attn_exact() if CFG.get("dit_attn_exact") else {"installed": False},
           "lazy_init": describe_lazy_init() if CFG.get("lazy_init") else {"installed": False}}
    if "apb_ptx1" in sys.modules:
        out["apb"] = sys.modules["apb_ptx1"].describe()          # the sampler attention levers' account (ditattn / ditattnfp16 / atomattn): engaged, cell, named, calls, census
    out["cores"] = describe_cores()                             # the triangle-attention provider words' account (tricuda, triexact): row, selections per key, census
    if CFG.get("trimul") in TRIMUL_TIER:                        # the TriMul lever's provider binding (report.kit_evidence: `bind=tier:<word>` beside row= / served_by= / selections=)
        out["trimul_" + CFG["trimul"]] = {"bind": "tier:%s" % trimul_tier(), "mode": kit_mode(), "stack": (_TRIMUL_STACK[0] if _TRIMUL_STACK else None)}
    if any(CFG.get(l) for l in ("gblock", "gflash", "xtr", "ttr")):
        try:
            out["pair_fused"] = _PF().describe()          # the core's serve layer: version, carried-cell shas, cell-table sha
        except Exception as e:
            out["pair_fused"] = {"error": repr(e)[:200]}
    if CFG.get("xtr") or CFG.get("ttr"):                             # the transition lever's provider account: the tier word asked, the Selection line served per (lever, family, c, hidden), refusals by kind
        out["transition"] = {"lever": "xtr" if CFG.get("xtr") else "ttr", "word": transition_tier(), "selections": dict(TRANSITION["selections"]), "refusals": dict(TRANSITION["refusals"])}
    if "trunk2_ptx1" in sys.modules:
        out["trunk2"] = sys.modules["trunk2_ptx1"].describe()    # the trunk-side fused-kernel levers' account (pfattn / opm_fused / pwa_fused): engaged, cell, named, calls, census
    out["cores"] = describe_cores()                             # the triangle-attention provider words' account (tricuda, triexact): row, selections per key, census
    if any(CFG.get(l) for l in ("gblock", "gflash", "xtr", "ttr")):
        try:
            out["pair_fused"] = _PF().describe()          # the core's serve layer: version, carried-cell shas, cell-table sha
        except Exception as e:
            out["pair_fused"] = {"error": repr(e)[:200]}
    if "ditfast_ptx1" in sys.modules:
        out["ditfast"] = sys.modules["ditfast_ptx1"].describe()  # the fused-sampler levers' account (cond_dedupe / dit_fused / dit_lowp / atom_fused / atom_attn_exact): state, served, gated, fallback
    return out

