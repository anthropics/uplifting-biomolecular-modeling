"""odde_sampler.py -- OPENDDE_SAMPLER unit: the diffusion-sampler levers of OpenDDE 1.1.1 (DiffusionTransformer 24 blocks x 16 heads x 48, atom
transformers 3+3 blocks x 4 heads x 32 with 32-query x 128-key windows, DiffusionConditioning on a sample-invariant noise level).
The two attention sites are served by the core's pair-bias-attention provider BY WORD (odde_apb_bind.py: opt_core.kernels.apb select() per call
class; no kernel, cell table or row import in the kit); the fused sampler schedules (conditioning de-dup, the fused token / atom stacks) are the
kit-namespaced package third_party/opendde_fpf_ditfast (orchestration; its row kernels are the shared core's, opt_core.kernels.apb.ditfast).
Installed by the served-levers hook through levers/ACCEL/odde_accel_v2.install_dit_attn (the DiT-attention routing site) AFTER the DITFAST hoist
and the ARM arm: the hoist's cached pair biases reach the attention rows through the unchanged AttentionPairBias / hoist slot protocol.

  lever (registry)   switch                         class       site (opendde/model/modules)                       served by
  dit_attn_exact     ODDE_DIT_ATTN=exact            exact       primitives._attention (fp32 SDPA [S,16,N,48])       odde_apb_bind.install_exact: the provider's exact tier (its bit-exact
                     (=ODDE_DIT_ATTN_EXACT=1, legacy alias)                                                          row where vouched on the running stack, else the lever steps aside by name)
  dit_attn_apb       ODDE_DIT_ATTN=fast|big|<row> tolerance   primitives.Attention.forward of the 24 token       odde_apb_bind.install_dit: the tier's fastest measured row per cell
                     (=apb, legacy alias)                       AttentionPairBias modules (transformer.py:1338)    (fp16 / bf16 operand forms are the provider's cell choice)
  atom_attn_apb      ODDE_ATOM_ATTN=fast|big|<row> tolerance  primitives.Attention.forward of the 3+3 atom       odde_apb_bind.install_atom (windowed rows: fpf_atom | dtk_window | sdpa_gather)
                                                                AtomTransformer modules (primitives._local_attention)
  cond_dedupe        ODDE_COND_DEDUPE=1             tolerance   DiffusionConditioning.forward (diffusion.py:1054)   opendde_fpf_ditfast.install_cond_dedupe
  dit_fused          ODDE_DIT_FUSED=1               tolerance   DiffusionTransformer.forward of diffusion_module    opendde_fpf_ditfast.install_dit_fast; requires dit_attn_apb (its attention is odde_apb_bind.dit_packed)
  dit_lowp           ODDE_DIT_LOWP=fp16|bf16        precision   dit_fused's a-path GEMM / attention operand dtype   (read by install_dit_fast; `off` = fp32)
  atom_fused         ODDE_ATOM_FUSED=1              tolerance   AtomTransformer.forward (encoder + decoder)         opendde_fpf_ditfast.install_atom_fast; requires atom_attn_apb (odde_apb_bind.atom_views)
Test hook (never exported by a line): ODDE_SAMPLER_PROBE=1 prints one `[odde_sampler] PROBE ...` census line per call signature of the sampler
sites (shapes, dtypes, strides) on stderr.
Refuse-by-name: every install error raises RuntimeError naming the lever (the hook's ODDE_SERVED_LEVERS_STRICT=1 ends the process); a pinned row the
card / stack cannot run is refused by name; the exact lever whose tier names the stock op on the running card / stack STEPS ASIDE by name instead
(REPORT['aside'], nothing installed); nothing falls back to stock silently.  Counters for the kit's ran-or-refuse census (opendde_opt/ran.py) are
the binder's / packages' own live dicts, published as SECTIONS (odde_accel_v2.STATS carries them).
"""
from __future__ import annotations

import os
import sys

__version__ = "0.2.0"                      # the unit's version word (printed on its marker lines)
HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.join(HERE, "third_party")

REPORT: dict = {"installed": [], "aside": {}, "records": {}, "errors": {}, "probe": False}
SECTIONS: dict = {}                       # section -> live counter dict (odde_accel_v2.STATS.update(SECTIONS); opendde_opt/ran.py reads them)
SWITCHES = ("ODDE_DIT_ATTN", "ODDE_ATOM_ATTN", "ODDE_COND_DEDUPE", "ODDE_DIT_FUSED", "ODDE_DIT_LOWP", "ODDE_ATOM_FUSED",
            "ODDE_DIT_ATTN_EXACT")             # the last one: a legacy spelling of ODDE_DIT_ATTN=exact (read as an alias, exported by no line)
PROBE_SWITCH = "ODDE_SAMPLER_PROBE"


def _log(msg):
    print(f"[odde_sampler] {msg}", file=sys.stderr, flush=True)


def _env(k, default=""):
    return os.environ.get(k, default).strip()


def _flag(k):
    return _env(k, "0") not in ("", "0", "off")


def requested() -> dict:
    """The levers the environment asks for: {lever: switch value}.  ODDE_DIT_ATTN: exact -> dit_attn_exact; fast | big | <row> -> dit_attn_apb."""
    import odde_apb_bind as B
    r = {}
    wd = B.word("dit")
    if wd is not None:
        r["dit_attn_exact" if B.KA.split_word(wd)[0] in ("exact", "faithful") else "dit_attn_apb"] = wd
    wa = B.word("atom")
    if wa is not None:
        r["atom_attn_apb"] = wa
    if _flag("ODDE_COND_DEDUPE"):
        r["cond_dedupe"] = "1"
    if _flag("ODDE_DIT_FUSED"):
        r["dit_fused"] = "1"
        if _env("ODDE_DIT_LOWP", "off") != "off":
            r["dit_lowp"] = _env("ODDE_DIT_LOWP")
    elif _env("ODDE_DIT_LOWP", "off") not in ("", "off"):
        r["dit_lowp"] = _env("ODDE_DIT_LOWP")
    if _flag("ODDE_ATOM_FUSED"):
        r["atom_fused"] = "1"
    return r


def _third_party_on_path():
    if TP not in sys.path:
        sys.path.insert(0, TP)


def install(model) -> dict:
    """Install every requested lever on `model` (the runner's OpenDDE module). Returns REPORT; raises RuntimeError('<lever>: <reason>') by name."""
    req = requested()
    probe = _flag(PROBE_SWITCH)
    if not req and not probe:
        return REPORT
    _third_party_on_path()
    # ---- refusals decided from the words alone (composition contracts)
    if "dit_lowp" in req and "dit_fused" not in req:
        raise RuntimeError("dit_lowp: requires lever dit_fused (ODDE_DIT_FUSED=1) -- the precision word of the fused token stack")
    if "dit_fused" in req and "dit_attn_apb" not in req:
        raise RuntimeError(f"dit_fused: requires lever dit_attn_apb (ODDE_DIT_ATTN=fast | big | <row>; got {_env('ODDE_DIT_ATTN')!r}) -- the fused token stack's attention is the provider-served site")
    if "atom_fused" in req and "atom_attn_apb" not in req:
        raise RuntimeError(f"atom_fused: requires lever atom_attn_apb (ODDE_ATOM_ATTN=fast | big | <row>; got {_env('ODDE_ATOM_ATTN')!r})")
    import odde_apb_bind as B
    SECTIONS["apb_launch"] = B.LAUNCHES                                        # kernel launches per site (n_apb, n_atom): opendde_opt/ran.py's counters
    SECTIONS["apb_state_dit"] = B.COUNTS["dit"]; SECTIONS["apb_state_atom"] = B.COUNTS["atom"]; SECTIONS["dit_attn_exact"] = B.COUNTS["exact"]
    # ---- exact tier first: the statement primitives._attention through the provider's exact tier (or the lever steps aside by name)
    if "dit_attn_exact" in req:
        marker = B.install_exact()
        ex = B.COUNTS["exact"]
        REPORT["records"]["dit_attn_exact"] = {"marker": marker, "selection": ex.get("selection"), "aside": ex.get("aside"), "word": ex.get("word")}
        if ex.get("aside"):                                                    # the provider's exact tier names the stock op here (or the probe refused): the lever STEPS ASIDE BY NAME
            REPORT["aside"]["dit_attn_exact"] = ex["aside"]; _log(marker)      # (nothing installed, the stock statement serves, the run complete: opendde_opt/stack.py reads REPORT["aside"])
        else:
            REPORT["installed"].append("dit_attn_exact"); _log(marker)
    # ---- the pair-bias attention sites (instance-level Attention.forward replacements served by the provider's row per call class)
    if "dit_attn_apb" in req:
        st = B.install_dit(model)
        REPORT["records"]["dit_attn_apb"] = {"modules": st.get("installed_on"), "word": st.get("word"), "row": st.get("row"), "selection": st.get("selection"), "version": B.__version__}
        REPORT["installed"].append("dit_attn_apb")
        _log(f"DITATTN_APB:on(word={st.get('word')} row={st.get('row')} modules={st.get('installed_on')} provider: {st.get('selection')}; binding {B.NAME} {B.__version__})")
    if "atom_attn_apb" in req:
        st = B.install_atom(model)
        REPORT["records"]["atom_attn_apb"] = {"modules": st.get("installed_on"), "word": st.get("word"), "row": st.get("row"), "selection": st.get("selection"), "version": B.__version__}
        REPORT["installed"].append("atom_attn_apb")
        _log(f"ATOMATTN_APB:on(word={st.get('word')} row={st.get('row')} modules={st.get('installed_on')} provider: {st.get('selection')}; binding {B.NAME} {B.__version__})")
    # ---- the fused sampler schedules
    if "cond_dedupe" in req or "dit_fused" in req or "atom_fused" in req:
        import opendde_fpf_ditfast as D
        try:
            if "cond_dedupe" in req:
                D.install_cond_dedupe(model)
                SECTIONS["cond_dedupe"] = D.cond_dedupe.COUNTS
                REPORT["records"]["cond_dedupe"] = {"guard": "stride0"}; REPORT["installed"].append("cond_dedupe")
            if "dit_fused" in req:
                rep = D.install_dit_fast(model)
                SECTIONS["dit_fused"] = D.dit_fast.COUNTS
                REPORT["records"]["dit_fused"] = {k: rep.get(k) for k in ("blocks", "act", "lowp", "attention", "bias_slots", "strip", "packed_param_MB")}
                REPORT["installed"].append("dit_fused")
                if "dit_lowp" in req:
                    REPORT["records"]["dit_lowp"] = {"word": rep.get("lowp")}; REPORT["installed"].append("dit_lowp")
            if "atom_fused" in req:
                rep = D.install_atom_fast(model)
                SECTIONS["atom_fused"] = D.atom_fast.COUNTS
                REPORT["records"]["atom_fused"] = {k: rep.get(k) for k in ("stacks", "act", "attention", "cond_slots")}
                REPORT["installed"].append("atom_fused")
        except D.LeverRefused as e:
            raise RuntimeError(str(e)) from e
    if probe:
        _install_probe(model)
    import atexit
    atexit.register(_exit_census)
    return REPORT


def _exit_census():
    """One line at exit: the levers installed and the CUDA caching allocator's retry / release counters of this process (a sampler schedule that
    fragments the pool shows here as alloc retries / segment frees before the confidence head's large score tensors)."""
    try:
        import torch
        st = torch.cuda.memory_stats() if torch.cuda.is_available() and torch.cuda.is_initialized() else {}
        alloc = {k: st.get(k) for k in ("num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free", "num_sync_all_streams")}
        alloc["reserved_peak_gib"] = round(st.get("reserved_bytes.all.peak", 0) / 2 ** 30, 2); alloc["allocated_peak_gib"] = round(st.get("allocated_bytes.all.peak", 0) / 2 ** 30, 2)
    except Exception as e:  # noqa: BLE001
        alloc = {"error": repr(e)[:120]}
    _log(f"EXIT installed={','.join(REPORT['installed']) or '-'} allocator={alloc}")


# ------------------------------------------------------------------------------------------------------------------ test hook: shape census
_PROBE_SEEN: dict = {}


def _sig(*tensors, **named):
    import torch
    parts = []
    for k, t in named.items():
        if torch.is_tensor(t):
            parts.append(f"{k}={tuple(t.shape)}:{str(t.dtype).replace('torch.', '')}:strides{tuple(t.stride())}")
        elif t is not None:
            parts.append(f"{k}={t!r}")
    return " ".join(parts)


def _probe_wrap(site, fn, argnames):
    def w(*a, **k):
        named = dict(zip(argnames, a)); named.update({kk: vv for kk, vv in k.items() if kk in argnames or kk in ("n_queries", "n_keys", "enable_efficient_fusion", "inplace_safe")})
        s = _sig(**named)
        key = (site, s)
        if key not in _PROBE_SEEN:
            _PROBE_SEEN[key] = 0
            _log(f"PROBE site={site} {s}")
        _PROBE_SEEN[key] += 1
        return fn(*a, **k)
    w.__wrapped__ = fn
    return w


def _install_probe(model):
    """Outermost wrappers on the sampler sites (after the levers): one census line per distinct call signature."""
    import opendde.model.modules.primitives as PR
    dm = model.diffusion_module
    for i, blk in enumerate(dm.diffusion_transformer.blocks):
        if i in (0, 23):
            att = blk.attention_pair_bias.attention
            att.forward = _probe_wrap(f"tok{i}.attention.forward", att.forward, ("q_x", "kv_x", "attn_bias", "trunked_attn_bias", "n_queries", "n_keys"))
    for tag, part in (("enc", dm.atom_attention_encoder), ("dec", dm.atom_attention_decoder)):
        at = part.atom_transformer
        at.forward = _probe_wrap(f"{tag}.atom_transformer.forward", at.forward, ("q", "c", "p"))
        blk = at.diffusion_transformer.blocks[0]
        att = blk.attention_pair_bias.attention
        att.forward = _probe_wrap(f"{tag}0.attention.forward", att.forward, ("q_x", "kv_x", "attn_bias", "trunked_attn_bias", "n_queries", "n_keys"))
    dc = dm.diffusion_conditioning
    dc.forward = _probe_wrap("diffusion_conditioning.forward", dc.forward, ("t_hat_noise_level", "relp_feature", "s_inputs", "s_trunk", "z_trunk", "pair_z"))
    dt = dm.diffusion_transformer
    dt.forward = _probe_wrap("diffusion_transformer.forward", dt.forward, ("a", "s", "z", "n_queries", "n_keys", "inplace_safe", "chunk_size", "enable_efficient_fusion", "extra_attn_bias"))
    orig_att = PR._attention
    PR._attention = _probe_wrap("primitives._attention", orig_att, ("q", "k", "v", "attn_bias", "use_efficient_implementation"))
    import atexit
    atexit.register(lambda: _log("PROBE census " + "; ".join(f"{site} x{n}" for (site, _s), n in sorted(_PROBE_SEEN.items()))))
    REPORT["probe"] = True
    _log("PROBE installed (test hook ODDE_SAMPLER_PROBE): token blocks 0/23, atom enc/dec block 0, conditioning, token stack, primitives._attention")
