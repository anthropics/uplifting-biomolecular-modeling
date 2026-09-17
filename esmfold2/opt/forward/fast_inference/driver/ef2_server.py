#!/usr/bin/env python3
"""The ESMFold2 kit's lever switchboard and output writer (the package esmfold2_opt drives it).

``MODES``: the two lever sets the package resolves by name (esmfold2_opt.modes.KIT_MODES: `exact` -> ``opt7x``, `fast` / `big` -> ``opt14_msa``),
each a tuple ``(base, ef2_opt flags, ef2_w4 flags, "<group>:<flags>;...")``. ``configure(model, mode, builder, off=)`` installs a set on a loaded
model in the one supported order (below) and returns the description string the package records; ``write_outputs`` writes one prediction's files
(mmCIF, PAE / pLDDT npz, the score row) — upstream's own outputs, nothing derived from them. ``EF2_MK=0/1`` and ``EF2_MSA=<flags>`` override the
``mk`` / ``msa:`` groups of the fourth field (the package exports them only for a mode that needs them and drops any caller-set value); every
other group is subtracted only through ``off`` (the package's line drops and its declared ablation variable, decided before configure runs).
"""
import os, sys, datetime
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_ef2_om as KIT

MODES = {'opt7x': ('fused', 'tg+sg+eg+ec+fc+pb+ls+rg', 't6,t3,t5', 'atom:ax;fz;msa2:mh;pair:xtr,xte;hoist:trimul,glue,disto;ln:xln;dit:ro'),
         'opt14_msa': ('fused', 'tg+sg+eg+ec+fc+pb+msa+ls+rg', 't3,t5,t10', 'trimul:tx;atom:ax,af;fz;msa2:m15,m16,m17,mh;pair:t15,t15msa,t16;hoist:disto;ln:xln;dit:ro,kd,dit')}
TRIMUL_TIER = {'opt7x': 'exact', 'opt14_msa': 'fast'}   # the shared core's TriMul tier word a set binds when it carries lever tx; the memory mode runs opt14_msa under the word `big` (ESMFOLD2_OPT=big).
                                                          # opt7x does not carry tx: its pair TriMul IS the upstream fused statement (the `exact` word names no byte-vouched exact-class row on the
                                                          # kit's stacks, so a binding would serve the same statement by name); `trimul:tx` on opt7x binds `exact` the day the shared core vouches one
# base 'fused' = set_kernel_backend('fused') + set_chunk_size(None); field 2 = ef2_opt.install flags (tg trunk graphs, sg sampler graphs, eg encoder
# graphs, ec ESMC cache, fc feature cache, pb pair-bias cache, msa = fused TriMul in the MSA encoder, ls = static loop I/O of the trunk recycle,
# rg = one graph per trunk recycle up to ef2_opt's internal token ceiling); field 3 = ef2_w4 flags (t3 tile table, t5 bf16 pre-cast, t6 / t10 fused
# transition); field 4 = ';'-separated groups, each `<module>:<its lever flags>` or a bare flag:
#   trimul:tx      ef2_w4        the pair TriMul (both directions of every PairUpdateBlock, and the M1 MSA path's fused TriMul) through the shared core's
#                                TriMul provider (opt_core.kernels.trimul) bound by the set's TIER WORD (TRIMUL_TIER: opt14_msa -> `fast`; the memory mode ->
#                                `big`, read off ESMFOLD2_OPT; an opt7x carrying it -> `exact`; EF2_TRIMUL_TIER=<word|row> overrides for an A/B): the
#                                provider's measured cell table decides the row per capability / stack / size class / direction on every card; a class whose
#                                cell names the stock op (the `exact` word today: no exact-class row is vouched byte-identical on this stack) and a call the
#                                provider refuses are served by the upstream fused TriMul BY NAME, printed once and counted
#   atom:<flags>   ef2_atom      ax = the atom path's exact hoists (adaLN factors, prefix var-len attention, RoPE tables, alias-not-copy),
#                                af = the fused atom-transformer block + segmented token mean (tensor-core GEMMs bf16-in / fp32-accumulate)
#   fz             ef2_feats     vectorised MSA featurisation (host; identical arrays)
#   msa2:<flags>   ef2_msa_v2    m15 / m16 / m17 = fused msa_transition / pair-weighted averaging / outer-product mean (Full model), mh = the
#                                loop-invariant MSA-encoder hoist (exact; engages per fold when no row subsample is drawn)
#   pair:<flags>   ef2_pair_v2   t15 = one-kernel pair Transition (+ t15msa: the MSA blocks' PairTransition through it) by the shared core's
#                                transition tier word; xtr = the MSA
#                                module's PairTransition (pair_transition c=256, msa_transition c=128) through the shared core's transition provider
#                                by word with the module's LayerNorm output given (bitwise; Full model; refused by name beside t16 / t15msa);
#                  ef2_xte       xte = the trunk's fused inference Transition (every d=256 C.Transition) through the shared core's transition provider
#                                row esm_fused_exact by word (bitwise: the statement in one kernel, self-checked at install; both models; installed
#                                after ef2_pair_v2; steps aside by name on a class / core without the row);
#                  ef2_transition_cute  t16 = every d=256 / h=1024 Transition AND the MSA blocks' PairTransition through the shared core's transition
#                                provider by the set's tier word: the 9.0 class's transition lever (the provider's sm_90a row) — it supersedes t15 / t15msa there, which
#                                serve the other classes (the package never hands configure() both: t15 with t16 is refused by name below)
#   hoist:<flags>  ef2_hoist     trimul / glue = the MSA module's reference TriMul contraction re-plumbed (exact; Full model, when ef2_opt's
#                                `msa` does not own those blocks), disto = the distogram logits moved to the host asynchronously (exact)
#   dit:<flags>    ef2_dit       ro = the device-resident sampler roll-out (graph per step boundary, constants staged once, fp32 bias twin, cusolver
#                                Kabsch: exact), kd = the sync-free device Kabsch, dit = the fused diffusion-transformer step (bf16 GEMMs, fused
#                                pair-bias attention): tolerance class
#   ln:<flags>     ef2_xln       xln = the release tree's exactln LayerNorm row (opt_core.kernels.ln) at the pair-sized nn.LayerNorm sites (exact; both models;
#                                steps aside by name on a stack the row does not serve)
#   mk             ef2_mk_sampler's step-invariant hoist and msa:<flags> = ef2_msa's t11-t14 remain known groups no set of this table selects.
# opt7x = the bitwise set; opt14_msa = the tolerance-class set. Install order (the one supported order): base -> ef2_mk_sampler -> ef2_msa -> ef2_w4 ->
# ef2_w4.enable_tx (the trimul group) -> ef2_atom -> ef2_feats -> ef2_opt.install (graph wrappers) -> ef2_msa_v2 -> ef2_pair_v2 | ef2_transition_cute -> ef2_xte -> ef2_hoist -> ef2_xln ->
# ef2_dit (wraps the sampler chain: last).
GROUPS = ("trimul", "atom", "fz", "msa2", "pair", "hoist", "ln", "dit", "mk", "msa")   # the fourth field's known groups (an unknown group is a table error, raised by name)
TX_WORDS = ("tx",)                                                        # the trimul group's one lever (ef2_w4.enable_tx): the pair TriMul by the shared core's provider under the set's tier word, every class
TRIMUL_LEVERS = TX_WORDS                                                  # the trimul group's known lever names
PACKAGE_MODE_ENV = "ESMFOLD2_OPT"                                         # the package's mode variable (esmfold2_opt exports it before configure): `big` binds the TriMul word `big` on opt14_msa
TRIMUL_TIER_ENV = "EF2_TRIMUL_TIER"                                       # engineering override of the TriMul word (a tier word or a provider row name, for an A/B); unset = the set's word
PAIR_V2_LEVERS = ("t15", "t15msa", "xtr")                       # the pair group's ef2_pair_v2 levers; t16 is ef2_transition_cute's
XTE_LEVERS = ("xte",)                                                     # the pair group's ef2_xte lever: the trunk Transition by the shared core's transition row word (the exact line's)
KNOBS = {"af.gemm": ("ef2_atom", "gemm"), "dit.gemm": ("ef2_dit", "gemm"), "dit.cond": ("ef2_dit", "cond"), "dit.attn": ("ef2_dit", "attn"),
         "dit.attn_precision": ("ef2_dit", "attn_precision")}               # <lever>.<knob> spellings configure(knobs=) accepts (the package's ablation variable is their only writer)
KNOB_DEFAULTS = {"af.gemm": "bf16", "dit.gemm": "bf16", "dit.cond": "bf16", "dit.attn": "flash", "dit.attn_precision": "bf16"}   # the compositions' own values (one precision vocabulary: bf16 operands, fp32 accumulate)


def utc():
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def parse_extra(extra):
    """The fourth field -> {group: [flags]} in table order ('fz' / 'mk' -> []). Raises ValueError naming an unknown group."""
    out = {}
    for tok in [t.strip() for t in str(extra or "").split(";") if t.strip()]:
        name, _, flags = tok.partition(":")
        if name not in GROUPS:
            raise ValueError(f"ef2_server.MODES: unknown group {name!r} in {extra!r} (known: {', '.join(GROUPS)})")
        out[name] = [f for f in flags.replace("+", ",").split(",") if f]
    return out


def _keep(flags, off):
    return [f for f in flags if f not in off]


def check_composition(mode, off=()):
    """The set MODES[mode] minus ``off`` must be self-consistent — raised BY NAME here, before configure() touches the model: unknown lever names in
    the trimul / pair groups; t16 beside t15 / t15msa (one class patch of C.Transition per process: the package keeps t16 on 9.0 and t15 / t15msa
    elsewhere, by class)."""
    if mode not in MODES:
        raise ValueError(f"ef2_server: unknown mode {mode!r} (known: {sorted(MODES)})")
    w4 = MODES[mode][2]
    groups = parse_extra(MODES[mode][3] if len(MODES[mode]) > 3 else "")
    fl4 = set(_keep([f for f in (w4 or "").replace("+", ",").split(",") if f], off))
    trimul_flags = _keep(groups.get("trimul", []), off)
    if trimul_flags:
        unknown = [f for f in trimul_flags if f not in TX_WORDS]
        if unknown:
            raise ValueError(f"ef2_server.MODES[{mode!r}]: trimul:{','.join(trimul_flags)} names unknown levers {unknown} (known: {', '.join(TX_WORDS)})")
    pair_flags = _keep(groups.get("pair", []), off)
    unknown = [f for f in pair_flags if f not in PAIR_V2_LEVERS + ("t16",) + XTE_LEVERS]
    if unknown:
        raise ValueError(f"ef2_server.MODES[{mode!r}]: pair:{','.join(pair_flags)} names unknown levers {unknown} (known: {', '.join(PAIR_V2_LEVERS + ('t16',) + XTE_LEVERS)})")
    if "xte" in pair_flags and ("t16" in pair_flags or "t15" in pair_flags):
        raise ValueError(f"ef2_server.MODES[{mode!r}]: pair:{','.join(pair_flags)} carries xte together with t16 / t15 — one owner of the trunk's C.Transition per process: "
                         "xte is the exact line's (the core transition row esm_fused_exact by word: the fused statement, bitwise), t16 / t15 the fast line's")
    if "xtr" in pair_flags and ("t16" in pair_flags or "t15msa" in pair_flags):
        raise ValueError(f"ef2_server.MODES[{mode!r}]: pair:{','.join(pair_flags)} carries xtr together with t16 / t15msa — one owner of the MSA blocks' PairTransition per process: "
                         "xtr is the exact line's (the core transition provider by word, LayerNorm given), t16 / t15msa the fast line's")
    if "t16" in pair_flags and ("t15" in pair_flags or "t15msa" in pair_flags):
        raise ValueError(f"ef2_server.MODES[{mode!r}]: pair:{','.join(pair_flags)} carries t16 together with t15 / t15msa — one class patch of C.Transition per process: "
                         "the 9.0 class runs t16, the other classes t15 / t15msa (esmfold2_opt subtracts by class before configure)")
    return dict(w4=sorted(fl4), trimul=trimul_flags, pair=pair_flags)


def trimul_tier(mode):
    """The TriMul provider word the set binds in this process: EF2_TRIMUL_TIER when set (an A/B), else `big` under the package's memory mode
    (ESMFOLD2_OPT=big runs opt14_msa), else the set's own word (TRIMUL_TIER; a table key without one binds `fast`)."""
    ov = (os.environ.get(TRIMUL_TIER_ENV) or "").strip()
    if ov:
        return ov
    if (os.environ.get(PACKAGE_MODE_ENV) or "").strip().lower() == "big":
        return "big"
    return TRIMUL_TIER.get(mode, "fast")


def configure(model, mode, builder=None, off=(), knobs=None):
    """Install the set ``MODES[mode]`` on ``model`` minus the lever names in ``off`` (registry flag names: the package's line drops and ablations),
    with ``knobs`` = {'<lever>.<knob>': value} sub-choices (KNOBS; absent = KNOB_DEFAULTS). Every lever engages or its module raises by name."""
    check_composition(mode, off)                                        # the set's self-consistency, by name, before the model is touched
    base, opt, w4 = MODES[mode][0], MODES[mode][1], MODES[mode][2]
    groups = parse_extra(MODES[mode][3] if len(MODES[mode]) > 3 else "")
    off = set(off or ())
    kn = dict(KNOB_DEFAULTS); kn.update(knobs or {})
    unknown = sorted(set(kn) - set(KNOBS))
    if unknown:
        raise ValueError(f"ef2_server.configure: unknown knob(s) {unknown} (known: {sorted(KNOBS)})")
    desc = []
    if base == "fused":
        model.set_kernel_backend("fused"); model.set_chunk_size(None); desc.append("set_kernel_backend('fused'); set_chunk_size(None)")
    else:
        raise ValueError(f"configure({mode}): base {base!r}; every mode of the table runs on the fused base")
    # ORDER CONTRACT: base -> ef2_mk_sampler -> ef2_msa -> ef2_w4 (+ enable_tx) -> ef2_atom -> ef2_feats -> ef2_opt.install (graph wrappers) -> ef2_msa_v2 -> ef2_pair_v2 | ef2_transition_cute -> ef2_xte -> ef2_hoist -> ef2_xln -> ef2_dit.
    # configure() is the only place levers are installed; assert nothing has captured graphs for this model yet so no lever can be silently absent from a replayed graph.
    _captured = [n for n, m in model.named_modules() if getattr(m, "_ef2opt_graphs", None)]
    assert not _captured, f"configure({mode}): ef2_opt already holds captured graphs on {len(_captured)} module(s) ({_captured[:3]}); call ef2_opt.clear_graphs(model) before re-configuring"
    env_mk = os.environ.get("EF2_MK")             # None -> follow MODES; "0"/"1" -> force
    env_msa = os.environ.get("EF2_MSA")           # None -> follow MODES; "" or "off" -> none; else flags for ef2_msa.enable
    want_mk = (env_mk == "1") if env_mk is not None else ("mk" in groups)
    want_mk = want_mk and "mk" not in off
    msa_flags = env_msa if env_msa is not None else ",".join(groups.get("msa", []))
    msa_flags = ",".join(_keep([f for f in msa_flags.split(",") if f], off))
    if want_mk:                            # the mk sampler hoist (vendored driver/ef2_mk_sampler.py): instance patch of structure_head.diffusion_module; EXACT; before ef2_opt.install
        import ef2_mk_sampler
        desc.append(f"ef2_mk_sampler.enable(model) -> {ef2_mk_sampler.enable(model)}")
    if msa_flags and msa_flags.lower() not in ("0", "off", "none"):
        import ef2_msa                    # MSA-module levers (vendored driver/ef2_msa.py): per-instance patches inside msa_encoder; before ef2_w4.enable and ef2_opt.install
        if getattr(model, "msa_encoder", None) is not None:
            desc.append(f"ef2_msa.enable(model, {msa_flags}) -> {ef2_msa.enable(model, **ef2_msa.parse_flags(msa_flags))}")
        else:
            desc.append("ef2_msa: model has no msa_encoder (Fast) -> skipped")
    fl4 = set(_keep([f for f in (w4 or "").replace("+", ",").split(",") if f], off))
    if fl4:
        import ef2_w4                     # W4 levers patch module-level kernels (process-wide); must precede ef2_opt graph capture
        d4 = ef2_w4.enable(model, transition_nolin=("t1" in fl4), tiles=(True if "t3" in fl4 else None), weight_cache=("t5" in fl4),
                           transition_rowblock=("t6" in fl4), transition_fused=("t10" in fl4))
        desc.append(f"ef2_w4.enable({sorted(fl4)}) -> {d4}")
    trimul_flags = _keep(groups.get("trimul", []), off)
    if "tx" in trimul_flags:               # lever tx (ef2_w4): the pair TriMul through the shared core's provider under the set's tier word; the provider is loaded and a canary served
        import ef2_w4                     # NOW (right after ef2_w4.enable, before any graph capture); a class / call the word does not serve takes the upstream fused TriMul by name
        word = trimul_tier(mode)
        desc.append(f"ef2_w4.enable_tx({word!r}) -> {ef2_w4.enable_tx(word)}")
    atom_flags = _keep(groups.get("atom", []), off)
    if atom_flags:                         # the atom path (vendored driver/ef2_atom.py): instance patches of the diffusion module's atom encoder / decoder; BEFORE ef2_opt.install
        import ef2_atom                   # (its static buffers are cleared together with ef2_opt's graphs: the module chains clear_static to ef2_opt.clear_graphs)
        kw = ef2_atom.parse_flags(",".join(atom_flags))
        if kw.get("fused_block"):
            kw["gemm"] = kn["af.gemm"]
        desc.append(f"ef2_atom.install(model, {atom_flags}{', gemm=' + kn['af.gemm'] if kw.get('fused_block') else ''}) -> {ef2_atom.install(model, **kw)}")
    if "fz" in groups and "fz" not in off:      # host-side MSA featurisation (vendored driver/ef2_feats.py): a module-attribute patch of upstream's paired_msa; identical arrays
        import ef2_feats
        desc.append(f"ef2_feats.enable() -> {ef2_feats.enable()}")
    if opt:
        import ef2_opt
        fl = set(_keep(opt.split("+"), off))
        info = ef2_opt.install(model, trunk_graphs=("tg" in fl), sampler_graphs=("sg" in fl), fuse_msa_trimul=("msa" in fl), encoder_graphs=("eg" in fl), pair_bias_cache=("pb" in fl),
                               esmc_cache=("ec" in fl), builder=(builder if "fc" in fl else None), loop_static=("ls" in fl or "rg" in fl), recycle_graph=("rg" in fl))
        desc.append(f"ef2_opt.install({sorted(fl)}) -> {info}")
    msa2_flags = _keep(groups.get("msa2", []), off)
    if msa2_flags:                         # MSA-module levers (vendored driver/ef2_msa_v2.py): per-instance block forwards on top of ef2_opt's M1 forward and encoder graph wrapper;
        import ef2_msa_v2                 # AFTER ef2_opt.install, before the first fold; the Fast model has no msa_encoder: the module raises by name (the package keeps these off its set there)
        desc.append(f"ef2_msa_v2.install(model, {','.join(msa2_flags)}) -> {ef2_msa_v2.install(model, ','.join(msa2_flags))}")
    pair_flags = _keep(groups.get("pair", []), off)   # (check_composition above refused unknown names and t16 beside t15 / t15msa)
    v2_flags = [f for f in pair_flags if f in PAIR_V2_LEVERS]
    if v2_flags:                           # pair Transition levers (vendored driver/ef2_pair_v2.py): C.Transition class patch + MSA-block instance wrappers; AFTER ef2_opt.install and ef2_msa_v2
        import ef2_pair_v2
        desc.append(f"ef2_pair_v2.install(model, {','.join(v2_flags)}) -> "
                    f"{ef2_pair_v2.install(model, trunk='t15' in v2_flags, msa='t15msa' in v2_flags, xtr='xtr' in v2_flags)}")
    if "t16" in pair_flags:                # the 9.0 class's pair Transition lever t16 (vendored driver/ef2_transition_cute.py): C.Transition + MOD.PairTransition class patches, word resolved and weights pre-packed now
        import ef2_transition_cute        # (nothing packed inside a graph capture); a failure raises by name, the package refuses the mode
        desc.append(f"ef2_transition_cute.install(model, t16) -> {ef2_transition_cute.install(model, trunk=True, msa=True)}")
    if "xte" in pair_flags:                # the trunk Transition by the shared core's transition row word (vendored driver/ef2_xte.py): instance bindings over ef2_pair_v2's class
        import ef2_xte                     # forward, resolved + self-checked + packed now (before any capture); a class / core / stack the row does not serve -> steps aside BY NAME
        desc.append(f"ef2_xte.install(model) -> {ef2_xte.install(model)}")
    hoist_flags = _keep(groups.get("hoist", []), off)
    if hoist_flags:                        # exact-class data-movement hoists (vendored driver/ef2_hoist.py): instance patches; strict: a flag that cannot engage raises by name
        import ef2_hoist
        desc.append(f"ef2_hoist.install(model, {','.join(hoist_flags)}) -> {ef2_hoist.install(model, strict=True, **{f: True for f in hoist_flags})}")
    ln_flags = _keep(groups.get("ln", []), off)
    if ln_flags:                           # exact-class: the release tree's exactln row at the pair-sized nn.LayerNorm sites (vendored driver/ef2_xln.py): instance patches, kernels
        import ef2_xln                     # compiled now (before any capture); a stack the row cannot serve makes the lever step aside BY NAME (ef2_xln.refusal()), the mode runs
        desc.append(f"ef2_xln.install(model, {','.join(ln_flags)}) -> {ef2_xln.install(model, **{f: True for f in ln_flags})}")
    if os.environ.get("EF2_CONF_PER_SAMPLE", "0") == "1":   # the memory mode's word (esmfold2_opt.modes.BIG_FAST.conf_per_sample): the confidence head once per
        import ef2_conf                                   # structure sample when num_diffusion_samples > 1 (driver/ef2_conf.py; S == 1 untouched)
        desc.append(f"ef2_conf.install(model) -> {ef2_conf.install(model)}")
    dit_flags = _keep(groups.get("dit", []), off)
    if dit_flags:                          # the sampler's token path (vendored driver/ef2_dit.py): wraps structure_head.sample and the diffusion module's forward; LAST (it wraps the
        import ef2_dit                    # chain the other levers installed); graph_budget_tokens=None: the roll-out's capture follows ef2_opt's sampler-site budget
        if "ro" not in dit_flags:
            raise ValueError(f"ef2_server.MODES[{mode!r}]: dit:{','.join(dit_flags)} without ro (kd and dit ride on the roll-out)")
        fast = "dit" in dit_flags
        kw = dict(rollout=True, graph=True, static_once=True, bias_f32=True, kabsch=("device" if "kd" in dit_flags else "torch"), graph_budget_tokens=None,
                  dit=fast, gemm=(kn["dit.gemm"] if fast else "fp32"), cond=(kn["dit.cond"] if fast else "fp32"), attn=(kn["dit.attn"] if fast else "sdpa"),
                  attn_precision=(kn["dit.attn_precision"] if fast else "ieee"), fused_ew=fast)
        desc.append(f"ef2_dit.install(model, {','.join(dit_flags)}: {kw}) -> {ef2_dit.install(model, **kw)}")
    if off:
        desc.append(f"off={','.join(sorted(off))}")
    return "; ".join(desc)


NPZ_SUFFIX = "_pae.npz"                 # cif_all/<stem>_pae.npz next to cif_all/<stem>.cif
ROW_RESULT_FIELDS = ("iptm", "ptm")     # upstream MolecularComplexResult scalars recorded in every row (KIT.fget)
NPZ_RESULT_FIELDS = ("pair_chains_iptm",)   # upstream MolecularComplexResult arrays written into the npz (float32) when the result carries them


def result_array(r, name, dtype):
    """An upstream result field as a numpy array of ``dtype`` — a torch tensor, a numpy array or a staged (JSON) list alike; None when the
    result does not carry it."""
    v = getattr(r, name, None)
    if v is None:
        return None
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    elif hasattr(v, "numpy"):
        v = v.numpy()
    return np.asarray(v, dtype=dtype)


def write_outputs(res, item, variant, seed, meta, cif_dir, save_struct, gpu_name, wall):
    """The kit's one writer, per sample k of one fold() result (a result or a list of results): cif_all/<id>__<variant>__s<seed>_x<k>.cif =
    upstream's ``result.complex.to_mmcif()`` text; cif_all/<stem>_pae.npz = upstream's ``pae`` / ``plddt`` (float16), the token features
    ``asym_id`` / ``mol_type`` from upstream's ``prepare_input`` (``meta``), and upstream's ``pair_chains_iptm`` (float32) when present; one row
    per sample: upstream's ``iptm`` / ``ptm``, the token count, the MSA depths and the bookkeeping (pred_file, gpu, wall_s, utc). Nothing is
    derived from the confidences. ``meta`` = (msa depths per chain, asym_id, mol_type, n_tokens). Returns the rows."""
    depths, asym, mol, ntok = meta
    rows = []
    for si, r in enumerate(res if isinstance(res, list) else [res]):
        pae = r.pae.numpy().astype(np.float32); plddt = r.plddt.numpy().astype(np.float32)
        stem = f"{item['complex_id']}__{variant}__s{'none' if seed is None else seed}_x{si}"
        if save_struct:
            with open(os.path.join(cif_dir, stem + ".cif"), "w") as f: f.write(r.complex.to_mmcif())
            arrays = dict(pae=pae.astype(np.float16), plddt=plddt.astype(np.float16), asym_id=asym, mol_type=mol)
            for name in NPZ_RESULT_FIELDS:
                a = result_array(r, name, np.float32)
                if a is not None:
                    arrays[name] = a
            np.savez_compressed(os.path.join(cif_dir, stem + NPZ_SUFFIX), **arrays)
        rows.append(dict(complex_id=item["complex_id"], variant=variant, seed=seed, sample_index=si, status="ok",
                         **{name: KIT.fget(r, name) for name in ROW_RESULT_FIELDS}, n_tokens=int(pae.shape[0]), msa_depths=depths,
                         pred_file=(f"{stem}.cif" if save_struct else None), gpu=gpu_name, wall_s=round(wall, 3), utc=utc()))
    return rows
