"""The tree's launcher for mode ``fast``: runs the FlashPairformer add-on's own launcher (argv[1]) unchanged and, at exit, prints the add-on's
served report as one kit line. The add-on serves only square bf16 pair activations with N % 128 == 0 and falls back to the stock classes
call by call outside that — silently, counted only in ``served_report()``; modes.lever_evidence reads it, so a kernel installed but not
served on a given input is a named event (``levers_short``), never a silent fallback. Nothing else changes: the add-on's launcher runs as
``__main__`` with the argv it expects.
"""
import atexit
import importlib.util
import os
import runpy
import sys

PREFIX = "[af3-jax-opt]"
HERE = os.path.dirname(os.path.abspath(__file__))
_HOIST = {"installed": False, "counted": False, "calls": 0, "probe_error": None}
_TREE = {"cond_share": None, "atom_cond_hoist": None, "dattn": None, "ttr": None, "triatt_xla": None, "sampler_bf16": None, "atom_attn": None, "trimul_cd": None, "lnp": None, "hoist_logits": None}                                                    # the tree's in-process lever modules loaded into this process (inprocess/)
# INSTALL ORDER — the one list. The tree's in-process levers rebind names inside `alphafold3.model` (each module's REBINDS says which);
# the order below is the install order and it is load-bearing in exactly two places, both relative to the FlashPairformer add-on's import:
#   pass 1 (before `import af3_flashpairformer`): cond_share, atom_cond_hoist, dattn, ttr, triatt_xla, sampler_bf16, atom_attn —
#          * cond_share FIRST: it REPLACES diffusion_head.sample (its REPLACES tuple) — every later lever that touches `sample` WRAPS whatever is bound
#            when it installs (atom_cond_hoist's precompute wrapper, then sampler_bf16's scope wrapper), so the chain is scope(precompute(shared body));
#          * atom_cond_hoist BEFORE the add-on: it rebinds diffusion_head.DiffusionHead to a subclass and the add-on derives HoistDiffusionHead from
#            whatever that name is when IT is imported (the TRIATT_XLA / GridSelfAttention pattern); it also rebinds the atom encoder / decoder functions;
#          * triatt_xla and trimul_cd BEFORE the add-on (FlashGridSelfAttention / FlashTriangleMultiplication derive from modules.GridSelfAttention /
#            modules.TriangleMultiplication at import);
#          * dattn, ttr, atom_attn, lnp rebind names no other lever and not the add-on touches — their relative order is free (lnp: haiku_modules.LayerNorm).
#          tests/test_install_order.py asserts the table: targets pairwise disjoint EXCEPT the declared `sample` chain (one REPLACES, installed first).
#   pass 2 (after the import, INSTALL_AFTER_ADDON): hoist_logits — it subclasses the add-on's HoistTransformer, which must exist first.
# big_launch.install_fast runs the same two passes around its own add-on import; nothing else installs tree levers.
TREE_LEVERS = (("cond_share", "COND_SHARE"), ("atom_cond_hoist", "ATOM_COND_HOIST"), ("dattn", "DATTN"), ("ttr", "TTR"), ("triatt_xla", "TRIATT_XLA"), ("sampler_bf16", "SAMPLER_BF16"), ("atom_attn", "ATOM_ATTN"), ("trimul_cd", "TRIMUL_CD"), ("lnp", "LNP"), ("hoist_logits", "HOIST_LOGITS"))   # inprocess module -> lever id, IN INSTALL ORDER: each installs when its own switch is set (the mode table sets it)
INSTALL_AFTER_ADDON_MODULES = ("hoist_logits",)                                                                   # pass 2 (the module's INSTALL_AFTER_ADDON flag is the mechanism; this tuple is the table's statement of it, asserted equal by the tests)
FPF_LEVERS = {"trimul": "FPF_TRIMUL", "triatt": "FPF_TRIATT"}                                  # the add-on's kernel names -> the mode's lever ids (modes.SERVED_KERNELS reads the same pairs)
EXIT_NOT_ACTIVE = 3                                                                            # the kit's NOT ACTIVE code (opt_core.report.EXIT_NOT_ACTIVE)


def load_inprocess(name: str):
    """Load af3_jax_opt/inprocess/<name>.py by file path as module ``af3_jax_opt.inprocess.<name>`` — no sys.path entry, no bare top-level
    name, and without importing the wrapper package (this interpreter is the fork's: neither af3_jax_opt nor the core is installed in it; the core's directory is on PYTHONPATH for
    the add-on's carried-kernel imports only)."""
    spec = importlib.util.spec_from_file_location(f"af3_jax_opt.inprocess.{name}", os.path.join(HERE, "inprocess", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_cache_key() -> str:
    """The persistent-cache key portability rebinding (inprocess/portable_cache_key.py): installed first in every model process the tree
    launches (modes exact, fast, big), before any compile; a jax it was not written for exits the process by name."""
    return load_inprocess("portable_cache_key").main_guard()


def install_template_guard() -> None:
    """The template census (inprocess/templates.py): wraps the fork's template featurisation so every featurised input prints its
    TEMPLATES census (monitoring only: the input runs exactly as the stock script runs it). A failure to install is printed by name
    (``TEMPLATES guard=failed:<Exc>``)."""
    try:
        load_inprocess("templates").install()
    except Exception as e:  # noqa: BLE001
        from opt_core.oom import is_oom
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        print(f"{PREFIX} TEMPLATES guard=failed:{type(e).__name__}:{str(e).replace(' ', '_')[:200]}", flush=True)
    else:
        print(f"{PREFIX} TEMPLATES guard=installed site=alphafold3.model.features.Templates.compute_features", flush=True)


def _install_tree_levers(late: bool = False) -> None:
    """The tree's in-process levers, each when its switch is set (the mode table sets it): installed BEFORE the add-on's import, so every later
    binding sees them; a module with INSTALL_AFTER_ADDON (it rebinds the add-on itself) installs in the late pass, after `import
    af3_flashpairformer`. A failure is printed by name here and surfaces as <word>=off on the SERVED line (levers_short downstream), never silently."""
    for name, lever in TREE_LEVERS:
        try:
            mod = (_TREE.get(name) if late else None) or load_inprocess(name)
        except Exception as e:
            from opt_core.oom import is_oom
            if is_oom(e): raise                                           # OOM propagates: no fallback applied (opt_core.oom.is_oom)
            print(f"{PREFIX} SERVED warning=inprocess/{name}.py failed to load: {type(e).__name__}: {e}", flush=True)
            continue
        _TREE[name] = mod
        if bool(getattr(mod, "INSTALL_AFTER_ADDON", False)) != late:
            continue
        if mod.wanted():
            try:
                mod.install()
            except Exception as e:
                from opt_core.oom import is_oom
                if is_oom(e): raise                                       # OOM propagates: no fallback applied (opt_core.oom.is_oom)
                print(f"{PREFIX} SERVED warning={lever} install failed: {type(e).__name__}: {e}", flush=True)


def _count_hoist() -> None:
    """After the add-on's import: when its hoist is installed (Model._sample_diffusion rebound to the hoisted sampler), wrap that bound
    function with a call counter. Anything unexpected leaves the install untouched and hoist reads off/0 — lever_evidence then names it."""
    try:
        from af3_flashpairformer import diffusion_hoist as dh
        from alphafold3.model import model as af3_model
    except Exception as e:                                                # the probe itself failed (the add-on's module or the fork not importable here): hoist=probe_error:<Exc>, never 'off'
        from opt_core.oom import is_oom
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        _HOIST["probe_error"] = type(e).__name__
        return
    stock = (getattr(dh, "_STOCK", None) or {}).get("sample")
    current = af3_model.Model._sample_diffusion
    if stock is None or current is stock:                                 # not installed (AF3_DIFFUSION_HOIST off) — report hoist=off
        return
    _HOIST["installed"] = True                                            # installed; 'uncounted' on the line unless the counter binds below

    def _sample_diffusion_counted(self, *args, **kwargs):
        _HOIST["calls"] += 1
        return current(self, *args, **kwargs)

    try:                                                                  # bound the way the add-on binds its own (diffusion_hoist.install: hk.transparent): no module name scope of its own
        import haiku as hk
        _sample_diffusion_counted = hk.transparent(_sample_diffusion_counted)
    except Exception as e:                                                # haiku always imports where the add-on installed; if not, leave the install as it is
        from opt_core.oom import is_oom
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        return
    af3_model.Model._sample_diffusion = _sample_diffusion_counted
    _HOIST["counted"] = True


def _served_line() -> str:
    try:
        import af3_flashpairformer as fpf
        sites = fpf.served_report()
    except Exception as e:                                                # the add-on did not install: the install line is absent too
        from opt_core.oom import is_oom
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        return f"{PREFIX} SERVED error={type(e).__name__}: {e}"
    counts = {(k, t): 0 for k in ("trimul", "triatt") for t in ("fused", "fallback")}
    shapes = []
    for (kernel, tag, shape, dtype), n in sorted(sites.items(), key=str):
        counts[(kernel, tag)] = counts.get((kernel, tag), 0) + int(n)
        if tag == "fallback":
            shapes.append(f"{kernel}:{'x'.join(str(d) for d in shape)}:{dtype}")
    hoist = (f"probe_error:{_HOIST['probe_error']}" if _HOIST["probe_error"] else "off") if not _HOIST["installed"] else (str(_HOIST["calls"]) if _HOIST["counted"] else "uncounted")
    held = {}
    try:                                                                  # the add-on's tile table for this GPU: its own entry, or its named fallback to the sm_90 table (patch.py status)
        st = fpf.status()
        tiles = str(st.get("tile_table") or "unknown").replace(" ", "_")
        cc = str(st.get("compute_capability") or "unknown").replace(" ", "")
        held = dict(st.get("held") or {})                                 # kernel -> fallback:no_tiles_cc<NN>(<kind>): a kernel whose class stayed stock on this part, by name
    except Exception as e:
        from opt_core.oom import is_oom
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        tiles, cc = f"error:{type(e).__name__}", "unknown"
    d = _TREE["dattn"].report() if _TREE["dattn"] else {"installed": False, "traced": 0, "sites": {}}
    dattn = str(d["traced"]) if d["installed"] else "off"
    sites = ",".join(f"{k}:{v}" for k, v in sorted(d["sites"].items())) or "none"
    t = _TREE["ttr"].report() if _TREE["ttr"] else {"installed": False, "fused": 0, "routed": 0, "fallback": {}}
    ttr = str(t["fused"]) if t["installed"] else "off"
    ttr_fb = ",".join(f"{k}:{v}" for k, v in sorted(t["fallback"].items())) or "none"
    b = _TREE["sampler_bf16"].report() if _TREE["sampler_bf16"] else {"installed": False, "traced": 0, "sites": {}}
    sbf16 = str(b["traced"]) if b["installed"] else "off"
    sbf16_sites = ",".join(f"{k}:{v}" for k, v in sorted(b["sites"].items())) or "none"
    x = _TREE["triatt_xla"].report() if _TREE["triatt_xla"] else {"installed": False, "traced": 0, "rows": {}, "aside": {}}
    txla = (str(x["traced"]) if x["installed"] else "off")                       # the core's triangle-attention kernels: served calls at trace time (inprocess/triatt_xla.py report), the rows that served, the step-asides by name
    txla_rows = ",".join(f"{k}:{v}" for k, v in sorted(x["rows"].items())) or "none"
    txla_aside = ",".join(f"{k}:{v}" for k, v in sorted(x["aside"].items())) or "none"
    aa = _TREE["atom_attn"].report() if _TREE["atom_attn"] else {"installed": False, "traced": 0, "sites": {}, "fallback": {}}
    atomattn = str(aa["traced"]) if aa["installed"] else "off"
    atomattn_sites = ",".join(f"{k}:{v}" for k, v in sorted(aa["sites"].items())) or "none"
    atomattn_fb = ",".join(f"{k}:{v}" for k, v in sorted(aa["fallback"].items())) or "none"
    h = _TREE["hoist_logits"].report() if _TREE["hoist_logits"] else {"installed": False, "traced": {"step": 0}, "skipped": "", "dtype": None}
    hlog = str(h["traced"]["step"]) if h["installed"] else ("skipped" if h["skipped"] else "off")
    hlog_dtype = (h["dtype"] or "none") if h["installed"] else (h["skipped"] or "none")
    c = _TREE["cond_share"].report() if _TREE["cond_share"] else {"installed": False, "traced": 0}
    cshare = str(c["traced"]) if c["installed"] else "off"
    a = _TREE["atom_cond_hoist"].report() if _TREE["atom_cond_hoist"] else {"installed": False, "precomputed": 0, "enc": 0, "dec": 0, "passed": 0, "aside": {}}
    ach_aside = ",".join(f"{k}:{v}" for k, v in sorted((a.get("aside") or {}).items())) or "none"
    achoist = (str(a["precomputed"]) if (a["precomputed"] or "no_hoist_step" not in (a.get("aside") or {})) else "skipped") if a["installed"] else "off"   # skipped = FPF_HOIST's step not in force: the lever steps aside by name with the hoist
    achoist_sites = (f"enc:{a['enc']},dec:{a['dec']},passed:{a['passed']}") if a["installed"] else "none"
    cd = _TREE["trimul_cd"].report() if _TREE["trimul_cd"] else {"installed": False, "traced": 0, "rows": {}, "aside": {}}
    tcd = (str(cd["traced"]) if cd["installed"] else "off")                    # the core's provider row at the triangle-multiplication site: served calls at trace time (inprocess/trimul_cd.py report), the arm(s) that served, the step-asides by name
    tcd_rows = ",".join(f"{k}:{v}" for k, v in sorted(cd["rows"].items())) or "none"
    tcd_aside = ",".join(f"{k}:{v}" for k, v in sorted(cd["aside"].items())) or "none"
    tcd_word = (cd.get("word") or "none") if cd["installed"] else "none" 
    tcd_uncov = ",".join(f"{k}:{v}" for k, v in sorted((cd.get("uncovered") or {}).items())) or "none"   # calls whose form has no measured cell on this card / stack (served by the provider's declared fallback; the provider prints UNCOVERED_CELL, opt_core.cell_census)
                                   # the provider word this process named (tier word `fast` by default; a row / arm word when AF3_JAX_TRIMUL_CD carries one)
    cnoise, cnoise_rule = "off", "none"                                   # constant words of the SERVED line: the sampler's noise is the fork's own draw
    return (f"{PREFIX} SERVED trimul fused={counts[('trimul', 'fused')]} fallback={counts[('trimul', 'fallback')]} "
            f"triatt fused={counts[('triatt', 'fused')]} fallback={counts[('triatt', 'fallback')]} fallback_shapes={','.join(shapes) or 'none'} hoist={hoist} "
            f"tiles={tiles} cc={cc} dattn={dattn} dattn_sites={sites} ttr={ttr} ttr_routed={t['routed']} ttr_fallback={ttr_fb} sbf16={sbf16} sbf16_sites={sbf16_sites} txla={txla} txla_rows={txla_rows} txla_aside={txla_aside} atomattn={atomattn} atomattn_sites={atomattn_sites} atomattn_fallback={atomattn_fb} hlog={hlog} hlog_dtype={hlog_dtype} cshare={cshare} achoist={achoist} achoist_sites={achoist_sites} achoist_aside={ach_aside if a['installed'] else 'none'} cnoise={cnoise} cnoise_rule={cnoise_rule} tcd={tcd} tcd_rows={tcd_rows} tcd_aside={tcd_aside} tcd_word={tcd_word} tcd_uncovered={tcd_uncov}"
            + (" held=" + ",".join(f"{k}:{w}" for k, w in sorted(held.items())) if held else ""))   # only on a part without a tile table


def held_kernels() -> dict:
    """The add-on's kernels the core REFUSED on this GPU (patch.status()['held']: kernel -> fallback:no_tiles_cc<NN>(<kind>); a part below cc 8.0 —
    an 8.x / 9.x / 10.x part without its own table is served the generation's safe rows and installs), {} when all installed."""
    try:
        import af3_flashpairformer as fpf
        return dict((fpf.status() or {}).get("held") or {})
    except Exception as e:  # noqa: BLE001 — an add-on that did not import is the SERVED warning's to name
        from opt_core.oom import is_oom
        if is_oom(e): raise
        return {}


def refusal_line(held: dict) -> str:
    """The ONE line of a mode refused because a FlashPairformer kernel cannot engage on this GPU: the lever(s), their words, the exit code, the escape."""
    names = ",".join(f"{FPF_LEVERS.get(k, k)}={w}" for k, w in sorted(held.items(), key=lambda kw: FPF_LEVERS.get(kw[0], kw[0])))
    return (f"{PREFIX} NOT ACTIVE: {names}: the lever cannot engage on this GPU (the core's Pallas serve layer refuses its compute capability); "
            f"exit {EXIT_NOT_ACTIVE} (a mode is all of its levers; --mode exact and --mode off run without it)")


def refuse_held() -> None:
    """A kernel of the mode that cannot install on this GPU refuses the run BEFORE the launcher starts — one line, exit 3. A mode is all of its
    levers: it never runs under its name with the stock class standing in for one of them (``--mode off`` / ``--mode exact`` are the modes without it)."""
    held = held_kernels()
    if held:
        print(refusal_line(held), flush=True)
        sys.exit(EXIT_NOT_ACTIVE)


def _lnp_line() -> str:
    """The LNP lever's own census line (inprocess/lnp.py line(); served=off when its switch is not set or the module did not install)."""
    mod = _TREE.get("lnp")
    if mod is None:
        return f"{PREFIX} LNP served=off word=none rows=none units=none routed=none aside=none uncovered=none"
    try:
        return mod.line()
    except Exception as e:                                                # the report is words; a module whose report raises is named on its line
        from opt_core.oom import is_oom
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        return f"{PREFIX} LNP served=error:{type(e).__name__} word=none rows=none units=none routed=none aside=none uncovered=none"


def _report() -> None:
    sys.stdout.flush()
    print(_served_line(), flush=True)
    print(_lnp_line(), flush=True)


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(f"usage: {sys.argv[0]} <add-on launcher> <script> [flags...]")
    atexit.register(_report)
    launcher = args[0]
    install_cache_key()
    install_template_guard()
    _install_tree_levers()
    sys.path.insert(0, os.path.dirname(os.path.abspath(launcher)))       # the add-on's package beside its launcher (its launcher's own first statement)
    try:
        import af3_flashpairformer  # noqa: F401 — installs per AF3_FLASHPAIRFORMER / AF3_DIFFUSION_HOIST, exactly as the launcher's import does
    except Exception as e:                                                # named here; the missing install line makes it levers_short downstream
        from opt_core.oom import is_oom
        if is_oom(e): raise                                               # OOM propagates: no fallback applied (opt_core.oom.is_oom)
        print(f"{PREFIX} SERVED warning=af3_flashpairformer import failed before launch: {type(e).__name__}: {e}", flush=True)
    else:
        _install_tree_levers(late=True)                                   # levers that rebind the add-on (INSTALL_AFTER_ADDON), after its import
        _count_hoist()
        refuse_held()                                        # a kernel that cannot install on this GPU: NOT ACTIVE by name, exit 3, before anything runs — a mode is all of its levers
    sys.argv = args
    runpy.run_path(launcher, run_name="__main__")


if __name__ == "__main__":
    main()
