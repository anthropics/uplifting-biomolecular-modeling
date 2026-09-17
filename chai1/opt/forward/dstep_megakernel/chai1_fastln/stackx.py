"""lever-arm builder on top of chai1_eager.stack: a private flat denoiser per lever set so that the base line's parts stay untouched.
Levers: ``compiled`` — the hoisted per-step function through torch.compile / Inductor (fast tier: fused fp32 glue, not bitwise); ``hoist2`` — the eager stack's value-taint hoister for the denoiser step (chai1_eager.hoist.HoistedForward2, bitwise to the base hoister:
the step-invariant atom-pair block, pair biases and AdaLN conditioning leave the per-step part); ``dit_attn`` — the DiT token attention through the
shared core's pair-bias-attention provider (dit_attn.py)."""

SURFACE = ("ALL_LEVERS", "HOIST_LEVERS", "COMPILE_LEVERS", "ATTN_LEVERS", "build_lever_parts", "apply_levers", "new_flat")   # the names the kit / this package's callers reach (chai1_opt tests/test_lever_surfaces.py checks they exist, statically)

HOIST_LEVERS = {"hoist2": "hoist2"}                 # lever -> chai1_eager.stack.HoistedDiffusionWrapper(hoister=...)
COMPILE_LEVERS = {"compiled": "default"}            # lever -> HoistedDiffusionWrapper.compile (torch.compile mode of the per-step function; fast tier, not bitwise)
ATTN_LEVERS = {"dit_attn": "tier"}                    # lever -> chai1_fastln.dit_attn (the DiT token attention through opt_core kernels.apb asked by the MODE's tier word; fast / big, not bitwise)
ALL_LEVERS = tuple(HOIST_LEVERS) + tuple(COMPILE_LEVERS) + tuple(ATTN_LEVERS)
TIER_WORD = {"fast": "fast", "big": "big", "exact": "exact"}   # mode -> the provider tier word its levers ask by (big never shares the fast word)


def new_flat(comps, tag):
    from chai1_eager import ts2eager as T
    sdk = {k: v.detach() for k, v in comps.stock("diffusion_module.pt").jit_module.state_dict().items()}
    return T.load_eager_component(f"{comps.downloads}/models_v2/diffusion_module.pt", device=None,
                                  code_dir=f"{comps.code_root}/diffusion_module.pt.{tag}.code", state_dict=sdk)


def apply_levers(flat, levers, mode="fast"):
    pols = {}
    for lv in levers:
        if lv in HOIST_LEVERS:
            pols["hoister"] = HOIST_LEVERS[lv]
        elif lv in COMPILE_LEVERS:
            pols["compile"] = COMPILE_LEVERS[lv]
        elif lv in ATTN_LEVERS:
            from . import dit_attn as DA                    # opt_core kernels.apb imported at the first served call, not here
            pols["dit_attn"] = DA.patch_flat(flat, word=TIER_WORD.get(mode, "fast"))
        else:
            raise ValueError(lv)
    return pols


def build_lever_parts(comps, base_parts, line, levers, *, sink=None, graphed=True, mode="fast"):
    """levers: tuple of tokens from ALL_LEVERS.  Shares the trunk wrapper of base_parts[line]; the denoiser is a private flat copy carrying the levers."""
    import chai1_eager.stack as S
    flat = new_flat(comps, f"{line}_{'_'.join(levers)}")
    pols = apply_levers(flat, levers, mode=mode)
    dw = S.HoistedDiffusionWrapper(flat, graphed=graphed, sink=sink, hoister=pols.get("hoister", "base"))
    dw.compile = pols.get("compile")
    if dw.compile:
        from . import aoti                                   # the compiled step's ahead-of-time packages ($MODEL_OPT_JIT_ROOT/<key>/aoti), when present
        dw.aoti = aoti
    dw.dit_attn = pols.get("dit_attn")                  # the Router bound in the flat's namespace (None: lever off)
    return dict(trunk=base_parts[line]["trunk"], diffusion=dw, flat_rest=True)
