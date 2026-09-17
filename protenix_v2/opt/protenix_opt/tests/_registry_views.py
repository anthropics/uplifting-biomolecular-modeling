"""Views of ``registry.LEVERS`` the conformance tests compare against env.sh (the switch names a mode's levers claim), and the knobs
env.sh reads that are not levers (``KNOBS``: honoured when pre-set, exported by no mode)."""
from typing import Dict

from protenix_opt.registry import LEVERS

KNOB_KEYS = ()             # protenix_opt 0.3.51: no TriMul row knob is probe-conditional (PTX_TRIMUL is exported by env.sh under both arms)
ATT_KEYS = ("PTX_BLK_ATT", "PTX_BLK_ATT_PROVIDER_MIN_TOKENS")           # env.sh L13 case (PTX_T_ATT=triattn_cuda | native selects the PTX_BLK_ATT word; native also exports the provider floor 0 and its own token floor)

# Opt-out / opt-in knobs env.sh reads (never exported by it); a pre-set value is honoured because env.sh is sourced on top of the environment.
KNOBS: Dict[str, str] = {
    "DEADSKIP": "0 disables deadskip (env.sh L23)",
    "PTX_BLK_GRAPH": "0 disables the stack graph (env.sh L53)",
    "PTX_SAMPLER_GRAPH": "0 disables the graphed sampler and the hoist (env.sh L59)",
    "PTX_FPF_XL": "0 disables the XL policy (env.sh L93-97)",
    "PTX_FPF_XL_ALLOC": "0 leaves PYTORCH_CUDA_ALLOC_CONF alone (env.sh L96)",
    "PTX_TRIMUL": "exact | tier: the pair-stack TriMul through the shared core by tier word (env.sh: exact under ARM=E, tier under ARM=T; src/ptx_trimul_routes.py)",
    "FPF_OPS_EXTRA": "extra registry ops merged after the arm's own (env.sh L40, L87)",
    "INFOPT_FASTLN_PREBUILT": "a pre-set prebuilt fast-LN directory is honoured; unset, env.sh L77-86 auto-selects by the installed torch version",
    "PTX_LAZY_INIT": "a pre-set 0 is honoured by modes.resolve() (default 1 in exact and fast)",
}


def env_keys_for(mode_levers) -> set:
    """All switch names the given levers claim (env.sh exports + conditional + extras)."""
    out = set()
    for n in mode_levers:
        lv = LEVERS[n]
        out.update(lv.env_keys)
        out.update(lv.conditional)
    return out


def conditional_keys_for(mode_levers) -> set:
    """Switch names env.sh exports only after a runtime probe (compute capability, installed torch) for these levers."""
    return {k for n in mode_levers for k in LEVERS[n].conditional}


def env_sh_keys_for(mode_levers) -> set:
    """Switch names env.sh itself exports in a clean environment for these levers (no extras, no probe-conditional ones)."""
    return {k for n in mode_levers for k in LEVERS[n].env_keys if not LEVERS[n].extra}
