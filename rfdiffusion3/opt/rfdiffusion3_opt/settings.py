"""The design line: `rfd3 design <upstream's own key=value overrides, verbatim> [ckpt_path=<the checkpoint>]`.

There is no kit preset and no kit name for an upstream setting: every run parameter is upstream's own hydra override — `inputs=`,
`out_dir=`, `seed=`, `diffusion_batch_size=`, `n_batches=`, `compile_model=`, `inference_sampler.*=` … (rfd3/configs/inference_engine/
rfdiffusion3.yaml names them and their defaults: ckpt_path :10, skip_existing True :12, diffusion_batch_size 8 :20, n_batches 1 :21,
inference_sampler.num_timesteps 200 :44, step_scale 1.5 :45, gamma_0 0.6 :48, compile_model false :70-71) — passed through in the order
given. A line with no override runs upstream's configuration untouched; the kit adds at most one token, `ckpt_path=<the checkpoint --ckpt
or RFD3_CKPT names>`, when the line does not name one itself. The same tokens serve every mode: the stock caller (`--mode off`) and the
kit modes run one line.

Upstream's own `compile_model=true` (rfdiffusion3.yaml:70-71, documented for inference by the comment at :70, the steady-state roll-out
saving; applied by rfd3/engine.py:212-269) under `--mode off` is the STOCK route (upstream's fastest documented environment), without it
the DEFAULT route (upstream exactly as shipped) — the routes table (modes.ROUTES) reads the line through `compile_requested`; under a kit
mode the token is refused by name (modes.COMPILE_REFUSED).
"""
from __future__ import annotations

from typing import Dict, List

BATCH_KEY = "diffusion_batch_size"                   # upstream's batch knob (rfdiffusion3.yaml:20)
BATCH_DEFAULT = "8"                                  # its shipped value (rfdiffusion3.yaml:20): the B a line without the token runs
CKPT_KEY = "ckpt_path"                               # upstream's checkpoint key (rfdiffusion3.yaml:10): the one token the kit may add (from --ckpt / RFD3_CKPT) when the line names none
INPUTS_KEY, OUT_DIR_KEY = "inputs", "out_dir"        # upstream's two mandatory design keys (the spec and the output directory), the caller's own tokens
COMPILE_KEY = "compile_model"                        # upstream's torch.compile switch (rfdiffusion3.yaml:70-71; engine.py:75,164,207,212-269)
COMPILE_TOKEN = f"{COMPILE_KEY}=true"                # the hydra override of the stock route, spelled as STOCK.md spells it
_TRUE_SPELLINGS = ("true", "1", "yes", "on", "y", "t")   # the values OmegaConf's bool grammar / python truth make `if self.compile_model:` take


def kv_of(tokens) -> Dict[str, str]:
    """{key: value} of a design line's `key=value` tokens (hydra's `+` / `++` / `~` prefixes stripped; last token wins; tokens without `=`
    skipped). Accepts a dict unchanged."""
    if isinstance(tokens, dict):
        return dict(tokens)
    kv: Dict[str, str] = {}
    for tok in tokens or []:
        key, eq, value = str(tok).partition("=")
        if not eq:
            continue
        kv[key.lstrip("+~").strip()] = value
    return kv


def is_true(value) -> bool:
    """Whether a line value (or a composed config value) is one of the truth spellings OmegaConf's bool grammar / python truth take
    (_TRUE_SPELLINGS; a bool passes through). The one truth reader: compile_requested here, the modes' unserved-feature readers."""
    if isinstance(value, bool):
        return value
    return value is not None and str(value).strip().lower() in _TRUE_SPELLINGS


def compile_requested(tokens_or_kv) -> bool:
    """True when the line turns upstream's `compile_model` on — the ONE reader of that fact (the routes table, the stock caller and the
    KERNELS census all ask here)."""
    return is_true(kv_of(tokens_or_kv).get(COMPILE_KEY))


def line(tokens: List[str]) -> List[str]:
    """The full upstream command line, as a list: `rfd3 design <the line's tokens, verbatim>`."""
    return ["rfd3", "design"] + [str(t) for t in tokens]
