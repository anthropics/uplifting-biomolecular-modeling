"""protenix_fpf_ditfast.cond_dedupe — lever `cond_dedupe`: ENGINEERING lever (class TOLERANCE: the conditioning GEMMs run with N rows instead of N_sample*N): the diffusion
conditioning `s` (DiffusionConditioning.forward's single output) depends on the noise level only through t_hat, which the sampler hands the
denoiser as ONE scalar expanded over the N_sample dimension (generator.py: t_hat.reshape(..., 1).expand(..., N_sample); the kit's graphed step
does the same) — stock computes the identical [N, 384] conditioning 5 times (noise embedding, two transitions) and every consumer downstream
(24 blocks x 6 Linear(384->768), diffusion_module.layernorm_s/linear_no_bias_s) 5 times too. This lever computes it once: when t_hat's sample
dimension is an expanded scalar (stride 0 — decided from the tensor's strides, no device read), DiffusionConditioning.forward runs on
t_hat[..., :1] and returns s with a sample dimension of 1; every stock consumer broadcasts it (a [.., 5, N, c] op [.., 1, N, c]). A t_hat that
is not an expanded scalar (training-style per-sample noise levels) takes the stock path — that is the input's property, printed in the census.
Kit switch: PTX_COND_DEDUPE=1 (the hook calls install). Numerics: fast tier = engineering (hoisting of provably identical rows;
TOLERANCE by rule because the conditioning GEMMs see 1 row-set instead of 5); an exact-tier proposal would need --det 1 byte identity (not proposed).
"""
import os, torch
from ._plumbing import LeverRefused, diffusion_module, say, register_exit

NAME = "cond_dedupe"
REPORT = {"installed": False, "census": {"dedupe_calls": 0, "stock_path_calls": 0, "rows_in": 0, "rows_computed": 0}}


def install(model):
    dm = diffusion_module(model)
    cond = dm.diffusion_conditioning
    if "forward" in cond.__dict__ and getattr(cond.__dict__["forward"], "_fpf_lever", None) == NAME:
        return REPORT
    inner = cond.forward

    def forward(t_hat_noise_level, *a, **k):
        t = t_hat_noise_level
        c = REPORT["census"]
        if torch.is_tensor(t) and t.dim() >= 1 and t.shape[-1] > 1 and t.stride(-1) == 0:
            c["dedupe_calls"] += 1; c["rows_in"] += int(t.shape[-1]); c["rows_computed"] += 1      # e.g. 5 sample rows in -> 1 conditioning row computed
            return inner(t[..., :1], *a, **k)
        c["stock_path_calls"] += 1; n = int(t.shape[-1]) if torch.is_tensor(t) and t.dim() >= 1 else 1; c["rows_in"] += n; c["rows_computed"] += n
        return inner(t, *a, **k)
    forward._fpf_lever = NAME
    cond.forward = forward
    REPORT.update(installed=True)
    say("CONDDEDUPE:on(guard=stride0)")
    register_exit(NAME, lambda: dict(REPORT["census"], note="Python-level DiffusionConditioning calls (inside the sampler graph: record + capture steps; replays reuse the captured single-row call)"))
    return REPORT
