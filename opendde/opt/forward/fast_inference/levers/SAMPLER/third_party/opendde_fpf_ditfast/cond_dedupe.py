"""opendde_fpf_ditfast.cond_dedupe — lever `cond_dedupe`: ENGINEERING lever (class TOLERANCE: the conditioning GEMMs run with N rows instead of N_sample*N): the diffusion
conditioning `s` (DiffusionConditioning.forward's single output) depends on the noise level only through t_hat, which the sampler hands the
denoiser as ONE scalar expanded over the N_sample dimension (generator.py: t_hat.reshape(..., 1).expand(..., N_sample); the kit's step graph
(opendde_opt.stepgraph) does the same) — stock computes the identical [N, 384] conditioning 5 times (noise embedding, two transitions) and every consumer downstream
(24 blocks x 6 Linear(384->768), diffusion_module.layernorm_s/linear_no_bias_s) 5 times too. This lever computes it once: when t_hat's sample
dimension is an expanded scalar (stride 0 — decided from the tensor's strides, no device read), DiffusionConditioning.forward runs on
t_hat[..., :1] and returns s with a sample dimension of 1; every stock consumer broadcasts it (a [.., 5, N, c] op [.., 1, N, c]). A t_hat that
is not an expanded scalar (training-style per-sample noise levels) takes the stock path — that is the input's property, printed in the census.
Capture consistency: a CAPTURING caller (the kit's `stepgraph`: opt_core GraphCache clones the step's arguments into static
buffers, so the expanded noise level arrives contiguous — the same one value per sample, re-materialised) would send the captured step down the stock
path while every eager step takes the deduplicated one, and the caller's replay-vs-eager check refuses the graph. Inside a capture
(`torch.cuda.is_current_stream_capturing()`, a host-side query, no device read) the wrapper therefore keeps the deduplicated path once this process
has seen the expanded form from the sampler (a latch; the sampler builds t_hat by `.expand` at every step, generator.py:221), counted apart
(`dedupe_calls_captured`); the capturing caller holds the replay to the eager answer, so a noise level that were not sample-invariant there
would be refused by that check, never silently deduplicated.
Kit switch: ODDE_COND_DEDUPE=1 (odde: OpenDDE's served-levers hook -> odde_accel_v2 -> odde_sampler calls install; DiffusionConditioning.forward
(opendde/model/modules/diffusion.py:1054) takes t_hat first and returns (s, pair_z): the wrapper passes everything else through). Numerics: fast tier = engineering (hoisting of provably identical rows;
TOLERANCE by rule because the conditioning GEMMs see 1 row-set instead of 5); the exact line does not export it (byte identity under --det 1 is not claimed).
"""
import os, torch
from ._plumbing import LeverRefused, diffusion_module, say, register_exit

NAME = "cond_dedupe"
REPORT = {"installed": False, "census": {"dedupe_calls": 0, "dedupe_calls_captured": 0, "stock_path_calls": 0, "rows_in": 0, "rows_computed": 0}}
LATCH = {"expanded_seen": False}                                                     # odde: the sampler's expanded (stride-0) noise level seen in this process


def _capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001
        return False
COUNTS = REPORT["census"]                                                            # odde: the live counter dict opendde_opt/ran.py reads


def install(model):
    dm = diffusion_module(model)
    cond = dm.diffusion_conditioning
    if "forward" in cond.__dict__ and getattr(cond.__dict__["forward"], "_fpf_lever", None) == NAME:
        return REPORT
    inner = cond.forward

    def forward(t_hat_noise_level, *a, **k):
        t = t_hat_noise_level
        c = REPORT["census"]
        if torch.is_tensor(t) and t.dim() >= 1 and t.shape[-1] > 1:
            if t.stride(-1) == 0:
                LATCH["expanded_seen"] = True
                c["dedupe_calls"] += 1; c["rows_in"] += int(t.shape[-1]); c["rows_computed"] += 1      # e.g. 5 sample rows in -> 1 conditioning row computed
                return inner(t[..., :1], *a, **k)
            if LATCH["expanded_seen"] and _capturing():                                          # odde: a capturing caller's contiguous static copy of the expanded level
                c["dedupe_calls"] += 1; c["dedupe_calls_captured"] += 1; c["rows_in"] += int(t.shape[-1]); c["rows_computed"] += 1
                return inner(t[..., :1], *a, **k)
        c["stock_path_calls"] += 1; n = int(t.shape[-1]) if torch.is_tensor(t) and t.dim() >= 1 else 1; c["rows_in"] += n; c["rows_computed"] += n
        return inner(t, *a, **k)
    forward._fpf_lever = NAME
    cond.forward = forward
    REPORT.update(installed=True)
    say("CONDDEDUPE:on(guard=stride0|captured-after-stride0)")
    register_exit(NAME, lambda: dict(REPORT["census"], note="Python-level DiffusionConditioning calls (inside the sampler graph: record + capture steps; replays reuse the captured single-row call)"))
    return REPORT
