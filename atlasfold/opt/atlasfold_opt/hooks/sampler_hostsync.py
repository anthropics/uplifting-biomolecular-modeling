"""Lever sampler_hostsync (class exact) — DiffusionHead.sample without its per-step HOST SYNCS.

Site: ``atlasfold.model.network.diffusion_head:DiffusionHead.sample`` (diffusion_head.py L283-373), re-stated.  The stock roll-out blocks the
host three times per step (STOCK_SYNCS_PER_STEP x num_steps per roll-out):
  * `if not mask.any():` in do_centering (utils/geometry/random_augment.py L91) and again in get_center (L47), reached from
    random_augmentation (diffusion_head.py L382) every step — a device->host bool of a mask that is CONSTANT over the roll-out
    (`mask = batch["atom14_mask"].unsqueeze(1)`, L317, viewed [B, 1, L*14]);
  * `torch.tensor(self.c_noise(t_hat), device=device)` in run_step (L340) — a blocking host->device upload of one fp32 scalar per step, whose
    values are a pure function of the sampling config (sigmas, gamma_0, gamma_min) known before the loop starts.
The re-statement makes exactly two changes and nothing else: the predicate is evaluated ONCE per roll-out (`bool(mask.any())`, one device->host
read-back) and answered from that bool inside do_centering / get_center (their statements otherwise verbatim, run in the same order under the
same autocast-off region), and the c_noise scalars are uploaded ONCE per roll-out as one fp32 table built from the loop's own float expressions
(`torch.tensor([self.c_noise(t_hat_1), ...], device=device)`: the same double->float32 rounding per element as the per-step `torch.tensor(float)`),
each step reading its [1, 1] row by a device-side view.  Same RNG draws in the same order, same statements, same dtypes: bit-identical to the
stock roll-out by construction (outputs byte-identical to stock under `--det 1`; the kit's CPU unit test compares the two roll-outs with
torch.equal).  Host syncs left inside sample() per roll-out: the one predicate read-back and the one schedule upload.

Install contract: the re-statement REPLACES the stock body, so this lever must be the innermost patch of DiffusionHead.sample — it installs
before every lever that wraps sample() (diffusion_bf16, denoiser_graph, sampler_hoist; the mode rows order it so) and refuses by name
(`not_innermost:<qualname>`) if it finds sample() already wrapped; a stock text whose digest is not the one re-stated here (sample,
random_augmentation, do_centering, get_center) refuses by name (`source:<fn>`).  AFO_SAMPLER_HOSTSYNC=0: installed, every roll-out runs the
stock body (counted `disabled`, an expected word; `switch=AFO_SAMPLER_HOSTSYNC=0` on the LEVER line).  LEVER line: served per roll-out as
`L<L>xB<B>`; counters rollouts, steps, host_syncs (predicate read-backs issued), host_syncs_avoided, stock_syncs_per_step."""
from __future__ import annotations

import math
import os
from typing import Dict

import torch

from opt_core.counters import Ledger

from . import Installed, rebind

NAME = "LOCAL.atlasfold.sampler_hostsync"
IMPL = "schedule-table+mask-once"
ENV_SWITCH = "AFO_SAMPLER_HOSTSYNC"
EXPECTED = ("disabled",)
T_HEAD = "atlasfold.model.network.diffusion_head"
T_RA = "atlasfold.utils.geometry.random_augment"
SOURCE_SHA256 = {                    # opt_core.diffusion_loop.source_guard.source_sha256 of the texts re-stated below (atlasfold v1.0.0 as carried in stock/src)
    "DiffusionHead.sample": ("3ed3e7037c73d38f7791898487555ecc0ccadb042441817b6815b6b0b6e375b0",),
    "DiffusionHead.random_augmentation": ("4e966d08000b18b603160566033fcf26e25ff046de6cf3d63d32781245522737",),
    "do_centering": ("26fb8eb0d2fb48c2c7b818be5cfcbd0cfbcf87473297e9cf86dc9c7188098034",),
    "get_center": ("6db549d460bc2ea0f4a30d5bea791f4b5156627eb555c65cacb3926d2bd789ab",),
}
STOCK_SYNCS_PER_STEP = 3             # do_centering mask.any() + get_center mask.any() + torch.tensor(c_noise) upload


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV_SWITCH, "1")).strip() not in ("0", "off", "false", "no")


# ----------------------------------------------------------------------------------------------------------------- the two sync sites, answered once
def _get_center(coords: torch.Tensor, mask: torch.Tensor, any_mask: bool) -> torch.Tensor:
    """random_augment.get_center (torch branch, L32-68) with `mask.any()` answered by the roll-out's bool."""
    if not any_mask:
        raise ValueError("Mask has no True values; cannot compute center.")
    assert isinstance(mask, torch.Tensor)
    assert coords.dtype == torch.float32
    # Sanitize coords
    safe_coords = coords.masked_fill(~mask[..., None], 0.0)
    total_mass = mask.sum(-1, keepdim=True).clamp(1)
    center = torch.sum(safe_coords, dim=-2, keepdim=True) / total_mass[..., None]
    return center


def _do_centering(coords: torch.Tensor, mask: torch.Tensor, any_mask: bool, mask_to_zero: bool = True) -> torch.Tensor:
    """random_augment.do_centering (L70-102) with `mask.any()` answered by the roll-out's bool."""
    assert coords.ndim == mask.ndim + 1
    assert coords.ndim >= 2
    if not any_mask:
        # If no positions are masked, return coords as is
        return coords
    center_pos = _get_center(coords, mask, any_mask)  # shape (*, 1, 3)
    centered_coords = coords - center_pos
    if mask_to_zero:
        centered_coords[~mask] = 0.0
    return centered_coords


def _random_augmentation(coords: torch.Tensor, mask: torch.Tensor, any_mask: bool) -> torch.Tensor:
    """DiffusionHead.random_augmentation (diffusion_head.py L375-391) verbatim but for do_centering's predicate."""
    from atlasfold.utils.geometry.random_augment import random_rotations_torch
    B, N, L, _, _ = coords.shape
    coords = coords.view(B, N, L * 14, 3)
    mask = mask.view(B, -1, L * 14)

    with torch.autocast(coords.device.type, enabled=False):
        coords = _do_centering(coords, mask, any_mask, mask_to_zero=False)

        R = random_rotations_torch((N,), coords.device)  # [N, 3, 3]
        coords = coords @ R.unsqueeze(0)  # [B, N, L*14, 3]

        noise = torch.randn((1, N, 1, 3), device=coords.device)  # [1, N, 1, 3]
        coords.add_(noise)

        # Mask out
        coords.masked_fill_(~mask[..., None], 0.0)
    return coords.view(B, N, L, 14, 3)


def schedule(config) -> Dict[str, list]:
    """The loop's own float expressions (diffusion_head.py L356-359), evaluated ahead of it: sigmas and the per-step t_hat."""
    sigmas = config.get_sigmas()
    t_hats = []
    for step in range(1, config.num_steps + 1):
        sigma_tm, sigma_t = sigmas[step - 1], sigmas[step]
        gamma = config.gamma_0 * (sigma_t > config.gamma_min)
        t_hat: float = sigma_tm * (1 + gamma)
        t_hats.append(t_hat)
    return {"sigmas": sigmas, "t_hats": t_hats}


# ----------------------------------------------------------------------------------------------------------------- the roll-out, re-stated
def make_sample(stock_sample, ledger: Ledger):
    from atlasfold.model.network.diffusion_head import SamplingConfig

    def sample(self, batch, s, z, num_samples: int = 1, config=None):
        """DiffusionHead.sample (diffusion_head.py L283-373): the stock statements, the mask predicate and the c_noise upload hoisted."""
        if not enabled():
            ledger.fallback("disabled"); ledger.count("rollouts")
            return stock_sample(self, batch, s, z, num_samples, config)
        config = SamplingConfig() if config is None else config
        device = s.device

        B, L = batch["aatype_int"].shape
        N = num_samples
        mask = batch["atom14_mask"].unsqueeze(1)  # (B, 1, L, 14)

        def sample_noise() -> torch.Tensor:
            """Sample noise with synchronized randomness across different inputs.
            This ensures that the resulting coordinates are the same regardless of
            batch size.
            """
            return torch.randn(
                size=(1, N, L, 14, 3), device=device, dtype=torch.float32
            ).expand(B, -1, -1, -1, -1)  # (B, N, L, 14, 3)

        # Get noise schedule
        sched = schedule(config)
        sigmas: list[float] = sched["sigmas"]
        sigma_0 = sigmas[0]

        x = sigma_0 * sample_noise()  # (B, N, L, 14, 3)
        x.masked_fill_(~mask[..., None], 0.0)  # apply atom mask

        # Compute time-independent variables
        # Algorithm 20 Line 1: DiffusionConditioning
        pair_bias = self.pair_conditioning(batch, z)
        del z

        # HOISTED (1/2): the c_noise scalars of every step as ONE fp32 upload — each element the value `torch.tensor(self.c_noise(t_hat), device=device)` holds
        c_noise_table = torch.tensor([self.c_noise(t_hat) for t_hat in sched["t_hats"]], device=device).view(-1, 1, 1)  # [steps, 1, 1]
        # HOISTED (2/2): do_centering's / get_center's `mask.any()` — the mask is constant over the roll-out: one device->host bool
        any_mask = bool(mask.view(B, -1, L * 14).any())
        ledger.count("host_syncs", 1)

        def run_step(x_noisy: torch.Tensor, t_hat: float, k: int) -> torch.Tensor:
            c_noise = c_noise_table[k]  # [1, 1] — stock (L340): torch.tensor(self.c_noise(t_hat), device=device).view(1, 1)
            # Algorithm 20 Line 1: DiffusionConditioning
            single_cond = self.single_conditioning(batch, s, c_noise)
            return self.inference_step(
                batch,
                x_noisy,
                t_hat,
                single_cond,
                pair_bias,
                chunk_size=config.chunk_size,
            )

        # Gradually denoise
        for step in range(1, config.num_steps + 1):
            # Apply centering and random augmentation.
            x = _random_augmentation(x, mask, any_mask)
            sigma_tm, sigma_t = sigmas[step - 1], sigmas[step]

            gamma = config.gamma_0 * (sigma_t > config.gamma_min)
            t_hat: float = sigma_tm * (1 + gamma)

            # Add noise
            noise_var: float = config.noise_scale**2 * (t_hat**2 - sigma_tm**2)
            eps = math.sqrt(noise_var) * sample_noise()  # (B, N, L, 14, 3)
            eps.masked_fill_(~mask[..., None], 0.0)  # apply atom mask
            x_noisy = x + eps

            # Denoise
            x_denoised = run_step(x_noisy, t_hat, step - 1)

            delta = (x_noisy - x_denoised) / t_hat
            dt = sigma_t - t_hat
            x = x_noisy + config.step_scale * dt * delta
        ledger.count("rollouts"); ledger.count("steps", config.num_steps); ledger.count("host_syncs_avoided", STOCK_SYNCS_PER_STEP * config.num_steps - 1)
        ledger.serve(f"L{L}xB{B}")
        return x
    sample.__qualname__ = "DiffusionHead.sample[atlasfold_opt:sampler_hostsync]"
    return sample


# ----------------------------------------------------------------------------------------------------------------- install
def source_check() -> Dict[str, str]:
    """{fn: 'observed digest'} for every re-stated text whose digest is NOT in SOURCE_SHA256 (empty = all match)."""
    import importlib
    from opt_core.diffusion_loop.source_guard import source_sha256
    H = importlib.import_module(T_HEAD); RA = importlib.import_module(T_RA)
    funcs = {"DiffusionHead.sample": H.DiffusionHead.sample, "DiffusionHead.random_augmentation": H.DiffusionHead.random_augmentation,
             "do_centering": RA.do_centering, "get_center": RA.get_center}
    bad = {}
    for fn, f in funcs.items():
        while hasattr(f, "__wrapped_stock__"):
            f = f.__wrapped_stock__
        try:
            d = source_sha256(f)
        except Exception as e:  # noqa: BLE001
            d = f"unreadable:{type(e).__name__}"
        if d not in SOURCE_SHA256[fn]:
            bad[fn] = d
    return bad


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        H = importlib.import_module(T_HEAD)
    except Exception as e:  # noqa: BLE001
        return Installed("sampler_hostsync", False, reason=f"import:{type(e).__name__}")
    head_cls = H.DiffusionHead
    cur = head_cls.sample
    if hasattr(cur, "__wrapped_stock__") or getattr(cur, "__qualname__", "") != "DiffusionHead.sample":
        return Installed("sampler_hostsync", False, reason=f"not_innermost:{getattr(cur, '__qualname__', type(cur).__name__)}")
    bad = source_check()
    if bad:
        return Installed("sampler_hostsync", False, reason="source:" + "+".join(sorted(bad)), facts={"digests": bad})
    ledger = Ledger(NAME, impl=IMPL, origin="kit", expected=EXPECTED)
    for k in ("rollouts", "steps", "host_syncs", "host_syncs_avoided"):
        ledger.set(k, 0)
    ledger.set("stock_syncs_per_step", STOCK_SYNCS_PER_STEP)
    stock_sample = cur
    rebind(head_cls, "sample", make_sample(stock_sample, ledger), stock_sample)

    def restore() -> None:
        if getattr(head_cls.sample, "__wrapped_stock__", None) is stock_sample:
            head_cls.sample = stock_sample

    def line():
        ev = {}
        if not enabled():
            ev["switch"] = f"{ENV_SWITCH}=0"
        return ledger.line(tag, **ev)

    def gate():
        return ledger.gate(require_served=False) if not enabled() else ledger.gate()
    return Installed("sampler_hostsync", True, lines=[line], gates=[gate], facts={"impl": IMPL, "ledger": ledger, "restore": restore})
