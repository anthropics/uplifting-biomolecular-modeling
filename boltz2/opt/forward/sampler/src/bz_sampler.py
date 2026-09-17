"""bz_sampler.py — the `sampler` levers: the diffusion sampler's roll-out
(boltz 2.2.1, inference only; runtime monkeypatch — no file of the boltz package is edited; nothing is installed unless install() is called).

Levers (each one is ONE row env word read by the kit adapter boltz2_opt/sampler.py, which calls install(); each reports on|off|skipped by name):

  EXACT tier (bitwise = stock; construction argument below; checked as sha256 of every output file vs `--mode off`):
    rollout   BOLTZ_SAMPLER_ROLLOUT=graph   AtomDiffusion.sample as a device-resident roll-out: the sigma/gamma schedule read to the host ONCE
              (one .tolist() per table; the stock loop reads the same fp32 values with 3 .item() per step), every random tensor pre-drawn on the
              CUDA generator in EXACTLY the stock order and shapes (init randn(shape); per step randn((m,4)), randn((m,1,3)), randn(shape)) so
              the Philox stream and every later draw are unchanged, and ONE CUDA graph per step BOUNDARY = [rigid-alignment tail of step k
              (rotation from the eager SVD's U, det-corrected F, V; aligned coords) + Euler update of step k + centre / random augmentation /
              noise of step k+1 + the preconditioned DiffusionModule forward of step k+1 + the alignment head of step k+1 (centroids, centred
              clouds, 3x3 weighted covariance)] captured once per sample() against the live conditioning tensors (no clones) and replayed S-1
              times; released when sample() returns. Between two replays only torch.linalg.svd(driver="gesvd") + torch.det stay eager
              (cuSOLVER: host-synchronising and not capturable; its bits ARE the stock bits). The two host-syncing guard prints of
              weighted_rigid_align are kept WITHOUT per-step syncs: the point-count guard depends on the step-invariant mask and is evaluated
              once per sample() (printed per step as stock would); the singular-value guard is evaluated for all steps after the loop from the
              kept S tensors (same warning lines, same count; printed after the loop instead of inside it).
              Per-step scalars live in device fp32 tables indexed by an on-device step counter: t_hat (== torch.full((b,), t_hat)),
              sqrt(noise_var) (stock `sqrt(noise_var) * randn` = one fp32 multiply by the rounded scalar), the Euler coefficient
              fp32(step_scale*(sigma_t - t_hat)) (stock `c * tensor` = one fp32 multiply by the rounded scalar) and the reciprocal ATen itself
              uses for `tensor / t_hat` (ATen divides a CUDA tensor by a CPU scalar by MULTIPLYING with a host-computed reciprocal —
              BinaryDivTrueKernel.cu; which rounding of the reciprocal this torch build uses is decided at install by probe_scalar_div()
              against torch's own division on the 200 t_hat values of the schedule, and the lever REFUSES BY NAME if neither candidate
              reproduces torch bit for bit).
              Construction argument: same kernels on the same shapes / dtypes / memory layouts in the same order for every value that reaches
              an output (cuBLAS / cuSOLVER calls identical; fp32 elementwise arithmetic identical incl. rounding order — mul then add as two
              kernels exactly as stock, no FMA regrouping), same RNG consumption; graph replay executes the captured kernels unchanged.
              Step 0 runs eagerly (its op list differs: no previous-denoised branch) and doubles as the capture's warm-up (the kit hoist's
              caches are (re)built there); the last step's tail runs eagerly with the stock Python-float expressions.
              Out of scope BY NAME (census `scope=<reason>`; the inner sampler serves the call unchanged): steering / guidance potentials
              (`--use_potentials`: the kit takes the sampler levers off for that run anyway), a non-CUDA device, training mode,
              alignment_reverse_diff off (never at inference). A non-OOM capture failure is named (`capture_failed`) and the SAME statements
              then run eagerly from the pre-drawn tables (still the stock arithmetic); BOLTZ_SAMPLER_ROLLOUT=eager is that path as the
              falsification control.

  FAST tier (tolerance class; each its own word, own state line):
    align     BOLTZ_SAMPLER_ALIGN=aligncap  (EXACT tier value) opt/forward/waste/aligncap.py: the same cusolverDnSgesvd torch calls, eager
              BETWEEN replays, straight into static U/S/Vh buffers with torch's own strides (no clone, no devInfo host read, no copies); det, F,
              R and the tail move INTO the graph (torch.det is capturable). Bitwise class.
    align     BOLTZ_SAMPLER_ALIGN=jacobi64  registry lever `align_jacobi64`: the rigid alignment's 3x3 SVD + det by ONE sync-free in-graph kernel
              (one-sided Jacobi on H^T H in fp64 registers, R = U diag(1,1,det(U Vh)) Vh rounded to fp32) -> no host sync
              left in the loop: S-1 back-to-back graph replays. Numerics: R agrees with cuSOLVER's fp32 gesvd to ~1e-6 (a change OUTSIDE the
              network -> its own word, own state line; absent = the stock gesvd + det eager between replays, bitwise).
    the fused bf16 token-transformer step levers live in bz_sampler_dit.py and compose INSIDE this roll-out (graph or eager).

Composition: install() wraps whatever AtomDiffusion.sample is current (the stock method); the kit's DiT hoist (boltz_dit_hoist.py, applied
later by the worker variant, outermost) wraps this roll-out: its step-invariant caches are (re)built at the eager step 0 of every sample() and
this module's graph — captured after step 0, released when sample() returns — reads them at stable addresses. The kit's per-step CUDA-graph
patch (boltz_graph_patch.py, BOLTZ_GRAPH_DIFFUSION) must be OFF in a row that turns the roll-out on (both own AtomDiffusion.sample; the
adapter refuses the combination by name).

Dev hook (not a lever, no environment word): the module attribute DEV_DUMP_DIR, when set in-process by a caller (nothing in
this tree sets it), writes the sampler's inputs of every sample() call served here (conditioning tensors, feats, the CUDA generator state) to
<dir>/dump_n<tokens>_<k>.pt. BOLTZ_SAMPLER_DEBUG=1 = verbose stderr (as BOLTZ_GRAPH_DEBUG / BOLTZ_DIT_HOIST_DEBUG).
"""
from __future__ import annotations

import os
import sys
import time
import traceback
from math import sqrt

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False

if _HAVE_TRITON:

    @triton.jit
    def _jacobi_rot(app, aqq, apq):
        """Jacobi rotation (c, s) annihilating apq of the symmetric 2x2 [[app, apq], [apq, aqq]] (fp64 scalars)."""
        small = tl.abs(apq) < 1e-300
        theta = (aqq - app) / (2.0 * tl.where(small, 1.0, apq))
        t = tl.where(theta >= 0, 1.0, -1.0) / (tl.abs(theta) + tl.sqrt(theta * theta + 1.0))
        c = 1.0 / tl.sqrt(t * t + 1.0)
        s = t * c
        c = tl.where(small, 1.0, c); s = tl.where(small, 0.0, s)
        return c, s

    @triton.jit
    def _kabsch3_kernel(Hp, Rp, DETp, SMINp, NSWEEP: tl.constexpr):
        """per batch element b: H = Hp[b] (3x3 fp32, row-major) = U S Vh (SVD, S descending); R = U diag(1, 1, det(U Vh)) Vh -> Rp[b] (== stock's
        rot_matrix after the det correction); det(U Vh) -> DETp[b]; the smallest singular value -> SMINp[b] (weighted_rigid_align's guard 2).
        Jacobi eigen-decomposition of H^T H in fp64 registers, U columns from H v_i re-orthonormalised (u2 = +-(u0 x u1)).
        One program per batch element b (the program id)."""
        b = tl.program_id(0).to(tl.int64)
        base = Hp + b * 9
        h00 = tl.load(base + 0).to(tl.float64); h01 = tl.load(base + 1).to(tl.float64); h02 = tl.load(base + 2).to(tl.float64)
        h10 = tl.load(base + 3).to(tl.float64); h11 = tl.load(base + 4).to(tl.float64); h12 = tl.load(base + 5).to(tl.float64)
        h20 = tl.load(base + 6).to(tl.float64); h21 = tl.load(base + 7).to(tl.float64); h22 = tl.load(base + 8).to(tl.float64)
        a00 = h00 * h00 + h10 * h10 + h20 * h20
        a01 = h00 * h01 + h10 * h11 + h20 * h21
        a02 = h00 * h02 + h10 * h12 + h20 * h22
        a11 = h01 * h01 + h11 * h11 + h21 * h21
        a12 = h01 * h02 + h11 * h12 + h21 * h22
        a22 = h02 * h02 + h12 * h12 + h22 * h22
        v00 = a00 * 0.0 + 1.0; v01 = a00 * 0.0; v02 = a00 * 0.0
        v10 = a00 * 0.0; v11 = a00 * 0.0 + 1.0; v12 = a00 * 0.0
        v20 = a00 * 0.0; v21 = a00 * 0.0; v22 = a00 * 0.0 + 1.0
        for _ in tl.static_range(NSWEEP):
            c, s = _jacobi_rot(a00, a11, a01)
            n00 = c * c * a00 - 2.0 * s * c * a01 + s * s * a11
            n11 = s * s * a00 + 2.0 * s * c * a01 + c * c * a11
            n02 = c * a02 - s * a12
            n12 = s * a02 + c * a12
            a00 = n00; a11 = n11; a01 = a00 * 0.0; a02 = n02; a12 = n12
            t0 = c * v00 - s * v01; t1 = s * v00 + c * v01; v00 = t0; v01 = t1
            t0 = c * v10 - s * v11; t1 = s * v10 + c * v11; v10 = t0; v11 = t1
            t0 = c * v20 - s * v21; t1 = s * v20 + c * v21; v20 = t0; v21 = t1
            c, s = _jacobi_rot(a00, a22, a02)
            n00 = c * c * a00 - 2.0 * s * c * a02 + s * s * a22
            n22 = s * s * a00 + 2.0 * s * c * a02 + c * c * a22
            n01 = c * a01 - s * a12
            n21 = s * a01 + c * a12
            a00 = n00; a22 = n22; a02 = a00 * 0.0; a01 = n01; a12 = n21
            t0 = c * v00 - s * v02; t1 = s * v00 + c * v02; v00 = t0; v02 = t1
            t0 = c * v10 - s * v12; t1 = s * v10 + c * v12; v10 = t0; v12 = t1
            t0 = c * v20 - s * v22; t1 = s * v20 + c * v22; v20 = t0; v22 = t1
            c, s = _jacobi_rot(a11, a22, a12)
            n11 = c * c * a11 - 2.0 * s * c * a12 + s * s * a22
            n22 = s * s * a11 + 2.0 * s * c * a12 + c * c * a22
            n01 = c * a01 - s * a02
            n02 = s * a01 + c * a02
            a11 = n11; a22 = n22; a12 = a11 * 0.0; a01 = n01; a02 = n02
            t0 = c * v01 - s * v02; t1 = s * v01 + c * v02; v01 = t0; v02 = t1
            t0 = c * v11 - s * v12; t1 = s * v11 + c * v12; v11 = t0; v12 = t1
            t0 = c * v21 - s * v22; t1 = s * v21 + c * v22; v21 = t0; v22 = t1
        l0 = a00; l1 = a11; l2 = a22
        sw = l1 > l0
        l0, l1 = tl.where(sw, l1, l0), tl.where(sw, l0, l1)
        x0, x1 = tl.where(sw, v01, v00), tl.where(sw, v00, v01); v00 = x0; v01 = x1
        x0, x1 = tl.where(sw, v11, v10), tl.where(sw, v10, v11); v10 = x0; v11 = x1
        x0, x1 = tl.where(sw, v21, v20), tl.where(sw, v20, v21); v20 = x0; v21 = x1
        sw = l2 > l1
        l1, l2 = tl.where(sw, l2, l1), tl.where(sw, l1, l2)
        x0, x1 = tl.where(sw, v02, v01), tl.where(sw, v01, v02); v01 = x0; v02 = x1
        x0, x1 = tl.where(sw, v12, v11), tl.where(sw, v11, v12); v11 = x0; v12 = x1
        x0, x1 = tl.where(sw, v22, v21), tl.where(sw, v21, v22); v21 = x0; v22 = x1
        sw = l1 > l0
        l0, l1 = tl.where(sw, l1, l0), tl.where(sw, l0, l1)
        x0, x1 = tl.where(sw, v01, v00), tl.where(sw, v00, v01); v00 = x0; v01 = x1
        x0, x1 = tl.where(sw, v11, v10), tl.where(sw, v10, v11); v10 = x0; v11 = x1
        x0, x1 = tl.where(sw, v21, v20), tl.where(sw, v20, v21); v20 = x0; v21 = x1
        u00 = h00 * v00 + h01 * v10 + h02 * v20; u10 = h10 * v00 + h11 * v10 + h12 * v20; u20 = h20 * v00 + h21 * v10 + h22 * v20
        nrm = tl.sqrt(u00 * u00 + u10 * u10 + u20 * u20)
        ok0 = nrm > 1e-200
        u00 = tl.where(ok0, u00 / nrm, 1.0); u10 = tl.where(ok0, u10 / nrm, 0.0); u20 = tl.where(ok0, u20 / nrm, 0.0)
        u01 = h00 * v01 + h01 * v11 + h02 * v21; u11 = h10 * v01 + h11 * v11 + h12 * v21; u21 = h20 * v01 + h21 * v11 + h22 * v21
        dt = u00 * u01 + u10 * u11 + u20 * u21
        u01 = u01 - dt * u00; u11 = u11 - dt * u10; u21 = u21 - dt * u20
        nrm = tl.sqrt(u01 * u01 + u11 * u11 + u21 * u21)
        ok1 = nrm > 1e-200
        f01 = -u10; f11 = u00; f21 = u00 * 0.0
        fn = tl.sqrt(f01 * f01 + f11 * f11)
        okf = fn > 1e-100
        f01 = tl.where(okf, f01 / tl.where(okf, fn, 1.0), 0.0); f11 = tl.where(okf, f11 / tl.where(okf, fn, 1.0), 0.0); f21 = tl.where(okf, f21, 1.0)
        u01 = tl.where(ok1, u01 / tl.where(ok1, nrm, 1.0), f01); u11 = tl.where(ok1, u11 / tl.where(ok1, nrm, 1.0), f11); u21 = tl.where(ok1, u21 / tl.where(ok1, nrm, 1.0), f21)
        c02 = u10 * u21 - u20 * u11; c12 = u20 * u01 - u00 * u21; c22 = u00 * u11 - u10 * u01
        w02 = h00 * v02 + h01 * v12 + h02 * v22; w12 = h10 * v02 + h11 * v12 + h12 * v22; w22 = h20 * v02 + h21 * v12 + h22 * v22
        sg = tl.where(c02 * w02 + c12 * w12 + c22 * w22 < 0, -1.0, 1.0)
        u02 = sg * c02; u12 = sg * c12; u22 = sg * c22
        detU = u00 * (u11 * u22 - u12 * u21) - u01 * (u10 * u22 - u12 * u20) + u02 * (u10 * u21 - u11 * u20)
        detV = v00 * (v11 * v22 - v12 * v21) - v01 * (v10 * v22 - v12 * v20) + v02 * (v10 * v21 - v11 * v20)
        det = detU * detV
        u02 = u02 * det; u12 = u12 * det; u22 = u22 * det
        r00 = u00 * v00 + u01 * v01 + u02 * v02; r01 = u00 * v10 + u01 * v11 + u02 * v12; r02 = u00 * v20 + u01 * v21 + u02 * v22
        r10 = u10 * v00 + u11 * v01 + u12 * v02; r11 = u10 * v10 + u11 * v11 + u12 * v12; r12 = u10 * v20 + u11 * v21 + u12 * v22
        r20 = u20 * v00 + u21 * v01 + u22 * v02; r21 = u20 * v10 + u21 * v11 + u22 * v12; r22 = u20 * v20 + u21 * v21 + u22 * v22
        ob = Rp + b * 9
        tl.store(ob + 0, r00.to(tl.float32)); tl.store(ob + 1, r01.to(tl.float32)); tl.store(ob + 2, r02.to(tl.float32))
        tl.store(ob + 3, r10.to(tl.float32)); tl.store(ob + 4, r11.to(tl.float32)); tl.store(ob + 5, r12.to(tl.float32))
        tl.store(ob + 6, r20.to(tl.float32)); tl.store(ob + 7, r21.to(tl.float32)); tl.store(ob + 8, r22.to(tl.float32))
        tl.store(DETp + b, det.to(tl.float32))
        tl.store(SMINp + b, tl.sqrt(tl.maximum(l2, 0.0)).to(tl.float32))


def kabsch_rotation_device(cov32, R_out, det_out, smin_out):
    """cov32 [B,3,3] fp32 -> R_out [B,3,3] (= stock's U diag(1,1,det(U Vh)) Vh), det_out [B], smin_out [B]; one sync-free kernel launch."""
    _kabsch3_kernel[(int(cov32.shape[0]),)](cov32, R_out, det_out, smin_out, NSWEEP=8, num_warps=1)


STATS = {
    "installed": {},            # lever -> word
    "calls": 0, "samples": 0, "replays": 0, "captures": 0, "capture_s": [], "pool_gib": [],
    "scope": {},                # reason -> count (calls served by the inner sampler, by name)
    "token_gated": 0,           # of which: calls above the memory row's token ceiling (scope word above_max_tokens; installed["max_tokens"] the ceiling)
    "capture_failed": 0, "capture_error": None,
    "div_recipe": None,         # 'double' | 'float': how ATen rounds the reciprocal of a CPU-scalar divisor on this build (probe_scalar_div)
    "guard_prints": 0,
    "predraw": None, "predraw_classes": {}, "predraw_offsets": {},   # predraw=batched: per (m, S, dtype) class, batched rotation statements bit-compared to the per-step ones
    "kabsch": "torch",
    "sampler_s": [],            # per sample() wall (one cuda synchronise at return; the sampler ends in host syncs anyway)
    "last": {},
}
_DEBUG = os.environ.get("BOLTZ_SAMPLER_DEBUG", "0") == "1"
_CFG = {"rollout": None, "kabsch": "torch", "dit": None, "max_tokens": 0}   # max_tokens: the memory row's token ceiling (BOLTZ_SAMPLER_MAX_TOKENS; 0 = none): a sample() of an input above it
                                                                            # is served by the inner (stock eager) loop BY NAME — scope word above_max_tokens — with the fused step prepared
_AC = {"mod": None}              # the aligncap module, opt/forward/waste (BOLTZ_SAMPLER_ALIGN=aligncap): the bitwise cusolverDnSgesvd seam into static buffers
_DIT = {"mod": None}            # bz_sampler_dit (the fused token-transformer step), when its words are installed


def _log(*a):
    if _DEBUG:
        print("[bz_sampler]", *a, file=sys.stderr, flush=True)


class _Timeline:
    """debug-only (BOLTZ_SAMPLER_DEBUG=1) host/GPU timeline of one sample(): host wall per phase (perf_counter after a device synchronise) and
    the GPU span of every graph replay (cuda events on the current stream). Touches no data; off => every method is a plain pass-through."""

    def __init__(self, on):
        self.on = bool(on); self.marks = {}; self.ev = []; self.host = {"replay_call": 0.0, "eager": 0.0, "n": 0}

    def mark(self, name):
        if self.on:
            torch.cuda.synchronize(); self.marks[name] = time.perf_counter()

    def replay(self, g):
        if not self.on:
            g.replay(); return
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter(); e0.record(); g.replay(); e1.record(); self.host["replay_call"] += time.perf_counter() - t0
        self.ev.append((e0, e1)); self.host["n"] += 1

    def eager(self, dt):
        if self.on:
            self.host["eager"] += dt

    def report(self, S, n_atoms, extra=""):
        if not self.on:
            return
        torch.cuda.synchronize()
        gpu_replays = sum(a.elapsed_time(b) for a, b in self.ev) / 1000.0
        m = self.marks; keys = list(m)
        phases = {f"{a}->{b}": round(m[b] - m[a], 4) for a, b in zip(keys, keys[1:])}
        loop = m.get("loop_end", 0.0) - m.get("loop_start", 0.0)
        n = max(self.host["n"], 1)
        _log(f"timeline atoms={n_atoms} S={S} phases_s={phases} | loop_wall={loop:.4f} replays={self.host['n']} gpu_in_replays={gpu_replays:.4f} "
             f"non_replay_wall={loop - gpu_replays:.4f} (={(loop - gpu_replays) / n * 1e3:.3f} ms/step) host_eager_section={self.host['eager']:.4f} "
             f"(={self.host['eager'] / n * 1e3:.3f} ms/step) host_replay_calls={self.host['replay_call']:.4f} {extra}")


# ----------------------------------------------------------------------------------------------------------------------------------------
# ATen's `CUDA tensor / Python float` multiplies by a host-computed reciprocal (BinaryDivTrueKernel.cu). Which rounding of the reciprocal
# (computed in double and rounded once, or fp32(1)/fp32(b)) is a property of the torch build: probe it against torch's own kernel.
# ----------------------------------------------------------------------------------------------------------------------------------------
def probe_scalar_div(device, values):
    """'double' | 'float' | None: the recipe r(b) with (x / b) == (x * torch.tensor([r(b)])) bit for bit for EVERY b in `values` on random x."""
    g = torch.Generator(device="cpu"); g.manual_seed(1234)
    x = (torch.randn(4099, generator=g, dtype=torch.float32) * 37.0).to(device)
    ok = {"double": True, "float": True}
    for b in values:
        ref = x / b
        if ok["double"] and not torch.equal(ref, x * torch.tensor([np.float32(1.0 / float(b))], dtype=torch.float32, device=device)):
            ok["double"] = False
        if ok["float"] and not torch.equal(ref, x * torch.tensor([np.float32(1.0) / np.float32(b)], dtype=torch.float32, device=device)):
            ok["float"] = False
        if not (ok["double"] or ok["float"]):
            return None
    return "double" if ok["double"] else "float"


def _recip32(b, recipe):
    return np.float32(1.0 / float(b)) if recipe == "double" else np.float32(1.0) / np.float32(b)


def _schedule_t_hats():
    """t_hat values of the stock Boltz-2 inference schedule (200 steps, main.py Boltz2DiffusionParams) folded as the loop folds them."""
    S = 200; sigma_min, sigma_max, sigma_data, rho, gamma_0, gamma_min = 0.0001, 160.0, 16.0, 7, 0.8, 1.0
    inv_rho = 1 / rho
    steps = torch.arange(S, dtype=torch.float32)
    sigmas = (sigma_max ** inv_rho + steps / (S - 1) * (sigma_min ** inv_rho - sigma_max ** inv_rho)) ** rho
    sigmas = torch.nn.functional.pad(sigmas * sigma_data, (0, 1), value=0.0)
    gammas = torch.where(sigmas > gamma_min, gamma_0, 0.0)
    sig, gam = sigmas.tolist(), gammas.tolist()
    return [sig[k] * (1 + gam[k + 1]) for k in range(S)]


# ----------------------------------------------------------------------------------------------------------------------------------------
# schedule + pre-drawn randomness (stock order)
# ----------------------------------------------------------------------------------------------------------------------------------------
def _host_schedule(diff, num_sampling_steps):
    sigmas_t = diff.sample_schedule(num_sampling_steps)
    gammas_t = torch.where(sigmas_t > diff.gamma_min, diff.gamma_0, 0.0)
    return sigmas_t, sigmas_t.tolist(), gammas_t.tolist()


def _rotations_from_quaternion_draws(o):
    """utils.random_quaternions' body after its draw + utils.quaternion_to_matrix, for a [n, 4] block of draws (the stock statements)."""
    from boltz.model.modules.utils import quaternion_to_matrix, _copysign
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return quaternion_to_matrix(o)


def _predraw(shape, m, S, device, dtype):
    """All random tensors of AtomDiffusion.sample in the stock draw order and shapes — init randn(shape); per step: random_quaternions'
    randn((m,4)), compute_random_augmentation's randn((m,1,3)), the noise randn(shape) — so the Philox stream (and every later draw of the
    process) is untouched. The deterministic quaternion -> rotation statements consume no randomness; they run AFTER the draws: per step
    (the stock shapes, [m,4] per call) or — `predraw=batched`, the default — ONCE over all steps ([S*m,4]): the same ATen functions at another
    batch size. The two are bit-compared on the live shape class (m, S, dtype) at its first encounter in the process (outside any capture);
    a class that does not compare identical stays on the per-step statements, by name (STATS["predraw"]). Returns (init, R [S,m,3,3],
    tr [S,m,1,3], noise [S,*shape])."""
    gen = torch.cuda.default_generators[device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else None
    off0 = gen.get_offset() if gen is not None else None
    init = torch.randn(shape, device=device)
    Q = torch.empty((S, m, 4), device=device, dtype=dtype)
    tr = torch.empty((S, m, 1, 3), device=device, dtype=dtype)
    noise = torch.empty((S,) + tuple(shape), device=device, dtype=dtype)
    for k in range(S):
        Q[k].copy_(torch.randn((m, 4), dtype=dtype, device=device))               # utils.random_quaternions' draw
        tr[k].copy_(torch.randn((m, 1, 3), dtype=dtype, device=device))           # utils.compute_random_augmentation's draw
        noise[k].copy_(torch.randn(shape, device=device))                         # stock: eps = sqrt(noise_var) * torch.randn(shape)
    if gen is not None:                                                           # the draws above ARE the stock calls; this pins their count/shapes against future edits
        key_off = f"m{m}_S{S}_{dtype}_shape{'x'.join(map(str, shape))}"
        want = STATS["predraw_offsets"].setdefault(key_off, gen.get_offset() - off0)
        if gen.get_offset() - off0 != want:
            raise RuntimeError(f"[bz_sampler] predraw: generator offset advanced by {gen.get_offset() - off0}, expected {want} for {key_off}")
    tr = tr * 1.0                                                                 # compute_random_augmentation: randn * s_trans (s_trans = 1.0), elementwise
    key = f"m{m}_S{S}_{dtype}"
    mode = _CFG.get("predraw", "batched")
    proven = STATS["predraw_classes"].get(key)
    if mode == "batched" and proven is True:
        R = _rotations_from_quaternion_draws(Q.reshape(S * m, 4)).reshape(S, m, 3, 3)
        STATS["predraw"] = "batched"
        return init, R, tr, noise
    R = torch.stack([_rotations_from_quaternion_draws(Q[k]) for k in range(S)])   # the stock shapes, step by step
    if mode == "batched" and proven is None:                                      # first encounter of this class: bit-compare the batched statements against the per-step ones
        Rb = _rotations_from_quaternion_draws(Q.reshape(S * m, 4)).reshape(S, m, 3, 3)
        same = bool(torch.equal(R, Rb))
        STATS["predraw_classes"][key] = same
        STATS["predraw"] = "batched" if same else f"looped:class_{m}x{S}_not_identical"
        _log(f"predraw class {key}: batched == per-step bitwise: {same}")
    elif mode != "batched":
        STATS["predraw"] = f"looped:word_{mode}"
    else:
        STATS["predraw"] = f"looped:class_{m}x{S}_not_identical"
    return init, R, tr, noise


def _step_scalars(diff, sig, gam, S, step_scale, recipe):
    """Per-step scalars exactly as the stock loop folds them (Python doubles) + their fp32 device-table values."""
    t_hat = np.zeros(S, np.float32); sq = np.zeros(S, np.float32); rc = np.zeros(S, np.float32); cf = np.zeros(S, np.float32)
    host = []
    for k in range(S):
        sigma_tm, sigma_t, gamma = sig[k], sig[k + 1], gam[k + 1]
        th = sigma_tm * (1 + gamma)
        noise_var = diff.noise_scale ** 2 * (th ** 2 - sigma_tm ** 2)
        c = step_scale * (sigma_t - th)
        host.append((th, sqrt(noise_var), c, sigma_t))
        t_hat[k] = np.float32(th); sq[k] = np.float32(sqrt(noise_var)); rc[k] = _recip32(th, recipe); cf[k] = np.float32(c)
    return host, t_hat, sq, rc, cf


# ----------------------------------------------------------------------------------------------------------------------------------------
# weighted_rigid_align (boltz/model/loss/diffusionv2.py) in three pieces: head (to the fp32 covariance), the eager SVD/det section, tail.
# Every statement is the stock statement; the two guard prints are lifted out (module doc).
# ----------------------------------------------------------------------------------------------------------------------------------------
_GUARD1 = "Warning: The size of one of the point clouds is <= dim+1. " + "`WeightedRigidAlign` cannot return a unique rotation."
_GUARD2 = ("Warning: Excessively low rank of " + "cross-correlation between aligned point clouds. "
           + "`WeightedRigidAlign` cannot return a unique rotation.")


def _align_head(true_coords, pred_coords, weights, mask):
    from einops import einsum
    weights = (mask * weights).unsqueeze(-1)
    true_centroid = (true_coords * weights).sum(dim=-2, keepdim=True) / weights.sum(dim=-2, keepdim=True)
    pred_centroid = (pred_coords * weights).sum(dim=-2, keepdim=True) / weights.sum(dim=-2, keepdim=True)
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid
    cov_matrix = einsum(weights * pred_coords_centered, true_coords_centered, "... n i, ... n j -> ... i j")
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    return cov_matrix_32, true_coords_centered, pred_centroid


def _align_svd(cov_matrix_32, batch_size):
    """The eager section (host-synchronising cuSOLVER): U, S, V, F exactly as stock forms them."""
    U, S, V = torch.linalg.svd(cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None)
    V = V.mH
    rot_matrix = torch.einsum("... i j, ... k j -> ... i k", U, V).to(dtype=torch.float32)
    F = torch.eye(3, dtype=cov_matrix_32.dtype, device=cov_matrix_32.device)[None].repeat(*batch_size, 1, 1)
    F[..., -1, -1] = torch.det(rot_matrix)
    return U, S, V, F


def _align_tail(U, F, V, true_coords_centered, pred_centroid, original_dtype):
    from einops import einsum
    rot_matrix = einsum(U, F, V, "... i j, ... j k, ... l k -> ... i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)
    aligned_coords = einsum(true_coords_centered, rot_matrix, "... n i, ... j i -> ... n j") + pred_centroid
    aligned_coords.detach_()
    return aligned_coords


# ----------------------------------------------------------------------------------------------------------------------------------------
# the roll-out
# ----------------------------------------------------------------------------------------------------------------------------------------
class _Roll:
    """State of one sample() call: device tables, the statics the graph reads/writes (allocated layout-preserving from the eager step-0
    tensors, so every captured kernel sees the stock layouts), the captured boundary graph."""

    def __init__(self, S, device):
        f32 = dict(device=device, dtype=torch.float32)
        self.ctr = torch.zeros(1, device=device, dtype=torch.long)
        self.t_hat_tab = torch.zeros(S, **f32); self.sqrt_nv_tab = torch.zeros(S, **f32)
        self.recip_tab = torch.zeros(S, **f32); self.coef_tab = torch.zeros(S, **f32)
        self.zadd_tab = torch.zeros(S, **f32)                     # the guidance block's `denoised += (+0.0)` at steps < S-1; -0.0 (the additive identity for every float) at S-1
        self.zadd_tab[S - 1] = -0.0
        self.cov = self.true_c = self.pcen = self.den = None      # written by the head of step k, read by the tail of step k
        self.U = self.F = self.V = None                           # written by the eager SVD section, read by the tail inside the graph (kabsch=torch)
        self.R = self.det_tab = self.smin_tab = None              # kabsch=device: the rotation written IN the graph; det / smallest singular value per step for the deferred guard
        self.Sv = self.Vh = self.S_tab = None                     # kabsch=aligncap: gesvd's S / Vh statics (torch strides) + the per-step S table
        self.sigma = {}                                           # chunk size -> static (n,) fp32 sigma (== torch.full((n,), t_hat))


def _guidance_active(steering_args, feats):
    """Stock CLI defaults carry contact_guidance_update=True: lines 443-473 run every step with the two Boltz-2 potentials of
    get_potentials(boltz2=True) — ContactPotentital and TemplateReferencePotential. Each returns torch.zeros_like(coords) from compute_gradient
    (potentials.py:95-96: `if index.shape[1] == 0`) unless the input carries contact constraints (feats['contact_pair_index'] non-empty) or
    forced templates (feats['template_mask_cb'][feats['template_force']] non-empty) — the same host-side tests the potentials make. Returns the
    name of the active potential, or None when the whole block provably computes +0.0 (its only effect on a live value is then
    `atom_coords_denoised += (+0.0)`, which the roll-out keeps as one kernel: it normalises the sign of an exact zero exactly as stock does)."""
    if not steering_args.get("contact_guidance_update"):
        return None
    idx = feats.get("contact_pair_index")
    if idx is not None and int(idx[0].shape[-1]) != 0:
        return "contact_constraints"
    if "template_mask_cb" in feats and "template_force" in feats:
        template_mask = feats["template_mask_cb"][feats["template_force"]]          # one host sync per sample() (boolean index), as the potential itself does per call
        if int(template_mask.shape[0]) != 0:
            return "template_force"
    return None


def _in_scope(diff, steering_args, device, feats=None):
    if steering_args is None:
        return "steering_args_none"
    if steering_args.get("fk_steering") or steering_args.get("physical_guidance_update"):
        return "steering"                       # --use_potentials / fk steering: per-step host-scheduled potentials (the kit drops the sampler levers for that run by name)
    if feats is not None:
        g = _guidance_active(steering_args, feats)
        if g is not None:
            return "guidance_" + g              # contact constraints / forced templates: the guidance block does real per-step work with a host schedule -> stock sampler
    if device.type != "cuda":
        return "device_" + device.type
    if not diff.alignment_reverse_diff:
        return "alignment_reverse_diff_off"
    if diff.training:
        return "training"
    return None


def sample_rollout(self, atom_mask, num_sampling_steps=None, multiplicity=1, max_parallel_samples=None, steering_args=None, **network_condition_kwargs):
    """AtomDiffusion.sample as the boundary-graph roll-out (module doc). Same signature, same return dict, same RNG consumption."""
    STATS["calls"] += 1
    inner = self.__class__._bzs_inner_sample
    device = atom_mask.device if torch.is_tensor(atom_mask) else self.device
    why = _in_scope(self, steering_args, device, network_condition_kwargs.get("feats"))
    if why is None and not _CFG["rollout"]:
        why = "rollout_off"
    feats_ = network_condition_kwargs.get("feats") or {}
    n_tok = int(feats_["token_pad_mask"].shape[1]) if torch.is_tensor(feats_.get("token_pad_mask")) else None
    if why is None and _CFG["max_tokens"] and n_tok is not None and n_tok > _CFG["max_tokens"]:
        why = "above_max_tokens"                                         # the memory row's token ceiling: this prediction samples on the stock eager loop (the fused step prepared below serves it)
        STATS["token_gated"] += 1
    dit = _DIT["mod"]
    if dit is not None and "feats" in network_condition_kwargs:      # the fused token step (fast tier): patch the instance once, pack this call's pair bias
        dit.attach(self.score_model)
        dit.prepare(self.score_model, network_condition_kwargs["feats"], network_condition_kwargs["diffusion_conditioning"], multiplicity)
    try:
        if why is not None:
            if why != "rollout_off":
                STATS["scope"][why] = STATS["scope"].get(why, 0) + 1
            STATS["last"] = {"sampler_mode": "inner:" + why, "n_tokens": n_tok, "max_tokens": _CFG["max_tokens"] or None, "token_gated": why == "above_max_tokens"}
            _log("inner sampler serves this call:", why)
            return inner(self, atom_mask, num_sampling_steps=num_sampling_steps, multiplicity=multiplicity, max_parallel_samples=max_parallel_samples,
                         steering_args=steering_args, **network_condition_kwargs)
        _maybe_dump(self, atom_mask, num_sampling_steps, multiplicity, max_parallel_samples, steering_args, network_condition_kwargs)
        t0 = time.time()
        out = _rollout(self, atom_mask, num_sampling_steps, multiplicity, max_parallel_samples, steering_args, network_condition_kwargs)
        torch.cuda.synchronize()
        STATS["sampler_s"].append(round(time.time() - t0, 4)); STATS["samples"] += 1
        STATS["last"]["sampler_s"] = STATS["sampler_s"][-1]
        _log("sample():", STATS["last"])
        return out
    finally:
        if dit is not None:
            dit.release()


sample_rollout._bzs_rollout = True


def _rollout(self, atom_mask, num_sampling_steps, multiplicity, max_parallel_samples, steering_args, nck):
    from boltz.model.modules.utils import default
    guided = bool(steering_args.get("physical_guidance_update") or steering_args.get("contact_guidance_update"))   # stock 443-446: the guidance block runs at steps < S-1;
    # here it provably computes +0.0 (_guidance_active checked the input) -> its one live statement `atom_coords_denoised += guidance_update` is kept as
    # `den += z[k]` with z[k] = +0.0 for k < S-1 (== adding stock's all-(+0.0) update: -0.0 -> +0.0 like stock) and z[S-1] = -0.0 (x + (-0.0) == x for every x: the
    # block does not run at the last step). The dead temporaries (energy_gradient, guidance_update, scaled_guidance_update and its rotation at 364-370 — read only
    # under fk_steering) are not materialised.
    if max_parallel_samples is None:
        max_parallel_samples = multiplicity
    num_sampling_steps = default(num_sampling_steps, self.num_sampling_steps)
    atom_mask = atom_mask.repeat_interleave(multiplicity, 0)
    shape = (*atom_mask.shape, 3)
    Bm, A = int(shape[0]), int(shape[1])
    S = int(num_sampling_steps)
    device = atom_mask.device
    TL = _Timeline(_DEBUG); TL.mark("enter")
    sigmas_t, sig, gam = _host_schedule(self, S)
    step_scale = self.step_scale
    host, t_hat_np, sq_np, rc_np, cf_np = _step_scalars(self, sig, gam, S, step_scale, STATS["div_recipe"])
    TL.mark("sched")
    # ---- all randomness, stock order ----
    init, R_tab, tr_tab, noise_tab = _predraw(shape, multiplicity, S, device, torch.float32)
    TL.mark("predrawn")
    init_sigma = sigmas_t[0]
    atom_coords = init_sigma * init                                              # stock 344-345: init_sigma (0-dim device tensor) * torch.randn(shape)
    roll = _Roll(S, device)
    for tab, arr in ((roll.t_hat_tab, t_hat_np), (roll.sqrt_nv_tab, sq_np), (roll.recip_tab, rc_np), (roll.coef_tab, cf_np)):
        tab.copy_(torch.from_numpy(arr).to(device))
    sample_ids = torch.arange(multiplicity).to(device)
    chunks = list(sample_ids.chunk(multiplicity % max_parallel_samples + 1))
    for c in chunks:
        roll.sigma[int(c.numel())] = torch.zeros((int(c.numel()),), device=device, dtype=torch.float32)
    guard1 = bool(torch.any(atom_mask.float().sum(dim=-1) < (3 + 1)))          # weighted_rigid_align guard 1: the mask is step-invariant
    num_points = A

    def network(atom_coords_noisy, sigma):
        """stock 380-396: zeros_like, chunked preconditioned forward, index_put. `sigma`: the Python float (eager step 0: stock's torch.full
        inside preconditioned_network_forward) or the (1,) tensor read from the table in-graph (copied into the static (n,) sigma)."""
        with torch.no_grad():
            atom_coords_denoised = torch.zeros_like(atom_coords_noisy)
            for ids in chunks:
                n = int(ids.numel())
                if torch.is_tensor(sigma):
                    sg = roll.sigma[n]; sg.copy_(sigma.expand(n))                # == torch.full((batch,), t_hat, device=device)
                else:
                    sg = sigma
                chunk = self.preconditioned_network_forward(atom_coords_noisy[ids], sg, network_condition_kwargs=dict(multiplicity=n, **nck))
                atom_coords_denoised[ids] = chunk
        return atom_coords_denoised

    def align_head(atom_coords_noisy, atom_coords_denoised, first):
        """stock 512-519: weighted_rigid_align(noisy.float(), denoised.float(), mask.float(), mask.float()) up to the fp32 covariance -> statics."""
        with torch.autocast("cuda", enabled=False):
            cov32, true_c, pcen = _align_head(atom_coords_noisy.float(), atom_coords_denoised.float(), atom_mask.float(), atom_mask.float())
        if first:
            roll.cov = cov32.clone(); roll.true_c = true_c.clone(); roll.pcen = pcen.clone(); roll.den = atom_coords_denoised.clone()
        else:
            roll.cov.copy_(cov32); roll.true_c.copy_(true_c); roll.pcen.copy_(pcen); roll.den.copy_(atom_coords_denoised)

    # ---- step 0, eager (stock statements; also the capture's warm-up: the hoist's caches are (re)built here) ----
    random_R, random_tr = R_tab[0], tr_tab[0]
    atom_coords = atom_coords - atom_coords.mean(dim=-2, keepdims=True)
    atom_coords = torch.einsum("bmd,bds->bms", atom_coords, random_R) + random_tr
    th0, sq0, c0, sigma_t0 = host[0]
    eps = sq0 * noise_tab[0]                                                     # == sqrt(noise_var) * torch.randn(shape)
    atom_coords_noisy = atom_coords + eps
    TL.mark("step0_pre_net")
    atom_coords_denoised = network(atom_coords_noisy, th0)
    TL.mark("step0_net")
    if guided and 0 < S - 1:
        atom_coords_denoised += 0.0                                              # stock 466: `atom_coords_denoised += guidance_update` (all +0.0 here, _guidance_active)
    align_head(atom_coords_noisy, atom_coords_denoised, first=True)

    kdev = _CFG["kabsch"] == "device"
    kcap = _CFG["kabsch"] == "aligncap"
    ksvd = not (kdev or kcap)                                                    # exact tier, torch's own gesvd: torch.linalg.svd(out=statics) eager between replays; det / F / R inside the graph
    if kdev:
        roll.R = torch.empty(Bm, 3, 3, device=device, dtype=torch.float32)
        roll.det_tab = torch.zeros(S, Bm, device=device, dtype=torch.float32); roll.smin_tab = torch.ones(S, Bm, device=device, dtype=torch.float32)
    if kcap:                                                                     # aligncap's seam: gesvd eager between replays into these statics (torch's own strides); det / F / R inside the graph
        AC = _AC["mod"]; AC.init(Bm, device); AC.reset_flags()
        roll.U = AC.new_U(Bm, device); roll.Sv = AC.new_S(Bm, device); roll.Vh = AC.new_Vh(Bm, device)
        roll.S_tab = torch.ones(S, Bm, 3, device=device, dtype=torch.float32)   # singular values per step for the deferred guard 2 (written in the graph)

    def tail_aligned():
        """the alignment tail: kabsch=torch reads the eager SVD section's U/F/V statics (stock statements); kabsch=device computes R in ONE
        sync-free kernel from the covariance static (fast tier) and applies stock's last two statements with rot_matrix = R."""
        from einops import einsum
        with torch.autocast("cuda", enabled=False):
            if kcap or ksvd:                                                     # stock loss/diffusionv2.py after the SVD: V = V.mH; rot = einsum(U, V); F = eye.repeat; F[..., -1, -1] = det(rot)
                roll.S_tab.index_copy_(0, roll.ctr, roll.Sv[None])
                V = roll.Vh.mH
                rot_matrix = torch.einsum("... i j, ... k j -> ... i k", roll.U, V).to(dtype=torch.float32)
                F = torch.eye(3, dtype=torch.float32, device=device)[None].repeat(Bm, 1, 1)
                F[..., -1, -1] = torch.det(rot_matrix)
                return _align_tail(roll.U, F, V, roll.true_c, roll.pcen, torch.float32)
            kabsch_rotation_device(roll.cov, roll.R, roll.det_tab[0], roll.smin_tab[0])          # det/smin of this step land in row ctr via the index_copy below
            roll.det_tab.index_copy_(0, roll.ctr, roll.det_tab[0:1].clone()); roll.smin_tab.index_copy_(0, roll.ctr, roll.smin_tab[0:1].clone())
            aligned = einsum(roll.true_c, roll.R, "... n i, ... j i -> ... n j") + roll.pcen
            return aligned.detach_()

    def graph_body():
        """[tail of step ctr] + [head of step ctr+1]; every per-step value from the device tables."""
        aligned = tail_aligned()
        atom_coords_noisy_ = aligned.to(roll.den)
        recip = roll.recip_tab.index_select(0, roll.ctr); coef = roll.coef_tab.index_select(0, roll.ctr)
        denoised_over_sigma = (atom_coords_noisy_ - roll.den) * recip           # == (noisy - denoised) / t_hat   (probe_scalar_div)
        atom_coords_next = atom_coords_noisy_ + denoised_over_sigma * coef       # == noisy + step_scale*(sigma_t - t_hat) * dos
        x = atom_coords_next
        roll.ctr.add_(1)                                                         # ---- head of step ctr+1 ----
        R = R_tab.index_select(0, roll.ctr)[0]; tr = tr_tab.index_select(0, roll.ctr)[0]
        x = x - x.mean(dim=-2, keepdims=True)
        x = torch.einsum("bmd,bds->bms", x, R) + tr
        den_prev = roll.den                                                      # stock 358-363 on the previous denoised (a dead value in stock too)
        den_prev -= den_prev.mean(dim=-2, keepdims=True)
        den_prev = torch.einsum("bmd,bds->bms", den_prev, R) + tr
        eps_ = roll.sqrt_nv_tab.index_select(0, roll.ctr) * noise_tab.index_select(0, roll.ctr)[0]
        noisy = x + eps_
        den = network(noisy, roll.t_hat_tab.index_select(0, roll.ctr))
        if guided:
            den += roll.zadd_tab.index_select(0, roll.ctr)                       # stock 466 at steps < S-1 (+0.0); the additive identity -0.0 at step S-1 (block not run there)
        align_head(noisy, den, first=False)

    use_graph = _CFG["rollout"] == "graph"
    g = None
    if kdev:                                                                     # ---- fast tier: no host sync in the loop; S-1 back-to-back replays ----
        if not STATS.get("kabsch_warm"):                                         # compile the Triton kernel OUTSIDE the capture, once per process (step 0 has no tail, so the
            _R = torch.empty_like(roll.R); _d = torch.empty(Bm, device=device); _m = torch.empty(Bm, device=device)   # graph body would be its first call): scratch outputs, discarded
            kabsch_rotation_device(roll.cov, _R, _d, _m); STATS["kabsch_warm"] = True
        roll.ctr.zero_()
        if guard1:
            for _ in range(S):
                print(_GUARD1); STATS["guard_prints"] += 1
        TL.mark("step0_done")
        if use_graph:
            g = _capture(self, graph_body)
        TL.mark("captured"); TL.mark("loop_start")
        for k in range(S - 1):
            if g is not None:
                TL.replay(g)
            else:
                graph_body()
        TL.mark("loop_end")
    elif kcap:                                                                   # ---- exact tier, aligncap seam: ONE eager cusolver call between replays, nothing else ----
        roll.ctr.zero_()
        if guard1:
            for _ in range(S):
                print(_GUARD1); STATS["guard_prints"] += 1
        TL.mark("step0_done")
        for k in range(S - 1):
            _t0 = time.perf_counter()
            AC.aligncap(roll.cov, roll.U, roll.Sv, roll.Vh)
            TL.eager(time.perf_counter() - _t0)
            if k == 0 and use_graph:
                g = _capture(self, graph_body)
                TL.mark("captured"); TL.mark("loop_start")
            if g is not None:
                TL.replay(g)
            else:
                graph_body()
        TL.mark("loop_end")
    else:                                                                        # ---- exact tier, torch's gesvd: ONE eager torch.linalg.svd(out=statics) between replays, nothing else ----
        TL.mark("step0_done")
        for k in range(S - 1):
            _t0 = time.perf_counter()
            with torch.autocast("cuda", enabled=False):
                if k == 0:                                                       # statics = layout-preserving clones of torch's own first outputs (U: cuSOLVER's layout; the graph reads them)
                    U0, S0, Vh0 = torch.linalg.svd(roll.cov, driver="gesvd")
                    roll.U, roll.Sv, roll.Vh = U0.clone(), S0.clone(), Vh0.clone()
                    roll.S_tab = torch.ones(S, Bm, 3, device=device, dtype=torch.float32)
                    roll.ctr.zero_()
                else:
                    torch.linalg.svd(roll.cov, driver="gesvd", out=(roll.U, roll.Sv, roll.Vh))
            TL.eager(time.perf_counter() - _t0)
            if guard1:
                print(_GUARD1); STATS["guard_prints"] += 1
            if k == 0:
                if use_graph:
                    g = _capture(self, graph_body)
                TL.mark("captured"); TL.mark("loop_start")
            if g is not None:
                TL.replay(g)
            else:
                graph_body()
        TL.mark("loop_end")
    if g is not None:
        STATS["replays"] += S - 1
    # ---- last step: eager SVD section (kabsch=torch) / the device kernel (kabsch=device) + eager tail with the stock Python-float expressions ----
    with torch.autocast("cuda", enabled=False):
        if kdev:
            atom_coords_noisy = tail_aligned()                                   # writes det/smin of step S-1 into row ctr (= S-1)
        elif kcap:
            AC.aligncap(roll.cov, roll.U, roll.Sv, roll.Vh)
            atom_coords_noisy = tail_aligned()                                   # writes S of step S-1 into row ctr (= S-1)
            fl = AC.flags()                                                      # ONE host read per sample(): devInfo of the S gesvd calls (torch raises at the step; we raise here, by name)
            if fl.get("gesvd_info_nonzero"):
                raise RuntimeError(f"[bz_sampler] align=aligncap: cusolverDnSgesvd reported devInfo != 0 in this sample() ({fl}) — torch.linalg.svd would have raised")
        else:
            if S - 1 == 0:                                                       # a 1-step schedule: no loop ran, the statics do not exist yet
                U0, S0, Vh0 = torch.linalg.svd(roll.cov, driver="gesvd"); roll.U, roll.Sv, roll.Vh = U0.clone(), S0.clone(), Vh0.clone()
                roll.S_tab = torch.ones(S, Bm, 3, device=device, dtype=torch.float32); roll.ctr.zero_()
            else:
                torch.linalg.svd(roll.cov, driver="gesvd", out=(roll.U, roll.Sv, roll.Vh))
            if guard1:
                print(_GUARD1); STATS["guard_prints"] += 1
            atom_coords_noisy = tail_aligned()                                   # writes S of step S-1 into row ctr (= S-1)
    atom_coords_noisy = atom_coords_noisy.to(roll.den)
    th, sqn, c, sigma_t = host[S - 1]
    denoised_over_sigma = (atom_coords_noisy - roll.den) / th
    atom_coords_next = atom_coords_noisy + step_scale * (sigma_t - th) * denoised_over_sigma
    atom_coords = atom_coords_next
    # ---- weighted_rigid_align guard 2 for every step: one host read after the loop ----
    if not (num_points < (3 + 1)):
        low = ((roll.smin_tab.abs() <= 1e-15).any(dim=1) if kdev else (roll.S_tab.abs() <= 1e-15).flatten(1).any(dim=1)).cpu()
        for flag in low.tolist():
            if flag:
                print(_GUARD2); STATS["guard_prints"] += 1
    n_tok = int(nck["feats"]["token_pad_mask"].shape[1]) if "feats" in nck and "token_pad_mask" in nck["feats"] else None
    STATS["last"] = {"sampler_mode": ("graph" if g is not None else "eager_tables"), "steps": S, "multiplicity": multiplicity, "n_tokens": n_tok,
                     "n_atoms": A, "n_replay": (S - 1 if g is not None else 0), "captures_total": STATS["captures"],
                     "capture_s_last": (STATS["capture_s"] or [None])[-1], "pool_gib_last": (STATS["pool_gib"] or [None])[-1], "kabsch": STATS["kabsch"], "div_recipe": STATS["div_recipe"]}
    if g is not None:                                                            # the graph read this call's live conditioning: drop it (and its pool) at return
        del g
        _release_graph(self)
    TL.mark("exit"); TL.report(S, A, extra=f"kabsch={_CFG['kabsch']} Bm={Bm} tokens={n_tok}")
    return dict(sample_atom_coords=atom_coords, diff_token_repr=None)


def _capture(diff, body):
    """Capture `body` into a CUDA graph on a side stream WITHOUT executing it (stream capture records only), in the graph's own private pool. No warm-up iterations: the eager step 0 just ran every kernel family of the body on this thread. Returns the graph, or None
    after a non-OOM capture failure (named and counted; the caller runs the same statements eagerly)."""
    from opt_core.oom import is_oom
    t0 = time.time()
    g = None
    try:
        # a FRESH private pool per capture: on this torch (2.12) a pool id whose graphs were all destroyed cannot be captured into again
        # (CUDACachingAllocator use_count assert), so the pool is the graph's own and dies with it at sample() return (direct mode)
        torch.cuda.synchronize()
        mem0 = torch.cuda.memory_reserved()
        s = torch.cuda.Stream(); cur = torch.cuda.current_stream(); s.wait_stream(cur)
        g = torch.cuda.CUDAGraph()
        # cuBLAS workspaces are keyed (handle, stream): the capture stream is new, so the first GEMM inside the capture allocates its workspace
        # from the graph's pool (lifetime = the graph's); _release_graph clears the map when the graph is dropped (scratch memory: no numerics).
        with torch.cuda.stream(s):
            g.capture_begin(capture_error_mode="thread_local")
            try:
                body()
            finally:
                g.capture_end()
        cur.wait_stream(s)
        torch.cuda.synchronize()
        STATS["captures"] += 1; STATS["capture_s"].append(round(time.time() - t0, 3))
        STATS["pool_gib"].append(round((torch.cuda.memory_reserved() - mem0) / 2 ** 30, 3))
        diff.__dict__["_bzs_graph_live"] = True
        _log(f"captured the boundary graph in {STATS['capture_s'][-1]} s, reserved +{STATS['pool_gib'][-1]} GiB")
        return g
    except Exception as e:  # noqa: BLE001
        if is_oom(e):
            raise
        STATS["capture_failed"] += 1; STATS["capture_error"] = traceback.format_exc()[-3000:]
        print("[bz_sampler] CAPTURE FAILED (capture_failed) -> the roll-out runs its statements eagerly for this call:\n" + STATS["capture_error"],
              file=sys.stderr, flush=True)
        del g
        torch.cuda.synchronize()
        return None


def _release_graph(diff):
    """After a graph is dropped: synchronise and clear cuBLAS's (handle, stream) workspace map so no entry points into the pool the next
    capture reuses (torch recycles stream ids) — the same call the kit's graph patch makes after deleting a step graph."""
    if diff.__dict__.pop("_bzs_graph_live", False):
        torch.cuda.synchronize()
        fn = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
        BGP = sys.modules.get("boltz_graph_patch")
        orig = (getattr(BGP, "_APPLIED", {}) or {}).get("clear_ws_orig") if BGP is not None else None
        f = orig or fn
        if f is not None:
            f()


# ----------------------------------------------------------------------------------------------------------------------------------------
# dev hook: dump the sampler's inputs (never set by a mode row)
# ----------------------------------------------------------------------------------------------------------------------------------------
DEV_DUMP_DIR = None             # dev hook, set in-process by a caller (nothing in this tree sets it); never by an environment word or a mode row


def _maybe_dump(diff, atom_mask, num_sampling_steps, multiplicity, max_parallel_samples, steering_args, nck):
    d = DEV_DUMP_DIR
    if not d:
        return
    import functools
    os.makedirs(d, exist_ok=True)
    n_tok = int(nck["feats"]["token_pad_mask"].shape[1])
    path = os.path.join(d, f"dump_n{n_tok:05d}_{STATS['calls']}.pt")
    dc = {}
    for key, v in nck["diffusion_conditioning"].items():
        if torch.is_tensor(v):
            dc[key] = v.detach().cpu()
        elif isinstance(v, functools.partial):
            dc[key] = {"__partial__": v.func.__name__, "keywords": {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.keywords.items()}}
        else:
            dc[key] = v
    feats = {kk: vv.detach().cpu() for kk, vv in nck["feats"].items() if torch.is_tensor(vv) and vv.numel() * vv.element_size() < (1 << 31) and "msa" not in kk}
    torch.save({"atom_mask": atom_mask.detach().cpu(), "num_sampling_steps": num_sampling_steps, "multiplicity": multiplicity,
                "max_parallel_samples": max_parallel_samples, "steering_args": dict(steering_args or {}),
                "s_trunk": nck["s_trunk"].detach().cpu(), "s_inputs": nck["s_inputs"].detach().cpu(), "feats": feats, "diffusion_conditioning": dc,
                "cuda_rng_state": torch.cuda.get_rng_state(atom_mask.device), "n_tokens": n_tok,
                "autocast": torch.is_autocast_enabled(), "matmul_precision": torch.get_float32_matmul_precision()}, path)
    print(f"[bz_sampler] dumped sampler inputs -> {path}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------------
# install / report
# ----------------------------------------------------------------------------------------------------------------------------------------
def _wrap_sample():
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    if not hasattr(AtomDiffusion, "_bzs_inner_sample"):
        AtomDiffusion._bzs_inner_sample = AtomDiffusion.sample
    if not getattr(AtomDiffusion.sample, "_bzs_rollout", False):
        AtomDiffusion.sample = sample_rollout


def install(rollout="graph", kabsch="torch", dit=None, aligncap_module=None, max_tokens=None):
    """Install the sampler levers. rollout: 'graph' | 'eager' (the same statements from the pre-drawn tables without a graph: the falsification
    control) | None / 'off'. dit: list of bz_sampler_dit words (fast tier) or None. max_tokens: the memory row's token ceiling (int > 0; None /
    0 = none): sample() calls of inputs above it go to the inner sampler by name (scope above_max_tokens), the fused step still prepared.
    Idempotent. Returns the installed lever words. Raises RuntimeError BY NAME when a precondition fails (the adapter turns that into a refusal)."""
    try:
        _CFG["max_tokens"] = max(0, int(max_tokens or 0))
    except (TypeError, ValueError):
        raise RuntimeError(f"max_tokens={max_tokens!r}: not a token count")
    STATS["installed"]["max_tokens"] = _CFG["max_tokens"] or None
    if isinstance(rollout, str) and "," in rollout:                       # sub-words of the roll-out word: graph,predraw=loop|batched
        parts = [w.strip().lower() for w in rollout.split(",") if w.strip()]
        rollout = parts[0]
        for w in parts[1:]:
            if w.startswith("predraw=") and w.split("=", 1)[1] in ("loop", "batched"):
                _CFG["predraw"] = w.split("=", 1)[1]
            else:
                raise RuntimeError(f"rollout sub-word {w!r}: not a value (predraw=loop|batched)")

    words = []
    rollout = (rollout or "off").lower()
    if rollout in ("graph", "eager"):
        if not torch.cuda.is_available():
            raise RuntimeError("rollout: no CUDA device")
        dev = torch.device("cuda", torch.cuda.current_device())
        recipe = probe_scalar_div(dev, _schedule_t_hats() + [3.0, 7.0, 0.1, 1e-3, 123.456, 6.928203230275509])
        if recipe is None:
            raise RuntimeError("rollout: neither reciprocal recipe reproduces torch's `tensor / float` on this build (probe_scalar_div) — refused")
        STATS["div_recipe"] = recipe
        _wrap_sample()
        _CFG["rollout"] = rollout
        STATS["installed"]["rollout"] = rollout
        words.append("rollout=" + rollout)
    elif rollout not in ("off", "0", ""):
        raise RuntimeError(f"rollout={rollout!r}: not a value (graph | eager | off)")
    align = (kabsch or "gesvd").lower()                     # word BOLTZ_SAMPLER_ALIGN: aligncap (exact: the aligncap module's bitwise gesvd seam) | jacobi64 (fast tier) | absent = torch's gesvd + det, eager between replays
    if align in ("torch", "gesvd", "off", ""):
        kabsch = "torch"
    elif align in ("jacobi64", "device"):
        kabsch = "device"
    elif align == "aligncap":
        kabsch = "aligncap"
        mod = aligncap_module
        if mod is None:
            raise RuntimeError("align=aligncap: the aligncap module (opt/forward/waste/aligncap.py) was not handed in by the adapter — refused")
        for fn in ("init", "new_U", "new_S", "new_Vh", "aligncap", "flags", "reset_flags"):
            if not callable(getattr(mod, fn, None)):
                raise RuntimeError(f"align=aligncap: aligncap.{fn} missing — refused")
        pin = str(getattr(mod, "PIN_TORCH", ""))
        if pin and not torch.__version__.startswith(pin):
            raise RuntimeError(f"align=aligncap: torch {torch.__version__} is off aligncap's pin {pin} — refused by name")
        if _CFG["rollout"] not in ("graph", "eager"):
            raise RuntimeError("align=aligncap rides the roll-out (BOLTZ_SAMPLER_ROLLOUT=graph|eager must be in the row) — refused")
        _AC["mod"] = mod
        words.append("align=aligncap")
        STATS["installed"]["align_aligncap"] = "aligncap"
    else:
        raise RuntimeError(f"align={align!r}: not a value (aligncap | jacobi64 | gesvd)")
    if kabsch == "device":
        if not _HAVE_TRITON:
            raise RuntimeError("align=jacobi64: triton is not importable in this process")
        if _CFG["rollout"] not in ("graph", "eager"):
            raise RuntimeError("align=jacobi64 rides the roll-out (BOLTZ_SAMPLER_ROLLOUT=graph|eager must be in the row) — refused")
        words.append("align=jacobi64")
        STATS["installed"]["align_jacobi64"] = "jacobi64"
    STATS["kabsch"] = kabsch; _CFG["kabsch"] = kabsch
    if dit:
        here = os.path.dirname(os.path.abspath(__file__))
        import importlib.util
        mod = sys.modules.get("bz_sampler_dit")
        if mod is None:
            spec = importlib.util.spec_from_file_location("bz_sampler_dit", os.path.join(here, "bz_sampler_dit.py"))
            mod = importlib.util.module_from_spec(spec); sys.modules["bz_sampler_dit"] = mod; spec.loader.exec_module(mod)
        words += mod.install(list(dit))
        _DIT["mod"] = mod
        _CFG["dit"] = list(dit); STATS["installed"]["dit_fused"] = ",".join(dit)
        _wrap_sample()
    _log("installed:", words, "div recipe:", STATS["div_recipe"])
    return words


def uninstall():
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    if hasattr(AtomDiffusion, "_bzs_inner_sample"):
        AtomDiffusion.sample = AtomDiffusion._bzs_inner_sample
        delattr(AtomDiffusion, "_bzs_inner_sample")
    _CFG["rollout"] = None
    STATS["installed"].clear()


def report():
    dit = _DIT["mod"]
    return {"stats": {k: v for k, v in STATS.items() if k != "capture_error"}, "capture_error": STATS["capture_error"], "cfg": dict(_CFG),
            "dit": (dit.report() if dit is not None else None)}
