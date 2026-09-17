"""infopt_graphs.protenix.sampler_prep — lever `sampler_prep` (exact class, engineering): the per-item and per-step HOST path of the graphed
diffusion sampler (GraphedDenoiseLoop in graphed.py), with no change to any kernel the model runs, to the RNG draws or their order, or to the
graph itself.  No environment switch — graphed.install(..., sampler_prep=True) builds the loop with
PrepConfig(True) (the kit's arm word `sampler_prep`, levers_ptx1.set_sampler); the census is report().  The parts, each named so the kit's LEVER
line says what is on (always all six; poison=ok|unlisted:…|failed).

  rot_async    the per-step random rotation (scipy on the host, as stock draws it) reaches the device through a pinned host ring with a
               non_blocking copy: same float32 bytes, no host<->device synchronisation per diffusion step.
  keycheck     the graph cache key is names/shapes/dtypes; the VALUES of every non-floating input feature (indices / masks a captured graph may
               bake into its launches) are compared on the device (torch.equal) with the live entry's static copies.  Same admission rule as the
               sha256 digests it replaces (any differing integer feature = a new capture), without copying integer features to the host per item.
  warmup1      one eager pass of step 0 before capture (the second pass recomputed the identical step in bias-cache 'hit' mode).
  poison_once  the bias-cache poison self-test (three extra replays) runs on the first capture of the process, with the pass criterion
               poisoned-vs-normal >= POISON_MIN_RATIO (10) x normal-vs-normal AND >= POISON_MIN_ABS (1.0): the normal-vs-normal replay difference is
               not zero outside the deterministic recipe (index-add atomics), and the record prints it as det0_replay_noise_max.
  pool_chain   when an item needs a new graph and the cache is full: the previous entry's static buffers are released first (the allocator hands the
               same blocks to the new ones); then, if the new signature is no larger than anything the previous graph's memory pool has served
               (N_token, N_atom, samples per chunk) and at least CHAIN_MIN_FRAC (0.8) of its token count, the new graph is captured INTO that pool while the previous graph is still alive (it is never
               replayed again and is dropped right after: one pool, no second set of buffers, no unmap/map cycle, and the pool cannot grow); otherwise
               the previous entry and its pool are dropped and returned to the device BEFORE the new capture allocates (pools never coexist).  The
               loop's own eviction + empty_cache after a capture is unchanged (it finds nothing left to evict).  Either way the DiT hoist's per-shape
               padding-bias cache is trimmed to the shape being captured (padbias_dropped).
  stepvec      the per-step scalars (t_hat, sqrt(t_hat^2 - c^2), dt) of all steps are three vector ops (element i is the same fp32 arithmetic as the
               per-step 0-dim ops).

Numerics: exact by construction — identical kernels in identical order on identical values; outputs are byte-identical to the loop without the
lever under the deterministic recipe.
"""
from __future__ import annotations
import os
import re
import time
from typing import Dict, Optional
import torch

PARTS = ("rot_async", "keycheck", "warmup1", "poison_once", "pool_chain", "stepvec")


class PrepConfig:
    """What the loop's `prep` selects (fixed when graphed.py builds its loop): all PARTS or none.  No environment variable is read; every other
    setting below is a module constant (a caller that wants a subset constructs PrepConfig itself; no mode of the kit selects a partial lever)."""
    def __init__(self, on: bool, parts=PARTS):
        self.on = bool(on)
        self.parts = tuple(p for p in PARTS if p in parts) if self.on else ()
        for p in PARTS:
            setattr(self, p, self.on and p in self.parts)

    def describe(self) -> str:
        if not self.on:
            return "off"
        return "+".join(self.parts) if self.parts else "none"



STATS: Dict[str, int] = {"rot_uploads": 0, "rot_ring_waits": 0, "key_hits": 0, "key_value_miss": 0, "key_shape_miss": 0,
                         "statics_released": 0, "pool_chained": 0, "pool_renewed": 0, "padbias_dropped": 0, "poison_skipped": 0}
POISON_MIN_RATIO = 10.0     # poison_once: the poisoned replay must move x_l by >= 10 x the replay-to-replay noise ...
POISON_MIN_ABS = 1.0        # ... and by >= 1.0 (coordinates, Angstrom) absolutely; a graph reading stale copies gives ratio ~ 1
# poison_once, per-class non-finite probe: classes of captured input the graphed step MUST read (finite output after NaN-poisoning one = a stale copy).
POISON_REQUIRED_STEP = ("st.rot", "st.trans", "st.eps", "st.t_hat", "st.delta", "st.dt")     # the per-step random draws and noise-schedule scalars
_WHY_COND = ("step-invariant conditioning consumed by the DiT hoist's RECORD pass (it becomes hoist.cond.single / hoist.tok.bias# / hoist.atomenc#|atomdec#.* "
             "slots, which the step reads and which are required below); read directly only when no hoist is bound")
_WHY_FEAT = "step-invariant input feature consumed by the record pass (atom / pair encoders); integer and mask features cannot hold NaN and are compared per item by keycheck"
_WHY_INTERM = "record-time intermediate of the hoist from which the per-step slots are built; the captured step never reads it"
POISON_EXPECTED_UNREAD: Dict[str, str] = {          # class -> why the graphed step legitimately leaves it unread (documentation: the pass rule does not depend on this table)
    "cond.s_inputs": _WHY_COND, "cond.s_trunk": _WHY_COND, "cond.z_trunk": _WHY_COND, "cond.pair_z": _WHY_COND, "cond.p_lm": _WHY_COND,
    "cond.input_feature_dict.restype": _WHY_FEAT, "cond.input_feature_dict.token_bonds": _WHY_FEAT, "cond.input_feature_dict.ref_pos": _WHY_FEAT,
    "cond.input_feature_dict.resolution": _WHY_FEAT, "cond.input_feature_dict.relp": _WHY_FEAT, "cond.input_feature_dict.d_lm": _WHY_FEAT,
    "hoist.glue.pair_z_clone": _WHY_INTERM, "hoist.enc.p_cl": _WHY_INTERM, "hoist.enc.p_cm": _WHY_INTERM, "hoist.enc.p_mlp": _WHY_INTERM, "hoist.tok.znorm": _WHY_INTERM,
}
def poison_class_key(name: str) -> str:
    """The KIND of a captured input: per-step buffers and top-level conditioning tensors are their own class; the input-feature dict is one class per
    feature; hoist slots are grouped by slot name with block indices folded (every instance of a kind is poisoned together, one replay per kind)."""
    if name.startswith("hoist."):
        return "hoist." + re.sub(r"\d+", "#", name[len("hoist."):])
    return name


# Hoist slot kinds the captured step reads with the DiT hoist bound and no other sampler lever serving their consumers (documentation:
# the pass rule below does not depend on this list).  A kind listed here that this process never produced (its producer wrapper was never entered
# because another lever serves that consumer — e.g. a fused attention or block path) is reported as `subsumed:<kinds>`; a produced kind in neither
# list is reported as `unlisted:<kinds>` (documentation drift) and is still required like any recorded slot.
POISON_KNOWN_HOIST = tuple(["hoist.cond.single", "hoist.enc.cl_s", "hoist.tok.bias#"] + [f"hoist.{blk}#.{k}" for blk in ("atomenc", "atomdec") for k in (
    "layernorm_a.sig", "layernorm_a.lin", "layernorm_kv.sig", "layernorm_kv.lin", "locbias", "gate_lin", "ctb.adaln.sig", "ctb.adaln.lin", "ctb.gate_lin")])
POISON_REQUIRED_HOIST = POISON_KNOWN_HOIST      # name kept for callers; the rule is poison_classify
# Also observed read: cond.c_l (read directly by the step even with the hoist bound).

POISON_STATUS = {"ran": False, "unlisted": [], "subsumed": [], "squash": [], "failed": False}     # filled by the per-class probe; shown on the SAMPLER record as poison=…


def poison_status() -> str:
    """`not-run` (no capture yet in this process) | `ok` | `failed`, followed by `;subsumed:<kinds>` (known hoist kinds this process never produced:
    another lever serves their consumer), `;squash:<kinds>` (read, but their consumer maps non-finite operands to finite output — seen by the
    magnitude stage only) and `;unlisted:<kinds>` (produced kinds in neither list) when non-empty."""
    if not POISON_STATUS["ran"]:
        return "not-run"
    out = "failed" if POISON_STATUS["failed"] else "ok"
    for key in ("subsumed", "squash", "unlisted"):
        if POISON_STATUS.get(key):
            out += f";{key}:" + ",".join(POISON_STATUS[key])
    return out


def poison_moved(d_max: float, replay_noise: float) -> bool:
    """Magnitude stage: the class is read when overwriting it moves x_l by >= POISON_MIN_RATIO x the replay-to-replay noise AND >= POISON_MIN_ABS
    (or turns it non-finite)."""
    return d_max == float("inf") or (d_max >= POISON_MIN_RATIO * max(replay_noise, 1e-6) and d_max >= POISON_MIN_ABS)


def poison_needs_magnitude(name: str, *, recorded, hoist_bound: bool) -> bool:
    """The magnitude stage runs only for a class whose verdict depends on it (it would be required): per-step inputs and recorded, non-exempt
    hoist kinds.  Conditioning / feature classes consumed at record time stay single-stage (their read/unread entry is documentation)."""
    if name in POISON_REQUIRED_STEP:
        return True
    return bool(hoist_bound and name.startswith("hoist.") and name not in POISON_EXPECTED_UNREAD and recorded.get(name, True))


def poison_classify(*, read, squash, unread, recorded, hoist_bound: bool, hoist_present=()):
    """The pass rule of the per-class probe, from the live objects only:
      required = every per-step input (POISON_REQUIRED_STEP) + every hoist class whose member slots were all RECORDED this process (dit_hoist
                 Slot.n_rec > 0: the producer wrapper was entered at the eager step, so the captured step must read the slot) except the kinds
                 POISON_EXPECTED_UNREAD names as consumed at record time;
      missing  = required and neither NaN- nor magnitude-read  -> the probe fails (a captured kernel reads a stale or private copy);
      subsumed = POISON_KNOWN_HOIST kinds with no slot this process (their producer was never entered: another lever serves that consumer);
      unlisted = produced hoist kinds in neither POISON_KNOWN_HOIST nor POISON_EXPECTED_UNREAD (documentation drift; still required).
    -> {"required": [...], "missing": [...], "subsumed": [...], "unlisted": [...]}"""
    seen = list(read) + list(squash) + list(unread)
    ok = set(read) | set(squash)
    required = []
    for n in seen:
        if n in POISON_REQUIRED_STEP:
            required.append(n)
        elif hoist_bound and n.startswith("hoist.") and n not in POISON_EXPECTED_UNREAD and recorded.get(n, True):
            required.append(n)
    missing = [n for n in required if n not in ok]
    present_kinds = {poison_class_key("hoist." + str(h)) for h in hoist_present}
    subsumed = sorted(k for k in POISON_KNOWN_HOIST if hoist_bound and k not in present_kinds) if hoist_present or hoist_bound else []
    unlisted = poison_unlisted(seen)
    return {"required": required, "missing": missing, "subsumed": subsumed, "unlisted": unlisted}


def poison_required(name: str, hoist_bound: bool) -> bool:
    """Compatibility helper: True for per-step inputs and, with the hoist bound, for any hoist kind not named as consumed at record time (the probe
    itself narrows this to slots actually recorded this process)."""
    if name in POISON_REQUIRED_STEP:
        return True
    return bool(hoist_bound and name.startswith("hoist.") and name not in POISON_EXPECTED_UNREAD)


def poison_unlisted(names) -> list:
    """Produced classes in neither list (documentation drift: a renamed or new input kind) — reported, still required."""
    known = set(POISON_REQUIRED_STEP) | set(POISON_KNOWN_HOIST) | set(POISON_EXPECTED_UNREAD) | {"cond.c_l"}
    return [n for n in names if n not in known]


RING_SLOTS = 256          # rot_async: pinned host ring depth (one 3x3 fp32 per step; the host never runs more than a few steps ahead)
CHAIN_MIN_FRAC = 0.8      # pool_chain re-uses a pool only for a signature within [0.8, 1.0] x its token count (a much smaller item lets the pool go instead of holding its blocks)
CHAIN_MAX_TOKENS = 1536   # pool_chain: above it an entry is large — always renew (nothing of the previous entry alive during the capture)


class RotationRing:
    """Pinned host ring for the per-step [N_augment, 3, 3] float32 rotations.  A slot is rewritten only after the event recorded behind its
    previous upload has completed; with SLOTS >= the steps of one sampler call that never waits in practice."""
    def __init__(self, slots: int = 256):
        self.slots = int(slots)
        self.rings: Dict[tuple, dict] = {}

    def upload(self, rot_cpu: torch.Tensor, device) -> torch.Tensor:
        key = (rot_cpu.numel(), rot_cpu.dtype)
        r = self.rings.get(key)
        if r is None:
            r = {"pin": torch.empty((self.slots, rot_cpu.numel()), dtype=rot_cpu.dtype, pin_memory=True), "ev": [None] * self.slots, "i": 0}
            self.rings[key] = r
        i = r["i"]; ev = r["ev"][i]
        if ev is not None and not ev.query():
            ev.synchronize(); STATS["rot_ring_waits"] += 1
        slot = r["pin"][i]
        slot.copy_(rot_cpu.reshape(-1))
        dev = slot.to(device, non_blocking=True).reshape(rot_cpu.shape)
        e = torch.cuda.Event(); e.record(); r["ev"][i] = e
        r["i"] = (i + 1) % self.slots
        STATS["rot_uploads"] += 1
        return dev


_RING: Optional[RotationRing] = None


def rotation_to_device(rot_cpu: torch.Tensor, device, prep: PrepConfig) -> torch.Tensor:
    """uniform_random_rotation(...)'s host tensor on `device`: through the pinned ring when rot_async is on, else the plain copy."""
    global _RING
    if prep.rot_async and torch.device(device).type == "cuda":
        if _RING is None:
            _RING = RotationRing(RING_SLOTS)
        return _RING.upload(rot_cpu, device)
    return rot_cpu.to(device)


def feature_shape_key(input_feature_dict) -> tuple:
    """(name, shape, dtype) of every tensor in input_feature_dict — the shape part of the graph cache key under keycheck."""
    return tuple(sorted((k, tuple(v.shape), str(v.dtype)) for k, v in input_feature_dict.items() if isinstance(v, torch.Tensor)))


def same_index_values(ent: dict, input_feature_dict) -> bool:
    """True iff every NON-floating tensor of input_feature_dict equals (torch.equal, on the device) the entry's static copy — the values a captured
    graph may have folded into its launches.  Floating tensors are inputs by value (copied into the static buffers before a replay)."""
    sifd = (ent.get("cond") or {}).get("input_feature_dict")
    if not isinstance(sifd, dict):
        return False
    for k, v in input_feature_dict.items():
        if not isinstance(v, torch.Tensor) or v.is_floating_point() or v.is_complex():
            continue
        sv = sifd.get(k)
        if not isinstance(sv, torch.Tensor) or sv.shape != v.shape or sv.dtype != v.dtype or not torch.equal(sv, v):
            return False
    return True


def release_statics(loop, key):
    """[pool_chain] Free the entry's static buffers (`st`, `cond`, `bc`, hoist buffers) and return its live CUDAGraph (or None).  The entry stays in
    loop.entries / loop.order for the loop's own eviction bookkeeping."""
    ent = loop.entries.get(key)
    if not ent:
        return None
    g = ent.get("graph")
    for k in ("st", "cond", "bc", "dit_hoist"):
        if k in ent:
            ent[k] = None
    STATS["statics_released"] += 1
    return g


def drop_padbias(loop, keep_n_atom: int) -> int:
    """[pool_chain] The DiT hoist keeps a padding-bias cache per atom-count shape across items (`_padbias`, key (N_atom, n_q, n_k, inf, dtype, device),
    up to 8 shapes): every shape other than the item about to be captured is dropped here (deterministic values,
    rebuilt on demand by the hoist; same-shape consecutive items keep theirs)."""
    H = getattr(loop, "biascache", None); d = getattr(H, "_padbias", None)
    if not isinstance(d, dict) or not d:
        return 0
    drop = [k for k in list(d.keys()) if not (isinstance(k, tuple) and len(k) > 0 and k[0] == keep_n_atom)]
    for k in drop:
        d.pop(k, None)
    STATS["padbias_dropped"] += len(drop)
    return len(drop)


def prepare_capture(loop, n_tok: int, n_atom: int, n_sample: int):
    """[pool_chain] A new signature is about to be captured and the cache is full.  Releases the newest entry's static buffers first; then either
    returns (graph, envelope) — the previous graph, kept alive by the caller until the new capture has ended so the capture can borrow its pool: only
    when (n_tok, n_atom, n_sample) is within the envelope that pool has served, so every request finds an existing block — or drops every cached entry
    and returns its pool to the device now (before the new pool exists) and returns (None, None)."""
    prev_key = loop.order[-1]
    prev = loop.entries.get(prev_key) or {}
    env = prev.get("prep_envelope")
    g = release_statics(loop, prev_key)
    drop_padbias(loop, n_atom)
    if (g is not None and env is not None and CHAIN_MIN_FRAC * env[0] <= n_tok <= env[0] and n_atom <= env[1] and n_sample <= env[2]
            and n_tok <= CHAIN_MAX_TOKENS):          # above it an entry is large: always renew (nothing of the previous entry alive during the capture)
        STATS["pool_chained"] += 1
        return g, env
    g = None; prev = None
    while loop.order:
        loop._evict_oldest()
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    STATS["pool_renewed"] += 1
    loop._prep_renewed = True        # the loop's post-capture trim (synchronize + empty_cache) runs once for this capture too: warm-up / capture transients
    return None, None


def envelope_after_capture(chained_env, n_tok: int, n_atom: int, n_sample: int):
    """[pool_chain] The envelope the new entry's pool has served: the previous envelope widened by this signature when chained, else this signature."""
    if chained_env is None:
        return (int(n_tok), int(n_atom), int(n_sample))
    return (max(int(n_tok), chained_env[0]), max(int(n_atom), chained_env[1]), max(int(n_sample), chained_env[2]))


def step_scalars(noise_schedule: torch.Tensor, gamma_pattern, gamma0: float):
    """[stepvec] (t_hat, sqrt(t_hat^2 - c_last^2), c_next - t_hat) for every step as 0-dim views of three vectors."""
    c_last = noise_schedule[:-1]; c_next = noise_schedule[1:]
    fac = torch.tensor([(float(gamma0) if gp else 0) + 1 for gp in gamma_pattern], dtype=torch.float64).to(dtype=c_last.dtype, device=c_last.device)
    t_hat = c_last * fac
    dnl = torch.sqrt(t_hat ** 2 - c_last ** 2)
    dt = c_next - t_hat
    return [(t_hat[i], dnl[i], dt[i]) for i in range(t_hat.shape[0])]


def report() -> dict:
    return dict(STATS)
