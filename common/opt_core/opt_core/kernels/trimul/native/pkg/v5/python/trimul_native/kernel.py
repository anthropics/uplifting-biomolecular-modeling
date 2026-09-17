# SPDX-License-Identifier: Apache-2.0
"""Launch wrappers of the trimul_native kernels (csrc/tmn_kernels.cuh): kernel selection by name from the tile table, tensor-map encoding, the
by-value parameter blocks (K1Params / K3Params) as pointer-patched ArgPacks, persistent-grid launches through launch.py (CUDA driver).

All shape / stride / descriptor logic lives here; the device code takes one POD parameter block per kernel.  Host-lean serving: every
(kernel, N, form) builds its ArgPack once (kept in the caller's cache); a call patches the tensor maps / pointers that changed and makes one
driver call per kernel.

Kernel names (csrc/tmn_sm90_unit.cuh):  tmn_k1_z<CZ>_h<CH>_<b|f>_t<BI>x<BJ>_s<NSLOT>k<SKCH>_m<0|1>_l<LNM>_v<0|1>[_x1]   |   tmn_k3_z<CZ>_h<CH>_<b|f|p>_t<BI>x<BJ>_s<NSLOT>a<NACC>_l<LNM>[_u]
(K3 operand modes: b = bf16 z tile, f = fp32 z tile (fp32 sum out), g = fp32 z tile with the bf16 update out (no residual), p = pre-normalised bf16 rows written by K1's _x1 variant with fp32 residual/output; LNM: 1 = the shared
numerics statement, 2/3 = reference-order LayerNorms (bitwise variant), 4 = trimul_tx 1.2's arithmetic (comparison only).)
"""
import os

import torch

from . import launch as L

NTHREADS = 384
BLOCK = (NTHREADS, 1, 1)

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Tile table (tuning data): (arch, c_z, c_hidden, 'b'|'f') -> default K1 config (BI, BJ, NSLOT, SKCH) and K3 config (BI, BJ, NSLOT, NACC);
# Optional per-entry key (bf16 entries; default 0): k3_u_late = the update-only K3 (<name>_u) releases its z chunks pair by pair after each pair's output
# pass (1) instead of all at once after the fragment load (0).
# every entry is an instantiation compiled into the width's unit (csrc/tmn_sm90_unit.cuh); `k1_variants` / `k3_variants` = the alternatives built.
# 'f' entries: k3_mode 'f' (default) = K3 reads the fp32 z tile; 'p' = the fp32 K1 also writes bf16(LN_in(z)) rows and K3 reads those; 'c' = z is cast
# to bf16 once, the bf16 K1 runs on the cast, K3 mode c normalises the cast tile and adds the fp32 residual from global (fp32 out).
INCOMING_MODE = "kt"       # incoming direction: "kt" = K1 reads z transposed and writes transposed planes so both directions run the NT contraction;
                           # "tn" = plain planes + TN contraction.  The single source for ops.py and the package face.

TILE_TABLE = {
    ("sm_90a", 256, 256, "b"): dict(k1=(2, 64, 4, 4), k3=(2, 64, 4, 1), k3_u_late=1,   # update-only K3 releasing z pair by pair: -18..-24 % K3 at N 800-2048 (per-kernel same-process A/B); 64 / 128 keep the eager release (measured faster there)   # K1 4 x 32 KB ring: -1..-3 % op vs (2,64,8,2) at N 1200-2048 (same-process A/B); K3 (1,128,4,1) at
                                     k1_variants=[(2, 64, 4, 4), (2, 64, 8, 2), (1, 128, 8, 2)], k3_variants=[(2, 64, 4, 1), (1, 128, 4, 1), (1, 64, 4, 1)]),   # N > 1200: -0.6 % without / +1..3 % with residual -> not tabled
    ("sm_90a", 256, 256, "f"): dict(k1=(2, 64, 4, 2), k3=(2, 64, 4, 1), k3_mode="c", k1_variants=[(2, 64, 4, 2)], k3_variants=[(2, 64, 4, 1)], k3_variants_f=[(1, 64, 4, 1)]),   # fp32 z at 256: route "c" = one bf16 cast pass + the bf16 K1 + K3 mode c (fp32 residual/out); "p" = fp32 K1 emitting bf16 LN rows + K3 mode p; "f" = fp32 tiles (split-N K3)
    ("sm_90a", 128, 128, "b"): dict(k1=(6, 32, 8, 2), k3=(2, 64, 8, 1), by_n={1536: dict(k3=(2, 64, 8, 2)), 4096: dict(k3=(2, 64, 8, 1))},   # measured: 192-token K1 tile -3..-5 % op vs (2,64,8,2) at every N (same-process A/B); K3 two accumulator sets N <= 1536
                                     k1_variants=[(6, 32, 8, 2), (3, 64, 8, 2), (2, 64, 8, 2), (1, 128, 8, 2), (2, 64, 4, 2)], k3_variants=[(2, 64, 8, 1), (2, 64, 8, 2), (1, 128, 8, 1), (2, 64, 4, 1), (1, 64, 8, 1)]),
    ("sm_90a", 128, 128, "f"): dict(k1=(2, 64, 8, 2), k3=(2, 64, 8, 1), k1_variants=[(2, 64, 8, 2)], k3_variants=[(2, 64, 8, 1), (1, 64, 8, 1)]),
    ("sm_90a", 64, 64, "b"): dict(k1=(3, 64, 4, 1), k3=(2, 64, 4, 1), by_n={400: dict(k1=(6, 32, 4, 1), k3=(2, 64, 4, 2)), 1536: dict(k1=(6, 32, 4, 1)), 4096: dict()},   # measured: 192-token K1 tiles (3 consumer warpgroups): (6,32) N<=1536, (3,64) above (-7..-16 % K1 vs the 128-token tiles); K3 a2 N<=400
                                   k1_variants=[(3, 64, 4, 1), (6, 32, 4, 1), (1, 192, 4, 1), (2, 64, 4, 1), (4, 32, 4, 1), (1, 128, 4, 1)], k3_variants=[(2, 64, 4, 1), (2, 64, 4, 2), (1, 128, 4, 1)]),
    ("sm_90a", 64, 64, "f"): dict(k1=(3, 64, 4, 1), k3=(2, 64, 4, 1), k1_variants=[(3, 64, 4, 1), (2, 64, 4, 1)], k3_variants=[(2, 64, 4, 1)]),
    ("sm_90a", 64, 256, "b"): dict(k1=(2, 64, 16, 1), k3=(2, 64, 4, 1), k1_variants=[(2, 64, 16, 1)], k3_variants=[(2, 64, 4, 1)]),
    ("sm_90a", 64, 256, "f"): dict(k1=(2, 64, 16, 1), k3=(2, 64, 4, 1), k1_variants=[(2, 64, 16, 1)], k3_variants=[(2, 64, 4, 1)]),
    ("sm_90a", 128, 64, "b"): dict(k1=(2, 64, 4, 2), k3=(2, 64, 4, 1), k1_variants=[(2, 64, 4, 2)], k3_variants=[(2, 64, 4, 1), (2, 64, 8, 2)]),
    ("sm_90a", 128, 64, "f"): dict(k1=(2, 64, 4, 2), k3=(2, 64, 8, 1), k1_variants=[(2, 64, 4, 2)], k3_variants=[(2, 64, 8, 1)]),
    ("sm_90a", 128, 256, "b"): dict(k1=(2, 64, 8, 2), k3=(2, 64, 4, 1), k1_variants=[(2, 64, 8, 2)], k3_variants=[(2, 64, 4, 1)]),
    ("sm_90a", 128, 256, "f"): dict(k1=(2, 64, 8, 2), k3=(1, 64, 4, 1), k1_variants=[(2, 64, 8, 2)], k3_variants=[(1, 64, 4, 1)]),
    ("sm_90a", 256, 64, "b"): dict(k1=(2, 64, 8, 2), k3=(2, 64, 4, 1), k1_variants=[(2, 64, 8, 2)], k3_variants=[(2, 64, 4, 1), (2, 64, 6, 1)]),
    ("sm_90a", 256, 64, "f"): dict(k1=(2, 64, 4, 2), k3=(1, 64, 4, 1), k1_variants=[(2, 64, 4, 2)], k3_variants=[(1, 64, 4, 1)]),
    ("sm_90a", 256, 128, "b"): dict(k1=(2, 64, 8, 2), k3=(2, 64, 4, 1), k1_variants=[(2, 64, 8, 2)], k3_variants=[(2, 64, 4, 1)]),
    ("sm_90a", 256, 128, "f"): dict(k1=(2, 64, 4, 2), k3=(1, 64, 4, 1), k1_variants=[(2, 64, 4, 2)], k3_variants=[(1, 64, 4, 1)]),
    ("sm_90a", 64, 128, "b"): dict(k1=(3, 64, 8, 1), k3=(2, 64, 4, 1), by_n={800: dict(k1=(6, 32, 8, 1)), 1536: dict(), 4096: dict(k3=(2, 64, 4, 2))},   # measured: 192-token K1 tiles (3 consumer warpgroups): (6,32) N<=800, (3,64) above (-9..-16 % K1 vs the 128-token tiles); K3 a2 N>=2048
                                    k1_variants=[(3, 64, 8, 1), (6, 32, 8, 1), (1, 192, 8, 1), (2, 64, 8, 1), (4, 32, 8, 1), (1, 128, 8, 1)], k3_variants=[(2, 64, 4, 1), (2, 64, 4, 2), (1, 128, 4, 1)]),
    ("sm_90a", 64, 128, "f"): dict(k1=(3, 64, 8, 1), k3=(2, 64, 4, 1), k1_variants=[(3, 64, 8, 1), (2, 64, 8, 1)], k3_variants=[(2, 64, 4, 1)]),
    # wide-K pairs (c_z/16 + c_hidden/16 > 24): K3 = the wide member (tmn_k3_wide.cuh), marked by a trailing "w" in the K3 config
    ("sm_90a", 384, 384, "b"): dict(k1=(2, 64, 6, 2), k3=(1, 64, 4, 1, "w"), k1_variants=[(2, 64, 6, 2), (1, 128, 6, 2), (2, 64, 4, 3)], k3_variants=[(1, 64, 4, 1, "w")]),
    ("sm_90a", 128, 384, "b"): dict(k1=(2, 64, 8, 2), k3=(2, 64, 4, 1, "w"), k1_variants=[(2, 64, 8, 2)], k3_variants=[(2, 64, 4, 1, "w"), (2, 64, 4, 2, "w")]),
    ("sm_90a", 384, 128, "b"): dict(k1=(2, 64, 6, 2), k3=(2, 64, 4, 1, "w"), k1_variants=[(2, 64, 6, 2)], k3_variants=[(2, 64, 4, 1, "w")]),
    ("sm_90a", 256, 384, "b"): dict(k1=(2, 64, 8, 2), k3=(1, 64, 4, 1, "w"), k1_variants=[(2, 64, 8, 2)], k3_variants=[(1, 64, 4, 1, "w")]),
    ("sm_90a", 384, 256, "b"): dict(k1=(2, 64, 6, 2), k3=(1, 64, 4, 1, "w"), k1_variants=[(2, 64, 6, 2)], k3_variants=[(1, 64, 4, 1, "w")]),
    ("sm_90a", 64, 384, "b"): dict(k1=(2, 64, 8, 1), k3=(2, 64, 4, 2, "w"), k1_variants=[(2, 64, 8, 1)], k3_variants=[(2, 64, 4, 2, "w"), (2, 64, 4, 1, "w")]),
    ("sm_90a", 384, 64, "b"): dict(k1=(2, 64, 6, 2), k3=(2, 64, 4, 1, "w"), k1_variants=[(2, 64, 6, 2)], k3_variants=[(2, 64, 4, 1, "w")]),
}


def k1_shape(bi, bj):
    """== K1Cfg::NCWG / NTHR / MINB: consumer warpgroups = tile tokens / 64, threads per CTA, CTAs per SM (the unit's launch bounds)."""
    ncwg = bi * bj // 64
    assert bi * bj == 64 * ncwg and 1 <= ncwg <= 4, (bi, bj)
    return ncwg, 128 * (ncwg + 1), (2 if ncwg == 1 else 1)


def k1_smem(cz, ch, zf32, bi, bj, nslot, skch):
    """== K1Cfg::SMEM (csrc/tmn_kernels.cuh)."""
    ncwg = k1_shape(bi, bj)[0]
    esz = 4 if zf32 else 2
    nkca = cz // (128 // esz)
    nbar = 2 + 2 * nslot
    return nkca * bi * bj * 128 + nslot * skch * 8192 + ncwg * 8192 + 2 * cz * 4 + ((nbar * 8 + 127) // 128) * 128


def k3_smem(cz, ch, mode, bi, bj, nslot, nacc=1):
    """== K3Cfg::SMEM (mode 'b' | 'f' | 'p'; bool accepted: True = 'f')."""
    if isinstance(mode, bool):
        mode = "f" if mode else "b"
    esz = 4 if mode in ("f", "g") else 2          # g: fp32 tile (staging is sized in the tile dtype), bf16 update out
    bmt = bi * bj
    nkcz = cz // (128 // esz)
    slot = max(cz, ch) * 64
    ob = 16 * 64 * esz
    nbar = 2 * nkcz + 2 + 2 * nslot
    return (bmt // 64) * ch * 128 + nkcz * bmt * 128 + nslot * slot + 8 * ob + (2 * cz + 2 * ch) * 4 + ((nbar * 8 + 127) // 128) * 128


def k3w_smem(cz, ch, mode, bi, bj, nslot, nacc=1):
    """== K3WCfg::SMEM (csrc/tmn_k3_wide.cuh; modes 'b' | 'f'): X sub-tiles + z chunks + kind-sized (projection, gate) slot pairs + staging + LN params + barriers."""
    if isinstance(mode, bool):
        mode = "f" if mode else "b"
    assert mode in ("b", "f"), "the wide K3 member serves modes b and f"
    esz = 4 if mode in ("f", "g") else 2          # g: fp32 tile (staging is sized in the tile dtype), bf16 update out
    bmt = bi * bj
    nkcz = cz // (128 // esz)
    nb = cz // 32
    nslot_eff = 2 * nb if nslot >= 2 * nb else nslot
    w = (nslot_eff // 2) * (ch * 64 + cz * 64)
    nbar = 2 * nkcz + 2 + 2 * nslot_eff
    return (bmt // 64) * ch * 128 + nkcz * bmt * 128 + w + 8 * (16 * 64 * esz) + (2 * cz + 2 * ch) * 4 + ((nbar * 8 + 127) // 128) * 128


def k3_is_wide(cfg):
    """K3 configs are (BI, BJ, NSLOT, NACC) for the register-resident K3 or (BI, BJ, NSLOT, NACC, "w") for the wide member."""
    return len(cfg) > 4 and cfg[4] == "w"


def k1_name(cz, ch, zf32, bi, bj, nslot, skch, mask, lnm, save, emitx=False):
    return "tmn_k1_z%d_h%d_%s_t%dx%d_s%dk%d_m%d_l%d_v%d%s" % (cz, ch, "f" if zf32 else "b", bi, bj, nslot, skch, int(mask), lnm, int(save), "_x1" if emitx else "")


def k3_name(cz, ch, mode, bi, bj, nslot, nacc, *rest):
    """k3_name(cz, ch, mode, *cfg, lnm) for both K3 config shapes: (BI, BJ, NSLOT, NACC) -> tmn_k3_..., (BI, BJ, NSLOT, NACC, "w") -> tmn_k3w_..."""
    if isinstance(mode, bool):
        mode = "f" if mode else "b"
    assert len(rest) in (1, 2) and (len(rest) == 1 or rest[0] == "w"), "k3_name(cz, ch, mode, BI, BJ, NSLOT, NACC[, 'w'], lnm)"
    return "tmn_k3%s_z%d_h%d_%s_t%dx%d_s%da%d_l%d" % ("w" if len(rest) == 2 else "", cz, ch, mode, bi, bj, nslot, nacc, rest[-1])


def lookup(arch, cz, ch, form, N=None, direction=None, residual=None, exact=False):
    """Deterministic launch-configuration lookup = the UNCOVERED-CELL rule's kernel side.
    Keys: arch ('sm_90a'; the sm_80 member has its own table), c_z, c_hidden, form ('b' bf16 z | 'f' fp32-resident z), N, direction
    ('outgoing' | 'incoming'), residual (bool), exact (numerics class: False = tolerance class, True = bitwise class).
    Returns None when (arch, c_z, c_hidden, form) has no entry: the op REFUSES BY NAME -- a width the kernels do not serve on this card is never
    inherited from another width, another card's table is never consulted, and the exact class never inherits a tolerance-class row (or vice versa).
    Otherwise returns dict(key, k1, k3, k3_mode, covered, token): the entry's defaults, refined by tuning buckets when present -- per call form
    `by_call[(direction, residual, "exact"|"fast")] = {n_hi: {k1:.., k3:..}}`, else the entry-wide `by_n = {n_hi: {...}}` -- choosing the bucket with
    the smallest n_hi >= N; 'nearest' is taken ONLY along N (an N above every bucket uses the largest bucket) and then `covered` is False and `token`
    = "UNCOVERED_CELL:<key>" for the face / provider info line (once per key per process; never an exception).  `key` = (arch, form, class, c_z,
    c_hidden, direction, residual, n_bucket)."""
    ent = TILE_TABLE.get((arch, cz, ch, form))
    if ent is None or not ent.get("k1") or not ent.get("k3"):
        return None
    cls = "exact" if exact else "fast"
    key = (arch, form, cls, cz, ch, direction, bool(residual) if residual is not None else None, None)
    out = dict(key=key, k1=tuple(ent["k1"]), k3=tuple(ent["k3"]), k3_mode=ent.get("k3_mode", "f" if form == "f" else "b"), covered=True, token=None)
    tbl = (ent.get("by_call") or {}).get((direction, bool(residual) if residual is not None else None, cls))   # this call form's own buckets only
    if tbl is None:
        tbl = ent.get("by_n")                                                                                  # entry-wide buckets (all call forms)
    if tbl and N is not None:
        his = sorted(tbl)
        sel = next((h for h in his if N <= h), None)
        out["covered"] = sel is not None
        sel = his[-1] if sel is None else sel
        out["key"] = key[:-1] + (sel,)
        out["k1"] = tuple(tbl[sel].get("k1", out["k1"])); out["k3"] = tuple(tbl[sel].get("k3", out["k3"]))
        if "k3_mode" in tbl[sel]:
            out["k3_mode"] = tbl[sel]["k3_mode"]
    if not out["covered"]:
        out["token"] = "UNCOVERED_CELL:" + ":".join(str(x) for x in out["key"])
    return out

def serve_names(arch, cz, ch, form, exact=False):
    """The kernel names the op launches for (c_z, c_hidden, form 'b' | 'f') with the table defaults and default numerics -- the list a sealed unit
    must contain (the package face derives its 'serves' records from this; kernel.py stays the single naming authority).
    bf16 z: K1 l2 (m0, m1) + K3 b l1 [+ K3 b l2, l3 for the exact word];  fp32 z by k3_mode: 'f' -> K1 f l1 (m0, m1) + K3 f l1;
    'p' -> K1 f l1 _x1 (m0, m1) + K3 p l1;  'c' -> the bf16 K1 l2 (m0, m1) + K3 c l1 + K3 b l1."""
    names = []
    tb = TILE_TABLE.get((arch, cz, ch, "b"))
    if form == "b":
        if tb is None or not tb.get("k1") or not tb.get("k3"):
            return []                               # no default K1 / K3 instantiation at this width (e.g. c_z 384: K3 is the wide member's): not served
        for k1c in _cfgs(tb, "k1"):
            names += [k1_name(cz, ch, False, *k1c, m, 2, False) for m in (False, True)]
        for k3c in _cfgs(tb, "k3"):
            names.append(k3_name(cz, ch, "b", *_k3cfg(k3c), 1))
            if not k3_is_wide(_k3cfg(k3c)):
                names.append(k3_name(cz, ch, "b", *_k3cfg(k3c), 1) + "_u")   # + its update-only specialisation (TMN_K3_SET_BF16 emits both)
        if exact:                                   # the exact word runs the entry defaults (reference-order K3 l2 / l3)
            names += [k3_name(cz, ch, "b", *_k3cfg(tb["k3"]), l) for l in (2, 3)]
        return list(dict.fromkeys(names))
    tf = TILE_TABLE.get((arch, cz, ch, "f"))
    if tf is None or not tf.get("k1") or not tf.get("k3"):
        return []
    mode = tf.get("k3_mode", "f")
    if mode == "c":
        if tb is None:
            return []
        for k1c in _cfgs(tb, "k1"):
            names += [k1_name(cz, ch, False, *k1c, m, 2, False) for m in (False, True)]
        for k3c in _cfgs(tf, "k3"):
            names.append(k3_name(cz, ch, "c", *_k3cfg(k3c), 1))
        for k3c in _cfgs(tb, "k3"):
            names.append(k3_name(cz, ch, "b", *_k3cfg(k3c), 1))
    elif mode == "p":
        for k1c in _cfgs(tf, "k1"):
            names += [k1_name(cz, ch, True, *k1c, m, 1, False, True) for m in (False, True)]
        for k3c in _cfgs(tf, "k3"):
            names.append(k3_name(cz, ch, "p", *_k3cfg(k3c), 1))
    else:
        for k1c in _cfgs(tf, "k1"):
            names += [k1_name(cz, ch, True, *k1c, m, 1, False) for m in (False, True)]
        for k3c in _cfgs(tf, "k3"):
            names.append(k3_name(cz, ch, "f", *_k3cfg(k3c), 1))          # residual=True: z + update in fp32
            if not k3_is_wide(k3c):
                names.append(k3_name(cz, ch, "g", *_k3cfg(k3c), 1))      # residual=False: the update in bf16 (the compute dtype)
    return list(dict.fromkeys(names))


def _cfgs(ent, which):
    """The entry default + every bucket override of `which` ('k1' | 'k3'), in a stable order ([] when the entry has no default: not served)."""
    if not ent.get(which):
        return []
    out = [tuple(ent[which])]
    for tbl in [ent.get("by_n") or {}] + list((ent.get("by_call") or {}).values()):
        for h in sorted(tbl):
            c = tbl[h].get(which)
            if c is not None and tuple(c) not in out:
                out.append(tuple(c))
    return out


def _k3cfg(c):
    """Normalised K3 config: (BI, BJ, NSLOT, NACC) or (BI, BJ, NSLOT, NACC, "w") (wide member); a 3-element config gets NACC = 1."""
    c = tuple(c)
    wide = c[-1:] == ("w",)
    core = c[:-1] if wide else c
    core = core if len(core) == 4 else core + (1,)
    return core + (("w",) if wide else ())


def unit_name(arch, cz, ch):
    return {"sm_90a": "tmn90"}[arch] + "_z%d_h%d" % (cz, ch)


K1_PARAMS_SIZE, K3_PARAMS_SIZE = 512, 768     # sizeof(K1Params), sizeof(K3Params) incl. the plane-store / output-store tensor maps (tmn_info reports the device-side layout)
# K1Params: 2 maps (256) + 6 pointers (48) + 7 int + eps (32) + 4 int (16) = 352 -> 384 with the 64-byte struct alignment


class Kernels:
    """Loaded units of one device + cached kernels; k1()/k3() launch on the current torch stream with pointer-patched parameter blocks."""

    def __init__(self, device=None, build_dir=None, verify=True):
        self.device_index = torch.cuda.current_device() if device is None else (device.index if isinstance(device, torch.device) else int(device))
        if self.device_index is None:
            self.device_index = torch.cuda.current_device()
        self.arch, self.cc = L.arch_of_device(self.device_index)
        self.build_dir = build_dir or os.environ.get("TRIMUL_NATIVE_BUILD_DIR") or L.DEFAULT_BUILD_DIR
        self._hasname = {}                       # (cz, ch, kernel name) -> the loaded unit carries it (the update-only K3 specialisations)
        self.verify = verify
        self._units = {}
        self._kern = {}
        self.sms = torch.cuda.get_device_properties(self.device_index).multi_processor_count
        self._has = {}

    # ---------------------------------------------------------------- loading
    def has_unit(self, cz, ch):
        r = self._has.get((cz, ch))
        if r is None:
            r = bool(self.arch) and os.path.isfile(os.path.join(self.build_dir, self.arch, unit_name(self.arch, cz, ch) + ".cubin"))
            self._has[(cz, ch)] = r
        return r

    def stream_handle(self):
        """The current stream of the unit's device as a raw handle (one cheap query per op call; launches take it explicitly)."""
        try:
            return int(torch._C._cuda_getCurrentRawStream(self.device_index))
        except AttributeError:
            return int(torch.cuda.current_stream(self.device_index).cuda_stream)

    def unit(self, cz, ch):
        u = self._units.get((cz, ch))
        if u is None:
            u = L.load_unit(unit_name(self.arch, cz, ch), arch=self.arch, device=self.device_index, build_dir=self.build_dir, verify=self.verify)
            self._units[(cz, ch)] = u
        return u

    def kernel(self, cz, ch, name, smem):
        k = self._kern.get(name)
        if k is None:
            k = self.unit(cz, ch).kernel(name)
            if k.max_dynamic_smem is None or k.max_dynamic_smem < smem:
                k.set_max_dynamic_smem(smem)
            self._kern[name] = k
        return k

    def attrs(self, cz, ch, name, smem):
        return self.kernel(cz, ch, name, smem).attrs()

    # ---------------------------------------------------------------- tensor maps (cached by geometry + base pointer in the caller's cache)
    @staticmethod
    def tmap(cache, key, tensor, box, dims, strides, swz, l2, dtype=None):
        """Descriptor for (geometry key, address): cached per key for the last few addresses seen (workspaces are stable; a caller's z moves
        between a handful of allocator blocks) -- never pins the tensor, bounded growth."""
        sub = cache.get(("tmn.tm",) + key)
        if sub is None:
            sub = {}
            cache[("tmn.tm",) + key] = sub
        ptr = tensor.data_ptr()
        tm = sub.get(ptr)
        if tm is None:
            tm = L.tensor_map(tensor, box, dims=dims, strides_bytes=strides, swizzle=swz, l2=l2, dtype=dtype)
            tm = L.TensorMap(tm.raw, keep=None, spec=tm.spec)
            if len(sub) >= 8:
                sub.clear()
            sub[ptr] = tm
        return tm

    # ---------------------------------------------------------------- K1
    def k1(self, z, mask, w, ab, *, N, Np, cz, ch, cfg=None, lnm=1, transpose=False, stats=None, xz=None, eps=1e-5, cache, stream=None, grid_limit=0):
        """z [N, N, cz] contiguous (bf16 or fp32); mask None or fp32 [N, N] contiguous; w = packed weights (ops.pack_weights); ab = planes
        [2 ch, Np, Np] bf16 (fully written incl. zero pads).  transpose=True reads z (and mask) with the token axes swapped: the planes then hold
        the transposed a/b (incoming direction contracts with the same NT GEMM as outgoing).  xz (fp32 z only): a bf16 [N, N, cz] buffer that receives
        bf16(LN_in(z)) in z's own layout (the _x1 kernels) for a mode-p K3."""
        zf32 = z.dtype == torch.float32
        esz = 4 if zf32 else 2
        bi, bj, nslot, skch = cfg or TILE_TABLE[(self.arch, cz, ch, "f" if zf32 else "b")]["k1"]
        name = k1_name(cz, ch, zf32, bi, bj, nslot, skch, mask is not None, lnm, stats is not None, xz is not None)
        ent = cache.get(("tmn.k1", name, N, Np, transpose, float(eps)))     # every scalar baked into the plan is part of its key (N, Np, transpose -> mask strides / vec, eps);
                                                                                # pointers / maps / the weight set are re-patched per call below
        if ent is None:
            smem = k1_smem(cz, ch, zf32, bi, bj, nslot, skch)
            k = self.kernel(cz, ch, name, smem)
            tiles_i, tiles_j = -(-Np // bi), -(-Np // bj)
            num_tiles = tiles_i * tiles_j
            ms_i, ms_j = (1, N) if transpose else (N, 1)
            tm_w = self.tmap(cache, ("w1", cz, ch), w["w1"], [64, 64], [cz, 4 * ch], [cz * 2], "128B", "256B")
            zero_tm = L.TensorMap(b"\0" * 128)
            st = L.Struct([zero_tm, tm_w, zero_tm, None, w["ln_in_w"], w["ln_in_b"], None, None, None, int(N), int(Np), int(tiles_j), int(num_tiles), int(Np % 8 == 0), int(ms_i), int(ms_j), float(eps),
                           int(ms_i * cz), int(ms_j * cz), 0, 0])
            pack = L.ArgPack([st])
            assert pack.sizes[0] == K1_PARAMS_SIZE, pack.sizes[0]
            ncwg, nthr, minb = k1_shape(bi, bj)
            ctas = self.sms * minb                # persistent CTAs: MINB per SM
            grid = (min(num_tiles, ctas) if not grid_limit else min(num_tiles, ctas, grid_limit), 1, 1)
            wsig = (w["w1"].data_ptr(), w["ln_in_w"].data_ptr(), w["ln_in_b"].data_ptr())
            ent = [pack, k, grid, smem, None, None, None, None, None, wsig, (nthr, 1, 1)]   # + last-patched (z map, mask ptr, ab ptr, stats ptr, xz ptr, weight signature), block
            cache[("tmn.k1", name, N, Np, transpose, float(eps))] = ent
        pack, k, grid, smem, block = ent[0], ent[1], ent[2], ent[3], ent[10]
        wsig = (w["w1"].data_ptr(), w["ln_in_w"].data_ptr(), w["ln_in_b"].data_ptr())
        if ent[9] != wsig:                       # another weight set through the same cache (one cache serving every layer): re-point the weight
            tm_w = self.tmap(cache, ("w1", cz, ch), w["w1"], [64, 64], [cz, 4 * ch], [cz * 2], "128B", "256B")   # stream and the LN_in affine
            pack.set_tensor_map(0, tm_w, field=1); pack.set_ptr(0, wsig[1], field=4); pack.set_ptr(0, wsig[2], field=5); ent[9] = wsig
        s_fast, s_slow = cz * esz, N * cz * esz
        if transpose:
            s_fast, s_slow = s_slow, s_fast
        tm_z = self.tmap(cache, ("z1", N, cz, esz, bi, bj, transpose), z, [128 // esz, bj, bi], [cz, N, N], [s_fast, s_slow], "128B", "128B")
        mp, abp, stp = (mask.data_ptr() if mask is not None else 0), ab.data_ptr(), (stats.data_ptr() if stats is not None else 0)
        if ent[4] is not tm_z:
            pack.set_tensor_map(0, tm_z, field=0); ent[4] = tm_z
        if ent[5] != mp:
            pack.set_ptr(0, mp, field=3); ent[5] = mp
        if ent[6] != abp:                        # planes pointer + the planes store descriptor (dims [Np (j)][Np (i)][2 ch], box [64][1][32]: one channel row of a 64-token run)
            tm_ab = self.tmap(cache, ("ab1", Np, ch), ab, [64, 1, 32], [Np, Np, 2 * ch], [Np * 2, Np * Np * 2], "128B", "128B")
            pack.set_tensor_map(0, tm_ab, field=2); pack.set_ptr(0, abp, field=6); ent[6] = abp
        if ent[7] != stp:
            pack.set_ptr(0, stp, field=7); ent[7] = stp
        xp = xz.data_ptr() if xz is not None else 0
        if ent[8] != xp:
            pack.set_ptr(0, xp, field=8); ent[8] = xp
        k.launch_packed(grid, block, pack, smem, self.stream_handle() if stream is None else stream)
        return ab

    # ---------------------------------------------------------------- K3
    def k3(self, x, z, w, out, *, N, Np, cz, ch, residual, cfg=None, lnm=1, eps=1e-5, cache, stream=None, grid_limit=0, prof=None, zres=None, prenorm=True):
        """x = contraction planes [ch, Np, Np] bf16 contiguous (Np % 8 == 0); z = the gate operand rows [N, N, cz] contiguous: the bf16 or fp32 pair
        tensor itself (modes b / f: LN_in applied here; out has z's dtype), or bf16(LN_in(z)) rows from K1's _x1 kernel with zres = the fp32 pair
        tensor (mode p: fp32 residual + fp32 out; prenorm=False = mode c: z holds a bf16 cast of zres and LN_in is applied here)."""
        mode = ("g" if out.dtype == torch.bfloat16 else "f") if z.dtype == torch.float32 else ("b" if zres is None else ("p" if prenorm else "c"))
        if mode in ("p", "c"):
            assert zres.dtype == torch.float32 and out.dtype == torch.float32
        if mode == "g":
            assert not residual, "mode g (fp32 z tile, bf16 update) never adds a residual: the fp32 sum is mode f"
        zf32 = mode in ("f", "g")
        esz = 4 if zf32 else 2
        ent_tt = TILE_TABLE[(self.arch, cz, ch, "b" if mode == "b" else "f")]
        cfg = _k3cfg(cfg or ent_tt["k3"])
        bi, bj, nslot, nacc = cfg[:4]
        name = k3_name(cz, ch, mode, *cfg, lnm)
        upd = False
        if not residual and not k3_is_wide(cfg):   # the update-only specialisation (name + "_u": residual add compiled out) when the unit carries it
            nu = name + "_u"
            hk = self._hasname.get((cz, ch, nu))
            if hk is None:
                hk = nu in set(self.unit(cz, ch).kernel_names())
                self._hasname[(cz, ch, nu)] = hk
            if hk:
                name, upd = nu, True
        ent = cache.get(("tmn.k3", name, N, Np, float(eps)))
        if ent is None:
            smem = (k3w_smem if k3_is_wide(cfg) else k3_smem)(cz, ch, mode, bi, bj, nslot, nacc)
            k = self.kernel(cz, ch, name, smem)
            tiles_i, tiles_j = -(-N // bi), -(-N // bj)
            num_tiles = tiles_i * tiles_j
            tm_wg = self.tmap(cache, ("wg", cz), w["wg"], [64, 32], [cz, cz], [cz * 2], "128B", "256B")
            tm_wo = self.tmap(cache, ("wo", cz, ch), w["wo"], [64, 32], [ch, cz], [ch * 2], "128B", "256B")
            zero_tm = L.TensorMap(b"\0" * 128)
            st = L.Struct([zero_tm, zero_tm, tm_wg, tm_wo, zero_tm, w["ln_in_w"], w["ln_in_b"], w["ln_out_w"], w["ln_out_b"], None, None, None,
                           int(N), int(Np), int(tiles_j), int(num_tiles), 0, 0, float(eps), 0])
            pack = L.ArgPack([st])
            assert pack.sizes[0] == K3_PARAMS_SIZE, pack.sizes[0]
            grid = (min(num_tiles, self.sms) if not grid_limit else min(num_tiles, self.sms, grid_limit), 1, 1)
            wsig = tuple(w[n].data_ptr() for n in ("wg", "wo", "ln_in_w", "ln_in_b", "ln_out_w", "ln_out_b"))
            ent = [pack, k, grid, smem, None, None, None, None, wsig, None]   # + last-patched (z map, x map, residual flag, prof ptr, weight signature, out map)
            cache[("tmn.k3", name, N, Np, float(eps))] = ent
        pack, k, grid, smem = ent[0], ent[1], ent[2], ent[3]
        wsig = tuple(w[n].data_ptr() for n in ("wg", "wo", "ln_in_w", "ln_in_b", "ln_out_w", "ln_out_b"))
        if ent[8] != wsig:                       # another weight set through the same cache: re-point W_og / W_o streams and the four LN affines
            tm_wg = self.tmap(cache, ("wg", cz), w["wg"], [64, 32], [cz, cz], [cz * 2], "128B", "256B")
            tm_wo = self.tmap(cache, ("wo", cz, ch), w["wo"], [64, 32], [ch, cz], [ch * 2], "128B", "256B")
            pack.set_tensor_map(0, tm_wg, field=2); pack.set_tensor_map(0, tm_wo, field=3)
            for f, p in zip((5, 6, 7, 8), wsig[2:]):
                pack.set_ptr(0, p, field=f)
            ent[8] = wsig
        tm_z = self.tmap(cache, ("z3", N, cz, esz, bi, bj), z, [128 // esz, bj, bi], [cz, N, N], [cz * esz, N * cz * esz], "128B", "128B")
        tm_x = self.tmap(cache, ("x3", Np, ch), x, [64, 1, 64], [Np, Np, ch], [Np * 2, Np * Np * 2], "128B", "128B")
        if ent[4] is not tm_z:
            pack.set_tensor_map(0, tm_z, field=0); pack.set_ptr(0, z if zres is None else zres, field=9); ent[4] = tm_z
        elif zres is not None:
            pack.set_ptr(0, zres, field=9)
        if ent[5] is not tm_x:
            pack.set_tensor_map(0, tm_x, field=1); ent[5] = tm_x
        pack.set_ptr(0, out, field=10)
        if mode == "b" and not k3_is_wide(cfg):   # bf16 output store descriptor: dims [cz (c)][N (j)][N (i)], box [64][16][1] = one warp's 16-token x 64-channel slice
            tm_o = self.tmap(cache, ("o3", N, cz), out, [64, 16, 1], [cz, N, N], [cz * 2, N * cz * 2], "128B", "128B")
            if ent[9] is not tm_o:
                pack.set_tensor_map(0, tm_o, field=4); ent[9] = tm_o
        # residual word: 1 on the residual path; on the update-only kernel it selects the z-chunk release schedule (table data per width): 1 = pair by pair
        # after each pair's output pass (as the residual path does), 0 = every chunk at once after the fragment load
        pp, rf = (prof.data_ptr() if prof is not None else 0), (int(ent_tt.get("k3_u_late", 0)) if upd else int(bool(residual)))
        if ent[7] != pp:
            pack.set_ptr(0, pp, field=11); ent[7] = pp
        if ent[6] != rf:
            pack.set_i32(0, rf, field=16); ent[6] = rf
        k.launch_packed(grid, BLOCK, pack, smem, self.stream_handle() if stream is None else stream)
        return out
