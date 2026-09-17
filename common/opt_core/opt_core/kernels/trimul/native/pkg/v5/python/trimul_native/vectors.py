# SPDX-License-Identifier: Apache-2.0
"""trimul_native.vectors -- the package's test vectors: closed-form inputs, expected output digests per device class, and the byte gate.

    python -m trimul_native.vectors make   [--out <tree>/testvectors] [--grid r2|r1|smoke] [--no-ref] [--merge]     # on a device of each class
    python -m trimul_native.vectors replay [--dir <tree>/testvectors] [--which gate|all|<id>[,<id>..]] [--ignore-build]
    python -m trimul_native.vectors list   [--dir ...]

Inputs are CLOSED FORMS in integer arithmetic (recipe ``closed_form_v1`` below): every value is an integer hash of its indices scaled by a
power of two, exactly representable in the target dtype, so the tensors are bit-identical on any device and framework version (no random
generator, no transcendental function).  Their sha256 digests are recorded anyway and checked first at replay, so a generator drift can never
be mistaken for a kernel difference.  Expected outputs are recorded per DEVICE CLASS (sm90 = cc 9.0 cubins, sm80 = cc 8.x cubins) as the
sha256 of the output bytes plus 16 row-slab digests (to localise a difference) and, for the small witness cases, the full output tensor.
``make`` also records informational numerics per case (max-abs / rel-RMS against an fp64 statement of the module math and, when the
reference library is importable, max-abs and the bitwise flag against cuequivariance_torch ``triangle_multiplicative_update``).

The gate (``replay --which gate``, and ``face.check``) replays the cases flagged ``gate`` through ``face.serve`` and compares bytes; ``all``
replays everything.  A vector set records the cubin digests it was made from; replaying it against other cubins is refused as stale unless
``--ignore-build`` (development).

Recipe closed_form_v1 (all integer operations on int64, M = 2^32 - 1; mix(a): a &= M; three rounds of (a ^= a >> 16; a = (a * 0x45d9f3b) & M);
a ^= a >> 16):
    h(i, j, c, salt) = mix((i * 2654435 + j * 40503 + c * 97 + salt * 7919) & M)
    bf16 z[i,j,c]    = (h(i,j,c,1) mod (2 r + 1) - r) / 128,  r = 100 + (7 i + 3 j) mod 155                       (|numerator| <= 254: exact in bf16)
    fp32 z[i,j,c]    = bf16 value + (h(i,j,c,2) mod 129 - 64) * 2^-16                                            (exact in fp32, not in bf16)
    batched z[b]     = the recipe with salt + 16 b
    mask tail3[i,j]  = 0 if i >= N-3 or j >= N-3 else 1 (fp32);  mask none = no mask tensor
    w[o,k] (w_ag 11, w_ap 12, w_bg 13, w_bp 14, w_o 15, w_og 16 = salt s) = (h(o,k,0,s) mod 255 - 127) / 2048     (bf16 for bf16 z, fp32 for fp32 z; exact in both)
    ln_in_w[c] = 1 + (h(c,0,0,21) mod 65 - 32) / 256;  ln_in_b[c] = (h(c,0,0,22) mod 65 - 32) / 512;  ln_out_w / ln_out_b the same with salts 23 / 24 (same dtype rule)
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_DIR = os.path.join(ROOT, "testvectors")
RECIPE = "closed_form_v1"
SCHEMA = 1
EPS = 1e-5

__all__ = ["VectorsError", "cases", "closed_form_inputs", "make", "replay", "load_manifest", "RECIPE"]


class VectorsError(RuntimeError):
    """``.kind``: absent | corrupt:<what> | stale:<unit/arch> | no_class:<cls> | inputs_digest:<case>."""

    def __init__(self, kind, detail=""):
        RuntimeError.__init__(self, kind + ((" (" + detail + ")") if detail else ""))
        self.kind = kind


# ---------------------------------------------------------------------------------------------------------------- case grid
def _cid(cz, ch, n, direction, form, residual, mask, variant, batch=1):
    return "c%dx%d_n%d%s_%s_%s_res%d_m%s_%s" % (cz, ch, n, ("_b%d" % batch) if batch > 1 else "", "out" if direction == "outgoing" else "in",
                                                form, int(residual), mask, variant)


def cases(grid="r2"):
    """The case list of a grid: r1 = c128 + c256 x N {256, 400, 403} x out/in x {bf16, f32z} x residual x mask {none, tail3} x {fast, exact(bf16)}
    + witnesses + one batched case; r2 = r1 + (64,64), (64,128) x N {256, 403} x out/in x {bf16 fast+exact, f32z fast} x residual x mask tail3
    + (384,384) x N {256, 403} x out/in x bf16 {fast, exact} x residual x tail3 + three off-diagonal pairs + a (64,128) witness (r1 case ids and bytes
    are a subset of r2); smoke = a handful."""
    out = []

    def add(cz, ch, n, direction, form, residual, mask, variant, batch=1, gate=False, witness=False):
        out.append({"id": _cid(cz, ch, n, direction, form, residual, mask, variant, batch), "c_z": cz, "c_hidden": ch, "n": n, "batch": batch,
                    "direction": direction, "form": form, "residual": bool(residual), "mask": mask, "variant": variant, "gate": bool(gate), "witness": bool(witness)})
    widths = [(128, 128), (256, 256)]
    if grid == "smoke":
        add(128, 128, 256, "outgoing", "bf16", 0, "none", "fast", gate=True)
        add(128, 128, 403, "incoming", "bf16", 1, "tail3", "exact", gate=True)
        add(256, 256, 403, "outgoing", "f32z", 1, "tail3", "fast", gate=True)
        add(128, 128, 32, "outgoing", "bf16", 1, "tail3", "fast", gate=True, witness=True)
        return out
    for cz, ch in widths:
        for n in (256, 400, 403):
            for direction in ("outgoing", "incoming"):
                for form in ("bf16", "f32z"):
                    for residual in (0, 1):
                        for mask in ("none", "tail3"):
                            for variant in (("fast", "exact") if form == "bf16" else ("fast",)):
                                gate = (n == 403 and mask == "tail3" and residual == 1) or (n == 256 and mask == "none" and residual == 0 and form == "bf16" and variant == "fast")
                                add(cz, ch, n, direction, form, residual, mask, variant, gate=gate)
    # witnesses: small cases whose full expected tensors are carried (a byte difference is then reported with its location and magnitude)
    add(128, 128, 32, "outgoing", "bf16", 1, "tail3", "fast", gate=True, witness=True)
    add(128, 128, 32, "incoming", "bf16", 0, "none", "exact", gate=True, witness=True)
    add(256, 256, 32, "outgoing", "f32z", 1, "tail3", "fast", gate=True, witness=True)
    # batched call (mask [B, N, N])
    add(128, 128, 256, "outgoing", "bf16", 0, "tail3", "fast", batch=2, gate=True)
    if grid == "r1":
        return out
    # r2: the 64-wide pairs (bf16 fast + exact, fp32-resident fast) and (384, 384) (bf16 fast + exact; fp32-resident z at 384-wide pairs is refused by name)
    for cz, ch, forms in ((64, 64, ("bf16", "f32z")), (64, 128, ("bf16", "f32z")), (384, 384, ("bf16",))):
        for n in (256, 403):
            for direction in ("outgoing", "incoming"):
                for form in forms:
                    for residual in (0, 1):
                        for variant in (("fast", "exact") if form == "bf16" else ("fast",)):
                            gate = (n == 403 and residual == 1 and direction == "outgoing") or (n == 403 and residual == 1 and variant == "exact")
                            add(cz, ch, n, direction, form, residual, "tail3", variant, gate=gate)
    # off-diagonal pairs (one case each, in the gate) and a small mixed-width witness
    add(128, 64, 256, "outgoing", "bf16", 1, "tail3", "fast", gate=True)
    add(256, 128, 403, "incoming", "bf16", 0, "tail3", "exact", gate=True)
    add(384, 128, 256, "outgoing", "bf16", 1, "tail3", "fast", gate=True)
    add(64, 128, 32, "outgoing", "bf16", 1, "tail3", "fast", gate=True, witness=True)
    return out


# ---------------------------------------------------------------------------------------------------------------- closed forms
_M32 = 0xFFFFFFFF


def _mix(a):
    a = a & _M32
    for _ in range(3):
        a = a ^ (a >> 16)
        a = (a * 0x45D9F3B) & _M32
    return a ^ (a >> 16)


def _h(torch, i, j, c, salt, device):
    """h(i, j, c, salt) over broadcast index tensors (int64)."""
    return _mix((i * 2654435 + j * 40503 + c * 97 + salt * 7919) & _M32)


def closed_form_inputs(case, device, weight_salt=0):
    """-> (z, mask or None, weights dict of the ten fp32 tensors) on ``device``, bit-reproducible (recipe in the module docstring).
    ``weight_salt`` k > 0 derives the k-th alternative weight set (salts shifted by 1000·k; the recorded vectors use k = 0)."""
    import torch
    cz, ch, n, B = case["c_z"], case["c_hidden"], case["n"], case.get("batch", 1)
    dev = torch.device(device)
    i = torch.arange(n, dtype=torch.int64, device=dev).view(n, 1, 1)
    j = torch.arange(n, dtype=torch.int64, device=dev).view(1, n, 1)
    c = torch.arange(cz, dtype=torch.int64, device=dev).view(1, 1, cz)
    zs = []
    for b in range(B):
        r = 100 + (7 * i + 3 * j) % 155                                       # [n, n, 1]
        k = _h(torch, i, j, c, 1 + 16 * b, dev) % (2 * r + 1) - r              # |k| <= 254
        zb = k.to(torch.float32) / 128.0
        if case["form"] == "f32z":
            k2 = _h(torch, i, j, c, 2 + 16 * b, dev) % 129 - 64
            zb = zb + k2.to(torch.float32) * (2.0 ** -16)
        else:
            zb = zb.to(torch.bfloat16)
        zs.append(zb)
    z = zs[0] if B == 1 else torch.stack(zs, 0)
    mask = None
    if case["mask"] == "tail3":
        m = torch.ones(n, n, dtype=torch.float32, device=dev)
        m[n - 3:, :] = 0
        m[:, n - 3:] = 0
        mask = m if B == 1 else m.unsqueeze(0).expand(B, n, n).contiguous()
    elif case["mask"] != "none":
        raise VectorsError("corrupt:mask=%s" % case["mask"])

    def mat(rows, cols, salt):
        o = torch.arange(rows, dtype=torch.int64, device=dev).view(rows, 1)
        q = torch.arange(cols, dtype=torch.int64, device=dev).view(1, cols)
        return ((_h(torch, o, q, 0, salt, dev) % 255) - 127).to(torch.float32) / 2048.0

    def vec(length, salt, base, scale):
        o = torch.arange(length, dtype=torch.int64, device=dev)
        return base + ((_h(torch, o, 0, 0, salt, dev) % 65) - 32).to(torch.float32) / scale
    ws = 1000 * int(weight_salt)
    w = {"ln_in_w": vec(cz, 21 + ws, 1.0, 256.0), "ln_in_b": vec(cz, 22 + ws, 0.0, 512.0),
         "w_ag": mat(ch, cz, 11 + ws), "w_ap": mat(ch, cz, 12 + ws), "w_bg": mat(ch, cz, 13 + ws), "w_bp": mat(ch, cz, 14 + ws),
         "ln_out_w": vec(ch, 23 + ws, 1.0, 256.0), "ln_out_b": vec(ch, 24 + ws, 0.0, 512.0),
         "w_o": mat(cz, ch, 15 + ws), "w_og": mat(cz, cz, 16 + ws)}
    if case["form"] == "bf16":                                              # a bf16 module holds bf16 parameters (every value above is exact in bf16)
        w = {k: v.to(torch.bfloat16) for k, v in w.items()}
    return z, mask, w


def tensor_sha256(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _slab_digests(t, nslab=16):
    import torch
    t2 = t.detach().contiguous().reshape(-1, *t.shape[-2:]) if t.dim() >= 3 else t.detach().contiguous()
    rows = t.reshape(-1, t.shape[-1]) if t.dim() > 1 else t.reshape(1, -1)
    n = rows.shape[0]
    out = []
    for s in range(nslab):
        a, b = (n * s) // nslab, (n * (s + 1)) // nslab
        out.append(hashlib.sha256(rows[a:b].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16] if b > a else "")
    return out


def input_digests(z, mask, w):
    return {"z": tensor_sha256(z), "mask": tensor_sha256(mask) if mask is not None else None, "weights": hashlib.sha256("".join(tensor_sha256(w[k]) for k in sorted(w)).encode()).hexdigest()}


# ---------------------------------------------------------------------------------------------------------------- references (make only)
def fp64_reference(z, mask, w, direction, residual, eps=EPS):
    """The module math in fp64 (the op statement of face.py), on z's device."""
    import torch
    z4 = z if z.dim() == 4 else z.unsqueeze(0)
    outs = []
    for b in range(z4.shape[0]):
        zd = z4[b].double()
        md = None
        if mask is not None:
            mm = mask if mask.dim() == 2 else mask[b if mask.shape[0] == z4.shape[0] else 0]
            md = mm.double().unsqueeze(-1)
        x = torch.nn.functional.layer_norm(zd, (zd.shape[-1],), w["ln_in_w"].double(), w["ln_in_b"].double(), eps)
        a = torch.sigmoid(x @ w["w_ag"].double().t()) * (x @ w["w_ap"].double().t())
        bb = torch.sigmoid(x @ w["w_bg"].double().t()) * (x @ w["w_bp"].double().t())
        if md is not None:
            a = a * md
            bb = bb * md
        if direction == "outgoing":
            X = torch.einsum("ikc,jkc->ijc", a, bb)
        else:
            X = torch.einsum("kic,kjc->ijc", a, bb)
        X = torch.nn.functional.layer_norm(X, (X.shape[-1],), w["ln_out_w"].double(), w["ln_out_b"].double(), eps)
        upd = torch.sigmoid(x @ w["w_og"].double().t()) * (X @ w["w_o"].double().t())
        outs.append(zd + upd if residual else upd)
        del x, a, bb, X, upd
    out = torch.stack(outs, 0)
    return out if z.dim() == 4 else out[0]


def _cueq_fn():
    try:
        import cuequivariance_torch as cqt
        fn = getattr(cqt, "triangle_multiplicative_update", None)
        if fn is None:
            from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update as fn
        ver = getattr(cqt, "__version__", "?")
        return fn, "cuequivariance_torch %s" % ver
    except Exception:                                                       # noqa: BLE001
        return None, None


CUEQ_MIN_TOKENS = 101                   # below this the reference library (0.11.x) leaves its fused path (other arithmetic, fp32-only operands)


def cueq_incomparable(case):
    """None when the reference library's fused triangle_multiplicative_update defines comparison bytes for ``case``; else the reason word:
    its fused path requires c_hidden == c_z, a single [N, N, c] problem (no batch axis here) and at least CUEQ_MIN_TOKENS tokens."""
    if int(case.get("batch", 1)) != 1:
        return "batched_case"
    if int(case["c_z"]) != int(case["c_hidden"]):
        return "reference_requires_c_hidden==c_z"
    if int(case["n"]) < CUEQ_MIN_TOKENS:
        return "reference_path_differs_below_%d_tokens" % CUEQ_MIN_TOKENS
    return None


def cueq_reference(fn, z, mask, w, direction, residual, form, eps=EPS):
    import torch
    kw = dict(norm_in_weight=w["ln_in_w"], norm_in_bias=w["ln_in_b"], p_in_weight=torch.cat([w["w_ap"], w["w_bp"]], 0).contiguous(),
              g_in_weight=torch.cat([w["w_ag"], w["w_bg"]], 0).contiguous(), norm_out_weight=w["ln_out_w"], norm_out_bias=w["ln_out_b"],
              p_out_weight=w["w_o"], g_out_weight=w["w_og"])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(form == "f32z")):
        upd = fn(z, direction=direction, mask=mask, eps=eps, **kw)
    return z + upd if residual else upd


def _numerics(out, ref64, cq):
    import torch
    o = out.double()
    d = (o - ref64)
    rec = {"max_abs_fp64": float(d.abs().max()), "rel_rms_fp64": float((d.pow(2).mean().sqrt() / ref64.pow(2).mean().sqrt()).item())}
    if cq is not None:
        if isinstance(cq, str):
            rec["cueq"] = cq
        else:
            cqd = cq.to(out.dtype) if cq.dtype != out.dtype else cq
            rec["bitwise_cueq"] = bool(torch.equal(cqd, out))
            rec["max_abs_cueq"] = float((out.double() - cq.double()).abs().max())
    return rec


# ---------------------------------------------------------------------------------------------------------------- manifest io
def load_manifest(tv_dir=None):
    p = os.path.join(tv_dir or DEFAULT_DIR, "manifest.json")
    if not os.path.isfile(p):
        raise VectorsError("absent", p)
    try:
        with open(p, encoding="utf-8") as f:
            man = json.load(f)
    except ValueError as e:
        raise VectorsError("corrupt:manifest.json", str(e)[:200])
    if man.get("recipe") != RECIPE or man.get("schema") != SCHEMA:
        raise VectorsError("corrupt:recipe=%s,schema=%s" % (man.get("recipe"), man.get("schema")))
    return man


def _build_digests(arch=None):
    from . import face as F
    man = F._read_manifest() or {}
    out = {}
    for key, ent in (man.get("units") or {}).items():
        if ent.get("dev"):
            continue
        if arch is None or ent.get("arch") == arch:
            out[key] = ent.get("cubin_sha256")
    return out


def _serve_case(F, case, z, mask, w, cache):
    import torch
    cfg = {"variant": case["variant"], "gate": False, "f32_resident_ok": True}
    cfg["cells"] = False                                                    # byte gates never consult the measured-cell table
    return F.serve(z, mask, direction=case["direction"], weights=w, residual=case["residual"], cache=cache, eps=EPS, config=cfg)


# ---------------------------------------------------------------------------------------------------------------- make
SANITY = {"max_abs_fp64": 0.08, "rel_rms_fp64": 0.02}              # a case whose output is outside the numerics class is never recorded as expected


def make(out_dir=None, grid="r2", with_ref=True, merge=False, device="cuda", quiet=False, exclude=None, isolated=True):
    """``exclude``: [(glob over case ids, reason)] -- cases recorded as unserved ("excluded:<reason>") instead of being run."""
    import fnmatch
    import torch
    from . import face as F
    from . import launch as L
    out_dir = out_dir or DEFAULT_DIR
    os.makedirs(os.path.join(out_dir, "witness"), exist_ok=True)
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    arch, cc = F._arch(idx)
    cls = F.device_class(arch)
    rep = F.check(idx, gate=False)
    fn_cq, cq_ver = _cueq_fn() if with_ref else (None, None)
    existing = None
    if merge:
        existing = load_manifest(out_dir)
    case_list = cases(grid)
    man = {"schema": SCHEMA, "package": "trimul_native", "version": F.VERSION, "recipe": RECIPE, "grid": grid,
           "made_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "classes": {}, "build_units": {}, "cases": []}
    if existing:
        man["classes"] = dict(existing.get("classes") or {})
        man["build_units"] = dict(existing.get("build_units") or {})
        prev = {c["id"]: c for c in existing.get("cases") or []}
    else:
        prev = {}
    man["classes"][cls] = {"device": torch.cuda.get_device_name(idx), "cc": "%d.%d" % cc, "torch": torch.__version__, "binding": rep["binding"],
                          "driver_version": rep["driver_version"], "reference": cq_ver, "made_utc": man["made_utc"]}
    man["build_units"].update(_build_digests(arch))
    cache = {}
    t0 = time.time()
    nbit = 0
    for ci, case in enumerate(case_list):
        word = F.admits(cc, "bf16", case["c_z"], case["c_hidden"], case["n"], case["direction"], residency=("fp32" if case["form"] == "f32z" else "bf16"),
                        residual=case["residual"], batch=case["batch"], variant=case["variant"], cells=False)
        rec = dict(prev.get(case["id"]) or case)
        rec.update({k: case[k] for k in case})
        rec.setdefault("expected", {})
        for pat, reason in (exclude or []):
            if fnmatch.fnmatch(case["id"], pat):
                word = "excluded:%s" % reason
                break
        if word is not None:
            rec["expected"][cls] = {"unserved": word}
            man["cases"].append(rec)
            if not quiet:
                print("[vectors] %3d/%d %-44s unserved on %s: %s" % (ci + 1, len(case_list), case["id"], cls, word))
            continue
        z, mask, w = closed_form_inputs(case, dev)
        dig = input_digests(z, mask, w)
        if "inputs_sha256" in rec and rec["inputs_sha256"] != dig:
            raise VectorsError("inputs_digest:%s" % case["id"], "the closed-form inputs differ from the recorded digests (recipe drift)")
        rec["inputs_sha256"] = dig
        try:
            out = _serve_case(F, case, z, mask, w, cache)
        except F.Refusal as e:                                              # a serve-time refusal is a named 'unserved', not a failure of the make
            rec["expected"][cls] = {"unserved": e.kind}
            man["cases"].append(rec)
            if not quiet:
                print("[vectors] %3d/%d %-44s unserved on %s at serve: %s" % (ci + 1, len(case_list), case["id"], cls, e.kind))
            continue
        torch.cuda.synchronize(dev)
        out2 = _serve_case(F, case, z, mask, w, cache)                        # determinism: a second call must give the same bytes
        torch.cuda.synchronize(dev)
        if not torch.equal(out, out2):
            raise VectorsError("corrupt:nondeterministic:%s" % case["id"], "two serves of the same inputs differ")
        exp = {"sha256": tensor_sha256(out), "slabs": _slab_digests(out), "shape": list(out.shape), "dtype": str(out.dtype).replace("torch.", "")}
        if with_ref:
            ref = fp64_reference(z, mask, w, case["direction"], case["residual"])
            cq = None
            if fn_cq is not None:
                why = cueq_incomparable(case)
                if why is not None:
                    cq = "skipped:" + why                                    # recorded, not an error: the reference library does not define these bytes
                else:
                    try:
                        cq = cueq_reference(fn_cq, z, mask, w, case["direction"], case["residual"], case["form"])
                    except Exception as e:                                   # noqa: BLE001  an unexpected reference-library failure is recorded by name
                        cq = "error:%s" % ((str(e).strip().splitlines() or [type(e).__name__])[0][:120])
            exp["numerics"] = _numerics(out, ref, cq)
            nbit += int(bool(exp["numerics"].get("bitwise_cueq")))
            for k, bound in SANITY.items():
                if not (exp["numerics"][k] <= bound):
                    raise VectorsError("corrupt:numerics:%s" % case["id"], "%s = %.4g exceeds %.3g -- refusing to record these bytes as expected" % (k, exp["numerics"][k], bound))
            del ref, cq
        if case["witness"]:
            wp = os.path.join("witness", "%s.%s.pt" % (case["id"], cls))
            torch.save(out.detach().cpu(), os.path.join(out_dir, wp))
            exp["witness"] = wp
        rec["expected"][cls] = exp
        man["cases"].append(rec)
        if not quiet:
            nm = exp.get("numerics", {})
            print("[vectors] %3d/%d %-44s %s %s maxabs64=%.3g relrms64=%.2g%s" % (
                ci + 1, len(case_list), case["id"], cls, exp["sha256"][:12], nm.get("max_abs_fp64", float("nan")), nm.get("rel_rms_fp64", float("nan")),
                ("  cueq: bitwise" if nm.get("bitwise_cueq") else ("  cueq maxabs=%.3g" % nm["max_abs_cueq"] if "max_abs_cueq" in nm else ("  " + nm["cueq"] if "cueq" in nm else "")))))
        del z, mask, w, out, out2
    man["seconds_" + cls] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write("\n")
    if isolated:                                                            # process-history independence: the gate cases once more in a FRESH
        import subprocess                                                   # process, reverse order, a fresh cache per case; bytes must equal
        cmd = [sys.executable, "-m", "trimul_native.vectors", "replay", "--dir", out_dir, "--which", "gate", "--reverse", "--fresh-cache"]
        r = subprocess.run(cmd, cwd=os.path.dirname(HERE), env=dict(os.environ, PYTHONPATH=os.path.dirname(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")),
                           capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [""])[-1]
        man["isolated_replay"] = {"status": "pass" if r.returncode == 0 else "FAIL", "detail": tail[-300:], "mode": "fresh process, gate cases, reversed, fresh cache per case"}
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
        if not quiet:
            print("[vectors] isolated gate replay (fresh process, reversed, fresh caches): %s %s" % (man["isolated_replay"]["status"], tail[-200:]))
        if r.returncode != 0:
            raise VectorsError("order_dependent:" + (tail.split("failed:")[-1].split()[0] if "failed:" in tail else "gate"),
                               "a gate case served in a fresh process (reverse order, fresh caches) does not reproduce the bytes recorded by the making process -- "
                               "the op's output depends on process history; the set is written but flagged isolated_replay=FAIL and cannot be sealed")
    served = [c for c in man["cases"] if "sha256" in (c["expected"].get(cls) or {})]
    print("[vectors] made %d cases (%d served on %s, %d bitwise == reference library) in %.0fs -> %s" % (len(man["cases"]), len(served), cls, nbit, time.time() - t0, out_dir))
    return man


# ---------------------------------------------------------------------------------------------------------------- replay (the byte gate)
MULTIWEIGHT_CASES = ("c128x128_n256_out_bf16_res1_mtail3_fast", "c128x128_n403_in_f32z_res0_mtail3_fast",
                     "c256x256_n403_out_f32z_res1_mtail3_fast", "c256x256_n400_in_bf16_res0_mnone_fast")


def multiweight_gate(F, man, cls, cache, quiet=False):
    """Process-history independence through the face, part of every gate replay: for each MULTIWEIGHT_CASES entry this device class serves,
    three weight sets (the recorded recipe + two alternatives) are served interleaved and repeated through ONE cache dict; every result must
    equal the same call through a FRESH cache, the recorded set must reproduce its expected bytes before and after the others, and the last
    call is repeated after the caching allocator has been handed (and has released) 256 MiB of NaN.  -> list of failure dicts."""
    import torch
    failed = []
    exp_of = {c["id"]: c for c in man["cases"]}
    for cid in MULTIWEIGHT_CASES:
        case = exp_of.get(cid)
        if case is None or "sha256" not in ((case.get("expected") or {}).get(cls) or {}):
            continue
        want = case["expected"][cls]["sha256"]
        z, mask, w0 = closed_form_inputs(case, "cuda")
        sets = [w0] + [closed_form_inputs(case, "cuda", weight_salt=k)[2] for k in (1, 2)]
        fresh = [tensor_sha256(_serve_case(F, case, z, mask, wk, {})) for wk in sets]
        if fresh[0] != want:
            failed.append({"id": "multiweight:%s" % cid, "why": "fresh-cache serve %s.. != expected %s.." % (fresh[0][:12], want[:12])}); continue
        if len(set(fresh)) != 3:
            failed.append({"id": "multiweight:%s" % cid, "why": "alternative weight sets did not change the output (recipe degenerate)"}); continue
        seq = [0, 1, 2, 1, 0, 2, 2, 0]
        got = [tensor_sha256(_serve_case(F, case, z, mask, sets[k], cache)) for k in seq]
        bad = [(i, k) for i, k in enumerate(seq) if got[i] != fresh[k]]
        if bad:
            i, k = bad[0]
            failed.append({"id": "multiweight:%s" % cid, "why": "call %d (weight set %d) through the shared cache gave %s.., a fresh cache gives %s.." % (i, k, got[i][:12], fresh[k][:12])}); continue
        junk = torch.full((256 * 2**20 // 4,), float("nan"), dtype=torch.float32, device="cuda")
        del junk
        after = tensor_sha256(_serve_case(F, case, z, mask, sets[0], cache))
        if after != want:
            failed.append({"id": "multiweight:%s" % cid, "why": "after allocator churn %s.. != expected %s.." % (after[:12], want[:12])}); continue
        if not quiet:
            print("[vectors] multiweight %s: 3 weight sets x 8 interleaved calls through one cache == fresh-cache bytes; allocator churn neutral" % cid)
    if not quiet and failed:
        for f in failed:
            print("[vectors] FAIL %-44s %s" % (f["id"], f["why"]))
    return failed


def replay(tv_dir=None, which="gate", device_index=None, device_class=None, raise_on_fail=True, quiet=False, ignore_build=False, reverse=False, fresh_cache=False):
    """Replay cases through face.serve and compare bytes.  -> {"n", "passed", "failed": [{"id", "why"}], "skipped", "seconds"}."""
    import torch
    from . import face as F
    tv_dir = tv_dir or DEFAULT_DIR
    man = load_manifest(tv_dir)
    idx = device_index if device_index is not None else torch.cuda.current_device()
    dev = torch.device("cuda", idx)
    arch, cc = F._arch(idx)
    cls = device_class or F.device_class(arch)
    if cls not in (man.get("classes") or {}):
        raise VectorsError("no_class:%s" % cls, "the vector set holds no expected bytes for device class %s" % cls)
    if not ignore_build:
        now = _build_digests(arch)
        for key, dig in now.items():
            rec = (man.get("build_units") or {}).get(key)
            if rec is not None and rec != dig:
                raise VectorsError("stale:%s" % key, "vectors made from cubin %s.., build holds %s.." % (rec[:12], dig[:12]))
    sel = []
    ids = None if which in ("gate", "all") else set(w.strip() for w in which.split(",") if w.strip())
    for case in man["cases"]:
        exp = (case.get("expected") or {}).get(cls)
        if ids is not None:
            if case["id"] in ids:
                sel.append(case)
        elif which == "all" or case.get("gate"):
            sel.append(case)
    if ids is not None and len(sel) != len(ids):
        raise VectorsError("corrupt:unknown_case:%s" % ",".join(sorted(ids - {c["id"] for c in sel})))
    cache = {}
    passed, failed, skipped = [], [], []
    t0 = time.time()
    if reverse:
        sel = list(reversed(sel))
    for case in sel:
        exp = (case.get("expected") or {}).get(cls) or {}
        if "sha256" not in exp:
            skipped.append({"id": case["id"], "why": exp.get("unserved", "no expected bytes for %s" % cls)})
            continue
        try:
            z, mask, w = closed_form_inputs(case, dev)
            dig = input_digests(z, mask, w)
            if dig != case["inputs_sha256"]:
                raise VectorsError("inputs_digest:%s" % case["id"], "regenerated inputs differ from the recorded digests")
            out = _serve_case(F, case, z, mask, w, {} if fresh_cache else cache)
            torch.cuda.synchronize(dev)
            got = tensor_sha256(out)
        except VectorsError:
            raise
        except F.Refusal as e:
            failed.append({"id": case["id"], "why": "refused:%s" % e.kind})
            if not quiet:
                print("[vectors] FAIL %-44s refused: %s" % (case["id"], e.kind))
            continue
        if got == exp["sha256"]:
            passed.append(case["id"])
            if not quiet:
                print("[vectors] ok   %-44s %s" % (case["id"], got[:12]))
        else:
            why = "sha256 %s.. != expected %s.." % (got[:12], exp["sha256"][:12])
            slabs = _slab_digests(out)
            diff = [k for k, (a, b) in enumerate(zip(slabs, exp.get("slabs") or [])) if a != b]
            if diff:
                why += "; differing row slabs %s of 16" % diff[:8]
            if exp.get("witness"):
                try:
                    ref = torch.load(os.path.join(tv_dir, exp["witness"]), map_location="cpu").to(dev)
                    d = (out.float() - ref.float()).abs()
                    nz = torch.nonzero(out.view(torch.int16 if out.dtype == torch.bfloat16 else torch.int32) != ref.view(torch.int16 if ref.dtype == torch.bfloat16 else torch.int32))
                    why += "; vs witness: max_abs %.4g, %d differing elements, first at %s" % (float(d.max()), int(nz.shape[0]), nz[0].tolist() if nz.shape[0] else None)
                except Exception as e:                                       # noqa: BLE001
                    why += "; witness unreadable (%s)" % str(e)[:80]
            failed.append({"id": case["id"], "why": why})
            if not quiet:
                print("[vectors] FAIL %-44s %s" % (case["id"], why))
        del out
    if which in ("gate", "all") and sel:                                    # weight-set changes through one cache + allocator history
        failed += multiweight_gate(F, man, cls, cache, quiet=quiet)
    res = {"class": cls, "n": len(sel), "passed": len(passed), "failed": failed, "skipped": skipped, "seconds": round(time.time() - t0, 2)}
    if not quiet:
        print("[vectors] replay %s on %s: %d passed, %d failed, %d skipped in %.1fs" % (which, cls, len(passed), len(failed), len(skipped), res["seconds"]))
    if failed and raise_on_fail:
        raise VectorsError("failed:%s" % failed[0]["id"], failed[0]["why"])
    return res


# ---------------------------------------------------------------------------------------------------------------- cli
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="verb", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", default=DEFAULT_DIR)
    m.add_argument("--grid", default="r2")
    m.add_argument("--no-ref", action="store_true")
    m.add_argument("--merge", action="store_true", help="add this device class's expected bytes to an existing set")
    m.add_argument("--device", default="cuda")
    m.add_argument("--no-isolated", action="store_true", help="development only: skip the isolated gate replay (fresh process, reversed order, fresh caches) after making")
    m.add_argument("--exclude", action="append", default=[], metavar="GLOB=REASON", help="development only: record matching cases as unserved (excluded:REASON) instead of running them; the sealing tool refuses such a set")
    r = sub.add_parser("replay")
    r.add_argument("--dir", default=DEFAULT_DIR)
    r.add_argument("--which", default="gate")
    r.add_argument("--ignore-build", action="store_true")
    r.add_argument("--reverse", action="store_true", help="replay the selection in reverse order")
    r.add_argument("--fresh-cache", action="store_true", help="a fresh cache dict per case (no workspace / plan reuse across cases)")
    ls = sub.add_parser("list")
    ls.add_argument("--dir", default=DEFAULT_DIR)
    ls.add_argument("--grid", default=None, help="list a grid's case ids instead of a made set")
    a = ap.parse_args(argv)
    if a.verb == "make":
        excl = [tuple(e.split("=", 1)) if "=" in e else (e, "excluded") for e in a.exclude]
        make(a.out, a.grid, with_ref=not a.no_ref, merge=a.merge, device=a.device, exclude=excl, isolated=not a.no_isolated)
        return 0
    if a.verb == "replay":
        try:
            res = replay(a.dir, a.which, raise_on_fail=False, ignore_build=a.ignore_build, reverse=a.reverse, fresh_cache=a.fresh_cache)
        except VectorsError as e:
            print("[vectors] %s" % e)
            return 2
        return 0 if not res["failed"] else 1
    if a.verb == "list":
        if a.grid:
            for c in cases(a.grid):
                print("%-46s%s%s" % (c["id"], " gate" if c["gate"] else "", " witness" if c["witness"] else ""))
            return 0
        man = load_manifest(a.dir)
        print("recipe %s, version %s, made %s, classes %s, %d cases" % (man["recipe"], man["version"], man["made_utc"], sorted(man["classes"]), len(man["cases"])))
        for c in man["cases"]:
            ex = "; ".join("%s:%s" % (k, (v.get("sha256", "")[:12] or v.get("unserved", "?"))) for k, v in sorted((c.get("expected") or {}).items()))
            print("%-46s%s %s" % (c["id"], " gate" if c.get("gate") else "     ", ex))
        return 0


if __name__ == "__main__":
    sys.exit(main())
