"""zbf16rows -- keep the sharded pair state bf16 past the trunk, instead of one whole fp32 plane.

The object.  `esmfold2_opt/rowpair.py:1161`, in `_forward_rows`, right after the coda:

    z = self.parcae_coda(z, pair_attention_mask=pair_mask)
    z = z.float()                       # <-- a WHOLE fp32 [B, R, N, c_z] plane

`z` leaves the trunk in bf16 (the kit's own P=2 OOM at 6912 asks 11.39 GiB, which is bf16
[1, 3456, 6912, 256]).  This statement doubles it and then holds it live across the distogram, the whole
sampler roll-out AND the confidence head.  At N=7936, P=2 (R=3968, c_z=256) that plane is **30.03 GiB per
rank**, held longer than any other pair-sized object of the fold.

There is a second half, and it is pure waste.  Line 1176 then passes `z=z.detach().float()` into the
confidence head, which with a bf16 `z` would re-materialise the very plane this lever removes -- while
`confrows.conf_prologue_rows` already does its own PER-ROW-BLOCK `.float()` before `z_norm`
(`head.z_norm(z[..., s:e, :, :].float())`).  So the whole-plane cast at the call site is redundant even
in the kit's own arrangement.  Both anchors are replaced together; replacing only one is worthless
(`.float()` on an fp32 tensor returns self, so the call-site edit alone is a no-op).

NUMERICS.  This is a numerics change of the same class as `confbf16`, NOT a memory-only lever: the
distogram head, the diffusion conditioning and the confidence prologue all read a bf16 pair state instead
of an fp32 one.  Do not assert equivalence and do not quote a ceiling won with it without saying so.

REFUSES BY NAME unless `confrows` is installed, because the stock sharded confidence statement does
`z_base = head.z_norm(z)` on the whole plane and would simply promote it again.

Two bf16-pair traps elsewhere in the head do NOT bite here, and it is worth writing down why:
 (a) `FoldingTrunk.forward` (modeling_esmfold2_common.py:2709) casts its input to bf16 when the fused
     trimul backend is on and casts back at the end, so a bf16 input makes it skip BOTH casts. That
     matters for a tensor fed to a FoldingTrunk. `z` past line 1161 is never fed to one -- it goes to
     `distogram_rows`, `structure_head.sample(z_trunk=z)` and `confidence_head(z=...)`; the confidence
     trunk is handed `pair`, built by the prologue, not `z`.
 (b) `add_embed_rows_` gates its in-place branch on `result_type(pair, probe) == pair.dtype` and
     silently re-materialises a full fp32 plane when that fails -- again on the confidence head's
     `pair`, not on `z`.
Both are checked against, not assumed: `stats()` reports the dtype actually seen at each consumer.

mode=conf is the default.  mode=full passes short inputs and then FAILS at 8192 tokens (P=2) in the
diffusion module's budget-selected pair-bias path --
`diffusion.py:470 pair_bias_rows_into -> rowpair_heads.py:588 bias -> LayerNorm: expected scalar type
BFloat16 but found Float`.  That path is chosen by `DiffusionSchedule.decide` from the free-byte budget,
so a short smoke never takes it.  mode=full is therefore NOT RECOMMENDED: the phases it additionally
bounds (distogram, sampler) are not the peak, so it buys little over mode=conf and costs the sampler's
whole dtype surface.

Knobs: ``apply(model=, min_N=, mb=)`` — set by ``rowchunk.install`` from EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS and EF2_ROWPAIR_ZBF16_MB;
mode=conf is what the kit installs (``mode`` stays a keyword for the full form); nothing here reads the environment.
"""
import inspect

STATS = {"calls": 0, "bf16": 0, "stock": 0, "N": None, "R": None, "C": None,
         "coda_dtype_in": None, "z_dtype_out": None, "plane_gib_fp32": None, "gib_avoided": None, "mode": None,
         "conf_calls": 0, "conf_blocks": 0, "conf_dtype": None,
         "floor_skips": 0}                                          # mode conf below the token floor (or off the GPU): the call site's own `z.detach().float()`, no copy
_CFG = {"min_N": 1024,                   # below this many tokens the fp32 path runs (rowchunk.install: EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS)
        "mode": "conf",                  # conf: a bf16 COPY for the confidence head only; full: z itself demoted after the coda (not installed by the kit)
        "mb": 1024.0}                    # MiB per row band of the copy (EF2_ROWPAIR_ZBF16_MB)
_ORIG = {}

ANCHOR_FLOAT = "            z = z.float()\n"
REPL_FLOAT = "            z = _RC_Z_KEEP(z)\n"
ANCHOR_CONF = "            s_inputs=x_inputs.detach(), z=z.detach().float(), x_pred=sample_coords.detach(),"
REPL_CONF = "            s_inputs=x_inputs.detach(), z=_RC_Z_CONF(z), x_pred=sample_coords.detach(),"

# --- mode "conf": leave `z = z.float()` alone (z stays fp32 for the distogram and the sampler) and
# convert to bf16 IN BANDS at the confidence call site, rebinding the caller's `z` so the fp32 plane's
# last reference is dropped there.  `z` is dead after this call -- nothing below it reads `z` -- so the
# rebinding is safe.  Lower risk than "full": it touches only the confidence consumer, which
# confrows.conf_prologue_rows already upcasts per row block.
ANCHOR_CONFCALL = """        confidence_output = self.confidence_head(
            s_inputs=x_inputs.detach(), z=z.detach().float(), x_pred=sample_coords.detach(),"""
REPL_CONFCALL = """        z = _RC_Z_CONF(z)
        confidence_output = self.confidence_head(
            s_inputs=x_inputs.detach(), z=z, x_pred=sample_coords.detach(),"""


class ZBf16Refused(RuntimeError):
    pass


def _on_device(t) -> bool:                                          # the bf16 copy serves CUDA tensors; tests rebind this to exercise the copy on CPU
    return bool(getattr(t, "is_cuda", False))


def z_keep(z):
    """`z.float()` -> keep the trunk's dtype when it is already a floating point shard we can use."""
    STATS["calls"] += 1
    R, N, C = int(z.shape[-3]), int(z.shape[-2]), int(z.shape[-1])
    STATS.update(R=R, N=N, C=C, coda_dtype_in=str(z.dtype).replace("torch.", ""))
    STATS["plane_gib_fp32"] = round(R * N * C * 4 / 2 ** 30, 3)
    if N < _CFG["min_N"] or not z.is_cuda or z.dtype not in (__import__("torch").bfloat16,):
        STATS["stock"] += 1
        STATS["gib_avoided"] = 0.0
        out = z.float()
    else:
        STATS["bf16"] += 1
        STATS["gib_avoided"] = round(R * N * C * 2 / 2 ** 30, 3)   # the fp32 plane never allocated
        out = z
    STATS["z_dtype_out"] = str(out.dtype).replace("torch.", "")
    return out


def z_conf(z):
    """mode "full": `z.detach().float()` -> `z.detach()` (z is already bf16; the whole-plane cast at the
    call site is redundant because confrows' prologue upcasts per row block before z_norm).
    mode "conf": z arrives fp32; return a BANDED bf16 copy.  The caller rebinds `z` to it, so the fp32
    plane is freed right here; the transient is fp32 + bf16 for the length of the copy only."""
    import torch
    if _CFG["mode"] != "full" and (int(z.shape[-2]) < int(_CFG["min_N"]) or not _on_device(z)):
        STATS["floor_skips"] += 1                                          # below the floor: exactly the unchunked route's argument expression
        return z.detach().float()
    STATS["conf_calls"] += 1
    if _CFG["mode"] == "full" or z.dtype == torch.bfloat16:
        STATS["conf_dtype"] = str(z.dtype).replace("torch.", "")
        return z.detach()
    R, N, C = int(z.shape[-3]), int(z.shape[-2]), int(z.shape[-1])
    rows = max(1, min(R, int(_CFG["mb"] * (1 << 20)) // max(1, N * C * 4)))
    out = torch.empty(z.shape, dtype=torch.bfloat16, device=z.device)
    for s0 in range(0, R, rows):
        e = min(s0 + rows, R)
        out[..., s0:e, :, :].copy_(z[..., s0:e, :, :])
        STATS["conf_blocks"] += 1
    STATS["conf_dtype"] = "bfloat16"
    STATS["gib_avoided"] = round(R * N * C * 2 / 2 ** 30, 3)
    return out.detach()


def apply(model=None, min_N=None, mode=None, mb=None):
    import sys
    import esmfold2_opt.rowpair as RP
    for k, v in (("min_N", min_N), ("mode", mode), ("mb", mb)):
        if v is not None:
            _CFG[k] = v
    if "fn" in _ORIG:
        return {"already": True}
    from . import confrows as cr
    if not getattr(cr, "_ORIG", None):
        raise ZBf16Refused(
            "zbf16rows: the confrows lever is not installed in this process. The whole-shard confidence "
            "statement does z_norm(z) on the WHOLE plane and would promote z back to fp32, so a bf16 z "
            "would cost memory rather than save it: zbf16 rides on confrows.")
    fn = RP._forward_rows
    src = inspect.getsource(fn)
    for name, anc in ((("z.float()", ANCHOR_FLOAT), ("confidence_head(z=...)", ANCHOR_CONF))
                      if _CFG["mode"] == "full" else ()):
        if src.count(anc) != 1:
            raise ZBf16Refused("zbf16rows: the kit's _forward_rows does not carry the %s anchor exactly "
                               "once (found %d) -- refusing" % (name, src.count(anc)))
    if _CFG["mode"] not in ("full", "conf"):
        raise ZBf16Refused("zbf16rows: mode=%r; expected 'full' or 'conf'" % _CFG["mode"])
    if _CFG["mode"] == "full":
        new_src = src.replace(ANCHOR_FLOAT, REPL_FLOAT).replace(ANCHOR_CONF, REPL_CONF)
    else:
        if src.count(ANCHOR_CONFCALL) != 1:
            raise ZBf16Refused("zbf16rows: the two-line confidence-call anchor occurs %d times; "
                               "expected 1" % src.count(ANCHOR_CONFCALL))
        new_src = src.replace(ANCHOR_CONFCALL, REPL_CONFCALL)
    STATS["mode"] = _CFG["mode"]
    g = RP.__dict__
    g["_RC_Z_KEEP"] = z_keep
    g["_RC_Z_CONF"] = z_conf
    ns = {}
    exec(compile(new_src, "<zbf16rows:%s>" % RP.__file__, "exec"), g, ns)
    new_fn = ns["_forward_rows"]
    new_fn.__zbf16rows__ = True
    _ORIG["fn"] = fn
    RP._forward_rows = new_fn
    inst = bind_instance(model) if model is not None else "not_requested"
    return {"version": "1.0", "rebound": "esmfold2_opt.rowpair._forward_rows", "instance": inst,
            "src_lines": src.count("\n"), "min_N": _CFG["min_N"], "mode": _CFG["mode"]}


def bind_instance(model):
    """install() does `_patch(model, "forward", types.MethodType(_forward_rows, model), "model")`, so the
    FUNCTION OBJECT is in the instance dict: the module rebind above is inert on its own after
    install_rank, exactly as for injrows/_run_one_loop."""
    import types
    import esmfold2_opt.rowpair as RP
    cur = vars(model).get("forward")
    if cur is None:
        return "not_installed_yet:module_attr_only"
    if getattr(getattr(cur, "__func__", None), "__zbf16rows__", False):
        return "already"
    if getattr(cur, "__func__", None) is not _ORIG.get("fn"):
        raise ZBf16Refused("zbf16rows: model.forward is %r, not rowpair._forward_rows -- refusing "
                           "(some other frame owns the rows forward)" % (getattr(cur, "__func__", cur),))
    model.__dict__["forward"] = types.MethodType(RP._forward_rows, model)
    _ORIG["model"] = model
    return "rebound:model.forward"


def unapply():
    import types
    import esmfold2_opt.rowpair as RP
    m = _ORIG.pop("model", None)
    if "fn" in _ORIG:
        fn = _ORIG.pop("fn")
        RP._forward_rows = fn
        if m is not None:
            m.__dict__["forward"] = types.MethodType(fn, m)


def stats():
    return dict(STATS)
