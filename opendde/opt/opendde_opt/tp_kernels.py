"""The row-sharded pair stack's triangle-attention KERNEL (line ``BIG_TP``, ``--mode big --n_gpu P``; lever ``tp_triatt``): the core's
row-block dispatch bound at the sharded schedule's tri-attention seam (``tp._triatt_fns``); the kernel words are the core's own (``KERNEL_WORDS``:
one vocabulary with ``opt_core.mem.rowpair``).

  ``opt_core.mem.rowpair.triatt.attention_core(kernel=KERNEL_WORDS["triatt"])``: the core's triangle-attention kernel dispatch on each row batch; the
  engine's own attention statement (``opendde/model/triangular/layers.py`` ``_attention`` behind ``Attention._prep_qkv`` / ``_wrap_up``) is
  its named per-call fallback over ``ROWPAIR_TRIATT_QBLOCK`` query sub-blocks (the core's ``LEVER name=F1.flash_triattn`` line); the core's
  opt-out ``ROWPAIR_TRIATT_CORE=torch`` runs the torch statement for every call, counted (``kernel_torch``); a process that cannot run the
  kernel (no CUDA device, the kernel not importable) is refused BY NAME at bind — never a silent torch statement.

The word is ``tier:big`` (the core's tier dispatch: the row ``opt_core.kernels.triattn`` selects for the ``big`` tier per window on this
card, the bias plane staged once per gathered plane under the line's ``ROWPAIR_TRIATT_STAGE=once``); the provider door below takes only the
fp32 row batches of the confidence stack. No variable of the kit selects a kernel: the word is a constant of the line (``KERNEL_WORDS``). One ``[opendde-opt tp] TRIATT …`` census line
per rank at exit, next to the core's own LEVER line; ``tp.kit_stats()["tp_kernels"]`` carries the same numbers. Numerics: tier 2 (the line's) —
online-softmax flash attention is not bitwise to the torch statement; run-to-run repeatable.

The sharded schedule's triangle MULTIPLICATION (lever word ``tp_trimul`` in this census): every TriangleMultiplication module's
row blocks go to the core's fused row-block provider (``trimul_provider``: ``opt_core.mem.rowpair.trimul_fused`` — fpf_trimul_v4 at pair
widths 128 / 256, its rows unit fpf_trimul_rows at this engine's 384 and 64 on a card whose rows-only table carries a MEASURED key);
declined units run the engine's torch statements (``tp._trimul_fns``), counted by reason.
"""
from __future__ import annotations

import atexit
import os
import sys

TAG = "opendde-opt tp"
KERNEL_WORDS: dict[str, str] = {"triatt": "tier:big", "trimul": "fpf_v4"}   # ONE vocabulary with the core (mem.rowpair.triatt: CORE_KERNELS + TIER_PREFIX words — tier:big = the row kernels.triattn selects for
                                                           # the big tier per window on this card, its bias operand staged ONCE per plane under ROWPAIR_TRIATT_STAGE=once; mem.rowpair.trimul_fused.KERNEL_WORDS:
                                                           # the fused row-block provider — fpf_trimul_v4 at pair widths 128 / 256, its rows unit fpf_trimul_rows at 64 / 384)
_CORE_SERVE_DTYPES = ("torch.bfloat16", "torch.float16")   # the compute dtypes the core's tier dispatch serves (attention_core serve_dtypes default); other dtypes (the fp32 confidence
                                                           # stack) are offered to the kit's provider door first
TRIMUL_EXPECTED = ("env_torch", "below_gate", "c=")         # declared decline reasons of the provider: the core's opt-out word, its size floor, a width / card its tables do not
                                                           # carry MEASURED (e.g. c=384/384 on a card without a rows-table key — every unit then runs the torch statements, counted)
TRIATT_EXPECTED = ("kernel_torch",)                          # declared fallback reason: the core's opt-out word; `unsupported:*` / `kernel_unavailable:*` are
                                                             # undeclared — named at exit (tp.fallbacks), the exit rule fails closed

TRIATT = {"kernel": None, "bound": 0, "calls": 0, "door": None, "door_calls": 0, "door_rows": {}, "door_asides": {}, "stage_word": None}
TRIMUL = {"kernels": None, "bound": 0, "modules": 0, "widths": {}, "refused": {}}
"""This process's triangle-multiplication record: ``kernels`` (the provider word bound: ``fpf_v4`` = opt_core.mem.rowpair.trimul_fused around the
engine's torch statements), ``bound`` (TriMulFns built on the provider), ``modules`` (distinct TriangleMultiplication modules), ``widths`` (``<c_z>/<c_h>`` -> modules),
``refused`` (reason -> n: a module the provider could not wrap — its torch statements run, named); launches / declines are the core provider's (``trimul_census``)."""
_TRIMUL_ATTR = "_odde_tp_trimul_fused"                     # the provider memo on each TriangleMultiplication module (one FusedTriMulFns per module per process)
"""This process's triangle-attention record: ``kernel`` (the word bound), ``bound`` (TriangleAttention modules bound), ``calls`` (row batches
handed to the core dispatch); served / fallback counts are the core dispatch's (``triatt_census``). ``door``: the line's triangle-attention
PROVIDER door: a row batch ``[rows, H, S, D]`` is the provider's own call form ``[1, rows, H, S, D]`` (row batches are the batch
dimension, queries = keys = S), so where the line installed the provider unit (``odde_triattn_bind``, the ARM's site, word = the line's
``ODDE_TRIATTN`` tier word) each batch is offered to the provider FIRST — ``door_calls`` / ``door_rows`` (the provider's row per call) —
and a batch the provider hands back by name (``Aside``: dtype / cell / form) runs on the core dispatch as before, counted in
``door_asides``. ``stage_word``: how the bias operand is staged for the provider (``ROWPAIR_TRIATT_STAGE``: ``once`` per gathered plane on
the row-sharded line, ``per_call`` otherwise)."""


class TpKernelRefused(RuntimeError):
    """A kernel the process cannot serve, refused by name at bind (the sentence names the lever and the core's opt-out)."""


def _attention_core(mha):
    """The engine's eager triangle-attention core: ``_attention(query, key, value, biases)`` of the module that defines ``mha``'s class
    (``opendde/model/triangular/layers.py:304-326``: logits = q·kᵀ + Σ biases, softmax without a cast, · v — the statement
    ``layers.Attention.forward`` runs on its torch route), with ``mha._prep_qkv`` / ``mha._wrap_up`` around it. A module without these three is
    refused by name (the dispatch's fallback must be the module's own statement, never a look-alike)."""
    mod = sys.modules.get(type(mha).__module__)
    core = getattr(mod, "_attention", None)
    if core is None or not callable(getattr(mha, "_prep_qkv", None)) or not callable(getattr(mha, "_wrap_up", None)):
        raise TpKernelRefused(f"refused: tp_triatt ({KERNEL_WORDS['triatt']}): {type(mha).__module__}.{type(mha).__name__} has no "
                              "_prep_qkv / _wrap_up / module _attention (the OpenFold attention statement the dispatch falls back to)")
    return core


def attention_dispatch(t, chunk_size=None):
    """``run(x_rows, mask_bias, tb) -> o``: the module's attention on one row batch through the core's ONE row-block dispatch — ``_prep_qkv``
    (q PRE-SCALED, so ``scale=1.0`` for the fused kernel and the dispatch's torch fallback is the engine's statement byte for byte), the dispatch,
    ``_wrap_up`` (gating + ``linear_o``). ``chunk_size``: the stock's query chunk (the fallback's ``stock_qblock``; None = ``ROWPAIR_TRIATT_QBLOCK``)."""
    from opt_core.mem.rowpair import RowpairRefused, triatt as _ta
    kernel = KERNEL_WORDS["triatt"]
    mha = t.mha
    core = _attention_core(mha)
    qblock = int(chunk_size) if chunk_size else _ta.lever_rows("TRIATT_QBLOCK")
    try:
        dispatch = _ta.attention_core(lambda q_, k_, v_, b: core(q_, k_, v_, list(b)),
                                      kernel=kernel, scale=1.0, layout="bnhsd", mask_from="bias0", tri_bias="bias1", stock_qblock=qblock, expected=TRIATT_EXPECTED)
    except RowpairRefused as e:                                                   # no CUDA device / the kernel not importable: refused by name with the opt-out, never a silent torch statement
        raise TpKernelRefused(f"refused: tp_triatt ({kernel}): {e}") from None
    TRIATT["kernel"] = kernel
    TRIATT["bound"] += 1
    TRIATT["stage_word"] = (os.environ.get("ROWPAIR_TRIATT_STAGE") or "per_call").strip()   # the line exports once; the core records triatt_stage=once:<kernel> when it engages
    door = _provider_door()
    TRIATT["door"] = "odde_triattn_bind" if door is not None else None
    _register_exit_lines()

    def run(x_rows, mask_bias, tb):
        q, k, v = mha._prep_qkv(x_rows, x_rows)                                   # [rows, H, N, d], q / sqrt(d) applied (layers.py:389-410, apply_scale=True: the torch route's operand)
        TRIATT["calls"] += 1
        o = None
        if door is not None and str(q.dtype) not in _CORE_SERVE_DTYPES:            # the provider door (fp32 row batches only — the confidence stack under skip_amp; bf16 / fp16 batches go to
                                                                                  # the core's tier dispatch below, which stages the bias plane once): the batch in the provider's call form [1, rows, H, S, D]
            try:
                o = door.serve(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), tb.unsqueeze(0), (mask_bias == 0)[None],   # bias [1,1,H,S,S], mask [1,rows,1,1,S] bool
                               1.0, stock5=lambda q5, k5, v5, b5, m5, *a, **kw: dispatch(q5[0], k5[0], v5[0], [mask_bias, tb]).unsqueeze(0))[0]
                TRIATT["door_calls"] += 1
            except door.Aside as e:                                               # handed back by name (dtype / cell / form): the core dispatch serves this batch
                r = str(e) or "aside"
                TRIATT["door_asides"][r] = TRIATT["door_asides"].get(r, 0) + 1
                o = None
        if o is None:
            o = dispatch(q, k, v, [mask_bias, tb])
        del q, k, v
        return mha._wrap_up(o.transpose(-2, -3), x_rows)                          # [rows, N, H, d] -> gate + linear_o (layers.py:412-426)
    return run


def _provider_door():
    """The triangle-attention provider unit of this process when the line installed it and it is active (``odde_triattn_bind``: imported by
    the ARM add-on under the line's ``ODDE_TRIATTN`` word; never imported here), else None. ``ODDE_TP_TRIATT_DOOR=0`` keeps the door shut
    (every batch on the core dispatch) — a named diagnostic opt-out, printed in the census as ``door=off``;
    the core's ``ROWPAIR_TRIATT_CORE=torch`` shuts it too (``door_asides=core_torch``), so that opt-out means the torch statement for every batch."""
    if os.environ.get("ODDE_TP_TRIATT_DOOR", "1").strip() == "0":
        TRIATT["door_asides"]["door_off"] = 0
        return None
    if (os.environ.get("ROWPAIR_TRIATT_CORE") or "").strip() == "torch":            # the core's opt-out (every window on the torch statement) shuts the door as well:
        TRIATT["door_asides"]["core_torch"] = 0                                     # under it no batch of this process reaches a provider kernel
        return None
    tb_mod = sys.modules.get("odde_triattn_bind")
    if tb_mod is None or not callable(getattr(tb_mod, "serve", None)) or not hasattr(tb_mod, "Aside"):
        return None
    try:
        if callable(getattr(tb_mod, "active", None)) and not tb_mod.active():
            return None
    except Exception:                                                             # noqa: BLE001 — a unit that cannot say whether it is active is not a door
        return None
    return tb_mod


def trimul_provider(m, stock):
    """The core's FUSED ROW-BLOCK triangle-multiplication provider (``opt_core.mem.rowpair.trimul_fused.fused_trimul_fns``) around this
    TriangleMultiplication module's torch statements ``stock`` (:func:`opendde_opt.tp._trimul_fns`): K1 (LayerNorm_in + gated a / b projections
    + mask, reading the channel-last row block in place) and K3 (LayerNorm_out + linear_z + output gate + residual on the column window in place)
    per row block — the fpf_trimul_v4 kernels at pair widths 128 / 256, the rows unit fpf_trimul_rows at this engine's 384 (trunk,
    MSA-module, structural stacks) and 64 (template stack) where its rows-only table carries a MEASURED key for the card (e.g. ``9.0|C384|H384``, each key with
    its own token floor). Units the provider declines (``ROWPAIR_TRIMUL_KERNELS=torch``, below the
    floor, a width / card without a key) run ``stock``, counted by reason on the core's ``LEVER name=F2.trimul_rows`` line. The weights are named
    once, by the x1 ARM add-on's map (``odde_trimul_bind.weights_of``: the ten canonical tensors, no biases — OpenDDE's projections carry none).
    Returns the provider (memoised on the module), or None with ``TRIMUL["refused"][reason]`` counted when it cannot be built here (the map or
    the core module not importable, unequal LayerNorm eps) — the caller then runs ``stock`` for that module, named in the census."""
    prov = m.__dict__.get(_TRIMUL_ATTR)
    if prov is not None:
        TRIMUL["bound"] += 1
        return prov
    try:
        from opt_core.mem.rowpair import trimul_fused as _tf
        import odde_trimul_bind as _otb                                              # the kit's ONE OpenDDE -> WEIGHT_KEYS name map (levers/ARMT, on every line's path)
    except ImportError as e:
        r = f"import:{getattr(e, 'name', None) or type(e).__name__}"
        TRIMUL["refused"][r] = TRIMUL["refused"].get(r, 0) + 1
        return None
    eps = float(getattr(m.layer_norm_in, "eps", 1e-5))
    if float(getattr(m.layer_norm_out, "eps", eps)) != eps:
        TRIMUL["refused"]["eps_unequal"] = TRIMUL["refused"].get("eps_unequal", 0) + 1
        return None
    prov = _tf.fused_trimul_fns(_otb.weights_of(m), stock, eps=eps)                # cells=None: the kernels' own tables (v4 table.json / the rows-only table); stock_round: the
    m.__dict__[_TRIMUL_ATTR] = prov                                                 # gated output is bf16 before the residual add, as the engine's bf16 statement adds it
    TRIMUL["kernels"] = KERNEL_WORDS["trimul"]
    TRIMUL["bound"] += 1
    TRIMUL["modules"] += 1
    w = f"{int(getattr(prov, 'C_z', 0))}/{int(prov.C_h)}"
    TRIMUL["widths"][w] = TRIMUL["widths"].get(w, 0) + 1
    _register_exit_lines()
    return prov


def trimul_census() -> dict:
    """``TRIMUL`` plus the core provider's account when bound: ``served`` (kernel launches k1 + k3), ``fallback`` (declined units), ``fallback_by``,
    the facts (``cells``, ``k1_impl``, ``unit``, ``rows_table``, ``min_tokens_source``), ``impl``, ``k1_launches`` / ``k3_launches``, and
    ``unexpected`` (decline reasons outside ``TRIMUL_EXPECTED``)."""
    c = dict(TRIMUL, served=0, fallback=0, fallback_by={}, facts={}, impl=None, k1_launches=0, k3_launches=0, unexpected={})
    c["widths"], c["refused"] = dict(TRIMUL["widths"]), dict(TRIMUL["refused"])
    if TRIMUL["kernels"]:
        from opt_core.mem.rowpair import trimul_fused as _tf
        d = _tf.describe()
        c.update(served=int(d.get("served") or 0), fallback=int(d.get("fallback") or 0), fallback_by=dict(d.get("fallback_by") or {}),
                 facts=dict(d.get("facts") or {}), impl=d.get("impl"), k1_launches=int(d.get("k1_launches") or 0), k3_launches=int(d.get("k3_launches") or 0))
        c["unexpected"] = {r: n for r, n in c["fallback_by"].items() if not str(r).startswith(TRIMUL_EXPECTED)}
    return c


def trimul_census_line() -> str:
    c = trimul_census()
    f = c["facts"]
    return (f"TRIMUL kernels={c['kernels']} bound={c['bound']} modules={c['modules']} widths={_fb(c['widths'])} served={c['served']} k1={c['k1_launches']} k3={c['k3_launches']}"
            f" fallback={c['fallback']} fallback_by={_fb(c['fallback_by'])} unit={f.get('unit') or '-'} cells={f.get('cells') or '-'} k1_impl={f.get('k1_impl') or '-'}"
            f" rows_table={f.get('rows_table') or '-'} refused={_fb(c['refused'])}")


# ----------------------------------------------------------------------------------------------------------------- census
def triatt_census() -> dict:
    """``TRIATT`` plus the core dispatch's counts when the flash word is bound: ``served`` / ``fallback`` (calls), ``fallback_by``, ``impl``,
    and ``unexpected`` (fallback reasons outside ``TRIATT_EXPECTED``)."""
    c = dict(TRIATT, served=0, fallback=0, fallback_by={}, impl=None, unexpected={})
    c["door_rows"], c["door_asides"] = dict(TRIATT.get("door_rows") or {}), dict(TRIATT.get("door_asides") or {})
    if TRIATT.get("door"):                                                        # the provider's own account of the rows it served (its describe(): rows per call)
        tb_mod = sys.modules.get(TRIATT["door"])
        try:
            c["door_rows"] = dict((tb_mod.describe() or {}).get("rows") or {}) if tb_mod is not None else c["door_rows"]
        except Exception:                                                         # noqa: BLE001
            pass
    if TRIATT["kernel"] not in (None, "torch"):
        from opt_core.mem.rowpair import triatt as _ta
        d = _ta.describe_core()
        c.update(served=int(d.get("core_served") or 0), fallback=int(d.get("core_fallback") or 0), fallback_by=dict(d.get("fallback_by") or {}), impl=d.get("impl"))
        c["unexpected"] = {r: n for r, n in c["fallback_by"].items() if r not in TRIATT_EXPECTED}
    return c


def _fb(d: dict) -> str:
    return ",".join(f"{k}:{v}" for k, v in sorted((d or {}).items())) or "none"


def triatt_census_line() -> str:
    c = triatt_census()
    door = "off" if "door_off" in (c.get("door_asides") or {}) else (c.get("door") or "none")
    return (f"TRIATT kernel={c['kernel']} bound={c['bound']} calls={c['calls']} served={c['served']} fallback={c['fallback']} fallback_by={_fb(c['fallback_by'])}"
            f" door={door} door_calls={c.get('door_calls', 0)} door_rows={_fb(c.get('door_rows'))} door_asides={_fb({k: v for k, v in (c.get('door_asides') or {}).items() if k != 'door_off'})}")


def stats() -> dict:
    """The manifest's ``tp_kernels`` block (tp.kit_stats): the census dict, plain data."""
    return {"triatt": dict(triatt_census()), "trimul": dict(trimul_census())}


def undeclared() -> list[str]:
    """Named events for the exit rule (tp.fallbacks): fallback reasons outside the declared set — one sentence."""
    a = triatt_census()
    out = [f"tp_triatt ({a['kernel']}): undeclared fallback {_fb(a['unexpected'])}"] if a["unexpected"] else []
    b = trimul_census()
    if b["unexpected"]:
        out.append(f"tp_trimul ({b['kernels']}): undeclared decline {_fb(b['unexpected'])}")
    return out


_EXIT_LINE = {"registered": False}


def _emit_exit_lines() -> None:
    """At interpreter exit, once per rank: the kit's TRIATT census line and the core's own LEVER line of the kernel bound (``F1.flash_triattn``)."""
    try:
        if TRIATT["kernel"] not in (None, "torch"):
            from opt_core.mem.rowpair import triatt as _ta
            sys.stderr.write(f"[{TAG}] {triatt_census_line()}\n")
            _ta.emit_core_line(TAG)
        if TRIMUL["kernels"] or TRIMUL["refused"]:                                  # the kit's TRIMUL census + the core provider's own LEVER name=F2.trimul_rows line
            sys.stderr.write(f"[{TAG}] {trimul_census_line()}\n")
            if TRIMUL["kernels"]:
                from opt_core.mem.rowpair import trimul_fused as _tf
                _tf.emit_line(TAG)
        sys.stderr.flush()
    except Exception:                                       # noqa: BLE001 — interpreter teardown: the manifest's tp_kernels block carries the same numbers
        pass


def _register_exit_lines() -> None:
    if not _EXIT_LINE["registered"]:
        _EXIT_LINE["registered"] = True
        atexit.register(_emit_exit_lines)
