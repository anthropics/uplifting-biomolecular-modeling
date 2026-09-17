"""The `trimul_exact` lever (exact class; the `exact` line): the pair stacks' triangle multiplicative updates (outgoing + incoming: pairformer,
MSA-module pair stack, template pair stack) served through the core's ONE triangle-multiplication provider (`opt_core.kernels.trimul`, cell
table `TRIMUL_CELLS.json`; binding `opt_core.trimul.by_word`).  A call's STATEMENT decides the word it asks (cells/trimul_form.py):

* library-statement classes (`use_cueq_triangle_kernels` True at the call -- the trunk, MSA-module and confidence pair stacks under the stock
  configuration): word `exact`.  Per call class (GPU, precision, c_z, c_hidden, token bucket, direction) the provider names the exact-class row
  the table marks bitwise AND at or above the line's own op: `tmk3_exact` (the cuEquivariance TriMul's floating-point operations in order on
  the core's kernels) at c_z 128 up to 400 tokens, H100 and A100; the line's own cuEquivariance call (`cueq`) everywhere else -- served, by
  name, as the line runs it.
* module-statement classes (the flag False at the call: the template pair stack, c_z 64, on every input; every pair block when the runner
  configuration switches the library off): the exact tier UNDER THIS ENGINE'S FORM KEY, `exact+of3_module` (opt_core >= 0.5.67.0: the row
  `of3_form` -- the module's own inference statement issued whole-tensor, exact class against the module -- where a form cell proves it,
  refused by name elsewhere) under the sub-lever `trimul_form` (OPENFOLD3_OPT_TRIMUL_EXACT_FORM=1; left off, those classes are the line's,
  counted `form_off`).

A kernel row serves a class only after its first eager call proved `torch.equal` against the line's own statement on that call's operands
(library classes: once per class; module classes: once per class and token count -- a class whose bits differ is refused for the process and
named; a class first met inside a CUDA-graph capture is the line's for that call).  Bitwise the line without it.

The call CLASS carries the call's LEADING BATCH EXTENT: the trunk, MSA-module and template pair stacks call
with one pair tensor (`[1, N, N, c]`: the cell key as before); the confidence head runs its pairformer over the diffusion samples as ONE batch
below `per_sample_token_cutoff` (`[S, N, N, c]`, S = 5: class `<cell key>|B5`).  The library op's batched call is not S calls of one sample byte
for byte, so a row proven / vouched on the batch-1 layout never serves the batched one on that evidence: the kit states `batch=` to the core's
select() (the core answers the stock op BY NAME for a batched layout no cell vouches: `exact_batch_unvouched`), takes its first-call proof per
layout, names the batched class on the census (`cells=…|B5:<row>`) and counts the batched calls the line's own statement serves
(`line=batched_cell:<row>:<n>`).

Switch: OPENFOLD3_OPT_TRIMUL_EXACT=1 (+ OPENFOLD3_OPT_TRIMUL_EXACT_FORM=1).  Evidence `[openfold3-opt/trimul_exact] installed …` and the exit line
`[openfold3-opt/trimul_exact] LEVER name=trimul_exact state=on word=exact form=<on|off|refused> rows=<row>:<n>,… line=<reason>:<n>,… proven=<classes>
bits_differ=<classes|none> cells=<cell>:<row>|…` (state=refused reason=… when it cannot bind), plus trimul_form's own LEVER line."""
import atexit
import os
import sys
import threading

ENV = "OPENFOLD3_OPT_TRIMUL_EXACT"
PREFIX = "[openfold3-opt/trimul_exact]"
VALUES = ("1",)
WORD = "exact"
LINE_ROWS = ("cueq", "torch_math")                 # the provider's named stock rows: the line's own statement serves, unchanged
STATE = {"installed": False, "state": "off", "reason": None, "served": {}, "line": {}, "proven": [], "bits_differ": {}, "cells": {}, "calls": 0, "form": "off"}
_UNITS = set()                                     # the proof units held on this process (cells/trimul_form.proof_unit)
_LOCK = threading.Lock()
BATCH_TAG = "|B"                                   # the class-key field of a call whose leading batch extent is > 1 (`<cell key>[|module]|B<b>`; batch 1: no field)


def batch_extent(shape) -> int:
    """The leading batch extent of a pair tensor `[..., N, N, c]`: the product of the dimensions in front of the last three (1 for a 3-d tensor)."""
    b = 1
    for d in tuple(shape)[:-3]:
        b *= int(d)
    return max(1, b)


def batch_key(key: str, b: int) -> str:
    """The class key of a call of leading batch extent `b`: the key as before at batch 1, `<key>|B<b>` above it (its own memo entry, its own
    first-call proof unit, its own census cell)."""
    return f"{key}{BATCH_TAG}{int(b)}" if int(b) > 1 else str(key)


def plan_class(KT, F, *, cc, prec, C, D, N, B, direction, statement, stack, use_form):
    """The exact line's decision for ONE call class, from the core's table (pure; no tensor): ("row", row, cell, provider word) -- a kernel row
    to prove and serve -- or ("line", reason) -- the line's own statement, counted by reason; plus the census cell entry (cell id, value) or None.
    The core's select() is asked the tier word exact WITH the batch extent (`batch=B`): a batched layout (B > 1) no cell vouches answers the
    stock op by name (`exact_batch_unvouched`) -> ("line", "batched_cell:<row>")."""
    if statement == F.MODULE_TAG and not use_form:                            # trimul_form left off (or the core without the row): the module's statement is the line's, by name
        return ("line", "form_off"), None
    dt = prec.split("_")[-1] if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec)
    pword = F.provider_word(statement)                                      # exact (library classes: no form) | exact+of3_module (module classes: this engine's form)
    try:
        sel = KT.select(cc, dt, C, D, N, direction, word="exact", form=(F.FORM if statement == F.MODULE_TAG else None),
                        residency=("fp32" if prec.startswith("f32z_") else None), stack=stack, tf32=(prec == "tf32"), has_cueq=True, batch=int(B))
        row, cell = sel.row, (sel.cell or "no-cell")
    except KT.Refusal as e:
        why = "".join(ch if (ch.isalnum() or ch in "._:<>=-|") else "_" for ch in f"{e.kind}")[:72]
        return ("line", f"declined:{why}"), None                             # the calls count under line=declined:<kind> (the table's by-name decline -- e.g. no form cell proven at this size)
    ccell = batch_key(cell, B)                                              # the census names the batched class beside the batch-1 class of the same cell
    line_reason = f"cell:{row}" if int(B) <= 1 else f"batched_cell:{row}"
    if statement == F.MODULE_TAG:
        hit = ("row", row, cell, pword) if row == F.ROW else ("line", line_reason)
        return hit, (ccell, 1)
    hit = ("line", line_reason) if row in LINE_ROWS else ("row", row, cell, row)
    return hit, (ccell, row)


def requested(environ=None) -> bool:
    v = ((os.environ if environ is None else environ).get(ENV) or "").strip()
    if not v:
        return False
    if v not in VALUES:
        raise ValueError(f"{ENV}={v!r} is not one of {'|'.join(VALUES)}")
    return True


def serving() -> bool:
    return bool(STATE.get("installed")) and STATE.get("state") == "on"


def _bump(d, k, n=1):
    with _LOCK:
        d[k] = d.get(k, 0) + n


def fields() -> str:
    j = lambda d: ",".join(f"{k}:{v}" for k, v in sorted(d.items())) or "none"
    return (f"word={WORD} form={STATE.get('form', 'off')} rows={j(STATE['served'])} line={j(STATE['line'])} proven={','.join(STATE['proven']) or 'none'} "
            f"bits_differ={','.join(f'{k}({v})' for k, v in sorted(STATE['bits_differ'].items())) or 'none'} "
            f"cells={'|'.join(f'{k}:{v}' for k, v in sorted(STATE['cells'].items())) or 'none'}")


def census_line() -> str:
    body = fields() if STATE.get("state") == "on" else f"reason={STATE.get('reason') or 'not_installed'}"
    return f"{PREFIX} LEVER name=trimul_exact state={STATE.get('state', 'off')} {body}"


def _log(msg):
    sys.stderr.write(f"{PREFIX} {msg}\n")


def install(environ=None) -> dict:
    """Rebind PairBlock.tri_mul_out_in on the exact line (idempotent): per module and direction, the provider's exact-class row for the call's
    class -- word exact for library-statement classes, row of3_form for module-statement classes under trimul_form -- when one is named and
    proven on this process, else the line's own statement for that block."""
    if STATE.get("installed") or not requested(environ):
        return STATE
    env = os.environ if environ is None else environ
    import torch
    if not torch.cuda.is_available():
        STATE.update(installed=True, state="refused", reason="no_cuda_device"); atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
        return STATE
    from opt_core.kernels import trimul as KT                             # pure: the cell table and select()
    from opt_core import trimul as T                                      # the binding: by_word providers, Call
    from openfold3.core.model.latent.base_blocks import PairBlock
    from openfold3.core.model.layers.triangular_multiplicative_update import TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
    from .pairfused import trimul_weights                                 # the ONE weight-name map of this kit (WEIGHT_KEYS + BIAS_KEYS vocabulary)
    from . import trimul_form as F                                        # the sub-lever: module-statement classes -> row of3_form (its record, words and proof rule)
    TMU = (TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming)
    _inner = PairBlock.tri_mul_out_in                                     # the line's own statement (the trunk-kernels hook's cuEquivariance route under OF3T_TRIMUL=cueq)
    use_cueq_line = (env.get("OF3T_TRIMUL") or "").strip() == "cueq"
    use_form = F.arm(KT, env)                                             # OPENFOLD3_OPT_TRIMUL_EXACT_FORM=1 and the core names the row
    STATE["form"] = F.STATE.get("state", "off")
    stack = {"word": None}
    providers = {}                                                        # provider word -> by_word provider (a row word for library classes; F.WORD for module classes)
    memo = {}                                                             # class key -> ("row", row, cell, provider word) | ("line", reason)

    def _provider(word):
        p = providers.get(word)
        if p is None:
            p = providers[word] = T.by_word(trimul_weights, word, name=f"openfold3:trimul_exact:{word}")
        return p

    def _class(zz, mod, direction, statement):
        cc = KT._device_cc(zz)
        prec, _ = KT.call_precision(zz if zz.dim() == 4 else zz.reshape(-1, *zz.shape[-3:]))
        C = int(zz.shape[-1]); D = int(mod.linear_a_p.weight.shape[0]); N = int(zz.shape[-2])
        B = batch_extent(zz.shape)                                            # the leading batch extent: 1 on the trunk / MSA / template stacks, S on the confidence head's batched samples
        key = KT.cell_key(cc, prec.split("_")[-1] if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec), C, D, N, direction,
                          residency=("fp32" if prec.startswith("f32z_") else None), tf32=(prec == "tf32"))
        key = key[0] if isinstance(key, tuple) else str(key)                 # cell_key -> (key, backward, note): the key string names the class
        return batch_key(F.class_key(key, statement), B), cc, prec, C, D, N, B

    def _select(key, cc, prec, C, D, N, direction, statement, B=1):
        hit = memo.get(key)
        if hit is not None:
            return hit
        if stack["word"] is None and not (statement == F.MODULE_TAG and not use_form):
            stack["word"] = KT.stack_word()
        hit, cell_entry = plan_class(KT, F, cc=cc, prec=prec, C=C, D=D, N=N, B=B, direction=direction, statement=statement, stack=stack["word"], use_form=use_form)
        if cell_entry is not None:
            (F.STATE if statement == F.MODULE_TAG else STATE)["cells"][cell_entry[0]] = cell_entry[1]
        memo[key] = hit
        return hit

    def _line(self, z, pair_mask, inplace_safe, uck, utk, why, n=2, form=False):   # the line's own statement for the whole block, untouched (counted by reason)
        _bump(STATE["line"], why, n)
        if form:
            F.bump("line", why, n)
        return _inner(self, z, pair_mask, inplace_safe, use_cueq_triangle_kernels=uck, use_triton_triangle_kernels=utk)

    def tri_mul_out_in(self, z, pair_mask, inplace_safe, use_cueq_triangle_kernels=False, use_triton_triangle_kernels=False):
        mods = ((self.tri_mul_out, "outgoing"), (self.tri_mul_in, "incoming"))
        uck = True if use_cueq_line else use_cueq_triangle_kernels            # the line's route (OF3T_TRIMUL=cueq sets it on every call)
        utk = use_triton_triangle_kernels
        statement = F.statement_word(bool(uck))                                # library | module: what the block executes for this call
        fm = statement == F.MODULE_TAG and use_form                            # count this block on trimul_form's census too
        if self.training:
            return _line(self, z, pair_mask, inplace_safe, uck, utk, "training", form=fm)
        if not all(type(m) in TMU for m, _ in mods):
            return _line(self, z, pair_mask, inplace_safe, uck, utk, "module", form=fm)
        if z.dim() < 3 or not z.is_cuda:
            return _line(self, z, pair_mask, inplace_safe, uck, utk, "rank_or_device", form=fm)
        if statement == F.MODULE_TAG and utk:                                  # use_triton_triangle_kernels: the module's Triton inference path -- a third statement, not the row's
            return _line(self, z, pair_mask, inplace_safe, uck, utk, "triton_flag", form=fm)
        if statement == F.MODULE_TAG and not inplace_safe:                     # the module's out-of-place path (mask, sigmoid, product in another order): not the statement the row states
            return _line(self, z, pair_mask, inplace_safe, uck, utk, "not_inplace", form=fm)
        hits = []
        for mod, direction in mods:
            key, cc, prec, C, D, N, B = _class(z, mod, direction, statement)
            hits.append((mod, direction, key, _select(key, cc, prec, C, D, N, direction, statement, B), N))
        STATE["calls"] += 2
        if any(h[3][0] == "line" for h in hits):                             # a module the table gives the line's op (c_z 128 above 400 tokens: cuEquivariance by cell; form_off): the
            for h in hits:                                                 #  block's own statement, untouched
                _bump(STATE["line"], h[3][1] if h[3][0] == "line" else "partner_on_line")
                if fm:
                    F.bump("line", h[3][1] if h[3][0] == "line" else "partner_on_line")
            return _inner(self, z, pair_mask, inplace_safe, use_cueq_triangle_kernels=uck, use_triton_triangle_kernels=utk)
        keys = tuple(h[2] for h in hits)
        if any(k in STATE["bits_differ"] for k in keys):
            return _line(self, z, pair_mask, inplace_safe, uck, utk, "bits_differ", form=fm)
        lead = z.shape[:-3]
        z4 = z.reshape(-1, *z.shape[-3:])
        m3 = None
        if pair_mask is not None:
            try:
                m3 = torch.broadcast_to(pair_mask, z.shape[:-1]).reshape(-1, *z.shape[-3:-1])
            except RuntimeError:
                return _line(self, z, pair_mask, inplace_safe, uck, utk, "mask_shape", form=fm)
        calls = []
        for mod, direction, key, hit, N in hits:
            prov = _provider(hit[3])
            call = T.Call(mod, z4, m3, direction, False, lambda: None)       # z is rebound per module below (call.z)
            try:
                prov.eligible(call)
            except T.Refused as e:
                if fm:
                    F.bump("refused", str(e).replace(" ", "_")[:60], 2)
                return _line(self, z, pair_mask, inplace_safe, uck, utk, f"declined:{str(e)[:52]}".replace(" ", "_"))
            calls.append((prov, call, hit[1], key, F.proof_unit(key, statement, N)))
        units = tuple(c[4] for c in calls)
        proven = all(u in _UNITS for u in units)
        if not proven and torch.cuda.is_current_stream_capturing():          # classes first met inside a capture: the line's statement for this call (no proof possible here)
            return _line(self, z, pair_mask, inplace_safe, uck, utk, "unproven_in_capture", form=fm)
        try:                                                                   # the rows, out of place: z + out-update, then + in-update (the block's arithmetic order)
            zz = z4
            for prov, call, row, key, unit in calls:
                call.z = zz
                zz = zz + prov.fn(call)
        except T.Refused as e:
            if fm:
                F.bump("refused", str(e).replace(" ", "_")[:60], 2)
            return _line(self, z, pair_mask, inplace_safe, uck, utk, f"declined:{str(e)[:52]}".replace(" ", "_"))
        except Exception as e:                                                 # a row that RAISES on its first (unproven) call of these classes -- e.g. a statement the runner's
            if proven:                                                         #  context does not admit on this stack -- refuses the classes BY NAME for the process; the line
                raise                                                          #  serves them (a proven unit raising later is a real error: re-raised)
            word = f"error:{type(e).__name__}"
            for k in keys:
                STATE["bits_differ"].setdefault(k, word)
            if fm:
                F.bump("refused", word, 2)
            _log(f"classes {keys[0]} + incoming -> {'+'.join(r for _, _, r, _, _ in calls)}: the row raised {type(e).__name__} on its first call "
                 f"({str(e).splitlines()[0][:80] if str(e) else ''}) - REFUSED for the process; the line serves them")
            return _line(self, z, pair_mask, inplace_safe, uck, utk, "row_error", form=fm)
        got = zz.reshape(*lead, *zz.shape[-3:])
        if proven:
            for _, _, row, _, _ in calls:
                _bump(STATE["served"], row)
            if fm:
                F.bump("served", "calls", 2)
            return got
        ref = _inner(self, z, pair_mask, inplace_safe, use_cueq_triangle_kernels=uck, use_triton_triangle_kernels=utk)   # the block's own statement on the same input (it may add into z in place: computed after the rows read z)
        rows = "+".join(r for _, _, r, _, _ in calls)
        if got.dtype == ref.dtype and got.shape == ref.shape and torch.equal(got, ref):
            with _LOCK:
                _UNITS.update(units)
                for k in keys:
                    if k not in STATE["proven"]:
                        STATE["proven"].append(k)
            if fm:
                for (_, _, _, key, unit) in calls:
                    F.note_proven(key, unit)
            _log(f"classes {keys[0]} + incoming -> {rows} ({statement} statement, {hits[0][4]} tokens): proven torch.equal against the line's statement on the "
                 f"first call; served from here")
        else:
            d = (got.float() - ref.float()).abs().max().item() if got.shape == ref.shape else float("nan")
            for k in keys:
                STATE["bits_differ"][k] = f"maxabs={d:.3g}"
            if fm:
                F.bump("refused", "bits_differ", 2)
            _log(f"classes {keys[0]} + incoming -> {rows} ({statement} statement, {hits[0][4]} tokens): bits differ from the line's statement (max abs {d:.3g}) - "
                 f"REFUSED for the process; the line serves them")
        _bump(STATE["line"], "first_call_proof", 2)
        if fm:
            F.bump("line", "first_call_proof", 2)
        return ref

    PairBlock.tri_mul_out_in = tri_mul_out_in
    STATE.update(installed=True, state="on", reason=None)
    _log(f"installed: PairBlock.tri_mul_out_in -> opt_core.kernels.trimul word={WORD} for library-statement classes, word {F.WORD} for module-statement classes "
         f"(form={STATE['form']}); rows proven per class against the line's {'cuEquivariance' if use_cueq_line else 'own'} statement; named stock rows = the line")
    atexit.register(lambda: sys.stderr.write(census_line() + "\n"))
    return STATE
