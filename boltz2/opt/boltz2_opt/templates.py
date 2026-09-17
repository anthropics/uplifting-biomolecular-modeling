"""boltz2_opt.templates — the template guard of the worker routes: a kit-level wrap of boltz 2.2.1's template path (no stock edit, the stock
pin stays), applied in the worker process by ``boltz2_opt.worker_launch --attach templ`` right after the worker's own
``boltz.model.models.boltz2`` import. It changes nothing the model computes; it COUNTS.

Why a guard: boltz's template featurizer (stock ``boltz/data/feature/featurizerv2.py`` process_template_features :1762-1823) maps each declared
template's tokens onto the query's residues by residue index (``toks = [t for t in toks if t["res_idx"] - offset in q_indices]``, :1801-1802)
and builds the template row from whatever mapped (compute_template_features :1696-1759): ``template_mask`` marks every index-mapped token
whether or not the template carries coordinates there; the structure lives in ``template_mask_cb`` / ``template_mask_frame`` (:1287-1368). A
declared template none of whose aligned residues maps becomes an all-zero row — a dummy slot, exactly what load_dummy_templates_features
(:1664-1693) produces for an
input that declared no template — with no line in any log. The model then runs untemplated (TemplateV2Module, ``trunkv2.py:444-505``:
``num_templates.clamp(min=1)``, ``u = (v * template_mask).sum(dim=1) / …`` = 0) — and a template whose aligned region carries NO resolved
coordinates (``template_mask_cb`` / ``template_mask_frame`` all zero) is fed as its residue types alone, structure-free, equally silently. This
guard counts COORDINATE-BEARING tokens (a CB or a backbone frame resolved) and makes every structure-free slot, and a templated input whose
slots are ALL dummy, a NAMED event; the input then runs exactly as `boltz predict` runs it (untemplated), as stock does.

The three parts:
  1. featurizer census (the DataLoader process, per record): after the stock ``process_template_features`` returns, one line per declared
     (template, query chain): ``[boltz2-opt] TEMPLATE record=<id> template=<name> chain=<query chain> aligned_residues_resolved=<n>/<N>``
     and, for n == 0, ``[boltz2-opt] TEMPLATE DROPPED record=<id> template=<name> chain=<chain> reason=unresolved_aligned_residues 0/<N>``
     (N = the residues the record's alignments declare for that chain, n = the tokens of that chain the slot's mask covers).
  2. model-entry census (the worker process, per item, at ``Boltz2.forward`` — and, under ``--n_gpu P``, at the entry of the row-sharded forward
     that replaces it, ``rowpair._forward_sharded``, on every rank): from ``feats["record"]`` (the records' declared
     TemplateInfo entries) and ``feats["template_mask_cb"]`` / ``feats["template_mask_frame"]`` (``[B, T, N]``; a live slot has at least one
     coordinate-bearing token): ``[boltz2-opt] TEMPLATES record=<id> declared=<k> names=<n>
     slots=<T> live_slots=<j> real=<j>/<T>`` (``real`` = the slots with coordinates over the slots the model receives; + ``form=row_born
     n_gpu=<P>`` under ``--n_gpu P``: the template pair features are born as each rank's pair rows, boltz2_opt.rowpair ``_template_rows`` —
     no rank forms the dense ``[T, N, N, F]`` features); a record that declared templates and shows ``live_slots=0`` also prints
     ``[boltz2-opt] TEMPLATES ALL DUMMY
     record=<id>: template slots all dummy (declared=<k> live_slots=0/<T>); the input runs untemplated, as `boltz predict` runs it`` — a
     named event in the transcript and the report (``all_dummy``), never a refusal: stock runs that input, so does every mode.
  3. ``report()`` (written into the worker log as ``templ_report`` by worker_launch at exit): ``{installed, items_checked, declared, live,
     all_dummy, per_item: [{record, declared, names, slots, live_slots}]}`` — the APPLIED line's ``templates=`` field and the run manifest.
An input that declares no template is untouched by all three (declared=0: no line).

Contract (worker_launch / stack.evidence): ``GUARDS``; ``apply()`` -> the guards installed (idempotent); ``report()``.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List

TAG = "[boltz2-opt]"
GUARDS = ("templates",)
ALL_DUMMY_MARK = f"{TAG} TEMPLATES ALL DUMMY"       # stack.evidence lists these lines (a named event of the run, not a problem)
DROPPED_MARK = f"{TAG} TEMPLATE DROPPED"
_STATE: Dict[str, Any] = {"installed": False, "per_item": [], "seen": set(), "all_dummy": []}


def _say(line: str) -> None:
    sys.stderr.write(line + "\n"); sys.stderr.flush()


# ---------------------------------------------------------------- part 1: the featurizer census (pure function + the wrap) ----------------------------------------------------------------
ROW_BORN = "row_born"                              # the form word under --n_gpu P: template pair features computed as each rank's pair rows, never dense then sliced


def item_lines(r: dict) -> List[str]:
    """The per-input census lines of one census row (item_census's dict, + ``form`` / ``n_gpu`` under --n_gpu P): the TEMPLATES line, and the
    ALL DUMMY sentence when no slot is live. Printed by the worker at the model's entry (check_batch) and echoed by the caller after the run
    from the worker's report (report.run_exit) — the same words in both places."""
    line = f"{TAG} TEMPLATES record={r['record']} declared={r['declared']} names={r['names']} slots={r['slots']} live_slots={r['live_slots']} real={r['live_slots']}/{r['slots']}"
    if r.get("form"):
        line += f" form={r['form']} n_gpu={r.get('n_gpu')}"
    out = [line]
    if r["live_slots"] == 0:
        out.append(f"{ALL_DUMMY_MARK} record={r['record']}: template slots all dummy (declared={r['declared']} live_slots=0/{r['slots']}); the input runs untemplated, as `boltz predict` runs it")
    return out


def tp_n_gpu() -> int:
    """P of this worker process: boltz2_opt.rowpair's installed world size when the row-sharded pair stack is attached (``--n_gpu P``,
    worker_launch ``--attach … tp``), else 1. Read from the already-imported module only — nothing is imported or installed here."""
    rp = sys.modules.get("boltz2_opt.rowpair")
    try:
        return int(rp.n_gpu()) if rp is not None else 1
    except Exception:
        return 1


def featurizer_census(record_id: str, templates: list, token_asym: list, chain_asym: Dict[str, int], names: List[str], template_mask, coord_mask) -> List[dict]:
    """Per (template name, query chain): ``{record, template, chain, declared_residues, aligned_tokens N, resolved_tokens n, dropped}``.
    ``templates`` = the record's TemplateInfo entries (name, query_chain, query_st, query_en); ``token_asym`` = asym_id per query token;
    ``chain_asym`` = query chain name -> asym_id; ``names`` = the template names in slot order (the featurizer's grouping order);
    ``template_mask`` = the featurizer's ``[T, N]`` index-mapped mask; ``coord_mask`` = ``[T, N]``, non-zero where the template token carries
    coordinates (``template_mask_cb`` or ``template_mask_frame``). ``n`` counts coordinate-bearing aligned tokens, ``N`` the aligned tokens;
    a (template, chain) with ``n == 0`` is dropped: structure-free."""
    rows = []
    for t, name in enumerate(names):
        declared: Dict[str, int] = {}
        for ti in templates:
            if getattr(ti, "name", None) == name:
                ch = getattr(ti, "query_chain", None)
                declared[ch] = declared.get(ch, 0) + max(0, int(getattr(ti, "query_en", 0)) - int(getattr(ti, "query_st", 0)))
        for ch, n_decl in declared.items():
            asym = chain_asym.get(ch)
            idx = [i for i, a in enumerate(token_asym) if a == asym] if asym is not None else []
            n_aligned = sum(1 for i in idx if float(template_mask[t][i]) > 0)
            n_res = sum(1 for i in idx if float(template_mask[t][i]) > 0 and float(coord_mask[t][i]) > 0)
            rows.append({"record": record_id, "template": str(name), "chain": str(ch), "declared_residues": int(n_decl), "aligned_tokens": int(n_aligned),
                         "resolved_tokens": int(n_res), "dropped": n_res == 0})
    return rows


def census_lines(rows: List[dict]) -> List[str]:
    out = []
    for r in rows:
        out.append(f"{TAG} TEMPLATE record={r['record']} template={r['template']} chain={r['chain']} aligned_residues_resolved={r['resolved_tokens']}/{r['aligned_tokens']} declared_residues={r['declared_residues']}")
        if r["dropped"]:
            out.append(f"{DROPPED_MARK} record={r['record']} template={r['template']} chain={r['chain']} reason=unresolved_aligned_residues 0/{r['aligned_tokens']}")
    return out


def _wrap_featurizer(FZ) -> None:
    stock = FZ.process_template_features

    def process_template_features(data, max_tokens):
        out = stock(data=data, max_tokens=max_tokens)
        try:
            rec = data.record
            names = []
            for ti in (rec.templates or []):
                if ti.name not in names:
                    names.append(ti.name)                              # the stock grouping order (dict insertion by first occurrence, :1781-1783)
            chain_asym = {str(c["name"]): int(c["asym_id"]) for c in data.structure.chains}
            token_asym = [int(a) for a in data.tokens["asym_id"]]
            rows = featurizer_census(str(rec.id), list(rec.templates or []), token_asym, chain_asym, names, out["template_mask"], coord_mask(out))
            for line in census_lines(rows):
                _say(line)
        except Exception as e:  # noqa: BLE001 — the census never alters the stock result; a census that cannot be taken says so
            _say(f"{TAG} TEMPLATE census not taken for record {getattr(getattr(data, 'record', None), 'id', '?')}: {type(e).__name__}: {e}")
        return out

    FZ.process_template_features = process_template_features


# ---------------------------------------------------------------- part 2: the model-entry census + refusal ----------------------------------------------------------------
def item_census(records: list, coord_masks) -> List[dict]:
    """Per record of the batch: ``{record, declared, names, slots, live_slots}`` from the records' TemplateInfo lists and the batch's coordinate
    mask (``[B, T, N]``, ``template_mask_cb`` or ``template_mask_frame`` non-zero; ``live`` = a slot with at least one coordinate-bearing token —
    an index-mapped slot without coordinates is structure-free, i.e. dummy)."""
    rows = []
    for b, rec in enumerate(records or []):
        tinfos = list(getattr(rec, "templates", None) or [])
        names = sorted({str(getattr(t, "name", "")) for t in tinfos})
        tm = coord_masks[b] if coord_masks is not None else []
        slots = len(tm)
        live = sum(1 for t in range(slots) if _any(tm[t]))
        rows.append({"record": str(getattr(rec, "id", b)), "declared": len(tinfos), "names": len(names), "slots": int(slots), "live_slots": int(live)})
    return rows


def coord_mask(feats):
    """``template_mask_cb`` OR ``template_mask_frame`` of a feature dict (featurizer output ``[T, N]`` or batch ``[B, T, N]``): non-zero where a
    template token carries coordinates. Both keys are boltz 2.2.1's (featurizerv2 :1688-1690, :1754-1756); a dict without them is refused by
    name rather than read as 'no coordinates'."""
    missing = [k for k in ("template_mask_cb", "template_mask_frame") if k not in feats]
    if missing:
        raise KeyError(f"template features lack {', '.join(missing)}: the guard counts coordinate-bearing tokens from boltz 2.2.1's template_mask_cb / template_mask_frame")
    cb, fr = feats["template_mask_cb"], feats["template_mask_frame"]
    try:
        return (cb > 0) | (fr > 0)                                       # tensors / arrays
    except TypeError:
        return _or_lists(cb, fr)


def _or_lists(a, b):
    if isinstance(a, (list, tuple)):
        return [_or_lists(x, y) for x, y in zip(a, b)]
    return 1.0 if (float(a) > 0 or float(b) > 0) else 0.0


def _any(row) -> bool:
    try:
        return bool(row.any())
    except AttributeError:
        return any(float(x) > 0 for x in row)


def check_batch(feats) -> List[dict]:
    """The census proper: lines once per record; the ALL DUMMY line (a named event, no refusal) for a record that declared templates and has
    no live slot. Returns the census rows for records seen for the first time."""
    rows = item_census(feats.get("record") or [], coord_mask(feats))
    fresh = []
    for r in rows:
        if r["declared"] == 0 or r["record"] in _STATE["seen"]:
            continue
        _STATE["seen"].add(r["record"]); _STATE["per_item"].append(r); fresh.append(r)
        P = tp_n_gpu()
        if P > 1:                                   # --n_gpu P: the pair features of every slot are born as this rank's rows (rowpair._template_rows) — the census words
            r["form"] = ROW_BORN; r["n_gpu"] = P
        if r["live_slots"] == 0:
            _STATE["all_dummy"].append(r["record"])
        for line in item_lines(r):
            _say(line)
    return fresh


def _wrap_forward(B2) -> None:
    stock = B2.Boltz2.forward

    def forward(self, feats, *args, **kwargs):
        check_batch(feats)
        return stock(self, feats, *args, **kwargs)

    B2.Boltz2.forward = forward


# ---------------------------------------------------------------- contract ----------------------------------------------------------------
def apply(spec=None) -> List[str]:
    """Install the featurizer census and the model-entry guard (idempotent). Returns the guards installed."""
    if _STATE["installed"]:
        return list(GUARDS)
    import boltz.data.feature.featurizerv2 as FZ
    import boltz.model.models.boltz2 as B2
    _wrap_featurizer(FZ)
    _wrap_forward(B2)
    _STATE["installed"] = True
    _say(f"{TAG} GUARD name=templates state=on impl=boltz2_opt/templates.py origin=kit sites=featurizerv2.process_template_features,Boltz2.forward")
    return list(GUARDS)


def report() -> Dict[str, Any]:
    items = list(_STATE["per_item"])
    return {"installed": bool(_STATE["installed"]), "items_checked": len(items), "declared": sum(r["declared"] for r in items),
            "live": sum(r["live_slots"] for r in items), "all_dummy": list(_STATE["all_dummy"]), "per_item": items}
