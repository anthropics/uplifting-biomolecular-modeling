"""The template census — a kit-level guard around the stock template featurizer (every kit mode; no stock edit; the stock pin stays).

The site: `protenix.data.template.template_featurizer.InferenceTemplateFeaturizer.make_template_feature` (called per item by
`protenix.data.inference.infer_dataloader` L180) reads, for every protein chain carrying `templatesPath` when `use_template` is set, the
hits file, and asks `protenix.data.template.template_utils.TemplateHitFeaturizer.get_templates` for at most 4 template feature sets.
A hit whose featurisation raises (`template_utils.py` `_process_single_hit`: `except Exception as e` -> `SingleHitResult(None, None,
"Error processing hit: <e>", None)` — e.g. `TemplateAtomMaskAllZerosError` when the aligned residues are unresolved in the deposited
model, `CaDistanceError`, a missing mmCIF) is DROPPED: its error string lands in `TemplateSearchResult.errors`, which the caller discards
(`result, _ = ...get_templates(...)`, `templates = result.features`, one INFO line with the count). A chain whose every hit dropped is
assembled with dummy (zero) template slots and the item runs untemplated — silently, in stock.

The guard makes each of those a named event and the all-dummy case a refusal:
- every dropped hit -> `[protenix-opt] TEMPLATE DROPPED seq=<sha8 of the chain sequence> reason=<the stock error text>`; every stock
  warning -> `TEMPLATE WARNING`;
- per requesting chain (a protein chain with `templatesPath`) -> `[protenix-opt] TEMPLATES entity=<n> seq=<sha8> path=<file> hits=<N>
  real=<k>/4 dropped=<d>` (the partial census: k < N is printed, never inferred), or `... skipped=use_template_off` when the run set
  `--use_template false` (the file is named, never read: stock behaviour, now printed);
- an item none of whose chains names a `templatesPath`, with `use_template` on -> `[protenix-opt] TEMPLATES templates=none requested=0
  entities=<n>`: nothing was declared, nothing is refused, the item runs untemplated exactly as stock runs it;
- an item that DECLARED templates (>= 1 requesting chain, `use_template` on) whose real-slot census is 0 ->
  `[protenix-opt] TEMPLATES templates=all_dummy entities=<n> hits=<N> real=0`: named, counted (`templ_all_dummy` on the FINAL line), and
  the item runs exactly as stock runs it (every slot dummy) — the kit refuses nothing stock accepts.
`state()` is the end-of-run record (chains requested, hits, real slots, drops with reasons, all-dummy items) the activation report carries
(`stack.status()["templates"]`) and the FINAL/EXIT evidence reads (`report`).
"""
from __future__ import annotations

import hashlib
import os
from typing import Optional

from opt_core.autoload import patch_attr_at_import

TAG = "protenix-opt"
PREFIX = f"[{TAG}]"
SITE_FEATURIZER = ("protenix.data.template.template_featurizer", "InferenceTemplateFeaturizer.make_template_feature")
SITE_HITS = ("protenix.data.template.template_utils", "TemplateHitFeaturizer.get_templates")
MAX_SLOTS = 4                                                   # template_featurizer.py: TemplateFeatureAssemblyLine(max_templates=4); TemplateHitFeaturizer(max_hits=4)

_STATE = {"installed": [], "items": 0, "items_no_templates": 0, "chains_requested": 0, "chains_skipped_use_template_off": 0, "hits": 0, "real": 0,
          "dropped": [], "warnings": 0, "all_dummy_items": 0, "calls": []}


def seq_id(seq: str) -> str:
    return hashlib.sha256((seq or "").encode()).hexdigest()[:8]


def _say(line: str) -> None:
    print(line, flush=True)


def _wrap_get_templates(orig):
    def get_templates(self, *args, **kwargs):
        result_track = orig(self, *args, **kwargs)
        result = result_track[0] if isinstance(result_track, tuple) else result_track
        seq = kwargs.get("query_sequence") or kwargs.get("sequence_uid") or (args[1] if len(args) > 1 else (args[0] if args else ""))
        hits = kwargs.get("hits") if "hits" in kwargs else (args[2] if len(args) > 2 else None)
        sid = seq_id(str(seq))
        errors = list(getattr(result, "errors", None) or []); warnings = list(getattr(result, "warnings", None) or [])
        real = len(getattr(result, "features", None) or [])
        n_hits = len(hits) if hits is not None else None
        for e in errors:
            _say(f"{PREFIX} TEMPLATE DROPPED seq={sid} reason={str(e).strip()[:300]}")
            _STATE["dropped"].append({"seq": sid, "reason": str(e).strip()[:300]})
        for w in warnings:
            _say(f"{PREFIX} TEMPLATE WARNING seq={sid} text={str(w).strip()[:300]}")
        _STATE["warnings"] += len(warnings)
        _STATE["calls"].append({"seq": sid, "hits": n_hits, "real": real, "dropped": len(errors)})
        return result_track
    get_templates._ptx_template_census = True
    return get_templates


def requesting_chains(bioassembly) -> list:
    """The protein chains of an item that name a templates file: [(entity index, sequence, path)] in input order."""
    out = []
    for eid, info in enumerate(bioassembly or []):
        c = info.get("proteinChain") if isinstance(info, dict) else None
        if c and c.get("templatesPath"):
            out.append((eid, c.get("sequence", ""), c["templatesPath"]))
    return out


def census_item(bioassembly, use_template: bool, featurizer_present: bool, calls: list) -> dict:
    """One item's census from its requesting chains and the get_templates calls made while featurising it. Pure (tests call it)."""
    req = requesting_chains(bioassembly)
    rows = []
    by_seq = {}
    for c in calls:
        by_seq.setdefault(c["seq"], []).append(c)
    for eid, seq, path in req:
        sid = seq_id(seq)
        if not use_template or not featurizer_present:
            rows.append({"entity": eid, "seq": sid, "path": os.path.basename(str(path)), "skipped": "use_template_off" if not use_template else "no_template_featurizer"})
            continue
        cs = by_seq.get(sid) or []
        c = cs.pop(0) if cs else {"hits": 0, "real": 0, "dropped": 0}
        rows.append({"entity": eid, "seq": sid, "path": os.path.basename(str(path)), "hits": c["hits"], "real": c["real"], "dropped": c["dropped"]})
    active = [r for r in rows if "skipped" not in r]
    real_total = sum(r["real"] for r in active)
    return {"rows": rows, "requested": len(req), "active": len(active), "real_total": real_total,
            "all_dummy": bool(active) and real_total == 0}


def _wrap_make_template_feature(orig):
    def make_template_feature(*args, **kwargs):
        bio = kwargs.get("bioassembly", args[0] if args else None)
        use_template = kwargs.get("use_template", args[2] if len(args) > 2 else True)
        feat = kwargs.get("online_template_featurizer", args[3] if len(args) > 3 else None)
        start = len(_STATE["calls"])
        out = orig(*args, **kwargs)
        cen = census_item(bio, bool(use_template), feat is not None, _STATE["calls"][start:])
        _STATE["items"] += 1
        for r in cen["rows"]:
            if "skipped" in r:
                _STATE["chains_skipped_use_template_off"] += int(r["skipped"] == "use_template_off")
                _say(f"{PREFIX} TEMPLATES entity={r['entity']} seq={r['seq']} path={r['path']} skipped={r['skipped']}")
            else:
                _STATE["chains_requested"] += 1; _STATE["hits"] += int(r["hits"] or 0); _STATE["real"] += int(r["real"])
                _say(f"{PREFIX} TEMPLATES entity={r['entity']} seq={r['seq']} path={r['path']} hits={r['hits']} real={r['real']}/{MAX_SLOTS} dropped={r['dropped']}")
        if bool(use_template) and cen["requested"] == 0:                       # nothing declared: a census line, never a refusal; the item runs untemplated as stock runs it
            _STATE["items_no_templates"] += 1
            _say(f"{PREFIX} TEMPLATES templates=none requested=0 entities={len(bio or [])}")
        if cen["all_dummy"]:                                                  # declared, every slot dummy: named and counted; the item runs exactly as stock runs it
            _STATE["all_dummy_items"] += 1
            ents = ",".join(str(r["entity"]) for r in cen["rows"] if "skipped" not in r)
            reasons = "; ".join(sorted({d["reason"] for d in _STATE["dropped"][-8:]}))[:400]
            _say(f"{PREFIX} TEMPLATES templates=all_dummy entities={ents} hits={sum(int(r.get('hits') or 0) for r in cen['rows'])} real=0/{MAX_SLOTS}"
                 + (f" reasons={reasons}" if reasons else ""))
        return out
    make_template_feature._ptx_template_census = True
    return staticmethod(make_template_feature)


def install() -> str:
    """Arm both sites (now if imported, else at first import; opt_core.autoload.patch_attr_at_import). Returns the applied marker."""
    states = []
    for mod, attr, factory in ((SITE_HITS[0], SITE_HITS[1], _wrap_get_templates), (SITE_FEATURIZER[0], SITE_FEATURIZER[1], _wrap_make_template_feature)):
        p = patch_attr_at_import(mod, attr, factory, tag=TAG, name="template_census")
        states.append(getattr(p, "state", "?"))
    _STATE["installed"] = states
    return f"TEMPLATE_CENSUS:{'+'.join(states)}"


def state() -> dict:
    """End-of-run record: items featurised (of which with no templates declared), requesting chains, hits, real slots, dropped hits (seq +
    reason), items whose every slot came out dummy."""
    return {"installed": list(_STATE["installed"]), "items": _STATE["items"], "items_no_templates": _STATE["items_no_templates"], "chains_requested": _STATE["chains_requested"],
            "chains_skipped_use_template_off": _STATE["chains_skipped_use_template_off"], "hits": _STATE["hits"], "real": _STATE["real"],
            "dropped": len(_STATE["dropped"]), "dropped_reasons": [d["reason"] for d in _STATE["dropped"]][:20],
            "warnings": _STATE["warnings"], "all_dummy_items": _STATE["all_dummy_items"]}


def evidence() -> list:
    """k=v pairs for the FINAL/EXIT evidence: templates_items chains hits real dropped all_dummy."""
    s = state()
    return [("templ_items", s["items"]), ("templ_chains", s["chains_requested"]), ("templ_hits", s["hits"]), ("templ_real", s["real"]),
            ("templ_dropped", s["dropped"]), ("templ_all_dummy", s["all_dummy_items"])]
