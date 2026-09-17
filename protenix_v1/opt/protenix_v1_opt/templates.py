"""The template guard: a templated run never proceeds silently untemplated (kit-level wrap of the stock's template path; no stock edit).

The stock's template path has two silent branches. (1) `InferenceTemplateFeaturizer.make_template_feature`
(protenix/data/template/template_featurizer.py:708-715) keeps `TemplateSearchResult.features` and discards `.errors` / `.warnings` — the
per-hit reasons `TemplateHitFeaturizer.get_templates` (template_utils.py:860-958) collected when a hit failed the prefilter (:896-900) or the
featurisation (:944-951: alignment / Cα-distance / missing-residue failures): a dropped template leaves no line. (2) When no template
survives for any chain, the inference dataloader substitutes DUMMY template features (protenix/data/inference/infer_dataloader.py:196-208,
protenix/data/utils.py:929-937: restype 31, all-zero atom masks) and the model's TemplateEmbedder runs on them — a run launched with
`--use_template true` computes untemplated and says nothing.

The guard, installed by each mode at activation (`install()`, stack.activate) and consulted at every item (`check_item()`, the
`InferenceRunner.predict` wrap in stack.py):
  - every dropped hit is a NAMED event: `TEMPLATE event=dropped query=<8-hex tag of the chain sequence> kind=error|warning
    reason=<the stock's own text, blank-free>`, and every search prints its census `TEMPLATE event=search query=<tag> hits=<n> kept=<k>
    errors=<e> warnings=<w>`;
  - every item of a templated run prints its slot census `TEMPLATE event=census item=<name> slots=<T> real=<k> dummy=<T-k> [form=row_born]
    per_chain=<asym_id:k,...>` (a real slot = a template whose mask covers at least one token of the chain; `form=row_born` under
    `--n_gpu P`: the item carries the featuriser's per-token precursors and every rank bears its own ROWS of the template pair features,
    tp.template_pair_rows — no rank forms them whole);
  - an item that DECLARED templates (its input JSON entry names a `templatesPath` on at least one protein chain: the upstream's per-chain
    declaration, template_featurizer.py:681) and whose real-slot census is 0 prints `TEMPLATE event=all_dummy item=<name> slots=<T>` and is
    counted (`all_dummy`): the item asked for templates and none reached the model; it runs untemplated exactly as the stock runs it;
  - an item that declares none prints `TEMPLATE event=none item=<name> declared=0` and runs untemplated exactly as the stock runs it (the
    dummy slots; `declared_templates()` reads the entry from `configs.input_json_path`, the JSON the dataloader loads). When that JSON cannot
    be read the census alone decides, as above.
`tally()` is the process record the EXIT line carries (report `templates`; `none` = the items that declared no
template). `--use_template false` (the stock CLI default) = `templates=off`: nothing is censused.
"""
from __future__ import annotations

import json
import zlib
from typing import Dict, Optional

from . import report as R

FEATURIZER_MODULE = "protenix.data.template.template_utils"
FEATURIZER_CLASS = "TemplateHitFeaturizer"
MASK_KEYS = ("template_pseudo_beta_mask", "template_backbone_frame_mask", "template_all_atom_mask")   # [T, N] / [T, N] / [T, N, 37]: the first present is censused
PRECURSOR_KEYS = ("template_aatype", "template_atom_positions", "template_atom_mask")   # the featuriser's per-token precursors (every templated item carries them)
DENSE_KEYS = ("template_distogram", "template_unit_vector")          # the dense pair features [T, N, N, F] the stock featuriser adds at n_gpu 1; absent under n_gpu > 1 (tp: born per rank as rows)
ROW_BORN = "row_born"                                                # the census word of that form: `form=row_born` (absent at n_gpu 1: the stock's dense features)

_TALLY = {"installed": False, "searches": 0, "hits": 0, "kept": 0, "dropped_error": 0, "dropped_warning": 0,
          "items": 0, "items_templated": 0, "none": 0, "slots": 0, "real": 0, "all_dummy": 0}
_DECLARED: Dict[str, list] = {}                                    # input_json_path -> the parsed item list (read once per path)


def tally() -> dict:
    return dict(_TALLY)


def _tag8(s) -> str:
    """An 8-hex-digit tag of a chain sequence (crc32; an identifier for the log line, not a pinned digest)."""
    return format(zlib.crc32(str(s).encode("utf-8")) & 0xFFFFFFFF, "08x")


def _emit(**kv) -> str:
    line = f"{R.PREFIX} TEMPLATE " + " ".join(f"{k}={R._token(v)}" for k, v in kv.items())
    R.log(line)
    return line


def install() -> bool:
    """Wrap TemplateHitFeaturizer.get_templates once per process: the stock call unchanged, its errors/warnings printed as named events.
    Returns True when the wrap is (already) in place; raises ImportError when the stock module is absent (the caller names it)."""
    import importlib
    mod = importlib.import_module(FEATURIZER_MODULE)
    cls = getattr(mod, FEATURIZER_CLASS)
    if getattr(cls.get_templates, "__wrapped_by_kit__", False):
        _TALLY["installed"] = True
        return True
    orig = cls.get_templates

    def get_templates(self, *a, _orig=orig, **kw):
        out = _orig(self, *a, **kw)
        result = out[0] if isinstance(out, tuple) else out
        query = kw.get("query_sequence") or kw.get("sequence_uid") or (a[1] if len(a) > 1 else (a[0] if a else ""))
        hits = kw.get("hits") if "hits" in kw else (a[2] if len(a) > 2 else [])
        errors = list(getattr(result, "errors", None) or [])
        warnings = list(getattr(result, "warnings", None) or [])
        kept = len(getattr(result, "features", None) or [])
        q = _tag8(query)
        _TALLY["searches"] += 1; _TALLY["hits"] += len(hits or []); _TALLY["kept"] += kept
        _TALLY["dropped_error"] += len(errors); _TALLY["dropped_warning"] += len(warnings)
        for e in errors:
            _emit(event="dropped", query=q, kind="error", reason=e)
        for w in warnings:
            _emit(event="dropped", query=q, kind="warning", reason=w)
        _emit(event="search", query=q, hits=len(hits or []), kept=kept, errors=len(errors), warnings=len(warnings))
        return out

    get_templates.__wrapped__ = orig
    get_templates.__wrapped_by_kit__ = True
    cls.get_templates = get_templates
    _TALLY["installed"] = True
    return True


def _cfg_get(configs, name: str, default=None):
    if configs is None:
        return default
    if isinstance(configs, dict):
        return configs.get(name, default)
    try:
        return configs.get(name, default)          # ml_collections ConfigDict
    except Exception:
        return getattr(configs, name, default)


def use_template(configs) -> bool:
    v = _cfg_get(configs, "use_template", False)
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _as_bool_rows(mask):
    """[T, ...] mask -> list of T flat lists of 0/1 over tokens (any trailing dims reduced with any())."""
    try:
        import torch
        if isinstance(mask, torch.Tensor):
            m = mask.detach()
            while m.dim() > 2:
                m = m.any(dim=-1) if m.dtype == torch.bool else (m != 0).any(dim=-1)
            m = (m != 0) if m.dtype != torch.bool else m
            return [[bool(x) for x in row] for row in m.cpu().tolist()] if m.dim() == 2 else [[bool(x) for x in m.cpu().tolist()]]
    except ImportError:
        pass
    rows = mask.tolist() if hasattr(mask, "tolist") else mask

    def flat_any(x):
        if isinstance(x, (list, tuple)):
            return any(flat_any(y) for y in x)
        return bool(x)
    return [[flat_any(tok) for tok in row] for row in rows]


def census(feats: dict) -> Optional[dict]:
    """The slot census of one item's feature dict: {key, slots T, real k, dummy, per_template [bool], per_chain {asym_id: k}}; None when the
    dict carries no template mask (an untemplated feature dict)."""
    key = next((k for k in MASK_KEYS if k in feats), None)
    if key is None:
        return None
    rows = _as_bool_rows(feats[key])
    T = len(rows)
    per_template = [any(r) for r in rows]
    asym = feats.get("asym_id")
    per_chain: Dict[str, int] = {}
    if asym is not None:
        ids = asym.detach().cpu().tolist() if hasattr(asym, "detach") else list(asym)
        ids = [int(round(float(x))) for x in ids]
        for a in sorted(set(ids)):
            idx = [i for i, x in enumerate(ids) if x == a]
            per_chain[str(a)] = sum(1 for r in rows if any(r[i] for i in idx if i < len(r)))
    real = sum(per_template)
    return {"key": key, "slots": T, "real": real, "dummy": T - real, "per_template": per_template, "per_chain": per_chain}


def declared_templates(configs, data, name: Optional[str]) -> Optional[int]:
    """How many protein chains of this item's input JSON entry name a `templatesPath` (the upstream's per-chain template declaration:
    InferenceTemplateFeaturizer.make_template_feature searches only those chains). The entry is the one the dataloader featurised —
    `configs.input_json_path`, matched by `sample_index`, else by name. None when the JSON or the entry cannot be read (the census alone
    decides then)."""
    path = _cfg_get(configs, "input_json_path")
    if not path:
        return None
    try:
        items = _DECLARED.get(str(path))
        if items is None:
            with open(str(path), "r", encoding="utf-8") as fh:
                items = json.load(fh)
            items = items if isinstance(items, list) else [items]
            _DECLARED[str(path)] = items
        idx = data.get("sample_index") if isinstance(data, dict) else None
        idx = int(idx) if idx is not None and not isinstance(idx, bool) and str(idx).strip().lstrip("-").isdigit() else None
        entry = items[idx] if idx is not None and 0 <= idx < len(items) and (name is None or items[idx].get("name") == name) else None
        if entry is None:
            named = [e for e in items if isinstance(e, dict) and e.get("name") == name]
            if len(named) != 1:
                return None
            entry = named[0]
        return sum(1 for s in (entry.get("sequences") or []) if isinstance(s, dict) and "proteinChain" in s
                   and str((s["proteinChain"] or {}).get("templatesPath") or "").strip())
    except Exception:
        return None


def check_item(configs, data, item: Optional[str] = None) -> Optional[dict]:
    """At an item's entry: nothing when the run is untemplated; `TEMPLATE event=none` (tallied `none`) and nothing else when the item declares
    no template — it runs untemplated exactly as the stock runs it; else the census line (per chain), tallied, and `TEMPLATE event=all_dummy`
    (tallied `all_dummy`) for an item whose declared templates all dropped — it runs untemplated as the stock runs it. Returns the census dict
    (or None)."""
    _TALLY["items"] += 1
    if not use_template(configs):
        return None
    feats = data.get("input_feature_dict", data) if isinstance(data, dict) else data
    c = census(feats if isinstance(feats, dict) else {})
    name = item or (data.get("sample_name") if isinstance(data, dict) else None) or f"item{_TALLY['items']}"
    if declared_templates(configs, data, name) == 0:                # use_template=true names the run, `templatesPath` names the item: none declared here
        _TALLY["none"] += 1
        _emit(event="none", item=name, declared=0)
        return None
    if c is None:
        c = {"key": None, "slots": 0, "real": 0, "dummy": 0, "per_template": [], "per_chain": {}}
    _TALLY["items_templated"] += 1; _TALLY["slots"] += c["slots"]; _TALLY["real"] += c["real"]
    form = {"form": ROW_BORN} if isinstance(feats, dict) and all(k in feats for k in PRECURSOR_KEYS) and not any(k in feats for k in DENSE_KEYS) else {}   # n_gpu > 1: precursors and no dense pair feature — each rank bears its template pair ROWS (tp.template_pair_rows)
    _emit(event="census", item=name, slots=c["slots"], real=c["real"], dummy=c["dummy"], **form,
          per_chain=",".join(f"{a}:{k}" for a, k in c["per_chain"].items()) or "-", mask=c["key"] or "none")
    if c["real"] == 0:                                              # every declared template dropped: named and counted; the item runs untemplated as the stock runs it
        _TALLY["all_dummy"] += 1
        _emit(event="all_dummy", item=name, slots=c["slots"])
    return c


def exit_fields() -> dict:
    """The EXIT-line fields: templates=off|on and, when on (an item was checked under use_template=true), the process tallies; tmpl_items /
    tmpl_slots / tmpl_real count the items that declared templates, tmpl_none those that declared none, tmpl_all_dummy (present only when
    non-zero) the items whose declared templates all dropped."""
    t = tally()
    if not t["items_templated"] and not t["none"]:
        return {"templates": "off"}
    return {"templates": "on", "tmpl_items": t["items_templated"], "tmpl_slots": t["slots"], "tmpl_real": t["real"],
            "tmpl_searches": t["searches"], "tmpl_hits": t["hits"], "tmpl_kept": t["kept"],
            "tmpl_dropped": t["dropped_error"] + t["dropped_warning"], "tmpl_none": t["none"], **({"tmpl_all_dummy": t["all_dummy"]} if t["all_dummy"] else {})}
