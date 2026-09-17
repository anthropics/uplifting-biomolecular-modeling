"""The template census: what a call's query DECLARES about templates, read from the query JSON before the run and printed as ONE line from
the primary process (report.process_role) on every route, stock included — this module changes nothing about templates (upstream OpenFold3
reads the template files a chain names in its own preprocessing); it only counts, so a transcript says how many chains
of the call carried templates and whether the files they name were on disk when the run started:

  ``[<TAG>] TEMPLATES DECLARED queries=<n> chains_templated=<m> chains_untemplated=<u> files_resolved=<r>/<t>[ missing=<path>[,<path>…]]``
or, when the call runs with templates disabled (upstream's ``--use-templates false``: a workload flag — template paths in the query are then not a
declared-templated workload, nothing is judged and the exit rule is unaffected), the one info line
  ``[<TAG>] TEMPLATES DISABLED (--use-templates false): queries_with_template_paths=<n>``

``queries`` = the query set's entries; ``chains_templated`` / ``chains_untemplated`` = its polymer chain ENTRIES (protein / RNA / DNA; a ligand
carries no template) with / without a template key (``template_alignment_file_path`` or ``template_cif_paths``,
openfold3/projects/of3_all_atom/config/inference_query_format.py); ``files_resolved`` = of the ``t`` template files those keys name, the ``r``
that exist (a relative path is taken as given: relative to the working directory, as upstream reads it); ``missing`` names the rest.
The same numbers feed the exit tally (``counters``: templ_declared, templ_untemplated, templ_files, templ_files_resolved) and any lever
that wants a per-call template census (one source: ``census``).

The guard (every route): the model process records, per predicted item, how many of the featurised template slots are REAL
(`observe`: a slot of upstream's `template_pseudo_beta_mask` / `template_backbone_frame_mask` with any non-zero entry — upstream
derives both masks from the loaded template coordinates, so a slot the preprocessor did not fill is all-zero:
core/data/pipelines/featurization/template.py, `~np.isnan(...)`); the primary process then judges (`judge`, from cli.gated_manifest; the
stock child hands its record over in its proof): a query that DECLARED templates (chains_templated >= 1) and reached the model with
real_slots == 0 had its templates DROPPED by upstream's preprocessing (fetch failed, structure absent, alignment unparsable — upstream logs
the cause and carries on untemplated, exit 0). That is a degraded protocol, so it is a NAMED event: one line per query
``[<TAG>] TEMPLATES DROPPED: query=<q> declared chains_templated=<n> real_slots=0 slots=<T> (<reason>)``, the exit census word
``fallbacks=templates_dropped:<reason>=<k>``, and exit code 5 (EXIT_TEMPLATES_DROPPED) after the outputs are written — unless the caller
passed ``--allow-template-drop``, the recorded opt-out (exit 0, the census word kept). A declared-templated query with no featurised record at
all is judged the same way (reason ``unobserved``): the guard never passes what it could not see. The same rule holds PER CHAIN: a query
that kept a real slot while fewer of its chain instances than it declared templated (an entry's `chain_ids` = its instances) carry a real
template entry on their tokens (the batch's `asym_id` against the slot masks) is dropped by name too —
``[<TAG>] TEMPLATES DROPPED: query=<q> declared chains_templated=<n> chain_instances=<i> chains_real=<r> real_slots=<k> slots=<T> (<reason>)``,
census reason ``chain_untemplated``, exit 5 under the same opt-out; its ``TEMPLATES FEATURISED`` line still prints (the query did keep
real slots). The run log's per-chain lines (templ_guard: TEMPLATE INPUT DROPPED / TEMPLATE DROPPED / TEMPLATES chain=) name which chain and why.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Dict, Mapping

from .inputs import POLYMERS

from .report import TAG                                                        # the tag of the package this module lives in (report.TAG), never spelled here
TEMPLATE_FILE_KEYS = ("template_alignment_file_path", "template_cif_paths")     # a chain's template inputs (openfold3/projects/of3_all_atom/config/inference_query_format.py): an alignment file (+ template_entry_chain_ids) or, from upstream 0.5.0 on, CIF files (+ template_cif_chain_ids)
MISSING_NAMED = 5                                                              # how many missing files the line names (the count is always exact)
EXIT_TEMPLATES_DROPPED = 5                                                     # the guard's exit code: declared templates dropped before the model (cli.exit_rule)
TEMPLATE_SLOT_MASKS = ("template_pseudo_beta_mask", "template_backbone_frame_mask")   # upstream's per-slot template masks in the featurised batch ([..., n_templ, n_token]); an unfilled slot is all-zero
DROP_LABEL = "templates_dropped"                                               # the exit census label (report.FALLBACK_SOURCES): fallbacks=templates_dropped:<reason>=<k>
REASON_DUMMY, REASON_UNOBSERVED = "dummy_only", "unobserved"                   # the featurised stack held no real template / the query never reached the recording model wrap
REASON_CHAIN = "chain_untemplated"                                             # a declared-templated chain instance carries no real template entry while the query kept a real slot
ALLOW_FLAG = "--allow-template-drop"                                           # the recorded opt-out (cli pred)
ROW_BORN_FLAG = "_lazy_template_real"                                            # the tp line's per-item flag (tp_rowpair.data.REAL_FLAG): non-zero = the item's real template slots are
                                                                                # computed per rank, per row block, from per-token precursors — form=row_born on the FEATURISED line
_FEATURISED: Dict[str, dict] = {}                                              # item (query id as the batch names it) -> {"slots", "real_slots", "key"}: this process's model calls (observe)
_DROPS: Optional[List[dict]] = None                                            # the last judgement of this process (judge); None before any

_LAST: Optional[dict] = None                                                   # the census of this process's call (record / counters)


def _files(chain: dict) -> List[str]:
    out: List[str] = []
    for key in TEMPLATE_FILE_KEYS:
        v = chain.get(key)
        if isinstance(v, str) and v:
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out += [str(x) for x in v if x]
    return out


def census(query_set: dict) -> dict:
    """The template census of a query set (the dict upstream's --query-json holds): counts only, nothing is modified."""
    queries = query_set.get("queries") or {}
    templated = untemplated = 0
    files: List[str] = []
    templated_queries: Dict[str, int] = {}                                     # query name -> its polymer chain ENTRIES that declare template files (the guard's declared side)
    templated_instances: Dict[str, int] = {}                                   # query name -> the chain INSTANCES of those entries (an entry's `chain_ids` list = its copies; the featurised batch has one asym id each)
    for name, q in queries.items():
        for ch in (q or {}).get("chains") or []:
            if str(ch.get("molecule_type", "")).lower() not in POLYMERS:
                continue
            named = _files(ch)
            if named:
                templated += 1; files += named
                templated_queries[str(name)] = templated_queries.get(str(name), 0) + 1
                templated_instances[str(name)] = templated_instances.get(str(name), 0) + _instances(ch)
            else:
                untemplated += 1
    missing = [f for f in files if not os.path.exists(f)]
    return {"queries": len(queries), "chains_templated": templated, "chains_untemplated": untemplated,
            "template_files": len(files), "template_files_resolved": len(files) - len(missing), "missing": missing,
            "templated_queries": templated_queries, "templated_instances": templated_instances}


def _instances(chain: dict) -> int:
    """A chain entry's instances in the prediction: the length of its `chain_ids` list (upstream's copies of one entry), 1 for a single id."""
    ids = chain.get("chain_ids")
    return max(1, len(ids)) if isinstance(ids, (list, tuple)) else 1


def line(c: dict) -> str:
    """The census line (module docstring): DECLARED when templates are enabled for the call, DISABLED when they are not."""
    if not c.get("enabled", True):
        return f"[{TAG}] TEMPLATES DISABLED (--use-templates false): queries_with_template_paths={len(c['templated_queries'])}"
    s = (f"[{TAG}] TEMPLATES DECLARED queries={c['queries']} chains_templated={c['chains_templated']} chains_untemplated={c['chains_untemplated']} "
         f"files_resolved={c['template_files_resolved']}/{c['template_files']}")
    if c["missing"]:
        more = len(c["missing"]) - MISSING_NAMED
        s += " missing=" + ",".join(c["missing"][:MISSING_NAMED]) + (f",+{more} more" if more > 0 else "")
    return s


def record(query_set: dict, stream=None, enabled: bool = True) -> dict:
    """Take the census of this process's call, keep it for the exit tally, and print the line once from the primary process. `enabled` = the
    call's templates switch (upstream's --use-templates): False makes the query's template paths a non-workload — the DISABLED line, no
    declared-templated query (declared() is empty: nothing is judged, no FEATURISED / DROPPED line, no census fallback word)."""
    global _LAST
    from .report import process_role
    c = census(query_set)
    c["enabled"] = bool(enabled)
    _LAST = c
    if process_role() == "main":
        print(line(c), file=stream or sys.stderr, flush=True)
    return c


def record_file(query_json: Optional[str], stream=None, enabled: bool = True) -> Optional[dict]:
    """`record` on a query JSON path; None (nothing printed) when the file is absent or unreadable — upstream's own error names that."""
    if not query_json:
        return None
    from . import inputs
    try:
        qs = inputs.load(query_json)
    except (OSError, ValueError):
        return None
    return record(qs, stream, enabled)


def counters() -> dict:
    """The exit tally's template fields of this process ({} before any census): templ_declared (chains templated), templ_untemplated,
    templ_files, templ_files_resolved — or, with templates disabled for the call, templ_declared=0 and templ_disabled_paths (the template files
    the query names, unused)."""
    if _LAST is None:
        return {}
    if not _LAST.get("enabled", True):                                          # templates disabled for the call: no declared-templated workload; the paths the query names are counted apart
        return {"templ_declared": 0, "templ_disabled_paths": _LAST["template_files"]}
    return {"templ_declared": _LAST["chains_templated"], "templ_untemplated": _LAST["chains_untemplated"],
            "templ_files": _LAST["template_files"], "templ_files_resolved": _LAST["template_files_resolved"]}


def reset() -> None:
    global _LAST, _DROPS
    _LAST = None
    _DROPS = None
    _FEATURISED.clear()


# ------------------------------------------------------------------------------------------------------------ the guard ----
def declared() -> Dict[str, int]:
    """{query: chains_templated} of this process's call (record / record_file): {} before any census and {} when templates are disabled for the
    call (record enabled=False) — the guard then judges nothing."""
    if not (_LAST or {}).get("enabled", True):
        return {}
    return dict((_LAST or {}).get("templated_queries") or {})


def declared_instances() -> Dict[str, int]:
    """{query: templated chain INSTANCES} of this process's call — the per-chain side of the guard (judge): {} as :func:`declared`."""
    if not (_LAST or {}).get("enabled", True):
        return {}
    return dict((_LAST or {}).get("templated_instances") or {})


def slots_featurised(batch) -> Optional[dict]:
    """{"slots": T, "real_slots": r, "key": <mask name>} read from a featurised predict batch: T = the template slots upstream allocated
    (n_templates, 4 as shipped), r = the slots whose TEMPLATE_SLOT_MASKS entry has any non-zero value (a slot the preprocessor filled from a
    loaded structure). None when the batch carries no template mask (nothing to judge from). Observe-only: the batch is not modified."""
    get = getattr(batch, "get", None)
    if get is None:
        return None
    for key in TEMPLATE_SLOT_MASKS:
        m = get(key)
        if m is None or not hasattr(m, "shape") or len(m.shape) < 2:
            continue
        try:
            t, n = int(m.shape[-2]), int(m.shape[-1])
            flat = m.reshape(-1, t, n)                                       # [batch, slots, tokens]
            per_slot = (flat != 0).any(-1).any(0)                            # [slots]: any real entry in the slot, over the batch
            real = int(per_slot.sum().item() if hasattr(per_slot.sum(), "item") else per_slot.sum())
        except Exception:                                                    # noqa: BLE001 — an unreadable mask is no evidence: the next key, else None
            continue
        rec = {"slots": t, "real_slots": real, "key": key}
        chains = _chains_real(get, flat)
        if chains is not None:
            rec["chains_real"], rec["chains"] = chains
        if _row_born(get):
            rec["form"] = "row_born"
        return rec
    return None


def _chains_real(get, flat) -> Optional[tuple]:
    """(chains_real, chains) from the batch's ``asym_id`` ([..., n_token], upstream's chain-instance id per token, 0 = padding) and the slot
    mask ``flat`` ([batch, slots, tokens]): ``chains`` = the distinct chain instances of the item, ``chains_real`` = those with a real template
    entry on at least one of their tokens in any slot. None when the batch carries no readable asym_id (nothing to judge per chain)."""
    a = get("asym_id")
    if a is None or not hasattr(a, "shape") or int(a.shape[-1]) != int(flat.shape[-1]):
        return None
    try:
        asym = a.reshape(-1, int(a.shape[-1]))[0]                               # [tokens]: one item's ids (seeds of an item share them)
        real_tok = (flat != 0).any(1).any(0)                                     # [tokens]: a real template entry in some slot, over the batch
        ids = {int(x) for x in asym.tolist() if int(x) != 0}
        real_ids = {int(x) for x, r in zip(asym.tolist(), real_tok.tolist()) if r and int(x) != 0}
    except Exception:                                                            # noqa: BLE001 — an unreadable asym_id is no evidence
        return None
    return len(real_ids), len(ids)


def _row_born(get) -> bool:
    """Whether the batch carries the tp line's real-slot flag set (ROW_BORN_FLAG non-zero): its template pair rows are born per rank from
    per-token precursors (tp_rowpair.template.computed_template_rows), never featurised densely. Absent or zero on every other route."""
    f = get(ROW_BORN_FLAG)
    try:
        return f is not None and bool((f.reshape(-1) != 0).any())
    except Exception:                                                        # noqa: BLE001 — an unreadable flag is no evidence
        return False


def observe(item: str, batch) -> Optional[dict]:
    """Record the featurised template slots of one model call under its item name (report's forward wrap calls this once per predicted item,
    in whichever process runs the model: the kit routes' own, the stock child, a tp rank). Several calls of one item (seeds) keep the
    largest real_slots seen. Returns the record, None when the batch carries no template mask."""
    rec = slots_featurised(batch)
    if rec is None:
        return None
    prev = _FEATURISED.get(item)
    if prev is None or (rec["real_slots"], rec.get("chains_real") or 0) >= (prev["real_slots"], prev.get("chains_real") or 0):
        _FEATURISED[item] = rec
    return rec


def featurised() -> Dict[str, dict]:
    """This process's featurised-template record, {item: {"slots", "real_slots", "key"}} (the stock child writes it into its proof)."""
    return {k: dict(v) for k, v in _FEATURISED.items()}


def judge(declared_by_query: Mapping[str, int], featurised_by_item: Mapping[str, dict], instances_by_query: Optional[Mapping[str, int]] = None) -> List[dict]:
    """The guard's verdict: one record per query that DECLARED templates (chains_templated >= 1) and whose featurised template stack held no
    real slot (REASON_DUMMY), that has no featurised record at all (REASON_UNOBSERVED — never passed unseen), or that kept a real slot while
    FEWER chain instances than it declared templated carry a real template entry (REASON_CHAIN: a declared-templated chain reached the model
    untemplated; ``instances_by_query`` = :func:`declared_instances`, this process's census when not given). Item names are upstream's query
    ids as the batch carries them (report.batch_item: spaces -> `_`). Kept for the exit census (fallbacks)."""
    global _DROPS
    instances = declared_instances() if instances_by_query is None else dict(instances_by_query)
    drops: List[dict] = []
    for query, n_chains in sorted((declared_by_query or {}).items()):
        if int(n_chains) < 1:
            continue
        rec = featurised_by_item.get(query) or featurised_by_item.get(str(query).replace(" ", "_"))
        if rec is None:
            drops.append({"query": query, "chains_templated": int(n_chains), "slots": None, "real_slots": 0, "reason": REASON_UNOBSERVED})
        elif int(rec.get("real_slots") or 0) == 0:
            drops.append({"query": query, "chains_templated": int(n_chains), "slots": rec.get("slots"), "real_slots": 0, "reason": REASON_DUMMY})
        elif rec.get("chains_real") is not None and instances.get(query) and int(rec["chains_real"]) < int(instances[query]):
            drops.append({"query": query, "chains_templated": int(n_chains), "slots": rec.get("slots"), "real_slots": int(rec["real_slots"]), "reason": REASON_CHAIN,
                          "chain_instances": int(instances[query]), "chains_real": int(rec["chains_real"])})
    _DROPS = drops
    return drops


REASON_WORDS = {REASON_DUMMY: "the featurised template stack holds no real template — upstream's preprocessing found or loaded none for this query "
                              "(its 'Preprocessing templates' / template-structure lines above name the cause: fetch failed, structure absent, alignment unread)",
                REASON_UNOBSERVED: "no featurised record of this query reached the guard (the model wrap did not see it)",
                REASON_CHAIN: "a chain that declared templates reached the model with no real template entry on its tokens while the query kept a real slot — "
                              "upstream's pipeline dropped that chain's templates (the TEMPLATE INPUT DROPPED / TEMPLATE DROPPED / TEMPLATES chain= lines above name which and why)"}


def drop_lines(drops: List[dict]) -> List[str]:
    """One `TEMPLATES DROPPED` line per dropped query (module docstring)."""
    out: List[str] = []
    for d in drops:
        if d["reason"] == REASON_CHAIN:
            out.append(f"[{TAG}] TEMPLATES DROPPED: query={d['query']} declared chains_templated={d['chains_templated']} chain_instances={d['chain_instances']} "
                       f"chains_real={d['chains_real']} real_slots={d['real_slots']} slots={d['slots']} ({REASON_WORDS[REASON_CHAIN]})")
        else:
            out.append(f"[{TAG}] TEMPLATES DROPPED: query={d['query']} declared chains_templated={d['chains_templated']} real_slots=0 "
                       f"slots={d['slots'] if d['slots'] is not None else 'unknown'} ({REASON_WORDS.get(d['reason'], d['reason'])})")
    return out


def featurised_lines(declared_by_query: Mapping[str, int], featurised_by_item: Mapping[str, dict]) -> List[str]:
    """One positive line per DECLARED-templated query whose featurised stack holds at least one real slot (the success case, printed at judge
    time on every route): `[<TAG>] TEMPLATES FEATURISED query=<q> real_slots=<r>/<T>`, plus ` form=row_born` on the tp line (`big --n_gpu P`:
    the rows of the real slots are computed per rank from per-token precursors — ROW_BORN_FLAG). Nothing for untemplated or dropped queries."""
    out: List[str] = []
    for query, n_chains in sorted((declared_by_query or {}).items()):
        rec = featurised_by_item.get(query) or featurised_by_item.get(str(query).replace(" ", "_"))
        if int(n_chains) >= 1 and rec is not None and int(rec.get("real_slots") or 0) >= 1:
            out.append(f"[{TAG}] TEMPLATES FEATURISED query={query} real_slots={int(rec['real_slots'])}/{rec.get('slots')}"
                       + (f" form={rec['form']}" if rec.get("form") else ""))
    return out


def fallbacks() -> Dict[str, int]:
    """The guard's census for report.FALLBACK_SOURCES: {reason: count} of the last judgement, {} when nothing was dropped or nothing judged."""
    out: Dict[str, int] = {}
    for d in _DROPS or []:
        out[d["reason"]] = out.get(d["reason"], 0) + 1
    return out
