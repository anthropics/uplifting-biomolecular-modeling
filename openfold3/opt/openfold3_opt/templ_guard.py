"""The template guard: what upstream OpenFold3's template pipeline does with a query's DECLARED templates, NAMED per chain and per
template. This file is written once for upstream 0.4.1 and upstream 0.5.0 alike; the words that differ by upstream version come from the
package's ``report`` module — ``TAG`` and ``TEMPL_POLICY``, the engine's one template behaviour at inference and where it comes from:

* ``("ignore", "upstream_0.4.1")`` — stock 0.4.1's ``InferenceDataset`` calls ``sample_templates(..., template_cache_directory=None)``
  and ``sample_templates`` returns ``{}`` unless that directory is set, so templates are PARSED (preprocessed under ``--use-templates
  true``) and not consumed: the prediction runs on the dummy template slots. An applied upstream fix that consumes templates in this
  process says so through :func:`consumed_by` (``cli.apply_upstream_fix``): the words become ``consume(<ID>)``.
* ``("consume", "upstream_0.5.0")`` — stock 0.5.0 hands its template cache directory to the sampler: a templated chain's top-k
  templates reach the featurizer on every route.

No line consumes or refuses templates by itself and the kit has no switch that changes either; this module wraps the stock functions at
import (no stock edit; the stock pin holds) in every active mode, to NAME what happens — events only, nothing here raises into the run:

* per parsed query set that declares template inputs (the query json's ``template_alignment_file_path``, or from 0.5.0 on
  ``template_cif_paths``), ONE line ``[<TAG>] TEMPLATES PARSED queries=<n> chains=<m>: <declared_words()>`` (``wrap_parse``,
  ``TemplatePreprocessor._parse_inference_query_set``);
* per declared chain the preprocessor left untemplated, the named event ``TEMPLATE INPUT DROPPED query=<q> chain=<ids> reason=<word>``
  (``wrap_update``, ``TemplatePreprocessor._update_inference_query_set``; stock only sets the chain's template path to None there,
  preprocessing/template.py 'No templates for chains whose preprocessing fails'). The word names the cache entry the engine looked for —
  ``cache_key_miss:<key>.npz`` (0.4.1 keys ``<sha256(sequence)>.npz``, 0.5.0 ``seq-<sha>.aln-<sha>.npz`` / ``.cif-<…>``) — or says the
  directory holds the sequence's entry under the other upstream version's key, ``of3_keyed_cache_entry:<sha>.npz`` (never adopted or re-keyed);
* per templated chain that reaches the sampler WITHOUT a template cache directory (the ignore policy's shape): the stock call untouched,
  then the census ``TEMPLATES chain=<id> ids=<n> sampled=0: <NOT_CONSUMED>`` (``wrap_sample``);
* for a sampler call WITH a template cache directory (0.5.0's own dataset; under 0.4.1 a consuming upstream fix or caller): stock drops a
  template SILENTLY when its alignment cannot be laid onto the deposited model — ``map_token_pos_to_template_residues``
  (primitives/structure/template.py) maps the query tokens onto the template's residues through the cache's ``idx_map`` (SEQRES
  numbering); when the idx_map covers residues NOT resolved in the deposited template model, ``residue_starts.shape != repeats.shape``
  and the function appends nothing ('Skip if still misaligned'). ``wrap_map`` names each drop ``TEMPLATE DROPPED chain=<id>
  template=<pdb_chain> reason=<word>`` (``unresolved_aligned_residues=n/N``: n of the N aligned idx_map rows point at residues absent
  from the template model; ``residue_token_shape_mismatch=<residues>/<tokens>`` when every aligned residue is present and the shapes
  still differ), ``wrap_align`` prints the per-chain census ``TEMPLATES chain=<id> real=<k>/<N> dropped=<d>`` (k real slots of N
  sampled) and ``wrap_process`` the per-query event ``TEMPLATE SLOTS ALL DUMMY (<chain>:<k>/<N> …)`` when templates were sampled and no
  chain kept a real slot.

A query with no templates configured (N = 0 for every chain, e.g. under ``--use-templates false``) passes through untouched, and the guard
changes no numeric on any query. ``CENSUS`` accumulates what THIS process saw (featurization runs in DataLoader workers when
``data_module_args.num_workers > 0``; the per-event lines reach the run log from any process, the counters below are the in-process view
the exit tally prints). Counts close: ``templ_parsed = templ_alive + templ_input_dropped`` (preprocessor view), ``templ_sampled =
templ_real + templ_dropped`` (slots; 0 under the ignore policy's own sampler call).
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Dict, Optional, Tuple

from .report import TAG, TEMPL_POLICY                     # the package's tag and its engine's template behaviour word (the two words of this file that its report module supplies)

STRUCT_MODULE = "openfold3.core.data.primitives.structure.template"          # map_token_pos_to_template_residues, align_template_to_query, sample_templates
PIPELINE_MODULE = "openfold3.core.data.pipelines.sample_processing.template"  # process_template_structures_of3 (reads align_template_to_query / sample_templates as globals)
PREPROC_MODULE = "openfold3.core.data.pipelines.preprocessing.template"       # TemplatePreprocessor (_parse_inference_query_set / _update_inference_query_set)
POLICY: Tuple[str, str] = tuple(TEMPL_POLICY)           # (word, source): the engine's one template behaviour on every line — ("ignore", "upstream_0.4.1") | ("consume", "upstream_0.5.0") — the exit tally's templ_policy=<word>(<source>)
_SOURCE_WORDS = POLICY[1].replace("_", " ")               # `upstream 0.4.1` / `upstream 0.5.0` as the census sentences spell it
NOT_CONSUMED = f"templates parsed; not consumed at inference ({_SOURCE_WORDS} behaviour)"     # the ignore policy's words (TEMPLATES PARSED / the sampler census without a cache directory)
CONSUMED_NATIVELY = f"templates parsed; consumed at inference ({_SOURCE_WORDS} behaviour)"    # the consume policy's words when the engine itself consumes (no upstream fix involved)
ALL_DUMMY = "TEMPLATE SLOTS ALL DUMMY"

CENSUS = {"installed": False, "declared": 0, "alive": 0, "input_dropped": 0, "ignored": 0, "sampled": 0, "real": 0, "dropped": 0,
          "all_dummy": 0, "events": []}
DECLARED: Dict[Tuple[str, int], dict] = {}              # (query name, chain index) -> {"chain": ids, "aln": path, "ids": declared template ids}: the preprocessor's declared registry
_LOCK = threading.Lock()
_CHAIN = threading.local()          # the per-query chain censuses collected between process_template_structures_of3 entry and exit


_CONSUMED_BY = None                 # the ID of an applied upstream fix that consumes templates at inference (cli.apply_upstream_fix -> consumed_by); None unless --upstream-fix names one


def consumed_by(fix_id: str) -> None:
    """Told by the ``--upstream-fix`` plumbing (cli.apply_upstream_fix) that an applied fix consumes templates at inference in this process
    (OF3-001): the census words then state it — :func:`policy` = ``("consume", <ID>)`` (the tally's ``templ_policy=consume(OF3-001)``) and the
    ``TEMPLATES PARSED`` line says consumed. No line, lever switch or variable reaches this; nothing here consumes anything itself."""
    global _CONSUMED_BY
    _CONSUMED_BY = str(fix_id)


def policy() -> Tuple[str, str]:
    """(word, source): the engine's :data:`POLICY` on every line — ``("ignore", "upstream_0.4.1")``: parsed, not consumed at inference;
    ``("consume", "upstream_0.5.0")``: consumed — unless an applied upstream fix consumes them in this process (:func:`consumed_by`): then
    ``("consume", <ID>)``."""
    return ("consume", _CONSUMED_BY) if _CONSUMED_BY else POLICY


def declared_words() -> str:
    """The tail of the ``TEMPLATES PARSED`` line: that an applied upstream fix consumes the templates, else the engine's own behaviour
    (:data:`CONSUMED_NATIVELY` under a consume policy, :data:`NOT_CONSUMED` under an ignore policy)."""
    if _CONSUMED_BY:
        return f"templates parsed; consumed at inference under --upstream-fix {_CONSUMED_BY}"
    return CONSUMED_NATIVELY if POLICY[0] == "consume" else NOT_CONSUMED


def _log(msg: str) -> None:
    sys.stderr.write(f"[{TAG}] {msg}\n")
    sys.stderr.flush()


def _chain_id(atom_array) -> str:
    try:
        ids = atom_array.chain_id
        return str(ids[0]) if len(ids) else "?"
    except Exception:  # noqa: BLE001
        return "?"


def drop_reason(template_cache_entry, atom_array_query_chain, atom_array_template_chain) -> str:
    """Why stock dropped this template: n of the N aligned idx_map rows name template residues absent from the template model."""
    import numpy as np
    idx_map = np.asarray(template_cache_entry.idx_map)
    n_rows = int(idx_map.shape[0]) if idx_map.ndim == 2 else 0
    try:
        tmpl_res = np.unique(np.asarray(atom_array_template_chain.res_id).astype(int))
        unresolved = int((~np.isin(idx_map[:, 1].astype(int), tmpl_res)).sum()) if n_rows else 0
    except Exception:  # noqa: BLE001
        unresolved = -1
    if unresolved:
        return f"unresolved_aligned_residues={unresolved}/{n_rows}"
    try:
        n_tok = int(np.isin(np.unique(np.asarray(atom_array_query_chain.res_id)), idx_map[:, 0]).sum()) if n_rows else 0
    except Exception:  # noqa: BLE001
        n_tok = -1
    return f"residue_token_shape_mismatch={len(tmpl_res) if unresolved == 0 else '?'}/{n_tok}"


def wrap_map(orig):
    """map_token_pos_to_template_residues: the stock call, then ONE named event when it appended nothing."""
    def map_token_pos_to_template_residues(template_slices, template_cache_entry, atom_array_query_chain, atom_array_template_chain):
        before = len(template_slices)
        out = orig(template_slices, template_cache_entry, atom_array_query_chain, atom_array_template_chain)
        if len(template_slices) == before:
            reason = drop_reason(template_cache_entry, atom_array_query_chain, atom_array_template_chain)
            tmpl = getattr(template_cache_entry, "template_pdb_chain_id", None) or f"index{getattr(template_cache_entry, 'index', '?')}"
            ev = {"chain": _chain_id(atom_array_query_chain), "template": str(tmpl), "reason": reason}
            with _LOCK:
                CENSUS["dropped"] += 1
                CENSUS["events"].append(ev)
            drops = getattr(_CHAIN, "drops", None)
            if drops is not None:
                drops.append(ev)
            _log(f"TEMPLATE DROPPED chain={ev['chain']} template={ev['template']} reason={reason}")
        return out
    map_token_pos_to_template_residues.__wrapped__ = orig
    map_token_pos_to_template_residues._of3opt_guard = True
    return map_token_pos_to_template_residues


def wrap_align(orig):
    """align_template_to_query: the per-chain census real=k/N (N = templates sampled for the chain, k = slots kept)."""
    def align_template_to_query(*args, **kwargs):
        sampled = kwargs.get("sampled_template_data", args[0] if args else None) or {}
        query_chain = kwargs.get("atom_array_query_chain", args[5] if len(args) > 5 else None)
        n = len(sampled)
        prev = getattr(_CHAIN, "drops", None)
        _CHAIN.drops = []
        try:
            slices = orig(*args, **kwargs)
            drops = _CHAIN.drops
        finally:
            _CHAIN.drops = prev
        k = len(slices) if slices is not None else 0
        if n:
            chain = _chain_id(query_chain) if query_chain is not None else "?"
            with _LOCK:
                CENSUS["sampled"] += n
                CENSUS["real"] += k
            rows = getattr(_CHAIN, "chains", None)
            if rows is not None:
                rows.append({"chain": chain, "real": k, "sampled": n, "dropped": drops})
            _log(f"TEMPLATES chain={chain} real={k}/{n} dropped={len(drops)}")
        return slices
    align_template_to_query.__wrapped__ = orig
    align_template_to_query._of3opt_guard = True
    return align_template_to_query


SAMPLE_PARAMS = ("assembly_data", "template_cache_directory", "n_templates", "take_top_k", "chain_id", "template_structure_array_directory",
                 "template_file_format", "use_roda_monomer_format")            # stock sample_templates' parameters in order (primitives/structure/template.py `sample_templates`; the same in 0.4.1 and 0.5.0)


def _templated_chain(chain_data) -> bool:
    """An inference chain that arrives with a preprocessed template cache entry and template ids (the preprocessor's output for a declared chain)."""
    return bool(chain_data) and chain_data.get("cache_entry_file_path") is not None and bool(chain_data.get("template_ids"))


def wrap_sample(orig):
    """sample_templates: the stock call, arguments untouched. For a templated inference chain (cache entry + ids,
    ``template_cache_directory=None``: stock returns ``{}``) the census ``TEMPLATES chain=<id> ids=<n> sampled=<k>: <NOT_CONSUMED>``;
    untemplated chains and a caller's own template cache directory: nothing added."""
    def sample_templates(*args, **kwargs):
        out = orig(*args, **kwargs)
        bound = dict(zip(SAMPLE_PARAMS, args))
        bound.update(kwargs)
        cid = bound.get("chain_id")
        chain_data = (bound.get("assembly_data") or {}).get(cid, {}) if cid is not None else {}
        if _templated_chain(chain_data) and bound.get("template_cache_directory") is None:
            with _LOCK:
                CENSUS["ignored"] += 1
            _log(f"TEMPLATES chain={cid} ids={len(chain_data['template_ids'])} sampled={len(out or {})}: {NOT_CONSUMED}")
        return out
    sample_templates.__wrapped__ = orig
    sample_templates._of3opt_guard = True
    return sample_templates


def _chain_label(chain) -> str:
    ids = getattr(chain, "chain_ids", None)
    return "/".join(str(i) for i in ids) if ids else "?"


def wrap_parse(orig):
    """TemplatePreprocessor._parse_inference_query_set: records every chain that declares template inputs (DECLARED) and prints ONE
    line ``TEMPLATES PARSED queries=<n> chains=<m>: <declared_words()>``; then the stock parse. Never refuses."""
    def _parse_inference_query_set(self):
        declared = {}
        for qn, q in self.input_set.queries.items():
            for idx, ch in enumerate(q.chains):
                aln, cifs = getattr(ch, "template_alignment_file_path", None), getattr(ch, "template_cif_paths", None)   # an alignment file, or (0.5.0 on) template CIF files
                if getattr(ch, "molecule_type", None) in self.moltypes and (aln is not None or cifs):
                    declared[(str(qn), idx)] = {"chain": _chain_label(ch), "aln": str(aln) if aln is not None else None, "cif": [str(c) for c in (cifs or [])],
                                                "ids": list(getattr(ch, "template_entry_chain_ids", None) or []), "seq": getattr(ch, "sequence", None)}
        if declared:
            with _LOCK:
                DECLARED.update(declared)
                CENSUS["declared"] += len(declared)
            _log(f"TEMPLATES PARSED queries={len({qn for qn, _ in declared})} chains={len(declared)}: {declared_words()}")
        return orig(self)
    _parse_inference_query_set.__wrapped__ = orig
    _parse_inference_query_set._of3opt_guard = True
    return _parse_inference_query_set


def cache_entry_name(mod, ch, d: dict) -> Optional[str]:
    """The file name of the chain's template cache entry as the ENGINE keys it, from the preprocessing module's own statements: 0.5.0's
    ``build_template_cache_key`` (``seq-<sha256(sequence)>.aln-<sha256(alignment file contents)>`` / ``.cif-<…>``) when the module has it,
    else 0.4.1's ``get_sequence_hash(sequence)``; None when neither can be computed here (the declared file unreadable, no sequence)."""
    try:
        build = getattr(mod, "build_template_cache_key", None)
        if build is not None:
            key = build(sequence=d.get("seq"), aln_path=d.get("aln"), cif_paths=d.get("cif") or None, cif_chain_ids=getattr(ch, "template_cif_chain_ids", None))
        else:
            key = mod.get_sequence_hash(d["seq"]) if d.get("seq") is not None else None
    except Exception:                                                           # noqa: BLE001 — a key the engine cannot compute here is reported as absent, never guessed
        return None
    return f"{key}.npz" if key else None


def absent_entry_reason(pre, ch, d: dict) -> str:
    """The reason word of a declared chain the preprocessor left WITHOUT a cache entry (stock sets its template path to None when
    ``<cache_directory>/<key>.npz`` does not exist after preprocessing): ``cache_key_miss:<key>.npz`` names the entry the engine looked for;
    ``of3_keyed_cache_entry:<sha>.npz`` says the directory holds this sequence's entry under the OTHER upstream version's key (0.4.1 keys an
    entry ``<sha256(sequence)>.npz``, 0.5.0 ``seq-<sha>.aln-<sha>.npz`` — a cache directory filled under one version is never read under the other,
    and the kit does not re-key it); ``preprocess_failed`` when the key cannot be computed here."""
    mod = sys.modules.get(PREPROC_MODULE)
    cache_dir = getattr(pre, "cache_directory", None)
    name = cache_entry_name(mod, ch, d) if mod is not None else None
    if name is None or cache_dir is None:
        return "preprocess_failed"
    other = getattr(mod, "get_sequence_hash", None)
    if getattr(mod, "build_template_cache_key", None) is not None and other is not None and d.get("seq") is not None:
        try:
            old = os.path.join(str(cache_dir), f"{other(d['seq'])}.npz")             # upstream 0.4.1's key of the same sequence
            if os.path.isfile(old):
                return f"of3_keyed_cache_entry:{os.path.basename(old)}"
        except Exception:                                                        # noqa: BLE001
            pass
    return f"cache_key_miss:{name}"


def wrap_update(orig):
    """TemplatePreprocessor._update_inference_query_set: the stock update, then per declared chain the named event when the preprocessor
    left it untemplated (``TEMPLATE INPUT DROPPED query=<q> chain=<ids> reason=<word>``; :func:`absent_entry_reason`:
    ``cache_key_miss:<key>.npz`` | ``of3_keyed_cache_entry:<sha>.npz`` | ``preprocess_failed``, or ``no_ids_in_cache`` when the entry
    names no template). Never refuses."""
    def _update_inference_query_set(self):
        orig(self)
        with _LOCK:
            declared = {k: v for k, v in DECLARED.items() if k[0] in {str(q) for q in self.input_set.queries}}
        if not declared:
            return
        alive = 0
        for (qn, idx), d in sorted(declared.items()):
            q = self.input_set.queries.get(qn) if hasattr(self.input_set.queries, "get") else self.input_set.queries[qn]
            ch = q.chains[idx]
            if getattr(ch, "template_alignment_file_path", None) is None:
                reason = absent_entry_reason(self, ch, d)
            elif not getattr(ch, "template_entry_chain_ids", None):
                reason = "no_ids_in_cache"
            else:
                alive += 1
                continue
            with _LOCK:
                CENSUS["input_dropped"] += 1
                CENSUS["events"].append({"chain": d["chain"], "template": "-", "reason": reason, "query": qn})
            _log(f"TEMPLATE INPUT DROPPED query={qn} chain={d['chain']} reason={reason}")
        with _LOCK:
            CENSUS["alive"] += alive
    _update_inference_query_set.__wrapped__ = orig
    _update_inference_query_set._of3opt_guard = True
    return _update_inference_query_set


def wrap_process(orig):
    """process_template_structures_of3: collects the per-chain censuses of one query and prints the per-query event ``TEMPLATE SLOTS ALL
    DUMMY (<chain>:<k>/<N> …; dropped: …)`` when templates were sampled (a caller's own cache directory) and no chain kept a real slot.
    An event, not a refusal: the stock result is returned either way."""
    def process_template_structures_of3(*args, **kwargs):
        prev = getattr(_CHAIN, "chains", None)
        _CHAIN.chains = []
        try:
            out = orig(*args, **kwargs)
            rows = _CHAIN.chains
        finally:
            _CHAIN.chains = prev
        sampled = sum(r["sampled"] for r in rows)
        real = sum(r["real"] for r in rows)
        if sampled and not real:
            with _LOCK:
                CENSUS["all_dummy"] += 1
            census = " ".join(f"{r['chain']}:{r['real']}/{r['sampled']}" for r in rows)
            drops = "; ".join(f"{d['chain']}/{d['template']}:{d['reason']}" for r in rows for d in r["dropped"])
            _log(f"{ALL_DUMMY} ({census}; dropped: {drops or 'none named'})")
        return out
    process_template_structures_of3.__wrapped__ = orig
    process_template_structures_of3._of3opt_guard = True
    return process_template_structures_of3


def patch_struct_module(mod) -> None:
    if not getattr(mod.map_token_pos_to_template_residues, "_of3opt_guard", False):
        mod.map_token_pos_to_template_residues = wrap_map(mod.map_token_pos_to_template_residues)
    if not getattr(mod.align_template_to_query, "_of3opt_guard", False):
        mod.align_template_to_query = wrap_align(mod.align_template_to_query)
    if not getattr(mod.sample_templates, "_of3opt_guard", False):
        mod.sample_templates = wrap_sample(mod.sample_templates)


def _rebind_global(mod, struct, name, wrap) -> None:
    """The pipeline module imported `name` from the struct module as a global: bind the struct module's guarded function (one wrapper), else wrap its own."""
    cur = getattr(mod, name)
    if getattr(cur, "_of3opt_guard", False):
        return
    theirs = getattr(struct, name, None) if struct is not None else None
    setattr(mod, name, theirs if theirs is not None and getattr(theirs, "_of3opt_guard", False) else wrap(cur))


def patch_pipeline_module(mod) -> None:
    struct = sys.modules.get(STRUCT_MODULE)
    _rebind_global(mod, struct, "align_template_to_query", wrap_align)
    _rebind_global(mod, struct, "sample_templates", wrap_sample)
    if not getattr(mod.process_template_structures_of3, "_of3opt_guard", False):
        mod.process_template_structures_of3 = wrap_process(mod.process_template_structures_of3)


def patch_preproc_module(mod) -> None:
    TP = mod.TemplatePreprocessor
    if not getattr(TP._parse_inference_query_set, "_of3opt_guard", False):
        TP._parse_inference_query_set = wrap_parse(TP._parse_inference_query_set)
    if not getattr(TP._update_inference_query_set, "_of3opt_guard", False):
        TP._update_inference_query_set = wrap_update(TP._update_inference_query_set)


_PATCHERS = {STRUCT_MODULE: patch_struct_module, PIPELINE_MODULE: patch_pipeline_module, PREPROC_MODULE: patch_preproc_module}


class _TemplGuardFinder:
    """Meta-path finder that patches the two stock modules right after their bodies run; delegates the find to the other finders."""

    def find_spec(self, fullname, path=None, target=None):
        patch = _PATCHERS.get(fullname)
        if patch is None:
            return None
        spec = None
        for finder in sys.meta_path:
            if finder is self or type(finder).__name__ == type(self).__name__:      # never delegate to another finder of this kind (a re-imported package's stale instance)
                continue
            try:
                spec = finder.find_spec(fullname, path, target)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                break
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig = spec.loader.exec_module

        def exec_module(module, _orig=orig, _patch=patch):
            _orig(module)
            _patch(module)
        spec.loader.exec_module = exec_module
        return spec


FINDER = _TemplGuardFinder()


def install() -> dict:
    """Idempotent: patch the modules already imported and arm the ONE finder for those not yet imported (any other finder of this kind on
    sys.meta_path — a re-imported package's stale instance — is dropped first). Returns CENSUS."""
    with _LOCK:
        for name, patch in _PATCHERS.items():
            mod = sys.modules.get(name)
            if mod is not None:
                patch(mod)
        sys.meta_path[:] = [f for f in sys.meta_path if f is FINDER or type(f).__name__ != type(FINDER).__name__]   # one finder of this kind per process: a stale instance (a re-imported package) is dropped
        if FINDER not in sys.meta_path:
            sys.meta_path.insert(0, FINDER)
        CENSUS["installed"] = True
    return CENSUS


def uninstall() -> None:
    """Disarm the finder and mark the guard off for this process (the wrappers already bound stay: they are inert without templates)."""
    with _LOCK:
        try:
            sys.meta_path.remove(FINDER)
        except ValueError:
            pass
        CENSUS["installed"] = False


def counters() -> dict:
    """The exit tally's fields: what this process saw — templ_guard=on|off; ``templ_policy=<word>(<source>)`` (the engine's one behaviour,
    :data:`POLICY`; ``consume(<ID>)`` when an applied upstream fix consumes templates in this process, :func:`consumed_by`);
    chains: declared / alive after preprocessing / input_dropped, ignored at the sampler (templated, not consumed); slots (a consuming
    sampler call only): sampled / real / dropped; templ_all_dummy events."""
    pol, src = policy()
    return {"templ_guard": "on" if CENSUS["installed"] else "off", "templ_policy": f"{pol}({src})",
            "templ_parsed": CENSUS["declared"], "templ_alive": CENSUS["alive"], "templ_input_dropped": CENSUS["input_dropped"],
            "templ_ignored": CENSUS["ignored"], "templ_sampled": CENSUS["sampled"], "templ_real": CENSUS["real"], "templ_dropped": CENSUS["dropped"],
            "templ_all_dummy": CENSUS["all_dummy"]}
