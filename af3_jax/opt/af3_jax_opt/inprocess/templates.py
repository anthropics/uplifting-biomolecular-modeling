"""Template census of the model process (loaded by file path by the kit's launchers; standard library at module level, numpy inside
the functions).

``alphafold3/model/features.py`` ``Templates.compute_features`` turns each declared template of a protein chain into one slot of
``template_aatype / template_atom_positions / template_atom_mask`` (at most ``max_templates``; fewer are zero-padded); an aligned residue
the template structure does not resolve keeps a zero atom mask, and a chain that is not a protein, has <= 4 tokens or is outside the crop
gets empty template features whatever it declared — all without a log line. A slot whose atom mask is all zero contributes nothing in the
template embedder. :func:`install` wraps ``Templates.compute_features`` once (the stock classmethod runs unchanged; the census reads its
inputs and its result) and prints, per featurised input (monitoring only — the run proceeds exactly as stock does):

    [af3-jax-opt] TEMPLATES chain=<id> declared=<n> kept=<k> real=<r>/<max_templates> dropped=<slot>:<reason>[,...]|none
    [af3-jax-opt] TEMPLATES census declared=<D> kept=<K> real=<R> chains_templated=<c> chains_all_dummy=<z>[ form=row_born] verdict=ok|untemplated|all_dummy

one ``chain=`` line per chain that declared >= 1 template, then the census line. Reasons: ``over_max_templates`` (declared beyond
``max_templates``: the first ones are kept), ``chain_skipped:<not_protein|le_4_tokens|not_in_crop>`` (the empty-features branch),
``unresolved_aligned_residues=<u>/<a>`` (a kept slot whose aligned residues resolve no atom). ``verdict=untemplated`` = no template declared
on any chain; ``all_dummy`` = a chain declared >= 1 template and none of its slots is real — a census word, not a refusal.
``form=row_born`` appears only under ``--n_gpu P``: the launcher calls :func:`set_form` once the row-sharded pair stack is installed
(inprocess/rowpair.py), where each device forms the template pair features for its row block only. :func:`census` is the pure-Python core
(no numpy, no alphafold3 import).
"""
import sys

PREFIX = "[af3-jax-opt]"
FORMS = (None, "row_born")                                                # None = one device: no form= word on the census line
_STATE = {"installed": False, "stock": None, "censuses": [], "form": None}


def set_form(form):
    """The placement word of the census line: the memory mode's launcher sets ``row_born`` once the row-sharded pair stack is installed."""
    if form not in FORMS:
        raise ValueError(f"templates.set_form: {form!r} is not one of {FORMS}")
    _STATE["form"] = form


def _skip_reason(chain_type, n_tokens, in_crop, protein_chain_type):
    if chain_type != protein_chain_type:
        return "not_protein"
    if n_tokens <= 4:
        return "le_4_tokens"
    if not in_crop:
        return "not_in_crop"
    return None


def census(chains, resolved, max_templates, form=None):
    """The per-chain template census of one featurised input (pure Python: no numpy, no alphafold3 import).

    ``chains``: ``[{'chain_id', 'declared' (templates declared in the input), 'aligned' ([len(query_to_template_map) per declared template]),
    'skip' (None | 'not_protein' | 'le_4_tokens' | 'not_in_crop')}]`` in the fork's chain order; ``resolved``: ``{chain_id: [tokens of that chain
    with any resolved atom in slot t, for t < max_templates]}`` read off the featurised ``template_atom_mask`` (:func:`_resolved_per_slot`);
    ``max_templates``: the fork's slot count; ``form``: ``'row_born'`` under the row-sharded pair stack, else None (FORMS). Returns ``{'chains': [{chain_id, declared, kept, real, dropped: [(slot, reason)], all_dummy}],
    'declared', 'kept', 'real', 'chains_templated', 'chains_all_dummy', 'verdict', 'lines': [...]}``."""
    out, lines = [], []
    for ch in chains:
        declared = int(ch.get("declared") or 0)
        if declared <= 0:
            continue
        cid, skip = str(ch["chain_id"]), ch.get("skip")
        kept = 0 if skip else min(declared, int(max_templates))
        aligned_of = list(ch.get("aligned") or [])
        counts = list(resolved.get(cid) or [])
        dropped, real = [], 0
        for slot in range(declared):
            if skip:
                dropped.append((slot, f"chain_skipped:{skip}"))
            elif slot >= max_templates:
                dropped.append((slot, "over_max_templates"))
            else:
                aligned = int(aligned_of[slot]) if slot < len(aligned_of) else 0
                n = int(counts[slot]) if slot < len(counts) else 0
                if n > 0:
                    real += 1                                             # a real slot (aligned residues the structure leaves unresolved keep the fork's own zero rows)
                else:
                    dropped.append((slot, f"unresolved_aligned_residues={aligned}/{aligned}"))
        out.append({"chain_id": cid, "declared": declared, "kept": kept, "real": real, "dropped": dropped, "all_dummy": real == 0})
        lines.append(f"{PREFIX} TEMPLATES chain={cid} declared={declared} kept={kept} real={real}/{max_templates} "
                     f"dropped={','.join(f'{s}:{r}' for s, r in dropped) or 'none'}")
    declared = sum(c["declared"] for c in out)
    z = [c for c in out if c["all_dummy"]]
    verdict = "untemplated" if declared == 0 else ("all_dummy" if z else "ok")
    summary = {"chains": out, "declared": declared, "kept": sum(c["kept"] for c in out), "real": sum(c["real"] for c in out),
               "chains_templated": len(out), "chains_all_dummy": len(z), "form": form, "verdict": verdict}
    lines.append(f"{PREFIX} TEMPLATES census declared={summary['declared']} kept={summary['kept']} real={summary['real']} "
                 f"chains_templated={summary['chains_templated']} chains_all_dummy={summary['chains_all_dummy']}" + (f" form={form}" if form else "") + f" verdict={verdict}")
    summary["lines"] = lines
    return summary


def _resolved_per_slot(atom_mask, col_chain_ids, chain_ids, max_templates):
    """``{chain_id: [tokens of the chain with any resolved atom in slot t]}`` from the featurised ``template_atom_mask`` ``[T, tokens(, atoms)]``."""
    import numpy as np
    mask = np.asarray(atom_mask)
    if mask.ndim > 2:
        mask = mask.reshape(mask.shape[0], mask.shape[1], -1).any(axis=-1)
    cols = np.asarray([str(c) if c is not None else "" for c in col_chain_ids])
    out = {}
    for cid in chain_ids:
        sel = cols == str(cid)
        out[str(cid)] = [int(mask[t][sel].sum()) if t < mask.shape[0] and sel.any() else 0 for t in range(int(max_templates))]
    return out


def emit(summary, stream=None):
    """Print the census lines."""
    stream = stream or sys.stdout
    for ln in summary["lines"]:
        stream.write(ln + "\n")
    stream.flush()


def install():
    """Wrap ``alphafold3.model.features.Templates.compute_features`` (once per process): the stock classmethod runs unchanged, then the census
    of its inputs and result is printed. Returns True when installed here, False when already installed."""
    if _STATE["installed"]:
        return False
    from alphafold3.constants import mmcif_names
    from alphafold3.model import features
    stock = features.Templates.__dict__["compute_features"]              # the classmethod object (kept: _STATE['stock'])
    stock_fn = stock.__func__

    def compute_features_censused(cls, all_tokens, standard_token_idxs, padding_shapes, templates_by_chain_id, max_templates, logging_name, *args, **kwargs):
        result = stock_fn(cls, all_tokens, standard_token_idxs, padding_shapes, templates_by_chain_id, max_templates, logging_name, *args, **kwargs)
        try:
            chains = _chains_of(all_tokens, templates_by_chain_id, logging_name, mmcif_names.PROTEIN_CHAIN)
            resolved = _resolved_per_slot(result.atom_mask, _col_chain_ids(all_tokens, standard_token_idxs, result.atom_mask.shape[1]),
                                          [c["chain_id"] for c in chains if c["declared"]], max_templates)
            summary = census(chains, resolved, max_templates, form=_STATE["form"])
        except Exception as e:  # noqa: BLE001 - the census failing is printed by name; the fork's result stands
            from opt_core.oom import is_oom
            if is_oom(e): raise                                           # OOM propagates: no fallback applied (opt_core.oom.is_oom)
            sys.stdout.write(f"{PREFIX} TEMPLATES census error={type(e).__name__}:{str(e).replace(' ', '_')[:200]} verdict=uncensused\n"); sys.stdout.flush()
            return result
        _STATE["censuses"].append({k: v for k, v in summary.items() if k != "lines"})
        emit(summary)
        return result

    features.Templates.compute_features = classmethod(compute_features_censused)
    _STATE.update(installed=True, stock=stock)
    return True


def _chains_of(all_tokens, templates_by_chain_id, logging_name, protein_chain_type):
    """The fork's own chain walk (features.py:744-769): chain order, type, token count, crop membership, and what the input declared."""
    import numpy as np
    from alphafold3.model.atom_layout import atom_layout
    substruct = atom_layout.make_structure(flat_layout=all_tokens, atom_coords=np.zeros(all_tokens.shape + (3,)), name=logging_name)
    nonempty = set(all_tokens.chain_id)
    chains = []
    for info in substruct.iter_chains():
        cid = info["chain_id"]
        n_tokens = len(all_tokens[all_tokens.chain_id == cid].atom_name)
        declared = list(templates_by_chain_id.get(cid) or [])
        chains.append({"chain_id": cid, "declared": len(declared),
                       "aligned": [len(dict(getattr(t, "query_to_template_map", {}) or {})) for t in declared],
                       "skip": _skip_reason(info["chain_type"], n_tokens, cid in nonempty, protein_chain_type)})
    return chains


def _col_chain_ids(all_tokens, standard_token_idxs, n_cols):
    """The chain id of each token column of the featurised template arrays (features.py:844: columns = standard_token_idxs; then padded)."""
    ids = [str(all_tokens.chain_id[i]) for i in standard_token_idxs]
    return ids + [""] * max(0, n_cols - len(ids))


def report():
    """``{'installed', 'censuses': [...]}`` — what this process featurised (the launchers' exit lines read it)."""
    return {"installed": _STATE["installed"], "censuses": list(_STATE["censuses"])}
