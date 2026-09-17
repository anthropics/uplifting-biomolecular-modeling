"""OF3-001 — templates parsed but not consumed at inference (openfold3 0.4.1).

Applied only on request: ``openfold3-opt pred|warm --upstream-fix OF3-001`` together with ``--use-templates true`` (the CLI refuses
the flag by name for a run whose templates are off: ``--upstream-fix OF3-001 requires --use-templates true``, exit 2, nothing launched);
The package's upstream_fix_registry (opt_core.upstream_fix) loads this file by path in the process that runs the model — the stock arm's process included — and prints
``[openfold3-opt] UPSTREAM-FIX OF3-001 applied …``. On a kit arm the template guard's census then reads ``TEMPLATES PARSED …: templates
parsed; consumed at inference under --upstream-fix OF3-001`` and the exit tally ``templ_policy=consume(OF3-001)``. Nothing imports this file
otherwise; no lever, hook or mode references it.

WHAT UPSTREAM 0.4.1 DOES. ``run_openfold predict --use-templates true`` preprocesses the template alignments a query declares
(``TemplatePreprocessor``: parses each chain's alignment file, fetches and caches the template structures, writes the chain's template
cache entry and its template ids onto the query) and then never uses them. ``InferenceDataset.create_template_features`` builds the
per-chain assembly data ``{"template_ids": …, "cache_entry_file_path": …}`` and calls
``process_template_structures_of3(..., template_cache_directory=None, ...)`` (openfold3/core/data/framework/single_datasets/
inference.py:246-266); ``sample_templates`` reads a cache entry only inside ``if k > 0 and template_cache_directory is not None:``
(openfold3/core/data/primitives/structure/template.py:210) and otherwise returns ``{}`` (:259-260). Every chain therefore gets zero
templates and the model runs on its dummy template slots — a templated query predicts exactly like the same query without templates.
The inference branch behind that gate (:224-226, ``template_cache_path = chain_data["cache_entry_file_path"]``) reads the chain's own
cache entry and never dereferences ``template_cache_directory``.

EVIDENCE. (1) The code path above is unconditional (``None`` ⇒ ``{}``). (2) The kit's template guard names it at run time on every
arm: ``[openfold3-opt] TEMPLATES chain=<id> ids=<n> sampled=0: templates parsed; not consumed at inference (upstream 0.4.1 behaviour)``
(opt/openfold3_opt/templ_guard.py ``wrap_sample``; CPU proof on the stock function compiled from source:
opt/openfold3_opt/tests/test_templ_guard.py ``test_a_templated_chain_is_not_consumed_exactly_as_stock_and_named`` and
opt/openfold3_opt/tests/test_upstream_fix.py). (3) On GPU a templated query gives bit-identical outputs under ``pred --mode off --det 1`` and
``pred --mode exact --det 1`` — both arms ignore the templates (README 'Known upstream issues').

OF3-004, BEHIND THIS ONE. Once the entries are consumed a second stock defect decides which hits reach the model: the predict-mode cache's
``idx_map`` keeps alignment gap columns as ``-1`` rows and the featurizer's residue mapper, written for gap-free rows, skips every hit whose
alignment has a gap (``map_token_pos_to_template_residues``: 'Skip if still misaligned') — only gap-free hits would be featurised. The wrapper
cuts each sampled entry's map to its residue-residue rows (``drop_gap_rows``; upstream_issues/OF3-004_gapped_template_hits_dropped.md) and
counts them on the chain's line (``gap_rows_dropped=<n>``).

WHAT THE FIX CHANGES. ``apply()`` wraps ``sample_templates`` (its definition in the structure module and the name the sample-processing
pipeline imported from it, pipelines/sample_processing/template.py:27-30, call site :113): for an INFERENCE chain that carries a
preprocessed cache entry and template ids and arrives with ``template_cache_directory=None``, the stock function is called with a
sentinel directory in place of the ``None`` — its own inference branch then loads the chain's cache entry and returns the top-k template
ids, which stock's own pipeline aligns, featurizes and embeds. Everything else is the stock call with the caller's arguments untouched:
untemplated chains (no ids or no cache entry), a caller's real template cache directory, training-style calls. No stock file is
edited; the wrap is installed at run time in the one process that runs the model. Per consumed chain the process prints
``[openfold3-opt] UPSTREAM-FIX OF3-001 chain=<id> template_ids=<n> sampled=<k> gap_rows_dropped=<g>``.

EXPECTED EFFECT ON OUTPUTS. Templated inputs only: the template embedder receives real template distograms / unit vectors instead of
dummy slots, so coordinates, pLDDT / PAE and the confidences JSON shift for chains whose templates align (typically higher pLDDT where a
template covers the chain). A query without templates, or a run with ``--use-templates false``, is bit-identical with and without the
fix (the wrap never fires: no chain carries a cache entry). Two arms that both apply the fix under the deterministic recipe stay as
comparable as they are without it (the same code runs on every arm). Under ``--mode big --n_gpu 2|4|8`` (the row-sharded line) the
launcher hands ``--upstream-fix OF3-001`` to every rank: each rank applies the fix in its own process and prints the per-chain census,
and the line's lazy template path carries each real slot's per-token precursors in place of the ``[T, N, N, F]`` pair features — every
rank computes its rows of them per row block (opt/openfold3_opt/tp_rowpair/data.py, template.py; on the host bitwise the dense features
sliced to rows, ``opt/openfold3_opt/tests/test_tp_rowpair_template_real_cpu.py``) and prints ``[rowpair data rank<r>] REAL templates: slots k/T real (…)``
per templated item. Real template hits on that line WITHOUT the fix are refused by name (``DataRefused``): the lazy path never runs a
real template as a dummy.
"""
from __future__ import annotations

import importlib
import sys

import numpy as np
from pathlib import Path

ID = "OF3-001"
SUMMARY = "sample_templates consumes the query's preprocessed template cache entries at inference (stock passes template_cache_directory=None and gets {})"
STRUCT_MODULE = "openfold3.core.data.primitives.structure.template"            # sample_templates: the definition
PIPELINE_MODULE = "openfold3.core.data.pipelines.sample_processing.template"    # process_template_structures_of3 calls sample_templates as a module global imported by name
SENTINEL_CACHE_DIR = Path("/nonexistent/openfold3-opt/OF3-001/template-cache-sentinel")   # stands in for the None; the inference branch never dereferences it (template.py:224-226)
SAMPLE_PARAMS = ("assembly_data", "template_cache_directory", "n_templates", "take_top_k", "chain_id", "template_structure_array_directory",
                 "template_file_format", "use_roda_monomer_format")            # stock sample_templates' parameters in order (structure/template.py:126-135)
MARK = "_of3_upstream_fix"                                                     # the wrapper's marker attribute (value: ID) — apply() is idempotent
CONSUMES_TEMPLATES = True                                                      # on a kit arm the template guard's census then states it (templ_policy=consume(OF3-001), TEMPLATES PARSED …: consumed)
REQUIRES_USE_TEMPLATES = "requires --use-templates true (only then does upstream preprocess the declared alignments into the cache entries the fix consumes; " \
                         "with templates off a declared alignment path would reach the sampler raw)"
GUARD_MARK = "_of3opt_guard"                                                   # the kit template guard's marker: a function carrying it is never re-wrapped by templ_guard.install()
TAG = "[openfold3-opt]"
GAP = -1                                                                       # upstream's gap marker in a PREDICT-mode cache entry's idx_map (io/sequence/template.py calculate_ids_hit: "-1 for
                                                                               # template positions aligned to gaps in the query" and the converse) — OF3-004, drop_gap_rows below


def _log(msg: str) -> None:
    print(f"{TAG} {msg}", file=sys.stderr, flush=True)


def refuse(run: dict):
    """The run-level precondition, checked by the CLI before anything launches (upstream_fix.resolve with the run's words): the fix consumes
    PREPROCESSED template cache entries, and upstream preprocesses templates only under ``--use-templates true`` — under ``false`` the query's
    declared alignment paths stay raw on the chains and must never be handed to the sampler. Returns the refusal words, or None."""
    if str(run.get("use_templates", "")).strip().lower() != "true":
        return REQUIRES_USE_TEMPLATES
    return None


def templated_chain(chain_data) -> bool:
    """An inference chain that arrives with a preprocessed template cache entry and template ids (the preprocessor's output for a declared chain)."""
    return bool(chain_data) and chain_data.get("cache_entry_file_path") is not None and bool(chain_data.get("template_ids"))


def wrap(orig, log=None):
    """``sample_templates`` with the fix: a templated inference chain given ``template_cache_directory=None`` runs stock's inference branch
    (the sentinel stands in for the None); every other call is ``orig`` with the caller's arguments untouched. Idempotent on a wrapped function."""
    if getattr(orig, MARK, None) == ID:
        return orig
    log = log or _log

    def sample_templates(*args, **kwargs):
        bound = dict(zip(SAMPLE_PARAMS, args))
        bound.update(kwargs)
        cid = bound.get("chain_id")
        chain_data = (bound.get("assembly_data") or {}).get(cid, {}) if cid is not None else {}
        if not (templated_chain(chain_data) and bound.get("template_cache_directory") is None):
            return orig(*args, **kwargs)
        bound["template_cache_directory"] = SENTINEL_CACHE_DIR
        out = orig(**bound)
        gap_rows = drop_gap_rows(out)                                           # OF3-004: the featurizer's residue mapper takes gap-free rows only
        log(f"UPSTREAM-FIX {ID} chain={cid} template_ids={len(chain_data['template_ids'])} sampled={len(out or {})} gap_rows_dropped={gap_rows}")
        return out

    sample_templates.__wrapped__ = orig
    sample_templates.__name__ = getattr(orig, "__name__", "sample_templates")
    sample_templates.__doc__ = getattr(orig, "__doc__", None)
    setattr(sample_templates, MARK, ID)
    setattr(sample_templates, GUARD_MARK, True)
    return sample_templates


def drop_gap_rows(sampled) -> int:
    """OF3-004 (reachable only through this fix: stock never hands a sampled entry to the featurizer). A PREDICT-mode template cache entry's
    ``idx_map`` (query residue index -> template residue index, one row per alignment column) keeps the alignment's gap columns as ``-1`` rows,
    while the featurizer's residue mapper (structure/template.py ``map_token_pos_to_template_residues``) is written for the training cache's
    gap-free rows (sequence/template.py ``create_residue_idx_map``: "columns where the template sequence is not a gap"): with one gap row the
    query-token count and the template-residue count differ, its "Skip if still misaligned" branch returns no slice, and the template's slot
    stays empty — every hit whose alignment to the query has a gap is dropped silently; only gap-free hits are featurised. Here each sampled
    entry's map is cut to the rows where both indices are residues (the documented invariant; the aligned residue pairs are untouched, so a
    gap-free hit is byte-unchanged); an entry left with no aligned pair is removed from the sample. Returns the number of rows dropped."""
    dropped = 0
    for tid in list(sampled or {}):
        entry = sampled[tid]
        idx_map = getattr(entry, "idx_map", None)
        if idx_map is None:
            continue
        rows = np.asarray(idx_map)
        if rows.ndim != 2 or rows.shape[-1] != 2 or rows.shape[0] == 0:
            continue
        keep = (rows[:, 0] != GAP) & (rows[:, 1] != GAP)
        n_gap = int((~keep).sum())
        if n_gap == 0:
            continue
        dropped += n_gap
        if keep.any():
            entry.idx_map = rows[keep]
        else:
            del sampled[tid]                                                    # no aligned residue pair: nothing of this hit can be featurised
    return dropped


def apply(log=None) -> dict:
    """Install the fix in this process: import the two stock modules and bind the wrapped ``sample_templates`` on both (the pipeline module
    holds the name it imported). Returns the record ``{"id", "summary", "targets", "sentinel"}``; raises when upstream's shape changed
    (a module or the function is missing) — the caller refuses the run by name, never runs without the requested fix."""
    struct = importlib.import_module(STRUCT_MODULE)
    pipeline = importlib.import_module(PIPELINE_MODULE)
    for mod in (struct, pipeline):
        if not callable(getattr(mod, "sample_templates", None)):
            raise AttributeError(f"{ID}: {mod.__name__}.sample_templates is missing (openfold3 changed shape; the fix targets 0.4.1)")
    fixed = wrap(struct.sample_templates, log)
    struct.sample_templates = fixed
    pipeline.sample_templates = fixed                                       # the call site's global: one function on both names
    return {"id": ID, "summary": SUMMARY, "targets": [f"{STRUCT_MODULE}.sample_templates", f"{PIPELINE_MODULE}.sample_templates"],
            "sentinel": str(SENTINEL_CACHE_DIR)}
