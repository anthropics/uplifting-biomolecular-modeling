"""`big` — the memory mode of this kit and its resource axis ``--n_gpu P`` (modes.BIG, modes.ENV_N_GPU).

What `big` is here (modes.TABLE): the `fast` lever set (modes.TABLE; fast-class numerics, tier 2 — its one-device pair levers, modes.ONE_DEVICE_PAIR_LEVERS, step aside at P > 1) plus the row-sharded
pair representation ``ROWPAIR`` (strategy F7.tensor_parallel) when P > 1: AlphaFold-Multimer runs END TO END with the pair representation
``[N, N, C]`` split by rows over P local GPUs — each GPU holds rows ``[N/P, N, C]`` from the statement that creates the pair (the prologue: relative
positions, the recycled pair, templates) through the extra-MSA / Evoformer stacks, the recycle carry, the structure module's pair reads and the
distogram / aligned-error heads (``opt_core.mem.rowpair_jax.alphafold``: the AlphaFold-2 recipe — the stock classes rebound to bodies running the
stock arithmetic on this device's row block, site groups ``trunk`` / ``model`` / ``heads``; ``opt_core.mem.rowpair_jax``: the mesh, the collectives,
the Haiku facts, the evidence). Replicated by the recipe's stated floor: the MSA representation ``[S, N, c_m]`` (linear in N), the single
representation and the parameters. The outputs that are N² by contract (distogram logits, aligned-error logits / PAE, the recycled pair) leave the
program row-sharded and are assembled on the HOST by the fetch (``jax.device_get`` in DEVICE_RESIDENT = the writer boundary; the LEVER line's
``outputs_rows=``). Templates take the same route — each GPU forms only its own rows of the per-template pair features from the per-residue template
fields (``opt_core.mem.rowpair_jax.alphafold_template``) — and each query's census is printed once as it starts: ``[colabfold-opt] TEMPLATES
job=<jobname> real=<r>/<T> form=row_born`` (:data:`TEMPLATES_FMT`, :func:`template_census`, :func:`templates_line`). P = 1 installs nothing: the process is byte for byte a `fast` process whose ACTIVE / LEVER lines carry ``n_gpu=1 sharding=none``.

The axis is explicit and refused by name, never sized to the box (``opt_core.mem.ngpu`` is the one producer of the sentences): ``--n_gpu`` absent
or ``1`` = one GPU; ``P`` not in ``modes.N_GPU_SUPPORTED`` → refused; ``P > 1`` under `exact` / `fast` / `off` → ``refused: n_gpu>1 requires --mode
big (sharded reductions are not bitwise)``; fewer than P visible GPUs → ``refused: n_gpu=P visible=K`` (the parent counts with nvidia-smi before
the launch, the model process again with ``jax.devices()`` when it builds the mesh). The recipe needs ``N % P == 0`` and refuses otherwise by
name; the kit pads: on the monomer route colabfold's own ``pad_len`` (``predict_structure``; colabfold pads the features with mask-0 positions
whenever ``pad_len`` exceeds the sequence length, batch.py:436-444, ``pad_input`` :330-362) is raised to the next multiple of P; on the multimer
route — which colabfold never pads (batch.py:431-434) — the feature dict is padded to the next multiple of P at ``RunModel.predict`` with
colabfold's own ``make_fixed_size`` over this kit's multimer residue-axis schema (``MULTIMER_FEATURES``) and every per-residue / pair output is cut
back to N before colabfold's writers and rankers see it (``MULTIMER_OUTPUTS``); every pad is a recorded event (``pad_events``, route named).

Evidence: the ACTIVE line's ``n_gpu=P sharding=rowpair|none`` (``opt_core.mem.rowpair_jax.evidence.active_fields``); at exit one LEVER line
``name=ROWPAIR`` rendered by ``evidence.line`` (state on with the mesh facts, the XLA memory settings, the per-device peak bytes and the
recipe's record at P > 1; ``state=off reason=n_gpu=1`` at P = 1); the manifest's ``n_gpu`` / ``sharding`` / ``lever_states_exit.ROWPAIR``.
"""
from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

from opt_core.mem import MemLeverRefused
from opt_core.mem import ngpu as _ngpu
from opt_core.mem.rowpair_jax import evidence as _evidence

from . import modes as _modes

NAME = _modes.TP_LEVER                                   # "ROWPAIR": the lever's name on the LEVER line and in the registry
STRATEGY = "F7.tensor_parallel"
MODULES = "alphafold.model.modules"                      # the stock module objects the recipe rebinds (opt_core.mem.rowpair_jax.alphafold REQUIRED / PIN)
MULTIMER = "alphafold.model.modules_multimer"
BATCH = "colabfold.batch"                                # predict_structure(pad_len=...) — N raised to a multiple of P (monomer route)
MODEL = "alphafold.model.model"                          # RunModel.predict — the multimer route's pad / un-pad boundary
RECIPE_WORDS = {"heads": "sharded", "conf": "sharded", "sites": ("trunk", "model", "heads")}   # what big asks of the recipe at P > 1 (alphafold.install); the LEVER line prints
                                                                                               # the recipe's EFFECTIVE words (describe()), never this request
# The multimer feature dict colabfold hands to ``RunModel.predict`` (``process_multimer_features``, colabfold/batch.py:804-859 →
# ``feature_processing.process_final`` keeps ``REQUIRED_FEATURES``, alphafold/data/feature_processing.py:24-33, :163-198; ``pipeline_multimer.pad_msa``):
# each key's axes as the placeholders ``make_fixed_size`` pads by (colabfold/alphafold/msa.py:12-43; "N" = residues — padded — "S" = MSA rows, "T" =
# templates, None = fixed). Padded positions are mask 0 (``seq_mask`` / ``msa_mask`` / template atom masks are zero-padded like every other key), the
# form colabfold's monomer route pads with (``pad_input``, batch.py:330-362, over ``config.data.eval.feat``). A key outside this table is refused by
# name (ROWPAIR never guesses a residue axis).
MULTIMER_FEATURES = {
    "aatype": ("N",), "residue_index": ("N",), "seq_mask": ("N",), "asym_id": ("N",), "sym_id": ("N",), "entity_id": ("N",), "entity_mask": ("N",),
    "deletion_mean": ("N",), "all_atom_mask": ("N", None), "all_atom_positions": ("N", None, None),
    "msa": ("S", "N"), "deletion_matrix": ("S", "N"), "msa_mask": ("S", "N"), "bert_mask": ("S", "N"), "cluster_bias_mask": ("S",),
    "template_aatype": ("T", "N"), "template_all_atom_mask": ("T", "N", None), "template_all_atom_positions": ("T", "N", None, None),
    "seq_length": (), "num_alignments": (), "num_templates": (), "assembly_num_chains": (), "resolution": (), "mem_peak": (), "queue_size": (),
}
# The result tree ``RunModel.predict`` returns on the multimer route (``modules_multimer.AlphaFold.__call__``, alphafold/model/modules_multimer.py:404-478:
# the heads of ``AlphaFoldIteration`` :287-401, ``get_prev`` :430-436, the in-graph confidence metrics of alphafold/common/confidence.py:172-212 —
# whose ``predicted_aligned_error`` [N, N] replaces the head's dict of the same name — the structure module's inference outputs,
# alphafold/model/folding_multimer.py:599-624, and ``representations`` as ``predict`` rebuilds it, alphafold/model/model.py:194-197): path → the
# residue axes cut back from the padded length to N. Leaves not listed: 0-d (ptm, iptm, ranking_confidence, mean_plddt, tol,
# max_predicted_aligned_error) pass; any other array is refused by name.
MULTIMER_OUTPUTS = {
    ("distogram", "logits"): (0, 1), ("distogram", "bin_edges"): (),
    ("experimentally_resolved", "logits"): (0,), ("masked_msa", "logits"): (1,), ("predicted_lddt", "logits"): (0,),
    ("predicted_aligned_error",): (0, 1), ("predicted_aligned_error", "logits"): (0, 1), ("predicted_aligned_error", "breaks"): (), ("predicted_aligned_error", "asym_id"): (0,),
    ("pae_matrix_with_logits", "logits"): (0, 1), ("pae_matrix_with_logits", "breaks"): (), ("pae_matrix_with_logits", "asym_id"): (0,),
    ("aligned_confidence_probs",): (0, 1), ("plddt",): (0,),
    ("structure_module", "final_atom_positions"): (0,), ("structure_module", "final_atom_mask"): (0,),
    ("prev", "prev_pos"): (0,), ("prev", "prev_msa_first_row"): (0,), ("prev", "prev_pair"): (0, 1),
    ("representations", "pair"): (0, 1), ("representations", "single"): (0,), ("representations", "msa"): (1,), ("representations", "msa_first_row"): (0,),
    ("representations", "structure_module"): (0,),
}

_STATE: Dict[str, Any] = {"enabled": False, "n_gpu": 1, "sharding": "none", "rmesh": None, "patches": None, "pad_multiple": 1,
                          "pad_events": [], "sites": "none", "reason": None}


def gate(mode: str, n_gpu: int, visible: Optional[int]) -> Optional[str]:
    """None when ``--n_gpu n_gpu`` may run under ``mode`` on a host showing ``visible`` GPUs (None = not counted here); else the refusal sentence
    (``opt_core.mem.ngpu``'s words for the mode and the visible count; this kit's for a P outside modes.N_GPU_SUPPORTED)."""
    try:
        p = _ngpu.check_n_gpu(n_gpu)
        _ngpu.refuse_unless_big(p, mode)
        if p not in _modes.N_GPU_SUPPORTED:
            return f"refused: n_gpu={p} not in {{{','.join(str(x) for x in _modes.N_GPU_SUPPORTED)}}} (the P set this kit ships, modes.N_GPU_SUPPORTED)"
        if p > 1 and visible is not None:
            _ngpu.refuse_unless_visible(p, int(visible))
    except MemLeverRefused as e:
        return e.reason
    except ValueError as e:
        return f"refused: {e}"
    return None


def active_fields(n_gpu: int) -> List:
    """``[("n_gpu", P), ("sharding", "rowpair"|"none")]`` — the ACTIVE line's tokens (evidence.active_fields, never spelled here)."""
    return list(_evidence.active_fields(int(n_gpu)))


def apply(n_gpu: int, modules: Any = None, multimer: Any = None, batch: Any = None, model: Any = None) -> Dict[str, Any]:
    """Install the axis in this process. P = 1: record the tokens, install nothing. P > 1: build the mesh over the first P devices
    (``mesh.build``: refuses by name when fewer are visible or when jax is not on the GPU platform), rebind the AlphaFold-2 pair classes
    (``opt_core.mem.rowpair_jax.alphafold.install`` on ``modules`` / ``multimer``, default the imported stock modules; the recipe refuses by
    name a stock tree whose bytes it does not pin; site groups trunk + model + heads, ``RECIPE_WORDS``), hand the mesh to DEVICE_RESIDENT (its
    jitted apply is bound to the mesh with the recipe's leaf shardings) and install the pad rule on both routes (``pad_len`` raised to a
    multiple of P at ``colabfold.batch.predict_structure``; the multimer feature dict padded / the outputs cut back at ``RunModel.predict`` of
    ``model``, default ``alphafold.model.model``). Returns the state dict; a refusal propagates (MemLeverRefused) — the activation turns it
    into NOT ACTIVE, never a silent P = 1 run."""
    import importlib
    p = _ngpu.check_n_gpu(n_gpu)
    fields = dict(active_fields(p))
    _STATE.update(n_gpu=p, sharding=fields.get("sharding", "none"))
    if p == 1:                                              # the activation drops ROWPAIR from the process's lever set at P = 1: fast's bytes, nothing installed
        _STATE.update(enabled=False, reason="n_gpu=1")
        return state_public()
    from opt_core.mem.rowpair_jax import mesh as _mesh  # noqa: PLC0415 — jax is imported by the mesh, in the model process only
    from . import device_resident as _dr  # noqa: PLC0415
    from opt_core.mem.rowpair_jax import alphafold as _recipe  # noqa: PLC0415 — the AlphaFold-2 recipe (one producer for the AF2 trees)
    rmesh = _mesh.build(p, lever=NAME, platform="gpu")                # platform pinned: a jax that fell back to CPU refuses by name instead of sharding over CPU devices
    M = modules if modules is not None else importlib.import_module(MODULES)
    MM = multimer if multimer is not None else importlib.import_module(MULTIMER)
    if not hasattr(_recipe, "jit_sharded"):                          # a recipe without the model / heads site groups: refused by name, never a trunk-only (partial) install
        raise MemLeverRefused(NAME, "refused: opt_core.mem.rowpair_jax.alphafold has no end-to-end site groups (model / heads) — big --n_gpu > 1 needs opt_core >= 0.4.3")
    patches = _recipe.install(rmesh, M, MM, lever=NAME, **RECIPE_WORDS)     # trunk + model + heads on rows; the trimul schedule is the recipe's default (never chosen here)
    try:
        _dr.set_mesh(rmesh)
        B = batch if batch is not None else importlib.import_module(BATCH)
        patches.replace(B, "predict_structure", _pad_to_multiple(getattr(B, "predict_structure"), p))   # monomer route
        RM = model if model is not None else getattr(importlib.import_module(MODEL), "RunModel")      # the class colabfold instantiates (DEVICE_RESIDENT's subclass when it is on)
        patches.replace(RM, "predict", _pad_predict(getattr(RM, "predict"), p))                        # multimer route: pad the features in, cut the outputs back
    except Exception:
        _recipe.uninstall()
        _dr.set_mesh(None)
        raise
    _STATE.update(enabled=True, rmesh=rmesh, patches=patches, pad_multiple=p, sites=_recipe.describe()["sites"], reason=None)
    return state_public()


def state_public() -> Dict[str, Any]:
    """A copy of the state (what activate records)."""
    return dict(_STATE)


def _pad_to_multiple(predict_structure, p: int):
    """colabfold's ``predict_structure`` with ``pad_len`` raised to the next multiple of P at or above the input's length (batch.py:381-388,
    :436-444: on the MONOMER route — ``"multimer" not in model_type`` — colabfold pads the features to ``pad_len`` whenever it exceeds the
    sequence length; the multimer route ignores ``pad_len``, batch.py:431-435, and is padded at ``RunModel.predict``, :func:`_pad_predict`).
    Every adjustment that reaches a monomer prediction is recorded; a raise the multimer route ignores is not an event. The same boundary —
    one call per query — prints the query's template census line (:func:`templates_line`)."""
    from opt_core.mem.rowpair_jax import shard as _shard  # noqa: PLC0415
    from .report import emit  # noqa: PLC0415

    def padded(*args, **kwargs):
        feat = kwargs.get("feature_dict", args[2] if len(args) > 2 else None)
        if isinstance(feat, dict):                                            # once per query: its template census (TEMPLATES_FMT)
            emit(templates_line(kwargs.get("prefix", args[0] if args else "?"), feat))
        lengths = kwargs.get("sequences_lengths")
        seq_len = int(sum(lengths)) if lengths else 0
        asked = kwargs.get("pad_len")
        target = _shard.next_multiple(max(int(asked or 0), seq_len), p)
        if (asked is None or int(asked) != target) and "multimer" not in str(kwargs.get("model_type", "")):
            _STATE["pad_events"].append({"route": "monomer", "asked": asked, "seq_len": seq_len, "pad_len": target})
        kwargs["pad_len"] = target
        return predict_structure(*args, **kwargs)

    padded.__wrapped__ = predict_structure   # type: ignore[attr-defined]
    return padded


def _pad_predict(predict, p: int):
    """``RunModel.predict(feat, random_seed=, return_representations=, callback=)`` (alphafold/model/model.py:123-208) on the MULTIMER route with
    the feature dict padded to the next multiple of P (:func:`pad_multimer_features`) and every result — the one returned and the one each
    recycle's ``callback`` receives — cut back to N (:func:`unpad_multimer_outputs`); the recycle loop inside runs at the padded length throughout
    (its ``prev`` is sized from the padded ``aatype``, :153-155). The monomer route (``multimer_mode`` false) passes through: colabfold padded those
    features itself to the ``pad_len`` :func:`_pad_to_multiple` raised. A feature or output outside the schemas exits NOT ACTIVE by name."""
    from opt_core.mem.rowpair_jax import shard as _shard  # noqa: PLC0415
    from .report import EXIT_NOT_ACTIVE, NOT_ACTIVE_FMT, PREFIX, emit  # noqa: PLC0415

    def padded_predict(self, feat, *args, **kwargs):
        if not getattr(self, "multimer_mode", False):
            return predict(self, feat, *args, **kwargs)
        n = int(feat["aatype"].shape[0])
        n_pad = _shard.next_multiple(n, p)
        if n_pad == n:
            return predict(self, feat, *args, **kwargs)
        try:
            feat_p = pad_multimer_features(feat, n_pad)
        except MemLeverRefused as e:
            emit(NOT_ACTIVE_FMT.format(prefix=PREFIX, reason=e.reason)); sys.exit(EXIT_NOT_ACTIVE)
        _STATE["pad_events"].append({"route": "multimer", "asked": None, "seq_len": n, "pad_len": n_pad})

        def cut(result):
            try:
                return unpad_multimer_outputs(result, n, n_pad)
            except MemLeverRefused as e:
                emit(NOT_ACTIVE_FMT.format(prefix=PREFIX, reason=e.reason)); sys.exit(EXIT_NOT_ACTIVE)

        cb = kwargs.get("callback")
        if cb is not None:
            kwargs["callback"] = lambda result, recycles: cb(cut(result), recycles)
        result, recycles = predict(self, feat_p, *args, **kwargs)
        return cut(result), recycles

    padded_predict.__wrapped__ = predict     # type: ignore[attr-defined]
    return padded_predict


def pad_multimer_features(feat: Dict[str, Any], n_pad: int) -> Dict[str, Any]:
    """A COPY of the multimer feature dict with every residue axis of ``MULTIMER_FEATURES`` zero-padded to ``n_pad`` by colabfold's
    ``make_fixed_size`` (colabfold/alphafold/msa.py:12-43; MSA rows and template count kept). A key without a schema row, or whose rank
    disagrees with its row, is refused by name."""
    from alphafold.model.tf import shape_placeholders as _ph  # noqa: PLC0415 — the placeholders make_fixed_size pads by
    from colabfold.alphafold.msa import make_fixed_size  # noqa: PLC0415
    unknown = sorted(k for k in feat if k not in MULTIMER_FEATURES and k != "extra_cluster_assignment")
    if unknown:
        raise MemLeverRefused(NAME, f"refused: multimer feature(s) {', '.join(unknown)} have no residue-axis schema (colabfold_opt.big.MULTIMER_FEATURES) — "
                                    f"the pad to a multiple of P cannot be applied without guessing")
    bad = sorted(f"{k}{tuple(getattr(v, 'shape', ()))}" for k, v in feat.items() if k in MULTIMER_FEATURES and len(getattr(v, "shape", ())) != len(MULTIMER_FEATURES[k]))
    if bad:
        raise MemLeverRefused(NAME, f"refused: multimer feature(s) {', '.join(bad)} disagree in rank with colabfold_opt.big.MULTIMER_FEATURES")
    ph = {"N": _ph.NUM_RES, "S": _ph.NUM_MSA_SEQ, "T": _ph.NUM_TEMPLATES, None: None}
    schema = {k: [ph[a] for a in MULTIMER_FEATURES[k]] for k in feat if k in MULTIMER_FEATURES}
    schema["extra_cluster_assignment"] = []
    rows = int(feat["msa"].shape[0]) if "msa" in feat else 0
    templates = int(feat["template_aatype"].shape[0]) if "template_aatype" in feat else 0
    return make_fixed_size(dict(feat), schema, msa_cluster_size=rows, extra_msa_size=0, num_res=int(n_pad), num_templates=templates)


def unpad_multimer_outputs(result: Any, n: int, n_pad: int, _path: tuple = ()) -> Any:
    """``result`` (the multimer route's output tree, dicts of arrays) with every residue axis of ``MULTIMER_OUTPUTS`` cut from ``n_pad`` back to
    ``n``, IN PLACE (dict entries reassigned; idempotent: an axis already at ``n`` is left). 0-d leaves pass; an array leaf without a schema
    row, or whose listed axis is neither ``n`` nor ``n_pad`` long, is refused by name."""
    if isinstance(result, dict):
        for k in list(result.keys()):
            result[k] = unpad_multimer_outputs(result[k], n, n_pad, _path + (k,))
        return result
    shape = getattr(result, "shape", None)
    if shape is None or len(shape) == 0:
        return result
    axes = MULTIMER_OUTPUTS.get(_path)
    if axes is None:
        raise MemLeverRefused(NAME, f"refused: multimer output {'/'.join(_path)} shape {tuple(shape)} has no residue-axis schema (colabfold_opt.big.MULTIMER_OUTPUTS) — "
                                    f"it cannot be cut back to N without guessing")
    index = [slice(None)] * len(shape)
    for ax in axes:
        if ax >= len(shape) or int(shape[ax]) not in (int(n), int(n_pad)):
            raise MemLeverRefused(NAME, f"refused: multimer output {'/'.join(_path)} shape {tuple(shape)}: axis {ax} is neither N={n} nor the padded {n_pad}")
        if int(shape[ax]) == int(n_pad):
            index[ax] = slice(0, int(n))
    return result[tuple(index)] if any(s != slice(None) for s in index) else result


def pad_routes() -> str:
    """``monomer:<a>,multimer:<b>`` — the pad events by route (a blank-free LEVER-line token; both routes always named)."""
    ev = _STATE["pad_events"]
    return ",".join(f"{r}:{sum(1 for e in ev if e.get('route') == r)}" for r in ("monomer", "multimer"))


TEMPLATE_MASK_KEYS = ("template_all_atom_mask", "template_all_atom_masks")     # the template atom mask as the multimer / monomer feature dicts name it
TEMPLATES_FMT = "{prefix} TEMPLATES job={job} real={real}/{slots} form={form}"     # once per query at P > 1 (templates_line): real template slots / slots, how their pair features are formed


def template_census(feat: Dict[str, Any]) -> tuple:
    """``(real, slots)`` of one query's feature dict: the template slots it carries and how many hold a real hit — a slot with any atom present in
    its atom mask (colabfold's mock slot, colabfold/batch.py mk_mock_template, is all-zero; padded residues are mask 0). Read on the host from the
    dict colabfold hands to ``predict_structure`` (both routes' key spellings; a leading ensemble axis, rank 4, is looked through). Nothing on the
    device path reads this. ``(0, 0)`` when the dict carries no template atom mask."""
    import numpy as np  # noqa: PLC0415
    mask = next((feat[k] for k in TEMPLATE_MASK_KEYS if k in feat), None)
    if mask is None or len(getattr(mask, "shape", ())) < 3:
        return (0, 0)
    m = np.asarray(mask)
    m = m.reshape((-1,) + m.shape[-3:])[0]                                   # [T, N, atoms]: the multimer dict as is, the monomer dict's first ensemble copy
    return (int(np.count_nonzero(m.reshape(m.shape[0], -1).any(axis=1))), int(m.shape[0]))


def templates_line(job: Any, feat: Dict[str, Any]) -> str:
    """The query's template census line, :data:`TEMPLATES_FMT`: ``real=<r>/<T>`` from :func:`template_census`; ``form`` = ``row_born`` when the recipe's
    template term is on rows (``describe()["template"] == "rows"``: each GPU forms its own rows of the per-template pair features from the per-residue
    template fields), else the recipe's own word for what it installed — named, never silent."""
    from opt_core.mem.rowpair_jax import alphafold as _recipe  # noqa: PLC0415
    from .report import PREFIX, slug  # noqa: PLC0415
    real, slots = template_census(feat)
    state = _recipe.describe().get("template") if _STATE["patches"] is not None else None
    return TEMPLATES_FMT.format(prefix=PREFIX, job=slug(str(job)), real=real, slots=slots, form="row_born" if state == "rows" else str(state or "none"))


def exit_evidence() -> Dict[str, Any]:
    """The LEVER line's evidence fields at exit (also the manifest's ``lever_states_exit.ROWPAIR``): the recipe's record (sites, trimul schedule,
    regions built, library shas, hazards), the per-shard attention kernel (the carried Pallas kernel when AF_PALLAS_ATTN is applied in this
    process, else XLA's jnp path — named, never silent) and the pad events."""
    kit = sys.modules.get(_modes.KIT_MODULE)
    kst = getattr(kit, "_STATE", None) if kit is not None else None
    kernel_on = isinstance(kst, dict) and bool(kst.get("enabled"))
    ev: Dict[str, Any] = {"pad_multiple": _STATE["pad_multiple"], "pad_events": len(_STATE["pad_events"]), "pad_routes": pad_routes()}
    if _STATE["patches"] is not None:                        # the recipe's record in the family's vocabulary (sites, schedule, the effective group words, outputs_rows, library, hazards)
        from opt_core.mem.rowpair_jax import alphafold as _recipe  # noqa: PLC0415
        ev.update(_recipe.describe())
    ev.update(kernel=_modes.KIT_PACKAGE if kernel_on else "jnp", kernel_reason="lever_applied" if kernel_on else "kernel_lever_not_applied")   # the per-shard attention kernel is this kit's fact: the carried Pallas kernel when AF_PALLAS_ATTN is applied in this process (its rebound Attention class is what the row bodies call), else XLA's jnp path
    return ev


def exit_line() -> str:
    """The ONE activation-evidence line of the axis, rendered by the family (evidence.line): ``state=on`` with the mesh facts, the XLA memory
    settings, the per-device allocator peaks and the recipe's record at P > 1; ``state=off reason=n_gpu=1`` with device 0's peak at P = 1;
    ``state=skipped`` with the activation's reason when the mode was refused or held."""
    from .report import TAG  # noqa: PLC0415
    rmesh = _STATE["rmesh"]
    if _STATE["enabled"] and rmesh is not None:
        return _evidence.line(TAG, "on", int(_STATE["n_gpu"]), lever=NAME, strategy=STRATEGY, rmesh=rmesh, peaks=_peaks(rmesh), **exit_evidence())
    if int(_STATE["n_gpu"]) == 1 and _STATE["reason"] in (None, "n_gpu=1"):
        return _evidence.line(TAG, "off", 1, lever=NAME, strategy=STRATEGY, reason="n_gpu=1", peaks=_peaks(None),
                              pad_multiple=1, pad_events=len(_STATE["pad_events"]), pad_routes=pad_routes())
    return _evidence.line(TAG, "skipped", int(_STATE["n_gpu"]), lever=NAME, strategy=STRATEGY, reason=_STATE["reason"] or "activation_refused")


def _peaks(rmesh):
    """evidence.device_peaks over the mesh devices (P > 1) or the process's first local device (P = 1); None when jax has no stats to give."""
    try:
        if rmesh is not None:
            return _evidence.device_peaks(rmesh)
        import jax  # noqa: PLC0415 — the model process only (exit of a big run)
        return _evidence.device_peaks(list(jax.local_devices())[:1])
    except Exception:  # noqa: BLE001 — a backend without memory stats: the line names the rest
        return None


def state() -> Optional[Dict[str, Any]]:
    """The registry probe's view (``module_state``): None until ``apply`` ran in this process. ``calls`` = the number of row-sharded regions the
    recipe built in this process (its ``describe()['regions_built']``; ≥ 1 is the proof a sharded pair stack was traced and ran) — the
    activation's exit check reads it like every other lever's counter."""
    if _STATE["reason"] is None and not _STATE["enabled"]:
        return None
    st = state_public()
    st["calls"] = None
    if _STATE["patches"] is not None:
        try:
            from opt_core.mem.rowpair_jax import alphafold as _recipe  # noqa: PLC0415
            built = _recipe.describe().get("regions_built")
            st["calls"] = len(built) if isinstance(built, (list, tuple, set, dict)) else (int(built) if built is not None else None)
        except Exception:  # noqa: BLE001
            st["calls"] = None
    return st


def reset_for_tests() -> None:
    p = _STATE.get("patches")
    if p is not None:
        try:
            from opt_core.mem.rowpair_jax import alphafold as _recipe  # noqa: PLC0415
            _recipe.uninstall()
        except Exception:  # noqa: BLE001
            pass
    dr = sys.modules.get(_modes.HOST_LEVER_MODULE)
    if dr is not None and hasattr(dr, "set_mesh"):
        dr.set_mesh(None)
    _STATE.update(enabled=False, n_gpu=1, sharding="none", rmesh=None, patches=None, pad_multiple=1, pad_events=[], sites="none", reason=None)
