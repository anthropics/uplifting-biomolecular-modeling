"""Template distinct-slot evaluation (lever ``template_dedupe``): the template embedder evaluates each DISTINCT template slot once.

AF3 featurisation pads the template features to a fixed number of slots (``template_aatype`` [T, N], ``template_atom_positions``
[T, N, 24, 3], ``template_atom_mask`` [T, N, 24]; T = 4); an input without templates carries T byte-identical dummy slots, and the stock
embedder (``xfold/nn/template.py`` ``TemplateEmbedding.forward``) runs its 2-block pair stack once per slot and sums the per-slot
embeddings in slot order. Here slots whose feature tensors are equal (``torch.equal`` on every field of ``xfold.features.Templates``) are
evaluated ONCE and the one result is added once per slot, in the stock slot order — the same addends in the same order: the output is
bitwise the stock forward's by construction (the embedder is deterministic: the tree's proof, ``off`` bitwise run to run). Slots that
differ (real templates) are each evaluated: a templated input is a counted non-event (``slots == evaluated``), never a skip.

The distinct-slot search (``distinct_slots``: ``torch.equal`` per field and slot pair — each a host synchronisation) is a fact of the
ITEM's template features, not of the call: it runs once per item and its map is reused by every later call on the same feature tensors
(the trunk's recycles; ``_MEMO``, keyed on the tensors' identity, cleared per item by ``begin_item()`` / ``take()``); a call on other
tensors searches again. The embedder's two constant index tables (``TABLES``: xfold.protein_data_processing's rigid-group atom index,
read by template.py's ``construct_input``, and pseudo-beta index, read by scoring.py's ``pseudo_beta_fn``) are CPU tensors the stock
statements move ``.to(device=…)`` on every slot evaluation (a host→device copy each); ``install`` makes them resident on the model's
device once (``place_tables``: the same values and dtype, so those statements' ``.to`` returns the tensor as it is) — bitwise either way.
The census (``COUNTS``: calls, slots, evaluated, scans; ``take()`` reads and resets it per item) rides the item's forward.json record
(``template_slots``, ``template_slots_evaluated``, ``template_scans``) and the pred's LEVER line (+ ``tables``: the tables made resident). ``install(model)`` binds the forward below to the model's
``evoformer.template_embedding`` instance; the class and the kit's files are untouched. The statements mirror the kit's own forward
(``tests/test_kit_statement_mirrors.py`` compares the restatement's source against the kit's, live, at test time).
"""
from __future__ import annotations

import dataclasses
import types

COUNTS = {"calls": 0, "slots": 0, "evaluated": 0, "scans": 0}
_MEMO = {"key": None, "reps": None}          # the item's distinct-slot map and the identity of the template feature tensors it was found on (begin_item / take clear it)
TABLES = ("RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX", "RESTYPE_PSEUDOBETA_INDEX")   # xfold.protein_data_processing's constant index tables the template embedder moves to the device per slot evaluation (template.py construct_input; scoring.py pseudo_beta_fn)
TABLES_MODULE = "xfold.protein_data_processing"


def take() -> dict:
    """The census since the last take (and reset it): {'calls', 'slots', 'evaluated', 'scans'}; forgets the item's distinct-slot map."""
    out = dict(COUNTS)
    for k in COUNTS:
        COUNTS[k] = 0
    begin_item()
    return out


def begin_item() -> None:
    """Forget the distinct-slot map: the next call searches again (forward.py: before every item, so a map never outlives its item's tensors)."""
    _MEMO["key"] = None; _MEMO["reps"] = None


def _slot_key(fields, num_templates: int):
    """The identity of a call's template feature tensors: which tensors (storage address, shape, dtype, device per field), not their values —
    host-side facts, no synchronisation. The same Templates object (every recycle of one item) gives the same key."""
    return (int(num_templates),) + tuple((k, v.data_ptr(), tuple(v.shape), str(v.dtype), str(v.device)) for k, v in fields)


def _slot_fields(templates):
    """The per-slot feature tensors of an xfold ``Templates`` (every dataclass field; a plain object: every tensor attribute)."""
    import torch
    if dataclasses.is_dataclass(templates):
        items = [(f.name, getattr(templates, f.name)) for f in dataclasses.fields(templates)]
    else:
        items = sorted(vars(templates).items())
    return [(k, v) for k, v in items if torch.is_tensor(v)]


def distinct_slots(templates, num_templates: int):
    """[representative slot index per slot]: slot t maps to the first slot u <= t whose every feature tensor equals slot t's. Found once per
    item: a call on the tensors the last search ran on returns that search's map (``_MEMO``); other tensors are searched (``COUNTS['scans']``)."""
    import torch
    fields = _slot_fields(templates)
    key = _slot_key(fields, num_templates)
    if _MEMO["key"] == key and _MEMO["reps"] is not None:
        return list(_MEMO["reps"])
    reps = []
    for t in range(num_templates):
        r = t
        for u in sorted(set(reps)):
            if all(torch.equal(v[t], v[u]) for _, v in fields):
                r = u
                break
        reps.append(r)
    COUNTS["scans"] += 1
    _MEMO["key"] = key; _MEMO["reps"] = tuple(reps)
    return reps


def place_tables(model) -> dict:
    """Make the template embedder's constant index tables (``TABLES``) resident on the model's device: {'tables': [names placed now],
    'device': str | None}. A model without parameters on a CUDA device (the CPU tests' stand-ins) places nothing. Idempotent: a table
    already on the device is left as it is."""
    import importlib
    import torch
    dev = None
    params = getattr(model, "parameters", None)
    if callable(params):
        for prm in params():
            dev = prm.device
            break
    if dev is None or dev.type != "cuda":
        return {"tables": [], "device": None if dev is None else str(dev)}
    mod = importlib.import_module(TABLES_MODULE)
    placed = []
    for name in TABLES:
        t = getattr(mod, name, None)
        if not torch.is_tensor(t):
            raise RuntimeError(f"template_dedupe: {TABLES_MODULE}.{name} is not a tensor (the kit's table changed shape)")
        if t.device != dev:
            setattr(mod, name, t.to(device=dev))
            placed.append(name)
    return {"tables": placed, "device": str(dev)}


def forward(self, query_embedding, templates, padding_mask_2d, multichain_mask_2d):
    """``TemplateEmbedding.forward`` with distinct-slot evaluation (bound to the instance by :func:`install`)."""
    import torch
    num_templates = templates.aatype.shape[0]
    num_res, _, _ = query_embedding.shape

    summed_template_embeddings = query_embedding.new_zeros(
        num_res, num_res, self.num_channels)

    reps = distinct_slots(templates, num_templates)
    evaluated = {}
    for template_idx in range(num_templates):
        r = reps[template_idx]
        if r not in evaluated:
            evaluated[r] = self.single_template_embedding(
                query_embedding, templates[r], padding_mask_2d, multichain_mask_2d
            )
        template_embedding = evaluated[r]

        summed_template_embeddings += template_embedding

    COUNTS["calls"] += 1; COUNTS["slots"] += int(num_templates); COUNTS["evaluated"] += len(evaluated)

    embedding = summed_template_embeddings / (1e-7 + num_templates)

    embedding = torch.relu(embedding)

    embedding = self.output_linear(embedding)

    return embedding


def install(model) -> dict:
    """Bind :func:`forward` to ``model.evoformer.template_embedding`` (the one TemplateEmbedding of the xfold tree) and make the embedder's
    constant index tables resident on the model's device (:func:`place_tables`). Idempotent. {'installed', 'already', 'tables', 'device'}."""
    te = model.evoformer.template_embedding
    if getattr(te, "_template_dedupe", False):
        return {"installed": True, "already": True, "tables": [], "device": None}
    for name in ("single_template_embedding", "output_linear", "num_channels"):
        if not hasattr(te, name):
            raise RuntimeError(f"template_dedupe: {type(te).__name__} has no {name!r} (the kit's TemplateEmbedding changed shape)")
    placed = place_tables(model)
    te.forward = types.MethodType(forward, te)
    te._template_dedupe = True
    begin_item()
    return {"installed": True, "already": False, **placed}
