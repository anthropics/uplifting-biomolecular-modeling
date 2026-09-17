"""The AlphaFold-2 TEMPLATE EMBEDDING born on pair ROWS (dm-haiku, ``alphafold.model.modules`` / ``alphafold.model.modules_multimer``): the
``model`` site group's template term of :mod:`alphafold` for the AF2 recipe, in one module that names no engine.

What it does. With the trunk plan of :func:`alphafold.install` in place (the template pair STACK — ``TemplatePairStack`` / ``TemplateEmbeddingIteration``
— and its pair sub-layers already rebound to row bodies), :func:`install_template` rebinds the template EMBEDDING classes so that one
``TemplateEmbedding.__call__`` runs as ONE rows region (``shard_map`` over: the query pair ROWS ``[N/P, N, c_z]``; the ``[N, N]`` masks REPLICATED and
sliced to this device's rows inside; the per-template fields ``[T, N]`` / ``[T, N, 37, 3]`` / ``[T, N, 37]`` REPLICATED — O(T·N); the key replicated):

| class | inside the region (rows ``i`` = this device's ``N/P`` residues, columns ``j`` = all ``N``) | collective | numerics class |
|---|---|---|---|
| ``modules.TemplateEmbedding.__call__`` (monomer; cf ``modules.py:2215-2280``, dl ``:2044-2109``) | THE REGION: per template the row block of the single-template embedding (below), stacked ``[T, N/P, N, c_t]``; the template-pointwise ``Attention`` over this device's ``N/P·N`` points (``flat_query [N/P·N, 1, c_z]`` vs ``flat_templates [N/P·N, T, c_t]``, the stock module and ``inference_subbatch`` policy unchanged, the reshapes on the LOCAL block); ``× (Σ template_mask > 0)``; rows OUT — the template pair is never whole on a device and never all-gathered | ``haiku.region`` (or the caller's region builder), ``shard.constrain`` | moves_bytes |
| ``modules.SingleTemplateEmbedding.__call__`` (cf ``:2108-2200``, dl ``:1941-2029``) | the ``[N/P, N, 88]`` feature block of my rows: :func:`dgram_rows` (my pseudo-beta rows vs all), pseudo-beta ``mask_2d`` rows, the aatype one-hot tiles (column term whole, row term = my rows), the unit vectors from my rows' backbone frames against every residue's translation (``quat_affine``), backbone ``mask_2d`` rows; ``embedding2d``; the (already rebound) ``TemplatePairStack`` on local rows — the NESTED path of :mod:`alphafold` (inside a region its entry runs the stock stack body on the blocks it is given); ``output_layer_norm`` on rows | ``shard.local_block`` (a ``dynamic_slice`` of a replicated operand) | row_local |
| ``modules_multimer.TemplateEmbedding.__call__`` (``modules_multimer.py:783-852``) | THE REGION: ``hk.scan`` over templates summing the single-template row blocks into ``[N/P, N, c]``; ``/ T``; ReLU; ``output_linear`` on rows; rows OUT | same | moves_bytes |
| ``modules_multimer.SingleTemplateEmbedding.__call__`` (``:863-1001``) | ``construct_input`` on my rows: dgram rows × pseudo-beta mask rows × multichain rows, aatype one-hots (``[1, N, 22]`` column term whole; ``[N/P, 1, 22]`` row term mine), unit vectors ``rigid[my rows, None]⁻¹ ∘ translation[all]`` (``folding_multimer.make_backbone_affine`` on all residues, the frames sliced to my rows), backbone mask rows, ``query_embedding_norm`` on the query rows, the nine ``template_pair_embedding_i`` Linears summed; the (already rebound) ``TemplateEmbeddingIteration`` stack on local rows with the ROW padding mask; ``output_layer_norm`` rows | ``shard.local_block`` | row_local |

Dispatch. Every rebound body runs the STOCK body untouched unless its switch is on: the two ``TemplateEmbedding`` entries build the region only when
``active()`` (the caller's predicate: the model group is installed and this call is not already inside a region) — otherwise stock; the two
``SingleTemplateEmbedding`` bodies run their row form only while :func:`template_rows` is set by the enclosing region body — otherwise stock. The
region is built by ``region`` (:func:`install_template`): the CALLER's region builder in :func:`haiku.region`'s positional form ``(body, rmesh, in_specs,
out_specs)`` so the caller's trace state is raised inside (the nested template pair stack and its sub-layers dispatch on the caller's predicates);
the default is :func:`haiku.region`. Inside the body the caller's ``in_region()`` must be True — refused BY NAME otherwise (a caller whose predicates do
not see this region would build a nested ``shard_map``).

Contract: ``N % P == 0`` at entry (:func:`shard.local_extent` refuses by name — the caller pads the model call or the kit buckets); P = 1 installs
nothing (the caller gates it); the transcribed files are PINNED (:data:`TEMPL_PIN` = the sha256 keys of :data:`alphafold.PIN`; refused by name on other
bytes); the trunk plan must already be in the caller's :class:`PatchSet` (``TemplatePairStack.__call__`` / ``TemplateEmbeddingIteration.__call__``
rebound: an un-rebound stack would run dense sub-layers on a row block). Tree facts the bodies branch on are read from the
pinned source (:func:`tree_facts`): the monomer ``float32`` cast of the template atom positions under ``global_config.bfloat16`` (cf ``:2145-2146``;
absent in dl) and the monomer ``output_layer_norm`` class (``common_modules.LayerNorm`` cf ``:2199`` / ``hk.LayerNorm`` dl ``:2028``).

HAZARDS (:data:`TEMPL_HAZARDS`): the monomer region draws ONE ``hk.next_rng_key()`` OUTSIDE the region and splits it per template inside, threading
each template's key through the per-template map as the batch field :data:`TEMPLATE_KEY_FIELD` (the stock draws ``hk.next_rng_key()`` inside
``TemplatePairStack`` once per template — not possible under the rng-less region frame; at inference dropout is off and no output depends on the keys;
the haiku rng sequence after the embedding differs from the stock's); masks enter whole and are sliced per device; module-state trace flag
(:func:`template_rows`, thread-local). DRY: every collective / slice / spec is :mod:`shard` / :mod:`triatt` / :mod:`haiku`; nothing of :mod:`alphafold`
is imported — the plan's facts arrive as :func:`install_template` arguments. Standard library at import; jax / haiku inside the bodies.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

from .. import MemLeverRefused
from . import LEVER as FAMILY_LEVER
from . import _lazy
from . import haiku as _hk
from . import shard as _shard
from . import triatt as _triatt
from .alphafold import PIN_FILES, _site_norm, check_pin, file_sha       # the family's ONE pin table / pin unit / pin gate / LayerNorm-class choice (alphafold imports this module lazily: no cycle)

TEMPL_PIN: Dict[str, str] = {**PIN_FILES["modules"], **PIN_FILES["modules_multimer"]}   # the module files the bodies transcribe (a view of alphafold.PIN)
TEMPL_HAZARDS = ("monomer_template_keys(one_draw_outside_region,split_per_template;inference_dropout_off)", "masks_replicated(sliced_per_device)",
                 "module_state_trace_flag(template_rows;thread_local)", "caller_in_region_checked_inside_region(refused_by_name)")
TEMPL_BODIES = {"TemplateEmbedding.__call__": ("moves_bytes", "cf modules.py:2215-2280 / dl modules.py:2044-2109 (the region; pointwise attention on local points)"),
                "SingleTemplateEmbedding.__call__": ("row_local", "cf modules.py:2108-2200 / dl modules.py:1941-2029 (feature rows; nested TemplatePairStack)"),
                "multimer.TemplateEmbedding.__call__": ("moves_bytes", "modules_multimer.py:783-852 (the region; template scan-sum, relu, output_linear on rows)"),
                "multimer.SingleTemplateEmbedding.__call__": ("row_local", "modules_multimer.py:863-1001 (construct_input rows; nested TemplateEmbeddingIteration stack)"),
                "dgram_rows": ("row_local", "cf modules.py:1677-1706 = dl modules.py:1517-1546 dgram_from_positions, the i operand sliced to my rows")}

MONOMER_TEMPLATE_FIELDS = ("template_mask", "template_aatype", "template_pseudo_beta", "template_pseudo_beta_mask", "template_all_atom_positions",
                           "template_all_atom_masks")                                                   # the template_batch leaves the monomer bodies read
MULTIMER_TEMPLATE_FIELDS = ("template_aatype", "template_all_atom_positions", "template_all_atom_mask")
TEMPLATE_KEY_FIELD = "template_rowpair_key"                                                              # per-template key [T, 2] threaded through the monomer map

_EMPTY: Dict[str, Any] = {"patches": None, "modules": None, "multimer": None, "rmesh": None, "axis": None, "n_gpu": 1, "lever": FAMILY_LEVER,
                          "active": None, "in_region": None, "region": None, "stock": {}, "bound": {}, "sites": [], "shas": {}, "facts": {},
                          "regions_built": 0}
_TPLAN: Dict[str, Any] = dict(_EMPTY)


class _TraceState(threading.local):
    """The trace-time switch of the single-template row bodies: raised by the enclosing template region body while it maps the templates (one thread
    traces a jitted function; a second thread sees its own state)."""
    def __init__(self):
        super().__init__()
        self.rows = 0


_T = _TraceState()


def template_rows() -> bool:
    """True while a template region body maps its templates: the ``SingleTemplateEmbedding`` bodies run their row form (query / masks are row blocks)."""
    return int(_T.rows) > 0


class _RowsFlag:
    def __enter__(self):
        _T.rows = int(_T.rows) + 1

    def __exit__(self, *exc):
        _T.rows = int(_T.rows) - 1
        return False


def installed() -> bool:
    """True while the rebinds of :func:`install_template` are live on the recorded modules (the caller's ``PatchSet.restore()`` ends it)."""
    M, bound = _TPLAN["modules"], _TPLAN["bound"]
    return M is not None and bool(bound) and getattr(M.TemplateEmbedding, "__call__", None) is bound.get("TemplateEmbedding")


def describe() -> Dict[str, Any]:
    """Evidence words of the template term: ``template`` (``rows`` while installed, else ``replicated``), ``template_sites``, ``template_masks``,
    ``template_regions_built``, ``template_library`` (short shas), ``template_hazards``."""
    return {"template": "rows" if installed() else "replicated", "template_sites": ",".join(_TPLAN["sites"]) if installed() else "none",
            "template_masks": "replicated", "template_regions_built": int(_TPLAN["regions_built"]),
            "template_library": ",".join(f"{k.rsplit('.', 1)[-1]}:{v[:8]}" for k, v in sorted(_TPLAN["shas"].items())) or "none",
            "template_hazards": ",".join(TEMPL_HAZARDS)}


def tree_facts(modules: Any) -> Dict[str, Any]:
    """Facts of the pinned monomer ``SingleTemplateEmbedding.__call__`` source the row body branches on: ``bf16_cast`` — the template atom positions are
    cast to float32 under ``global_config.bfloat16`` (the 2.3.x multimer-era tree's modules.py:2145-2146; absent in the dl_binder_design monomer tree);
    ``output_norm_common`` — ``output_layer_norm`` is ``common_modules.LayerNorm`` (cf :2199) rather than ``hk.LayerNorm`` (dl :2028)."""
    import inspect  # noqa: PLC0415
    src = inspect.getsource(_hk.stock_body(modules.SingleTemplateEmbedding, "__call__"))
    return {"bf16_cast": "raw_atom_pos = raw_atom_pos.astype(jnp.float32)" in src,
            "output_norm_common": "common_modules.LayerNorm([-1], True, True, name='output_layer_norm')" in src}


def dgram_rows(positions_rows: Any, positions_all: Any, num_bins: Any, min_bin: Any, max_bin: Any, lever: str = FAMILY_LEVER):
    """``dgram_from_positions`` (the 2.3.x tree's modules.py:1677-1706 = dl_binder_design modules.py:1517-1546) for a ROW block: the distogram
    one-hots ``[n_rows, N, num_bins]`` of ``positions_rows [n_rows, 3]`` (my residues ``i``) against ``positions_all [N, 3]`` (every ``j``) with the
    stock's exact bin arithmetic (``lower_breaks = linspace(min_bin, max_bin, num_bins)²``, ``upper = [lower[1:], 1e8]``, ``dist2 = Σ (x_i − x_j)²`` with
    ``keepdims``, ``(dist2 > lower) · (dist2 < upper)`` in float32). ``dgram_rows(x[rows], x, …) == dgram_from_positions(x, …)[rows]`` bit-exact. Accepts
    ``**config.dgram_features`` / ``**config.prev_pos``."""
    jnp = _lazy.jnp(lever)
    lower_breaks = jnp.linspace(min_bin, max_bin, num_bins)
    lower_breaks = jnp.square(lower_breaks)
    upper_breaks = jnp.concatenate([lower_breaks[1:], jnp.array([1e8], dtype=jnp.float32)], axis=-1)
    dist2 = jnp.sum(jnp.square(jnp.expand_dims(positions_rows, axis=-2) - jnp.expand_dims(positions_all, axis=-3)), axis=-1, keepdims=True)
    dgram = ((dist2 > lower_breaks).astype(jnp.float32) * (dist2 < upper_breaks).astype(jnp.float32))
    return dgram


def install_template(patches: Any, modules: Any, multimer: Any = None, *, lever: Optional[str] = None, rmesh: Any, axis: str, n_gpu: int,
                     active: Callable[[], bool], in_region: Callable[[], bool], region: Optional[Callable[..., Any]] = None,
                     pin: Optional[Dict[str, str]] = None, _test_single_device_mesh: bool = False) -> List[str]:
    """Rebind ``modules.TemplateEmbedding.__call__`` + ``modules.SingleTemplateEmbedding.__call__`` (and the two ``modules_multimer`` classes when ``multimer``
    is given) INTO ``patches`` (the caller's :class:`PatchSet`; its ``restore()`` undoes them) so the template embedding runs as one rows region on ``rmesh``
    (axis ``axis``, ``n_gpu`` = P). ``active()`` / ``in_region()``: the caller's predicates (build the region here / already inside a region → stock);
    ``region``: the caller's region builder ``(body, rmesh, in_specs, out_specs) -> mapped`` (default :func:`haiku.region`). Returns the site names.
    Refused by name: an unpinned tree (:func:`check_pin`)."""
    lv = str(lever or FAMILY_LEVER)
    n = int(n_gpu or 0)
    shas = check_pin(modules, multimer, TEMPL_PIN if pin is None else pin, lv)
    facts = tree_facts(modules)
    reg = region if region is not None else (lambda body, rmesh_, in_specs, out_specs: _hk.region(body, rmesh_, in_specs, out_specs, lever=lv))
    stock: Dict[str, Any] = {}
    sites: List[str] = []
    stock["TemplateEmbedding"] = _hk.rebind(patches, modules.TemplateEmbedding, "__call__", _template_embedding_call, lv)
    sites.append("TemplateEmbedding.__call__")
    stock["SingleTemplateEmbedding"] = _hk.rebind(patches, modules.SingleTemplateEmbedding, "__call__", _single_template_call, lv)
    sites.append("SingleTemplateEmbedding.__call__")
    if multimer is not None:
        stock["multimer.TemplateEmbedding"] = _hk.rebind(patches, multimer.TemplateEmbedding, "__call__", _template_embedding_multimer_call, lv)
        sites.append("multimer.TemplateEmbedding.__call__")
        stock["multimer.SingleTemplateEmbedding"] = _hk.rebind(patches, multimer.SingleTemplateEmbedding, "__call__", _single_template_multimer_call, lv)
        sites.append("multimer.SingleTemplateEmbedding.__call__")
    bound = {"TemplateEmbedding": getattr(modules.TemplateEmbedding, "__call__")}
    _T.rows = 0
    _TPLAN.update(patches=patches, modules=modules, multimer=multimer, rmesh=rmesh, axis=str(axis), n_gpu=n, lever=lv, active=active, in_region=in_region,
                  region=reg, stock=stock, bound=bound, sites=sites, shas=shas, facts=facts, regions_built=0)
    return list(sites)


def forget() -> None:
    """Drop the recorded plan (after the caller's ``PatchSet.restore()``; :func:`installed` is already False then — this only clears the record)."""
    _TPLAN.clear()
    _TPLAN.update(dict(_EMPTY))
    _T.rows = 0


# ----------------------------------------------------------------------------------------------------------------- helpers inside the bodies
def _refuse_unless_in_region(where: str):
    if not _TPLAN["in_region"]():
        raise MemLeverRefused(_TPLAN["lever"], f"install_template: the caller's in_region() is False inside the template region ({where}) — pass region= "
                                                "(the caller's region builder that raises its trace state); a nested shard_map is never built")


# ----------------------------------------------------------------------------------------------------------------- monomer bodies
def _template_embedding_call(self, query_embedding, template_batch, mask_2d, is_training):
    """``modules.TemplateEmbedding.__call__`` (cf modules.py:2215-2280, dl :2044-2109) as ONE rows region: per-template row blocks, the pointwise attention
    over this device's points, rows out. Outside the model group or inside a region: the stock body."""
    P = _TPLAN
    stock = P["stock"]["TemplateEmbedding"]
    if not (P["active"]() and not P["in_region"]()):
        return stock(self, query_embedding, template_batch, mask_2d, is_training)
    M, rmesh, axis, lv = P["modules"], P["rmesh"], P["axis"], P["lever"]
    jax, jnp = _lazy.jax(lv), _lazy.jnp(lv)
    missing = [k for k in MONOMER_TEMPLATE_FIELDS if k not in template_batch]
    if missing:
        raise MemLeverRefused(lv, f"install_template: template_batch lacks {','.join(missing)} (the monomer template fields the row bodies read)")
    num_templates = template_batch["template_mask"].shape[0]
    num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
    num_res = query_embedding.shape[0]
    n_loc = _shard.local_extent(int(num_res), int(P["n_gpu"]), lv)          # N % P != 0 is refused by name (the caller pads / the kit buckets)
    query_num_channels = query_embedding.shape[-1]
    key = M.hk.next_rng_key()                                                # ONE draw outside the region (HAZARD monomer_template_keys)
    fields = {k: template_batch[k] for k in MONOMER_TEMPLATE_FIELDS}
    rows, rep = _shard.rows_spec(axis, 3, 0), _shard.replicated_spec()
    query_rows = _shard.constrain(query_embedding, rmesh)                     # the query pair enters in the row layout

    def body(q_rows, mask_full, tb, key_b):
        _refuse_unless_in_region("TemplateEmbedding")
        dtype = q_rows.dtype
        template_mask = tb["template_mask"].astype(dtype)
        mask_rows = _triatt.mask_rows(mask_full, axis, n_loc)               # my rows of the [N, N] padding mask
        tb = dict(tb)
        tb[TEMPLATE_KEY_FIELD] = jax.random.split(key_b, num_templates)     # [T, 2]: one key per template, mapped with the template fields
        # Make sure the weights are shared across templates by constructing the embedder here (stock :2242).
        template_embedder = M.SingleTemplateEmbedding(self.config, self.global_config)

        def map_fn(batch):
            return template_embedder(q_rows, batch, mask_rows, is_training)

        with _RowsFlag():
            template_pair_representation = M.mapping.sharded_map(map_fn, in_axes=0)(tb)        # [T, N/P, N, c_t]
        # Cross attend from the query to the templates along the residue dimension by flattening everything else into the batch dimension —
        # on THIS device's N/P·N points (stock :2251-2274 on N·N points).
        flat_query = jnp.reshape(q_rows, [n_loc * num_res, 1, query_num_channels])
        flat_templates = jnp.reshape(jnp.transpose(template_pair_representation, [1, 2, 0, 3]), [n_loc * num_res, num_templates, num_channels])
        bias = (1e9 * (template_mask[None, None, None, :] - 1.))
        template_pointwise_attention_module = M.Attention(self.config.attention, self.global_config, query_num_channels)
        nonbatched_args = [bias]
        batched_args = [flat_query, flat_templates]
        embedding = M.mapping.inference_subbatch(template_pointwise_attention_module, self.config.subbatch_size, batched_args=batched_args,
                                                 nonbatched_args=nonbatched_args, low_memory=not is_training)
        embedding = jnp.reshape(embedding, [n_loc, num_res, query_num_channels])
        # No gradients if no templates (stock :2278).
        embedding *= (jnp.sum(template_mask) > 0.).astype(embedding.dtype)
        return embedding

    mapped = P["region"](body, rmesh, (rows, rep, {k: rep for k in fields}, rep), rows)
    P["regions_built"] += 1
    out = mapped(query_rows, mask_2d, fields, key)
    return _shard.constrain(out, rmesh)


def _single_template_call(self, query_embedding, batch, mask_2d, is_training):
    """``modules.SingleTemplateEmbedding.__call__`` (cf modules.py:2108-2200, dl :1941-2029) on my rows: ``query_embedding`` / ``mask_2d`` are ROW blocks
    ``[N/P, N, ·]``, ``batch`` holds ONE template's whole-N fields (+ :data:`TEMPLATE_KEY_FIELD`). Outside a template region: the stock body."""
    P = _TPLAN
    stock = P["stock"]["SingleTemplateEmbedding"]
    if not template_rows():
        return stock(self, query_embedding, batch, mask_2d, is_training)
    M, axis, lv = P["modules"], P["axis"], P["lever"]
    jax, jnp = _lazy.jax(lv), _lazy.jnp(lv)
    quat_affine, residue_constants, common_modules = M.quat_affine, M.residue_constants, M.common_modules
    assert mask_2d.dtype == query_embedding.dtype
    dtype = query_embedding.dtype
    num_res = batch["template_aatype"].shape[0]
    n_loc = int(query_embedding.shape[0])
    assert int(mask_2d.shape[0]) == n_loc and int(mask_2d.shape[1]) == int(num_res), (mask_2d.shape, query_embedding.shape, num_res)

    def rows_of(x, dim=0):                                                  # my rows of a whole-N per-residue operand
        return _shard.local_block(x, axis, n_loc, dim)

    num_channels = (self.config.template_pair_stack.triangle_attention_ending_node.value_dim)
    template_mask = batch["template_pseudo_beta_mask"]
    template_mask_2d = rows_of(template_mask)[:, None] * template_mask[None, :]
    template_mask_2d = template_mask_2d.astype(dtype)

    template_dgram = dgram_rows(rows_of(batch["template_pseudo_beta"]), batch["template_pseudo_beta"], lever=lv, **self.config.dgram_features)
    template_dgram = template_dgram.astype(dtype)

    to_concat = [template_dgram, template_mask_2d[:, :, None]]

    aatype = jax.nn.one_hot(batch["template_aatype"], 22, axis=-1, dtype=dtype)

    to_concat.append(jnp.tile(aatype[None, :, :], [n_loc, 1, 1]))           # (i, j) ← aatype[j]: every column, my rows
    to_concat.append(jnp.tile(rows_of(aatype)[:, None, :], [1, num_res, 1]))  # (i, j) ← aatype[i]: my rows' residues

    n, ca, c = [residue_constants.atom_order[a] for a in ("N", "CA", "C")]
    raw_atom_pos = batch["template_all_atom_positions"]
    if P["facts"].get("bf16_cast") and self.global_config.bfloat16:         # cf :2145-2146 (the dl tree has no such line)
        raw_atom_pos = raw_atom_pos.astype(jnp.float32)

    rot, trans = quat_affine.make_transform_from_reference(n_xyz=raw_atom_pos[:, n], ca_xyz=raw_atom_pos[:, ca], c_xyz=raw_atom_pos[:, c])
    points = [jnp.expand_dims(x, axis=-2) for x in jnp.moveaxis(trans, -1, 0)]   # every residue's translation, unstacked as QuatAffine(unstack_inputs=True).translation: [1, N] each
    rot_rows, trans_rows = rows_of(rot), rows_of(trans)                       # my rows' backbone frames
    affines = quat_affine.QuatAffine(quaternion=quat_affine.rot_to_quat(rot_rows, unstack_inputs=True), translation=trans_rows, rotation=rot_rows,
                                     unstack_inputs=True)
    affine_vec = affines.invert_point(points, extra_dims=1)                  # [N/P, N] each: R_iᵀ (t_j − t_i) for my rows i
    inv_distance_scalar = jax.lax.rsqrt(1e-6 + sum([jnp.square(x) for x in affine_vec]))

    # Backbone affine mask: whether the residue has C, CA, N (the template mask defined above only considers pseudo CB).
    template_mask = (batch["template_all_atom_masks"][..., n] * batch["template_all_atom_masks"][..., ca] * batch["template_all_atom_masks"][..., c])
    template_mask_2d = rows_of(template_mask)[:, None] * template_mask[None, :]

    inv_distance_scalar *= template_mask_2d.astype(inv_distance_scalar.dtype)

    unit_vector = [(x * inv_distance_scalar)[..., None] for x in affine_vec]

    unit_vector = [x.astype(dtype) for x in unit_vector]
    template_mask_2d = template_mask_2d.astype(dtype)
    if not self.config.use_template_unit_vector:
        unit_vector = [jnp.zeros_like(x) for x in unit_vector]
    to_concat.extend(unit_vector)

    to_concat.append(template_mask_2d[..., None])

    act = jnp.concatenate(to_concat, axis=-1)                                # [N/P, N, 88]

    # Mask out non-template regions so we don't get arbitrary values in the distogram for these regions.
    act *= template_mask_2d[..., None]

    act = common_modules.Linear(num_channels, initializer="relu", name="embedding2d")(act)

    # the template pair stack on my rows: the trunk plan's rebound entry runs the stock stack body on the blocks it is given inside a region
    act = M.TemplatePairStack(self.config.template_pair_stack, self.global_config)(act, mask_2d, is_training, safe_key=M.prng.SafeKey(batch[TEMPLATE_KEY_FIELD]))

    act = _site_norm(M, "output_layer_norm")(act)                         # cf modules.py:2199 common_modules.LayerNorm / dl modules.py:2028 hk.LayerNorm (the tree's pair LayerNorm class)
    return act


# ----------------------------------------------------------------------------------------------------------------- multimer bodies
def _template_embedding_multimer_call(self, query_embedding, template_batch, padding_mask_2d, multichain_mask_2d, is_training, safe_key=None):
    """``modules_multimer.TemplateEmbedding.__call__`` (modules_multimer.py:783-852) as ONE rows region: the per-template row blocks summed by ``hk.scan``,
    ``/ T``, ReLU, ``output_linear`` — rows out. Outside the model group or inside a region: the stock body."""
    P = _TPLAN
    stock = P["stock"]["multimer.TemplateEmbedding"]
    if not (P["active"]() and not P["in_region"]()):
        return stock(self, query_embedding, template_batch, padding_mask_2d, multichain_mask_2d, is_training, safe_key=safe_key)
    MM, rmesh, axis, lv = P["multimer"], P["rmesh"], P["axis"], P["lever"]
    jax, jnp = _lazy.jax(lv), _lazy.jnp(lv)
    missing = [k for k in MULTIMER_TEMPLATE_FIELDS if k not in template_batch]
    if missing:
        raise MemLeverRefused(lv, f"install_template: template_batch lacks {','.join(missing)} (the multimer template fields the row bodies read)")
    c = self.config
    if safe_key is None:
        safe_key = MM.prng.SafeKey(MM.hk.next_rng_key())                     # stock :812-813 — outside the region, at the stock's position

    num_templates = template_batch["template_aatype"].shape[0]
    num_res, _, query_num_channels = query_embedding.shape
    n_loc = _shard.local_extent(int(num_res), int(P["n_gpu"]), lv)          # N % P != 0 is refused by name
    safe_key, unsafe_key = safe_key.split()
    unsafe_keys = jax.random.split(unsafe_key._key, num_templates)           # stock :830-831
    fields = {k: template_batch[k] for k in MULTIMER_TEMPLATE_FIELDS}
    rows, rep = _shard.rows_spec(axis, 3, 0), _shard.replicated_spec()
    query_rows = _shard.constrain(query_embedding, rmesh)

    def body(q_rows, pad_full, multi_full, tb, ukeys):
        _refuse_unless_in_region("multimer.TemplateEmbedding")
        pad_rows = _triatt.mask_rows(pad_full, axis, n_loc)                 # my rows of the [N, N] padding / multichain masks
        multi_rows = _triatt.mask_rows(multi_full, axis, n_loc)
        # Embed each template separately (stock :819-828).
        template_embedder = MM.SingleTemplateEmbedding(self.config, self.global_config)

        def partial_template_embedder(template_aatype, template_all_atom_positions, template_all_atom_mask, unsafe_key):
            safe_key = MM.prng.SafeKey(unsafe_key)
            return template_embedder(q_rows, template_aatype, template_all_atom_positions, template_all_atom_mask, pad_rows, multi_rows,
                                     is_training, safe_key)

        def scan_fn(carry, x):
            return carry + partial_template_embedder(*x), None

        scan_init = jnp.zeros((n_loc, num_res, c.num_channels), dtype=q_rows.dtype)
        with _RowsFlag():
            summed_template_embeddings, _ = MM.hk.scan(scan_fn, scan_init, (tb["template_aatype"], tb["template_all_atom_positions"],
                                                                          tb["template_all_atom_mask"], ukeys))

        embedding = summed_template_embeddings / num_templates
        embedding = jax.nn.relu(embedding)
        embedding = MM.common_modules.Linear(query_num_channels, initializer="relu", name="output_linear")(embedding)
        return embedding

    mapped = P["region"](body, rmesh, (rows, rep, rep, {k: rep for k in fields}, rep), rows)
    P["regions_built"] += 1
    out = mapped(query_rows, padding_mask_2d, multichain_mask_2d, fields, unsafe_keys)
    return _shard.constrain(out, rmesh)


def _single_template_multimer_call(self, query_embedding, template_aatype, template_all_atom_positions, template_all_atom_mask, padding_mask_2d,
                                   multichain_mask_2d, is_training, safe_key):
    """``modules_multimer.SingleTemplateEmbedding.__call__`` (modules_multimer.py:863-1001) on my rows: ``query_embedding``, ``padding_mask_2d`` and
    ``multichain_mask_2d`` are ROW blocks; the template fields are whole-N. Outside a template region: the stock body."""
    P = _TPLAN
    stock = P["stock"]["multimer.SingleTemplateEmbedding"]
    if not template_rows():
        return stock(self, query_embedding, template_aatype, template_all_atom_positions, template_all_atom_mask, padding_mask_2d,
                     multichain_mask_2d, is_training, safe_key)
    MM, axis, lv = P["multimer"], P["axis"], P["lever"]
    jax, jnp = _lazy.jax(lv), _lazy.jnp(lv)
    modules, common_modules, geometry, folding_multimer = MM.modules, MM.common_modules, MM.geometry, MM.folding_multimer
    gc = self.global_config
    c = self.config
    assert padding_mask_2d.dtype == query_embedding.dtype
    dtype = query_embedding.dtype
    num_channels = self.config.num_channels
    n_loc = int(query_embedding.shape[0])

    def rows_of(x, dim=0):                                                  # my rows of a whole-N per-residue operand
        return _shard.local_block(x, axis, n_loc, dim)

    def construct_input(query_embedding, template_aatype, template_all_atom_positions, template_all_atom_mask, multichain_mask_2d):
        # Compute distogram feature for the template (stock :897-908) — rows i mine, columns j all.
        template_positions, pseudo_beta_mask = modules.pseudo_beta_fn(template_aatype, template_all_atom_positions, template_all_atom_mask)
        pseudo_beta_mask_2d = (rows_of(pseudo_beta_mask)[:, None] * pseudo_beta_mask[None, :])
        pseudo_beta_mask_2d *= multichain_mask_2d
        template_dgram = dgram_rows(rows_of(template_positions), template_positions, lever=lv, **self.config.dgram_features)
        template_dgram *= pseudo_beta_mask_2d[..., None]
        template_dgram = template_dgram.astype(dtype)
        pseudo_beta_mask_2d = pseudo_beta_mask_2d.astype(dtype)
        to_concat = [(template_dgram, 1), (pseudo_beta_mask_2d, 0)]

        aatype = jax.nn.one_hot(template_aatype, 22, axis=-1, dtype=dtype)
        to_concat.append((aatype[None, :, :], 1))                            # [1, N, 22]: the column term, whole (broadcast over my rows)
        to_concat.append((rows_of(aatype)[:, None, :], 1))                   # [N/P, 1, 22]: the row term, my rows

        # Compute a feature representing the normalized vector between each backbone affine (stock :915-937): my rows' frames against every translation.
        raw_atom_pos = template_all_atom_positions
        if gc.bfloat16:
            # Vec3Arrays are required to be float32
            raw_atom_pos = raw_atom_pos.astype(jnp.float32)

        atom_pos = geometry.Vec3Array.from_array(raw_atom_pos)
        rigid, backbone_mask = folding_multimer.make_backbone_affine(atom_pos, template_all_atom_mask, template_aatype)
        points = rigid.translation
        rigid_rows = jax.tree_util.tree_map(rows_of, rigid)                  # my rows' backbone frames
        rigid_vec = rigid_rows[:, None].inverse().apply_to_point(points)
        unit_vector = rigid_vec.normalized()
        unit_vector = [unit_vector.x, unit_vector.y, unit_vector.z]

        if gc.bfloat16:
            unit_vector = [x.astype(jnp.bfloat16) for x in unit_vector]
            backbone_mask = backbone_mask.astype(jnp.bfloat16)

        backbone_mask_2d = rows_of(backbone_mask)[:, None] * backbone_mask[None, :]
        backbone_mask_2d *= multichain_mask_2d
        unit_vector = [x * backbone_mask_2d for x in unit_vector]

        # Note that the backbone_mask takes into account C, CA and N (unlike pseudo beta mask which just needs CB) so we add both masks as features.
        to_concat.extend([(x, 0) for x in unit_vector])
        to_concat.append((backbone_mask_2d, 0))

        query_embedding = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name="query_embedding_norm")(query_embedding)
        # Allow the template embedder to see the query embedding (my rows of it).
        to_concat.append((query_embedding, 1))

        act = 0

        for i, (x, n_input_dims) in enumerate(to_concat):
            act += common_modules.Linear(num_channels, num_input_dims=n_input_dims, initializer="relu", name=f"template_pair_embedding_{i}")(x)
        return act

    act = construct_input(query_embedding, template_aatype, template_all_atom_positions, template_all_atom_mask, multichain_mask_2d)

    template_iteration = MM.TemplateEmbeddingIteration(c.template_pair_stack, gc, name="template_embedding_iteration")

    def template_iteration_fn(x):
        act, safe_key = x

        safe_key, safe_subkey = safe_key.split()
        # the trunk plan's rebound entry: inside a region it runs the stock block on the row block with the ROW padding mask it is given
        act = template_iteration(act=act, pair_mask=padding_mask_2d, is_training=is_training, safe_key=safe_subkey)
        return (act, safe_key)

    if gc.use_remat:
        template_iteration_fn = MM.hk.remat(template_iteration_fn)

    safe_key, safe_subkey = safe_key.split()
    template_stack = MM.layer_stack.layer_stack(c.template_pair_stack.num_block)(template_iteration_fn)
    act, safe_key = template_stack((act, safe_subkey))

    act = common_modules.LayerNorm(axis=[-1], create_scale=True, create_offset=True, name="output_layer_norm")(act)
    return act
