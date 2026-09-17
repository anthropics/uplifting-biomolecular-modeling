"""`--mode big --n_gpu P>1`: protenix 1.1.0 END-TO-END tensor-parallel over `opt_core.mem.rowpair` — the pair representation z [N, N, c_z]
is ROW-SHARDED from the statement that creates it to the last statement that reads it; rank q holds rows [r0, r1) of the layout
(`opt_core.mem.rowpair.dist.Layout`, boundaries multiples of `ALIGN` = the pinned chunk grid) and nothing N x N x c is ever whole on a rank.
This module is the engine's ADAPTER: it names protenix's modules, packs their sub-module statements into the core's callables (TriMulFns,
TriAttFns, PairBlockFns, DiTBlockFns, rows functions) and installs them; every collective, block schedule and budget is the core's
(`opt_core/mem/rowpair/API.md`). The kit's process half — `--n_gpu` grammar, worker launch, feature replication, evidence — is `rowpair.py`.

Flow (stock statement -> what runs here; protenix/model/protenix.py `get_pairformer_output` / `main_inference_loop`):
  relp one-hot plane [N,N,139]  RelativePositionEncoding.generate_relp   -> `RelpRows`: the per-token ids only; rows [g0,g1) built per row block
                                                                           (`_relp.relp_rows_into`, the statements of embedders.py:150-201)
  z_init outer sum + relpos + bonds (protenix.py:209-228)                -> trunk.init_pair_shard(rows_fn) — z BORN as this rank's rows
  z = 0; per cycle z = z_init + linear(LN(z)) (:230, :246)               -> trunk.recycle_shard_ (cycle 0 = a zeros ROW BLOCK; the z_init shard
                                                                           placed between recycles by ROWPAIR_PARK_ZINIT (ngpu.TP_EXPORTS:
                                                                           `recompute` = row blocks re-run the init statements, `1` = parked on
                                                                           pinned host, trunk.ShardPark; census park_z_init=<where> ...)
  z += TemplateEmbedder(z) (pairformer.py:982-1098)                       -> template.template_embed_rows: v_t rows, the 2-block stack on rows
                                                                           through pairstack.pair_stack_, mean/relu/linear rows; the dense
                                                                           template pair features are sliced to LOCAL ROWS once per item
                                                                           (template.slice_template_inputs_to_rows)
  z = MSAModule(z) (:809-917; 4 blocks: OPM, pair-weighted averaging,     -> msa.opm_rows_budgeted (a rows x b) into the shard in place;
      transition, pair block)                                              msa.pwa_bias_rows + msa.pwa_rows per 2048-sequence chunk (softmax
                                                                           over j is row-local; o gathered over tokens before linear_out);
                                                                           transition_m on the replicated m; pairstack.pair_block_ on rows
  s, z = PairformerStack(s, z) (48 blocks)                                -> pairstack.pair_stack_ (tri-mult out ring / in bands, tri-att
                                                                           start row-local + gathered bias, tri-att end on transposed shards,
                                                                           transition row-local, pair-biased single attention = local query
                                                                           rows + all-gather of s rows)
  distogram logits + logits^T -> contact_probs (head.py:44-56)            -> heads.sym_logit_rows (column slabs exchanged) -> contact rows [R,N];
                                                                           taken at ROLL-OUT ENTRY from the device shard (rollout_entry), the
                                                                           trunk shard then PARKED on pinned host, storage released
                                                                           (heads.ZTrunkPlan.park_now; ROWPAIR_CONF_PARK_ZTRUNK=1, the line's export)
  ConfidenceHead (confidence.py:133-348), N_sample passes                 -> per sample = one pass of heads.ZTrunkPlan: heads.embed_rows (z_init
                                                                           outer sum + z_trunk rows served from the host park, or IN PLACE into
                                                                           the shard at a single same-dtype last-use pass: ROWPAIR_FREE_ZTRUNK=1)
                                                                           + distance one-hot rows, the 4-block stack IN PLACE on the pass's
                                                                           rows, PAE rows and PDE rows of z + z^T (sym exchange, fp32 per row
                                                                           block) consumed per row block by confidence.RowBlockReducer (exact
                                                                           finish) -> ptm/iptm/... on rank 0; device: ONE pair-shard-equivalent
                                                                           per pass at any N_sample (census conf_ztrunk=<word per pass>)
  DiffusionConditioning.prepare_cache (diffusion.py:86-107)               -> diffusion.pair_cond_rows (z_cond rows once per item)
  AtomAttentionEncoder.prepare_cache token-pair windows (transformer.py)  -> diffusion.pair_band_rows + band_lookup (the replicated band)
  DiffusionTransformer 24 blocks x steps x samples (transformer.py)       -> diffusion.diffusion_transformer_sharded (bias rows of local rows,
                                                                           queries = local rows, ONE all-gather of a rows per block)
  compute_full_data_and_summary (sample_confidence.py:811)                -> rank 0 assembles the stock keys from the reducers (`_assemble_rank0`)

THE MSA REPRESENTATION m [S_msa<=16384, N|R, 64] is TOKEN-SHARDED by default (`PROTENIX_V1_TP_MSA_M=token_sharded`: rank q embeds and
holds m[:, r0:r1, :]; the OPM's b operand [N, S, 32] and the pair-weighted averaging's values source per S-chunk are the only gathers,
named with their bytes) or replicated by name (`=replicated`, census msa_m=replicated).

REPLICATED BY DESIGN (named; printed with their bytes in the schedule census `tp_replicated=`): s_inputs [N,449], s [N,384], the
per-token template features, the gathered triangle bias [N,N,4] per tri-att call, the diffusion activations a/s and atom tensors. HOST-RESIDENT
(never whole on a device): token_bonds [N,N] (rows move per init row block), the template pair features (sliced to this rank's rows on the
host), the raw MSA features [S_msa<=16384, N] (msa int64, has_deletion, deletion_value: ROWPAIR_MSA_HOST=rank0, the line's export — rank 0's
pinned host copy is the only one, the per-cycle SUBSAMPLED rows reach every rank's device by broadcast, msa_host.rows_to_device; `=all`:
every rank keeps a host copy and gathers its own rows; unset: replicated on the device as the stock, census msa_raw=device), and on RANK 0
ONLY the per-sample expected-value planes token_pair_pae/pde + contact_probs the writer needs ([N,N] fp16 each, host).

REPLACED / SUBSUMED P=1 levers (never silently dropped; off by name in every rank's selection — big.rowpair_switches over ngpu.SUPERSEDED_LEVERS):
relp_lean -> RelpRows (no plane at all), recycle_carry -> ShardPark of the z_init SHARD, diffusion_cond_chunk -> pair_cond_rows, conf_head_chunk
-> the row-block PAE/PDE producers; hoist / sg forced off (their pools are per rank); the fast / exact TriMul cells, gflash and the
cuEquivariance tri-mult kernel -> the core's torch tri-mult statement on rows (`trimul=rowpair_torch`); the triangle-attention kernel of the
run (`--triatt_kernel`) serves this rank's query rows unchanged; ttr (fused transition) composes (row-local); tg never captures (no stock
stack forward runs). No size gate: every seam is sharded at every admitted N (the layout refuses by name what it cannot shard).

REFUSED BY NAME under n_gpu>1: an ENABLED constraint embedder (pocket / contact / contact_atom / substructure: no rows form of ConstraintEmbedder,
`tp_constraint_unsupported`; all are off in the stock configs, when the dead pair-shaped constraint features are dropped before H2D), a leading batch dim,
(MC dropout — the stock's per-item `random.random() < mc_dropout_apply_rate` draw, proven identical across ranks — is SERVED: at N <=
`PROTENIX_V1_TP_MC_DROPOUT_FULL_MAX` (default 2048) every rank draws the stock statement's ONE keep-mask over the whole plane through the global generator,
exactly as `F.dropout` consumes it, and applies its rows — census `tp_mc_dropout=stock_mask(p)`, the process's later draws (MSA sample, diffusion
noise) stay the stock's; above it the rows' keep-mask comes from a rank-forked generator and the global generator is NOT consumed —
census `tp_mc_dropout=rank_streams(p)` + a NOTE `tier-2 by construction (rank_streams)`: an independent realisation of the seed from that item on),
fp16 autocast, --enable_cache false (`tp_needs_shared_vars_cache`), training mode.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Tuple

from . import _relp as RL

try:                                                         # instrumentation only (census marks; no-op unless OPT_CORE_TP_CENSUS=1)
    from opt_core.mem.rowpair import census as CENSUS
except ImportError:                                          # a core without the census module: a no-op
    CENSUS = None

ALIGN_ENV = "ROWPAIR_CHUNK_ALIGN"
MSA_M_ENV = "PROTENIX_V1_TP_MSA_M"                      # the MSA representation's layout under n_gpu>1: token_sharded (default: rank q holds m[:, r0:r1, :]) | replicated
MSA_M_LAYOUTS = ("token_sharded", "replicated")
MC_DROPOUT_FULL_MAX_ENV = "PROTENIX_V1_TP_MC_DROPOUT_FULL_MAX"   # n_gpu>1 MC dropout: N <= value -> `stock_mask` (the stock keep-mask over the whole plane on every rank,
MC_DROPOUT_FULL_MAX_DEFAULT = 2048                            # the global generator consumed as F.dropout consumes it); N > value -> `rank_streams` (tier-2 by construction). 0 = rank_streams at every N.
MC_DROPOUT_FORMS = ("stock_mask", "rank_streams")
TRIMUL_INPLACE_CHUNK = 256                              # protenix's in-place triangle-multiplication column chunk (triangular.py:476 `_inplace_chunk_size=256`): the ring schedule's grid
CONF_FINISH_ENV = "PROTENIX_V1_TP_CONF_FINISH"               # exact (default) | rowsum — confidence.RowBlockReducer finish
BAND_W_ENV = "ROWPAIR_DIFF_BAND_W"                           # the adapter's explicit cap of the atom-pair band half-width (None = as wide as needed)

STATE: Dict[str, object] = {"P": 1, "rank": 0, "kernel": "torch", "chunk": 128, "installed": False, "undo": [], "contact_rows": {}, "model": None}
COUNTS: Dict[str, int] = {k: 0 for k in (
    "items", "trunk_inits", "recycles", "template_calls", "template_rows", "msa_blocks_rows", "opm_rows", "pwa_calls", "pairstack_calls",
    "pair_blocks", "gathers_in_trunk", "distogram_rows", "conf_samples", "conf_rows", "zcond_rows", "band_calls", "dit_blocks", "dit_calls",
    "summaries", "weights_guard", "mc_dropout_items", "msa_host_calls", "dit_sdpa", "dit_stock_attn", "dit_rows_attn", "dit_rollouts", "dit_bias_computes")}
REPLICATED: Dict[str, int] = {}                              # name -> bytes (the census's tp_replicated field)
PARKS: Dict[str, dict] = {}                                  # name -> the park's record (opt_core.mem.rowpair.trunk.ShardPark.record(): where, gib, device_release; report.rowpair.parks)
SYNC = {"mode": "bcast", "det": False, "source": "default"}   # the replicated-tensor sync policy of this rank process (rowpair.install sets it: det 1 -> guard, det 0 -> bcast,
                                                             # ROWPAIR_DIFF_NOISE_SYNC=<mode> a named opt-in); read at every defined sync point below (RNG state, denoiser state in, denoiser update out, MSA sample, MC decision)


# ---------------------------------------------------------------- the xP line's triangle KERNELS (the core's row-sharded fused kernels, opt_core.mem.rowpair)
# The default statements are the torch contraction on rows (triangle multiplication) and the run's `--triatt_kernel` through the stock attention
# module on this rank's query rows (triangle attention) — the torch statements, which stay the named fallback of every unit the fused
# kernels decline (the core's size gate below 2048 tokens, an unsupported shape or dtype, a missing kernel: each a counted reason on the
# core's LEVER line). The xP line selects the fused row kernels here, by name; a selected kernel the installed core does not carry is a NAMED
# fallback to the torch statements (NOTE line + census word), never silent.
TP_TRIMUL_KERNELS = "fpf_v4"         # "torch" = opt_core.mem.rowpair.trimul's contraction with this adapter's torch TriMulFns (census tp_trimul=rowpair_torch)
                                     # | "fpf_v4" = the core's fused row-sharded TriMul (mem.rowpair.trimul_fused.fused_trimul_fns: K1 -> bmm -> K3 on row blocks, this kit's
                                     #   lib/fpf_trimul_v4/cells.json row for the card; tier 2; census tp_trimul=rowpair_fpf_v4 + the core's F2.trimul_rows LEVER line)
TP_TRIATT_CORE = "tier:big"        # None = the stock attention module with the run's --triatt_kernel on this rank's query rows (census tp_triatt_kernel=<word>)
                                     # | "flash_triattn" | "cueq" | "torch" = the core's per-row-block attention core (triatt.attention_core(kernel=<word>); the
                                     #   projections, gating and output stay the stock module's; census tp_triatt_core=<word> + the core's F1.flash_triattn LEVER line)
_TPX: Dict[str, object] = {}         # resolved once per process: {"RF": module|None, "RA_core": callable|None, "reason": str|None}


def tp_kernels() -> Dict[str, object]:
    """The row kernels this process CAN serve for the two switches above: the core's fused TriMul module (RF) when TP_TRIMUL_KERNELS names it
    and the installed core carries it (the module / the constructor is importable: presence, not a version number), the core's
    attention-core constructor (RA_core) likewise; else None with the reason word (`core_<version>_lacks_row_kernels`), announced once as a
    NOTE. Nothing is imported when both switches name the torch statements."""
    if _TPX:
        return _TPX
    want_trimul, want_triatt = TP_TRIMUL_KERNELS != "torch", TP_TRIATT_CORE is not None
    RF = RA_core = None; reason = None
    if want_trimul or want_triatt:
        try:
            import opt_core
            ver = str(getattr(opt_core, "__version__", "?"))
            if want_trimul:
                from opt_core.mem.rowpair import trimul_fused as RF   # noqa: N812
            if want_triatt:
                from opt_core.mem.rowpair import triatt as _RA
                RA_core = _RA.attention_core
        except (ImportError, AttributeError) as e:
            RF = RA_core = None
            reason = "core_%s_lacks_row_kernels" % ver
            _note_once("tp_kernels", f"row-sharded fused kernels not in the installed core ({e}): triangle multiplication = the torch contraction on rows, "
                                     f"triangle attention = the run's kernel through the stock module (trimul={TP_TRIMUL_KERNELS!r} triatt_core={TP_TRIATT_CORE!r} requested)")
    _TPX.update(RF=RF, RA_core=RA_core, reason=reason,
                trimul_word=("rowpair_fpf_v4" if RF is not None else "rowpair_torch"),
                triatt_word=(TP_TRIATT_CORE if RA_core is not None else None))
    return _TPX


def emit_tp_kernel_lines() -> List[str]:
    """Print the core's LEVER lines of the row kernels this process constructed (F2.trimul_rows via trimul_fused.emit_line, F1.flash_triattn via
    triatt.emit_core_line: the core's one grammar, this package's tag; the core prints them) right after the rowpair LEVER line, and return
    their texts; nothing when the line ran the torch statements."""
    out: List[str] = []
    if not _TPX:
        return out
    from . import report as R
    tag = R.PREFIX.strip("[]")
    if _TPX.get("RF") is not None:
        out.append(_TPX["RF"].emit_line(tag))
    if _TPX.get("RA_core") is not None:
        from opt_core.mem.rowpair import triatt as _RA
        out.append(_RA.emit_core_line(tag))
    return out


def sync(t, name: str):
    """The ONE replicated-tensor sync point statement (opt_core.mem.rowpair.diffusion.sync_replicated in this process's mode): `bcast` -> rank
    0's tensor on every rank (det 0: replicated kernels are not bit-reproducible run-to-run, rank 0 is authoritative); `guard` -> proven bit-equal
    identical or RowpairRefused naming `name` (det 1: a mismatch is a real defect)."""
    return C("DF").sync_replicated(t.contiguous(), name, mode=SYNC["mode"])


def synced_denoiser(fn, state_kw: str = "r_noisy"):
    """A denoiser call whose sampler state IN (`state_kw`, the noisy coordinates) and update OUT (its return) both pass through `sync`: with
    the sampler's every other operation elementwise on synced tensors, EVERY rank's diffusion trajectory is identical by construction (det 0:
    rank 0's; det 1: proven), whatever the token count or step count — the sampler-exit rank spread reads 0.0. DiffusionModule.f_forward under
    n_gpu>1 is this wrapper over the row-sharded body (f_forward_tp)."""
    def call(*a, **kw):
        if state_kw in kw:
            kw[state_kw] = sync(kw[state_kw], "diffusion_state")
        else:
            a = (a[0], sync(a[1], "diffusion_state")) + tuple(a[2:])                       # (self, r_noisy, ...) positional form
        return sync(fn(*a, **kw), "diffusion_update")
    call.__wrapped__ = getattr(fn, "__wrapped__", fn)
    return call


class TPRefused(RuntimeError):
    """A condition the row-sharded path does not serve, refused by name (never a silent fallback to replicated)."""


def _refuse(what: str):
    raise TPRefused(f"refused: rowpair tp: {what}")


_NOTED: set = set()


def _note_once(key: str, text: str) -> None:
    """One `[protenix-v1-opt] NOTE …` line per process for a condition decided up front and named; the run proceeds."""
    if key in _NOTED:
        return
    _NOTED.add(key)
    from . import report as R
    R.log(R.note_line(text))


def _mark(stage: str) -> None:
    if CENSUS is not None:
        CENSUS.mark(stage)


class PairShard(object):
    """The pair representation under TP as it travels through the stock call sites in place of z: this rank's rows `t` [R, N, C] + the layout.
    A TRUNK shard also carries, from roll-out entry on, `plan` — the core's placement of the trunk shard across the confidence passes
    (opt_core.mem.rowpair.heads.ZTrunkPlan: parked on the host | embedded in place | resident, per pass) — and `contact`, this rank's
    distogram / contact rows taken from the device shard before any park (rollout_entry)."""
    __slots__ = ("t", "layout", "plan", "contact")

    def __init__(self, t, layout):
        self.t, self.layout, self.plan, self.contact = t, layout, None, None

    @property
    def shape(self):
        return tuple(self.t.shape)

    @property
    def dtype(self):
        return self.t.dtype

    def clone(self):
        return PairShard(self.t.clone(), self.layout)

    def rows_source(self):
        """What a row-block reader of the TRUNK shard reads right now: the live host park (`.zrows`, the device storage released) or the tensor."""
        return self.plan.source() if self.plan is not None else self.t

    def float_rows(self):
        """The stock's `z.to(torch.float32)` (autocasting_disable_decorator / f_forward upcasts) on the shard."""
        import torch
        return self.t if self.t.dtype == torch.float32 else self.t.to(torch.float32)


class RelpRows(object):
    """`input_feature_dict["relp"]` under TP: the per-token id vectors; `rows(g0, g1)` = rows [g0, g1) of the stock one-hot plane [.., N, N, 139]
    built by the statements of embedders.py:150-201 (`_relp.relp_rows_into`). The plane itself never exists."""
    __slots__ = ("rpe", "feats", "N", "width")

    def __init__(self, rpe, feats):
        self.rpe, self.feats = rpe, feats
        self.N = int(feats["asym_id"].shape[-1])
        self.width = RL.relp_width(rpe)

    def rows(self, g0: int, g1: int):
        import torch
        asym = self.feats["asym_id"]
        out = torch.empty(tuple(asym.shape[:-1]) + (int(g1) - int(g0), self.N, self.width), dtype=torch.float32, device=asym.device)
        RL.relp_rows_into(self.rpe, self.feats, int(g0), int(g1), out)
        return out


# =========================================================================================================== layout / helpers
def _core():
    from opt_core.mem.rowpair import dist as D, shard as SH, trunk as TK, pairstack as PS, trimul as TM, triatt as TA, transition as TR  # noqa: E401
    from opt_core.mem.rowpair import msa as MS, msa_host as MH, template as TE, heads as HD, confidence as CF, diffusion as DF, evidence as EV, bcast as BC  # noqa: E401
    return dict(D=D, SH=SH, TK=TK, PS=PS, TM=TM, TA=TA, TR=TR, MS=MS, MH=MH, TE=TE, HD=HD, CF=CF, DF=DF, EV=EV, BC=BC)


_C: Dict[str, object] = {}


def C(name: str):
    if not _C:
        _C.update(_core())
    return _C[name]


def choose_align(N: int, P: int, chunk: int) -> int:
    """Row boundaries are multiples of the pinned chunk (128) when every rank can own a chunk; else the largest power of two >= 4 with a*P <= N
    (tri-attention query blocks are multiples of 4); refused by name below that."""
    a = int(chunk)
    while a >= 4:
        if a * P <= N:
            return a
        a //= 2
    _refuse(f"N={N} tokens cannot be row-sharded over P={P} ranks (needs N >= {4 * P}): run --n_gpu 1")


def world() -> Tuple[int, int]:
    """(P, rank) of the calling rank from the core's group (thread-aware: threaded test ranks each see their own)."""
    P, rank = C("D").world()
    return int(P), int(rank)


def layout_for(N: int):
    D = C("D")
    P, rank = world()
    if P <= 1:
        _refuse("layout at n_gpu=1 (the row-sharded statements run only in a group of world > 1)")
    align = choose_align(N, P, int(STATE["chunk"]))
    os.environ[ALIGN_ENV] = str(align)
    lay = D.Layout.checked(N, P, rank, B=align, lever="rowpair", align=align)          # the refusing form (aligned policy): never a replicated layout
    D.require_sharded(lay, "protenix_v1 tp.layout_for")
    C("EV").record_schedule(tp_align=align, tp_layout=f"N={N},P={P},bounds={';'.join(f'{a}-{b}' for a, b in lay.bounds)}")
    STATE["layout"] = lay
    return lay


def _record_replicated(**named) -> None:
    for k, t in named.items():
        if t is not None and hasattr(t, "numel"):
            REPLICATED[k] = int(t.numel()) * int(t.element_size())
    C("EV").record_schedule(tp_replicated=",".join(f"{k}:{v}" for k, v in sorted(REPLICATED.items())))


def _record_park(rec) -> None:
    """A host park's record (opt_core.mem.rowpair.trunk.ShardPark.record(): where, gib, device_release, ...) into report.rowpair.parks; its
    schedule-census words `park_<name>=<where> park_<name>_gib= park_<name>_release=` are the core's (ShardPark.census_words in run_trunk_sharded)."""
    if not rec:
        return
    PARKS[str(rec.get("name") or "shard")] = dict(rec)


# =========================================================================================================== pair-block callables (S4)
def trimul_fns(mod):
    """TriMulFns of a protenix TriangleMultiplicativeUpdate (triangular/triangular.py forward: LN_in; a|b = mask * sigmoid(linear_*_g(x)) *
    linear_*_p(x); LN_out -> linear_z; gate sigmoid(linear_g(LN_in(z))))."""
    import torch
    TM = C("TM")

    def proj(z_blk, mask_blk, is_a):
        x = mod.layer_norm_in(z_blk)
        g, p = (mod.linear_a_g, mod.linear_a_p) if is_a else (mod.linear_b_g, mod.linear_b_p)
        out = mask_blk
        out = out * torch.sigmoid(g(x))
        out = out * p(x)
        return out.to(z_blk.dtype)                                                      # the ring streams operands in z's dtype (the schedule dtype): an autocast
                                                                                        # activation dtype never reaches the wire (equal byte counts on every rank)

    def out(x):
        return mod.linear_z(mod.layer_norm_out(x))

    def gate(z_blk):
        return torch.sigmoid(mod.linear_g(mod.layer_norm_in(z_blk)))

    fns = TM.TriMulFns(proj, out, gate, int(mod.linear_a_p.weight.shape[0]))
    RF = tp_kernels()["RF"]
    if RF is None:                                                                      # the default statements: the torch contraction on rows
        return fns
    return RF.fused_trimul_fns(trimul_weights(mod), fns, eps=float(getattr(mod.layer_norm_in, "eps", 1e-5)), cells=None, ledger=None)


def trimul_weights(mod) -> Dict[str, object]:
    """A protenix TriangleMultiplicativeUpdate's tensors in the core's TriMul vocabulary (opt_core.trimul.WEIGHT_KEYS — the mapping the fused
    providers take at n_gpu 1 and the fused row kernels take here); protenix's projections carry no bias."""
    return dict(ln_in_w=mod.layer_norm_in.weight, ln_in_b=mod.layer_norm_in.bias, w_ag=mod.linear_a_g.weight, w_ap=mod.linear_a_p.weight,
                w_bg=mod.linear_b_g.weight, w_bp=mod.linear_b_p.weight, ln_out_w=mod.layer_norm_out.weight, ln_out_b=mod.layer_norm_out.bias,
                w_o=mod.linear_z.weight, w_og=mod.linear_g.weight)


def _tb_operand(tb_full):
    """[N, N, H] gathered bias -> the stock `triangle_bias` operand [1, H, N, N] (permute_final_dims(.., (2, 0, 1)).unsqueeze(-4))."""
    return tb_full.permute(2, 0, 1).unsqueeze(0)


def triatt_fns(mod, kernel: str):
    """TriAttFns of a protenix TriangleAttention (starting form; the ending node is the same module on transposed rows): LN, bias = linear(x)
    [rows, N, H], attend = mod.mha(q_x=x_rows, kv_x=x_rows, biases=[inf*(mask_rows-1), tb_full]) with the run's kernel."""
    TA = C("TA")
    core = triatt_core(mod, kernel)

    def attend(x_rows, mask_rows, tb_full, _ii):
        m = mask_rows if mask_rows is not None else x_rows.new_ones(x_rows.shape[:-1])
        mask_bias = (mod.inf * (m - 1))[..., :, None, None, :]                        # [rows, 1, 1, N]
        if core is None or int(x_rows.shape[-2]) <= SMALL_Q:                            # the default statement; and the stock module's own small-input rule (layers.Attention.forward: Q <= 16 -> torch)
            return mod.mha(q_x=x_rows, kv_x=x_rows, biases=[mask_bias, _tb_operand(tb_full)], triangle_attention=kernel)
        return attend_rows_core(mod.mha, core, x_rows, mask_bias, _tb_operand(tb_full))

    return TA.TriAttFns(mod.layer_norm, mod.linear, attend)


SMALL_Q = 16                                                                            # protenix layers.Attention.forward: `if q.shape[-2] <= 16: triangle_attention = "torch"`


def stock_attention_core(mha, kernel: str):
    """The stock attention core of a protenix triangular layers.Attention as the core's `stock(q, k, v, biases) -> o` fallback: exactly
    layers.Attention.forward's own dispatch after `_prep_qkv` for the run's `--triatt_kernel` — "cuequivariance": the cuEquivariance kernel
    with the module's `scale = 1/sqrt(c_hidden)`, the fp32 triangle bias and the boolean key mask; "torch": q divided by sqrt(c_hidden) as
    `_prep_qkv(apply_scale=True)` does, then layers._attention — on the module's own 4-D shapes (the adapter's leading batch dim of 1 is
    stripped here; the result is reshaped to q's shape), so a declined unit runs the stock statements, at their memory. Convention:
    unscaled q in, scale=None -> the KERNEL applies D**-0.5, the stock callable scales itself."""
    import math
    from protenix.model.triangular import layers as LY
    root = math.sqrt(mha.c_hidden)
    # The result comes back in q's own shape whatever rank the kernel returns (cuEquivariance prepends singleton dims to a 4-D call and
    # returns 5-D; layers._attention keeps the rank) — the shape the core and the module's `_wrap_up` expect.
    if kernel == "cuequivariance":
        def stock(q, k, v, biases):
            return LY.cuequivariance_triangular_attn(q[0], k[0], v[0], biases[1][0].float(), (biases[0][0] == 0).bool(), 1.0 / root).reshape(q.shape)
    elif kernel == "torch":
        def stock(q, k, v, biases):
            return LY._attention(q[0] / root, k[0], v[0], [b[0] for b in biases]).reshape(q.shape)     # `q / math.sqrt(self.c_hidden)`: a division, as _prep_qkv states it
    else:
        raise C("D").RowpairRefused(f"triangle attention on rows: --triatt_kernel {kernel!r} has no row-block core here (torch | cuequivariance)")
    return stock


def triatt_core(mod, kernel: str):
    """The core's per-row-block attention core for this TriangleAttention when TP_TRIATT_CORE names one and the installed core carries it
    (tp_kernels), else None (the default: the whole stock module call). Core kernel = TP_TRIATT_CORE ("flash_triattn" | "cueq" |
    "torch"); its fallback = the module's own dispatch for the run's `--triatt_kernel` (stock_attention_core); q/k/v in the stock layout
    [B, rows, H, S, D] (`bnhsd`), the key mask read off biases[0] (the additive 0 / -inf mask bias), the triangle bias = biases[1]; scale
    None = 1/sqrt(D), the module's own."""
    make = tp_kernels()["RA_core"]
    if make is None:
        return None
    return make(stock_attention_core(mod.mha, kernel), kernel=str(TP_TRIATT_CORE), min_tokens=0, ledger=None, scale=None, layout="bnhsd", mask_from="bias0", tri_bias="bias1")


def attend_rows_core(mha, core, x_rows, mask_bias, tb):
    """The stock module's forward around a core: its own `_prep_qkv` (unscaled: the core applies the scale; a declined unit's stock callable
    scales itself), `core(q, k, v, biases)` over a leading batch dim of 1 ([1, rows, H, N, D]; the mask bias [1, rows, 1, 1, N]; the triangle
    bias [1, 1, H, N, N]), then its own `_wrap_up` (gating by q_x, head merge, output projection) — layers.Attention.forward's statements with
    the core in place of its kernel switch."""
    q, k, v = mha._prep_qkv(x_rows, x_rows, apply_scale=False)                         # [rows, H, N, D] each
    o = core(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), [mask_bias.unsqueeze(0), tb.unsqueeze(0)])
    o = o.squeeze(0).transpose(-2, -3)                                                 # [rows, N, H, D]
    return mha._wrap_up(o, x_rows)


def apb_fn(apb):
    """Pair-biased attention of the single representation for LOCAL query rows (AttentionPairBias with has_s=False, pairformer.py:218):
    transition.apb_local_queries all-gathers the rows."""
    from protenix.model.utils import permute_final_dims
    TR = C("TR")

    def attn_fn(s_q, s_all, z_rows):
        a_q = apb.layernorm_a(s_q)
        a_kv = apb.layernorm_a(s_all)
        bias = permute_final_dims(apb.linear_nobias_z(apb.layernorm_z(z_rows)), [2, 0, 1])   # [H, rows, N]
        while bias.dim() < a_q.dim() + 1:                                                       # bias rank = q rank ([*, H, q, N]): the stock Attention
            bias = bias.unsqueeze(0)                                                            # unsqueezes a lower-rank bias at the HEAD axis
        return apb.attention(q_x=a_q, kv_x=a_kv, attn_bias=bias)

    def apb_update(s, z_loc, lay):
        return s + TR.apb_local_queries(attn_fn, s, z_loc, lay, gather=True)
    return apb_update


def pair_block_fns(blk, kernel: str, chunk: int):
    """PairBlockFns of one protenix PairformerBlock (pairformer.py:103-224, the inplace inference branch; c_s > 0 adds the single track)."""
    PS = C("PS")
    has_s = bool(getattr(blk, "c_s", 0)) and blk.c_s > 0
    return PS.bind(trimul_out=trimul_fns(blk.tri_mul_out), trimul_in=trimul_fns(blk.tri_mul_in),
                   triatt_start=triatt_fns(blk.tri_att_start, kernel), triatt_end=triatt_fns(blk.tri_att_end, kernel),
                   transition=lambda x_rows, _mask_u: blk.pair_transition(x_rows), chunk=int(chunk),
                   apb=apb_fn(blk.attention_pair_bias) if has_s else None,
                   single_transition=(lambda s: s + blk.single_transition(s)) if has_s else None,
                   trimul_kw={"inplace_chunk": TRIMUL_INPLACE_CHUNK})


def pair_stack_tp(stack, s, z_loc, lay, kernel: str, chunk: int, what: str):
    """`PairformerStack.forward` on the PRE-SHARDED rows (the ONE core driver; never gathers). s replicated or None. Returns (s, z_loc)."""
    PS = C("PS")
    blocks = [pair_block_fns(b, kernel, chunk) for b in stack.blocks]
    COUNTS["pairstack_calls"] += 1
    COUNTS["pair_blocks"] += len(blocks)
    z_loc, s = PS.pair_stack_(blocks, z_loc, None, lay, s=s, transition_mask=False)
    return s, z_loc


# =========================================================================================================== trunk (S2 / S3 / S5)
CONSTRAINT_KINDS = ("pocket", "contact", "contact_atom", "substructure")


def constraint_embedders_enabled(source) -> List[str]:
    """The enabled sub-embedders of protenix's ConstraintEmbedder (embedders.py:335-437: each kind embeds only when its config says
    `enable`; all off in the stock model configs, when `forward` returns None and the trunk adds nothing). `source` is the module (its
    `<kind>_embedder_config` dicts) or the run's configs (`model.constraint_embedder.<kind>_embedder.enable`); [] when neither carries one."""
    out = []
    for kind in CONSTRAINT_KINDS:
        cfg = None
        if source is not None and hasattr(source, f"{kind}_embedder_config"):
            cfg = getattr(source, f"{kind}_embedder_config")
        elif source is not None:
            try:
                cfg = source.model.constraint_embedder[f"{kind}_embedder"]
            except (AttributeError, KeyError, TypeError):
                cfg = None
        try:
            on = bool(cfg.get("enable", False)) if cfg is not None else False
        except AttributeError:
            on = bool(getattr(cfg, "enable", False))
        if on:
            out.append(kind)
    return out


def _check_trunk_inputs(model, feats, s_inputs, mc_dropout: bool) -> None:
    import torch
    if model.training:
        _refuse("training mode (inference only)")
    enabled = constraint_embedders_enabled(getattr(model, "constraint_embedder", None))
    if enabled and "constraint_feature" in feats:                                     # the stock adds constraint_embedder(...) [N, N, c_z] only when a sub-embedder is enabled
        _refuse(f"tp_constraint_unsupported: constraint embedders {','.join(enabled)} enabled with constraint_feature present (no row form of "
                "ConstraintEmbedder under n_gpu>1; run --n_gpu 1)")
    if s_inputs.dim() != 2:                                                              # structural: the row-sharded domain is one item per fold call
        _refuse(f"s_inputs {tuple(s_inputs.shape)}: one item per fold call under n_gpu>1 (a leading batch dim is outside the row-sharded domain); "
                "run the items separately or --n_gpu 1")
    if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.float16:
        _note_once("fp16", "fp16 autocast under n_gpu>1: the line's numerics and memory statements were established on bf16 / fp32 only; proceeding in fp16")
    if not isinstance(feats.get("relp"), RelpRows):
        _refuse("input_feature_dict['relp'] is not RelpRows: the TP generate_relp patch is not installed (install order)")


def msa_m_layout(environ=None) -> str:
    env = os.environ if environ is None else environ
    v = (env.get(MSA_M_ENV) or "token_sharded").strip()
    if v not in MSA_M_LAYOUTS:
        _refuse(f"{MSA_M_ENV}={v!r}: one of {'|'.join(MSA_M_LAYOUTS)}")
    return v


def mc_dropout_full_max(environ=None) -> int:
    """`PROTENIX_V1_TP_MC_DROPOUT_FULL_MAX` (default 2048; a non-negative integer, else refused by name)."""
    env = os.environ if environ is None else environ
    v = (env.get(MC_DROPOUT_FULL_MAX_ENV) or "").strip()
    if not v:
        return MC_DROPOUT_FULL_MAX_DEFAULT
    try:
        n = int(v)
    except ValueError:
        n = -1
    if n < 0:
        _refuse(f"{MC_DROPOUT_FULL_MAX_ENV}={v!r}: a non-negative token count (N <= it: stock_mask; above: rank_streams; 0: rank_streams at every N)")
    return n


def mc_dropout_form(N: int, environ=None) -> str:
    """The n_gpu>1 MC-dropout statement of an item of N tokens: `stock_mask` when N <= the ceiling, else `rank_streams`."""
    return "stock_mask" if int(N) <= mc_dropout_full_max(environ) else "rank_streams"


def make_mc_dropout_rows(lay, p: float, device, form: str):
    """protenix.py:240-245 under n_gpu>1 — the MC-dropout statement `F.dropout(linear(LN(z)) [N, N, c_z], p)` of the recycling projection on
    this rank's rows. Returns `drop(upd_rows [.., g1-g0, N, c_z]) -> rows`, called on the CONSECUTIVE row blocks `trunk.recycle_shard_` visits
    (rows r0 .. r1 of the layout in order, every block once per cycle).

    stock_mask    the stock statement's own draw: `F.dropout(x, p, training=True)` is ONE fused Philox dropout over the whole plane from the GLOBAL
                  generator whose keep-mask and generator advance depend on (numel, dtype, device), never on the values. At the cycle's first
                  row block every rank makes exactly that draw (`torch.native_dropout` — the kernel F.dropout takes — on an uninitialised plane of
                  the projection's shape and dtype), keeps the bool mask [N, N, c_z] for the cycle, applies rows [g0, g1) of it with the stock
                  arithmetic (fp32: x · mask · 1/(1−p) -> dtype) and drops it after the rank's last block. The global generator is left where the
                  stock statement leaves it on EVERY rank (the trunk guard's premise), so every later replicated draw of the process — this item's
                  MSA sample and diffusion noise and every later item's (protenix seeds once per seed and carries the generators across items) —
                  is the stock's. Transient memory at the draw N²·c_z·(2+2+1) B (plane + the kernel's output + mask), N²·c_z·1 B held through the
                  cycle: 1,340 tok 1.1 GiB / 0.23 GiB; 2,048 tok 2.7 / 0.5 GiB (4,096 would be 10.7 / 2.1). No collective.
    rank_streams  (above PROTENIX_V1_TP_MC_DROPOUT_FULL_MAX) this rank's rows draw i.i.d. keep-masks from a
                  rank-forked private generator seeded from (initial_seed, rank) WITHOUT consuming the global generator: a valid MC-dropout
                  realisation, but not the stock's mask, and the global Philox offset falls behind the stock's by the dropout's draw from this
                  item on — every later CUDA draw of the process is an independent realisation of the seed (tier-2 by construction)."""
    import torch
    p = float(p)
    scale = 1.0 / (1.0 - p)
    if form == "stock_mask":
        cur = {"g": int(lay.r0), "mask": None}

        def drop_stock(upd):
            if cur["mask"] is None:                                                          # the cycle's first row block on this rank: the stock statement's ONE draw over the whole plane
                plane = torch.empty((int(lay.N), int(lay.N), int(upd.shape[-1])), dtype=upd.dtype, device=upd.device)
                _out, cur["mask"] = torch.native_dropout(plane, p, True)                     # (values irrelevant: the mask and the generator advance depend on numel / dtype / device only)
                del _out, plane
            g0 = int(cur["g"]); g1 = g0 + int(upd.shape[-3])
            out = (upd.float() * cur["mask"][g0:g1].to(torch.float32) * scale).to(upd.dtype)
            if g1 >= int(lay.r1):                                                            # the rank's last block of the cycle: the next call is the next cycle's first
                cur["g"], cur["mask"] = int(lay.r0), None
            else:
                cur["g"] = g1
            return out
        return drop_stock
    if form != "rank_streams":
        _refuse(f"mc dropout form {form!r}: one of {'|'.join(MC_DROPOUT_FORMS)}")
    gen = torch.Generator(device=device)                                                     # a rank-forked stream: this rank's rows draw i.i.d. keep-masks WITHOUT consuming the
    gen.manual_seed((int(torch.initial_seed()) * 1000003 + 7919 * (int(lay.rank) + 1)) % (2 ** 63))   # global generator

    def drop_streams(upd):                                                                   # F.dropout(upd, p): keep ~ Bernoulli(1-p) per element, kept values scaled by 1/(1-p)
        keep = torch.rand(upd.shape, generator=gen, device=upd.device, dtype=torch.float32) >= p
        return upd * keep.to(upd.dtype) * scale
    return drop_streams


def msa_module_tp(msa, feats, z_loc, s_inputs, lay, kernel: str, chunk: int):
    """MSAModule.forward (pairformer.py:809-917) with z as rows. The MSA representation m is TOKEN-SHARDED by default (`PROTENIX_V1_TP_MSA_M=
    token_sharded`: rank q embeds and holds m[:, r0:r1, :] — the embedding, the OPM `a` operand, the pair-weighted-averaging update and the
    transition are per-token statements on the shard; the OPM `b` operand and the averaging's values source are the only all-gathers) or
    REPLICATED by name (`=replicated`: every rank holds m[S, N, c_m]; census msa_m=replicated). The per-cycle subsample is the stock's own
    statement (utils.py:314 sample_msa_feature_dict_random_without_replacement: the size from the default generator, the permutation on the
    features' device) on every rank from generator states proven identical before the call (trunk.guard_rng_replicated), made rank 0's / proven
    identical by the run's sync policy before anything is sliced. Where the raw features live decides what the statement indexes: on the
    DEVICE (ROWPAIR_MSA_HOST unset: the sampled [k, N] features, as the stock), or on pinned HOST (host_side_inputs parked them; the line exports
    ROWPAIR_MSA_HOST=rank0): the statement runs on an [S_msa, 1] index stand-in on the compute device — the same two generator draws — and the
    drawn row indices are applied to the host copies (msa_host.HostTensor.rows_to: only the k selected rows reach the device); in mode rank0
    rank 0 gathers and every other rank receives the rows by a chunked device broadcast (msa_host.rows_to_device), in mode all every rank
    gathers its own."""
    import torch
    from protenix.model.modules.pairformer import sample_msa_feature_dict_random_without_replacement
    MS, MH, PS, TK, SH, EV = C("MS"), C("MH"), C("PS"), C("TK"), C("SH"), C("EV")
    if msa.n_blocks < 1:
        return z_loc
    N = int(z_loc.shape[-2])
    hm = None
    md = MH.host_mode()
    if md is not None:                                                                  # ROWPAIR_MSA_HOST: this adapter moved the raw MSA features out of the item at entry — their
        host = host_inputs()                                                            # absence from `feats` is the normal state, so only the entry's explicit record decides
        hm = host.get("msa")
        if hm is None or host.get("N") != N:
            _refuse(f"msa_host {md}: no host-side MSA record for this item on rank {lay.rank} (host_side_inputs N={host.get('N')} vs N={N}) — "
                    "the raw MSA features left the item at entry; running the MSA module without them would silently drop it")
        if hm.get("absent"):                                                            # the item carries no MSA features (stated at entry): pairformer.py:846-851, z unchanged
            EV.record_schedule(msa_raw=f"absent:{md}")
            return z_loc
    elif "msa" not in feats or feats["msa"].dim() < 2:                                  # the stock placement: pairformer.py:846-851, no MSA features -> z unchanged
        return z_loc
    # ---- embedding (pairformer.py:846-898; inference branch)
    layout_word = msa_m_layout()
    tok = layout_word == "token_sharded"
    r0, r1 = (lay.r0, lay.r1) if tok else (0, N)
    sample_kw = dict(cutoff=msa.msa_configs["test_cutoff"], lower_bound=msa.msa_configs["test_lowerb"], strategy=msa.msa_configs["strategy"])
    if hm is None:                                                                      # features on the device: the stock statement on them, the sampled features synced
        msa_feat = sample_msa_feature_dict_random_without_replacement(feat_dict=feats, dim_dict={feat_name: -2 for feat_name in msa.input_feature}, **sample_kw)
        for name in sorted(msa.input_feature):                                          # the draw: rank 0's sampled MSA features on every rank (det 0) / proven identical (det 1) — before any slicing
            if name in msa_feat and hasattr(msa_feat[name], "dim"):
                msa_feat[name] = sync(msa_feat[name], f"msa.sample.{name}")
        _record_replicated(msa_raw=feats["msa"])
        EV.record_schedule(msa_raw="device")
    else:                                                                               # features on the host (ROWPAIR_MSA_HOST): the stock statement draws the row indices on the device,
        dev = s_inputs.device                                                           # the host copies serve the selected rows only
        S_msa = int(hm["n_rows"])
        stand_in = {"msa": torch.arange(S_msa, device=dev).unsqueeze(-1)}              # [S_msa, 1]: dim -2 = the MSA rows, .device = the compute device (utils.py:305-307 draw there)
        idx = sample_msa_feature_dict_random_without_replacement(feat_dict=stand_in, dim_dict={"msa": -2}, **sample_kw)["msa"].reshape(-1)
        idx = sync(idx, "msa.sample.index")                                             # rank 0's drawn indices on every rank (det 0) / proven identical (det 1)
        k_sel = int(idx.numel())
        msa_feat = {}
        for name in msa.input_feature:                                                  # the stock's key order: msa, has_deletion, deletion_value
            shape, dtype_word = hm["meta"][name]
            ht = hm["parked"].get(name)                                               # HostTensor (rank 0; every rank in mode all) | None (ranks > 0 in mode rank0: the rows arrive by broadcast)
            if ht is None and not (hm["mode"] == "rank0" and lay.rank != 0):
                _refuse(f"msa_host {hm['mode']}: rank {lay.rank} holds no host copy of {name!r}")
            rows = MH.rows_to_device((lambda _ht=ht: _ht.rows_to(idx, dev)), shape=tuple(shape[:-2]) + (k_sel, int(shape[-1])),
                                     dtype=getattr(torch, dtype_word), device=dev, mode=hm["mode"], name=f"msa_rows.{name}")
            msa_feat[name] = rows
            COUNTS["msa_host_calls"] += 1                                               # one rows_to_device call per feature per cycle (k_sel rows each: census msa_sample_rows)
        REPLICATED["msa_raw"] = f"host:{sum(h.nbytes for h in hm['parked'].values())}" if hm["parked"] else "rank0"
        EV.record_schedule(msa_raw=f"host:{hm['mode']}", msa_sample_rows=k_sel)
    if tok:                                                                             # a per-token statement: this rank's token columns only ([S, R, *] — the [S, N, 32] one-hot never exists)
        msa_feat = {name: (t[..., r0:r1] if t.dim() == 2 else t[..., r0:r1, :]) if name in msa.input_feature else t for name, t in msa_feat.items()}
    if N > 2000:
        msa_feat["msa"] = msa.one_hot_fp32(msa_feat["msa"], num_classes=msa.input_feature["msa"])
    else:
        msa_feat["msa"] = torch.nn.functional.one_hot(msa_feat["msa"], num_classes=msa.input_feature["msa"])
    target_shape = msa_feat["msa"].shape[:-1]
    msa_sample = torch.cat([msa_feat[name].reshape(*target_shape, d) for name, d in msa.input_feature.items()], dim=-1)
    del msa_feat
    m = msa.linear_no_bias_m(msa_sample)
    del msa_sample
    m = m + msa.linear_no_bias_s(s_inputs[r0:r1] if tok else s_inputs)                # [S, R|N, c_m]
    if tok:
        REPLICATED.pop("msa_m", None)
        EV.record_schedule(msa_m="token_sharded", msa_m_shard_bytes=int(m.numel()) * m.element_size())
    else:
        _record_replicated(msa_m=m)
        EV.record_schedule(msa_m="replicated")
    S_ = int(m.shape[-3])
    for i, blk in enumerate(msa.blocks):
        # ---- OPM rows into the shard in place (pairformer.py:660; triangular/layers.py:724-765)
        opm = blk.outer_product_mean_msa
        ln = opm.layer_norm(m)
        a = opm.linear_1(ln)                                                            # mask = ones at inference (stock: mask None)   [S, R|N, C]
        b = opm.linear_2(ln)
        del ln
        a = a.transpose(-2, -3)                                                         # [R|N, S, C]
        b = b.transpose(-2, -3)
        if tok:
            b = SH.unshard_rows(b.contiguous(), lay, dim=0).contiguous()               # the OPM's non-local operand: every token's b (ONE all_gather; [N, S, C] transient, named)
            EV.record_schedule(opm_b_gather_bytes=int(b.numel()) * b.element_size())

        def outer_fn(a_blk, b_all, _opm=opm, _S=S_):
            outer = torch.einsum("...bac,...dae->...bdce", a_blk, b_all)               # [rows, N, C, C]
            outer = outer.reshape(outer.shape[:-2] + (-1,))
            outer = _opm.linear_out(outer)
            return outer / (_opm.eps + float(_S))                                       # norm = einsum of all-ones masks == S (stock: outer / (eps + norm))
        MS.opm_rows_budgeted(a, b, lay, outer_fn, rows=None, C_z=int(z_loc.shape[-1]), out=z_loc, add=True,
                             align=lay.align if getattr(lay, "align", None) else None, row_dim=0, a_local=tok)
        del a, b
        COUNTS["opm_rows"] += 1
        if not blk.is_last_block:
            # ---- pair-weighted averaging on LOCAL rows + transition (pairformer.py:547-573 chunk loop, :385-422 statements)
            st = blk.msa_stack
            pwa = st.msa_pair_weighted_averaging
            H, c = int(pwa.n_heads), int(pwa.c)
            bias_loc = MS.pwa_bias_rows(lambda z_rows, g0, g1: pwa.linear_no_bias_z(pwa.layernorm_z(z_rows)).permute(2, 0, 1), z_loc, lay)   # [H, R, N]

            def values_fn(m_chunk):                                                     # m_chunk: every token of this S-chunk ([chunk, N, c_m]; gathered by the core when m is token-sharded)
                m_ln = pwa.layernorm_m(m_chunk)
                v = pwa.linear_no_bias_mv(m_ln).reshape(*m_ln.shape[:-1], H, c)        # [chunk, N, H, c]
                g = torch.sigmoid(pwa.linear_no_bias_mg(m_ln)).reshape(*m_ln.shape[:-1], H, c)
                return (v, g)

            def attend_fn(w, state, g0, g1):
                v, g = state
                wv = torch.einsum("hij,mjhc->mihc", w, v)                               # stock '...ijh,...mjhc->...mihc' with w heads-first
                o = g[:, g0:g1] * wv
                return o.reshape(*o.shape[:-2], H * c)                                  # [chunk, q, H*c]

            softmax_fn = lambda x: torch.softmax(x, dim=-1)                            # stock nn.Softmax(dim=-2) on [i, j, h] == over j  # noqa: E731
            chunk_size = st.msa_chunk_size or S_
            for s0 in range(0, S_, chunk_size):
                s1 = min(S_, s0 + chunk_size)
                upd = MS.pwa_rows(m[s0:s1], bias_loc, lay, values_fn=values_fn, attend_fn=attend_fn, out_fn=pwa.linear_no_bias_out,
                                  softmax_fn=softmax_fn, s_chunk=None, m_layout=layout_word)
                m[s0:s1] += upd
                del upd
                m[s0:s1] += MS.msa_transition_rows(lambda mm, t0, t1: st.transition_m(mm), m[s0:s1], lay, shard_tokens=tok, token_dim=-2, m_layout=layout_word)
            del bias_loc
            COUNTS["pwa_calls"] += 1
        # ---- pair block on rows (pairformer.py:667)
        fns = pair_block_fns(blk.pair_stack, kernel, chunk)
        z_loc, _ = PS.pair_block_(fns, z_loc, None, lay, s=None, transition_mask=False, census=(i == 0))
        COUNTS["msa_blocks_rows"] += 1
        COUNTS["pair_blocks"] += 1
        if N > 2000:
            torch.cuda.empty_cache() if z_loc.is_cuda else None
    del m
    return z_loc


def template_tp(te, feats, z_loc, lay, kernel: str, chunk: int):
    """TemplateEmbedder.forward (pairformer.py:982-1098) with z as rows; 0 when the run carries no templates (stock: returns 0)."""
    import torch
    import torch.nn.functional as F
    from protenix.model.modules.pairformer import STD_RESIDUES_WITH_GAP  # noqa: F401
    TE, TK = C("TE"), C("TK")
    if te.n_blocks < 1 or "template_aatype" not in feats:
        return z_loc
    n_templ = int(feats["template_aatype"].shape[0])
    N = lay.N
    r0 = lay.r0
    all_keys = ("template_distogram", "template_pseudo_beta_mask", "template_unit_vector", "template_backbone_frame_mask")
    host = host_inputs()
    if host.get("N") == N and "template_rows" in host:                                 # the rows sliced on the host at item entry (host_side_inputs)
        rows = host["template_rows"]
        pair_keys = tuple(rows.keys())
    else:                                                                               # features still whole in the dict (a caller without the entry hook)
        pair_keys = tuple(k for k in all_keys if feats[k].dim() >= 3)                  # [T,N,N(,F)] pair features; a [T,N] mask broadcasts along rows as in the stock statement
        if "_tp_template_rows" not in feats:                                           # ONCE per item: dense [T,N,N,*] -> this rank's rows [T,R,N,*]
            dense = {k: (feats[k] if feats[k].dim() == 4 else feats[k].unsqueeze(-1)) for k in pair_keys}   # the core slices [T, N, N, F]: masks get F=1
            sl = TE.slice_template_inputs_to_rows(dense, lay, pair_keys)
            feats["_tp_template_rows"] = {k: (sl[k] if feats[k].dim() == 4 else sl[k].squeeze(-1)) for k in pair_keys}
            del dense, sl
            COUNTS["template_rows"] += 1
        rows = feats["_tp_template_rows"]
    asym = feats["asym_id"]
    aatype_oh = F.one_hot(feats["template_aatype"], num_classes=len(STD_RESIDUES_WITH_GAP))   # [T, N, 32] per-token (replicated by design)

    def feat_rows(key, slot, b0, b1, device):
        if key in pair_keys:
            return rows[key][slot][b0:b1].to(device, non_blocking=False)                # [rows, N(,F)] of this rank's rows (host-resident rows move per block)
        return feats[key][slot].to(device)                                              # [N]: a 1-D mask broadcasts over rows, the stock's own statement

    def unit_rows_fn(z_rows, slot, gg):
        g0, g1 = gg
        b0, b1 = g0 - r0, g1 - r0
        mc = (asym[g0:g1, None] == asym[None, :]).to(z_rows.dtype)                     # multichain_mask rows (pairformer.py:1012); pair_mask = ones (:1020)
        pm = z_rows.new_ones(mc.shape)
        dgram = feat_rows("template_distogram", slot, b0, b1, z_rows.device) * mc[..., None] * pm[..., None]
        pbm = feat_rows("template_pseudo_beta_mask", slot, b0, b1, z_rows.device) * mc * pm
        uv = feat_rows("template_unit_vector", slot, b0, b1, z_rows.device) * mc[..., None] * pm[..., None]
        bbm = feat_rows("template_backbone_frame_mask", slot, b0, b1, z_rows.device) * mc * pm
        aa = aatype_oh[slot]
        at = torch.concat([dgram, pbm.unsqueeze(-1), aa[None, :, :].expand(g1 - g0, -1, -1), aa[g0:g1, None, :].expand(-1, N, -1), uv,
                           bbm.unsqueeze(-1)], dim=-1)
        v = te.linear_no_bias_z(te.layernorm_z(z_rows)) + te.linear_no_bias_a(at)      # [rows, N, c]  (pairformer.py:1020, :1085)
        return v.unsqueeze(0)                                                           # [1, rows, N, c_t]

    def pair_stack_fn(u, _mask_loc):
        u2 = u[0]
        _, u2 = pair_stack_tp(te.pairformer_stack, None, u2.contiguous(), lay, kernel, chunk, "template")
        return te.layernorm_v(u2).unsqueeze(0)

    def finish_fn(t):                                                                   # t [T, rows, N, c_t] in slot order (pairformer.py:1022-1032)
        u = 0
        for k in range(int(t.shape[0])):                                                # u = u + v_t, sequentially in slot order (the stock's sum order)
            u = u + t[k]
        u = u / (1e-7 + n_templ)
        return te.linear_no_bias_u(te.relu(u))

    COUNTS["template_calls"] += 1
    return TE.template_embed_rows(z_loc, lay, n_templ=n_templ, c_t=int(te.c), unit_rows_fn=unit_rows_fn, pair_stack_fn=pair_stack_fn,
                                 finish_fn=finish_fn, mask_loc=None, slot_groups=None, add=True)


def get_pairformer_output_tp(self, input_feature_dict, N_cycle, inplace_safe=False, chunk_size=None, mc_dropout=False, mc_dropout_rate=0.4):
    """Protenix.get_pairformer_output (protenix.py:170-303) with z row-sharded from birth: returns (s_inputs, s, PairShard)."""
    import torch
    TK, EV = C("TK"), C("EV")
    feats = input_feature_dict
    s_inputs = self.input_embedder(feats, inplace_safe=False, chunk_size=chunk_size)  # [N, 449] replicated
    _check_trunk_inputs(self, feats, s_inputs, mc_dropout)
    _bind_model(self, chunk_size)
    params = getattr(self, "parameters", None)
    if not STATE.get("weights_guarded") and callable(params):                          # once per process: the weights are identical on every rank, PROVEN (one all_reduce over
        with torch.no_grad():                                                           # a parameter checksum) — a per-rank weight desync refuses by name here, never folds
            checksum = torch.stack([p.detach().float().sum() for p in params()]).sum().reshape(1)
        TK.guard_replicated(checksum, name="tp_weights_checksum")
        STATE["weights_guarded"] = True
        COUNTS["weights_guard"] += 1
    COUNTS["items"] += 1
    N = int(s_inputs.shape[-2])
    lay = layout_for(N)
    kernel, chunk = str(STATE["kernel"]), int(STATE["chunk"])
    _mark("trunk_entry")
    s_init = self.linear_no_bias_sinit(s_inputs)                                        # [N, 384]
    zi1 = self.linear_no_bias_zinit1(s_init)                                            # row term  [N, 128]
    zi2 = self.linear_no_bias_zinit2(s_init)                                            # column term
    relp: RelpRows = feats["relp"]
    host = host_inputs()
    bonds = host["token_bonds"] if host.get("N") == N and "token_bonds" in host else feats["token_bonds"]   # host-resident [N, N] (host_side_inputs); rows move per block
    _record_replicated(s_inputs=s_inputs)
    REPLICATED["token_bonds"] = f"host:{int(bonds.numel()) * bonds.element_size()}" if bonds.device.type == "cpu" else int(bonds.numel()) * bonds.element_size()

    def init_rows_fn(g0, g1):                                                           # protenix.py:209-228 on rows [g0, g1)
        z = TK.outer_sum_rows(zi1, zi2, g0, g1)
        z = z + self.relative_position_encoding(relp.rows(g0, g1))
        tb = TK.feature_rows(bonds, g0, g1, device=zi1.device, row_dim=-2, non_blocking=False).to(dtype=z.dtype)
        z = z + self.linear_no_bias_token_bond(tb.unsqueeze(-1))
        return z

    mc = bool(sync(torch.tensor([int(bool(mc_dropout))], dtype=torch.int64, device=zi1.device), "mc_dropout_decision").item())   # the stock's `random.random() < rate` draw:
    mc_rows = None                                                                      # rank 0's decision on every rank (det 0) / proven identical (det 1)
    if mc:                                                                              # protenix.py:240-245: z = z_init + F.dropout(proj(z), p) at inference (MC dropout, p = mc_dropout_rate)
        p_drop = float(self.configs.mc_dropout_rate)
        COUNTS["mc_dropout_items"] += 1
        form = mc_dropout_form(int(lay.N))                                              # stock_mask at N <= PROTENIX_V1_TP_MC_DROPOUT_FULL_MAX (default 2048), rank_streams above
        mc_rows = make_mc_dropout_rows(lay, p_drop, zi1.device, form)
        EV.record_schedule(tp_mc_dropout=f"{form}(p={p_drop})")
        if form == "rank_streams":
            _note_once("tp_mc_dropout_rank_streams",
                       f"tp_mc_dropout=rank_streams(p={p_drop}) at N={int(lay.N)} > {MC_DROPOUT_FULL_MAX_ENV}={mc_dropout_full_max()}: tier-2 by construction (rank_streams) "
                       "- the rows' keep-mask comes from rank-forked generators and the stock statement's global draw is not consumed, so every later draw of this "
                       "process (MSA sample, diffusion noise; this item and the following ones) is an independent realisation of the seed, not the stock's")

    def recycle_update_fn(z_rows):                                                      # protenix.py:240-246
        upd = self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z_rows))
        return mc_rows(upd) if mc_rows is not None else upd

    def single_recycle_fn(s, _cycle):                                                   # protenix.py:283
        return s_init + self.linear_no_bias_s(self.layernorm_s(s))

    def template_fn(z_loc, _cycle):                                                     # protenix.py:249-256
        out = template_tp(self.template_embedder, feats, z_loc, lay, kernel, chunk)
        _mark("after_template")
        return out

    def msa_fn(z_loc, _cycle):                                                          # protenix.py:257-277
        out = msa_module_tp(self.msa_module, feats, z_loc, s_inputs, lay, kernel, chunk)
        _mark("after_msa")
        return out

    def pairstack_fn(s, z_loc, _cycle):                                                 # protenix.py:288-295
        s, z_loc = pair_stack_tp(self.pairformer_stack, s, z_loc, lay, kernel, chunk, "trunk")
        _mark("after_pairstack")
        COUNTS["recycles"] += 1
        return s, z_loc

    COUNTS["trunk_inits"] += 1
    out = TK.run_trunk_sharded(lay, n_cycles=int(N_cycle), init_rows_fn=init_rows_fn, init_like=zi1, recycle_update_fn=recycle_update_fn,
                               s_init=s_init, single_recycle_fn=single_recycle_fn, template_fn=template_fn, msa_fn=msa_fn,
                               pairstack_fn=pairstack_fn, s_input=s_inputs, init_channels=3 * int(zi1.shape[-1]) + relp.width, gather="none")
    s, z_loc = out.s, out.z
    _record_park(out.record.get("park"))                                                # the z_init shard's placement through the trunk (ROWPAIR_PARK_ZINIT: the line exports recompute, ngpu.TP_EXPORTS)
    _mark("no_gather")
    feats.pop("_tp_template_rows", None)
    zsh = PairShard(z_loc, lay)
    return s_inputs, s, zsh


# =========================================================================================================== the trunk shard after the trunk (S6-S8)
def n_conf_passes() -> int:
    """The confidence stage's passes = the roll-out's samples (configs.sample_diffusion.N_sample: main_inference_loop rolls every sample out, then
    the confidence head embeds the trunk shard once PER SAMPLE — confidence.py:221 `z_trunk.clone()` per sample is one embed_rows pass here)."""
    model = STATE.get("model")
    try:
        return max(1, int(model.configs.sample_diffusion["N_sample"]))
    except (AttributeError, KeyError, TypeError):
        return 1


def conf_plan(zsh: PairShard, passes: int):
    """The ONE placement plan of this item's trunk shard across the confidence passes (heads.ZTrunkPlan; ROWPAIR_FREE_ZTRUNK /
    ROWPAIR_CONF_PARK_ZTRUNK — the line exports both =1: parked on pinned host from roll-out entry to the last pass, the device storage
    released; in place at a single same-dtype last-use pass when the park is off; resident only with both off). Built once per shard."""
    if zsh.plan is None:
        zsh.plan = C("HD").ZTrunkPlan(zsh.t, passes=int(passes), name="z_trunk", log=_say_conf)
    elif zsh.plan.passes != int(passes):
        _refuse(f"confidence: the trunk shard's plan has {zsh.plan.passes} passes, the stage runs {passes} (N_sample changed between roll-out entry and the confidence head)")
    return zsh.plan


def _say_conf(msg: str) -> None:
    """The plan's park / restore lines on this rank's stderr under ROWPAIR_VERBOSE (the core group's logger: `[rowpair r<rank>] ...`)."""
    C("D").comm().log(msg)


def rollout_entry(zsh: PairShard) -> str:
    """At ROLL-OUT ENTRY (DiffusionConditioning.prepare_cache, the trunk shard's first reader after the trunk; protenix.py:527): (1) this rank's
    distogram / contact rows are taken from the DEVICE shard now (distogram_tp — the stock computes them after the roll-out, protenix.py:576; the
    same statements on the same z: heads.sym_logit_rows exchanges column slabs of the device shard and refuses a parked source by name), (2) the
    plan is built for N_sample passes and (3) `plan.park_now()`: under ROWPAIR_CONF_PARK_ZTRUNK=1 the shard is copied to pinned host and its
    device storage RELEASED before the conditioned pair rows are allocated — the roll-out, the distogram and every confidence pass then run with
    ONE pair-shard-equivalent less on the device; the conditioning and the confidence passes read the trunk rows through `plan.source()`.
    Returns the entry word (census conf_ztrunk_entry: parked:host_pinned | resident[:<reason>])."""
    model = STATE.get("model")
    if model is not None and getattr(model, "distogram_head", None) is not None and zsh.contact is None:
        zsh.contact = distogram_tp(model.distogram_head, zsh)
        STATE["contact_rows"][world()[1]] = zsh.contact
    plan = conf_plan(zsh, n_conf_passes())
    return plan.park_now()


# =========================================================================================================== distogram / contact rows (S6)
class ContactRows(object):
    """`pred_dict["contact_probs"]` under TP: this rank's rows [R, N] fp32 of the stock contact probabilities + the layout; the full [N, N]
    plane reaches rank 0 only inside the summary assembly (confidence.RowBlockReducer collects it)."""
    __slots__ = ("rows", "layout")

    def __init__(self, rows, layout):
        self.rows, self.layout = rows, layout


def distogram_tp(head, zsh: PairShard):
    """DistogramHead.forward (head.py:44-56: logits = linear(z); logits + logits^T) -> compute_contact_prob, per row block: heads.sym_logit_rows
    exchanges the column slabs so logits^T rows arrive without a transposed copy; contact probabilities in fp32 with autocast off (the stock's
    autocasting_disable_decorator(True) around compute_contact_prob). Returns ContactRows."""
    import torch
    from protenix.model import sample_confidence as SC
    HD = C("HD")
    lay, z_loc = zsh.layout, zsh.t
    model = STATE.get("model")
    bins = SC.get_bin_params(model.configs.loss.distogram) if model is not None else {"min_bin": 2.3125, "max_bin": 21.6875, "no_bins": 64}
    R, N = lay.R, lay.N
    out = torch.empty((R, N), dtype=torch.float32, device=z_loc.device)
    dev_type = "cuda" if z_loc.is_cuda else "cpu"
    logits_fn = head_rows_fn(head.linear, z_loc)                                        # per row block in the n_gpu 1 path's arithmetic: under autocast of the shard's dtype for a
    C("EV").record_schedule(disto_gemm=logits_fn.word)                                  # bf16|fp16 shard (the stock's ambient regime at protenix.py:576), plain for fp32 — never a whole-shard cast
    for i0, i1, logits in HD.sym_logit_rows(logits_fn, z_loc, lay, rows=None, bins=int(head.linear.weight.shape[0])):
        with torch.autocast(device_type=dev_type, enabled=False):
            out[i0:i1] = SC.compute_contact_prob(distogram_logits=logits.to(torch.float32), **bins)
        del logits
        COUNTS["distogram_rows"] += 1
    _mark("distogram")
    return ContactRows(out, lay)


def head_rows_fn(module, z_shard):
    """`rows -> module(rows)` for a head the core applies to ROW BLOCKS of a pair shard (heads.sym_logit_rows / logit_rows / embed_rows
    callbacks), in the arithmetic of the engine's own n_gpu 1 path — the ×P line is a PLACEMENT lever. The stock computes `distogram_head(z)`
    (protenix.py:576) under the runner's AMBIENT autocast (runner/inference.py:220-227: a bf16 / fp16 GEMM and symmetric sum under
    `--dtype bf16|fp16`, plain fp32 under fp32); the trunk shard carries that regime in its dtype (a bf16 / fp16 shard is a trunk that ran under
    autocast of that dtype), so: a bf16 / fp16 shard -> the block runs under torch.autocast(<device>, dtype=<the shard's dtype>) whatever the
    caller's autocast state (DiffusionConditioning.prepare_cache and run_confidence_head run with autocast DISABLED by
    autocasting_disable_decorator, which casts tensor ARGUMENTS only — never a PairShard: without this a bf16 block would meet the fp32
    weight unconverted); an fp32 shard -> plain, in the weight's dtype (no autocast). Never a whole-shard cast: the block is the only transient.
    Census: `disto_gemm=autocast:<dtype>` | `fp32` (evidence.record_schedule at the distogram rows)."""
    import torch
    amp = z_shard.dtype if z_shard.dtype in (torch.bfloat16, torch.float16) else None
    w = next((p for p in module.parameters()), None)
    w_dtype = w.dtype if w is not None else z_shard.dtype

    def fn(rows):
        if amp is not None:
            with torch.autocast(device_type="cuda" if rows.is_cuda else "cpu", dtype=amp, enabled=True):
                return module(rows)
        return module(rows if rows.dtype == w_dtype else rows.to(w_dtype))
    fn.word = f"autocast:{str(amp).replace('torch.', '')}" if amp is not None else str(w_dtype).replace("torch.", "").replace("float32", "fp32")
    return fn


def compute_contact_prob_tp(stock):
    def compute_contact_prob(distogram_logits, *a, **k):
        if isinstance(distogram_logits, ContactRows):                                  # produced per row block by distogram_tp already
            return distogram_logits
        return stock(distogram_logits, *a, **k)
    compute_contact_prob.__wrapped__ = stock
    return compute_contact_prob


# =========================================================================================================== confidence head (S6 + core reducer)
class TPHead(object):
    """`pred_dict["pae"]` / `["pde"]` under TP: the per-sample confidence.RowBlockReducer of this rank's rows (logits reduced on the fly); the
    summary keys are finished on rank 0 by `compute_full_data_and_summary_tp`."""
    __slots__ = ("reducers", "chains", "layout")

    def __init__(self, reducers, chains, layout):
        self.reducers, self.chains, self.layout = reducers, chains, layout


def _token_is_ligand(token_asym_id, atom_to_token_idx, atom_is_polymer):
    import torch
    atom_is_ligand = (1 - atom_is_polymer).long()
    til = torch.zeros_like(token_asym_id).long().scatter_add(0, atom_to_token_idx.long(), atom_is_ligand)
    return til > 0


def confidence_tp(head, input_feature_dict, s_inputs, s_trunk, zsh: PairShard, pair_mask, x_pred_coords, use_embedding=True,
                  triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
    """ConfidenceHead.forward (confidence.py:133-256 + memory_efficient_forward :258-348) with z_trunk as rows: per sample the pair input is
    built per row block (z_init outer sum + z_trunk rows, distance one-hot rows), the 4-block stack runs on rows WITH the single track, PAE rows
    and PDE rows of (z + z^T) are consumed per row block by confidence.RowBlockReducer; plddt / resolved on the replicated s. Returns
    (plddt [S, N_atom, b], TPHead, TPHead, resolved)."""
    import torch
    from protenix.model.utils import broadcast_token_to_atom, one_hot
    from protenix.model import sample_confidence as SC
    HD, CF, TK, EV = C("HD"), C("CF"), C("TK"), C("EV")
    feats = input_feature_dict
    lay = zsh.layout
    kernel, chunk = str(STATE["kernel"]), int(STATE["chunk"])
    if pair_mask is not None:
        _refuse("confidence pair_mask is not None (inference passes None)")
    conf_dtype = torch.float32 if not torch.is_autocast_enabled() else zsh.t.dtype      # run_confidence_head's autocasting_disable_decorator casts tensor inputs to fp32 (the stock's skip_amp ladder);
    r0, r1, R, N = lay.r0, lay.r1, lay.R, lay.N                                         # under autocast the pair input keeps the trunk's dtype — cast PER ROW BLOCK either way
    dev = zsh.t.device
    dev_type = "cuda" if zsh.t.is_cuda else "cpu"
    # ---- forward prologue (confidence.py:178-201)
    s_trunk = head.input_strunk_ln(torch.clamp(s_trunk, min=-512, max=512))
    x_rep_atom_mask = feats["distogram_rep_atom_mask"].bool()
    x_pred_rep_coords = x_pred_coords[..., x_rep_atom_mask, :]
    N_sample = int(x_pred_rep_coords.size(-3))
    s1 = head.linear_no_bias_s1(s_inputs)                                               # column term  (z_init = s1[None, :, :] + s2[:, None, :])
    s2 = head.linear_no_bias_s2(s_inputs)                                               # row term

    def zrows_fn(z_blk, g0, g1):                                                         # confidence.py:199-201 + :221 on rows [g0, g1): z_init rows + z_trunk rows, in the pass's dtype
        zi = TK.outer_sum_rows(s2, s1, g0, g1).to(conf_dtype)
        return zi + (z_blk.to(conf_dtype) if use_embedding else 0 * z_blk.to(conf_dtype))
    model = STATE.get("model")
    cfg = model.configs if model is not None else None
    pae_bins = SC.get_bin_params(cfg.loss.pae) if cfg is not None else {"min_bin": 0.0, "max_bin": 32.0, "no_bins": 64}
    pde_bins = SC.get_bin_params(cfg.loss.pde) if cfg is not None else {"min_bin": 0.0, "max_bin": 32.0, "no_bins": 64}
    contact = zsh.contact if zsh.contact is not None else STATE["contact_rows"].get(world()[1])
    if not isinstance(contact, ContactRows) or contact.layout.N != N:
        _refuse("confidence head before the distogram rows of this item (main_inference_loop order)")
    atom_is_polymer = 1 - feats["is_ligand"]
    chains = CF.ChainIndex(feats["asym_id"], feats["has_frame"], _token_is_ligand(feats["asym_id"], feats["atom_to_token_idx"], atom_is_polymer))
    finish = os.environ.get(CONF_FINISH_ENV, "exact").strip() or "exact"
    plddt_preds, resolved_preds, reducers = [], [], []
    atom_to_token_idx, atom_to_tokatom_idx = feats["atom_to_token_idx"], feats["atom_to_tokatom_idx"]
    plan = conf_plan(zsh, N_sample)                                                     # the trunk shard across the N_sample passes: parked | in place | resident (heads.ZTrunkPlan decides)
    d_bins = int(head.linear_no_bias_d.weight.shape[1])
    stack_form = None
    for i in range(N_sample):
        COUNTS["conf_samples"] += 1
        src, inplace = plan.begin(i, out_dtype=conf_dtype)                              # parked: the device storage is released BEFORE the pass's pair input is allocated
        z_pair = HD.embed_rows(zrows_fn, src, lay, inplace=inplace, out_dtype=conf_dtype, bins=d_bins)   # [R, N, C]: this pass's pair input (one per sample: confidence.py:221)
        plan.retire(i)                                                                  # the LAST pass drops the trunk park now, before its pair stack (census
                                                                                        # ztrunk_retired=pass<i>@embed; earlier passes: kept:not_last; in place / resident: nothing_parked)
        # ---- distance embedding rows (confidence.py:276-304)
        with torch.autocast(device_type=dev_type, enabled=False):
            xr = x_pred_rep_coords[..., i, :, :].to(torch.float32)
            d_rows = torch.cdist(xr[r0:r1], xr)                                         # [R, N]: this rank's rows of the stock [N, N] distances (never the whole matrix)
        rb, _src = HD.conf_rows(N, d_bins + int(z_pair.shape[-1]), None, 1)
        for b0 in range(0, R, rb):
            b1 = min(R, b0 + rb)
            z_pair[b0:b1] += head.linear_no_bias_d(one_hot(x=d_rows[b0:b1], lower_bins=head.lower_bins, upper_bins=head.upper_bins))
            z_pair[b0:b1] += head.linear_no_bias_d_wo_onehot(d_rows[b0:b1].unsqueeze(dim=-1))
        del d_rows
        # ---- the 4-block pair stack on rows, with s (confidence.py:307): IN PLACE on z_pair (pairstack.pair_stack_ reuse_storage)
        s_single, z_out = pair_stack_tp(head.pairformer_stack, s_trunk.clone() if inplace_safe else s_trunk, z_pair, lay, kernel, chunk, "confidence")
        stack_form = "inplace" if z_out is z_pair else "copy"
        z_pair = z_out
        del z_out
        s_single = s_single.to(torch.float32)
        with torch.autocast(device_type=dev_type, enabled=False):
            a = broadcast_token_to_atom(x_token=s_single, atom_to_token_idx=atom_to_token_idx)
            plddt_pred = torch.einsum("...nc,ncb->...nb", head.plddt_ln(a), head.plddt_weight[atom_to_tokatom_idx])
            resolved_pred = torch.einsum("...nc,ncb->...nb", head.resolved_ln(a), head.resolved_weight[atom_to_tokatom_idx])
            # ---- PAE rows / PDE rows of (z + z^T), reduced on the fly (confidence.py:318-331: `z_pair.to(torch.float32)` PER ROW BLOCK, then the heads)
            red = CF.RowBlockReducer(chains, r0, r1, dev, pae_bins=(pae_bins["min_bin"], pae_bins["max_bin"], pae_bins["no_bins"]),
                                     pde_bins=(pde_bins["min_bin"], pde_bins["max_bin"], pde_bins["no_bins"]), finish=finish, keep_value_rows=True)
            for i0, i1, zsym in HD.sym_logit_rows(lambda t: t, z_pair, lay, rows=None, bins=2 * int(z_pair.shape[-1]) + 128):
                pae = head.linear_no_bias_pae(head.pae_ln(z_pair[i0:i1].to(torch.float32)))
                pde = head.linear_no_bias_pde(head.pde_ln(zsym.to(torch.float32).contiguous()))
                red.consume(r0 + i0, r0 + i1, pae, pde, contact.rows[i0:i1])
                COUNTS["conf_rows"] += 1
                del pae, pde, zsym
        reducers.append(red)
        del z_pair, s_single, a, src
        plddt_preds.append(plddt_pred)
        resolved_preds.append(resolved_pred)
        if N > 2000 and zsh.t.is_cuda:
            torch.cuda.empty_cache()                                                    # the stock's own release site (confidence.py:233, :347)
        plan.end(i, device_needed_next=False)                                           # nothing reads the device trunk shard whole after a pass (protenix.py:586 is z's last reader)
    del s1, s2
    PARKS["z_trunk"] = plan.record()
    EV.record_schedule(conf_pairstack=stack_form or "-", conf_dtype=str(conf_dtype).replace("torch.", ""))
    _mark("confidence")
    th = TPHead(reducers, chains, lay)
    return torch.stack(plddt_preds, dim=-3), th, th, torch.stack(resolved_preds, dim=-3)


def _assemble_rank0(configs, stats: dict, red, *, plddt_logits, token_asym_id, token_has_frame, atom_coordinate, atom_to_token_idx,
                    atom_is_polymer, N_recycle, return_full_data, interested_atom_mask, mol_id, elements_one_hot):
    """sample_confidence._compute_full_data_and_summary (sample_confidence.py:84-140) for ONE sample on rank 0, the pair statistics taken from
    the reducer's finished keys (their finishing statements ARE the stock's: opt_core/mem/rowpair/confidence.py header)."""
    import torch
    from protenix.model import sample_confidence as SC
    full_data = {}
    full_data["atom_plddt"] = SC.logits_to_score(plddt_logits, **SC.get_bin_params(configs.loss.plddt))
    if "token_pair_pde_f16" in stats:
        full_data["token_pair_pde"] = stats["token_pair_pde_f16"].float().unsqueeze(0)
        full_data["contact_probs"] = stats["contact_probs_f16"].float()
        full_data["token_pair_pae"] = stats["token_pair_pae_f16"].float().unsqueeze(0)
    summary = {}
    summary["plddt"] = full_data["atom_plddt"].mean(dim=-1) * 100
    if red.finish == "exact":
        token_pair_pde, contact_probs = stats["token_pair_pde_f32"], stats["contact_probs_f32"]
        summary["gpde"] = (token_pair_pde * contact_probs).sum(dim=[-1, -2]) / contact_probs.sum(dim=[-1, -2])
    else:
        summary["gpde"] = stats["gpde"]
    summary["ptm"], summary["iptm"] = stats["ptm"], stats["iptm"]
    if red.finish == "exact":
        summary.update(SC.calculate_chain_based_gpde(token_pair_pde=token_pair_pde, contact_probs=contact_probs, asym_id=token_asym_id))
    else:
        summary.update({"chain_gpde": stats["chain_gpde"], "chain_pair_gpde": stats["chain_pair_gpde"]})
    summary.update({"chain_ptm": stats["chain_ptm"], "chain_iptm": stats["chain_iptm"], "chain_pair_iptm": stats["chain_pair_iptm"],
                    "chain_pair_iptm_global": stats["chain_pair_iptm_global"]})
    summary.update(SC.calculate_chain_based_plddt(full_data["atom_plddt"], token_asym_id, atom_to_token_idx))
    summary["has_clash"] = SC.calculate_clash(atom_coordinate, token_asym_id, atom_to_token_idx, atom_is_polymer, configs.metrics.clash.af3_clash_threshold)
    summary["num_recycles"] = torch.tensor(N_recycle, device=atom_coordinate.device)
    summary["disorder"] = torch.zeros_like(summary["ptm"])
    summary["ranking_score"] = 0.8 * summary["iptm"] + 0.2 * summary["ptm"] + 0.5 * summary["disorder"] - 100 * summary["has_clash"]
    if interested_atom_mask is not None:
        token_idx = atom_to_token_idx[interested_atom_mask[0].bool()].long()
        asym_ids = token_asym_id[token_idx]
        if len(torch.unique(asym_ids)) != 1:
            _refuse("interested_atom_mask spans more than one chain (stock asserts the same)")
        interested_asym_id = asym_ids[0].item()
        N_chains = token_asym_id.max().long().item() + 1
        pb = summary["chain_pair_iptm_global"][:, interested_asym_id, torch.arange(N_chains) != interested_asym_id]
        summary["pb_ranking_score"] = pb[:, 0]
        if elements_one_hot is not None and mol_id is not None:
            vdw = SC.calculate_vdw_clash(pred_coordinate=atom_coordinate, asym_id=token_asym_id, mol_id=mol_id, is_polymer=atom_is_polymer,
                                         atom_token_idx=atom_to_token_idx, elements_one_hot=elements_one_hot, threshold=configs.metrics.clash.vdw_clash_threshold)
            flag = vdw[:, interested_asym_id, :].reshape(atom_coordinate.shape[0], -1).max(dim=-1)[0]
            summary["has_vdw_pl_clash"] = flag
            summary["pb_ranking_score_vdw_penalized"] = summary["pb_ranking_score"] - 100 * flag
    summary = SC.break_down_to_per_sample_dict(summary, shared_keys=["num_recycles"])
    if not return_full_data:
        return summary, [{}]
    full_data["token_has_frame"] = token_has_frame.clone()
    full_data["token_asym_id"] = token_asym_id.clone()
    full_data["atom_to_token_idx"] = atom_to_token_idx.clone()
    full_data["atom_is_polymer"] = atom_is_polymer.clone()
    full_data["atom_coordinate"] = atom_coordinate.clone()
    full_data = SC.break_down_to_per_sample_dict(full_data, shared_keys=["contact_probs", "token_has_frame", "token_asym_id", "atom_to_token_idx", "atom_is_polymer"])
    return summary, full_data


def compute_full_data_and_summary_tp(stock):
    def compute_full_data_and_summary(configs, pae_logits, plddt_logits, pde_logits, contact_probs, token_asym_id, token_has_frame,
                                      atom_coordinate, atom_to_token_idx, atom_is_polymer, N_recycle, interested_atom_mask=None,
                                      return_full_data=False, mol_id=None, elements_one_hot=None):
        if not isinstance(pae_logits, TPHead):
            return stock(configs=configs, pae_logits=pae_logits, plddt_logits=plddt_logits, pde_logits=pde_logits, contact_probs=contact_probs,
                         token_asym_id=token_asym_id, token_has_frame=token_has_frame, atom_coordinate=atom_coordinate,
                         atom_to_token_idx=atom_to_token_idx, atom_is_polymer=atom_is_polymer, N_recycle=N_recycle,
                         interested_atom_mask=interested_atom_mask, return_full_data=return_full_data, mol_id=mol_id, elements_one_hot=elements_one_hot)
        import torch
        th: TPHead = pae_logits
        lay = th.layout
        rank = world()[1]
        N_sample = len(th.reducers)
        summaries: List[dict] = []
        fulls: List[dict] = []
        bounds = [tuple(b) for b in lay.bounds]
        dev_type = "cuda" if atom_coordinate.is_cuda else "cpu"
        for i, red in enumerate(th.reducers):
            stats = red.finalize(bounds, collect_full=bool(return_full_data))          # COLLECTIVE: every rank
            COUNTS["summaries"] += 1
            if rank != 0:
                continue
            with torch.autocast(device_type=dev_type, enabled=False):
                s_i, f_i = _assemble_rank0(configs, stats, red, plddt_logits=plddt_logits[i:i + 1].float(), token_asym_id=token_asym_id,
                                           token_has_frame=token_has_frame, atom_coordinate=atom_coordinate[i:i + 1].float(),
                                           atom_to_token_idx=atom_to_token_idx, atom_is_polymer=atom_is_polymer, N_recycle=N_recycle,
                                           return_full_data=return_full_data, interested_atom_mask=interested_atom_mask, mol_id=mol_id,
                                           elements_one_hot=elements_one_hot)
            summaries.extend(s_i)
            fulls.extend(f_i)
            del stats
        STATE["contact_rows"].pop(rank, None)
        _mark("done")
        if rank != 0:                                                                   # ranks > 0 write nothing (DataDumper.dump is rank-0 only): placeholders of the stock's list shape
            return [{} for _ in range(N_sample)], [{} for _ in range(N_sample)]
        return summaries, (fulls if return_full_data else [{}])
    compute_full_data_and_summary.__wrapped__ = stock
    return compute_full_data_and_summary


# =========================================================================================================== diffusion (S7)
def prepare_cache_cond_tp(cond, relp: RelpRows, zsh: PairShard, inplace_safe=False):
    """DiffusionConditioning.prepare_cache (diffusion.py:86-107) on rows: linear_z(LN(cat[z_rows, relpe(relp rows)])) + transition_z1 + z2."""
    import torch
    DF = C("DF")
    lay = zsh.layout
    entry = rollout_entry(zsh)                                                          # distogram rows first, then the trunk shard parked (ROWPAIR_CONF_PARK_ZTRUNK) before z_cond is allocated
    src = zsh.rows_source()                                                             # the host park's .zrows | the device tensor
    want = torch.float32 if not torch.is_autocast_enabled() else zsh.t.dtype           # the runner's autocasting_disable_decorator(skip_amp.sample_diffusion) casts z (and relp) to fp32 when
                                                                                        # it disables autocast: cast PER ROW BLOCK here (no whole-shard fp32 copy)

    def embed_fn(z_rows, g0, g1):
        z_rows = z_rows.to(want)
        p = torch.cat([z_rows, cond.relpe(relp.rows(g0, g1).to(z_rows.dtype) if z_rows.dtype != torch.float32 else relp.rows(g0, g1))], dim=-1)
        return cond.linear_no_bias_z(cond.layernorm_z(p))

    def t1(x_rows, g0, g1):
        return cond.transition_z1(x_rows)

    def t2(x_rows, g0, g1):
        return cond.transition_z2(x_rows)
    out = torch.empty((lay.R, lay.N, int(cond.linear_no_bias_z.weight.shape[0])), dtype=want, device=zsh.t.device)
    out = DF.pair_cond_rows(embed_fn, src, lay, c_out=int(cond.linear_no_bias_z.weight.shape[0]), rows=256, transitions=[t1, t2], out=out)
    COUNTS["zcond_rows"] += 1
    C("EV").record_schedule(conf_ztrunk_entry_site=f"prepare_cache:{entry}", zcond_src="parked" if hasattr(src, "zrows") else "device", zcond_dtype=str(want).replace("torch.", ""))
    _mark("diffusion_start")
    return PairShard(out, lay)


def encoder_prepare_cache_tp(stock):
    """AtomAttentionEncoder.prepare_cache (transformer.py:728-830): everything verbatim except the token-pair windows of the conditioned pair,
    which are read from the replicated BAND diffusion.pair_band_rows builds from this rank's rows (== the stock gather of padded indices)."""
    def prepare_cache(self, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, atom_to_token_idx, d_lm, v_lm, pad_info,
                      r_l=None, z=None, inplace_safe=False):
        if not isinstance(z, PairShard):
            return stock(self, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, atom_to_token_idx, d_lm, v_lm, pad_info,
                         r_l=r_l, z=z, inplace_safe=inplace_safe)
        import torch
        from protenix.model.modules.primitives import rearrange_qk_to_dense_trunk
        DF = C("DF")
        p_lm, c_l = stock(self, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, atom_to_token_idx, d_lm, v_lm, pad_info,
                          r_l=None, z=None, inplace_safe=inplace_safe)                  # every statement up to the pair windows (r_l=None skips them)
        if r_l is None:
            return p_lm, c_l
        lay = z.layout
        z_loc = z.float_rows() if not torch.is_autocast_enabled() else z.t
        idx_q, idx_k, _pad = rearrange_qk_to_dense_trunk(atom_to_token_idx, atom_to_token_idx, dim_q=-1, dim_k=-1,
                                                        n_queries=self.n_queries, n_keys=self.n_keys, compute_mask=False)
        q3 = idx_q.long().reshape(1, *idx_q.shape[-2:])
        k3 = idx_k.long().reshape(1, *idx_k.shape[-2:])
        valid = pad_info["mask_trunked"].bool().reshape(1, *pad_info["mask_trunked"].shape[-3:]) if "mask_trunked" in pad_info else torch.ones(q3.shape + (k3.shape[-1],), dtype=torch.bool, device=q3.device)
        w_env = os.environ.get(BAND_W_ENV, "").strip()
        plan = DF.band_plan(q3, k3, valid, lay.N, max_w=int(w_env) if w_env else None)
        band, extras = DF.pair_band_rows(lambda zr: self.linear_no_bias_z(self.layernorm_z(zr)), z_loc, lay, plan)
        zg = DF.band_lookup(band, extras, plan)[0]                                      # [nb, nq, nk, c_atompair] == z_proj[idx_q, idx_k]
        del band, extras
        COUNTS["band_calls"] += 1
        p_lm = p_lm.unsqueeze(dim=-5) + zg.reshape(idx_q.shape[:-2] + zg.shape[-4:]) if idx_q.dim() > 2 else p_lm.unsqueeze(dim=-5) + zg
        return p_lm, c_l
    prepare_cache.__wrapped__ = stock
    return prepare_cache


DIT_ATTN_ENV = "ROWPAIR_DIT_ATTN"                  # rows (default) | sdpa | stock: the DiT token attention's CORE on this rank's query rows under n_gpu>1 —
DIT_ATTN_CORES = ("rows", "sdpa", "stock")         #   rows = the shared core's rows face (mem.rowpair.diffusion.dit_attention_rows on
DIT_ROWS_ENV = "ROWPAIR_DIT_ROWS"                  #   kernels.apb.pair_bias_attention_rows; row word DIT_ROWS_ENV, default the tier word `big` = the rows
DIT_ROWS_DEFAULT = "big"                         #   cell's row for this card else apb_attn by name: k|v projected + cast to bf16 once per block, q per query block, the bias
                                                   #   rows read in place, fp32 statistics/accumulation; its fallback BY NAME is the sdpa statement; census
                                                   #   dit_rows=<row>:<n> | stock:<kind>:<n>, dit_rows_core=<describe>);
                                                   #   (no [S, H, q, N] logits tensor; fp32 operands, the bias enters as the additive mask) then the
                                                   #   module's gating + output projection; stock = the Attention module's forward (the stock statement:
                                                   #   logits, + bias, softmax, PV in torch — its fused single-GPU kernel declines q rows != kv rows)
DIT_CACHE_POLICY_ENV = "ROWPAIR_DIFF_BIAS_CACHE_POLICY"   # fit (default) | core: whether the 24 DiT blocks' pair-bias rows of a roll-out are CACHED
DIT_CACHE_POLICIES = ("fit", "core")                      #   (computed once from the conditioned rows, reused by every step and sample):
DIT_CACHE_DTYPE_ENV = "ROWPAIR_DIFF_BIAS_CACHE_DTYPE"     #   fit = on iff n_blocks*H*Rmax*N*elt + FIT_MARGIN fits FIT_FRAC of the agreed usable device bytes
DIT_CACHE_DTYPES = ("fp16", "bf16", "fp32")               #   at the roll-out's first denoiser call (device free + the allocator's free pool, min over ranks);
DIT_FIT_FRAC = 0.9                                        #   core = opt_core's rule (ROWPAIR_DIFF_BIAS_CACHE=0|1 pins, else fits ROWPAIR_DIFF_BIAS_CACHE_GB).
DIT_FILL_ROWS = 256                                       #   The fill runs in DIT_FILL_ROWS-row blocks (its LN/projection transient stays small next to the set).
DIT_ZCOND_RELEASE_ENV = "ROWPAIR_DIFF_ZCOND_RELEASE"     #   1 (default) | 0: release the conditioned pair rows' device storage once the set is cached (dit_zcond_release).
DIT_FIT_MARGIN_GB = 4.0                                   #   A cached set is held in DIT_CACHE_DTYPE (default fp16 — the operand class of the single-GPU
                                                          #   line's ditattnfp16 on this card; a recomputed row block stays fp32); without the cache
                                                          #   every block's bias rows are recomputed at every step (24 x 200 x S LayerNorm+Linear passes over [R, N, 128]).


def dit_attn_core() -> str:
    word = (os.environ.get(DIT_ATTN_ENV) or "rows").strip().lower()
    if word not in DIT_ATTN_CORES:
        _refuse(f"{DIT_ATTN_ENV}={word!r}: one of {'|'.join(DIT_ATTN_CORES)}")
    if word == "rows" and not hasattr(C("DF"), "dit_attention_rows"):
        return "sdpa"                                                                   # a core without the rows face: the kit's own SDPA rows
    return word


def dit_rows_word() -> str:
    w = (os.environ.get(DIT_ROWS_ENV) or DIT_ROWS_DEFAULT).strip().lower()
    return "" if w in ("engine", "off", "none", "stock", "0") else w


def dit_cache_dtype():
    import torch
    word = (os.environ.get(DIT_CACHE_DTYPE_ENV) or "fp16").strip().lower()
    if word not in DIT_CACHE_DTYPES:
        _refuse(f"{DIT_CACHE_DTYPE_ENV}={word!r}: one of {'|'.join(DIT_CACHE_DTYPES)}")
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[word]


def dit_bias_cache_decision(cache_bytes: int):
    """-> (decision or None, census word). None hands the rule to the core's DiffusionSchedule.decide (policy `core`, or a caller's
    ROWPAIR_DIFF_BIAS_CACHE=0|1 pin: word `env`). Policy `fit`: ONE decision for all ranks (int64 min-allreduce of the usable bytes)."""
    import torch
    D, DF = C("D"), C("DF")
    pin = (os.environ.get(DF.ENV_BIAS_CACHE) or "").strip().lower()
    if pin in ("0", "1", "on", "off"):
        return None, "env"
    policy = (os.environ.get(DIT_CACHE_POLICY_ENV) or "fit").strip().lower()
    if policy not in DIT_CACHE_POLICIES:
        _refuse(f"{DIT_CACHE_POLICY_ENV}={policy!r}: one of {'|'.join(DIT_CACHE_POLICIES)}")
    if policy == "core" or not torch.cuda.is_available():
        return None, policy
    free, _total = torch.cuda.mem_get_info()
    room = int(free) + int(torch.cuda.memory_reserved()) - int(torch.cuda.memory_allocated())
    if D.is_dist():
        c = D.comm()
        t = torch.tensor([room], dtype=torch.int64, device=c.device)
        c.allreduce_(t, "min")
        room = int(t.item())
    on = int(cache_bytes) + int(DIT_FIT_MARGIN_GB * 2 ** 30) <= int(room * DIT_FIT_FRAC)
    return on, "fit:%s:%.1f/%.1fGiB" % ("on" if on else "off", cache_bytes / 2 ** 30, room / 2 ** 30)


def dit_rollout(dm, zsh: "PairShard", N_sample: int, fusion: bool) -> dict:
    """The DiT state of ONE `sample_diffusion` call on this rank (every step of every sample it rolls out shares the conditioned pair rows,
    hence the pair biases), built at the call's FIRST denoiser call and dropped by rowpair's sample_diffusion seam on exit
    (`dit_rollout_clear`; a state is never keyed on tensor identity — every rank takes the decision's collective exactly once per call):
    the core's DiffusionSchedule (row blocks + the bias-cache decision), ONE PairBiasCache, the blocks' callables (bias rows in the cache
    dtype when cached), the attention core word. Census: diff_bias_cache_policy= diff_bias_cache_dtype= dit_attn= (+ the schedule's fields)."""
    import torch
    DF = C("DF")
    lay = zsh.layout
    st = STATE.get("dit")
    if st is not None:
        if st["key"] != (int(lay.N), int(lay.P), int(N_sample)):
            _refuse("dit_rollout: the layout or sample count changed inside one sample_diffusion call")
        return st
    key = (int(lay.N), int(lay.P), int(N_sample))
    dt = dm.diffusion_transformer
    H = int(dt.n_heads)
    dtype = dit_cache_dtype()
    elt = torch.empty((), dtype=dtype).element_size()
    cache_bytes = len(dt.blocks) * H * int(lay.Rmax) * int(lay.N) * elt
    on, word = dit_bias_cache_decision(cache_bytes)
    fill_rows = None if not on else DIT_FILL_ROWS                                       # a cached set is filled ONCE: small row blocks keep the fill's transient
    core = dit_attn_core()                                                              #   (LN + projection over [rows, N, C] fp32) small next to the set itself
    rows_word, attn_core, rows_why = "", None, "engine"
    if core == "rows":                                                                  # the rows face's STATIC admission, once per call, identical on every rank
        att0 = dt.blocks[0].attention_pair_bias.attention
        rows_word = dit_rows_word()
        attn_core, _sel, rows_why = DF.dit_rows_core(rows_word, dtype=torch.float32, heads=H, head_dim=int(att0.c_hidden), samples=max(1, int(N_sample)))
        if not (attn_core == "kernel" or rows_word == "naive"):
            core, rows_word = "sdpa", ""                                                # refused by name (word / card / operand class): the kit's SDPA rows
    kw = {} if attn_core is None else {"attn_core": attn_core}
    sched = DF.DiffusionSchedule.decide(lay, c_z=int(dm.c_z), c_in=2 * int(dm.c_z), c_cond=int(dm.c_z), H=H, S=max(1, int(N_sample)),
                                      n_blocks=len(dt.blocks), elt=4, bias_cache=on, bias_rows=fill_rows, **kw)
    cache = DF.PairBiasCache(enabled=bool(sched.bias_cache))
    bias_dtype = dtype if cache.enabled else None
    q_stock = int(getattr(sched, "q_rows_stock", 0) or 0) or None
    blocks = [dit_fns(b, fusion, bias_dtype=bias_dtype, core=core, normalize=dm.normalize if fusion else None, rows_word=rows_word, stock_q_rows=q_stock)
              for b in dt.blocks]
    dtype_word = str(dtype).replace("torch.", "") if cache.enabled else "fp32:recompute"
    entry_gib = torch.cuda.memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0
    C("EV").record_schedule(diff_bias_cache_policy=word, diff_bias_cache_dtype=dtype_word, dit_attn=core if core != "rows" else f"rows:{rows_word}",
                            diff_entry_alloc_gib=round(entry_gib, 2), diff_bias_cache_gib=round(cache_bytes / 2 ** 30, 2) if cache.enabled else 0)
    os.write(2, (f"[protenix-v1-opt] TP-DIFF sampler_entry rank={lay.rank}/{lay.P} diff_bias_cache={int(cache.enabled)} diff_bias_cache_policy={word} "
                 f"diff_bias_cache_dtype={dtype_word} dit_attn={core}{(':' + rows_word) if rows_word else ''}({rows_why}) q_rows={sched.q_rows} bias_rows={sched.bias_rows} "
                 f"entry_alloc_gib={entry_gib:.2f} N={lay.N} R={lay.R} S={int(N_sample)}\n").encode())
    COUNTS["dit_rollouts"] += 1
    st = {"key": key, "sched": sched, "cache": cache, "blocks": blocks, "fusion": bool(fusion), "n_blocks": len(dt.blocks), "core": core, "zcond_released": False,
          "lay": lay}
    STATE["dit"] = st
    return st


def dit_zcond_release(st: dict, zsh: "PairShard") -> None:
    """Once every block's bias rows are cached, the conditioned pair rows have no reader left in this sample_diffusion call (the atom
    encoder reads the cached band p_lm / c_l, the conditioning returns the shard object unread, the confidence stage reads the trunk park):
    their DEVICE storage is released for the rest of the roll-out (DIT_ZCOND_RELEASE_ENV, default on; census zcond_released_gib=). The shard
    object keeps its shape (the core's shape check reads nothing else)."""
    import torch
    if st["zcond_released"] or not (st["cache"].enabled and len(st["cache"].store) == st["n_blocks"]):
        return
    if (os.environ.get(DIT_ZCOND_RELEASE_ENV) or "1").strip() not in ("1", "on", "true"):
        return
    t = zsh.t
    gib = t.untyped_storage().nbytes() / 2 ** 30
    t.untyped_storage().resize_(0)
    st["zcond_released"] = True
    C("EV").record_schedule(zcond_released_gib=round(gib, 2))
    if torch.cuda.is_available():
        os.write(2, (f"[protenix-v1-opt] TP-DIFF bias_cache_filled rank={zsh.layout.rank}/{zsh.layout.P} cache_gib={st['cache'].nbytes / 2 ** 30:.2f} "
                     f"zcond_released_gib={gib:.2f} alloc_gib={torch.cuda.memory_allocated() / 2 ** 30:.2f} max_alloc_gib={torch.cuda.max_memory_allocated() / 2 ** 30:.2f}\n").encode())


def dit_rollout_clear() -> None:
    """Drop the roll-out's DiT state (the cached bias rows leave the device): rowpair's sample_diffusion seam calls it on exit, `undo` too."""
    st = STATE.pop("dit", None)
    if st is not None:
        DF, EV = C("DF"), C("EV")
        if hasattr(DF, "dit_rows_census"):
            rows_c, bias_c = DF.dit_rows_census(), DF.dit_bias_census()
            EV.record_schedule(dit_rows=rows_c, dit_bias=bias_c)
            lay = st.get("lay")
            if lay is not None:
                os.write(2, (f"[protenix-v1-opt] TP-DIFF sampler_done rank={lay.rank}/{lay.P} dit_rows={rows_c} dit_bias={bias_c} "
                             f"cache_gib={st['cache'].nbytes / 2 ** 30:.2f}\n").encode())
        st["cache"].clear()
        st["blocks"] = []


def dit_fns(block, fusion: bool, bias_dtype=None, core: Optional[str] = None, normalize=None, rows_word: str = "", stock_q_rows: Optional[int] = None):
    """DiTBlockFns of one protenix DiffusionTransformerBlock (transformer.py:257-355; AttentionPairBias with has_s, standard attention):
    norm = AdaLN(a, s); attn = attention(q rows, kv all, bias rows); update = sigmoid(linear_a_last(s rows)) gate, + a rows, conditioned
    transition residual; bias = linear_nobias_z(layernorm_z(z_rows)) or, under enable_efficient_fusion, the stock conv2d with the LN weight
    folded, on the pre-normalised rows (or, given `normalize` = DiffusionModule.normalize, on RAW rows it normalises per block: row-local, the
    same statement, no whole-shard normalised copy) — left in `bias_dtype` when given (a cached set's dtype). `core`: the attention core (DIT_ATTN_ENV;
    None = the env's word): `sdpa` = the module's _prep_qkv -> scaled_dot_product_attention(q, k, v, attn_mask=bias) -> _wrap_up (gating,
    output projection) = the module's global-attention forward without the [*, H, q, N] logits tensor; `stock` = the module's forward."""
    import torch
    import torch.nn.functional as F
    DF = C("DF")
    apb = block.attention_pair_bias
    core = dit_attn_core() if core is None else core
    if core == "rows" and not rows_word:
        core = "sdpa"                                                                   # a direct caller without an admitted word: the SDPA rows
    mod = apb.attention                                                                 # protenix.model.modules.primitives.Attention (q|k|v|g|o linears)
    H = int(mod.num_heads)

    def norm(a, s):
        return apb.layernorm_a(a=a, s=s)

    def kv(x):
        return x

    def attn(x_q, x_all, bias_q, gg):                                                   # the kit's statements: sdpa | stock (the module's own forward); the rows face's fallback by name
        bias = bias_q
        while bias.dim() < x_q.dim() + 1:                                               # [*, H, q, N] with a's leading (sample) dims explicit: the stock
            bias = bias.unsqueeze(0)                                                    # Attention unsqueezes a bias of lower rank at the HEAD axis
        if core == "stock":
            COUNTS["dit_stock_attn"] += 1
            return apb.attention(q_x=x_q, kv_x=x_all, attn_bias=bias if bias.dtype == x_q.dtype else bias.to(dtype=x_q.dtype))
        q, k, v = mod._prep_qkv(q_x=x_q, kv_x=x_all, apply_scale=False)               # [*, H, q|N, c]; SDPA's default scale = 1/sqrt(c) = the module's
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias if bias.dtype == q.dtype else bias.to(dtype=q.dtype))
        COUNTS["dit_sdpa"] += 1
        return mod._wrap_up(o.transpose(-2, -3), x_q)                                  # [*, q, H, c] -> gate(x_q) -> linear_o: primitives.Attention.forward's tail

    if core == "rows":                                                                  # the rows face; `attn` / `kv` above are captured BEFORE the names are rebound
        stock_attn = attn

        def kv(x):                                                                      # k | v of ALL rows ONCE per block, bf16 under the tier word (cast16); stock = x (the sdpa statement's kv)
            k = mod.linear_k(x)
            v = mod.linear_v(x)
            k = k.view(k.shape[:-1] + (H, -1)).transpose(-2, -3)
            v = v.view(v.shape[:-1] + (H, -1)).transpose(-2, -3)
            return DF.DitKV(DF.cast16(k), DF.cast16(v), None, x)

        def q_fn(x_q):                                                                  # _prep_qkv's q for these rows, UNSCALED (the face applies c ** -0.5)
            q = mod.linear_q(x_q)
            return q.view(q.shape[:-1] + (H, -1)).transpose(-2, -3)

        def out_fn(o, x_q):                                                             # o [.., S, q, H*c] -> _wrap_up(o [.., q, H, c], q_x): gating + output projection
            COUNTS["dit_rows_attn"] += 1
            return mod._wrap_up(o.view(o.shape[:-1] + (H, -1)), x_q)

        attn = DF.dit_attention_rows(q_fn=q_fn, out_fn=out_fn, stock_fn=stock_attn, num_heads=H, core_word=rows_word, scale=None, stock_q_rows=stock_q_rows)

    def update(a, o_rows, s, rr):
        r0, r1 = rr
        s_r = s[..., r0:r1, :]
        o = o_rows * torch.sigmoid(apb.linear_a_last(s_r))                              # transformer.py:249-253 (has_s output gate)
        attn_out = o + a[..., r0:r1, :]                                                 # block: attn_out = attn + a
        ff = block.conditioned_transition_block(a=attn_out, s=s_r)
        return ff + attn_out

    if fusion:
        def bias(zn_rows):                                                              # zn_rows: pre-normalised rows [rows, N, C] (f_forward's normalize) or raw rows + `normalize`
            if normalize is not None:
                zn_rows = normalize(zn_rows.to(dtype=torch.float32))                    # the stock per-call normalize(z), on this row block only
            weight = (apb.linear_nobias_z.weight * apb.layernorm_z.weight[None, :])[:, :, None, None]
            y = F.conv2d(zn_rows.permute(2, 0, 1).unsqueeze(0), weight)[0]             # [H, rows, N]
            y = y.permute(1, 2, 0)                                                      # -> [rows, N, H] (pair_bias_rows moves heads first)
            COUNTS["dit_bias_computes"] += 1
            return y if bias_dtype is None else y.to(dtype=bias_dtype)
    else:
        def bias(z_rows):
            y = apb.linear_nobias_z(apb.layernorm_z(z_rows.to(dtype=torch.float32)))   # [rows, N, H]
            COUNTS["dit_bias_computes"] += 1
            return y if bias_dtype is None else y.to(dtype=bias_dtype)
    if not hasattr(DF, "DitBias"):
        return DF.DiTBlockFns(norm, kv, attn, update, bias)
    ln = apb.layernorm_z                                                                # the producer as WEIGHTS (ROWPAIR_DIFF_BIAS=ln_proj: one-pass LN + projection into the
    bias_into = DF.DitBias(engine=bias, ln_weight=getattr(ln, "weight", None), ln_bias=None if fusion else getattr(ln, "bias", None),   # [H, rows, N] block; engine (default)
                           weight=apb.linear_nobias_z.weight, linear_bias=None, eps=float(getattr(ln, "eps", 1e-5)))         # = bias() value for value); the fused path
    return DF.DiTBlockFns(norm, kv, attn, update, bias, bias_into=bias_into)                                                  # folds the LN weight: no LN offset term


def f_forward_tp(stock):
    """DiffusionModule.f_forward (diffusion.py:350-478) with the conditioned pair as rows: conditioning returns the shard, the atom encoder reads
    the cached p_lm (prepared from the band), the token transformer runs diffusion.diffusion_transformer_sharded; everything else verbatim."""
    def f_forward(self, r_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z=None, p_lm=None, c_l=None,
                  inplace_safe=False, chunk_size=None, use_conditioning=True, enable_efficient_fusion=False):
        if not isinstance(pair_z, PairShard) and not isinstance(z_trunk, PairShard):
            return stock(self, r_noisy=r_noisy, t_hat_noise_level=t_hat_noise_level, input_feature_dict=input_feature_dict, s_inputs=s_inputs,
                         s_trunk=s_trunk, z_trunk=z_trunk, pair_z=pair_z, p_lm=p_lm, c_l=c_l, inplace_safe=inplace_safe, chunk_size=chunk_size,
                         use_conditioning=use_conditioning, enable_efficient_fusion=enable_efficient_fusion)
        import torch
        DF = C("DF")
        if pair_z is None or p_lm is None or c_l is None:
            _refuse("tp_needs_shared_vars_cache: the diffusion shared-variable cache is off (--enable_cache false) under n_gpu>1")
        if not use_conditioning:
            _refuse("use_conditioning=False under n_gpu>1")
        N_sample = r_noisy.size(-3)                                                     # r_noisy arrives synced and the update leaves synced (synced_denoiser, below)
        s_single, z_pair = self.diffusion_conditioning(t_hat_noise_level, input_feature_dict["relp"], s_inputs=s_inputs, s_trunk=s_trunk,
                                                       z_trunk=None, pair_z=pair_z, inplace_safe=inplace_safe, use_conditioning=use_conditioning)
        lay = z_pair.layout
        from protenix.model.utils import expand_at_dim
        s_trunk_e = expand_at_dim(s_trunk, dim=-3, n=1)
        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
            input_feature_dict["atom_to_token_idx"], input_feature_dict["ref_pos"], input_feature_dict["ref_charge"], input_feature_dict["ref_mask"],
            input_feature_dict["ref_atom_name_chars"], input_feature_dict["ref_element"], input_feature_dict["d_lm"], input_feature_dict["v_lm"],
            input_feature_dict["pad_info"], r_l=r_noisy, s=s_trunk_e, z=z_pair, p_lm=p_lm, c_l=c_l, inplace_safe=inplace_safe, chunk_size=chunk_size)
                                                                                        # z: the stock asserts `z is not None` then reads it only in prepare_cache (skipped: p_lm / c_l are
                                                                                        # the cached band tensors); the shard is what the patched prepare_cache takes were the cache absent
        a_token = a_token.to(dtype=torch.float32)
        if inplace_safe:
            a_token += self.linear_no_bias_s(self.layernorm_s(s_single))
        else:
            a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s_single))
        fusion = bool(enable_efficient_fusion)
        ro = dit_rollout(self, z_pair, N_sample, fusion)                                # the roll-out's schedule, bias cache, block callables (built at its first call)
        if ro["fusion"] != fusion:
            _refuse("enable_efficient_fusion changed inside one roll-out")
        z_loc = z_pair.t                                                                # the blocks' bias() casts + normalises ITS row block (fused path: dm.normalize per
        blocks = ro["blocks"]                                                           # block — row-local, the stock statement; no whole-shard fp32 / normalised copy per step);
        COUNTS["dit_calls"] += 1                                                        # with every block's bias rows cached the rows are not read at all (shape check only)
        COUNTS["dit_blocks"] += len(blocks)
        a_token = DF.diffusion_transformer_sharded(blocks, a_token.to(dtype=torch.float32), s_single.to(dtype=torch.float32), z_loc, lay,
                                                  schedule=ro["sched"], bias_cache=ro["cache"])
        dit_zcond_release(ro, z_pair)                                                   # first call: the set is now cached -> the conditioned rows' device storage is released
        _mark("diffusion_peak")
        a_token = self.layernorm_a(a_token)
        r_update = self.atom_attention_decoder(atom_to_token_idx=input_feature_dict["atom_to_token_idx"], a=a_token, q_skip=q_skip, c_skip=c_skip,
                                               p_skip=p_skip, inplace_safe=inplace_safe, chunk_size=chunk_size)
        return r_update
    f_forward.__wrapped__ = stock
    body = f_forward

    def f_forward(self, r_noisy, *a, **kw):                                             # noqa: F811 — the stock signature; PairShard absent -> the stock verbatim (no sync, P=1 form)
        pair_z, z_trunk = kw.get("pair_z"), kw.get("z_trunk", a[4] if len(a) > 4 else None)   # positional: (t_hat, feats, s_inputs, s_trunk, z_trunk, ...)
        if not isinstance(pair_z, PairShard) and not isinstance(z_trunk, PairShard):
            return body(self, r_noisy, *a, **kw)
        return synced(self, r_noisy, *a, **kw)
    synced = synced_denoiser(body)
    f_forward.__wrapped__ = stock
    return f_forward


# =========================================================================================================== install / undo / evidence
def install(P: int, rank: int) -> List[str]:
    """Bind the statements above onto protenix's classes for a P>1 rank process (refused by name at P == 1: the structural n_gpu=1 rule).
    The run's triangle-attention kernel and chunk size are read off the model at its first trunk call (`_bind_model`). Returns the list of
    patched sites (the census's `tp_sites`). `undo()` restores every site."""
    if int(P) <= 1:
        _refuse("install at n_gpu=1 (the single-GPU path installs nothing)")
    import protenix.model.protenix as PX
    import protenix.model.modules.embedders as EM
    import protenix.model.modules.diffusion as DM
    import protenix.model.modules.transformer as TF
    import protenix.model.modules.head as HDM
    import protenix.model.modules.confidence as CO
    from protenix.model import sample_confidence as SC
    STATE.update(P=int(P), rank=int(rank), model=None, contact_rows={})
    sites: List[str] = []

    def patch(obj, name, new):
        old = getattr(obj, name)
        setattr(obj, name, new)
        STATE["undo"].append(lambda o=obj, n=name, v=old: setattr(o, n, v))
        sites.append(f"{getattr(obj, '__name__', obj.__class__.__name__)}.{name}")
        return old

    # relp: ids only (no [N,N,139] plane)
    stock_gen = EM.RelativePositionEncoding.generate_relp

    def generate_relp(self, input_feature_dict):
        input_feature_dict["relp"] = RelpRows(self, input_feature_dict)
        return input_feature_dict
    generate_relp.__wrapped__ = stock_gen
    patch(EM.RelativePositionEncoding, "generate_relp", generate_relp)
    # trunk
    get_pairformer_output_tp.__wrapped__ = PX.Protenix.get_pairformer_output
    patch(PX.Protenix, "get_pairformer_output", get_pairformer_output_tp)
    # distogram -> contact rows
    stock_dh = HDM.DistogramHead.forward

    def distogram_forward(self, z):
        if not isinstance(z, PairShard):
            return stock_dh(self, z)
        cr = z.contact if z.contact is not None else distogram_tp(self, z)              # taken at roll-out entry from the device shard (rollout_entry) | a caller without it
        STATE["contact_rows"][world()[1]] = cr
        return cr
    distogram_forward.__wrapped__ = stock_dh
    patch(HDM.DistogramHead, "forward", distogram_forward)
    patch(SC, "compute_contact_prob", compute_contact_prob_tp(SC.compute_contact_prob))
    # confidence head + summaries
    stock_cf = CO.ConfidenceHead.forward

    def confidence_forward(self, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_mask, x_pred_coords, use_embedding=True,
                           triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        if not isinstance(z_trunk, PairShard):
            return stock_cf(self, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, pair_mask=pair_mask,
                            x_pred_coords=x_pred_coords, use_embedding=use_embedding, triangle_multiplicative=triangle_multiplicative,
                            triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        return confidence_tp(self, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_mask, x_pred_coords, use_embedding=use_embedding,
                             triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe,
                             chunk_size=chunk_size)
    confidence_forward.__wrapped__ = stock_cf
    patch(CO.ConfidenceHead, "forward", confidence_forward)
    patch(SC, "compute_full_data_and_summary", compute_full_data_and_summary_tp(SC.compute_full_data_and_summary))
    # diffusion: conditioning rows, encoder band, f_forward
    stock_pc = DM.DiffusionConditioning.prepare_cache

    def prepare_cache(self, relp_feature, z_trunk, inplace_safe=False):
        if not isinstance(z_trunk, PairShard):
            return stock_pc(self, relp_feature, z_trunk, inplace_safe)
        return prepare_cache_cond_tp(self, relp_feature, z_trunk, inplace_safe)
    prepare_cache.__wrapped__ = stock_pc
    patch(DM.DiffusionConditioning, "prepare_cache", prepare_cache)
    patch(TF.AtomAttentionEncoder, "prepare_cache", encoder_prepare_cache_tp(TF.AtomAttentionEncoder.prepare_cache))
    patch(DM.DiffusionModule, "f_forward", f_forward_tp(DM.DiffusionModule.f_forward))
    import importlib
    TFZ = importlib.import_module(TEMPLATE_FEATURIZER_MODULE)                          # templated inputs: the featuriser emits per-token precursors, host_side_inputs bears this rank's rows
    patch(TFZ.Templates, TEMPLATE_DENSE_SITE, templates_as_precursors(getattr(TFZ.Templates, TEMPLATE_DENSE_SITE)))
    STATE["installed"] = True
    tk = tp_kernels()
    C("EV").record_schedule(tp_sites=";".join(sites), tp_msa_m=msa_m_layout(), tp_conf_finish=os.environ.get(CONF_FINISH_ENV, "exact") or "exact",
                            tp_trimul=tk["trimul_word"], tp_trimul_inplace_chunk=TRIMUL_INPLACE_CHUNK)
    if tk["triatt_word"] is not None:
        C("EV").record_schedule(tp_triatt_core=tk["triatt_word"])
    if tk["reason"] is not None:
        C("EV").record_schedule(tp_kernels_fallback=tk["reason"])
    return sites


def _bind_model(model, chunk_size, configs=None) -> None:
    """The run's facts the callables need, read off the Protenix module (or the runner's configs at item entry): the triangle-attention kernel
    the stock routes (`configs.triangle_attention`, the `--triatt_kernel` of the run: it serves this rank's query rows unchanged), the pinned
    chunk size (the layout's alignment and the tri-attention query block), the loss bin parameters (distogram / PAE / PDE expected values)."""
    cfg = configs if configs is not None else getattr(model, "configs", None)
    kernel = str(getattr(cfg, "triangle_attention", "torch")) if cfg is not None else "torch"
    chunk = chunk_size if chunk_size else (getattr(getattr(cfg, "infer_setting", None), "chunk_size", None) if cfg is not None else None)
    if model is not None or STATE.get("model") is None:
        STATE["model"] = model if model is not None else _ConfigsOnly(cfg)
    STATE.update(kernel=kernel, chunk=int(chunk or 128))
    C("EV").record_schedule(tp_triatt_kernel=kernel, tp_chunk=int(chunk or 128))


class _ConfigsOnly(object):
    __slots__ = ("configs",)

    def __init__(self, configs):
        self.configs = configs


# ---------------------------------------------------------------- template pair features BORN AS ROWS per rank (templated inputs under n_gpu > 1)
# The stock featuriser (data/template/template_featurizer.py:580-620, the `Templates` dense statement TEMPLATE_DENSE_SITE) forms the template
# pair features DENSE on the host — template_distogram [T,N,N,39], template_unit_vector [T,N,N,3] and the two 2-D masks [T,N,N] — before the
# model sees the item. Under n_gpu > 1 that dense birth is replaced (install patches the site): the featuriser emits only its per-token
# precursors (template_aatype [T,N], template_atom_positions [T,N,24,3], template_atom_mask [T,N,24]; O(T·N)) plus the per-token pseudo-beta /
# backbone masks the slot census reads, and each rank computes ITS OWN rows [T, R, N(, F)] of the four pair features at the item's entry
# (host_side_inputs -> template_pair_rows) with the stock's numpy statements applied to row slices — the same ufuncs, dtypes and reduction
# axes as data/template/template_utils.py:196-311, so the rows equal the dense features sliced to those rows bit for bit
# (tests/test_tp_template_rows.py). No rank ever holds [T, N, N, F]; the template embedder consumes the rows as before (template_tp).
TEMPLATE_FEATURIZER_MODULE = "protenix.data.template.template_featurizer"
TEMPLATE_DENSE_SITE = "as_protenix_dict"                                                # the `Templates` method that adds the dense pair features to the per-token ones (`as_data_dict`)
TEMPLATE_PRECURSOR_KEYS = ("template_atom_positions", "template_atom_mask")           # leave the item once the rows are born; template_aatype stays (the embedder's one-hot)


def template_token_precursors(aatype, atom_positions, atom_mask) -> list:
    """Per slot, the per-token quantities the pair statements read — O(T·N), every statement the stock's, token-local: masked positions
    (template_featurizer.py:590), pseudo-beta position + mask (`TemplateFeatures.pseudo_beta_fn`, template_utils.py:196-227), the backbone
    frame (C, CA, N by the featuriser's `RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX[aatype, 0]`; Gram-Schmidt with the additive 1e-6,
    template_utils.py:262-292) and its mask. numpy in (aatype [T,N] int, atom_positions [T,N,24,3] float32, atom_mask [T,N,24] bool), numpy out."""
    import importlib
    import numpy as np
    TF = importlib.import_module(TEMPLATE_FEATURIZER_MODULE).TemplateFeatures       # the stock's per-token statements (pseudo_beta_fn) ...
    table = importlib.import_module(TF.__module__).RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX  # ... and the backbone index table its unit-vector statement reads (template_utils)
    out = []
    for t in range(int(aatype.shape[0])):
        aa, mask = aatype[t], atom_mask[t]
        pos = atom_positions[t] * mask[..., None]
        pb_pos, pb_mask = TF.pseudo_beta_fn(aa, pos, mask)
        bb = table[aa, 0]
        idx = np.arange(aa.shape[0])
        c_pos, ca_pos, n_pos = pos[idx, bb[:, 0]], pos[idx, bb[:, 1]], pos[idx, bb[:, 2]]
        bb_mask = (mask[idx, bb[:, 0]] * mask[idx, bb[:, 1]] * mask[idx, bb[:, 2]]).astype(np.float32)
        eps = 1e-6
        v1 = c_pos - ca_pos
        v2 = n_pos - ca_pos
        e1 = v1 / (np.linalg.norm(v1, axis=-1, keepdims=True) + eps)
        e2 = v2 - np.sum(v2 * e1, axis=-1, keepdims=True) * e1
        e2 = e2 / (np.linalg.norm(e2, axis=-1, keepdims=True) + eps)
        e3 = np.cross(e1, e2)
        out.append({"pb_pos": pb_pos, "pb_mask": pb_mask, "ca_pos": ca_pos, "e1": e1, "e2": e2, "e3": e3, "bb_mask": bb_mask})
    return out


def template_token_masks(prec) -> dict:
    """The per-token pseudo-beta / backbone-frame masks [T, N] float32 of the precursors (the slot census's keys, templates.MASK_KEYS; the
    stock's per-token shapes, data/utils.py:1022-1023). Their outer products are the 2-D masks the pair rows carry."""
    import numpy as np
    return {"template_pseudo_beta_mask": np.stack([p["pb_mask"] for p in prec]), "template_backbone_frame_mask": np.stack([p["bb_mask"] for p in prec])}


def template_pair_rows(prec, g0: int, g1: int) -> dict:
    """The four template pair features of GLOBAL rows [g0, g1) against all N columns, [T, g1-g0, N(, F)] float32: the featuriser's dense
    statements (template_featurizer.py:595-618; `dgram_from_positions` template_utils.py:229-249, `compute_template_unit_vector` :252-311)
    with the row operand sliced — distogram one-hot `(d2 > lower) * (d2 < upper)` over the 39 squared edges times the 2-D pseudo-beta
    mask, unit vector `frame_i · (CA_j - CA_i)` normalised with the additive 1e-6 times the 2-D backbone mask, and the two masks themselves."""
    import numpy as np
    lower = np.square(np.linspace(3.25, 50.75, 39))                                    # DistogramFeaturesConfig defaults (template_featurizer.py:71-75)
    upper = np.concatenate([lower[1:], np.array([1e8], dtype=np.float32)], axis=-1)
    rows = slice(int(g0), int(g1))
    dg, pbm, uv, bbm = [], [], [], []
    for p in prec:
        dist2 = np.sum(np.square(np.expand_dims(p["pb_pos"][rows], axis=-2) - np.expand_dims(p["pb_pos"], axis=-3)), axis=-1, keepdims=True)
        dgram = (dist2 > lower).astype(np.float32) * (dist2 < upper).astype(np.float32)
        pb2 = p["pb_mask"][rows][:, None] * p["pb_mask"][None, :]
        dg.append(dgram * pb2[..., None])
        pbm.append(pb2)
        diff = p["ca_pos"][None, :, :] - p["ca_pos"][rows][:, None, :]
        u = np.stack([np.sum(p["e1"][rows][:, None, :] * diff, axis=-1), np.sum(p["e2"][rows][:, None, :] * diff, axis=-1),
                      np.sum(p["e3"][rows][:, None, :] * diff, axis=-1)], axis=-1)
        u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-6)
        bb2 = p["bb_mask"][rows][:, None] * p["bb_mask"][None, :]
        uv.append(u * bb2[..., None])
        bbm.append(bb2)
    return {"template_distogram": np.stack(dg), "template_pseudo_beta_mask": np.stack(pbm),
            "template_unit_vector": np.stack(uv), "template_backbone_frame_mask": np.stack(bbm)}


def templates_as_precursors(stock_fn):
    """install's replacement of the `Templates` dense statement (TEMPLATE_DENSE_SITE) for a P>1 rank process: the featuriser's per-token
    precursors (`as_data_dict`) plus the per-token census masks — never the dense [T, N, N(, F)] pair features (host_side_inputs bears this
    rank's rows from them)."""
    def as_precursor_dict(self):
        feats = self.as_data_dict()
        prec = template_token_precursors(feats["template_aatype"], feats["template_atom_positions"], feats["template_atom_mask"])
        feats.update(template_token_masks(prec))
        return feats
    as_precursor_dict.__wrapped__ = stock_fn
    return as_precursor_dict


TEMPLATE_PAIR_KEYS = ("template_distogram", "template_pseudo_beta_mask", "template_unit_vector", "template_backbone_frame_mask")
MSA_HOST_KEYS = ("msa", "has_deletion", "deletion_value")                             # the raw MSA features [S_msa, N] (protenix/data/utils.py:1012-1015; MSAModule.input_feature,
                                                                                        # pairformer.py:730-734) the MSA module subsamples per cycle: HOST-resident under ROWPAIR_MSA_HOST
                                                                                        # (opt_core.mem.rowpair.msa_host; the line exports rank0, ngpu.TP_EXPORTS)


def host_inputs() -> dict:
    """This rank's host-resident input features of the current item (host_side_inputs; `{}` for a caller without the entry hook). Keyed by
    rank so the threaded CPU ranks of the tests each see their own; a rank process holds one entry."""
    return (STATE.get("host_inputs") or {}).get(world()[1]) or {}


def msa_host_take(feats: dict) -> dict:
    """Under ROWPAIR_MSA_HOST (the line exports `rank0`) at P>1 the raw MSA features (MSA_HOST_KEYS, [S_msa, N] tensors) LEAVE `feats`:
    returned as {key: tensor}; {} when nothing is held (no mode, P == 1, an item without MSA tensors). No collective — rank 0 calls it on the
    item it featurised before the item travels (rowpair.replicate_inputs), so those tensors are never staged whole on a device by the item
    broadcast nor enter its digest."""
    import torch
    P, _rank = world()
    if C("MH").host_mode() is None or P <= 1:
        return {}
    return {k: feats.pop(k) for k in MSA_HOST_KEYS if torch.is_tensor(feats.get(k)) and feats[k].dim() >= 2}


def msa_host_hold(feats: dict, mine: Optional[dict] = None):
    """On every rank, around the item broadcast (rowpair.replicate_inputs calls it right after the item arrived): the raw MSA features leave
    this rank's `feats` (msa_host_take — or `mine`, the tensors the caller already took: rank 0 takes its own BEFORE its item travels; a rank
    that received rank 0's item holds none) and rank 0's shapes and dtypes travel by one object broadcast so every rank can declare what it
    never receives. Mode `rank0`: only rank 0's copy exists from here on (equality across ranks is by construction; the per-cycle selected
    rows reach every rank by a device broadcast, msa_host.rows_to_device). Mode `all`: msa_host_land makes every rank's HOST copy equal to
    rank 0's (msa_host.sync_host_features_: staged through the device in <= ROWPAIR_BCAST_CHUNK_GB pieces, never whole; a rank without the
    tensor ends with rank 0's). Unset: nothing is held (the stock placement: the features travel with the item and enter its digest).
    Returns the hold record (or None); _msa_host_place consumes it from STATE["msa_hold"][rank]."""
    MH, BC = C("MH"), C("BC")
    P, rank = world()
    md = MH.host_mode()
    if md is None or P <= 1:
        return None
    if mine is None:
        mine = msa_host_take(feats)
    meta = BC.broadcast_obj({k: (tuple(int(d) for d in v.shape), str(v.dtype).replace("torch.", "")) for k, v in mine.items()} if rank == 0 else None, src=0)
    hold = {"mode": md, "meta": dict(meta or {}), "mine": mine if rank == 0 else {}}
    STATE.setdefault("msa_hold", {})[rank] = hold
    return hold


def msa_host_land(feats: dict, hold, device=None) -> None:
    """After the broadcast and its proof (rowpair.replicate_inputs): rank 0's held raw MSA tensors go back into its item; mode `all`: every
    rank's host copy is made rank 0's (msa_host.sync_host_features_, chunked through `device`); mode `rank0`: ranks > 0 hold nothing (their
    rows arrive per cycle). _msa_host_place parks what this rank holds."""
    if not hold:
        return
    feats.update(hold["mine"])
    if hold["mode"] == "all" and hold["meta"]:
        C("MH").sync_host_features_(feats, list(hold["meta"]), device if device is not None else "cpu", src=0, mode="all")


def _msa_host_place(feats: dict, host: dict, facts: dict) -> None:
    """host_side_inputs' MSA part (ROWPAIR_MSA_HOST set): the raw MSA features present in this rank's item become PINNED host copies
    (msa_host.park_features -> HostTensor: rank 0's in mode rank0, every rank's in mode all; census msa_host_mode / msa_host_pinned_gib /
    msa_host_pageable_gib are the core's words) and LEAVE the item (the runner's to_device never moves them; runner/inference.py:226);
    `host["msa"]` = {mode, n_rows (S_msa of rank 0's item), meta {key: (shape, dtype)}, parked {key: HostTensor}} is what msa_module_tp reads —
    or {mode, absent: True} for an item without MSA features (the one state in which the MSA module leaves z unchanged under the lever).
    The host copies live for the item (the next item's entry replaces them)."""
    import torch
    MH = C("MH")
    P, rank = world()
    md = MH.host_mode()
    hold = (STATE.get("msa_hold") or {}).pop(rank, None)
    if md is None:
        return
    present = tuple(k for k in MSA_HOST_KEYS if torch.is_tensor(feats.get(k)) and feats[k].dim() >= 2)
    meta = dict(hold["meta"]) if hold else {k: (tuple(int(d) for d in feats[k].shape), str(feats[k].dtype).replace("torch.", "")) for k in present}
    if "msa" not in meta:                                                               # an item WITHOUT MSA features, stated explicitly: msa_module_tp takes the stock's
        host["msa"] = {"mode": md, "absent": True}                                      # `no msa -> z unchanged` exit (pairformer.py:846-851) only on this word
        facts["tp_msa_host"] = f"{md}:absent"
        return
    if md == "rank0" and rank != 0 and present:
        _refuse(f"msa_host rank0: rank {rank} holds raw MSA features {present} (rowpair.replicate_inputs drops them before the broadcast)")
    if (md == "all" or rank == 0) and tuple(meta) != present:
        _refuse(f"msa_host {md}: rank {rank} lacks raw MSA features {sorted(set(meta) - set(present))} to park")
    MH.park_features(feats, present, mode=md, row_dims=-2)                              # in place: pinned host copies (pageable named when page-locking fails); records msa_host_mode
    parked = {k: MH.HostTensor(feats.pop(k), name=k, row_dim=-2, log=False) for k in present}
    host["msa"] = {"mode": md, "n_rows": int(meta["msa"][0][-2]), "meta": meta, "parked": parked}
    facts["tp_host_inputs"] += f",msa:{md}"
    facts["tp_msa_host"] = f"{md}:S={int(meta['msa'][0][-2])}:keys={len(meta)}:parked={len(parked)}"


def _host(t, name: str):
    """A host-resident input feature: PINNED host through the core (msa_host.to_host — a refused page-lock is answered with pageable host
    memory BY NAME on stderr, never silently; the bytes land in the schedule census msa_host_pinned_gib / msa_host_pageable_gib)."""
    return C("MH").to_host(t.contiguous(), name=name)


def host_side_inputs(feats: dict, configs=None) -> dict:
    """At an item's entry on every rank of a P>1 launch, BEFORE the runner moves the features to the device: the pair-shaped INPUT features
    never become device-resident whole. `token_bonds` [N, N] stays on pinned host (msa_host.to_host: a refused page-lock is pageable host, named; its rows are copied per row
    block where the trunk initialises z); the template pair features reach this rank as ROWS [T, R, N(, F)] only (pinned; copied per row block
    by the template embedder): born here from the featuriser's per-token precursors (template_pair_rows — install's featuriser site makes it
    emit precursors, never the dense [T, N, N(, F)] features), or, for an item that still carries the dense features, sliced to this rank's
    rows on the host (template.slice_template_inputs_to_rows); under ROWPAIR_MSA_HOST (the line exports `rank0`) the raw MSA features [S_msa, N] stay on pinned host too — rank 0's copy only
    in mode rank0, every rank's in mode all (_msa_host_place; only the per-cycle subsampled rows reach the device, msa_module_tp). The keys
    leave `feats`; the per-token features stay (replicated by design). Returns the census facts (`tp_host_inputs`, `tp_msa_host`)."""
    TE = C("TE")
    if configs is not None:
        _bind_model(None, None, configs=configs)
    N = int(feats["token_index"].shape[-1])
    lay = layout_for(N)
    host: Dict[str, object] = {"N": N}
    facts = {"tp_host_inputs": "token_bonds"}
    if "token_bonds" in feats:
        host["token_bonds"] = _host(feats.pop("token_bonds"), "token_bonds")
    tkeys = tuple(k for k in TEMPLATE_PAIR_KEYS if k in feats and hasattr(feats[k], "dim") and feats[k].dim() >= 3)
    if tkeys:
        dense = {k: (feats[k] if feats[k].dim() == 4 else feats[k].unsqueeze(-1)) for k in tkeys}     # the core slices [T, N, N, F]: masks get F=1
        sl = TE.slice_template_inputs_to_rows(dense, lay, tkeys)
        host["template_rows"] = {k: _host(sl[k] if feats[k].dim() == 4 else sl[k].squeeze(-1), f"template_rows.{k}") for k in tkeys}
        for k in tkeys:
            feats.pop(k)
        del dense, sl
        facts["tp_host_inputs"] = "token_bonds,template_pair_rows"
        COUNTS["template_rows"] += 1
    elif "template_aatype" in feats and all(k in feats for k in TEMPLATE_PRECURSOR_KEYS):   # row_born: the featuriser emitted per-token precursors (install's site) —
        import torch                                                                    # this rank's rows of the four pair features are computed here, on the host
        arrs = {k: (feats[k].detach().cpu().numpy() if torch.is_tensor(feats[k]) else feats[k]) for k in ("template_aatype",) + TEMPLATE_PRECURSOR_KEYS}
        prec = template_token_precursors(arrs["template_aatype"], arrs["template_atom_positions"], arrs["template_atom_mask"])
        born = template_pair_rows(prec, lay.r0, lay.r1)
        host["template_rows"] = {k: _host(torch.from_numpy(born[k]), f"template_rows.{k}") for k in TEMPLATE_PAIR_KEYS}
        for k in TEMPLATE_PRECURSOR_KEYS:                                               # the atom coordinates leave the item; template_aatype and the per-token masks stay (replicated by design)
            feats.pop(k)
        del arrs, prec, born
        C("EV").record_schedule(templ_pair_keys=",".join(TEMPLATE_PAIR_KEYS), templ_pair_rows=lay.R)   # the schedule's words for the rows this rank holds (the dense-slice route records the same two through the core)
        facts["tp_host_inputs"] = "token_bonds,template_pair_rows"
        COUNTS["template_rows"] += 1
    if "constraint_feature" in feats and not constraint_embedders_enabled(configs):   # every constraint embedder off (the stock configs): the
        feats.pop("constraint_feature")                                                 # pair-shaped constraint features are read by nothing — dropped before H2D
        facts["tp_host_inputs"] += ",constraint_feature_dropped"
    _msa_host_place(feats, host, facts)                                                 # ROWPAIR_MSA_HOST: the raw MSA features parked on pinned host (rank 0's only in mode rank0)
    STATE.setdefault("host_inputs", {})[world()[1]] = host
    C("EV").record_schedule(**facts)
    return facts


def undo() -> None:
    while STATE["undo"]:
        STATE["undo"].pop()()
    STATE["installed"] = False
    dit_rollout_clear()
    for k in ("model", "layout", "host_inputs", "msa_hold", "host_facts", "weights_guarded", "kernel", "chunk"):
        STATE.pop(k, None)


def record() -> dict:
    lay = STATE.get("layout")
    return {"installed": bool(STATE["installed"]), "P": STATE["P"], "rank": STATE["rank"], "counts": dict(COUNTS),
            "replicated_bytes": dict(REPLICATED), "parks": {k: dict(v) for k, v in PARKS.items()},
            "rows": None if lay is None else {"r0": lay.r0, "r1": lay.r1, "N": lay.N, "align": getattr(lay, "align", None)}}
