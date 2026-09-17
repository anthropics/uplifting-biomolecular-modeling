"""The multi-GPU line's rank-side routing: the carried PTX_TP unit keeps its engine glue (trunk / MSA / template / diffusion /
confidence seams, launcher) and imports its ENGINE-FREE layers — the row layout and collectives (``ptx_tp.dist``), the block
reducer of the confidence head (``ptx_tp.blockreduce``), the feature broadcast (``ptx_tp.bcast``) and the sharded TriMul
contractions (``ptx_tp.contract``) — from the shared core ``opt_core.mem.rowpair`` at runtime: every public function / class of
those carried modules named in :data:`ROUTES` is REBOUND to the core's object when the carried module is imported in a rank
(``opt_core.autoload.patch_attr_at_import``, one hook per carried module: the carried body runs, then the names are replaced; the carried bytes on disk are
untouched and sha-manifested). What stays the carried module's own is named: :data:`CARRIED` (``contract.rowsplit_attention``
— the core's family form ``rowpair.triatt.triatt_starting`` has another signature) and the constants.

A second table, :data:`BIND`, uses the same hook for the carried unit's ENGINE seams whose replacement needs Protenix glue between the
carried call site and the core driver: the named public functions of ``ptx_tp.trunk`` / ``ptx_tp.template`` / ``ptx_tp.diffusion`` are
rebound to the kit's BINDING modules ``protenix_opt.tp_bind.*`` (engine attribute paths + the stock per-(i, j) statements on rows; every
schedule, collective and checkpoint is one core driver call — ``run_trunk_sharded``, ``template_embed_rows``, ``heads.sym_logit_rows``,
``diffusion.pair_cond_rows`` / ``dit_block_sharded`` / ``pair_band_rows``). Evidence: ``[protenix-opt] TP-BIND <carried module> -> <binding
module> names=<n>`` per bound module.

Installed ONLY inside the line's rank processes (``tp.rank_env`` sets :data:`ENV` = :data:`WORD`; the kit's interpreter-start hook
calls :func:`install`); a ``--n_gpu 1`` run installs nothing. The core's ``dist.init_from_env`` reads the ``ROWPAIR_*`` rank
environment; the carried launcher starts torchrun ranks, so :func:`bridge_env` derives ``ROWPAIR_RANK / _WORLD / _LOCAL_RANK /
_ADDR / _PORT`` from torchrun's names and points ``ROWPAIR_STORE`` at a node-local file store of this run (one node; never a
volume). Evidence: each rank prints ``[protenix-opt] TP-ROUTE <carried module> -> <core module> names=<n>`` when the rebinding
happens (``tp.events`` reads those lines: a rank whose route lines are short of :data:`ROUTES` makes the run not ok)."""
import os
import sys

ENV, WORD = "PROTENIX_OPT_TP_ROUTE", "rowpair"          # set by tp.rank_env in every rank; declared in _autoload
CORE_PKG = "opt_core.mem.rowpair"
TAG = "protenix-opt"

# carried module -> (core module, names rebound). Fixed by the carried unit's public names and the core's rowpair API; tests/test_tp_route.py
# re-derives both sides (every public function/class of the carried module is here or in CARRIED; every name exists in the core module).
ROUTES = {
    "ptx_tp.dist": ("opt_core.mem.rowpair.dist", (
        "init_from_env", "is_dist", "barrier", "Layout", "zmeta", "all_gather_rows", "gather_rows_to_rank0", "alltoall_window",
        "transpose_shards", "ring_blocks", "allreduce_checksum", "broadcast_obj", "world", "allreduce_", "zlen", "checksum", "zclone",
        "zalloc_like", "zblocks", "gather_cat_to_rank0", "transpose_blocks")),
    "ptx_tp.blockreduce": ("opt_core.mem.rowpair.confidence", (
        "tm_bin_centers", "tm_d0", "tm_bin_weight", "expected_value_rows", "Context", "ChainIndex", "RowBlockReducer", "gpde_from_full",
        "max_row_chunk", "gather_cat_to_rank0")),
    "ptx_tp.bcast": ("opt_core.mem.rowpair.bcast", ("comm_device", "broadcast_tensordict", "tensordict_checksum", "assert_replicated")),
    "ptx_tp.contract": ("opt_core.mem.rowpair.trimul", (
        "gemm_a_layout", "gemm_b_layout", "max_rows_per_launch", "default_rows_a", "default_rows_b", "ContractStats",
        "rowshard_outer_contract", "colshard_inner_contract")),
}
# names routed to another core module than their carried module's counterpart
ELSEWHERE = {("ptx_tp.blockreduce", "gather_cat_to_rank0"): "opt_core.mem.rowpair.dist"}
# public functions of the routed carried modules that stay the carried module's own, with the reason
CARRIED = {("ptx_tp.contract", "rowsplit_attention"): "the core's family form is rowpair.triatt.triatt_starting (another signature)"}
BIND_PKG = "protenix_opt.tp_bind"
# carried ENGINE-seam module -> (kit binding module, names rebound): the binding holds Protenix's attribute paths and row statements and makes
# ONE core driver call per seam; tests/test_tp_rebase_route.py checks both sides by source (names exist; the binding imports the core only).
BIND = {
    "ptx_tp.trunk": ("protenix_opt.tp_bind.trunk", (
        "get_pairformer_output_tp", "zinit_rows", "zinit_rows_block", "recycle_rows", "relp_rows", "relp_block", "contact_probs_rows")),
    "ptx_tp.template": ("protenix_opt.tp_bind.template", ("tp_template_embedder", "template_embedder_is_active")),
    "ptx_tp.diffusion": ("protenix_opt.tp_bind.diffusion", ("tp_prepare_cache", "tp_sample_diffusion", "report")),
    "ptx_tp.pairformer": ("protenix_opt.tp_bind.pairstack", ("tp_pairformer_block", "tp_pairformer_stack")),     # the trunk's 48 blocks and the confidence head's 4:
    "ptx_tp.msa": ("protenix_opt.tp_bind.pairstack", ("tp_pair_stack_block",)),                                   # the core pair-block driver with its row kernels; the MSA
}                                                                                                                  # module's pair stack likewise (its OPM / PWA rows stay the unit's)
BOUND = tuple(m.split(".", 1)[1] for m in BIND)          # ('trunk', 'template', 'diffusion') — the ACTIVE line's bound= value
BIND_LINE = "TP-BIND"
ROUTED = tuple(m.split(".", 1)[1] for m in ROUTES)      # ('dist', 'blockreduce', 'bcast', 'contract') — the ACTIVE line's routed= value
# the unit's launcher statement served by the kit (not a core mechanism, so not in ROUTES / the routed= census): torchrun's rendezvous
# pinned to the loopback address — the launcher process rebinds ptx_tp.launch_core.torchrun_cmd to protenix_opt.tp_bind.launch
LAUNCH_SITE = ("ptx_tp.launch_core", "torchrun_cmd", "protenix_opt.tp_bind.launch")
LINE = "TP-ROUTE"

_announced = set()
_patches = []
_bind_patches = []


def bridge_env(environ=None):
    """The core's rank environment from torchrun's, in place: ROWPAIR_RANK/_WORLD/_LOCAL_RANK/_ADDR/_PORT from RANK/WORLD_SIZE/
    LOCAL_RANK/MASTER_ADDR/MASTER_PORT when the former are absent, and ROWPAIR_STORE = a node-local file store of this run (keyed by
    torchrun's run id, else addr:port) unless already set. Returns the names it set. A process without WORLD_SIZE is left alone."""
    env = os.environ if environ is None else environ
    if "WORLD_SIZE" not in env:
        return {}
    pairs = (("ROWPAIR_RANK", "RANK"), ("ROWPAIR_WORLD", "WORLD_SIZE"), ("ROWPAIR_LOCAL_RANK", "LOCAL_RANK"),
             ("ROWPAIR_ADDR", "MASTER_ADDR"), ("ROWPAIR_PORT", "MASTER_PORT"))
    set_ = {}
    for dst, src in pairs:
        if dst not in env and src in env:
            env[dst] = set_[dst] = env[src]
    if not env.get("ROWPAIR_STORE", "").strip():
        run = env.get("TORCHELASTIC_RUN_ID") or f"{env.get('MASTER_ADDR', '127.0.0.1')}_{env.get('MASTER_PORT', '29500')}"
        run = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run)
        d = os.path.join(store_parent(env), run)
        os.makedirs(d, mode=0o700, exist_ok=True)
        env["ROWPAIR_STORE"] = set_["ROWPAIR_STORE"] = os.path.join(d, "c10d_store")
    return set_


class StoreDirRefused(RuntimeError):
    """The per-user directory for the ranks' file store is not this user's own private directory — refused by name (every rank sees the same
    directory, so every rank refuses alike; a private directory picked per process would split the rendezvous)."""


def store_parent(environ=None) -> str:
    """The parent of every run's file-store directory when ROWPAIR_STORE is not set: ``<tmp>/protenix_opt_rowpair-uid<uid>``, made here with
    mode 0700 or made so by an earlier run of this user. The ranks exchange rendezvous data through the store, so a directory another account
    could have planted or written is never used: one that is a symbolic link, belongs to another uid, or carries a group/other write bit raises
    :class:`StoreDirRefused` naming it, the reason and the fix (remove it, or set ROWPAIR_STORE to a path of your own)."""
    import stat
    import tempfile
    env = os.environ if environ is None else environ
    tmp = env.get("TMPDIR") or tempfile.gettempdir()
    d = os.path.join(tmp, "protenix_opt_rowpair-uid%d" % os.geteuid())
    if not os.path.lexists(d):
        try:
            os.mkdir(d, 0o700)
        except FileExistsError:                                       # another rank of this launch made it first
            pass
    st = os.lstat(d)
    why = None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        why = "is a symbolic link or not a directory"
    elif st.st_uid != os.geteuid():
        why = f"belongs to uid {st.st_uid}, not to this process (uid {os.geteuid()})"
    elif st.st_mode & 0o022:
        why = f"is writable by group or other (mode {stat.S_IMODE(st.st_mode):04o})"
    if why is not None:
        raise StoreDirRefused(f"[protenix-opt] rank file store: REFUSED {d}: it {why}; fix: remove it, or set ROWPAIR_STORE to a path of your own")
    return d


def _core_object(carried, name, table=None):
    import importlib
    table = ROUTES if table is None else table
    return getattr(importlib.import_module(ELSEWHERE.get((carried, name), table[carried][0])), name)


def _factory(carried, table=None, line=LINE):
    """The wrapper factory of the ONE patch per carried module (site = its first routed name): rebinds every other routed name of the
    module to the core's object as well (one import hook per module — the core's seam nests hooks per site, so several armed hooks on
    one not-yet-imported module would ask each other for the module's spec), prints the evidence line once, returns the first name's
    core object."""
    table = ROUTES if table is None else table
    core_module, names = table[carried]

    def make(_original, _m=carried):
        mod = sys.modules[_m]
        for n in names[1:]:
            setattr(mod, n, _core_object(_m, n, table))
        obj = _core_object(_m, names[0], table)
        if _m not in _announced:
            _announced.add(_m)
            core = sys.modules.get(core_module)
            sys.stderr.write(f"[{TAG}] {line} {_m} -> {core_module} names={len(names)} core={getattr(core, '__file__', '?')}\n")
            sys.stderr.flush()
        return obj
    return make


def install(environ=None):
    """Arm the rebinding of every name in :data:`ROUTES` (fires at each carried module's import; immediate for a module already
    imported) after :func:`bridge_env`. Idempotent per process. Returns the armed patches."""
    global _patches
    if _patches:
        return _patches
    bridge_env(environ)
    from . import _core  # noqa: F401 — the pinned core importable in a PYTHONPATH-only rank
    from opt_core.autoload import patch_attr_at_import
    env = os.environ if environ is None else environ
    if (env.get("ROWPAIR_RANK_THREADS") or "").strip():                  # the per-rank CPU-thread cap the line exports (tp.RANK_ENV_DEFAULTS):
        from opt_core.mem.rowpair.dist import apply_rank_threads          # cores // ranks under `auto`, census rank_threads=<n>; unset = untouched
        apply_rank_threads(env.get("ROWPAIR_RANK_THREADS"), local_world=int(env.get("PTX_TP") or env.get("LOCAL_WORLD_SIZE") or 1))
    out = []
    for carried, (core_module, names) in ROUTES.items():
        out.append(patch_attr_at_import(carried, names[0], _factory(carried), tag=TAG, name=f"tp_route:{carried}"))
    out.append(patch_attr_at_import(LAUNCH_SITE[0], LAUNCH_SITE[1], _launch_factory, tag=TAG, name="tp_route:launch"))
    from .tp_bind import census as _census                                  # the core census's stage marks on the unit's phase log + seam returns
    out.extend(_census.install())
    from .tp_bind import layout_guard as _lg                              # the unit's row layout must shard on a supported grid with no empty rank, else refused by name
    out.append(patch_attr_at_import(_lg.SITE[0], _lg.SITE[1], _lg.guard_factory, tag=TAG, name="tp_route:layout_guard"))
    for carried, (bind_module, names) in BIND.items():                     # the engine seams -> the kit's bindings on the core drivers
        _bind_patches.append(patch_attr_at_import(carried, names[0], _factory(carried, BIND, BIND_LINE), tag=TAG, name=f"tp_bind:{carried}"))
    _patches = out
    return out


def _launch_factory(_original):
    """The launcher's torchrun argv builder -> :func:`protenix_opt.tp_bind.launch.torchrun_cmd` (loopback rendezvous); prints the evidence
    line once in the process where the unit's launcher module is imported."""
    import importlib
    mod = importlib.import_module(LAUNCH_SITE[2])
    if LAUNCH_SITE[0] not in _announced:
        _announced.add(LAUNCH_SITE[0])
        sys.stderr.write(f"[{TAG}] TP-ROUTE-LAUNCH {LAUNCH_SITE[0]}.{LAUNCH_SITE[1]} -> {LAUNCH_SITE[2]} ({mod.NAME})\n"); sys.stderr.flush()
    return getattr(mod, LAUNCH_SITE[1])


def selected(environ=None):
    env = os.environ if environ is None else environ
    return (env.get(ENV) or "").strip() == WORD


def route_lines(text):
    """The carried modules a rank log's TP-ROUTE lines name (in order of appearance)."""
    import re
    return [m.group(1) for m in re.finditer(rf"\[{TAG}\] {LINE} (ptx_tp\.\w+) -> ", text)]


def bind_lines(text):
    """The carried modules a rank log's TP-BIND lines name (in order of appearance)."""
    import re
    return [m.group(1) for m in re.finditer(rf"\[{TAG}\] {BIND_LINE} (ptx_tp\.\w+) -> ", text)]
