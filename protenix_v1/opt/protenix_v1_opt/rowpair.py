"""`--mode big --n_gpu P>1`: the PROCESS half of protenix 1.1.0's tensor-parallel memory mode over `opt_core.mem.rowpair` — the P rank
processes, the group, rank-0 feature replication, rank-0-only output writing, the RNG guard, the mode's evidence lines. The STATEMENT half
(what is row-sharded and how every protenix module is bound onto the core's callables) is `tp.py`: the pair representation is row-sharded
end to end — born as rows in the trunk, through recycling, templates, the MSA module, the Pairformer, the distogram, the confidence head and
diffusion conditioning / the diffusion transformer — and nothing N x N x c is ever whole on a rank (tp.py's docstring lists what stays
replicated by design and what is refused by name).

Rank 0 featurises, every rank computes on its item (the core's `rank0_bcast` data form, opt_core.mem.rowpair.rankdata): the stock dataset's
`__getitem__` runs upstream's featurisation (`process_one`: CCD/RDKit reference features, MSA pairing, template search) on RANK 0 ONLY and
the item reaches every rank by `replicate_inputs` INSIDE that call — before the stock loop reads its sizes — through the core's
`broadcast_features` (ranks > 0 wait at a store rendezvous with no collective pending while rank 0 featurises, receive the tensors and the
pickled non-tensor leaves, and adopt rank 0's host RNG state, so every later draw is the one all ranks would have made); ranks > 0 keep the
cheap input parse (the JSON is read on every rank) and build no features. Replication is proven, not assumed: every rank digests the item it
holds (`rankdata.feature_digest`) and differing digests refuse the run on every rank (`refused: feats_ranks_differ`); the torch RNG state is
checksummed across ranks before every `sample_diffusion` call (`diffusion_rng_state`); the MSA module's subsample draw is guarded the same
way inside the trunk driver. Each is a RowpairRefused by name on a mismatch. A featurisation error stays stock's per-item event: the error
item reaches every rank, all ranks skip it in step, the ITEMS census counts it failed. Rank 0 alone writes outputs
(`runner.dumper.DataDumper.dump` is a no-op elsewhere).

Numerics class: tier 2 (fast-class) against `--n_gpu 1` by construction — the triangle multiplication is the core's torch statement on rows
where the single-GPU line runs a fused kernel, reductions run over per-rank row blocks, the triangle-attention kernel of the run
(`--triatt_kernel`) serves this rank's query rows; per-element arithmetic equals the dense statement everywhere else. `--n_gpu 1` never
enters tp.py (nothing is installed outside a rank process of a P>1 launch: RowpairRefused by name).

Process shape: `pred --mode big --n_gpu P` (the parent) admits P (ngpu.py), then `launch.run_rank_processes` runs P copies of
`python -m protenix_v1_opt pred <same arguments>` under the rank environment (`ROWPAIR_RANK/WORLD/...`, one visible device per rank, and
ONE `PYTHONHASHSEED` for all P interpreters — an exported integer value is kept, unset / empty / `random` give the core's default `0`; the launcher prints
`[protenix-v1-opt] RANKENV hashseed=<v> source=default|inherited ranks=P` once per launch, opt_core.mem.rowpair.rankdata — so a
featurisation statement ordered by `str` hash yields the same bytes on every rank);
each rank activates the mode as any single-GPU run does, joins the group (`init_rank`, NCCL, bounded timeout) and installs the patches
(`install` -> tp.install); any rank that fails ends every rank (`launch.RankFailed`; the parent prints the ROWPAIR event line and exits
non-zero); the parent relays rank 0's transcript, prints the EXIT line and returns rank 0's exit code.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Sequence

from opt_core.mem.rowpair import dist as D
from opt_core.mem.rowpair import evidence as EV
from opt_core.mem.rowpair import launch as L
from opt_core.mem.rowpair import rankdata as RKD

from . import report as R
from . import tp as TP

from opt_core.mem import rowpair as RP

STRATEGY = RP.LEVER                  # the family's canonical strategy id of the row-sharded pair stack (one producer: opt_core.mem.rowpair)
LEVER = "rowpair"
MODEL_MODULE = "protenix.model.protenix"            # binds `sample_diffusion` (from protenix.model.generator) as a module global: the name the trunk calls
DUMPER_MODULE = "runner.dumper"
DATASET_MODULE = "protenix.data.inference.infer_dataloader"   # InferenceDataset.__getitem__: the one caller of process_one per item x seed (infer_dataloader.py:278-293), in the rank process itself (num_workers 0)
DIST_MODULE = "protenix.utils.distributed"          # DIST_WRAPPER: the world the stock sampler splits items over (torchrun's RANK / WORLD_SIZE — 1 in every rank process: the launcher never exports them)
DATA_FORM = RKD.check_data_form("rank0_bcast")      # the core's data-form token of this line (rankdata.DATA_FORMS): rank 0 featurises, every rank receives its item; census `data_form=` on the rowpair LEVER line
ITEM_ERROR = "item_error"                           # the control word rank 0 stores for an item whose featurisation stock caught (infer_dataloader.py:288-290), followed by the error's first line: nothing is broadcast, every rank skips the item in step
REFUSE_UNREPLICATED = "rowpair_item_unreplicated: the item entering predict ({got}) is not the one item_of_rank0 replicated last on this rank ({want}) — the dataset item did not come through the line's __getitem__"
NCCL_TIMEOUT_S = 300.0                # the collective timeout of the ×P line, seconds: passed to the launcher (every rank's ROWPAIR_NCCL_TIMEOUT_S) and to a rank's own init — a peer that died mid-collective is an error within it, not a hang
STATE = {"P": 1, "rank": 0, "device": None, "group": False, "installed": False, "items": 0, "sites": []}   # items: dataset item fetches through the line (items x seeds, error items included; LEVER `items=`)


def _refuse(msg: str):
    raise D.RowpairRefused(msg)


# ----------------------------------------------------------------------------------------------------------------- rank environment
def world_size() -> int:
    """P of this process from the rank environment the launcher sets (1 in the parent and in every single-GPU run)."""
    return L.world_size()


def rank() -> int:
    return L.rank()


def is_rank_process() -> bool:
    return world_size() > 1


def is_output_rank() -> bool:
    return L.is_output_rank()


# ----------------------------------------------------------------------------------------------------------------- parent side
def rank_argv(kit_args: Sequence[str]) -> List[str]:
    """The command line every rank runs: this package's `pred` with the parent's own arguments."""
    return [sys.executable, "-m", "protenix_v1_opt", "pred", *kit_args]


def rank_log(out_dir: str, rank: int) -> str:
    """<out_dir>/rowpair/rank<r>.log — a rank's transcript (launch.run_rank_processes' log_dir naming)."""
    return os.path.join(out_dir, "rowpair", f"rank{int(rank)}.log")


def run_ranks(n_gpu: int, kit_args: Sequence[str], out_dir: str, on_line: Optional[Callable[[str], None]] = None) -> List[dict]:
    """Run P rank processes of `pred` (launch.run_rank_processes: rank environment, one visible device per rank, one `PYTHONHASHSEED` for every
    rank, fail-fast); the base environment is this process's own carrying the kit's line tag (launch.ENV_TAG), so the launcher's once-per-launch
    line reads `[protenix-v1-opt] RANKENV hashseed=<v> source=default|inherited ranks=P`; transcripts in <out_dir>/rowpair/rank<r>.log, rank 0's
    relayed through `on_line`. Returns the per-rank records; raises launch.RankFailed."""
    log_dir = os.path.join(out_dir, "rowpair")
    os.makedirs(log_dir, exist_ok=True)
    return L.run_rank_processes(n_gpu, argv_of=lambda r: rank_argv(kit_args), mode="big", env=dict(os.environ, **{L.ENV_TAG: R.TAG}),
                                log_dir=log_dir, isolate_devices=True, nccl_timeout_s=NCCL_TIMEOUT_S,
                                on_line=on_line or (lambda line: R.log(line.rstrip("\n"))))


def rank_failed_line(exc) -> str:
    return EV.rank_failed_line(R.PREFIX.strip("[]"), exc)


# ----------------------------------------------------------------------------------------------------------------- rank side
def init_rank(timeout_s: float = NCCL_TIMEOUT_S) -> dict:
    """Join the group from the rank environment (NCCL; the one visible device made current first). Idempotent."""
    if STATE["group"]:
        return dict(STATE)
    P, r, device = D.init_from_env(backend=None, timeout_s=timeout_s, device_index=0)
    STATE.update(P=int(P), rank=int(r), device=device, group=True)
    return dict(STATE)


REFUSE_INSTALL_P1 = "rowpair.install: refused at n_gpu=1 — the adapter installs nothing outside a rank process of a P>1 launch (the engine's single-GPU statement runs unchanged)"


SYNC_ENV = "ROWPAIR_DIFF_NOISE_SYNC"                      # the core's replicated-draw sync switch (opt_core.mem.rowpair.diffusion): an explicit value is the named opt-in


def sync_policy(det: bool, environ=None) -> dict:
    """The replicated-tensor sync policy of a rank process, by determinism level: `--det 1` -> guard (strict bit-equality: a mismatch is a real
    defect and refuses by name); `--det 0` -> bcast (rank 0 authoritative at every defined sync point — the diffusion RNG state at sampler
    entry, the per-denoiser-call state and update, the MSA sample, the MC-dropout decision: replicated kernels are not bit-reproducible run-to-run at det 0). An
    explicit ROWPAIR_DIFF_NOISE_SYNC=<guard|bcast> is the named opt-in over either default. Printed on the LEVER line (noise_sync=, det=)."""
    environ = os.environ if environ is None else environ
    explicit = (environ.get(SYNC_ENV) or "").strip().lower()
    if explicit in ("guard", "bcast"):
        return {"mode": explicit, "det": bool(det), "source": SYNC_ENV}
    if explicit:
        _refuse(f"{SYNC_ENV}={explicit!r} is not one of guard|bcast")
    return {"mode": "guard" if det else "bcast", "det": bool(det), "source": "det"}


def install(det: bool = False, on_item_error: Optional[Callable[[dict, str], None]] = None) -> bool:
    """Install the row-sharded statements (tp.install: every protenix site tp.py names), the replicated-tensor sync policy of this determinism
    level (sync_policy), rank-0 featurisation (the stock dataset's `__getitem__`: item_of_rank0; `on_item_error(data, message)` is told of an
    item whose featurisation failed, on every rank — the activation's ITEMS census), the diffusion sampler's RNG-state sync + rank-0
    coordinates and the rank-0 output writer — ONLY inside a rank process of a P>1 launch: at n_gpu=1 nothing is wrapped (RowpairRefused by
    name). Idempotent; True when in place."""
    import importlib
    if not is_rank_process():
        _refuse(REFUSE_INSTALL_P1)
    if STATE["installed"]:
        return True
    TP.SYNC.update(sync_policy(det))
    STATE["det"] = bool(det)
    STATE["on_item_error"] = on_item_error
    TP.C("EV").record_schedule(noise_sync=TP.SYNC["mode"], noise_sync_source=TP.SYNC["source"], det=int(bool(det)), data_form=DATA_FORM)
    STATE["sites"] = TP.install(STATE["P"], STATE["rank"])
    ds = importlib.import_module(DATASET_MODULE)                    # rank 0 featurises, every rank receives its item — inside the stock dataset's __getitem__, before the stock loop reads the item's sizes
    gorig = ds.InferenceDataset.__getitem__
    if not getattr(gorig, "__rowpair__", False):
        def __getitem__(self, index, _orig=gorig):
            return item_of_rank0(self, index, _orig)
        __getitem__.__rowpair__ = True
        __getitem__.__wrapped__ = gorig
        ds.InferenceDataset.__getitem__ = __getitem__
    pm = importlib.import_module(MODEL_MODULE)                      # the diffusion sampler's RNG guard: every rank must draw the same noise
    sorig = pm.sample_diffusion
    if not getattr(sorig, "__rowpair__", False):
        def sample_diffusion(*a, _orig=sorig, **kw):
            sync_rng_from_rank0("diffusion_rng_state")                   # entry: rank 0's CPU + CUDA generator state REPLACES every rank's (the stock's noise draws are then identical by construction)
            try:
                out = _orig(*a, **kw)
            finally:
                TP.dit_rollout_clear()                                   # the roll-out's cached DiT bias rows leave the device before the confidence stage
            return coordinates_of_rank0(out)                             # exit: rank 0's sampled coordinates on every rank; the cross-rank spread is computed, printed, bounded by name
        sample_diffusion.__rowpair__ = True
        sample_diffusion.__wrapped__ = sorig
        pm.sample_diffusion = sample_diffusion
    dm = importlib.import_module(DUMPER_MODULE)
    dorig = dm.DataDumper.dump
    if not getattr(dorig, "__rowpair__", False):
        def dump(self, *a, _orig=dorig, **kw):
            if STATE["P"] > 1 and not is_output_rank():
                return None
            return _orig(self, *a, **kw)
        dump.__rowpair__ = True
        dump.__wrapped__ = dorig
        dm.DataDumper.dump = dump
    STATE["installed"] = True
    return True


REFUSE_LOADER_WORKER = "rowpair rank0 featurisation: refused — the dataset item is fetched in a DataLoader worker process (num_workers > 0), outside the rank's group; the line featurises in the rank process itself (stock's num_workers 0)"
REFUSE_SAMPLER_WORLD = "rowpair rank0 featurisation: refused — the stock sampler's world is {world} (RANK/WORLD_SIZE set): the ranks would iterate different items; the line's ranks each iterate every item (world 1)"


def _assert_item_pairing() -> None:
    """The k-th `__getitem__` call is the same item on every rank: fetched in the rank process (no loader worker) by a sampler of world 1 (every
    rank iterates every index in file order). Either failing is refused by name before any collective."""
    import importlib
    from torch.utils.data import get_worker_info
    if get_worker_info() is not None:
        _refuse(REFUSE_LOADER_WORKER)
    w = int(getattr(importlib.import_module(DIST_MODULE).DIST_WRAPPER, "world_size", 1) or 1)
    if w != 1:
        _refuse(REFUSE_SAMPLER_WORLD.format(world=w))


def item_of_rank0(dataset, index: int, stock_getitem):
    """The stock dataset's `__getitem__` under the line (installed on every rank of a P>1 launch, rowpair.install): rank 0 runs the stock
    statement — upstream's featurisation of item `index`, its per-item error capture included — and every rank returns rank 0's
    `(data, atom_array, error_message)` through replicate_inputs; ranks > 0 build no features (their `atom_array` is None: only rank 0 writes
    outputs). The input parse stays per rank (`dataset.inputs`, read at construction on every rank), so every rank names the item itself.
    A refusal here (the pairing guards, `feats_rank0_failed`, `feats_ranks_differ`, `feats_bcast_malformed`, a rendezvous timeout) is
    raised where upstream's per-JSON handler would swallow it (runner/batch_inference.py:545-549 around the loader loop,
    runner/inference.py:453): it is logged by name (`ROWPAIR event=refused`), the item is counted failed (`on_item_error`) and the rank
    ends with SystemExit(EXIT_NOT_ACTIVE) — past the loader, upstream's handler and click, to the verb's EXIT line, non-zero on every rank."""
    if STATE["P"] <= 1:                                          # never installed at P == 1: the stock statement as it is
        return stock_getitem(dataset, index)
    name = str(dataset.inputs[index]["name"])                   # the per-rank input parse: every rank names the item itself
    try:
        _assert_item_pairing()
        return replicate_inputs(lambda: stock_getitem(dataset, index), name, index)
    except D.RowpairRefused as e:
        R.log(f"{R.PREFIX} ROWPAIR event=refused rank={TP.world()[1]} item={name} reason={R._token(e)}")
        cb = STATE.get("on_item_error")
        if cb is not None:
            cb({"sample_name": name, "sample_index": index}, str(e))
        raise SystemExit(R.EXIT_NOT_ACTIVE) from e


def replicate_inputs(featurise: Callable[[], tuple], name: str, index: int):
    """At an item's entry (item_of_rank0) on every rank: RANK 0 ALONE calls `featurise()` — the stock triple `(data, atom_array,
    error_message)` of item `index` — and every rank returns the triple holding rank 0's `data` (feature dict, sizes, names; `atom_array` stays
    rank 0's own, None elsewhere). The transport is the core's `rankdata.broadcast_features` (store rendezvous first — ranks > 0 wait outside any collective
    while rank 0 featurises —, meta, tensors, pickled non-tensor leaves, rank 0's host RNG state adopted by every receiver: `feats_rng=carried`)
    and the replication is PROVEN: every rank digests the item it holds (`rankdata.feature_digest`; the raw MSA features held on the host are
    outside it by name) and differing digests are `refused: feats_ranks_differ` on every rank (`assert_ranks_agree`, refusing). Under
    ROWPAIR_MSA_HOST (the line exports `rank0`) rank 0's raw MSA features leave its item BEFORE the broadcast (tp.msa_host_take) and never
    travel; their shapes reach every rank after it (tp.msa_host_hold) and rank 0's tensors return to its item (tp.msa_host_land; mode all:
    every rank's host copy made rank 0's, chunked through the device) — the event line names what was held. An item whose featurisation
    stock caught (error_message set, empty data) is the control word ITEM_ERROR at the rendezvous: nothing is broadcast, every rank returns
    the error triple (the stock loop skips the item on every rank in step; `on_item_error` counts it failed); anything `featurise()` or the
    hold raises on rank 0 is the core's `failed:` status — `refused: feats_rank0_failed` on EVERY rank, no peer left waiting.
    P == 1: returns `featurise()`, the stock statement untouched."""
    P, rank = TP.world()
    if P <= 1:
        return featurise()
    STATE["items"] += 1
    calls = STATE.setdefault("calls", {})
    calls[rank] = calls.get(rank, 0) + 1                         # this rank's call ordinal: the rendezvous key, unique per call, equal across ranks by construction
    key = f"feats/{calls[rank]}"
    STATE.setdefault("last_item", {})[rank] = (str(name), int(index))   # the item this rank replicated last: host_side_inputs admits only it into predict
    src = rank == 0                                              # rank 0 featurised; every other rank receives
    data, atom_array, error_message, mine, status = None, None, "", {}, None
    if src:
        try:
            data, atom_array, error_message = featurise()       # upstream's featurisation of the item, its per-item error capture included
            error_message = str(error_message or "")
            if error_message:
                head = " ".join((error_message.strip().splitlines() or [""])[0].split())[:RKD.STATUS_MAX]
                status = f"{ITEM_ERROR} {head}"                 # stock caught the featurisation error: the control word + the error's first line; nothing is broadcast
            else:
                feats_in = data.get("input_feature_dict", data)
                mine = TP.msa_host_take(feats_in)                # mode rank0|all: rank 0's raw MSA tensors leave the item before it travels (no collective here)
        except Exception as e:                                   # anything raised here would leave the receivers at the rendezvous: the core's failure word ends every rank alike
            status = RKD.status_word(e)
    fb = RKD.broadcast_features(data if src else None, src=0, key=key, status=status, device=STATE["device"], carry_rng=True, what="feats")
    if (fb.status.split() or [""])[0] == ITEM_ERROR:             # every rank: the error item, skipped in step by the stock loop (runner/inference.py:460-467)
        if not src:
            data = {"sample_name": name, "sample_index": index}
            error_message = f"{name}: featurisation failed on rank 0: {fb.status[len(ITEM_ERROR):].strip()} (rank 0's transcript and the error file carry the traceback)"
        R.log(f"{R.PREFIX} ROWPAIR event=feats rank={rank} item={name} {RKD.data_form_word(DATA_FORM)} {fb.words()}")
        cb = STATE.get("on_item_error")
        if cb is not None:
            cb(data, error_message)
        return data, atom_array, error_message
    data = fb.feats
    feats = data.get("input_feature_dict", data)
    hold = TP.msa_host_hold(feats, mine=mine if src else None)   # rank 0's raw-MSA shapes / dtypes to every rank (one object broadcast, after rank 0 finished featurising); ranks > 0 hold none
    dg = RKD.digest_features(data)
    agree = RKD.assert_ranks_agree(dg.digest, what="feats", mode="census", log=R.log)   # the core's `[feats] rank r digest <d16> feats_ranks_equal=… digests=…` line on every rank
    if agree is False:                                           # differing digests: every rank named its digest above; `refused: feats_ranks_differ` on every rank
        raise RKD.differ_refusal("feats", rank, dg.digest)
    TP.msa_host_land(feats, hold, STATE["device"])               # rank 0's raw MSA tensors back into its item; mode all: every rank's host copy made rank 0's (chunked); rank0: ranks > 0 hold none
    R.log(f"{R.PREFIX} ROWPAIR event=feats rank={rank} item={name} {RKD.data_form_word(DATA_FORM)} digest={dg.digest[:RKD.DIGEST_HEX]} "
          f"{RKD.agree_word('feats', agree)} {dg.words()} held={','.join(hold['meta']) if hold else 'none'} {fb.words()}")
    if hold is not None:                                         # the raw MSA features never travelled and are outside the digest: named
        R.log(f"{R.PREFIX} ROWPAIR event=msa_host_hold rank={rank} mode={hold['mode']} held={','.join(hold['meta']) or '-'} "
              f"landed={'sync_host_features' if hold['mode'] == 'all' else 'rank0_only'}")
    return data, atom_array, error_message


def host_side_inputs(data, configs=None):
    """After replicate_inputs and the item checks, before the runner moves the features to the device: tp.host_side_inputs keeps the
    pair-shaped input features (token_bonds; the template pair features, sliced to this rank's rows) on pinned host. Only the item this rank
    replicated last (replicate_inputs stamps its name and index) is admitted: any other item entering `predict` is refused by name
    (`rowpair_item_unreplicated`) — it came around the line's `__getitem__`, so no broadcast and no digest gate ran. P == 1: untouched."""
    if STATE["P"] <= 1:
        return data
    want = (STATE.get("last_item") or {}).get(TP.world()[1])
    got = (str(data.get("sample_name")), _int_or_none(data.get("sample_index"))) if isinstance(data, dict) else (None, None)
    if want is None or got != want:                              # fail-closed: an item that did not come through item_of_rank0 (no broadcast, no digest gate) never reaches the model
        _refuse(REFUSE_UNREPLICATED.format(got=f"{got[0]}#{got[1]}", want=(f"{want[0]}#{want[1]}" if want else "none")))
    feats = data.get("input_feature_dict", data) if isinstance(data, dict) else data
    STATE["host_facts"] = TP.host_side_inputs(feats, configs)
    return data


def _int_or_none(v):
    try:
        return int(v.item()) if hasattr(v, "item") else int(v)
    except Exception:
        return None


RANK_SPREAD_MAX_A = 1.0                                    # Å: the largest |x_rank - x_rank0| over atoms a completed diffusion may show across ranks (per-step state
                                                           # broadcast leaves one step's kernel-order noise, ~1e-3 Å; a draw or state that de-synchronised shows ≫ 1 Å)


def coordinates_of_rank0(x):
    """At the diffusion sampler's exit (n_gpu>1): rank 0's sampled coordinates replace every rank's (one broadcast of [N_sample, N_atom, 3]),
    after the cross-rank spread max|x_rank - x_rank0| is all-reduced and recorded (census diff_rank_spread_A): identical inputs and draws by
    construction leave only the kernels' own run-to-run order noise between ranks (0 under --det); a spread above RANK_SPREAD_MAX_A is a
    de-synchronised rank — RowpairRefused by name (`tp_rank_spread`), never a written structure. P == 1: x."""
    if STATE["P"] <= 1:
        return x
    import torch
    DF = TP.C("DF")
    x0 = DF.sync_replicated(x.detach().clone().contiguous(), "diffusion_coordinates", mode="bcast")   # both det levels: rank 0's structure is THE output
    spread = (x.detach().float() - x0.float()).abs().max().reshape(1)
    D.allreduce_(spread, op="max")
    val = float(spread.item())
    STATE["rank_spread_A"] = max(float(STATE.get("rank_spread_A", 0.0)), val)
    TP.C("EV").record_schedule(diff_rank_spread_A=round(STATE["rank_spread_A"], 6))
    if not (val <= RANK_SPREAD_MAX_A):
        _refuse(f"tp_rank_spread: sampled coordinates differ across ranks by {val:.3f} Å > {RANK_SPREAD_MAX_A} Å (a de-synchronised draw or state)")
    if STATE.get("det") and val != 0.0:                                  # det 1: deterministic kernels + identical inputs and draws => bit-equal ranks; any spread is a defect
        _refuse(f"tp_rank_spread_det: sampled coordinates differ across ranks by {val:.3e} Å under --det 1 (must be 0)")
    return x0.to(dtype=x.dtype)


def sync_rng_from_rank0(name: str = "diffusion_rng_state") -> None:
    """At the diffusion sampler's entry (n_gpu>1): rank 0's torch RNG state — the CPU generator and this rank's CUDA generator (seed + Philox
    offset) — is BROADCAST and set on every rank at det 0 / proven identical at det 1 (tp.sync: opt_core.mem.rowpair.diffusion.sync_replicated in
    this process's mode), so the stock sampler's initial noise and every per-step draw are identical on all ranks by construction (census
    diff_noise=bcast_rank0_state | guard_state). Collective; P == 1: nothing."""
    if STATE["P"] <= 1:
        return
    import torch
    DF = TP.C("DF")
    dev = STATE["device"]
    cpu_state = TP.sync(torch.get_rng_state().to(dev), f"{name}:cpu")
    torch.set_rng_state(cpu_state.to("cpu"))
    if torch.cuda.is_available():
        cuda_state = TP.sync(torch.cuda.get_rng_state(dev).to(dev), f"{name}:cuda")
        torch.cuda.set_rng_state(cuda_state.to("cpu"), dev)
    STATE["rng_syncs"] = STATE.get("rng_syncs", 0) + 1
    TP.C("EV").record_schedule(diff_noise=("bcast_rank0_state" if TP.SYNC["mode"] == "bcast" else "guard_state"), diff_noise_syncs=STATE["rng_syncs"])


def assert_rng_replicated(name: str = "rng_state") -> None:
    """Every rank's torch RNG state (CPU generator and the bound CUDA generator: seed + offset) is bit-equal — so the diffusion
    sampler's noise draws are identical on every rank (the stock seeds every rank from the same `--seeds` entry; a rank whose state
    drifted is RowpairRefused by name, never a silently different sample). Collective; P == 1: nothing."""
    if STATE["P"] <= 1:
        return
    import torch
    dev = STATE["device"]
    cpu_state = torch.get_rng_state().to(dev)
    D.allreduce_checksum(cpu_state, f"{name}:cpu")
    if torch.cuda.is_available():
        D.allreduce_checksum(torch.cuda.get_rng_state().to(dev), f"{name}:cuda")


# ----------------------------------------------------------------------------------------------------------------- evidence
def lever_line(state: str = None, reason: str = None, peaks_gib=None) -> str:
    """The rowpair LEVER line (the family's producer and strategy id, opt_core.mem.rowpair.LEVER): on with the layout facts of the last stack when P > 1,
    off with reason=n_gpu_1 at P == 1."""
    P = STATE["P"]
    if state is None:
        state = "on" if P > 1 else "off"
    if reason is None and P <= 1:
        reason = "n_gpu_1"
    c = TP.COUNTS
    ev = dict(stacks=c["pairstack_calls"], blocks=c["pair_blocks"], items=STATE["items"])
    if P > 1:
        ev.update(rank=STATE["rank"], peak_alloc_gib=_own_peak_gib(), trunk_inits=c["trunk_inits"], recycles=c["recycles"],
                  msa_blocks_rows=c["msa_blocks_rows"], template_calls=c["template_calls"], gathers_in_trunk=c["gathers_in_trunk"],
                  distogram_rows=c["distogram_rows"], conf_samples=c["conf_samples"], conf_rows=c["conf_rows"], zcond_rows=c["zcond_rows"],
                  band_calls=c["band_calls"], dit_calls=c["dit_calls"],
                  replicated=",".join(f"{k}:{v}" for k, v in sorted(TP.REPLICATED.items())) or "-")
    return EV.lever_line(R.PREFIX.strip("[]"), state, P, name=LEVER, layout=TP.STATE.get("layout"), reason=reason, peaks_gib=peaks_gib, **ev)


def emit_kernel_lines() -> List[str]:
    """Print the core's LEVER lines of the row kernels the xP line constructed (tp.emit_tp_kernel_lines: F2.trimul_rows / F1.flash_triattn),
    right after the rowpair LEVER line; prints nothing when the line ran the torch contraction and the stock attention module."""
    return TP.emit_tp_kernel_lines()


def _own_peak_gib():
    """This rank's torch peak allocation in GiB (every rank prints its own LEVER line)."""
    try:
        import torch
        if torch.cuda.is_available():
            from . import phase as PHASE                      # the process-wide peak: phase.py resets torch's counters per item (PEAK line) and carries the max
            return round(max(int((PHASE.process_peak() or {}).get("alloc") or 0), int(torch.cuda.max_memory_allocated())) / 2 ** 30, 2)
    except Exception:
        pass
    return "none"


def record() -> dict:
    tp = TP.record()
    return {"P": STATE["P"], "rank": STATE["rank"], "installed": STATE["installed"], "items": STATE["items"], "sites": list(STATE.get("sites") or []),
            "counts": tp["counts"], "replicated_bytes": tp["replicated_bytes"], "parks": tp["parks"], "rows": tp["rows"]}
