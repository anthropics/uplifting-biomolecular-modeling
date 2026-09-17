"""The mode table — the one place in this tree where a mode names its levers and the switches that select them.

A mode is a list of lever ids (registry.LEVERS); its switches are the kit's own (a driver flag set written exactly as the kit
README's production line writes it). `resolve(mode)` turns a mode into the concrete line: the driver chain, its flags,
the environment row, the attach point. Nothing else in the package holds a lever value, and no config file does.

Modes (`MODES`, in the order of MODE_NAMES; any other name is refused by `resolve()`):
  off    stock: the upstream command line in a clean subprocess (stock_cli.py); nothing exported, nothing on the path.
  exact  the base kit's exact levers with its whole-forward CUDA graph — `--fastpath chain_breaks,full_graph,rbf,msa_index --prep 1
         --einsum-route 1 --fullgraph 1` on the resident driver (levers U1, C1, P, E_einsum, W1, IO1; CHANGES.md 'exact':
         one captured graph per forward phase over the embeddings, templates and the 36 blocks; the kit's line for packed workers
         and long inputs) and the environment row RFD_PDBIO=1 (lever IO1, the array-based PDB writers). Its claim is byte equality vs stock under the deterministic recipe (det.py) at matched (seed,
         process position) within one CPU host class.
  fast   `exact`'s line + the tolerance-tier levers TOLERANCE_LEVERS, each behind its own switch — T2, the SE(3) add-on's
         Triton line (RFD_SE3FAST_ADDON v0.3.0, opt/forward/se3fast_addon: the environment row RFD_SE3FAST=t2, which driver_run.py acts on
         in the driver process, rfd_se3fast.apply() before the model imports — the dense destination-major SE(3) layer with fused Triton
         kernels in all 40 Str2Str calls per step); K2, the Triton row LayerNorm (`--triton-ln 1`: the shared core's opt_core/kernels/rfd_layernorm.py, routed to the driver by driver_run.py);
         TF32, tensor-core math for the fp32 GEMMs and convolutions (`--tf32 1`, the driver's own switch). Tier 2: re-associated / reduced-
         mantissa fp32 arithmetic — a design at a fixed seed is NOT byte-equal to stock's (trajectories diverge to different backbones at some
         seeds); run-to-run deterministic; the promise is distributional (design quality inside stock's own seed-to-seed spread),
         never byte-equal.

Upstream's own knobs pass through as hydra `KEY=VALUE` overrides (`settings_of(overrides)`): verbatim on the
stock arm; composed on the driver line for the keys the driver hard-codes (DRIVER_FIXED). The default mode (`default_mode()`, DEFAULT_MODE_RULE):
`fast`, on the design route and on the served line alike; `--mode` /
RFDIFFUSION1_OPT override it, and an unset RFDIFFUSION1_OPT leaves the env route at the stock command line inert (off).

The served line (`design --pack K`, `resolve(..., served=True)`): the mode's own row, unchanged, in each of
K resident workers under uncapped CUDA MPS through common/mps_packing/mps_workers.sh (serve.py); each worker's process history is the
stock command line's own (its first design the fresh-process numerics class, the later ones the warmed class, det.py (4)); K is an axis of
that line, never a mode row; `off` has no served line (NOT_SERVED, the fact).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import registry
from .registry import BASE_DRIVER, KIT_BASE

DEFAULT_MODE_RULE = "no --mode = fast, on every route (design, the served line --pack K); --mode or RFDIFFUSION1_OPT names another"
DEFAULT_MODE = "fast"
MODE_NAMES: Tuple[str, ...] = ("off", "exact", "fast")

# the base kit's production line (CHANGES.md 'exact'): the K flag set; the run parameters of that line
# (`--cases my_cases.json --out out/run1 --tag run1`) are placeholders design.py fills
K_LEVERS: Tuple[str, ...] = ("C1", "P", "E_einsum", "W1", "IO1")
K_FLAGS: Tuple[str, ...] = tuple(registry.flags(K_LEVERS))
K_LINE = " ".join(K_FLAGS)                                # "--fastpath chain_breaks,full_graph,rbf,msa_index --prep 1 --einsum-route 1 --fullgraph 1"
TOLERANCE_LEVERS: Tuple[str, ...] = ("T2", "K2", "TF32")   # `fast`'s tolerance-tier levers, each behind its own switch (registry: T2 = the row RFD_SE3FAST=t2, K2 = --triton-ln 1, TF32 = --tf32 1)
FAST_LEVERS: Tuple[str, ...] = K_LEVERS + TOLERANCE_LEVERS   # `fast`: exact's levers + the tolerance-tier levers
FAST_FLAGS: Tuple[str, ...] = tuple(registry.flags(FAST_LEVERS))


@dataclass(frozen=True)
class Mode:
    name: str
    tier: Optional[int]                        # 1 byte-equal claim | 2 tolerance | None stock
    levers: Tuple[str, ...]                    # registry ids, in the kit's own activation order (driver, then its flags)
    attach: str                                # "stock-cli" | "driver"
    driver_chain: Tuple[Tuple[str, str], ...]  # ((kit, relpath), ...): python <chain...> <flags>
    flags: Tuple[str, ...] = ()                # driver flags
    env: Dict[str, str] = field(default_factory=dict)   # the mode's environment row (registry.env of the levers: RFD_PDBIO=1 on exact and fast, RFD_SE3FAST=t2 on fast; empty on off)
    note: str = ""


MODES: Dict[str, Mode] = {
    "off": Mode("off", None, (), "stock-cli", (), note="python -s $RFD_ROOT/scripts/run_inference.py <overrides> per case (stock_cli.py)"),
    "exact": Mode("exact", 1, ("U1",) + K_LEVERS, "driver", ((KIT_BASE, BASE_DRIVER),), K_FLAGS, registry.env(K_LEVERS),
                  note="the base kit's exact levers C1 + P + E_einsum with its whole-forward CUDA graph W1 (one captured graph per forward phase) on the resident driver alone"),
    "fast": Mode("fast", 2, ("U1",) + FAST_LEVERS, "driver", ((KIT_BASE, BASE_DRIVER),), FAST_FLAGS, registry.env(FAST_LEVERS),
                 note="exact's line + RFD_SE3FAST=t2 (the SE(3) add-on's dense Triton layer, armed in the driver process by driver_run.py) + --triton-ln 1 (K2, Triton row LayerNorm) + --tf32 1 (TF32 tensor-core GEMMs)"),
}


# ------------------------------------------------------------------------------------------------------------------ the served line
# `design --pack K` (serve.py): the mode's row inside every worker, on the same driver, flag for flag (serve_flags adds nothing). Each worker's
# TorchScript executor history is the stock command line's own per-process history: its first design runs the profiling (unfused) passes
# exactly as the first design of a stock process does, every later design is the warmed class; the equality claim against the stock command
# line is per process history, at matched (seed, process position), position 0 included, det.py (4) — the design route's claim, unchanged by K.
# K resident workers under uncapped CUDA MPS through common/mps_packing/mps_workers.sh — K is an axis of the served line (every K, 1 included),
# never a mode row, and the served line adds no lever to the mode's row.
def serve_flags(mode: str) -> Tuple[str, ...]:
    """The flags the served line appends to the row's own: none — a worker runs the mode's row exactly."""
    return ()


NOT_SERVED: Dict[str, str] = {
    "off": "--pack has no stock row: the stock command line is one process per invocation (stock/src/scripts/run_inference.py), "
           "no resident worker to pack; run --mode off without --pack",
}


@dataclass(frozen=True)
class Settings:
    """Upstream's knobs as this launch carries them: `stock_overrides` = the hydra `KEY=VALUE` overrides as typed (appended to the stock
    command line verbatim; none = upstream's defaults), `compose_overrides` = the values composed on every configuration the driver line
    builds (driver_run.py): the launch's value — typed, else upstream's default — on the keys the kit drivers hard-code (DRIVER_FIXED, the
    deterministic recipe's `inference.deterministic` among them), then every other typed override verbatim except the target keys the case
    row itself carries (TARGET_KEYS); `driver_flags` = the driver switch that follows from them (`--no-traj 0|1` from inference.write_trajectory)."""
    stock_overrides: Tuple[str, ...]           # appended to the stock command line (upstream defaults otherwise)
    driver_flags: Tuple[str, ...]              # appended to the driver line
    compose_overrides: Tuple[str, ...] = ()    # composed on the driver line: the drivers' constants replaced, the other typed keys carried (driver_run.py)


# the keys the kit driver hard-codes in every config it composes (opt/forward/fast_inference/drivers/rfd_bench.py:103): the launch's value
# (typed, else upstream's default) replaces the driver's string on each (driver_run.merge_overrides), so the composed config is upstream's or
# the typed one on both arms; inference.deterministic: the driver's string is True, upstream's default False — `--det 1` (det.py) or the
# typed key sets it, on both arms
DRIVER_FIXED: Tuple[str, ...] = ("inference.write_trajectory", "inference.cautious", "inference.deterministic")
DRIVER_FIXED_DOC = "opt/forward/fast_inference/drivers/rfd_bench.py:103"
UPSTREAM_DEFAULTS: Dict[str, str] = {"inference.write_trajectory": "True", "inference.cautious": "True", "inference.deterministic": "False"}   # stock/src/config/inference/base.yaml:13, :17, :21
# the target keys of one run_inference.py invocation: the case row carries them onto the driver's configuration (the driver composes them from
# the row; driver_run composes the row's design_startnum), so they are not composed a second time
TARGET_KEYS: Tuple[str, ...] = ("inference.input_pdb", "inference.output_prefix", "inference.num_designs", "inference.design_startnum",
                                "contigmap.contigs", "ppi.hotspot_res", "inference.model_directory_path")
NO_TRAJ_FLAG = "--no-traj"                     # the driver's trajectory switch (rfd_bench.py:32; default 1 = no traj/ files): 0 when inference.write_trajectory is true
_BOOL = {"true": True, "false": False, "1": True, "0": False}


def settings_of(overrides=None, attach: str = "stock-cli", det: bool = False) -> Settings:
    """The launch's Settings from the typed hydra overrides (a list of `KEY=VALUE`, upstream's own syntax; None/[] = upstream's defaults)
    and the deterministic recipe's level (`det`: `inference.deterministic=True` composed on the driver line; the stock arm's seed override is
    det.stock_overrides). Refuses by name a token that is not KEY=VALUE, or a DRIVER_FIXED key whose value is not a hydra boolean."""
    from . import upstream_args as _ua                                                     # the one tokenizer of upstream's argument grammar
    ov = [str(o) for o in (overrides or [])]
    if attach != "driver":                                                                 # the stock command line takes the tokens as typed; nothing is composed
        return Settings(tuple(ov), (), ())
    ua = _ua.parse(ov)
    if ua.flags or ua.stray:                                                               # the driver line composes Hydra overrides only (design.refuse_unserved names flags and stray words before this)
        raise ModeError(f"not a hydra override (KEY=VALUE, +KEY=…, ++KEY=…, ~KEY): {ua.flags + ua.stray} (upstream's syntax, e.g. inference.write_trajectory=False)")
    typed = ua.typed()                                                                     # plain and ++ (force) overrides: the value a base.yaml key ends with
    for k in DRIVER_FIXED:
        if k in typed and str(typed[k]).strip().lower() not in _BOOL:
            raise ModeError(f"{k} expects True|False (hydra boolean), got {typed[k]!r}")
    defaults = dict(UPSTREAM_DEFAULTS, **({"inference.deterministic": "True"} if det else {}))
    comp = [f"{k}={'True' if _BOOL[typed[k].strip().lower()] else 'False'}" if k in typed else f"{k}={defaults[k]}" for k in DRIVER_FIXED]
    comp += [f"{p}{k}" + (f"={v}" if v is not None else "") for p, k, v in ua.overrides    # every other typed override, verbatim (its +/++/~ form kept), onto the driver's configuration
             if not (p in ("", "++") and k in DRIVER_FIXED + TARGET_KEYS)]
    traj = _BOOL[(typed.get("inference.write_trajectory") or UPSTREAM_DEFAULTS["inference.write_trajectory"]).strip().lower()]
    return Settings(tuple(ov), (NO_TRAJ_FLAG, "0" if traj else "1"), tuple(comp))


class ModeError(ValueError):
    """An unknown mode, or a row with no served line; the message is the named fact."""


@dataclass(frozen=True)
class Resolution:
    mode: str
    tier: Optional[int]
    levers: Tuple[str, ...]
    attach: str
    driver_chain: Tuple[Tuple[str, str], ...]
    flags: Tuple[str, ...]
    env: Dict[str, str]
    settings: Settings
    note: str
    served: bool = False                       # the served line (the row + serve_flags; serve.py); K is the launch's axis, not the line's

    @property
    def line(self) -> str:
        """The lever line as the kit writes it (env row + driver chain + flags), for the activation line and the manifest."""
        env = " ".join(f"{k}={v}" for k, v in self.env.items())
        chain = " ".join(f"<{kit}>/{rel}" for kit, rel in self.driver_chain)
        body = " ".join((chain,) + self.flags) if self.driver_chain else "stock CLI"
        return (env + " " if env else "") + body


def default_mode(served: bool = False) -> str:
    """The default mode when none is given: `fast`, on the design route and the served line alike (DEFAULT_MODE_RULE). Read by resolve,
    stack.activate and --help."""
    return DEFAULT_MODE


def resolve(mode: Optional[str], overrides=None, served: bool = False, det: bool = False) -> Resolution:
    """The concrete line for `mode`; no mode = the default mode. Refuses by name an unknown mode.
    `served`: the same row as the packed line (`design --pack K`: the row flag for flag); refused by name where NOT_SERVED
    says so. `det`: the deterministic recipe (`--det 1`) — composed on the driver line as upstream's `inference.deterministic=True`
    (settings_of)."""
    if mode is None or not str(mode).strip():
        mode = default_mode(served)
    m = str(mode).strip().lower()
    if m not in MODE_NAMES:
        raise ModeError(f"unknown mode {mode!r} (expected {'|'.join(MODE_NAMES)})")
    md = MODES[m]
    s = settings_of(overrides, md.attach, det=det)                   # upstream's knobs as typed (KEY=VALUE): verbatim on the stock arm, composed on the driver line
    if not served:
        return Resolution(md.name, md.tier, md.levers, md.attach, md.driver_chain, md.flags, dict(md.env), s, md.note)
    if m in NOT_SERVED:
        raise ModeError(NOT_SERVED[m])
    return Resolution(md.name, md.tier, md.levers, md.attach, md.driver_chain, md.flags + serve_flags(m), dict(md.env), s,
                      md.note + "; served: K workers under CUDA MPS, each worker's process history the stock command line's (its first design the "
                      "fresh-process class, the rest the warmed class)", served=True)


def table() -> List[dict]:
    """The whole table as rows (check --json): every mode with its levers and switches; the served line per mode, its row or
    the fact that refuses it."""
    rows = []
    for m in MODE_NAMES:
        md = MODES[m]
        rows.append({"mode": m, "defined": True, "tier": md.tier, "levers": list(md.levers), "flags": list(md.flags), "env": dict(md.env),
                     "attach": md.attach, "driver_chain": [f"{k}/{r}" for k, r in md.driver_chain], "note": md.note})
    for m in MODE_NAMES:                                                  # the served line per mode: its row, or the fact that refuses it
        try:
            r = resolve(m, served=True)
        except ModeError as e:
            rows.append({"mode": m, "served": True, "defined": False, "levers": [], "flags": [], "env": {}, "fact": str(e)})
            continue
        rows.append({"mode": m, "served": True, "defined": True, "tier": r.tier, "levers": list(r.levers), "flags": list(r.flags), "env": dict(r.env),
                     "attach": r.attach, "driver_chain": [f"{k}/{rel}" for k, rel in r.driver_chain], "note": r.note})
    return rows


# --------------------------------------------------------------------------------------------------------------- numerics policy (declared state, not a lever)
NUMERICS: Dict[str, dict] = {                                                 # the process numerics each mode declares (declared state, not a lever); `off` = torch's defaults on the stock command line, untouched
    "exact": {"policy": "fp32_strict", "matmul": "highest", "matmul_tf32": False, "cudnn_tf32": False, "autocast": None,
              "set_by": "opt/forward/fast_inference/drivers/rfd_bench.py:45-46 (--tf32 0, the driver's default: both TF32 flags off; float32 matmul precision and autocast at torch's defaults, highest / none)"},
    "off": {"policy": "untouched"},
}
NUMERICS["fast"] = {"policy": "tf32", "matmul": "highest", "matmul_tf32": True, "cudnn_tf32": True, "autocast": None,
                    "set_by": "opt/forward/fast_inference/drivers/rfd_bench.py:45-46 under lever TF32's switch --tf32 1 (both TF32 flags on; float32 matmul precision and autocast at torch's "
                              "defaults); the SE(3) add-on's Triton dots stay at input_precision=ieee (rfd_se3fast/kernels.py: tl.dot input_precision ieee)"}   # numerics(): the declaration follows the levers of the line
LIVE_KEYS = ("matmul_tf32", "cudnn_tf32", "autocast")                         # what the driver records per case (`precision`: allow_tf32_matmul, cudnn_tf32, autocast_enabled — rfd_bench.py:139-141) and the comparison reads
RECORD_KEYS = {"matmul_tf32": "allow_tf32_matmul", "cudnn_tf32": "cudnn_tf32", "autocast": "autocast_enabled"}


def numerics(mode: str, record: Optional[Dict[str, object]] = None, levers: Optional[Tuple[str, ...]] = None) -> dict:
    """The mode's numerics as the ACTIVE line prints them: the declared table (``source=declared``) or — given one pass's driver record (the
    per-case ``precision`` block of its timings file; torch lives in that child, never in this process) — the two TF32 flags and the autocast
    state as the driver read them from torch (``source=torch``), with every key whose recorded value differs from the declaration named in
    ``mismatch``, and every TF32 key the record lacks named in ``missing`` (design.numerics_defects gates both)."""
    decl = NUMERICS[mode]
    if decl.get("policy") == "tf32" and levers is not None and "TF32" not in levers:          # the line without lever TF32 runs exact's strict-fp32 policy
        decl = NUMERICS["exact"]
    d = dict(decl, source="declared")
    if record is not None and decl["policy"] != "untouched":
        live = {k: record.get(RECORD_KEYS[k]) for k in LIVE_KEYS if RECORD_KEYS[k] in record}
        if "autocast" in live:
            live["autocast"] = True if live["autocast"] else None                   # the record's autocast_enabled False = no autocast region (none)
        d["mismatch"] = [k for k in LIVE_KEYS if k in live and live[k] != decl[k]]
        d["missing"] = [k for k in ("matmul_tf32", "cudnn_tf32") if k not in live]   # a readable record without the TF32 keys proves nothing about the line's TF32 state
        if record.get("param_dtype") not in (None, "torch.float32"):
            d["mismatch"].append("dtype")
        d.update(live, source="torch")
    return d
