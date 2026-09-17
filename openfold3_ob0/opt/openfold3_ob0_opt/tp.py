"""The big/tp line's process form: one `pred` process per GPU, joined into one NCCL group, launched and censused by this CLI.

`pred --mode big --n_gpu P` (P > 1) in a process that is not a rank (no OF3TP_RANK) is the LAUNCHER: it pins the chunk plan into the runner yaml
(the line's contract, `pinned_yaml`: `settings.memory.eval.chunk_size` pinned and the chunk tuner off in all five stacks, the trunk's msa/template host offload off, the fused attention /
triangle kernels off; the shipped confidence-head offload and per-sample cutoffs stay as the row's yaml has them — the sharded confidence head
names them), spawns N `python -m openfold3_ob0_opt pred` RANK processes (OF3TP_RANK=r, OF3TP_WORLD=N, one GPU each through CUDA_VISIBLE_DEVICES,
the same OF3TP_ADDR/OF3TP_PORT), streams rank 0's log to its own stderr, samples every GPU's used memory once a second, fails fast when a
rank fails (the peers would otherwise wait on NCCL until its timeout), and prints the census (`census …` line; the tp block of the run record): the
world size, the backend and mode the ranks reported, the chunk, every rank's exit code / GPU / peak MiB / rows of the pair representation
(from the ranks' own `trunk done N=… rows a:b` lines), and whether the ranks' outputs are byte-identical (rank 0 writes the prediction; the
line's contract says the other ranks' copies are identical — checked, never assumed). Every one of those is a NAMED event: a rank that
died, a rank missing from the census, a shard map that does not cover N rows, differing rank outputs, a GPU census short of N — each ends the
run with the reason on stderr, never a silent single-GPU or partial run.

A RANK process (OF3TP_RANK set) runs the kit's ordinary in-process route (cli.cmd_pred): `enable("big", n_gpu=P)` executes the
line's hook (tp_rowpair/hook/sitecustomize.py: the tensor-parallel finder, world > 1) chained with the fast-inference kit's fast init, then
`run_openfold predict` with the pinned yaml. Every rank hands its run record (cli.gated_manifest) to this launcher through the per-launch records
directory (manifest.RECORDS_ENV); rank 0's is the run's: its exit rule and template guard are read back here, then the directory is removed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

from . import modes

NGPU_MISMATCH = "n_gpu_mismatch"                        # the census' first reason when the ranks that armed and joined a P-wide group are not exactly P: NOT ACTIVE, exit 3 — a P-GPU request never passes on fewer
MSA_CACHE_DIR = "msas"                                     # upstream's MSA cache root under every --output-dir (experiment_runner.py:649: msas/msa-root-<time>-<id>/…/*.npz, one per process; never a prediction)
RANK_DIR = "_tp"                                        # under the output directory: rank<r>/ (ranks > 0 outputs, removed after the equality check), rank<r>.log (every rank)
PINNED_YAML = "tp_predict.yml"                          # the run's pinned runner yaml (written by the launcher next to the outputs)
# the chunk plan (the value and its rule are recorded per run): (upper bound on polymer residues, chunk) — 128 up to 2500, 64 up to
# 4000, 32 up to 6500, 16 above; the chunk halves as the tri-attention logits transient (chunk x N^2 per rank) grows on an 80 GB card (the top
# entry is the chunk mid-size queries run at; 16 is the 80 GB value at the largest sizes, where a larger card would fit a larger chunk)
CHUNK_PLAN: Tuple[Tuple[int, int], ...] = ((2500, 128), (4000, 64), (6500, 32), (10 ** 9, 16))
from .tp_rowpair import floor as FLOOR                      # the sharding floor: ONE predicate for this launcher (polymer residues) and the ranks (featurised tokens)
from .tp_rowpair import FEATS_TAG, FEATS_IDENTICAL          # the cross-rank input digest gate's log line: its tag and verdict word, spelled once (tp_rowpair), read back here (parse_rank_log)

CHUNK_FLOOR = FLOOR.BLOCK_UNIT                             # 16: the smallest chunk the plan reduces to for a short query = the layout's finest row-block unit (floor.BLOCK_UNIT)
ENV_CHUNK = "OF3TP_CHUNK"
MODE_S, BACKEND = "S", "nccl"                                   # the launch line's words: the row-sharded pair representation (the one form) over an NCCL group
TUNER_OFF_PATHS = (("template", "template_pair_stack"), ("msa", "msa_module"), ("pairformer",), ("diffusion_module", "diffusion_conditioning"),
                   ("heads", "pairformer_embedding", "pairformer"))                    # the five chunk tuners (model_config.py:250,286,307,326,417; all follow settings.memory.eval.tune_chunk_size)
OFFLOAD_OFF = ("msa_module", "template_module")              # settings.memory.eval.offload_inference: host offload of the trunk's MSA / template modules does not apply under row sharding
                                                           # (the row shards bound those modules' memory; confidence_heads stays as the yaml has it — the sharded head names it)
KERNEL_FLAGS_OFF = ("use_triton_triangle_kernels", "use_deepspeed_evo_attention", "use_cueq_triangle_kernels", "use_lma")   # settings.memory.eval (model_config.py:116-120); tp_rowpair.trunk.KERNEL_FLAGS
TRIMUL_TAG = "[tp_rowpair] TRIMUL "                          # the tp_trimul lever's per-rank census line (tp_rowpair/pairstack.trimul_census_line): kernels= bound= served= fallback= fallback_by= cells= k1_impl=
TRIATT_TAG = "[tp_rowpair] TRIATT "                          # the tp_triatt lever's per-rank census line (tp_rowpair/pairstack.triatt_census_line): kernel= bound= calls= served= fallback= fallback_by=
KERNEL_GATES = {"triatt": ("unsupported:dtype_float32",), "trimul": ("below_gate", "c=64/64", "layout")}   # the DOCUMENTED gates per kernel lever (CHANGES.md `tp`), each running
                                                                   #  OpenFold3's torch statements by design, counted: F1 — the confidence head's pair stack, which OpenFold3 0.5.0 runs in fp32
                                                                   #  (heads `pairformer_dtype=float32`; the flash kernel serves bf16 / fp16: `unsupported:dtype_float32`); F2 — its pair-size
                                                                   #  gate (pairs under 2048 tokens: `below_gate`), its width gate (the template embedder's c=64 pair stack; the kernels serve
                                                                   #  c 128 / 256: `c=64/64`) and its bf16 staging contract, which the same fp32 confidence pair stack does not meet (`layout`)
KERNEL_OPT_OUTS = ("kernel_torch", "env_torch")                    # the explicit, user-typed engineering opt-outs (ROWPAIR_TRIATT_CORE=torch / ROWPAIR_TRIMUL_KERNELS=torch): every unit declined by request
REFUSAL_RE = re.compile(r"\b(\w*Refused): (.+)$")                      # `<module.>SomethingRefused: <sentence>` — the last line of a rank's refusal traceback
POLL_S = 2.0
PRED_PATTERNS = ("_model.cif", "_confidences.json", "_confidences_aggregated.json", ".npz")   # upstream's prediction files (its timing.json / experiment_config.json / summary.txt are bookkeeping)


def log(msg: str) -> None:
    sys.stderr.write(f"[openfold3_ob0-opt tp] {msg}\n")
    sys.stderr.flush()


# ----------------------------------------------------------------------------------------------------------------- inputs ----
def polymer_residues(query_json: str) -> int:
    """The polymer residue count of the query file (every protein / RNA / DNA sequence x the chain ids it is given, upstream's query form:
    inputs.py): the chunk plan's input (ligand atoms are not counted; the count is recorded beside the chunk so the plan can be read back)."""
    with open(query_json, encoding="utf-8") as fh:
        q = json.load(fh)
    n = 0
    for query in (q.get("queries") or {}).values():
        for ch in query.get("chains") or []:
            seq = ch.get("sequence")
            if isinstance(seq, str):
                ids = ch.get("chain_ids")
                n += len(seq) * (len(ids) if isinstance(ids, list) and ids else 1)
    return n


def chunk_for(n_residues: int) -> int:
    for bound, chunk in CHUNK_PLAN:
        if n_residues <= bound:
            return chunk
    return CHUNK_PLAN[-1][1]


MIN_CARD_BYTES = 79 * 10 ** 9                            # CHUNK_PLAN is the 80 GB-card table (an H100/A100-80GB reports ~79.6-81.5e9 B total); a smaller card gets a NOTE line, never a refusal


def card_note(mem: Optional[int], environ=None) -> Optional[str]:
    """The NOTE line of a launch on a card below the chunk plan's memory class (None on an 80 GB-class card, when the memory cannot be read, or when
    the caller set OF3TP_CHUNK): the plan is the {MIN}+ GB-card table; the run proceeds on it and may run out of memory — OF3TP_CHUNK=<chunk> tunes it."""
    environ = os.environ if environ is None else environ
    if mem is None or mem >= MIN_CARD_BYTES or (environ.get(ENV_CHUNK) or "").strip():
        return None
    return (f"NOTE chunk plan table (CHUNK_PLAN) is for >={MIN_CARD_BYTES // 10**9} GB cards; using it on {mem / 1e9:.1f} GB (GPU 0, opt_core.arch.device_memory) "
            f"— may OOM; {ENV_CHUNK}=<chunk> tunes it")


def card_memory(index: int = 0) -> Optional[int]:
    """Total memory of GPU `index` in bytes as opt_core.arch reads it (nvidia-smi / the framework), None when it cannot be read."""
    try:
        from opt_core import arch
        got = arch.device_memory(index)
    except Exception:  # noqa: BLE001  (no arch registry in the core, no device, probe failure: the plan runs unrefused and the launch log says so)
        return None
    total = (got or {}).get("total_bytes") if isinstance(got, dict) else getattr(got, "total_bytes", None)
    return int(total) if total else None


def chunk_of_run(query_json: str, environ=None, world: int = 1) -> Tuple[int, str]:
    """(chunk, how): a caller's OF3TP_CHUNK is honoured and recorded as the caller's; else the plan's value for the query's polymer residues,
    halved (down to CHUNK_FLOOR) until the row partition at that chunk — the chunk is also the row layout's alignment (ROWPAIR_CHUNK_ALIGN) —
    gives every one of the `world` ranks at least one full block of CHUNK_FLOOR rows (`tp_rowpair.floor.verdict`; a short query at a large
    chunk would leave a rank with zero or a few ragged rows); the reduction is named in `how`. `launch` then applies the sharding floor to the
    plan's chunk and serves a query still below it unsharded (`small_n_unsharded`)."""
    environ = os.environ if environ is None else environ
    preset = (environ.get(ENV_CHUNK) or "").strip()
    if preset:
        try:
            c = int(preset)
        except ValueError:
            raise ValueError(f"{ENV_CHUNK}={preset!r} is not an integer")
        if c <= 0:
            raise ValueError(f"{ENV_CHUNK}={preset!r} must be positive")
        return c, f"caller ({ENV_CHUNK})"
    from .tp_rowpair import floor as _floor
    n = polymer_residues(query_json)
    c0 = c = chunk_for(n)
    while int(world) > 1 and c > CHUNK_FLOOR and not _floor.verdict(max(n, 1), int(world), c)["ok"]:     # every rank must own a full block at this alignment
        c //= 2
    if c != c0:
        return c, f"plan {c0} for {n} polymer residues (CHUNK_PLAN) reduced to {c}: {-(-max(n, 1) // c)} row chunks for {int(world)} ranks"
    return c, f"plan for {n} polymer residues (CHUNK_PLAN)"


def pins_overlay(chunk: int) -> dict:
    """The tp line's REQUIRED runner-yaml keys as one `model_update.custom` document (laid over the run's base yaml by `pinned_yaml`):
    ``settings.memory.eval.chunk_size`` = the plan's chunk, ``tune_chunk_size`` false at the settings level and in the five stacks
    (``TUNER_OFF_PATHS``), the fused attention / triangle kernels off (``KERNEL_FLAGS_OFF``: the row shards run the torch statements), and the
    trunk's MSA / template host offload off (``OFFLOAD_OFF``: row shards bound those modules' memory on each rank; offloading them whole does not
    apply). NOT required: ``offload_inference.confidence_heads`` (as the yaml has it; the sharded head names it) and the per-sample cutoffs."""
    ev = {"chunk_size": int(chunk), "tune_chunk_size": False, **{flag: False for flag in KERNEL_FLAGS_OFF}, "offload_inference": {k: False for k in OFFLOAD_OFF}}
    arch: dict = {}
    for path in TUNER_OFF_PATHS:
        node = arch
        for k in path:
            node = node.setdefault(k, {})
        node["tune_chunk_size"] = False
    return {"model_update": {"custom": {"settings": {"memory": {"eval": ev}}, "architecture": arch}}}


def pinned_yaml(base_yaml: str, chunk: int, out_path: str) -> dict:
    """The runner yaml of the run: `base_yaml` (the line's member, or a caller's yaml cli.cmd_pred already composed under it) with the line's required keys
    (`pins_overlay`) laid on top by the kit's one composition primitive (`cli.deep_overlay`), written to `out_path`. ONE note line names the
    composition and every base key the pins overrode (`none` when nothing) — and the caller's extra presets the pins are laid over (a preset such
    as upstream's `low_mem` sets the trunk offload switches; `custom` keys apply after presets). Returns the pins written (the census' record)."""
    import yaml
    from .cli import deep_overlay
    with open(base_yaml, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    doc, overrides = deep_overlay(doc, pins_overlay(chunk), "")
    presets = [p for p in ((doc.get("model_update") or {}).get("presets") or []) if p != "predict"]
    log(f"note: runner yaml {base_yaml} composed under the tp line's required keys (chunk_size {int(chunk)}, chunk tuners off, fused pair kernels off, "
        f"msa/template host offload off) -> {out_path}; tp overrides runner-yaml keys: {'; '.join(overrides) if overrides else 'none'}"
        + (f"; laid over the caller's presets {','.join(map(str, presets))} (custom keys apply after presets)" if presets else ""))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    return {"base_yaml": os.path.abspath(base_yaml), "chunk_size": int(chunk), "tune_chunk_size": False, "tuners": ["/".join(p) for p in TUNER_OFF_PATHS],
            **{flag: False for flag in KERNEL_FLAGS_OFF}, "offload_inference": {**{k: False for k in OFFLOAD_OFF}, "confidence_heads": "as_yaml"},
            "per_sample_cutoffs": "as_yaml", "overrides": overrides, "presets_laid_over": presets}


# ----------------------------------------------------------------------------------------------------------------- GPUs ----
def launch_cc(index: Optional[str] = None):
    """The compute capability ``(major, minor)`` of the run's first GPU as ``nvidia-smi --query-gpu=compute_cap`` reads it (the ranks read theirs
    through torch; one device class per box), ``None`` when it cannot be read (the tp_triatt word is then the word by name, ``env.triatt_word``)."""
    gpus = gpu_census()
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=60).stdout
    except Exception:                                                        # noqa: BLE001
        return None
    want = str(index if index is not None else (gpus[0] if gpus else "0")).strip()
    first = None
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 3 or "." not in parts[2]:
            continue
        cc = tuple(int(x) for x in parts[2].split(".")[:2])
        first = first or cc
        if want in (parts[0], parts[1]) or parts[1].startswith(want):
            return cc
    return first


def gpu_census() -> List[str]:
    """The GPUs this process may use: CUDA_VISIBLE_DEVICES when set (its entries, in order), else nvidia-smi's list (indices)."""
    preset = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if preset:
        return [x.strip() for x in preset.split(",") if x.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return [str(i) for i, l in enumerate(out.splitlines()) if l.startswith("GPU ")]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MemorySampler:
    """nvidia-smi memory.used once a second per GPU: peak MiB per GPU over the run, keyed by the GPU's index with
    its UUID and PCI bus id recorded beside (`gpus`): the index is the launcher's rank↔device mapping, the UUID is the card's identity."""

    def __init__(self, period_s: float = 1.0):
        self.period_s, self.peak, self.gpus, self.samples, self._stop = period_s, {}, {}, 0, threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,uuid,pci.bus_id,name,memory.total", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=20).stdout
                for l in out.splitlines():
                    idx, used, uuid, bus, name, total = [x.strip() for x in l.split(",")[:6]]
                    self.peak[idx] = max(self.peak.get(idx, 0), int(float(used)))
                    self.gpus[idx] = {"uuid": uuid, "bus_id": bus, "name": name, "memory_total_mib": int(float(total))}
                self.samples += 1
            except Exception:                                        # nvidia-smi absent or busy: the record says how many samples were taken
                pass
            self._stop.wait(self.period_s)

    def start(self):
        self.thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self.thread.join(timeout=30)
        return {"period_s": self.period_s, "samples": self.samples, "peak_mib": dict(sorted(self.peak.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])),
                "gpus": dict(sorted(self.gpus.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]))}


class HostSampler:
    """Resident host memory once a second per rank: the rank process and its descendants (upstream's DataLoader workers featurise in child
    processes) summed with shared pages split among the processes that map them — ``Pss`` of /proc/<pid>/smaps_rollup, `method=pss`: the
    memory a container limit charges the rank. Where smaps_rollup is unreadable (sandboxed /proc) only ``VmRSS`` of /proc/<pid>/status exists,
    which counts a shared page once in EVERY process that maps it — a pool forked from a 30-GB rank reads as workers x 30 GB — so under
    `method=rss_own` the per-rank figure (`host_rss_peak_mib`) is the rank PROCESS's own resident peak and the tree's VmRSS sum rides beside it
    as `host_tree_peak_mib` (an over-count by construction, printed for what it is); under pss both are the tree's Pss sum. Beside them: the
    whole machine's used memory peak (`host_used_peak_mib`: /proc/meminfo MemTotal - MemAvailable, every rank and everything else on the machine).
    Rounds are `period_s` apart (5 s under pss — an smaps_rollup read walks the target's page tables — 1 s under rss); the rank's process tree is
    read from /proc/<p>/task/<p>/children, else from one PPid scan of /proc per round (`tree`); `nproc_peak` = the most processes seen in a
    rank's tree. A process that vanished mid-round counts 0 that round; `samples` = rounds taken."""

    def __init__(self, pids: dict, period_s: Optional[float] = None):
        self.pids, self.peak, self.nproc, self.samples, self._stop = dict(pids), {str(r): 0 for r in pids}, {str(r): 0 for r in pids}, 0, threading.Event()
        self.own = {str(r): 0 for r in pids}                                                 # the rank process's own resident peak (MiB)
        self.method = "pss" if os.path.isfile(f"/proc/{os.getpid()}/smaps_rollup") else "rss_own"
        self.period_s = float(period_s) if period_s is not None else (5.0 if self.method == "pss" else 1.0)   # an smaps_rollup read walks the target's page tables under its mmap lock: seconds apart, not every second
        self.tree = "children" if os.path.isfile(f"/proc/{os.getpid()}/task/{os.getpid()}/children") else "ppid_scan"   # /proc/<p>/task/<p>/children needs CONFIG_PROC_CHILDREN; else one PPid scan of /proc per round
        self.used_peak = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _tree(self, pid: int, children: Optional[Dict[int, List[int]]] = None) -> List[int]:
        out, todo = [], [int(pid)]
        while todo:
            p = todo.pop()
            out.append(p)
            if children is not None:
                todo.extend(children.get(p, ()))
                continue
            try:
                with open(f"/proc/{p}/task/{p}/children") as fh:
                    todo.extend(int(x) for x in fh.read().split())
            except OSError:
                pass
        return out

    @staticmethod
    def _ppid_map() -> Dict[int, List[int]]:
        children: Dict[int, List[int]] = {}
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/status") as fh:
                    for ln in fh:
                        if ln.startswith("PPid:"):
                            children.setdefault(int(ln.split()[1]), []).append(int(d))
                            break
            except (OSError, ValueError):
                pass
        return children

    def _kib(self, pid: int) -> int:
        path, key = (f"/proc/{pid}/smaps_rollup", "Pss:") if self.method == "pss" else (f"/proc/{pid}/status", "VmRSS:")   # kB either way
        try:
            with open(path) as fh:
                for ln in fh:
                    if ln.startswith(key):
                        return int(ln.split()[1])
        except (OSError, ValueError):
            pass
        return 0

    @staticmethod
    def _used_kib() -> int:
        total = avail = None
        try:
            with open("/proc/meminfo") as fh:
                for ln in fh:
                    if ln.startswith("MemTotal:"):
                        total = int(ln.split()[1])
                    elif ln.startswith("MemAvailable:"):
                        avail = int(ln.split()[1])
        except (OSError, ValueError):
            pass
        return (total - avail) if (total is not None and avail is not None) else 0

    def _run(self):
        while not self._stop.is_set():
            self.round()
            self._stop.wait(self.period_s)

    def round(self) -> None:
        """One sampling round over every rank's process tree (the thread's body; callable on its own)."""
        children = self._ppid_map() if self.tree == "ppid_scan" else None
        for r, pid in self.pids.items():
            tree = self._tree(pid, children)
            kibs = [self._kib(p) for p in tree]                                                  # tree[0] is the rank process itself
            self.own[str(r)] = max(self.own[str(r)], (kibs[0] if kibs else 0) >> 10)
            self.peak[str(r)] = max(self.peak[str(r)], sum(kibs) >> 10)
            self.nproc[str(r)] = max(self.nproc[str(r)], len(tree))
        self.used_peak = max(self.used_peak, self._used_kib() >> 10)
        self.samples += 1

    def report(self) -> dict:
        """The record `stop` returns: per rank the reported figure (`host_rss_peak_mib`: the tree's Pss sum under pss, the rank process's own
        VmRSS under rss_own) and the tree sum as read (`host_tree_peak_mib`), the method and tree words, process counts, rounds, machine peak."""
        of_record = self.peak if self.method == "pss" else self.own
        return {"period_s": self.period_s, "samples": self.samples, "method": self.method, "tree": self.tree, "nproc_peak": dict(self.nproc),
                "host_rss_peak_mib": dict(of_record), "host_tree_peak_mib": dict(self.peak), "host_used_peak_mib": self.used_peak}

    def start(self):
        self.thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self.thread.join(timeout=30)
        return self.report()


# ----------------------------------------------------------------------------------------------------------------- ranks ----
def launch_env(records: str, world: int, environ: Optional[dict] = None) -> dict:
    """The environment `launch` hands `spawn_ranks` as the ranks' base: the launcher's own plus its per-launch records directory
    (manifest.RECORDS_ENV) — the ONE place a name under the package prefix is added for a rank, so every OPENFOLD3_OB0_OPT* name a rank
    inherits is this function's output through `rank_env` (each must be one `_autoload.DECLARED` knows: the .pth gate runs first in a rank).
    Plus the launch's ONE hash seed: `PYTHONHASHSEED` for all `world` ranks (opt_core.mem.rowpair.rankdata.ranks_env — the launcher's own value
    when it names one, else 0; a rank interpreter reads it at start-up), so an upstream featurisation step ordered by str hashing yields
    the same bytes in every rank; `launch` prints the seed's census word (`hashseed=… source=default|inherited ranks=P`)."""
    from .manifest import RECORDS_ENV
    from opt_core.mem.rowpair.rankdata import ranks_env
    env, _word = ranks_env(os.environ if environ is None else environ, world)   # + the launch's ONE hash seed (PYTHONHASHSEED, every rank the same: the core's rank input contract)
    return {**env, RECORDS_ENV: records}


RANK_KNOBS = ("OPENFOLD3_OB0_OPT_CONF_DTYPE", "OPENFOLD3_OB0_OPT_CONF_DTYPE_SOURCE", "OPENFOLD3_OB0_OPT_Z_DTYPE", "OPENFOLD3_OB0_OPT_Z_DTYPE_SOURCE")   # the package's own knobs a rank keeps from the launcher's environment (modes.ENV_CONF_DTYPE / _SOURCE: the confidence phase's dtype the caller chose, with its source word)


def rank_env(base: dict, rank: int, world: int, gpu: str, port: int, chunk: int, addr: str = "127.0.0.1") -> dict:
    """The rank's environment: the launcher's minus the kit's own route variables (the rank takes its route from argv), plus the line's
    run variables (modes.RUN_VARS) and its one GPU. The launcher's records directory (manifest.RECORDS_ENV in `base`) passes through, and
    each rank's TMPDIR is its own directory under it (`rank_tmpdir`): upstream's data pipeline keeps per-host defaults under the temporary
    directory — the template preprocessor's `template_data/template_cache/<chain>.npz` entries (written with an exists()-then-skip check and
    a non-atomic save) — and every rank runs that pipeline, so a shared TMPDIR would have P processes rewriting the files the others read."""
    from .manifest import RECORDS_ENV
    env = {k: v for k, v in base.items() if not (k.startswith("OPENFOLD3_OB0_OPT") and k not in ("OPENFOLD3_OB0_OPT_HOME", RECORDS_ENV, *RANK_KNOBS))}
    env.update({"OF3TP_WORLD": str(world), "OF3TP_RANK": str(rank), "OF3TP_ADDR": addr, "OF3TP_PORT": str(port), ENV_CHUNK: str(chunk),
                "CUDA_VISIBLE_DEVICES": gpu})
    env.setdefault("ROWPAIR_CONF_REDUCER", "fast")                          # the label-indexed confidence reducer is the tp line's default; a caller's export (classic | both) wins
    from .tp_rowpair import core as _core
    env.update(_core.fn("dist", "rank_thread_env")(env.get("ROWPAIR_RANK_THREADS", "auto"), world))   # OMP/MKL/OPENBLAS_NUM_THREADS = cpus // P per rank (+ the resolved ROWPAIR_RANK_THREADS the rank applies at install)
    if base.get(RECORDS_ENV):
        env["TMPDIR"] = rank_tmpdir(base[RECORDS_ENV], rank)
    return env


def rank_tmpdir(records: str, rank: int) -> str:
    """Rank `rank`'s private temporary directory: <records>/rank<r>/tmp (inside the caller's own TMPDIR, where `launch` made the records
    directory; removed with it when the launcher returns). `spawn_ranks` creates it before the rank starts (tempfile.gettempdir() falls back
    to the shared default for a TMPDIR that does not exist)."""
    return os.path.join(records, f"rank{rank}", "tmp")


def rank_dir(out_dir: str, rank: int) -> str:
    return out_dir if rank == 0 else os.path.join(out_dir, RANK_DIR, f"rank{rank}")


def rank_log(out_dir: str, rank: int) -> str:
    return os.path.join(out_dir, RANK_DIR, f"rank{rank}.log")


def _tee(pipe, sink, mirror) -> None:
    for line in iter(pipe.readline, b""):
        sink.write(line)
        sink.flush()
        if mirror is not None:
            mirror.write(line.decode("utf-8", "replace"))
            mirror.flush()
    pipe.close()


def spawn_ranks(argv_of_rank, out_dir: str, world: int, gpus: List[str], port: int, chunk: int, base_env: Optional[dict] = None) -> List[dict]:
    """Start the ranks: `argv_of_rank(r, rank_out_dir)` is each rank's command line. Rank 0's log is mirrored to this process's stderr
    (the run's log: its ACTIVE line, the line's `[tp_rowpair hook r0]` / `[rowpair r0]` lines), every rank's goes to <out>/_tp/rank<r>.log."""
    base_env = os.environ if base_env is None else base_env
    os.makedirs(os.path.join(out_dir, RANK_DIR), exist_ok=True)
    procs = []
    for r in range(world):
        od = rank_dir(out_dir, r)
        os.makedirs(od, exist_ok=True)
        env = rank_env(base_env, r, world, gpus[r], port, chunk)
        if env.get("TMPDIR") and env.get("TMPDIR") != base_env.get("TMPDIR"):
            os.makedirs(env["TMPDIR"], exist_ok=True)                        # the rank's private TMPDIR (rank_tmpdir) must exist before its interpreter starts
        sink = open(rank_log(out_dir, r), "wb")
        p = subprocess.Popen(argv_of_rank(r, od), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        t = threading.Thread(target=_tee, args=(p.stdout, sink, sys.stderr if r == 0 else None), daemon=True)
        t.start()
        procs.append({"rank": r, "gpu": gpus[r], "proc": p, "sink": sink, "tee": t, "dir": od, "log": rank_log(out_dir, r), "rc": None, "started": time.time()})
    return procs


def wait_ranks(procs: List[dict], poll_s: float = POLL_S) -> Optional[int]:
    """Wait for every rank; the first non-zero exit kills the others (fail-fast) and is the rank failure returned (None when all exited 0)."""
    failed = None
    while any(p["rc"] is None for p in procs):
        for p in procs:
            if p["rc"] is None:
                rc = p["proc"].poll()
                if rc is not None:
                    p["rc"], p["ended"] = rc, time.time()
                    log(f"rank {p['rank']} exited rc={rc} after {p['ended'] - p['started']:.0f}s")
                    if rc != 0 and failed is None:
                        failed = p["rank"]
                        for q in procs:
                            if q["rc"] is None:
                                log(f"rank {p['rank']} failed: terminating rank {q['rank']}")
                                q["proc"].terminate()
        if failed is not None:
            deadline = time.time() + 30
            for q in procs:
                if q["rc"] is None:
                    try:
                        q["rc"] = q["proc"].wait(timeout=max(0.1, deadline - time.time()))
                    except subprocess.TimeoutExpired:
                        q["proc"].kill()
                        q["rc"] = q["proc"].wait()
                    q["ended"] = time.time()
            break
        time.sleep(poll_s)
    for p in procs:
        p["tee"].join(timeout=10)
        p["sink"].close()
    return failed


# ----------------------------------------------------------------------------------------------------------------- census ----
ALLOC_TAG = "allocator peak:"


def allocator_peak_line() -> str:
    """This process's CUDA allocator high-water (torch.cuda.max_memory_allocated / max_memory_reserved, MiB) for its log — the
    instrument beside the launcher's nvidia-smi sampler; 'no cuda' when torch has no device."""
    try:
        import torch
        if not torch.cuda.is_available():
            return f"{ALLOC_TAG} no cuda"
        return f"{ALLOC_TAG} max_allocated_mib={torch.cuda.max_memory_allocated() // (1024 * 1024)} max_reserved_mib={torch.cuda.max_memory_reserved() // (1024 * 1024)}"
    except Exception as e:                                                   # noqa: BLE001 — the line names the failure rather than dropping the census field
        return f"{ALLOC_TAG} unavailable ({type(e).__name__})"


FALLBACK_TAG = "] FALLBACK "                              # the adapter's marker of a named per-call fallback in a rank log (tp_rowpair.core.log_fallback writes it)
_LAST_CENSUS: dict = {}                                    # the census block of this process's last launch (counters() reads it for the exit line)


POCKET_KEY = "pocket_constraint"                            # upstream's query-level field that turns pocket-conditioned sampling on (of3_all_atom/config/inference_query_format.py: Query.pocket_constraint)


def pocket_constrained(query_json: str) -> List[str]:
    """The names of the queries in `query_json` that carry a pocket constraint (a non-null `pocket_constraint`); [] when none or unreadable."""
    try:
        with open(query_json, encoding="utf-8") as fh:
            q = json.load(fh)
    except (OSError, ValueError):
        return []
    queries = q.get("queries") if isinstance(q, dict) else None
    return [n for n, qq in (queries or {}).items() if isinstance(qq, dict) and qq.get(POCKET_KEY)] if isinstance(queries, dict) else []


def serve_unsharded(a, home: str, ckpt: str, ckpt_info: dict, det_level: int, world: int, reason: str, why: str) -> int:
    """A request the row-sharded line does not shard is SERVED, on one card, through the stock route (`pred --mode off`'s clean child: stock
    arithmetic for the whole pass, bitwise equal to the stock call under the det recipe) — a named per-call fallback of the tp line
    (`FALLBACK <reason>=1` in the log, `fallbacks=tp:<reason>=1` on the census and exit lines), never a refusal."""
    import argparse
    from .cli import resolve_upstream_fix, run_stock_subprocess
    global _LAST_CENSUS
    t0 = time.time()
    stock_call = argparse.Namespace(**dict(vars(a), mode="off"))            # served AS the stock route: `pred --mode off` with the caller's own arguments (its runner yaml if it gave one, else the
    log(f"{FALLBACK_TAG.strip('] ')} {reason}=1: {why} — this request is served UNSHARDED on one card through the stock route (n_gpu={world} requested; "   # stock route's default) — the
        f"`pred --mode off` with {'the caller runner yaml ' + str(a.runner_yaml) if getattr(a, 'runner_yaml', None) else 'the stock configuration'}: stock arithmetic, "               # mode's own yaml members never
        "outputs bitwise equal to that stock call under the det recipe)")                                                                                                            # enter (cli.row_yaml composes per mode)
    _LAST_CENSUS = {"world": world, "ranks_alive_at_exit": 0, "unsharded": reason, "fallbacks": {reason: 1}}
    fix_ids = resolve_upstream_fix(a, home, "pred") if getattr(a, "upstream_fix", None) else None
    rc = run_stock_subprocess(stock_call, home, ckpt, a.query_json, mode_env="big", ckpt_info=ckpt_info, det=det_level, upstream_fix=fix_ids)
    log(f"census world={world} ranks_alive=0/{world} unsharded={reason} fallbacks={fallbacks_word(_LAST_CENSUS['fallbacks'])} wall={time.time() - t0:.1f}s -> exit {rc}")
    log(f"exit mode=big line={getattr(a, 'line', None) or 'tp'} n_gpu={world} served=unsharded fallbacks={fallbacks_word(_LAST_CENSUS['fallbacks'])} -> {rc}")   # this process's exit census (no rank ran: the launcher states the fallback it served)
    return rc


def fallbacks() -> dict:
    """The launcher process's per-call fallback record for the EXIT line (report.FALLBACK_SOURCES, label `tp`): `{<name>=<word>: count}` — the
    named per-call fallbacks the ranks logged in this launch (their FALLBACK_TAG lines, e.g. `sample_loop_pocket=one_pass_forced` for a
    pocket-conditioned query whose sample loop ran as one batched pass), summed over the ranks. Empty before a launch and when no rank fell back;
    a rank process reports its own through tp_rowpair.core.fallbacks."""
    return dict((_LAST_CENSUS.get("fallbacks") or {}) if _LAST_CENSUS else {})


def fallbacks_word(fb: dict) -> str:
    """`none` or `tp:<reason>=<count>,…` — the census line's spelling of a fallback record (the EXIT line's grammar: `<label>:<reason>=<n>`)."""
    return ",".join(f"tp:{k}={v}" for k, v in sorted((fb or {}).items())) or "none"


def parse_rank_log(path: str) -> dict:
    """What the line's own log lines say for one rank: the process group (backend / world / mode / nccl), the rows of the pair representation the
    rank held (`trunk done N=… P=… rows a:b`), the hook's armed line, the rank's own allocator high-water line (allocator_peak_line), the
    named per-call fallbacks it logged (`fallbacks`: `<name>=<word>` -> count, FALLBACK_TAG lines), and the cross-rank input digest gate's
    verdict (`feats`: {gates, ranks_identical, shared} over its `[feats] rank r … ranks_identical=` lines, one per forward — False if any said
    False, True if one said True and none False, None for n/a only; None when the rank wrote no such line)."""
    rec = {"armed": None, "group": None, "rows": None, "N": None, "alloc": None, "fallbacks": {}, "triatt": None, "trimul": None, "refusal": None, "feats": None}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if FALLBACK_TAG in line:                                   # a named per-call fallback of the line's adapter (tp_rowpair.core.FALLBACK_TAG grammar: `FALLBACK <name>=<word>`)
                    tok = line.split(FALLBACK_TAG, 1)[1].split()[:1]
                    if tok and "=" in tok[0]:
                        key = tok[0].rstrip(":")
                        rec["fallbacks"][key] = rec["fallbacks"].get(key, 0) + 1
                if "process group ready:" in line and rec["group"] is None:
                    kv = dict(tok.split("=", 1) for tok in line.split("process group ready:", 1)[1].split() if "=" in tok)
                    rec["group"] = {k: kv[k] for k in ("backend", "world", "mode", "nccl") if k in kv}
                elif "[tp_rowpair hook r" in line and "armed:" in line and rec["armed"] is None:
                    rec["armed"] = line.split("armed:", 1)[1].strip()
                elif TRIATT_TAG in line or TRIMUL_TAG in line:                  # the kernel levers' census, one line each per rank at exit (the last one wins)
                    tag, key = (TRIATT_TAG, "triatt") if TRIATT_TAG in line else (TRIMUL_TAG, "trimul")
                    kv = dict(tok.split("=", 1) for tok in line.split(tag, 1)[1].split() if "=" in tok)
                    rec[key] = {k: (int(v) if v.isdigit() else v) for k, v in kv.items()}
                elif "Refused: " in line and rec["refusal"] is None:           # a rank's refusal sentence (opt_core RowpairRefused: a lever that cannot run and its opt-out; the kit's
                    m = REFUSAL_RE.search(line)                              #  PairstackRefused / ConfidenceRefused / DiffusionRefused / TemplateRefused): the launcher repeats it by name
                    if m:
                        rec["refusal"] = f"{m.group(1)}: {m.group(2).strip()}"
                elif FEATS_TAG in line and f"{FEATS_IDENTICAL}=" in line:            # the cross-rank input digest gate (tp_rowpair/model.feature_digest), one line per forward: its verdict read back
                    word = line.split(f"{FEATS_IDENTICAL}=", 1)[1].split(";", 1)[0].split(")", 1)[0].strip()
                    verdict = {"True": True, "False": False}.get(word)                 # n/a (no multi-rank group) -> None
                    toks = line.split(FEATS_TAG, 1)[1].split()
                    prev = rec["feats"] or {"gates": 0, "ranks_identical": None, "shared": None}
                    seen = (prev["ranks_identical"], verdict)
                    rec["feats"] = {"gates": prev["gates"] + 1, "ranks_identical": False if False in seen else (True if True in seen else None),
                                    "shared": toks[2] if len(toks) > 2 and toks[1] == "shared" else prev["shared"]}
                elif ALLOC_TAG in line and rec["alloc"] is None:
                    kv = dict(tok.split("=", 1) for tok in line.split(ALLOC_TAG, 1)[1].split() if "=" in tok)
                    rec["alloc"] = {k: int(v) for k, v in kv.items() if v.isdigit()} or {"unavailable": line.split(ALLOC_TAG, 1)[1].strip()}
                elif "trunk done N=" in line and rec["rows"] is None:
                    kv = dict(tok.split("=", 1) for tok in line.split("trunk done", 1)[1].split() if "=" in tok)
                    toks = line.split("trunk done", 1)[1].split()
                    if "rows" in toks:
                        a, b = toks[toks.index("rows") + 1].split(":")
                        rec["rows"] = [int(a), int(b)]
                    rec["N"] = int(kv["N"]) if "N" in kv else None
    except OSError:
        pass
    return rec


def other_files(d: str) -> List[str]:
    """Every file under `d` that is not a prediction file (relative paths; the rank census excluded)."""
    out = []
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x != RANK_DIR]
        for f in files:
            if not f.endswith(PRED_PATTERNS):
                out.append(os.path.relpath(os.path.join(root, f), d))
    return sorted(out)


def prediction_files(d: str) -> Dict[str, str]:
    """sha256 of every prediction file under `d` (upstream's structures, confidences and PAE files; its bookkeeping, the kit's records and the
    rank census excluded), keyed by relative path."""
    from opt_core.gates import sha256_file                     # the shared core's one file hasher (no private digest helpers)
    out = {}
    for root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if x != RANK_DIR and not (root == d and x == MSA_CACHE_DIR)]     # the rank tree and upstream's per-process MSA cache are not predictions
        for f in files:
            if f.endswith(PRED_PATTERNS):
                p = os.path.join(root, f)
                out[os.path.relpath(p, d).replace(os.sep, "/")] = sha256_file(p)
    return out


def census(out_dir: str, procs: List[dict], world: int, mode: str, backend: str, chunk: int, chunk_how: str, port: int, pins: dict, memory: dict,
           kernels: Optional[dict] = None) -> dict:
    """The tp block of the run record and its named failures (a list of reasons; empty = every event as expected). ``kernels``: the line's declared
    triangle kernels (tp_rowpair/env.KERNEL_WORDS: {'triatt': 'flash_triattn', 'trimul': 'fpf_v4'}); the ranks' TRIATT / TRIMUL census lines are checked against them."""
    kernels = {"triatt": "torch", "trimul": "torch", **{k: (v or "torch").strip().lower() for k, v in (kernels or {}).items()}}
    ranks, reasons = [], []
    rank0 = prediction_files(rank_dir(out_dir, 0))
    groups = set()
    for p in procs:
        rec = parse_rank_log(p["log"])
        smi = memory["peak_mib"].get(str(p["gpu"]))
        alloc = rec["alloc"] or {}
        entry = {"rank": p["rank"], "gpu": p["gpu"], "gpu_uuid": (memory.get("gpus") or {}).get(str(p["gpu"]), {}).get("uuid"),
                 "rc": p["rc"], "wall_s": round((p.get("ended") or time.time()) - p["started"], 1), "fallbacks": rec.get("fallbacks") or {},   # the named per-call fallbacks this rank logged
                 "peak_mib": smi,                                                 # nvidia-smi high-water of the rank's GPU (the launcher's sampler)
                 "host_rss_peak_mib": ((memory.get("host") or {}).get("host_rss_peak_mib") or {}).get(str(p["rank"])),   # peak resident host memory of the rank: its tree's Pss sum (pss) or the rank process's own VmRSS (rss_own) (HostSampler)
                 "host_tree_peak_mib": ((memory.get("host") or {}).get("host_tree_peak_mib") or {}).get(str(p["rank"])), # the rank tree's summed reading (== the above under pss; a shared-page over-count under rss_own)
                 "allocator_mib": {"max_allocated": alloc.get("max_allocated_mib"), "max_reserved": alloc.get("max_reserved_mib")},   # the rank's own counters
                 "peak_of_record_mib": max(x for x in (smi, alloc.get("max_reserved_mib")) if x is not None) if (smi is not None or alloc.get("max_reserved_mib") is not None) else None,
                 "armed": rec["armed"], "group": rec["group"], "rows": rec["rows"], "N": rec["N"], "triatt": rec["triatt"], "trimul": rec["trimul"],
                 "feats": rec["feats"],                                            # the cross-rank input digest gate as this rank reported it ({gates, ranks_identical, shared} | None: no line)
                 "log": os.path.relpath(p["log"], out_dir)}
        for key, tag, word in (("triatt", TRIATT_TAG, kernels["triatt"]), ("trimul", TRIMUL_TAG, kernels["trimul"])):   # the kernel levers fail-loud: a kernel the line declared
            if word == "torch":                                                       #  and that bound nothing / another word / served nothing for an unexpected reason is a named failure
                continue
            t = rec[key]
            lever = "tp_triatt" if key == "triatt" else "tp_trimul"
            if t is None:
                reasons.append(f"rank {p['rank']}: lever {lever} ({word}) but no `{tag.strip()}` line in its transcript (the lever's census was not written)")
                continue
            bound = str(t.get("kernel" if key == "triatt" else "kernels"))
            fb = {} if str(t.get("fallback_by") or "none") == "none" else dict(x.rsplit(":", 1) for x in str(t["fallback_by"]).split(","))
            undeclared = sorted(r for r in fb if not (r in KERNEL_OPT_OUTS or any(r == g or r.startswith(g + ":") for g in KERNEL_GATES[key])))
            if bound != word:
                reasons.append(f"rank {p['rank']}: lever {lever} ({word}) but the rank bound {bound}")
            elif undeclared:                                                          # the mode is a contract: a unit the kernel declined for any reason but a documented gate or an explicit opt-out fails the run
                reasons.append(f"rank {p['rank']}: lever {lever} ({word}) declined units for an undocumented reason: "
                               f"{','.join(f'{r}:{fb[r]}' for r in undeclared)} (documented gates: {', '.join(KERNEL_GATES[key]) or 'none'}); "
                               f"to run without it set {'ROWPAIR_TRIATT_CORE' if key == 'triatt' else 'ROWPAIR_TRIMUL_KERNELS'}=torch (the torch statements, counted) or use --mode off")
        if rec["alloc"] is None:
            reasons.append(f"rank {p['rank']}: no `{ALLOC_TAG}` line in its transcript (the rank's allocator high-water was not recorded)")
        if p["rc"] != 0:
            reasons.append(f"rank {p['rank']} exited rc={p['rc']}" + (f" — {rec['refusal']}" if rec["refusal"] else ""))
        if rec["armed"] is None:
            reasons.append(f"rank {p['rank']}: no `[tp_rowpair hook r{p['rank']}] armed:` line in its transcript (the line's hook did not arm)")
        if rec["group"] is None:
            reasons.append(f"rank {p['rank']}: no `process group ready` line in its transcript (the ranks never joined the group)")
        else:
            groups.add((rec["group"].get("backend"), rec["group"].get("world"), rec["group"].get("mode")))
            if rec["group"].get("world") != str(world) or rec["group"].get("mode") != mode or rec["group"].get("backend") != backend:
                reasons.append(f"rank {p['rank']}: group {rec['group']} is not the launch's (backend={backend} world={world} mode={mode})")
        if rec["rows"] is None:
            reasons.append(f"rank {p['rank']}: no `trunk done … rows a:b` line (the sharded trunk did not report)")
        if rec["feats"] is None:
            if world > 1:                                                       # the gate runs on every rank of a P>1 group before the forward (tp_rowpair/model.forward_synced): a rank without its line did not reach it
                reasons.append(f"rank {p['rank']}: no `{FEATS_TAG}{p['rank']} … {FEATS_IDENTICAL}=` line in its transcript (the cross-rank input digest gate did not report)")
        elif rec["feats"]["ranks_identical"] is False:                           # the ranks refused by name as well (feats_ranks_differ); the census repeats the verdict
            reasons.append(f"rank {p['rank']}: its input digest gate reported {FEATS_IDENTICAL}=False (the ranks featurised the query differently: feats_ranks_differ)")
        if p["rank"] > 0:                                              # rank 0 writes the prediction; a rank that also wrote one must have written the same bytes
            mine = prediction_files(p["dir"])
            if mine and mine != rank0:
                diff = sorted(set(mine) ^ set(rank0)) + sorted(k for k in mine if k in rank0 and mine[k] != rank0[k])
                reasons.append(f"rank {p['rank']}: its outputs differ from rank 0's ({', '.join(diff[:6])})")
            entry["outputs"] = {"prediction_files": len(mine), "identical_to_rank0": (mine == rank0) if mine else None,
                                "other_files": other_files(p["dir"])}            # upstream's bookkeeping (timing.json, experiment_config.json, ...): named, never compared
        ranks.append(entry)
    rows = sorted((r["rows"] for r in ranks if r["rows"]), key=lambda ab: ab[0])
    n_tok = next((r["N"] for r in ranks if r["N"]), None)
    covered = bool(rows) and rows[0][0] == 0 and all(rows[i][1] == rows[i + 1][0] for i in range(len(rows) - 1)) and (n_tok is None or rows[-1][1] == n_tok)
    if rows and not covered:
        reasons.append(f"the ranks' rows {rows} do not tile 0:{n_tok}")
    verdicts = [r["feats"]["ranks_identical"] if r["feats"] else None for r in ranks]   # the gate's verdict per rank (None: n/a or no line)
    ranks_identical = False if False in verdicts else (True if ranks and all(v is True for v in verdicts) else None)   # True only when EVERY rank reported True; one False anywhere is False; otherwise not claimed
    active = sum(1 for r in ranks if r["armed"] is not None and r["group"] is not None and str(r["group"].get("world")) == str(world))
    if active != world:                                                   # the axis fail-closed: the ranks that armed AND joined a `world`-wide group must number exactly the requested P
        reasons.insert(0, f"{NGPU_MISMATCH} requested={world} active={active}")
    return {"world": world, "mode": mode, "backend": backend, "port": port, "chunk": chunk, "chunk_how": chunk_how, "pins": pins, "n_tokens": n_tok, "n_gpu_active": active,
            "shard_map": {str(r["rank"]): r["rows"] for r in ranks}, "shard_map_covers_N": covered, "groups_reported": sorted(map(list, groups)),
            "ranks": ranks, "ranks_alive_at_exit": sum(1 for r in ranks if r["rc"] == 0), "ranks_identical": ranks_identical,                                                # the cross-rank INPUT digest gate's verdict over the ranks (every rank's `[feats] … ranks_identical=` word): True | False | None (a rank reported no verdict)
            "outputs_identical": (None if not any(r.get("outputs", {}).get("identical_to_rank0") is not None for r in ranks)       # ranks > 0 that also wrote prediction files wrote rank 0's bytes: True | False | None — no rank other than 0 wrote one (rank 0 is the writer: nothing to match)
                                  else all(r.get("outputs", {}).get("identical_to_rank0") is not False for r in ranks)),
            "rank0_files": len(rank0),
            "memory": memory, "peak_mib_per_rank": {str(r["rank"]): r["peak_mib"] for r in ranks},
            "peak_mib_sum": sum(r["peak_mib"] or 0 for r in ranks), "peak_mib_max": max((r["peak_mib"] or 0) for r in ranks) if ranks else None,
            "host_rss_method": (memory.get("host") or {}).get("method"),                     # pss: the tree's Pss, shared pages split (smaps_rollup) | rss_own: the rank process's own VmRSS (sandboxed /proc: no smaps_rollup)
            "host_used_peak_mib": (memory.get("host") or {}).get("host_used_peak_mib"),       # the whole machine's used-memory peak over the run (MemTotal - MemAvailable)
            "host_rss_tree": (memory.get("host") or {}).get("tree"), "host_nproc_peak_per_rank": (memory.get("host") or {}).get("nproc_peak"),   # how the rank's process tree was read; the most processes seen per rank (the rank + its DataLoader workers)
            "host_rss_peak_mib_per_rank": {str(r["rank"]): r["host_rss_peak_mib"] for r in ranks},
            "host_rss_peak_mib_sum": sum(r["host_rss_peak_mib"] or 0 for r in ranks), "host_rss_peak_mib_max": max((r["host_rss_peak_mib"] or 0) for r in ranks) if ranks else None,
            "host_tree_peak_mib_per_rank": {str(r["rank"]): r["host_tree_peak_mib"] for r in ranks},               # the trees' summed readings (VmRSS over-counts shared pages under rss_own)
            "peak_of_record_mib_per_rank": {str(r["rank"]): r["peak_of_record_mib"] for r in ranks},           # max(smi high-water, the rank's max_reserved)
            "peak_of_record_mib_max": max((r["peak_of_record_mib"] or 0) for r in ranks) if ranks else None,
            "peak_of_record_mib_sum": sum(r["peak_of_record_mib"] or 0 for r in ranks),
            "triatt_kernel": kernels["triatt"], "trimul_kernels": kernels["trimul"], "triatt": {str(r["rank"]): r["triatt"] for r in ranks}, "trimul": {str(r["rank"]): r["trimul"] for r in ranks},
            "fallbacks": _sum_fallbacks(ranks),                                                                 # name=word -> count over all ranks; {} = none (the exit line's tp_fallbacks, counters())
            "reasons": reasons, "rank_copies": "removed after the identity check (rank logs and manifests kept)"}


def _sum_fallbacks(ranks) -> dict:
    out: dict = {}
    for r in ranks:
        for k, v in (r.get("fallbacks") or {}).items():
            out[k] = out.get(k, 0) + int(v)
    return out


def remove_rank_copies(out_dir: str, world: int) -> None:
    for r in range(1, world):
        d = rank_dir(out_dir, r)
        for name in os.listdir(d) if os.path.isdir(d) else []:
            p = os.path.join(d, name)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)


# ----------------------------------------------------------------------------------------------------------------- the launcher ----
def judge_templates(a, man: dict, code: int, reason: str, block: dict, ok_codes: Tuple[int, ...] = (0,)) -> Tuple[int, str, dict]:
    """The template guard of a sharded run, judged HERE — the process that took the declaration (`cli.cmd_pred`: `templ_census.record_file`, the
    `TEMPLATES DECLARED` line) — against what rank 0's model process FEATURISED (its run record's "templates" entry: `report`'s forward wrap
    observes every predicted item in the rank), exactly as the stock route judges its child's proof (`cli.templates_guard(rc, stock_proof)`): one
    `TEMPLATES FEATURISED` line per declared-templated query that kept real slots, one `TEMPLATES DROPPED` line per query that lost them, printed once
    in this process. A drop is the run's exit `EXIT_TEMPLATES_DROPPED` unless `--allow-template-drop` (cli.exit_rule's clause, applied to this
    launcher's code) and is counted in the census (`fallbacks=templates_dropped:<n>`). Returns (code, reason, the guard record)."""
    from .cli import EXIT_TEMPLATES_DROPPED, templates_guard
    feats = dict(((man.get("templates") or {}).get("featurised")) or {})   # rank 0's featurised record ({} when the rank never reached the model)
    guard = templates_guard(code if code in ok_codes else 1, {"templates_featurised": feats})
    dropped = len(guard.get("dropped") or [])
    if dropped:
        block.setdefault("fallbacks", {})["templates_dropped"] = dropped      # the census word (report's name for the guard's fallback), beside the ranks' own
        allow = bool(getattr(a, "allow_template_drop", False))
        if code in ok_codes and not allow:
            code, reason = EXIT_TEMPLATES_DROPPED, (f"TEMPLATES DROPPED: {dropped} templated quer{'y' if dropped == 1 else 'ies'} reached the model with a declared chain's templates dropped "
                                                    "(fallbacks=templates_dropped); --allow-template-drop is the recorded opt-out")
        elif code in ok_codes:
            reason = f"{reason}; template drop accepted by --allow-template-drop (recorded; {dropped} quer{'y' if dropped == 1 else 'ies'})"
    return code, reason, guard


def launch(a, home: str, ckpt: str, ckpt_info: dict, det_level: int, world: int, exit_codes: Tuple[int, int, int, int]) -> int:
    """`pred --mode big --n_gpu P` (P > 1) from a non-rank process: spawn the ranks (their run records go to a per-launch temporary directory,
    manifest.RECORDS_ENV, read back for rank 0's exit rule and template guard and removed), census, return the exit code (rank 0's exit rule; a
    named tp failure -> EXIT_FAIL, the reasons printed)."""
    from . import manifest as _manifest
    from .cli import row_yaml
    from .tp_rowpair.env import KERNEL_WORDS, triatt_word
    EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = exit_codes
    pocket = pocket_constrained(a.query_json)
    if pocket:                                                              # pocket-conditioned sampling (SampleDiffusion's restart from parents drawn across samples): served exactly, not sharded —
        return serve_unsharded(a, home, ckpt, ckpt_info, det_level, world, "pocket_unsharded",   # the sharded sampler draws the restart parents in another order than stock's process
                               f"quer{'y' if len(pocket) == 1 else 'ies'} {','.join(pocket)} carr{'ies' if len(pocket) == 1 else 'y'} `{POCKET_KEY}` (pocket-conditioned sampling); "
                               "the row-sharded sampler's pocket restart draws its parents in another order than stock's process (sharded pocket restart with an audited draw order is the follow-on)")
    try:
        chunk, chunk_how = chunk_of_run(a.query_json, world=world)          # the chunk plan (polymer residues; a caller's OF3TP_CHUNK) = the row layout's alignment
        n_res = polymer_residues(a.query_json)
    except ValueError as e:
        log(f"NOT ACTIVE: {e}")
        return EXIT_NOT_ACTIVE
    except OSError:                                                          # an unreadable query is the run's to refuse by name (upstream's parser), not this check's
        chunk, chunk_how, n_res = None, None, None
    if n_res is not None:
        v = FLOOR.verdict(n_res, world, chunk)                               # the sharding floor on the residue count (a lower bound of the token count; every rank re-checks the
        if not v["ok"]:                                                      # featurised count before any collective, floor.check_featurised): served exactly on one card, not sharded
            return serve_unsharded(a, home, ckpt, ckpt_info, det_level, world, FLOOR.REASON,
                                   f"the query's {FLOOR.words(v, 'polymer residues')} — polymer residues are the launcher's lower bound of the token count "
                                   f"(chunk plan: {chunk_how})")
    line = getattr(a, "line", None) or "tp"
    ln = modes.LINES[("big", line)]
    mode_s, backend = MODE_S, BACKEND
    from opt_core.mem import ngpu
    gpus = gpu_census()
    try:                                                                     # fewer visible GPUs than P: refused by name (opt_core.mem.ngpu's sentence), never a smaller run
        ngpu.refuse_unless_visible(world, len(gpus))
    except ngpu.NGpuRefused as e:
        log(f"NOT ACTIVE: {e.reason} ({', '.join(gpus) or 'no GPU'}; CUDA_VISIBLE_DEVICES / nvidia-smi -L)")
        return EXIT_NOT_ACTIVE
    gpus = gpus[:world]
    mem = card_memory()
    note = card_note(mem)
    if note:                                                                 # a smaller card than the plan's table: named up front, the run proceeds on the table's chunk (may OOM; OF3TP_CHUNK tunes it)
        log(note)
    if chunk is None:                                                        # the query file was unreadable above: the plan reads it again here and its error is the run's
        try:
            chunk, chunk_how = chunk_of_run(a.query_json, world=world)
        except (ValueError, OSError) as e:
            log(f"NOT ACTIVE: {e}")
            return EXIT_NOT_ACTIVE
    os.makedirs(a.output_dir, exist_ok=True)
    yml = os.path.join(a.output_dir, PINNED_YAML)
    pins = pinned_yaml(a.runner_yaml or row_yaml(home, ln, mode="big"), chunk, yml)   # a caller's yaml arrives here already composed under the line's member (cli.cmd_pred, once); else the member itself
    port = free_port()
    from opt_core.mem.rowpair import rankdata
    from .tp_rowpair import model as TPM
    try:                                                                     # the data form (OF3TP_DATA_FORM: rank0_bcast | per_rank), refused by name before any rank starts
        form = TPM.data_form(os.environ)
    except Exception as e:                                                   # noqa: BLE001 — the refusal carries the sentence
        log(f"NOT ACTIVE: {getattr(e, 'reason', e)}")
        return EXIT_NOT_ACTIVE
    log(f"launch {ngpu.active_fields(world, 'rowpair')} mode={mode_s} backend={backend} gpus={','.join(gpus)} card_bytes={mem if mem is not None else 'unknown'} "
        f"chunk={chunk} ({chunk_how}) port={port} {rankdata.hashseed_word(os.environ, world)} {rankdata.data_form_word(form)} yaml={yml}")

    def argv_of_rank(r: int, od: str) -> List[str]:                  # a rank is a PRIMARY process: this launcher never activates the package (no model runs here), so each rank resolves
        # the line itself (its ACTIVE line, gates and exit tally are the rank's; rank 0's log is mirrored here) and the processes a rank starts (0.5.0's forkserver DataLoader
        # workers) inherit that rank's frozen resolution (modes.ENV_RESOLVED).
        sub = [sys.executable, "-m", "openfold3_ob0_opt", "pred", "--mode", "big", "--n_gpu", str(world), "--query-json", a.query_json,
               "--output-dir", od, "--det", str(a.det), *(["--ckpt", ckpt] if ckpt else ["--inference-ckpt-name", a.inference_ckpt_name]), "--runner-yaml", yml]
        for k, v in (("--num-model-seeds", a.num_model_seeds), ("--num-diffusion-samples", a.num_diffusion_samples), ("--use-templates", a.use_templates),
                     ("--use-msa-server", a.use_msa_server)):
            if v is not None:
                sub += [k, str(v)]
        if getattr(a, "upstream_fix", None):
            sub += ["--upstream-fix", a.upstream_fix]                   # every rank applies the requested upstream fixes in its own process (cli.apply_upstream_fix)
        return sub

    records = tempfile.mkdtemp(prefix="openfold3_ob0_opt_tp_")               # the ranks' run records (cli.gated_manifest): read back below, removed before this returns
    mem = MemorySampler().start()
    t0 = time.time()
    procs = spawn_ranks(argv_of_rank, a.output_dir, world, gpus, port, chunk, base_env=launch_env(records, world))
    log(f"ranks started: pids {','.join(str(p['proc'].pid) for p in procs)}; rank logs {os.path.join(a.output_dir, RANK_DIR)}/rank<r>.log; "
        f"per-rank TMPDIR {records}/rank<r>/tmp (upstream's template/MSA pipeline defaults are per rank)")
    hostmem = HostSampler({p["rank"]: p["proc"].pid for p in procs if getattr(p["proc"], "pid", None)}).start()
    failed = wait_ranks(procs)
    memory = mem.stop()
    memory["host"] = hostmem.stop()                                          # per-rank peak resident host memory (the rank process + its DataLoader workers)
    below = [p["rank"] for p in procs if p["rc"] == FLOOR.EXIT_BELOW_FLOOR]
    if below:                                                                # the featurised token count (ligand atoms, modified residues counted) put the query below the floor at the pinned
        log(f"rank{'s' if len(below) > 1 else ''} {','.join(map(str, below))} exited {FLOOR.EXIT_BELOW_FLOOR}: the featurised query is below the sharding floor "   # chunk: no rank issued a collective (under rank0_bcast the ranks joined the group and left it);
            f"(rank 0's line above); no collective was issued — the request is served unsharded")                                                                    # the same answer as the residue check
        shutil.rmtree(records, ignore_errors=True)
        return serve_unsharded(a, home, ckpt, ckpt_info, det_level, world, FLOOR.REASON,
                               f"the ranks' featurised token count is below the sharding floor at the pinned chunk {chunk} (rank transcript: `below the sharding floor: …`; "
                               f"the launcher's residue count {n_res} was a lower bound)")
    block = census(a.output_dir, procs, world, mode_s, backend, chunk, chunk_how, port, pins, memory,
                   kernels=dict(KERNEL_WORDS, triatt=triatt_word(launch_cc())))   # the tp_triatt word is a function of the device class (tier door on 9.0 / 8.0), read here as the ranks read it
    block.update(rankdata.hashseed_fields(os.environ))                     # the launch's one hash seed (the `launch` line's hashseed= word) in the run record's tp block
    block["data_form"] = form                                               # who featurised: rank0_bcast | per_rank (the `launch` line's data_form= word)
    block["wall_s"] = round(time.time() - t0, 1)
    rec0 = os.path.join(records, "rank0.json")                            # rank 0's run record (its exit rule and template guard); a rank that died before its exit rule left none
    man = json.load(open(rec0, encoding="utf-8")) if os.path.isfile(rec0) else _manifest.build(None, command="pred", out_dir=a.output_dir, checkpoint=ckpt_info,
                                                                                                    runner_yaml=yml, det=det_level)

    for note in (man.get("notes") or []):                                   # K.T24a: e.g. 'confidence not written: N structure-only model file(s) ...'
        log(f"NOTE {note}")
    shutil.rmtree(records, ignore_errors=True)
    rc0 = procs[0]["rc"]
    if block["reasons"] and block["reasons"][0].startswith(NGPU_MISMATCH):
        code, reason = EXIT_NOT_ACTIVE, "; ".join(block["reasons"])
        log(f"NOT ACTIVE: reason={block['reasons'][0]} (the ranks that armed and joined a {world}-wide group; never a pass on fewer GPUs than requested)")
    elif block["reasons"]:
        code, reason = (EXIT_FAIL if rc0 in (0, None) else rc0), "; ".join(block["reasons"])
    else:
        code, reason = (rc0 if rc0 is not None else EXIT_FAIL), man.get("exit_rule", {}).get("reason", "rank 0's exit rule")
    code, reason, man["templates"] = judge_templates(a, man, code, reason, block, (EXIT_OK,))
    block["exit_code"], block["reason"] = code, reason
    man["tp"] = block
    man["exit_code"] = code
    global _LAST_CENSUS
    _LAST_CENSUS = block
    if not block["reasons"]:
        remove_rank_copies(a.output_dir, world)
    peak = ", ".join(f"gpu{r['gpu']}={r['peak_mib']}" for r in block["ranks"])
    hostpeak = ", ".join(f"r{r['rank']}={r['host_rss_peak_mib']}" for r in block["ranks"])
    treepeak = ", ".join(f"r{r['rank']}={r['host_tree_peak_mib']}" for r in block["ranks"])
    log(f"census world={world} ranks_alive={block['ranks_alive_at_exit']}/{world} group={block['groups_reported']} shard_map={block['shard_map']} covers_N={block['shard_map_covers_N']} "
        f"ranks_identical={block['ranks_identical']} outputs_identical={block['outputs_identical']} smi_peak_mib[{peak}] smi_max={block['peak_mib_max']} smi_sum={block['peak_mib_sum']} "
        f"host_rss_peak_mib[{hostpeak}] host_max={block['host_rss_peak_mib_max']} host_sum={block['host_rss_peak_mib_sum']} host_method={block['host_rss_method']} host_tree_peak_mib[{treepeak}] host_used_peak_mib={block['host_used_peak_mib']} data_form={block.get('data_form')} "
        f"peak_of_record_max={block['peak_of_record_mib_max']} peak_of_record_sum={block['peak_of_record_mib_sum']} triatt_kernel={block['triatt_kernel']} trimul_kernels={block['trimul_kernels']} "
        f"fallbacks={fallbacks_word(block['fallbacks'])} wall={block['wall_s']}s -> exit {code} ({reason})")
    if block["reasons"]:
        log("FAILED: " + " | ".join(block["reasons"]))
    return code
