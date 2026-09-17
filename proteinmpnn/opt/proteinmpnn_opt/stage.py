"""Staging: the kit's own install step, performed per run on copies — never on the carried bytes.

The worker runs from ``kit/`` of the parser's directory (``kit/mpnn_worker2.py`` beside ``kit/fast_parse.py``), a write into a carried
directory. The tree keeps one byte-identical copy of both opt/forward/ directories, so every pass of the kit route stages: ``stage_base``
copies both directories into a scratch directory, checks every copied file against its source and performs the overlay there
(``addon/mpnn_worker2.py`` -> ``kit/mpnn_worker2.py``).
``stage_base`` then derives the exact executable: ``stage_lowmem`` writes ``kit/mpnn_worker2_lowmem.py`` = the
checked staged worker with ``lowmem.py`` inlined after its ``protein_mpnn_utils`` import (installing the low-memory featuriser into that
module) and its own two dense decoding-order masks replaced (``lowmem_worker_source``: every anchor must occur exactly once, else
ActivationError — a changed worker is refused, never guessed at). The executable then carries the ITEM line's clocks
(``itemtime_worker_source``: timing and allocator reads around the executable's own calls, nothing of its
arithmetic). The stock route stages nothing: mode ``off`` runs upstream's own files where they are (stock_run.py). A staged directory is
disposable and removed when the pass ends. A caller staging by hand through this module (``python -c "from proteinmpnn_opt import stage;
..."``) makes it an entry: statement one is the core pin gate (``core_gate``), before the modules below that import the core.
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Dict, List, Sequence

from . import core_gate

core_gate()

from . import TAG, ActivationError, modes, stack  # noqa: E402  (after the gate by design)

OVERLAY_BASE = ("mpnn_worker2.py",)                                    # addon/<f> -> <parser dir>/kit/<f>
LOWMEM_WORKER = "mpnn_worker2_lowmem.py"                                 # kit/<f>: the staged worker transformed by lowmem_worker_source then itemtime_worker_source (the base variants' exact line)

# lowmem_worker_source anchors in addon/mpnn_worker2.py (each exactly once): the utils import the inlined lowmem.py follows, and the end-of-run
# record the lever evidence joins ("lowmem": kit_run.lever_evidence reads it).
WORKER_UTILS_IMPORT = "from protein_mpnn_utils import tied_featurize, ProteinMPNN, StructureDatasetPDB, _scores, _S_to_seq, gather_nodes, cat_neighbors_nodes\n"
WORKER_RECORD_TAIL = '"stock_shape_enc": args.stock_shape_enc, "graph_stats": w.graph_stats})'

# The per-item time line, printed by the kit executable after stage-time instrumentation, once per item at its loop boundary
# (itemtime_worker_source), each followed by its PEAK line, and the STACK line, printed once per designing process
# before its first item is stamped. One grammar (ITEM_LINE_RE; the tail ITEM_STAMP_FMT is spelled once — the forward_s / batch_* fields are
# optional in the grammar so a line carrying only the name and the two stamps parses too):
#   ``[proteinmpnn-opt] ITEM name=<item> forward_s=<s> batch_items=<K> batch_forward_s=<s> batch_call_s=<s> pid=<pid> t_start=<epoch s> t_end=<epoch s>``
#   ``[proteinmpnn-opt] PEAK item=<item> pid=<pid> alloc_gib=<GiB> reserved_gib=<GiB>`` (PEAK_LINE_FMT / PEAK_LINE_RE) after it
# forward_s is the item's equal share of its batch's forward span — the same work upstream's own `generated in` line covers (sampling, the
# re-scoring forward, the output writes; featurise and the native-score forward excluded): the worker's sync'd `sample` + `rescore` +
# `d2h_write` timers over one padded batch of K backbones (each item holds one Lmax slot, so the share is the item's cost).
# batch_call_s: the batch's own wall, featurisation to last file written (batches overlap: the worker pipelines them across CUDA streams).
# pid: os.getpid() of the designing process (several may share one device; ITEM_LINE_RE group ``proc``).
# t_start / t_end: time.time() wall-clock epoch seconds (comparable across processes), each taken after a cuda.synchronize, immediately around
# the span batch_call_s measures — the worker's batch call (every backbone of a K-batch carries its batch's two stamps). The PEAK line's alloc_gib / reserved_gib:
# torch.cuda.max_memory_allocated() / max_memory_reserved() of the process read at t_end, bytes / 2**30 — the caching allocator's running peaks
# since the process started, read as found and never reset; 0.000 without a CUDA device.
# name= is the item's own name (protein['name'], the seqs/<name>.fa stem).
# The OUTPUTS_WRITTEN line closes the pass: printed once per designing process after its last output file is written and closed —
# ``[proteinmpnn-opt] OUTPUTS_WRITTEN pid=<pid> t=<epoch s> n_items=<items designed by this process> n_designs=<sequences written>`` — by the
# worker right after its batch loop (its writes are synchronous: a failed write raises and the process exits non-zero). n_designs = n_items x the
# sequences written per item (the worker's ``seqs_per_item``: batch_size x the (temperature, batch) rounds).
# Timing and reads only: the clocks, the synchronize, the two allocator reads and the prints sit outside the timed statements; no tensor, RNG,
# reduction, backend switch or allocator counter is touched, no write is moved.
ITEM_STAMP_FMT = " pid=%d t_start=%.6f t_end=%.6f"
ITEM_LINE_FMT = "[" + TAG + "] ITEM name=%s forward_s=%.4f batch_items=%d batch_forward_s=%.4f batch_call_s=%.4f" + ITEM_STAMP_FMT
ITEM_LINE_RE = (r"\[" + re.escape(TAG) + r"\] ITEM name=(?P<name>\S+)"
                r"(?: forward_s=(?P<s>[\d.]+) batch_items=(?P<batch_items>\d+) batch_forward_s=(?P<batch_forward_s>[\d.]+) batch_call_s=(?P<batch_call_s>[\d.]+))?"
                r" pid=(?P<proc>\d+) t_start=(?P<t_start>[\d.]+) t_end=(?P<t_end>[\d.]+)")
PEAK_LINE_FMT = "[" + TAG + "] PEAK item=%s pid=%d alloc_gib=%.3f reserved_gib=%.3f"
PEAK_LINE_RE = r"\[" + re.escape(TAG) + r"\] PEAK item=(?P<item>\S+) pid=(?P<proc>\d+) alloc_gib=(?P<alloc_gib>[\d.]+) reserved_gib=(?P<reserved_gib>[\d.]+)"
# The STACK line: ``[proteinmpnn-opt] STACK pid=<os.getpid()> torch=<torch.__version__> cuda=<torch.version.cuda> device=<cuda device 0 name, blanks as _ | cpu>
# cc=<major.minor | none> tf32_matmul=<torch.backends.cuda.matmul.allow_tf32> cudnn_tf32=<torch.backends.cudnn.allow_tf32>
# f32_matmul_precision=<torch.get_float32_matmul_precision()> cudnn_benchmark=<torch.backends.cudnn.benchmark>
# deterministic_algorithms=<torch.are_deterministic_algorithms_enabled()> alloc_conf=<$PYTORCH_CUDA_ALLOC_CONF | unset>
# cublas_workspace=<$CUBLAS_WORKSPACE_CONFIG | unset>`` — every field one whitespace-free token, read from the process's own torch as found
# (``none`` in each torch field when the process holds no torch module); nothing is set. STACK_LINE_RE names the pid group ``proc``, as ITEM_LINE_RE does.
OUTPUTS_WRITTEN_FMT = "[" + TAG + "] OUTPUTS_WRITTEN pid=%d t=%.6f n_items=%d n_designs=%d"
# The PREPASS line: the kit worker's stream-offset pre-pass reads every input backbone once (length and gap status, `featurize_one`) before the
# batch loop — work proportional to the inputs, outside every ITEM span; printed once when that loop ends, its own epoch stamps around it:
# ``[proteinmpnn-opt] PREPASS pid=<pid> t_start=<epoch s> t_end=<epoch s> n=<backbones read>``.
PREPASS_FMT = "[" + TAG + "] PREPASS pid=%d t_start=%.6f t_end=%.6f n=%d"
PREPASS_RE = r"\[" + re.escape(TAG) + r"\] PREPASS pid=(?P<proc>\d+) t_start=(?P<t_start>[\d.]+) t_end=(?P<t_end>[\d.]+) n=(?P<n>\d+)"
OUTPUTS_WRITTEN_RE = r"\[" + re.escape(TAG) + r"\] OUTPUTS_WRITTEN pid=(?P<proc>\d+) t=(?P<t>[\d.]+) n_items=(?P<items_n>\d+) n_designs=(?P<designs>\d+)"
STACK_FIELDS = ("pid", "torch", "cuda", "device", "cc", "tf32_matmul", "cudnn_tf32", "f32_matmul_precision", "cudnn_benchmark", "deterministic_algorithms", "alloc_conf", "cublas_workspace")
STACK_LINE_FMT = "[" + TAG + "] STACK " + " ".join(f + "=%s" for f in STACK_FIELDS)
STACK_LINE_RE = r"\[" + re.escape(TAG) + r"\] STACK " + " ".join(f + "=" + (r"(?P<proc>\d+)" if f == "pid" else "(?P<" + f + r">\S+)") for f in STACK_FIELDS)

ITEMTIME_MAIN_DEF = "def main("                                          # both kit executables: the helpers go in front of the one line holding its main() definition (module level, found once)
ITEMTIME_WORKER_PHASES = ("sample", "rescore", "d2h_write")            # the worker's own timers whose per-batch delta is the forward span (Worker.timers)
ITEMTIME_WORKER_DONE = 'done += fin["n"]'                                                      # main(): the one statement run per completed batch, `fin` its record (found once)
ITEMTIME_WORKER_PREPASS = 'w.offsets = offs; T["t_offsets_s"] = time.time()-t0; T["stream_total_offset"] = o'   # main(): the stream-offset pre-pass's closing statement (found once; its t0 = time.time() at the loop's start)
ITEMTIME_WORKER_LOOP = 'for fin in w.run_pipelined([[proteins[j] for j in bi] for bi in batches], chain_id_dict, args.out_folder, args.batch_size):'   # main(): the batch loop's header (found once); the STACK line precedes it, the OUTPUTS_WRITTEN line follows its block

# The helpers every designing executable receives in front of its main() (``_insert_helpers``): the clocks, the synchronize, the allocator reads and
# the STACK line, spelled once. They reach torch through sys.modules — the process's own torch once the executable imported it (both import it
# before their first item) — so the block imports nothing of torch itself: without a torch module the torch fields read ``none``, without a CUDA
# device the peaks read 0.0 and the device ``cpu``.
ITEMTIME_HELPERS = (
    "# ---- proteinmpnn_opt stage: the ITEM line's clocks and the STACK line (stage.ITEMTIME_HELPERS; timing and reads only) ----\n"
    "import os as _it_os, sys as _it_sys, time as _it_time\n"
    "_IT = {\"forward_s\": 0.0, \"stack_printed\": False, \"items\": 0}\n"
    "def _it_torch():\n"
    "    return _it_sys.modules.get(\"torch\")                                # the process's torch once the executable imported it; nothing is imported here\n"
    "def _it_cuda():\n"
    "    _m = _it_torch()\n"
    "    return _m if _m is not None and _m.cuda.is_available() else None\n"
    "def _it_sync():\n"
    "    if _it_cuda() is not None: _it_cuda().cuda.synchronize()\n"
    "def _it_now():\n"
    "    _it_sync(); return _it_time.time()                                 # an epoch stamp taken once the device is idle\n"
    "def _it_peak():\n"
    "    _m = _it_cuda()                                                     # the caching allocator's running peaks, read as found (GiB); never reset\n"
    "    return (0.0, 0.0) if _m is None else (_m.cuda.max_memory_allocated() / 2 ** 30, _m.cuda.max_memory_reserved() / 2 ** 30)\n"
    "def _it_tok(_v):\n"
    "    _s = \"none\" if _v is None else str(_v).replace(\" \", \"_\")\n"
    "    return _s if _s else \"unset\"\n"
    "def _it_stack():\n"
    "    if _IT[\"stack_printed\"]: return                                      # once per process, before the first item is stamped\n"
    "    _IT[\"stack_printed\"] = True\n"
    "    _m, _c = _it_torch(), _it_cuda()\n"
    "    if _m is None:\n"
    "        _f = (None,) * 9\n"
    "    else:\n"
    "        _f = (_m.__version__, _m.version.cuda, _c.cuda.get_device_name(0) if _c is not None else \"cpu\",\n"
    "              \"%d.%d\" % tuple(_c.cuda.get_device_capability(0)) if _c is not None else None,\n"
    "              _m.backends.cuda.matmul.allow_tf32, _m.backends.cudnn.allow_tf32, _m.get_float32_matmul_precision(), _m.backends.cudnn.benchmark,\n"
    "              _m.are_deterministic_algorithms_enabled())\n"
    "    _f = (_it_os.getpid(),) + tuple(_it_tok(_x) for _x in _f) + (_it_tok(_it_os.environ.get(\"PYTORCH_CUDA_ALLOC_CONF\") or \"\"), _it_tok(_it_os.environ.get(\"CUBLAS_WORKSPACE_CONFIG\") or \"\"))\n"
    "    print(" + repr(STACK_LINE_FMT) + " % _f, flush=True)\n"
    "def _it_outputs_written(_n_items, _n_designs):\n"
    "    print(" + repr(OUTPUTS_WRITTEN_FMT) + " % (_it_os.getpid(), _it_now(), _n_items, _n_designs), flush=True)   # once, after the last output file is written and closed\n"
    "# ---- end of the ITEM line's clocks ----\n")


def _copytree(src: str, dst: str) -> int:
    shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=False, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return sum(len(fs) for _, _, fs in os.walk(dst))


def _check_copy(dst: str, src: str) -> List[str]:
    """Every file of ``src`` against its copy at ``dst`` (src-relative names) — a staging copy is disposable, so this catches a copy that
    landed wrong, not tampering: the tree's git commit names the bytes, nothing is re-hashed here."""
    bad = []
    for dp, dns, fs in os.walk(src):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for f in fs:
            if f.endswith(".pyc"):
                continue
            rel = os.path.relpath(os.path.join(dp, f), src)
            got = stack.sha256(os.path.join(dst, rel)) if os.path.isfile(os.path.join(dst, rel)) else None
            if got != stack.sha256(os.path.join(dp, f)):
                bad.append(rel)
    return bad


def stage_base(stage_dir: str) -> Dict[str, str]:
    """Copy both opt/forward/ directories under ``stage_dir``, check each copy against its source, overlay, derive kit/mpnn_worker2_lowmem.py. Returns the paths
    a run needs."""
    os.makedirs(stage_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    for sub in (modes.PARSER_DIR, modes.WORKER_DIR):
        src, dst = os.path.join(stack.kit_home(), sub), os.path.join(stage_dir, sub)
        _copytree(src, dst)
        bad = _check_copy(dst, src)
        if bad:
            raise ActivationError(f"staged copy of {sub} does not match its source: {', '.join(bad[:5])}")
        paths[sub] = dst
    parser, worker = paths[modes.PARSER_DIR], paths[modes.WORKER_DIR]
    for f in OVERLAY_BASE:
        src, dst = os.path.join(worker, "addon", f), os.path.join(parser, "kit", f)
        shutil.copy2(src, dst)
        if stack.sha256(dst) != stack.sha256(src):
            raise ActivationError(f"overlay {f}: the copy in the staged kit does not match {modes.WORKER_DIR}/addon/{f}")
    paths["worker"] = os.path.join(parser, "kit", "mpnn_worker2.py")
    paths.update(stage_lowmem(paths))                                     # kit/mpnn_worker2_lowmem.py: the base variants' exact executable
    paths["fast_parse"] = os.path.join(parser, "kit", "fast_parse.py")
    return paths


def lowmem_worker_source(worker_src: str) -> str:
    """The base variants' exact executable, from the carried worker's text: lowmem.py inlined after the utils import and installed into protein_mpnn_utils,
    the worker's two dense decoding-order masks (decoder_scores: upstream's spelling; sample: its own) replaced by lowmem.mask_attend, and
    ``lowmem`` / ``lowmem_sites`` / ``peak_alloc_GB`` joined to the end-of-run record. ValueError when an anchor is not found exactly once."""
    from . import lowmem
    for anchor in (WORKER_UTILS_IMPORT, WORKER_RECORD_TAIL):
        if worker_src.count(anchor) != 1:
            raise ValueError(f"worker anchor found {worker_src.count(anchor)} time(s), expected 1: {anchor[:60]}...")
    with open(lowmem.__file__, encoding="utf-8") as fh:
        inline = fh.read()
    block = (WORKER_UTILS_IMPORT
             + "# ---- proteinmpnn_opt/lowmem.py inlined by stage.stage_lowmem (the low-memory featuriser + decoding-order mask) ----\n"
             + inline.rstrip("\n") + "\n"
             + "import protein_mpnn_utils as _lm_utils\n"
             + "LOWMEM_SITES = install(_lm_utils) + ['Worker.decoder_scores', 'Worker.sample']\n"
             + "_lm_mask_attend = mask_attend\n"
             + "print('[proteinmpnn-opt] lowmem: low-memory featuriser + decoding-order mask installed (' + ', '.join(LOWMEM_SITES) + ')', flush=True)\n"
             + "# ---- end of the inlined lowmem.py ----\n")
    src = lowmem.replace_dense_mask(worker_src, lowmem.DENSE_MASK_UPSTREAM)   # Worker.decoder_scores (upstream's spelling)
    src = lowmem.replace_dense_mask(src, lowmem.DENSE_MASK_WORKER)            # Worker.sample (the worker's spelling)
    src = src.replace(WORKER_UTILS_IMPORT, block)                            # after the sites: the inlined text quotes the statements it replaces
    src = src.replace(WORKER_RECORD_TAIL, WORKER_RECORD_TAIL[:-2]
                      + ', "lowmem": True, "lowmem_sites": LOWMEM_SITES, "peak_alloc_GB": (round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None)})')
    compile(src, LOWMEM_WORKER, "exec")                                   # a syntax error here is the package's, raised before any launch
    return src



def _wrap_statement(src: str, stmt: str, before: Sequence[str], after: Sequence[str], what: str) -> str:
    """``src`` with the one line holding ``stmt`` preceded by ``before`` and followed by ``after`` at that line's own indentation.
    ValueError unless ``stmt`` is found on exactly one line."""
    lines = src.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if stmt in ln]
    if len(hits) != 1:
        raise ValueError(f"{what}: statement found {len(hits)} time(s), expected 1: {stmt[:60]}...")
    i = hits[0]
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines[i:i + 1] = [indent + b + "\n" for b in before] + [lines[i]] + [indent + a + "\n" for a in after]
    return "".join(lines)


def _after_block(src: str, header: str, lines_after: Sequence[str], what: str) -> str:
    """``src`` with ``lines_after`` inserted right after the block opened by the one line holding ``header`` (a ``for`` / ``with`` / ``try``
    header), at the header's own indentation: before the first non-blank line indented no deeper than the header, or at the end of the text.
    ValueError unless ``header`` is found on exactly one line."""
    lines = src.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if header in ln]
    if len(hits) != 1:
        raise ValueError(f"{what}: statement found {len(hits)} time(s), expected 1: {header[:60]}...")
    i = hits[0]
    indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    j = i + 1
    while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > len(indent)):
        j += 1
    if j > 0 and not lines[j - 1].endswith("\n"):
        lines[j - 1] += "\n"
    lines[j:j] = [indent + a + "\n" for a in lines_after]
    return "".join(lines)


def insert_helpers(src: str, extra: str, what: str) -> str:
    """``src`` with ITEMTIME_HELPERS (then ``extra``: the executable's own item generator, if any) in front of the one line holding its ``def main(``
    (ITEMTIME_MAIN_DEF, module level). ValueError unless that line is found exactly once."""
    block = ITEMTIME_HELPERS + extra
    return _wrap_statement(src, ITEMTIME_MAIN_DEF, before=block.rstrip("\n").split("\n"), after=[], what=f"{what} main() definition")


def itemtime_worker_source(worker_src: str, filename: str = LOWMEM_WORKER) -> str:
    """The worker's text with the helpers in front of ``main()`` and the ITEM line (ITEM_LINE_FMT) printed once per backbone as each batch of ``main()``'s loop
    completes (ITEMTIME_WORKER_DONE, ``fin`` = the worker's own record of the batch: names, n, t_start, t_end, forward_s, call_s): the batch's forward span is
    the worker's own sampling + re-scoring device spans plus its write (``fin["forward_s"]``), shared equally over the batch's K backbones; ``batch_call_s`` is
    the batch's wall from its featurisation to its last file (``fin["call_s"]`` — batches overlap: the worker pipelines them), ``t_start`` / ``t_end`` its epoch
    stamps, the peaks read when it completed; the STACK line is printed before the loop and the OUTPUTS_WRITTEN line right after it (ITEMTIME_WORKER_LOOP:
    n_items = the worker's own ``done``, n_designs = done x its ``seqs_per_item``), the PREPASS line when the stream-offset pre-pass ends (ITEMTIME_WORKER_PREPASS: its own
    ``t0`` and ``len(proteins)``). Nothing is synchronised or timed here: the worker's record is read. ValueError when the statement, the loop header, the
    ``main()`` definition or a phase timer is not in the text (nothing is timed by a guess). Every line carries the process's pid."""
    for phase in ITEMTIME_WORKER_PHASES:
        if f'self.timers["{phase}"]' not in worker_src:
            raise ValueError(f'worker phase timer self.timers["{phase}"] not found (the record of a batch accumulates it)')
    src = insert_helpers(worker_src, "", "worker")
    src = _wrap_statement(src, ITEMTIME_WORKER_LOOP, before=["_it_stack()   # proteinmpnn_opt stage: the STACK line, once, before the first item"], after=[], what="worker batch loop")
    src = _wrap_statement(src, ITEMTIME_WORKER_DONE, before=[],
                          after=["_it_pa, _it_pr = _it_peak()   # proteinmpnn_opt stage: the ITEM line per backbone of the completed batch (the worker's own record; reads only)",
                                 f"for _it_n in fin['names']: print({ITEM_LINE_FMT!r} % (_it_n, fin['forward_s'] / fin['n'], fin['n'], fin['forward_s'], fin['call_s'], _it_os.getpid(), fin['t_start'], fin['t_end']), flush=True); "
                                 f"print({PEAK_LINE_FMT!r} % (_it_n, _it_os.getpid(), _it_pa, _it_pr), flush=True)"],
                          what="worker batch record statement")
    src = _wrap_statement(src, ITEMTIME_WORKER_PREPASS, before=[],
                          after=[f"print({PREPASS_FMT!r} % (_it_os.getpid(), t0, _it_time.time(), len(proteins)), flush=True)   # proteinmpnn_opt stage: the pre-pass's span (timing only)"],
                          what="worker stream-offset pre-pass")
    src = _after_block(src, ITEMTIME_WORKER_LOOP, ["_it_outputs_written(done, done * seqs_per_item)   # proteinmpnn_opt stage: every backbone's files are written and closed inside finalize"],
                       what="worker batch loop")
    compile(src, filename, "exec")
    return src


def stage_lowmem(paths: Dict[str, str]) -> Dict[str, str]:
    """Write kit/mpnn_worker2_lowmem.py (the lowmem transform, then the ITEM line's clocks) beside the staged worker ``paths["worker"]``; returns ``worker_lowmem`` (its path)."""
    with open(paths["worker"], encoding="utf-8") as fh:
        base = fh.read()
    try:
        src = lowmem_worker_source(base)
    except ValueError as e:
        raise ActivationError(f"lowmem transform: {e}") from None
    try:
        src = itemtime_worker_source(src)                                 # the ITEM line per backbone (timing only)
    except ValueError as e:
        raise ActivationError(f"itemtime transform: {e}") from None
    dst = os.path.join(os.path.dirname(paths["worker"]), LOWMEM_WORKER)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(src)
    return {"worker_lowmem": dst}


def remove(stage_dir: str) -> None:
    shutil.rmtree(stage_dir, ignore_errors=True)
