"""Upstream's affinity leg for ONE processed input, verbatim, in a clean stock interpreter — ``python -s -m boltz2_opt.affinity_leg <request.json>``.

Why. An input declaring ``properties: - affinity: {binder: <chain>}`` makes ``boltz predict`` run a SECOND model pass right after the structure
pass, in the same process (boltz/main.py ``predict``, the block after "Compute structure predictions": ``filter_inputs_affinity``,
``BoltzAffinityWriter``, ``Boltz2InferenceDataModule(affinity=True)``, ``Boltz2.load_from_checkpoint(<cache>/boltz2_aff.ckpt, …)``,
``trainer.predict``) and write ``predictions/<id>/affinity_<id>.json``. The kit's persistent worker predicts structures with its levers; the
affinity pass is stock's own second model on stock's own code path — no lever exists for it and none may touch it. So the worker runs it HERE:
a fresh interpreter without the worker's hooks, routes or lever patches (the worker's environment carries no BOLTZ2_OPT, so the kit's autoload
is inert), started right after the input's structure pass and handed the four RNG streams (torch CPU, numpy, python ``random``, CUDA) exactly as
they stand at that point — which is where stock's one process stands when it reaches this block: the affinity model's constructor, its
DataLoader's base seed and its sampling continue those streams as they do in stock. The request names the input's ``processed/`` paths, the cache,
and the ``diffusion_process_args`` / ``pairformer_args`` / ``msa_args`` the worker built for the structure model (main.py passes the same objects
to both models); the affinity options are the stock CLI's defaults (``DEFAULTS``, = main.py's ``--sampling_steps_affinity 200
--diffusion_samples_affinity 5``, ``--affinity_mw_correction`` off, ``--affinity_checkpoint`` unset → ``<cache>/boltz2_aff.ckpt``).

Exit 0 = the leg ran (or upstream found the affinity prediction already present); any failure raises (traceback on stderr, exit 1) and the
worker names it. The worker runs every leg of its pass in ONE such interpreter (``--serve``, form `process`, ``BOLTZ_AFFINITY_LEG``;
`item` = one interpreter per leg): request paths on its stdin, per request the same header, the streams SET from the request and ``run`` —
upstream's imports, the CUDA context and the affinity model held across requests as stock's one process holds them across its inputs
(``Boltz2.load_from_checkpoint`` memoised on (checkpoint, arguments, pre-construction stream states) with the constructor's draws replayed on a
hit: ``memoize_constructor``), one ``RESULT rc=<n> request=<path>`` line back per request; between requests the model returns to the host and
the allocator's cache is released (``_after_request``). Standard library only at import: the worker imports ``request`` / ``write_request`` / ``command`` / ``execute`` from here.

Under ``--n_gpu P > 1`` (P rank processes of the worker on one node; rank 0's outputs are the run's) the leg runs on rank 0 ONLY
(``affinity_leg=rank0_only``, :func:`execute`): rank 0 starts the clean interpreter as above and, when it has exited, sends its exit code and the
``affinity_*`` files it wrote to every other rank (:func:`opt_core.mem.rowpair.rankdata.broadcast_features` — the other ranks wait at the store
meanwhile, no leg process, no second model, no featurization of their own); a rank > 0 copies those files into its own ``predictions/``
directory and returns rank 0's exit code, so the worker's accounting after the leg reads the same on every rank and a failed leg is failed
on every rank.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from typing import List, Optional

DEFAULTS = {"sampling_steps_affinity": 200, "diffusion_samples_affinity": 5, "affinity_mw_correction": False, "affinity_checkpoint": None}   # boltz main.py predict's click defaults
AFFINITY_CKPT = "boltz2_aff.ckpt"                  # main.py: `affinity_checkpoint = cache / "boltz2_aff.ckpt"` when --affinity_checkpoint is unset
REQUEST, RNG = "request.json", "rng.pkl"
TP_FORM = "rank0_only"                             # at n_gpu > 1 the leg runs on rank 0 only; the word `affinity_leg=rank0_only` (tp_word) names it on the ranks' lines and the LEVER line
TP_WHAT = "affinity"                               # the `what` / store-key stem of the leg's status broadcast (rankdata.broadcast_features: key affinity/<n>, one per leg, the same n on every rank)
NOT_TAKEN = "affinity_result_not_taken"            # refused by this name on EVERY rank when a rank could not copy rank 0's result files
_CALLS = {TP_WHAT: 0}                              # legs run or received by this process: numbers the store key of each
FORM_ENV, FORMS = "BOLTZ_AFFINITY_LEG", ("process", "item")   # `process` (the default): ONE clean interpreter per worker pass serves every (input, seed) leg of the pass
                                                   # (`--serve`: the same block per request, upstream's imports / CUDA context / affinity model held across requests —
                                                   # as stock's one process holds them across its inputs); `item`: one interpreter per leg, by name
SERVE_FLAG = "--serve"
RESULT = "[boltz2-opt affinity] RESULT"           # the served interpreter's one line per request on its stdout: `… RESULT rc=<n> request=<path>` (execute reads up to it)
_SERVER = {"proc": None, "python": None, "starts": 0, "requests": 0}   # the worker side's handle on its served interpreter
_MEMO = {"installed": False, "entries": {}, "builds": 0, "hits": 0, "disabled": None}   # the served side's model memo (load_from_checkpoint by (checkpoint, arguments, pre-constructor RNG state))


def request(*, out_dir, targets_dir, msa_dir, constraints_dir, template_dir, extra_mols_dir, cache, mol_dir, num_workers: int,
            diffusion_process_args: dict, pairformer_args: dict, msa_args: dict, **overrides) -> dict:
    """The leg's inputs, JSON-able: the input's ``boltz_results_<stem>`` directory and its processed/ paths (as the worker's BoltzProcessedInput
    holds them; None stays None), the cache, the DataLoader's ``num_workers`` (the worker's own), the three model-argument dicts the worker built
    for the structure model, and the stock CLI defaults for the affinity options (``overrides`` for a caller that states them otherwise)."""
    s = lambda p: None if p is None else os.fspath(p)   # noqa: E731
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown affinity option(s) {sorted(unknown)}; the stock CLI's are {sorted(DEFAULTS)}")
    return {"out_dir": s(out_dir), "targets_dir": s(targets_dir), "msa_dir": s(msa_dir), "constraints_dir": s(constraints_dir), "template_dir": s(template_dir),
            "extra_mols_dir": s(extra_mols_dir), "cache": s(cache), "mol_dir": s(mol_dir), "num_workers": int(num_workers),
            "diffusion_process_args": dict(diffusion_process_args), "pairformer_args": dict(pairformer_args), "msa_args": dict(msa_args), **DEFAULTS, **overrides}


def write_request(leg_dir, req: dict, rng: dict) -> str:
    """Write ``<leg_dir>/request.json`` (+ ``rng.pkl``: {"torch", "np", "py", "cuda"} as the worker's ``rng_state()`` returns them) and return
    the request's path — the one argument of ``command``."""
    os.makedirs(leg_dir, exist_ok=True)
    rng_path = os.path.join(os.fspath(leg_dir), RNG)
    with open(rng_path, "wb") as f:
        pickle.dump(rng, f, protocol=pickle.HIGHEST_PROTOCOL)
    path = os.path.join(os.fspath(leg_dir), REQUEST)
    with open(path, "w") as f:
        json.dump(dict(req, rng=rng_path), f, indent=1)
    return path


def command(request_path: str, python: Optional[str] = None) -> List[str]:
    """``[python, "-s", "-m", "boltz2_opt.affinity_leg", request_path]`` — the interpreter flags of the kit's stock route (cli.cmd_pred_off)."""
    return [python or sys.executable, "-s", "-m", "boltz2_opt.affinity_leg", os.fspath(request_path)]


def leg_form(environ=None) -> str:
    """`process` | `item` (FORM_ENV; anything else reads as the default `process`)."""
    env = os.environ if environ is None else environ
    w = (env.get(FORM_ENV) or "").strip().lower()
    return w if w in FORMS else "process"


def server_command(python: Optional[str] = None) -> List[str]:
    """``[python, "-s", "-m", "boltz2_opt.affinity_leg", "--serve"]`` — the clean interpreter that serves every leg of the pass (same flags as ``command``)."""
    return [python or sys.executable, "-s", "-m", "boltz2_opt.affinity_leg", SERVE_FLAG]


def _server(python: Optional[str] = None):
    """The pass's served interpreter, started on first use (request paths on its stdin, its transcript + one RESULT line per request on its
    stdout, its stderr this process's); a server that has exited is started again for the next request."""
    import atexit
    import subprocess
    p = _SERVER["proc"]
    if p is not None and p.poll() is None:
        return p
    p = subprocess.Popen(server_command(python or _SERVER["python"]), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    if _SERVER["starts"] == 0:
        atexit.register(close_server)
    _SERVER.update(proc=p, python=python or _SERVER["python"], starts=_SERVER["starts"] + 1)
    sys.stderr.write(f"[boltz2-opt affinity] served interpreter started (pid {p.pid}, start #{_SERVER['starts']}): every affinity leg of this pass runs in it\n"); sys.stderr.flush()
    return p


def warm(python: Optional[str] = None) -> bool:
    """Start the pass's served interpreter now (form `process`) so upstream's imports and its CUDA context are up before the first request —
    the worker calls this when it parses an input declaring the affinity property, ahead of that input's structure pass. Best effort:
    False (nothing started) in form `item` or when the start failed (the request itself then names the failure)."""
    if leg_form() != "process":
        return False
    try:
        _server(python); return True
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[boltz2-opt affinity] served interpreter not started ahead of time ({type(e).__name__}: {e}); the first leg starts it\n")
        return False


def close_server(timeout: float = 120.0) -> Optional[int]:
    """End the served interpreter (its stdin closed = no more requests; it exits after the request in hand). Returns its exit code, None if none ran."""
    p = _SERVER.get("proc")
    if p is None:
        return None
    _SERVER["proc"] = None
    try:
        if p.stdin and not p.stdin.closed:
            p.stdin.close()
        try:
            return p.wait(timeout=timeout)
        except Exception:  # noqa: BLE001 — a served interpreter that will not finish: terminated, named
            p.terminate(); sys.stderr.write(f"[boltz2-opt affinity] served interpreter pid {p.pid} terminated at exit (did not finish within {timeout:.0f} s)\n")
            return p.wait(timeout=30)
    except Exception:  # noqa: BLE001
        return None


def serve_request(request_path: str, python: Optional[str] = None) -> int:
    """Hand ``request_path`` to the pass's served interpreter and return that leg's exit code: its transcript lines are echoed to this stdout as
    they come (the leg's words stay in this transcript, as with the one-shot interpreter), up to its ``RESULT rc=<n> request=<path>`` line. A
    served interpreter that dies mid-request: its exit code (or 1) is this leg's, named on stderr, and the next request starts a fresh one; one
    that cannot be started or written to at all: this leg runs in a one-shot interpreter instead (``command``), named."""
    import subprocess
    path = os.fspath(request_path)
    for attempt in (1, 2):
        try:
            p = _server(python)
            p.stdin.write(path + "\n"); p.stdin.flush()
            break
        except Exception as e:  # noqa: BLE001 — the pipe of a server that died between requests: one restart, then the one-shot form by name
            _SERVER["proc"] = None
            if attempt == 2:
                sys.stderr.write(f"[boltz2-opt affinity] served interpreter unavailable ({type(e).__name__}: {e}): this leg runs in a one-shot interpreter (form item)\n"); sys.stderr.flush()
                return int(subprocess.run(command(path, python)).returncode)
    _SERVER["requests"] += 1
    for line in iter(p.stdout.readline, ""):
        if line.startswith(RESULT):
            toks = dict(t.split("=", 1) for t in line[len(RESULT):].split() if "=" in t)
            if toks.get("request") == path:
                return int(toks.get("rc", 1))
        sys.stdout.write(line); sys.stdout.flush()
    rc = p.wait(); _SERVER["proc"] = None                                       # stdout closed without this request's RESULT line: the served interpreter died with the leg in hand
    sys.stderr.write(f"[boltz2-opt affinity] served interpreter pid {p.pid} exited {rc} during the leg of {path}: that leg is failed by it; the next leg starts a fresh interpreter\n"); sys.stderr.flush()
    return int(rc) if rc else 1


def run_leg(request_path: str, python: Optional[str] = None) -> int:
    """This process's own run of one leg: the pass's served interpreter (form `process`) or a one-shot interpreter (form `item`)."""
    import subprocess
    if leg_form() == "item":
        return int(subprocess.run(command(request_path, python)).returncode)
    return serve_request(request_path, python)


def tp_word() -> str:
    """``affinity_leg=rank0_only``: how the ranks of an ``n_gpu > 1`` run come by the leg's result."""
    return f"affinity_leg={TP_FORM}"


def _prefix() -> str:
    from . import report                                                       # the kit's line tag lives in report (which imports this module at load)
    return report.PREFIX


def _leg_key() -> str:
    _CALLS[TP_WHAT] += 1
    return f"{TP_WHAT}/{_CALLS[TP_WHAT]}"


def _leg_files(out_dir: str) -> List[str]:
    """The leg's result files under ``out_dir``, relative: ``predictions/<id>/affinity_<id>*`` (main.py's BoltzAffinityWriter)."""
    import glob
    root = os.fspath(out_dir)
    return sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, "predictions", "*", "affinity_*")))


def _take_files(own_out_dir: str, rank0_out_dir: str, files: List[str]) -> List[str]:
    """A copy of each of rank 0's result files at ``<own_out_dir>/<rel>`` (bytes as rank 0 wrote them; an entry already there is replaced);
    nothing when the two directories are one. Returns the copies made."""
    import shutil
    if os.path.realpath(os.fspath(own_out_dir)) == os.path.realpath(os.fspath(rank0_out_dir)):
        return []
    made = []
    for rel in files:
        src, dst = os.path.join(os.fspath(rank0_out_dir), rel), os.path.join(os.fspath(own_out_dir), rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.unlink(dst)
        shutil.copyfile(src, dst)
        made.append(dst)
    return made


def execute(request_path: str, python: Optional[str] = None) -> int:
    """Run the leg of ``request_path`` and return its exit code. At n_gpu = 1: ``run_leg`` — the pass's served clean interpreter (form
    `process`: ``serve_request``) or a one-shot one (form `item`: ``subprocess.run(command(request_path)).returncode``), this
    transcript, this environment, this working directory. At n_gpu > 1 (``affinity_leg=rank0_only``): on rank 0 the same,
    then its exit code and result files go to every other rank (``rankdata.broadcast_features``, key ``affinity/<n>``) and rank 0 returns once
    every rank holds them (one all-gather: the worker moves rank 0's files right after this call); on a rank > 0 no process is started — the
    rank waits at the store for rank 0's leg however long it runs (no collective pending), copies rank 0's ``affinity_*`` files into its own
    ``predictions/`` directory and returns rank 0's exit code. One line per leg on every rank: ``[boltz2-opt] affinity_leg=rank0_only rank=<r>
    ranks=<P> rc=<rc> files=<n> [wait_s=<s> copied=<n> from=<rank 0's out_dir>]``. Refused by name on EVERY rank: a leg rank 0 could not start
    (``refused: affinity_rank0_failed: <Type>: <msg>`` — the status travels in place of the result, no rank waits on), a rank that could not
    copy the files (``refused: affinity_result_not_taken: rank(s) [...]`` — said in the acknowledgement), a rank of a ×P launch without its
    rank group."""
    import subprocess
    from opt_core.mem.rowpair import launch
    if int(launch.world_size()) <= 1:
        return run_leg(request_path, python)
    from opt_core.mem.rowpair import RowpairRefused, dist as D, rankdata as RD
    if not D.is_dist():
        raise RowpairRefused(f"refused: rank {launch.rank()} of {launch.world_size()} has no rank group for the affinity leg ({tp_word()}: rank 0 runs it, the other ranks receive its result)", lever="n_gpu")
    cm = D.comm()
    req = json.load(open(request_path))
    key = _leg_key()                                                         # numbered first, on every rank, whatever follows
    if int(cm.rank) == 0:
        try:
            rc = int(run_leg(request_path, python))
            files = _leg_files(req["out_dir"]) if rc == 0 else []
        except BaseException as e:                                           # the leg could not be started / its files not listed: every rank learns it at the store and raises refused: affinity_rank0_failed, this one too (`e` chained under it)
            RD.broadcast_features(None, src=0, key=key, status=RD.status_word(e), what=TP_WHAT, carry_rng=False)
            raise                                                            # not reached (the line above raises)
        got = {"rc": rc, "out_dir": os.fspath(req["out_dir"]), "files": files}
        RD.broadcast_features(got, src=0, key=key, what=TP_WHAT, carry_rng=False)
        word, tail = "ok", ""
    else:
        fb = RD.broadcast_features(None, src=0, key=key, what=TP_WHAT, carry_rng=False)   # waits at the store while rank 0's leg runs
        got = fb.feats
        try:
            copied = _take_files(req["out_dir"], got["out_dir"], list(got["files"])) if int(got["rc"]) == 0 else []
            word = "ok"
        except BaseException as e:                                           # this rank could not take the files: said in the acknowledgement, refused by name on every rank below
            copied, word = [], RD.status_word(e)
        tail = f" wait_s={float(fb.wait_s):.1f} copied={len(copied)} from={got['out_dir']}"
    acks = cm.allgather_obj((int(cm.rank), word))                            # every rank's word; rank 0's files may move once this returns (the worker moves them next)
    print(f"{_prefix()} {tp_word()} rank={cm.rank} ranks={cm.world} rc={int(got['rc'])} files={len(got['files'])}{tail}", flush=True)
    bad = [(r, w) for r, w in acks if w != "ok"]
    if bad:
        raise RowpairRefused(f"refused: {NOT_TAKEN}: rank(s) {[r for r, _ in bad]} of {cm.world} could not copy rank 0's affinity result "
                             f"({'; '.join(f'rank {r}: {w}' for r, w in bad)})", lever="n_gpu")
    return int(got["rc"])


def reset_for_tests() -> None:
    _CALLS[TP_WHAT] = 0
    close_server(timeout=10)
    _SERVER.update(proc=None, python=None, starts=0, requests=0)
    _MEMO.update(installed=False, entries={}, builds=0, hits=0, disabled=None)


# ---------------------------------------------------------------- the leg (imports boltz; runs in the clean interpreter only) ----------------------
def preamble() -> None:
    """main.py ``predict``'s process preamble, minus ``seed_everything`` (the streams are SET from the request instead): warnings filter,
    no grad, highest float32 matmul precision, the RDKit pickle policy, the cuEquivariance environment defaults."""
    import warnings
    import torch
    from rdkit import Chem
    warnings.filterwarnings("ignore", ".*that has Tensor Cores. To properly utilize them.*")
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
        os.environ[key] = os.environ.get(key, "1")


def set_rng(rng: dict) -> None:
    """The four streams as the worker captured them right after the structure pass (its ``rng_state()``)."""
    import random
    import numpy as np
    import torch
    torch.set_rng_state(rng["torch"]); np.random.set_state(rng["np"]); random.setstate(rng["py"]); torch.cuda.set_rng_state(rng["cuda"])


def _rng_snapshot():
    """(torch CPU, numpy global, python `random`) generator states of this process — the three streams a model constructor draws on."""
    import random
    import numpy as np
    import torch
    return torch.get_rng_state().clone(), np.random.get_state(), random.getstate()


def _rng_restore(snap) -> None:
    import random
    import numpy as np
    import torch
    torch.set_rng_state(snap[0]); np.random.set_state(snap[1]); random.setstate(snap[2])


def _rng_digest(snap) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(snap[0].numpy().tobytes())                                   # torch CPU generator state (a uint8 tensor)
    h.update(repr((snap[1][0], snap[1][2:])).encode()); h.update(snap[1][1].tobytes())   # numpy: (name, pos, has_gauss, cached_gaussian) + the key array
    h.update(pickle.dumps(snap[2], protocol=4))                           # python `random`: (version, state tuple, gauss_next)
    return h.hexdigest()


def memoize_constructor(cls, name: str = "load_from_checkpoint"):
    """Memoise ``cls.<name>`` (a classmethod that CONSTRUCTS a module: ``Boltz2.load_from_checkpoint``) for the served interpreter: the key is
    (checkpoint path + its size and mtime, the call's arguments, the digest of the three CPU-side generator states at the call); a MISS runs
    the classmethod as is and records the generator states it leaves; a HIT returns the module built then and SETS the three states to the
    recorded ones — the constructor's draws replayed exactly, so everything after the call (the DataLoader's base seed, the sampling) reads
    the streams where the one-shot interpreter's fresh construction leaves them. The CUDA generator is checked untouched by the construction
    (it is: ``map_location="cpu"``); were it not, the memo disables itself by name and every call constructs. Returns the wrapper (idempotent)."""
    import functools
    cur = cls.__dict__.get(name)
    if getattr(cur, "_boltz2_opt_memo", False):
        return cur
    orig = getattr(cls, name)                                             # the constructor as upstream resolves it on the class: a bound classmethod, or the plain function
                                                                          # pytorch_lightning's restricted-classmethod descriptor hands back (the class bound inside it)

    def _key(checkpoint_path, args, kwargs, digest):
        p = os.fspath(checkpoint_path) if isinstance(checkpoint_path, (str, os.PathLike)) else repr(checkpoint_path)
        try:
            st = os.stat(p); stamp = f"{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            stamp = "-"
        return json.dumps([p, stamp, repr(args), json.dumps(kwargs, sort_keys=True, default=repr), digest])

    @functools.wraps(getattr(orig, "__func__", orig))
    def memo(klass, checkpoint_path, *args, **kwargs):
        pre = _rng_snapshot()
        key = _key(checkpoint_path, args, kwargs, _rng_digest(pre))
        ent = None if _MEMO["disabled"] else _MEMO["entries"].get(key)
        if ent is not None:
            _MEMO["hits"] += 1
            _rng_restore(ent["post"])
            print(f"[boltz2-opt affinity] affinity model held from request #{ent['built_at']} of this interpreter (same checkpoint, arguments and pre-construction "
                  f"stream states; its constructor's draws replayed); memo hits={_MEMO['hits']} builds={_MEMO['builds']}", flush=True)
            return ent["module"]
        cuda_before = _cuda_rng_state()
        raw = getattr(orig, "__func__", None) or getattr(orig, "__wrapped__", None)   # the undecorated `f(cls, checkpoint_path, …)` behind either form
        if klass is not cls and raw is not None:                          # called on a subclass: construct THAT class, as the classmethod would
            module = raw(klass, checkpoint_path, *args, **kwargs)
        else:                                                             # the class it was fetched from: call it exactly as upstream's block does
            module = orig(checkpoint_path, *args, **kwargs)
        _MEMO["builds"] += 1
        cuda_after = _cuda_rng_state()
        if cuda_before is not None and cuda_after is not None and not bool((cuda_before == cuda_after).all()):
            _MEMO["disabled"] = "constructor_draws_on_the_cuda_generator"
            print("[boltz2-opt affinity] the affinity model's construction drew on the CUDA generator: the model memo is OFF by that name (every leg constructs)", flush=True)
        elif not _MEMO["disabled"]:
            _MEMO["entries"][key] = {"module": module, "post": _rng_snapshot(), "built_at": _SERVER.get("requests") or _MEMO["builds"]}
        return module

    memo._boltz2_opt_memo = True
    wrapper = classmethod(memo)
    wrapper._boltz2_opt_memo = True
    setattr(cls, name, wrapper)
    _MEMO["installed"] = True
    return wrapper


def _cuda_rng_state():
    try:
        import torch
        return torch.cuda.get_rng_state().clone() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    except Exception:  # noqa: BLE001
        return None


def _after_request() -> None:
    """Between requests the served interpreter holds what stock's process holds between inputs — the imports, the CUDA context, the model on
    the host: every memoised module goes back to the CPU (a fresh ``load_from_checkpoint(map_location="cpu")`` hands it over there) and the
    caching allocator's blocks are released, so the worker's next structure pass meets only this interpreter's CUDA context on the card."""
    try:
        import gc
        import torch
        for ent in _MEMO["entries"].values():
            try:
                ent["module"].cpu()
            except Exception:  # noqa: BLE001
                pass
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def handle_request(request_path: str) -> int:
    """One leg in the served interpreter: exactly ``main`` for that request — the header line, the streams SET from the request, ``run`` — with
    ``Boltz2.load_from_checkpoint`` memoised (``memoize_constructor``)."""
    req = json.load(open(request_path))
    with open(req["rng"], "rb") as f:
        rng = pickle.load(f)
    _SERVER["requests"] = int(_SERVER.get("requests") or 0) + 1
    print(f"[boltz2-opt affinity] upstream's affinity leg (boltz main.py predict, after the structure pass) for {req['out_dir']} in the pass's clean served interpreter "
          f"(pid {os.getpid()}, request #{_SERVER['requests']}); RNG streams handed over from the structure pass; options {json.dumps({k: req[k] for k in DEFAULTS})}", flush=True)
    from boltz.model.models.boltz2 import Boltz2
    memoize_constructor(Boltz2)
    set_rng(rng)
    return run(req)


def serve(stdin=None, handle=None) -> int:
    """``--serve``: read request paths, one per line, from stdin until it closes; per request: ``handle`` (``handle_request``), any failure
    printed (traceback) as rc 1, then ONE line ``RESULT rc=<n> request=<path>`` on stdout and the between-requests release (``_after_request``).
    Upstream's imports and the process preamble run once, up front (as stock's process has them before its affinity block)."""
    import traceback
    stdin = sys.stdin if stdin is None else stdin
    handle = handle_request if handle is None else handle
    if handle is handle_request:
        try:
            preamble()                                                    # torch / rdkit + main.py's process settings, once
            import pytorch_lightning  # noqa: F401
            import boltz.main  # noqa: F401
            import boltz.model.models.boltz2  # noqa: F401
            import torch
            if torch.cuda.is_available():
                torch.cuda.init(); torch.empty(1, device="cuda")          # the CUDA context up before the first request (no generator draw)
        except Exception:  # noqa: BLE001 — named now; the request that needs it fails by the same error, accounted there
            traceback.print_exc()
        print(f"[boltz2-opt affinity] served interpreter ready (pid {os.getpid()}): upstream imported, CUDA context up; waiting for the pass's affinity legs", flush=True)
    for raw in stdin:
        path = raw.strip()
        if not path:
            continue
        try:
            rc = int(handle(path) or 0)
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
        except BaseException:  # noqa: BLE001 — the leg failed: its traceback is the transcript, rc 1, the interpreter serves on
            traceback.print_exc(); rc = 1
        sys.stderr.flush()
        print(f"{RESULT} rc={rc} request={path}", flush=True)
        if handle is handle_request:
            _after_request()
    return 0


def run(req: dict) -> int:
    """boltz/main.py ``predict``, the affinity block after the structure pass, verbatim but for three substitutions of where a name comes from,
    none of which touches a number: (1) ``manifest`` / ``processed`` / ``out_dir`` / ``cache`` / ``mol_dir`` / ``num_workers`` / ``override`` and the
    affinity options are the request's (the worker's paths; the CLI defaults); (2) ``asdict(diffusion_params)`` / ``asdict(pairformer_args)`` /
    ``asdict(msa_args)`` arrive as those very dicts, built by the worker for the structure model as main.py builds them once for both;
    (3) main.py reuses the structure pass's Trainer with its callback swapped (``trainer.callbacks[0] = pred_writer``) — here that Trainer is
    constructed with the same arguments main.py gave it (strategy auto at one device, the gpu accelerator, bf16-mixed) around ``pred_writer``."""
    from dataclasses import asdict
    from pathlib import Path
    import click
    from pytorch_lightning import Trainer
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.data.write.writer import BoltzAffinityWriter
    from boltz.main import BoltzProcessedInput, BoltzSteeringParams, filter_inputs_affinity
    from boltz.model.models.boltz2 import Boltz2

    P = lambda k: None if req.get(k) is None else Path(req[k])   # noqa: E731
    out_dir, cache, mol_dir = P("out_dir"), P("cache"), P("mol_dir")
    processed = BoltzProcessedInput(manifest=Manifest.load(out_dir / "processed" / "manifest.json"), targets_dir=P("targets_dir"), msa_dir=P("msa_dir"),
                                    constraints_dir=P("constraints_dir"), template_dir=P("template_dir"), extra_mols_dir=P("extra_mols_dir"))
    manifest = processed.manifest
    sampling_steps_affinity, diffusion_samples_affinity = int(req["sampling_steps_affinity"]), int(req["diffusion_samples_affinity"])
    affinity_mw_correction = bool(req["affinity_mw_correction"])
    affinity_checkpoint = None if req.get("affinity_checkpoint") is None else Path(req["affinity_checkpoint"])
    num_workers, override = int(req["num_workers"]), False
    asdict_diffusion_params, asdict_pairformer_args, asdict_msa_args = req["diffusion_process_args"], req["pairformer_args"], req["msa_args"]   # (2)

    # ---- boltz/main.py predict(): "Check if affinity predictions are needed" .. the end of the affinity block (verbatim; comments upstream's) ----
    # Check if affinity predictions are needed
    if any(r.affinity for r in manifest.records):
        # Print header
        click.echo("\nPredicting property: affinity\n")

        # Validate inputs
        manifest_filtered = filter_inputs_affinity(
            manifest=manifest,
            outdir=out_dir,
            override=override,
        )
        if not manifest_filtered.records:
            click.echo("Found existing affinity predictions for all inputs, skipping.")
            return 0

        msg = f"Running affinity prediction for {len(manifest_filtered.records)} input"
        msg += "s." if len(manifest_filtered.records) > 1 else "."
        click.echo(msg)

        pred_writer = BoltzAffinityWriter(
            data_dir=processed.targets_dir,
            output_dir=out_dir / "predictions",
        )

        data_module = Boltz2InferenceDataModule(
            manifest=manifest_filtered,
            target_dir=out_dir / "predictions",
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers,
            constraints_dir=processed.constraints_dir,
            template_dir=processed.template_dir,
            extra_mols_dir=processed.extra_mols_dir,
            override_method="other",
            affinity=True,
        )

        predict_affinity_args = {
            "recycling_steps": 5,
            "sampling_steps": sampling_steps_affinity,
            "diffusion_samples": diffusion_samples_affinity,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        # Load affinity model
        if affinity_checkpoint is None:
            affinity_checkpoint = cache / "boltz2_aff.ckpt"

        steering_args = BoltzSteeringParams()
        steering_args.fk_steering = False
        steering_args.physical_guidance_update = False
        steering_args.contact_guidance_update = False

        model_module = Boltz2.load_from_checkpoint(
            affinity_checkpoint,
            strict=True,
            predict_args=predict_affinity_args,
            map_location="cpu",
            diffusion_process_args=asdict_diffusion_params,
            ema=False,
            pairformer_args=asdict_pairformer_args,
            msa_args=asdict_msa_args,
            steering_args=asdict(steering_args),
            affinity_mw_correction=affinity_mw_correction,
        )
        model_module.eval()

        trainer = Trainer(default_root_dir=out_dir, strategy="auto", callbacks=[pred_writer], accelerator="gpu", devices=1, precision="bf16-mixed")   # (3)
        trainer.predict(
            model_module,
            datamodule=data_module,
            return_predictions=False,
        )
    return 0


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == [SERVE_FLAG]:
        return serve()
    if len(argv) != 1:
        sys.stderr.write("usage: python -s -m boltz2_opt.affinity_leg <request.json> | --serve (request paths on stdin)\n"); return 2
    req = json.load(open(argv[0]))
    with open(req["rng"], "rb") as f:
        rng = pickle.load(f)
    print(f"[boltz2-opt affinity] upstream's affinity leg (boltz main.py predict, after the structure pass) for {req['out_dir']} in a clean interpreter "
          f"(pid {os.getpid()}); RNG streams handed over from the structure pass; options {json.dumps({k: req[k] for k in DEFAULTS})}", flush=True)
    preamble()
    set_rng(rng)
    return run(req)


if __name__ == "__main__":
    sys.exit(main())
