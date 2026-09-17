"""Input preprocessing in a fresh process per input — the ZYGOTE the persistent worker parses through.

Why. The stock CLI is one process per input: ``boltz predict a.yaml`` parses ``a.yaml`` (boltz/main.py ``process_inputs``) in a process that
has parsed nothing before it. A SMILES ligand is given its reference conformer there by RDKit's ETKDG (boltz/data/parse/schema.py
``compute_3d_conformer``: ``AllChem.EmbedMolecule(mol, ETKDGv3())`` with no ``randomSeed``), and RDKit's conformer RNG is PROCESS-GLOBAL: the
first embedding of a process is reproducible across processes, the second depends on the first. Under stock every input's ligands are the
first embeddings of a process; a persistent worker that parsed input after input in its own process would hand the 2nd, 3rd, … SMILES input of
its life a different conformer than stock does (different ``ref_pos`` features → a different structure for every SMILES input after the first).
Seeding is not an option: the outputs must equal stock's AS SHIPPED.

What. :class:`Zygote` is a small fork server. ``start()`` forks it at worker launch — before CUDA is initialised, before any lever is attached,
before anything is parsed — and it imports ``boltz.main`` once (in the background, while the worker builds the model). ``run(target, **kwargs)``
asks it to fork a FRESH child of itself; the child imports/looks up ``target`` (``"module:function"``), calls it with ``kwargs``, and returns
``(rc, stdout, stderr)``: its Python-level stdout / stderr are captured (the caller prints them where it wants them; boltz's own words —
``Failed to process … Skipping`` — are among them). Every input is thus parsed by a process in the state stock's process is in when it parses:
nothing embedded before it, no kit hook installed, ``Chem.SetDefaultPickleProperties`` set as main.py sets it (:func:`parse_one`). The fork
costs milliseconds; the worker's speed is unchanged.

The worker's one call: ``prep.zygote().parse(yaml, out_dir, ccd_path, mol_dir, **kw)`` (= ``run(PARSE_ONE, …)``) where ``kw`` are the
exact ``process_inputs`` keyword arguments the worker states once (bz_worker_lev*.py ``PROCESS_INPUTS_KW``). Standard library only at import.

Under ``--n_gpu P > 1`` (the tensor-parallel line: P rank processes of the same worker on one node) the input is parsed ONCE, on rank 0: rank 0's
zygote parses as above and rank 0 sends the parse's :class:`Result` and the path of its ``processed`` directory to every other rank
(:func:`opt_core.mem.rowpair.rankdata.broadcast_features`); a rank > 0 forks no zygote and parses nothing — its :class:`Receiver` takes rank 0's
result, checks that rank 0's ``processed`` directory is readable from this process (the ranks share the node's filesystem; refused BY NAME on
every rank when one rank cannot read it: ``rank0_processed_dir_not_visible``) and links its own ``<out_dir>/processed`` to it, so the worker's
manifest / structure reads resolve to rank 0's files. The model-input features follow the same route (``rowpair_msa``: rank 0 featurizes,
ranks > 0 receive; ``data_form=rank0_bcast``).
"""
from __future__ import annotations

import atexit
import contextlib
import io
import os
import sys
import threading
import traceback
from typing import NamedTuple, Optional, Sequence, Tuple

PREFIX = "[boltz2-opt prep]"
PARSE_ONE = "boltz2_opt.prep:parse_one"          # the target the worker runs per input (below)
PRELOAD = ("boltz.main",)                        # imported once in the zygote so a per-input child forks with boltz already loaded
PROCESSED = "processed"                          # boltz's processed-input directory under an input's out_dir (main.py process_inputs; the worker's Manifest.load reads it)
NOT_VISIBLE = "rank0_processed_dir_not_visible"  # the refusal word when a rank cannot read rank 0's processed directory (n_gpu > 1: one node, one filesystem is the line's premise)
PARSE_WHAT = "parse"                             # the `what` / store-key stem of the parse result's broadcast (rankdata.broadcast_features: key parse/<n>, one per input, the same n on every rank)


def parse_one(yaml: str, out_dir: str, ccd_path: str, mol_dir: str, **process_inputs_kw) -> None:
    """One input parsed exactly as ``boltz predict <yaml>`` parses it: main.py ``predict``'s parsing preamble (the RDKit pickle policy the
    processed mols are written under) then ``check_inputs`` and ``process_inputs`` with the caller's keyword arguments. Runs in a fresh child
    of the zygote (nothing embedded before it in that process)."""
    from pathlib import Path
    from rdkit import Chem
    from boltz.main import check_inputs, process_inputs
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)        # main.py predict(), before process_inputs
    data = check_inputs(Path(yaml))
    process_inputs(data=data, out_dir=Path(out_dir), ccd_path=Path(ccd_path), mol_dir=Path(mol_dir), **process_inputs_kw)


class Result(NamedTuple):
    """rc 0 = the target returned; rc 1 = it raised (``err`` ends with its traceback); rc 2 = the child died without reporting (``err`` names
    the wait status). ``out`` / ``err``: the child's Python-level stdout / stderr, verbatim."""
    rc: int
    out: str
    err: str


class Zygote:
    """A fork server: forked once by :meth:`start` from a process that has parsed nothing; :meth:`run` forks a fresh child of the zygote per
    call. Not thread-safe (one caller: the worker's main thread)."""

    def __init__(self, preload: Sequence[str] = PRELOAD):
        self.preload = tuple(preload)
        self.pid: Optional[int] = None
        self.forked_at: Optional[str] = None          # "launch" (worker_launch, before CUDA / hooks) | "first_use"
        self.cuda_initialized_at_fork: Optional[bool] = None
        self.threads_at_fork: Optional[int] = None
        self._conn = None
        self._seq = 0

    # ------------------------------------------------------------------ parent side
    def start(self, forked_at: str = "first_use") -> "Zygote":
        if self.pid is not None:
            return self
        from multiprocessing import Pipe
        parent_end, child_end = Pipe(duplex=True)
        torch = sys.modules.get("torch")
        self.cuda_initialized_at_fork = bool(torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_initialized())
        self.threads_at_fork = threading.active_count()
        self.forked_at = forked_at
        sys.stdout.flush(); sys.stderr.flush()
        pid = os.fork()
        if pid == 0:                                   # the zygote
            parent_end.close()
            try:
                self._serve(child_end)
            finally:
                os._exit(0)
        child_end.close()
        self.pid, self._conn = pid, parent_end
        atexit.register(self.close)
        sys.stderr.write(f"{PREFIX} zygote pid={pid} forked_at={forked_at} cuda_initialized={self.cuda_initialized_at_fork} "
                         f"threads={self.threads_at_fork} preload={','.join(self.preload) or '-'} (every input is parsed in a fresh child of it)\n")
        sys.stderr.flush()
        return self

    def run(self, target: str, **kwargs) -> Result:
        """Fork a fresh child of the zygote, call ``target`` (``"module:function"``) with ``kwargs`` there, return :class:`Result`."""
        if self.pid is None:
            self.start()
        self._seq += 1
        self._conn.send(("run", self._seq, target, kwargs))
        while True:
            try:
                tag, seq, rc, out, err = self._conn.recv()
            except EOFError:
                raise RuntimeError(f"{PREFIX} the zygote (pid {self.pid}) is gone") from None
            if seq == self._seq:                       # a stale message of an earlier call (a child that reported and then died oddly) is skipped
                return Result(int(rc), out, err)

    def parse(self, yaml: str, out_dir: str, ccd_path: str, mol_dir: str, **process_inputs_kw) -> Result:
        """One input parsed by :func:`parse_one` in a fresh child of the zygote — the worker's per-input call. On rank 0 of an ``n_gpu > 1`` run
        the result and this input's processed directory then go to every other rank (:func:`share_parse`) BEFORE the caller acts on ``rc``:
        a failed parse fails every rank alike, no rank waits on a parse that will not come."""
        res = self.run(PARSE_ONE, yaml=yaml, out_dir=out_dir, ccd_path=ccd_path, mol_dir=mol_dir, **process_inputs_kw)
        return share_parse(res, out_dir)

    def describe(self) -> dict:
        return {"zygote_pid": self.pid, "forked_at": self.forked_at, "cuda_initialized_at_fork": self.cuda_initialized_at_fork,
                "threads_at_fork": self.threads_at_fork, "preload": list(self.preload), "calls": self._seq}

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(("stop", 0, None, None))
            except (BrokenPipeError, OSError):
                pass
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None
        if self.pid is not None:
            try:
                os.waitpid(self.pid, 0)
            except ChildProcessError:
                pass
            self.pid = None

    # ------------------------------------------------------------------ zygote side
    def _serve(self, conn) -> None:
        for m in self.preload:                          # once, here: the per-input children fork with it loaded
            try:
                __import__(m)
            except Exception as e:                      # named here; the child that needs the module reports the ImportError again, with its traceback
                sys.stderr.write(f"{PREFIX} zygote preload of {m} failed: {type(e).__name__}: {e}\n"); sys.stderr.flush()
        while True:
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                return                                  # the worker is gone
            if not isinstance(msg, tuple) or msg[0] != "run":
                return
            _, seq, target, kwargs = msg
            sys.stdout.flush(); sys.stderr.flush()
            pid = os.fork()
            if pid == 0:                                # the fresh child: one input's parse, then exit
                rc, out, err = _call_captured(target, kwargs)
                code = 0
                try:
                    conn.send(("done", seq, rc, out, err))
                except BaseException:                   # could not report: exit non-zero so the zygote reports for it
                    code = 3
                os._exit(code)
            _, status = os.waitpid(pid, 0)
            if status != 0:                             # the child died without reporting (a report is always followed by exit 0)
                how = f"signal {os.WTERMSIG(status)}" if os.WIFSIGNALED(status) else f"exit {os.WEXITSTATUS(status)}"
                conn.send(("done", seq, 2, "", f"{PREFIX} the child parsing this input died without reporting ({how})\n"))


def _call_captured(target: str, kwargs: dict) -> Tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    rc = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            mod_name, _, fn_name = target.partition(":")
            mod = __import__(mod_name, fromlist=["_"])
            getattr(mod, fn_name)(**kwargs)
        except BaseException:                            # SystemExit from a target included: reported, never swallowed silently
            rc = 1
            err.write(traceback.format_exc())
    return rc, out.getvalue(), err.getvalue()


# ------------------------------------------------------------------ n_gpu > 1: rank 0 parses, every other rank receives
def rank_env() -> Tuple[int, int]:
    """``(P, rank)`` of this process from the core launcher's rank environment (``ROWPAIR_WORLD`` / ``ROWPAIR_RANK``); ``(1, 0)`` outside a ×P launch."""
    from opt_core.mem.rowpair import launch
    return int(launch.world_size()), int(launch.rank())


def _processed_dir(out_dir: str) -> str:
    return os.path.join(os.path.abspath(out_dir), PROCESSED)


def _readable(processed_dir: str) -> bool:
    return os.path.isdir(processed_dir) and os.access(processed_dir, os.R_OK | os.X_OK) and os.path.isfile(os.path.join(processed_dir, "manifest.json"))


def _all_ranks_see(processed_dir: str, ok: bool) -> None:
    """Every rank states whether it can read rank 0's processed directory; when any rank cannot, EVERY rank refuses by name (one object
    all-gather: all ranks learn the same answer, none is left in a collective)."""
    from opt_core.mem.rowpair import RowpairRefused, dist as D
    seen = D.comm().allgather_obj((int(D.comm().rank), bool(ok)))
    blind = [r for r, good in seen if not good]
    if blind:
        raise RowpairRefused(f"refused: {NOT_VISIBLE}: {processed_dir} — rank(s) {blind} cannot read it "
                             f"(the n_gpu > 1 line runs its ranks on ONE node sharing one filesystem: every rank reads rank 0's processed directory)", lever="n_gpu")


_CALLS = {"parse": 0}                            # parse results sent or received by this process: numbers the store key of each (parse/<n>), identical on every rank


def _parse_key() -> str:
    _CALLS["parse"] += 1
    return f"{PARSE_WHAT}/{_CALLS['parse']}"


def share_parse(res: Result, out_dir: str) -> Result:
    """Rank 0 of an ``n_gpu > 1`` run, right after its parse: send ``(rc, out, err, processed dir)`` to every other rank
    (``rankdata.broadcast_features``: the ranks > 0 wait at the store for rank 0 however long the parse takes — no collective pending — then
    receive), then take part in the visibility census (:func:`_all_ranks_see`). A pass-through without a rank group (n_gpu = 1): nothing is sent."""
    from opt_core.mem.rowpair import RowpairRefused, dist as D
    if not D.is_dist():
        if rank_env()[0] > 1:                                                 # a rank of a ×P launch without its group: refused, never a per-rank parse in silence
            raise RowpairRefused(f"refused: {PREFIX} rank {rank_env()[1]} of {rank_env()[0]} has no rank group to send its parse result on (the ×P line attaches before the first parse)", lever="n_gpu")
        return res
    from opt_core.mem.rowpair import rankdata as RD
    pdir = _processed_dir(out_dir)
    RD.broadcast_features({"rc": int(res.rc), "out": res.out, "err": res.err, "processed_dir": pdir}, src=0, key=_parse_key(), what=PARSE_WHAT, carry_rng=False)
    _all_ranks_see(pdir, _readable(pdir) if int(res.rc) == 0 else True)       # a failed parse wrote no directory: the rc, not the directory, is the news
    return res


class Receiver:
    """The parse seam of a rank > 0 of an ``n_gpu > 1`` run: no zygote, no child, nothing parsed in this process. :meth:`parse` receives rank 0's
    :class:`Result` and processed directory (:func:`share_parse` is the sending side), refuses by name when this process cannot read that
    directory (every rank refuses then), links ``<out_dir>/processed`` to it and returns rank 0's result verbatim — so the worker's statements
    after the parse (manifest load, the SKIPPED census of an input stock's parser skipped, the writer's structure reads) run on every rank
    exactly as on rank 0. The interface is :class:`Zygote`'s."""

    def __init__(self):
        self.P, self.rank = rank_env()
        self.calls = 0
        self.last: Optional[dict] = None

    def start(self, forked_at: str = "first_use") -> "Receiver":
        return self

    def parse(self, yaml: str, out_dir: str, ccd_path: str, mol_dir: str, **process_inputs_kw) -> Result:
        from opt_core.mem.rowpair import RowpairRefused, dist as D, rankdata as RD
        if not D.is_dist():
            raise RowpairRefused(f"refused: {PREFIX} rank {self.rank} of {self.P} has no rank group to receive rank 0's parse from (the ×P line attaches before the first parse)", lever="n_gpu")
        self.calls += 1
        got = RD.broadcast_features(None, src=0, key=_parse_key(), what=PARSE_WHAT, carry_rng=False).feats   # waits at the store for rank 0's parse, however long it takes
        pdir = str(got["processed_dir"])
        ok = _readable(pdir) if int(got["rc"]) == 0 else True                      # a failed parse wrote no directory: the rc, not the directory, is the news
        _all_ranks_see(pdir, ok)
        linked, set_aside = _link_processed(out_dir, pdir) if int(got["rc"]) == 0 else (None, None)
        self.last = {"rank": self.rank, "rc": int(got["rc"]), "processed": pdir, "link": linked, "set_aside": set_aside}
        sys.stderr.write(f"{PREFIX} rank={self.rank} processed=rank0 {RD.data_form_word(data_form())} rc={int(got['rc'])} dir={pdir}"
                         + (f" link={linked}" if linked else "") + (f" set_aside={set_aside}" if set_aside else "") + "\n"); sys.stderr.flush()
        return Result(int(got["rc"]), str(got["out"]), str(got["err"]))

    def run(self, target: str, **kwargs) -> Result:
        from opt_core.mem.rowpair import RowpairRefused
        raise RowpairRefused(f"refused: {PREFIX} rank {self.rank} runs no zygote child ({target}): under n_gpu > 1 rank 0 parses and the other ranks receive", lever="n_gpu")

    def describe(self) -> dict:
        return {"zygote_pid": None, "forked_at": None, "rank": self.rank, "ranks": self.P, "processed_from": "rank0", "data_form": data_form(),
                "calls": self.calls, "last": self.last}

    def close(self) -> None:
        return None


def _link_processed(out_dir: str, rank0_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """``<out_dir>/processed -> rank0_dir`` (absolute symlink) → ``(link, set_aside)``. A symlink already there is re-pointed; a real directory
    there (this rank's own parse product of an earlier run into the same out_dir) is set aside as ``processed.rank_local.<n>`` — never read,
    never deleted, named on the rank's line (``set_aside=``); when ``<out_dir>/processed`` IS rank 0's directory (a rank launched on rank 0's
    out_dir) nothing is renamed or linked: ``(None, None)``."""
    mine = _processed_dir(out_dir)
    if not os.path.islink(mine) and os.path.realpath(mine) == os.path.realpath(rank0_dir):     # this rank's processed entry IS rank 0's directory
        return None, None
    os.makedirs(os.path.dirname(mine), exist_ok=True)
    set_aside = None
    if os.path.islink(mine):
        os.unlink(mine)
    elif os.path.exists(mine):
        n = 0
        while os.path.exists(f"{mine}.rank_local.{n}"):
            n += 1
        set_aside = f"{mine}.rank_local.{n}"
        os.rename(mine, set_aside)
    os.symlink(rank0_dir, mine)
    return mine, set_aside


def data_form() -> str:
    """The word for how the ranks of an ``n_gpu > 1`` run come by their parse and features (``rowpair_msa.FEATS_FORM``, one of the core's
    ``rankdata.DATA_FORMS``: ``rank0_bcast``)."""
    from .rowpair_msa import FEATS_FORM
    return FEATS_FORM


# ------------------------------------------------------------------ the worker's one zygote (a Receiver on a rank > 0 of an n_gpu > 1 run)
_ZYGOTE = None


def _new(preload: Sequence[str] = PRELOAD):
    """A :class:`Zygote` — or, on a rank > 0 of an ``n_gpu > 1`` launch (the core launcher's rank environment), a :class:`Receiver`."""
    P, rank = rank_env()
    if P > 1 and rank > 0:
        sys.stderr.write(f"{PREFIX} rank={rank} of {P}: no zygote — rank 0 parses every input, this rank receives its result and processed directory\n"); sys.stderr.flush()
        return Receiver()
    return Zygote(preload)


def start(forked_at: str = "launch", preload: Sequence[str] = PRELOAD):
    """Fork the process-wide zygote now (idempotent). worker_launch calls this first — before CUDA, hooks, routes and the worker script.
    On a rank > 0 of an ``n_gpu > 1`` launch nothing is forked (:class:`Receiver`)."""
    global _ZYGOTE
    if _ZYGOTE is None:
        _ZYGOTE = _new(preload)
    return _ZYGOTE.start(forked_at=forked_at)


def zygote():
    """The process-wide zygote (or receiver); forked here on first use when worker_launch did not start it (named ``forked_at=first_use`` on its line)."""
    global _ZYGOTE
    if _ZYGOTE is None:
        _ZYGOTE = _new()
    if isinstance(_ZYGOTE, Zygote) and _ZYGOTE.pid is None:
        _ZYGOTE.start(forked_at="first_use")
    return _ZYGOTE


def close() -> None:
    global _ZYGOTE
    if _ZYGOTE is not None:
        _ZYGOTE.close()
        _ZYGOTE = None


def reset_for_tests() -> None:
    """Close the zygote and clear the process state (the tests' fixture): the store keys of a launch (``parse/<n>``) restart at 1."""
    close()
    _CALLS["parse"] = 0
