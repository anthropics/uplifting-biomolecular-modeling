"""Lever IO1 — the PDB writers of ``rfdiffusion.util`` (``writepdb``, ``writepdb_multi``) re-expressed over numpy arrays, byte for byte.

Upstream writes every ATOM record through one ``%`` format per atom after indexing torch tensors element by element (``util.py:272-414``
``writepdb``; ``:661-722`` ``writepdb_multi``: per atom a ``torch.all(torch.isnan(...))`` and three 0-d tensor reads); on the trajectory files of
one design (2 files × 50 models × L residues × 14 atom slots) that is seconds of CPU inside every design's completion window. These two
functions produce the same bytes from the same arguments — the same format string, the same operands converted to the same Python numbers
(``tolist()`` of a float32 array is the exact widening ``float(tensor)`` performs), the same atom order, skips, counters, chain letters,
residue numbers and ``ENDMDL`` records — with the per-atom tensor indexing replaced by one host copy per model.

Proof in the process, not only in the tests: the first call of each writer per argument signature (shapes, dtypes, flags) also runs
upstream's own function into a sibling file and compares the bytes (``_verify``); unequal bytes keep UPSTREAM's file as the output, count
``n_mismatch`` and print ``pdb writer lever IO1: MISMATCH …`` — the driver process then exits 5 at the end of the pass (``driver_run``), so a
divergent writer can neither ship a wrong byte nor pass silently. ``arm()`` patches the two names on ``rfdiffusion.util`` before the kit driver
imports them (``from rfdiffusion.util import writepdb_multi, writepdb`` — ``rfd_bench.py:81``); the stock command
line never runs this module.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional

TAG = "pdb writer lever IO1"
ENV, VALUE = "RFD_PDBIO", "1"                       # the lever's switch: the mode's environment row (registry IO1), exported by stack.driver_environment
FMT = "%-6s%5s %4s %3s %s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f\n"   # util.py:298 = :316 = :335 = :399 = :700, verbatim

_state = dict(armed=False, n_calls=0, n_verified=0, n_mismatch=0, verified_keys=[], mismatches=[], seconds=0.0)
_orig: Dict[str, Callable] = {}


def stats() -> dict:
    return dict(_state, verified_keys=list(_state["verified_keys"]), mismatches=list(_state["mismatches"]))


# ----------------------------------------------------------------------------------------------------------------- the text (no torch)
HIS_D = (" N  ", " CA ", " C  ", " O  ", " CB ", " CG ", " NE2", " CD2", " CE1", " ND1", None, None, None, None, " H  ", " HA ", "1HB ", "2HB ",
         " HD2", " HE1", " HD1", None, None, None, None, None, None)          # util.py:363-391, upstream's protonated-histidine atom names


def text_single(al, ndim, natoms, s_list, idx, Bf, binderlen, chain_idx, his_d, aa2long, num2aa) -> str:
    """The ATOM records ``writepdb`` writes (util.py:285-414) from plain Python numbers: ``al`` the coordinates as nested lists (``tolist()`` of the
    squeezed array), ``ndim`` / ``natoms`` its rank and atom-axis length, ``s_list`` / ``idx`` / ``Bf`` the residue types, residue numbers and clamped
    B-factors as lists, ``his_d[i]`` upstream's histidine test result per residue (util.py:358-361; read only on the 14/27-atom branch)."""
    lines = []
    ctr = 1
    for i, s in enumerate(s_list):
        if chain_idx is None:
            if binderlen is not None:
                chain = "A" if i < binderlen else "B"
            elif binderlen is None:
                chain = "A"
        else:
            chain = chain_idx[i]
        rn, ri, B = num2aa[s], idx[i], Bf[i]
        if ndim == 2:
            x = al[i]
            lines.append(FMT % ("ATOM", ctr, " CA ", rn, chain, ri, x[0], x[1], x[2], 1.0, B)); ctr += 1
        elif natoms == 3:
            row = al[i]
            for j, atm_j in enumerate((" N  ", " CA ", " C  ")):
                x = row[j]; lines.append(FMT % ("ATOM", ctr, atm_j, rn, chain, ri, x[0], x[1], x[2], 1.0, B)); ctr += 1
        elif natoms == 4:
            row = al[i]
            for j, atm_j in enumerate((" N  ", " CA ", " C  ", " O  ")):
                x = row[j]; lines.append(FMT % ("ATOM", ctr, atm_j, rn, chain, ri, x[0], x[1], x[2], 1.0, B)); ctr += 1
        else:
            if natoms != 14 and natoms != 27:
                print("bad size!", (len(al), natoms, 3))
                assert False
            atms = HIS_D if (s == 8 and his_d[i]) else aa2long[s]
            row = al[i]
            for j, atm_j in enumerate(atms):
                if j < natoms and atm_j is not None:
                    x = row[j]; lines.append(FMT % ("ATOM", ctr, atm_j, rn, chain, ri, x[0], x[1], x[2], 1.0, B)); ctr += 1
    return "".join(lines)


def text_multi(models, seq_rows, Bf, chain_ids, stop, aa2long, num2aa) -> str:
    """The records ``writepdb_multi`` writes (util.py:680-722): ``models`` a list of (coordinates as nested lists, all-NaN mask as nested lists of
    bool) per model, ``seq_rows`` the per-model residue types, ``Bf`` the clamped B-factors, ``stop`` the atom index the two ``break``s of
    util.py:688-691 amount to (3 backbone-only, 14 without hydrogens, None otherwise); one ``ENDMDL`` per model."""
    chunks = []
    for (al, nanall), srow in zip(models, seq_rows):
        lines = []
        ctr = 1
        for i, s in enumerate(srow):
            atms = aa2long[s]
            rn, B = num2aa[s], Bf[i]
            chain_id = chain_ids[i] if chain_ids is not None else "A"
            row, nrow = al[i], nanall[i]
            for j, atm_j in enumerate(atms):
                if stop is not None and j >= stop:
                    break
                if atm_j is None:
                    continue
                if nrow[j]:                                                 # IndexError past the atom axis exactly where upstream's atomscpu[i, j] raises
                    continue
                x = row[j]
                lines.append(FMT % ("ATOM", ctr, atm_j, rn, chain_id, i + 1, x[0], x[1], x[2], 1.0, B)); ctr += 1
        lines.append("ENDMDL\n")
        chunks.append("".join(lines))
    return "".join(chunks)


# ----------------------------------------------------------------------------------------------------------------- the writers (torch in, upstream's signatures)
def writepdb(filename, atoms, seq, binderlen=None, idx_pdb=None, bfacts=None, chain_idx=None):
    """``rfdiffusion.util.writepdb`` (util.py:272-414), the same bytes."""
    import torch
    from rfdiffusion.chemical import aa2long, num2aa
    scpu = seq.cpu().squeeze()
    atomscpu = atoms.cpu().squeeze()
    if bfacts is None:
        bfacts = torch.zeros(atomscpu.shape[0])
    if idx_pdb is None:
        idx_pdb = 1 + torch.arange(atomscpu.shape[0])
    Bf = torch.clamp(bfacts.cpu(), 0, 1).tolist()
    a = atomscpu.detach().numpy()
    s_list = scpu.tolist()
    if not isinstance(s_list, list):                                  # a 0-d squeeze (L == 1): upstream iterates a 0-d tensor, which raises; so does this
        raise TypeError("iteration over a 0-d tensor")
    idx = [int(v) for v in (idx_pdb.tolist() if hasattr(idx_pdb, "tolist") else idx_pdb)]
    natoms = a.shape[1] if a.ndim > 1 else 0
    his_d = None
    if a.ndim == 3 and natoms in (14, 27):                              # upstream's histidine test (util.py:358-361), evaluated with torch as upstream evaluates it
        his_d = [bool(s == 8 and torch.linalg.norm(atomscpu[i, 9, :] - atomscpu[i, 5, :]) < 1.7) for i, s in enumerate(s_list)]
    text = text_single(a.tolist(), a.ndim, natoms, s_list, idx, Bf, binderlen, chain_idx, his_d, aa2long, num2aa)
    with open(filename, "w") as f:
        f.write(text)


def writepdb_multi(filename, atoms_stack, bfacts, seq_stack, backbone_only=False, chain_ids=None, use_hydrogens=True):
    """``rfdiffusion.util.writepdb_multi`` (util.py:661-722), the same bytes."""
    import numpy as np
    import torch
    from rfdiffusion.chemical import aa2long, num2aa
    from rfdiffusion.util import N_BACKBONE_ATOMS, N_HEAVY
    if seq_stack.ndim != 2:
        T = atoms_stack.shape[0]
        seq_stack = torch.tile(seq_stack, (T, 1))
    seq_stack = seq_stack.cpu()
    Bf = torch.clamp(bfacts.cpu(), 0, 1).tolist()
    stop = N_BACKBONE_ATOMS if backbone_only else (N_HEAVY if not use_hydrogens else None)   # the two `break`s of util.py:688-691, in their order
    models = []
    for atoms, _srow in zip(atoms_stack, seq_stack):                        # upstream's zip: the shorter of the two stacks bounds the models written
        ac = atoms.detach().cpu() if isinstance(atoms, torch.Tensor) else atoms
        a = ac.numpy() if isinstance(ac, torch.Tensor) else np.asarray(ac)
        models.append((a.tolist(), np.isnan(a).all(axis=-1).tolist()))       # util.py:692 `torch.all(torch.isnan(atomscpu[i, j]))` per atom, as one mask per model
    text = text_multi(models, seq_stack.tolist(), Bf, chain_ids, stop, aa2long, num2aa)
    with open(filename, "w") as f:
        f.write(text)


# ----------------------------------------------------------------------------------------------------------------- proof + arming
def _key(name: str, args: tuple, kwargs: dict) -> str:
    def d(v):
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            return f"{tuple(v.shape)}:{v.dtype}"
        if isinstance(v, (list, tuple)):
            return f"seq{len(v)}"
        return repr(v) if isinstance(v, (bool, int, float, str, type(None))) else type(v).__name__
    return name + "(" + ",".join(d(a) for a in args[1:]) + ";" + ",".join(f"{k}={d(v)}" for k, v in sorted(kwargs.items())) + ")"


def _verified(name: str, fast: Callable) -> Callable:
    import time

    def wrapper(filename, *args, **kwargs):
        t0 = time.time()
        fast(filename, *args, **kwargs)
        _state["n_calls"] += 1
        key = _key(name, (filename,) + args, kwargs)
        if key not in _state["verified_keys"]:                              # first call per signature: upstream's own writer beside ours, bytes compared
            ref = f"{filename}.upstream~"
            _orig[name](ref, *args, **kwargs)
            with open(filename, "rb") as fa, open(ref, "rb") as fb:
                same = fa.read() == fb.read()
            if same:
                os.remove(ref)
                _state["n_verified"] += 1
                _state["verified_keys"].append(key)
                print(f"{TAG}: confirmed {key} byte-equal to rfdiffusion.util.{name}", flush=True)
            else:
                os.replace(ref, filename)                                   # the output is upstream's bytes; the event is counted and loud
                _state["n_mismatch"] += 1
                _state["mismatches"].append(key)
                print(f"{TAG}: MISMATCH {key} — upstream's bytes kept for {os.path.basename(filename)}; the pass exits 5", flush=True)
        _state["seconds"] += time.time() - t0
    wrapper.__name__ = name
    wrapper.__wrapped_upstream__ = True
    return wrapper


def arm(value: Optional[str]) -> Optional[dict]:
    """Under ``RFD_PDBIO=1``: replace ``rfdiffusion.util.writepdb`` / ``writepdb_multi`` with the confirmed writers above before the kit
    driver imports them; prints the lever's applied-line. Any other value: nothing (the mode's row does not carry the lever)."""
    if value != VALUE:
        return None
    import rfdiffusion.util as U
    if _state["armed"]:
        return stats()
    for name, fast in (("writepdb", writepdb), ("writepdb_multi", writepdb_multi)):
        _orig[name] = getattr(U, name)
        setattr(U, name, _verified(name, fast))
    _state["armed"] = True
    print(f"{TAG}: armed (rfdiffusion.util.writepdb, writepdb_multi -> rfdiffusion1_opt.pdbio; first call per signature confirmed against upstream's writer)", flush=True)
    return stats()


def final_line() -> str:
    return "PDBIO_FINAL " + json.dumps(stats(), default=str)
