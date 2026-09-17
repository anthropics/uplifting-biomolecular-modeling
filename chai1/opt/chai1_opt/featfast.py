"""The feature-build levers (registry: ``prefetch``, ``confmemo``; modes.KitMode.postproc). ``prefetch`` — the NEXT item's ``make_all_atom_feature_context`` (FASTA parse,
reference conformers, MSA parquet load + pairing, template / embedding / restraint contexts: CPU work whose cost does not grow with the crop, no RNG draw,
built before ``set_seed`` in stock — the premise of the driver's W2) computed on one helper THREAD while the current item folds on the GPU, so the
driver's per-item ``features_s`` leaves the critical path from the second item of a process on. An install-style rebinding of
``chai_lab.chai1.make_all_atom_feature_context`` (the attribute the kit driver ``chai_worker.py`` calls per item); the function that runs, its
arguments and therefore the context it returns are upstream's own: exact by construction (the fold receives the same tensors).

How the next item is known: the driver runs the kit worker as ``__main__`` with its own command line (``chai_worker.py <uid,uid,…> <seeds> <out>
… --msa_dir …``): the uid list is ``sys.argv[1]``, each uid's FASTA source is the worker's ``chai_proto.input_spec(uid)["fasta_src"]`` (the file the
worker copies to its work dir before its own call), and every other argument of the call is per-process (``chai_proto.RUN_KW``) — the helper's call
for item i+1 is the current call with item i+1's FASTA (a private byte-identical copy) and a private ``output_dir`` (upstream writes there only in
its server modes, where this lever stands aside). The worker's later call is matched by FASTA CONTENT (sha256) + the bound keyword arguments; a call
that matches no prefetched context is computed inline as upstream's statement (counted ``fallback_by=first_item|miss``), never guessed.

No prefetch is started, BY NAME (``prefetch_aside_by=``; the next item is then computed inline): ``server_mode`` (``use_msa_server`` / ``use_templates_server``: network I/O and
files under ``output_dir``), ``esm_embeddings`` (``use_esm_embeddings=True``: the ESM-2 encode would run on the GPU beside the fold — not this
lever's to schedule), ``no_worker`` (not under the kit driver: no uid list to read), ``last_item`` (no item follows), ``error:<exc>`` (the kick raised). A helper
build that raised is ``fallback_by=error:<exc>``: the inline call then raises or succeeds on its own.

LEVER-line evidence / EXIT tally: ``prefetch_served= prefetch_fallback= prefetch_kicked= prefetch_build_s= prefetch_wait_s= [prefetch_fallback_by=…] [prefetch_aside_by=…]``.

``confmemo`` — the reference-conformer library loaded ONCE per process. Upstream's ``load_chains_from_raw(inputs, …, tokenizer=None)`` constructs
``AllAtomResidueTokenizer(RefConformerGenerator())`` on every call, and ``RefConformerGenerator.__init__`` unpickles the whole CCD conformer library
(``_load_apkl_conformers``: the bulk of the per-item feature build at any size; upstream's own docstring calls constructing the conformer
generator expensive and meant to be cached). The lever rebinds ``chai_lab.chai1.load_chains_from_raw`` (the attribute ``make_all_atom_feature_context``
calls) to the same function with ``tokenizer=`` the process's one tokenizer (built by upstream's own two constructors on first use) whenever the caller
passed none: the same objects upstream would have built, read-only reference data + RDKit's seeded embedding for residues outside the library -> the same
chains, exact by construction (W1's pattern for the model files). EXIT tally: ``confmemo_served= confmemo_builds=``.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Dict, Optional, Tuple

LEVERS = ("confmemo", "prefetch")
FEAT_NAME = "make_all_atom_feature_context_prefetch"     # registry probe: ("attr", "chai_lab.chai1", "make_all_atom_feature_context", FEAT_NAME)
CHAINS_NAME = "load_chains_from_raw_confmemo"           # registry probe: ("attr", "chai_lab.chai1", "load_chains_from_raw", CHAINS_NAME)
EXPECTED_FALLBACKS = ("first_item", "miss", "server_mode", "esm_embeddings", "no_worker", "last_item")
_STATE: Dict[str, object] = {"installed": (), "orig": None, "st": None, "orig_chains": None, "cm": None}


class LeverUnavailable(RuntimeError):
    pass


class _State:
    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor                           # at install time only (see postproc._TailState)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chai1_prefetch")
        self.lock = threading.Lock()
        self.futures: Dict[tuple, object] = {}
        self.served = 0; self.fallback_by: Dict[str, int] = {}; self.aside_by: Dict[str, int] = {}
        self.build_s = 0.0; self.wait_s = 0.0; self.kicked = 0; self.cursor = 0
        self.sha_by_uid: Dict[str, str] = {}
        self.priv = None

    def fallback(self, why):
        self.fallback_by[why] = self.fallback_by.get(why, 0) + 1

    def aside(self, why):
        self.aside_by[why] = self.aside_by.get(why, 0) + 1

    def privdir(self):
        if self.priv is None:
            self.priv = tempfile.mkdtemp(prefix=f"chai1_prefetch_{os.getpid()}_")
        return self.priv


def _sha(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _key(orig, fasta_file, output_dir, kw) -> tuple:
    """(FASTA content sha256, the keyword arguments as passed except output_dir) — what makes two calls return the same context. The kit worker
    passes every argument by keyword (chai_proto.RUN_KW), and the helper's build passes the same dictionary."""
    return (_sha(fasta_file), tuple(sorted((k, str(v)) for k, v in kw.items() if k != "output_dir")))


def _worker_uids() -> Optional[list]:
    cp = sys.modules.get("chai_proto")
    if cp is None or not hasattr(cp, "input_spec") or len(sys.argv) < 4:
        return None
    try:
        return [u for u in str(sys.argv[1]).split(",") if u]
    except Exception:  # noqa: BLE001
        return None


def _kick_next(st: _State, orig, fasta_file, output_dir, kw):
    if kw.get("use_msa_server") or kw.get("use_templates_server"):
        st.aside("server_mode"); return
    if kw.get("use_esm_embeddings", True):                                      # upstream's default is True: an absent word means ESM on
        st.aside("esm_embeddings"); return
    uids = _worker_uids()
    if not uids:
        st.aside("no_worker"); return
    cp = sys.modules["chai_proto"]
    cur = _sha(fasta_file)
    nxt = None
    for i in range(st.cursor, len(uids)):
        u = uids[i]
        if u not in st.sha_by_uid:
            try:
                st.sha_by_uid[u] = _sha(cp.input_spec(u)["fasta_src"])
            except BaseException:  # noqa: BLE001 — an unreadable uid is the worker's to name; no prefetch past it
                st.sha_by_uid[u] = ""
        if st.sha_by_uid[u] == cur:
            st.cursor = i + 1
            nxt = uids[i + 1] if i + 1 < len(uids) else None
            break
    if nxt is None:
        st.aside("last_item"); return
    spec = cp.input_spec(nxt)
    d = os.path.join(st.privdir(), f"item{st.cursor}")
    os.makedirs(d, exist_ok=True)
    priv = os.path.join(d, f"{spec.get('key', 'next')}.fasta")
    shutil.copy(spec["fasta_src"], priv)
    from pathlib import Path
    key = _key(orig, Path(priv), Path(d) / "out", kw)

    def build():
        t0 = time.perf_counter()
        try:
            return orig(fasta_file=Path(priv), output_dir=Path(d) / "out", **kw)     # by keyword, as the worker calls it (the kit's own wrappers read fasta_file= by name)
        finally:
            with st.lock:
                st.build_s += time.perf_counter() - t0
    with st.lock:
        st.futures[key] = st.pool.submit(build); st.kicked += 1


def _make_prefetch(st: _State, orig):
    def make_all_atom_feature_context_prefetch(fasta_file=None, *, output_dir=None, **kw):
        """chai_lab.chai1.make_all_atom_feature_context — the same call; served from the helper thread's earlier build of THIS call when there is one (prefetch)."""
        from pathlib import Path
        if fasta_file is None or output_dir is None:                             # arguments upstream would reject: its own statement raises the way it does
            return orig(fasta_file=fasta_file, output_dir=output_dir, **kw)
        fasta_file = Path(fasta_file)
        try:
            key = _key(orig, fasta_file, output_dir, kw)
        except Exception:  # noqa: BLE001
            return orig(fasta_file=fasta_file, output_dir=output_dir, **kw)
        with st.lock:
            fut = st.futures.pop(key, None)
        ctx = None
        if fut is not None:
            t0 = time.perf_counter()
            try:
                ctx = fut.result()
                st.served += 1
            except BaseException as e:  # noqa: BLE001 — the helper's build raised: the inline statement runs (and raises or not on its own)
                st.fallback("error:" + type(e).__name__)
                sys.stderr.write(f"[chai1-opt] LEVER prefetch fallback_by=error:{type(e).__name__}: {str(e)[:200]} — the inline statement serves this item\n")
            finally:
                st.wait_s += time.perf_counter() - t0
        else:
            st.fallback("first_item" if st.served == 0 and not st.fallback_by else "miss")
        if ctx is None:
            ctx = orig(fasta_file=fasta_file, output_dir=output_dir, **kw)
        try:
            _kick_next(st, orig, fasta_file, output_dir, kw)
        except BaseException as e:  # noqa: BLE001 — no prefetch for the next item; it will be computed inline (miss)
            st.aside("error:" + type(e).__name__)
        return ctx

    make_all_atom_feature_context_prefetch.__name__ = FEAT_NAME
    make_all_atom_feature_context_prefetch.chai1_opt_lever = "prefetch"
    make_all_atom_feature_context_prefetch.__wrapped_statement__ = orig
    return make_all_atom_feature_context_prefetch


# --------------------------------------------------------------------------------------------------------------------------- confmemo
class _ConfMemo:
    def __init__(self):
        self.tokenizer = None; self.lock = threading.Lock(); self.served = 0; self.builds = 0; self.build_s = 0.0; self.passed_through = 0

    def get(self):
        with self.lock:
            if self.tokenizer is None:
                t0 = time.perf_counter()
                from chai_lab.data.dataset.structure.all_atom_residue_tokenizer import AllAtomResidueTokenizer
                from chai_lab.data.sources.rdkit import RefConformerGenerator
                self.tokenizer = AllAtomResidueTokenizer(RefConformerGenerator())        # upstream's own two constructors (inference_dataset.load_chains_from_raw), once
                self.builds += 1; self.build_s += time.perf_counter() - t0
            return self.tokenizer


def _make_chains(cm: _ConfMemo, orig):
    def load_chains_from_raw_confmemo(*a, tokenizer=None, **kw):
        """chai_lab.data.dataset.inference_dataset.load_chains_from_raw with the process's one tokenizer when the caller built none (confmemo)."""
        if tokenizer is None:
            tokenizer = cm.get(); cm.served += 1
        else:
            cm.passed_through += 1
        return orig(*a, tokenizer=tokenizer, **kw)

    load_chains_from_raw_confmemo.__name__ = CHAINS_NAME
    load_chains_from_raw_confmemo.chai1_opt_lever = "confmemo"
    load_chains_from_raw_confmemo.__wrapped_statement__ = orig
    return load_chains_from_raw_confmemo


def install(levers: Tuple[str, ...], chai1_mod=None) -> dict:
    """Rebind ``chai_lab.chai1``'s feature-build attributes for ``levers`` (idempotent per lever; unknown names -> LeverUnavailable)."""
    unknown = [l for l in levers if l not in LEVERS]
    if unknown:
        raise LeverUnavailable(f"unknown feature levers {unknown}; known: {LEVERS}")
    todo = [l for l in levers if l not in _STATE["installed"]]
    if not todo:
        return {"featfast_levers": list(_STATE["installed"])}
    import importlib
    C1 = chai1_mod or importlib.import_module("chai_lab.chai1")
    if "confmemo" in todo:
        cm = _ConfMemo()
        _STATE["orig_chains"] = getattr(C1, "load_chains_from_raw", None)
        C1.load_chains_from_raw = _make_chains(cm, _STATE["orig_chains"])
        _STATE["cm"] = cm
    if "prefetch" in todo:
        st = _State()
        _STATE["orig"] = getattr(C1, "make_all_atom_feature_context", None)
        C1.make_all_atom_feature_context = _make_prefetch(st, _STATE["orig"])
        _STATE["st"] = st
        import atexit
        try:
            atexit.register(_shutdown)
        except RuntimeError:                                                         # registered during shutdown: the helper is a daemon-free pool with nothing queued; nothing to arm
            pass
    _STATE["installed"] = tuple(l for l in LEVERS if l in todo or l in _STATE["installed"])
    return {"featfast_levers": list(_STATE["installed"])}

def _shutdown():
    st = _STATE["st"]
    if st is not None:
        try:
            st.pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        if st.priv:
            shutil.rmtree(st.priv, ignore_errors=True)


def applied() -> Tuple[str, ...]:
    return tuple(_STATE["installed"])


def census() -> dict:
    out = {}
    cm = _STATE["cm"]
    if cm is not None:
        out["confmemo"] = {"served": cm.served, "builds": cm.builds, "build_s": round(cm.build_s, 3), "passed_through": cm.passed_through}
    st = _STATE["st"]
    if st is None:
        return out
    out["prefetch"] = {"served": st.served, "fallback": sum(st.fallback_by.values()), "fallback_by": dict(sorted(st.fallback_by.items())),
                         "aside_by": dict(sorted(st.aside_by.items())), "kicked": st.kicked, "build_s": round(st.build_s, 3), "wait_s": round(st.wait_s, 3)}
    return out


def evidence(name: str) -> dict:
    return census().get(name, {}) if name in _STATE["installed"] else {}


def gate() -> Tuple[bool, str]:
    c = census().get("prefetch")
    if c is None:
        return True, ""
    bad = [k for k in c["fallback_by"] if k not in EXPECTED_FALLBACKS and not k.startswith("error:")]
    return (not bad), (f"prefetch:fallback_by={','.join(bad)}" if bad else "")


def verdict() -> Optional[str]:
    ok, why = gate()
    return None if ok or not _STATE["installed"] else why


def tally_fields() -> list:
    out = []
    cm = census().get("confmemo")
    if cm is not None:
        out.append(f"confmemo_served={cm['served']} confmemo_builds={cm['builds']}")
    c = census().get("prefetch")
    if c is None:
        return out
    f = f"prefetch_served={c['served']} prefetch_fallback={c['fallback']} prefetch_kicked={c['kicked']} prefetch_build_s={c['build_s']} prefetch_wait_s={c['wait_s']}"
    if c["fallback_by"]:
        f += " prefetch_fallback_by=" + ",".join(f"{k}:{n}" for k, n in c["fallback_by"].items())
    if c["aside_by"]:
        f += " prefetch_aside_by=" + ",".join(f"{k}:{n}" for k, n in c["aside_by"].items())
    return out + [f]


def reset_for_tests(chai1_mod=None) -> None:
    C1 = chai1_mod if chai1_mod is not None else sys.modules.get("chai_lab.chai1")
    if C1 is not None:
        for attr, key in (("make_all_atom_feature_context", "orig"), ("load_chains_from_raw", "orig_chains")):
            cur = getattr(C1, attr, None)
            if getattr(cur, "chai1_opt_lever", None) in LEVERS:
                if _STATE[key] is not None:
                    setattr(C1, attr, _STATE[key])
                else:
                    delattr(C1, attr)
    st = _STATE["st"]
    if st is not None:
        st.pool.shutdown(wait=True)
    _STATE.update(installed=(), orig=None, st=None, orig_chains=None, cm=None)
