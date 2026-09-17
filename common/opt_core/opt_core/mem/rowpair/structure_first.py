"""opt_core.mem.rowpair.structure_first — STRUCTURE FIRST on the row-sharded line: a sample's structure file is on disk the moment
the sampler returns, BEFORE that sample's confidence heads start.

What it does (rank 0 only; ordering, not arithmetic). After a sample-loop pass's roll-out the binding calls :func:`write_early` with the pass's
coordinates ``x`` ``[1, k, N_atom, 3]`` and the engine's own STATIC structure writer (``write_structure(atom_array=, predicted_coords=, plddt=,
output_file=)`` — the statement the engine's output writer applies after the heads: same atom order, same bonds, same tables, same numeric
formatting). Each sample's file is written to the path that writer will use, ``<output_dir>/<query>/seed_<s>/<query>_seed_<s>_sample_<k>_model.<fmt>``,
with the per-atom pLDDT column — which does not exist yet — set to ``0.00``, plus a sidecar ``<prefix>_structure_first.json``
(``confidence_written: false``, the file's sha256, sample index, UTC stamp, seconds spent). When the heads finish, the engine's UNCHANGED
``on_predict_batch_end`` overwrites the file with the pLDDT-filled bytes (identical to today's file: same writer, same inputs) and
:func:`mark_final` flips the sidecar to ``confidence_written: true`` with the final sha256 — only when the engine actually rewrote the bytes
(``outputs`` present AND the sha256 changed). A death inside the confidence stage therefore leaves the structure on disk: the model file
(B-factor column 0.00) + the sidecar saying ``confidence not written``; :func:`scan` gives the run record its ``structure_first`` block, the
names of the structure-only files (the binding's exit tally does NOT count them as structures: a run whose confidences were not written exits
non-zero by name) and the ``notes`` the launcher prints.

Why the B-factor column: the engine stores pLDDT in ``_atom_site.B_iso_or_equiv``, so no structure file written before the heads can carry it;
every other byte of the early file equals the final file.

Census (:func:`.evidence.record_schedule`, rank 0): ``structure_first=on|off|skipped:<why>``, ``structure_first_files=<n>`` (this process),
``structure_first_write_s=<s,...>`` (seconds per early write — rank-0 CPU work on the critical path while the peers wait in the heads' first
collective; :data:`WRITE_WARN_S` is the stated bound above which the line says ``SLOW`` by name; the peers' watchdog is the row-sharded
line's collective timeout, minutes), ``structure_first_not_written=<k>`` after :func:`mark_final`. The same words ride the ``[structure_first]``
log lines. Off switch: the binding's kit switch (``write_early(..., enabled=False)`` names it and writes nothing: today's order). A salvage step
outside the run can key off the sidecar: the structure exists the moment ``*_structure_first.json`` does."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
from typing import Callable, Dict, List, Optional

SIDE_SUFFIX = "_structure_first.json"
PLDDT_PLACEHOLDER = 0.0
WRITE_WARN_S = 60.0                                  # an early write slower than this is named SLOW on its line (the peers wait in a collective meanwhile)

STATE: Dict[str, object] = {"writer": None, "batch": None, "written": {}, "words": {}, "write_s": []}   # rank-local


def utc() -> str:
    """``2026-09-09T12:44:30.123Z`` — the log lines of this module carry wall-clock stamps (the ordering evidence)."""
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + (".%03dZ" % int((t % 1) * 1000))


def _record(**facts) -> None:
    try:
        from .evidence import record_schedule
        record_schedule(**facts)
    except ImportError:                              # a bare unit test of this file without the package: the words stay in STATE["words"]
        pass


def _set_words(**facts) -> None:
    STATE["words"].update(facts)
    _record(**facts)


def capture(pl_module, batch, is_writer: Callable[[object], bool]) -> None:
    """Remember the engine's output-writer callback instance (``output_dir``, ``structure_format``; ``is_writer(cb)`` picks it among the
    trainer's callbacks) and the batch object the writer will see (``query_id``, ``seed``, ``atom_array``) — called from the rank's
    batch-transfer hook, i.e. before ``predict_step``."""
    try:
        tr = getattr(pl_module, "trainer", None) or getattr(pl_module, "_trainer", None)
        for cb in (getattr(tr, "callbacks", None) or []):
            if is_writer(cb):
                STATE["writer"] = cb
                break
    except Exception:  # noqa: BLE001 - a capability probe (OOM census NOT_REROUTES): no trainer / no writer -> write_early says `skipped:no_writer_or_batch_fields` by name
        pass
    STATE["batch"] = batch


def _sha256(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except OSError:
        return None


def _field(batch, key, fallback):
    v = batch.get(key) if isinstance(batch, dict) else None
    if v is None and isinstance(fallback, dict):
        v = fallback.get(key)
    return v


def naming(batch) -> Optional[dict]:
    """The writer's naming inputs for the ONE element of an inference batch: ``{query_id, seed, atom_array, subdir, fmt, output_dir, batch_size}``,
    or None when the writer instance or the fields are unavailable."""
    w = STATE.get("writer")
    fb = STATE.get("batch")
    qids = _field(batch, "query_id", fb); seeds = _field(batch, "seed", fb); arrays = _field(batch, "atom_array", fb)
    if w is None or qids is None or seeds is None or arrays is None:
        return None
    try:
        n = len(qids)
        seed = seeds[0]
        if hasattr(seed, "item"):
            seed = seed.item()
        query_id = str(qids[0])
    except Exception:  # noqa: BLE001 - a field probe (OOM census NOT_REROUTES): malformed naming fields -> `skipped:` by name, the heads run as usual
        return None
    out_dir = getattr(w, "output_dir", None)
    if out_dir is None:
        return None
    return {"query_id": query_id, "seed": seed, "atom_array": arrays[0], "fmt": getattr(w, "structure_format", "cif") or "cif", "batch_size": int(n),
            "subdir": Path(str(out_dir)) / query_id / f"seed_{seed}", "output_dir": str(out_dir)}


def _key(query_id, seed) -> str:
    return f"{query_id}|{seed}"


def write_early(batch, x, sample0: int, write_structure: Callable, log=None, enabled: bool = True, switch: str = "the kit's structure-first switch") -> List[str]:
    """Write the structures of samples ``[sample0, sample0 + k)`` from ``x`` ``[1, k, N_atom, 3]`` (rank 0's adopted coordinates) with the
    pLDDT column = 0.00 through the engine's static writer ``write_structure``; one sidecar per file. Returns the paths written ([] when off /
    skipped: named on the line and in the census)."""
    log = log or (lambda m: None)
    if not enabled:
        _set_words(structure_first="off")
        log(f"[structure_first] off ({switch}=0): structures are written after the confidence heads (today's order)")
        return []
    pf = naming(batch)
    if pf is None:
        _set_words(structure_first="skipped:no_writer_or_batch_fields")
        log("[structure_first] skipped: the engine's output-writer instance or the batch's query_id/seed/atom_array is not available on this rank (named; the heads run as usual)")
        return []
    if pf["batch_size"] != 1:                                   # inference batches hold ONE query; anything else is named, never written from element 0 silently
        _set_words(structure_first=f"skipped:batch_size_{pf['batch_size']}")
        log(f"[structure_first] skipped: batch size {pf['batch_size']} != 1 (the early write handles one query per batch, as the engine's inference does; named)")
        return []
    import numpy as np
    t0 = time.time()
    pf["subdir"].mkdir(parents=True, exist_ok=True)
    coords = x[0].detach().float().cpu().numpy()                  # [k, N_atom, 3]: the statement the engine applies to outputs["atom_positions_predicted"][b]
    n_atom = int(coords.shape[-2])
    written = []
    for j in range(int(coords.shape[0])):
        s = int(sample0) + j
        prefix = pf["subdir"] / f"{pf['query_id']}_seed_{pf['seed']}_sample_{s + 1}"
        out = Path(f"{prefix}_model.{pf['fmt']}")
        write_structure(atom_array=pf["atom_array"], predicted_coords=coords[j], plddt=np.full((n_atom,), PLDDT_PLACEHOLDER, dtype=np.float32), output_file=out)
        side = {"schema": 1, "what": "structure_first", "query_id": pf["query_id"], "seed": pf["seed"], "sample": s + 1, "structure_file": out.name,
                "confidence_written": False, "note": "confidence not written (yet): model file carries the structure with the pLDDT (B-factor) column = 0.00",
                "structure_first_sha256": _sha256(out), "structure_first_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "n_atoms": n_atom}
        Path(f"{prefix}{SIDE_SUFFIX}").write_text(json.dumps(side, indent=1) + "\n", encoding="utf-8")
        STATE["written"].setdefault(_key(pf["query_id"], pf["seed"]), {})[str(out)] = side
        written.append(str(out))
    dt = round(time.time() - t0, 2)
    STATE["write_s"].append(dt)
    n_files = sum(len(v) for v in STATE["written"].values())
    _set_words(structure_first="on", structure_first_files=n_files, structure_first_write_s=",".join(f"{v:g}" for v in STATE["write_s"]))
    slow = f" SLOW: above the stated bound {WRITE_WARN_S:g}s (the peers wait in the heads' first collective meanwhile)" if dt > WRITE_WARN_S else ""
    log(f"[structure_first] {utc()} wrote {len(written)} structure(s) BEFORE the confidence heads in {dt:g}s{slow} (pLDDT column = 0.00 until the heads finish): "
        + ", ".join(os.path.basename(p) for p in written)
        + f" | census structure_first=on structure_first_files={n_files} structure_first_write_s={STATE['words']['structure_first_write_s']}")
    return written


def mark_final(batch, outputs=None, log=None) -> int:
    """After the engine's ``on_predict_batch_end`` ran on rank 0 for THIS batch (query, seed): every structure-first file of that query/seed the
    writer REWROTE (``outputs`` present AND the file's sha256 no longer equals the structure-first sha256) gets ``confidence_written: true`` with
    the final sha256; a file it did not rewrite (``outputs`` is None: ``predict_step`` caught an exception after the structure was written; or
    the bytes are unchanged) keeps ``confidence_written: false`` and the 'confidence not written' note. Files of OTHER queries/seeds are left as
    they are (already final or still pending their own batch end). Returns how many sidecars were flipped."""
    log = log or (lambda m: None)
    pf = naming(batch)
    keys = [_key(pf["query_id"], pf["seed"])] if pf is not None else list(STATE["written"].keys())   # no naming (writer gone): every pending file of this process
    n = kept = 0
    for key in keys:
        files = STATE["written"].get(key) or {}
        for path, side in list(files.items()):
            if side.get("confidence_written"):
                continue                                    # flipped by an earlier batch end: final already, not re-counted
            p = Path(path)
            sidep = Path(str(p).rsplit("_model.", 1)[0] + SIDE_SUFFIX)
            now_sha = _sha256(p)
            rewritten = outputs is not None and now_sha is not None and now_sha != side.get("structure_first_sha256")
            if rewritten:
                side = dict(side, confidence_written=True, note="confidence written: model file rewritten by the engine's writer with the pLDDT column",
                            final_sha256=now_sha, final_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            else:
                kept += 1
                side = dict(side, confidence_written=False,
                            note="confidence not written: the prediction ended before the engine's writer rewrote this file; it holds the structure with the pLDDT (B-factor) column = 0.00")
            try:
                sidep.write_text(json.dumps(side, indent=1) + "\n", encoding="utf-8")
                n += int(rewritten)
            except OSError:
                pass
            files[path] = side
    not_written = sum(1 for v in STATE["written"].values() for s in v.values() if not s.get("confidence_written"))
    _set_words(structure_first_not_written=not_written)
    if n:
        log(f"[structure_first] {utc()} confidence written: {n} structure file(s) rewritten by the engine's writer with the pLDDT column; sidecars updated"
            f" | census structure_first_not_written={not_written}")
    if kept:
        log(f"[structure_first] {utc()} confidence NOT written for {kept} structure file(s): they stay on disk as structure-only model files (pLDDT column 0.00); "
            f"the run record says so and the exit tally does not count them | census structure_first_not_written={not_written}")
    return n


def scan(out_dir: Optional[str]) -> Dict[str, object]:
    """The run record's ``structure_first`` block from the sidecars under ``out_dir``: ``{files, confidence_not_written: [names],
    structure_only_paths: [paths], notes: [...]}`` — ``structure_only_paths`` are the model files the exit tally must NOT count as structures."""
    block = {"files": 0, "confidence_not_written": [], "structure_only_paths": [], "notes": []}
    if not out_dir or not os.path.isdir(out_dir):
        return block
    for root, _dirs, files in os.walk(out_dir):
        for f in sorted(files):
            if f.endswith(SIDE_SUFFIX):
                block["files"] += 1
                try:
                    with open(os.path.join(root, f), encoding="utf-8") as fh:
                        side = json.load(fh)
                except (OSError, ValueError):
                    continue
                if not side.get("confidence_written"):
                    name = side.get("structure_file") or f[: -len(SIDE_SUFFIX)] + "_model.cif"
                    block["confidence_not_written"].append(name)
                    block["structure_only_paths"].append(os.path.join(root, name))
    if block["confidence_not_written"]:
        names = block["confidence_not_written"]
        block["notes"].append(f"confidence not written: {len(names)} structure-only model file(s) (pLDDT/B-factor column = 0.00, written before the confidence heads; "
                              f"not counted as structures by the exit tally): " + ", ".join(sorted(names)[:8]) + (" ..." if len(names) > 8 else ""))
    return block


def words() -> Dict[str, object]:
    """The census words of this process (also recorded through :func:`.evidence.record_schedule`)."""
    return dict(STATE["words"])


def reset() -> None:
    """Forget this process's structure-first state (tests; a second trainer in one process)."""
    STATE.update({"writer": None, "batch": None, "written": {}, "words": {}, "write_s": []})
