"""templates — structural templates on `pred`, on every route alike: what the query names, where the engine reads it, what the engine found.

Upstream reads a protein chain's templates through two inputs (stock/src): the chain's `templatesPath` — a HITS file, `.a3m` in the
hmmsearch convention or `.hhr` (docs/infer_json_format.md:55-64), read only under `--use_template true`
(opendde/data/template/template_featurizer.py:292-306) — and the hit's coordinates, `<prot_template_mmcif_dir>/<pdb>.cif`
(template_utils.py:407-427; the directory is `$OPENDDE_ROOT_DIR/search_database/mmcif`, opendde/config/data.py:21-41 — no CLI flag names
it). A hit's pdb id is mapped through `common/obsolete_to_successor.json` first (template_utils.py:685); a cif absent from the directory
is fetched from PDBe (`fetch_remote`, config/data.py:39 — no CLI flag either); a protein chain WITHOUT `templatesPath` makes the engine
SEARCH templates (hmmsearch + the pdb_seqres download, runner/template_search.py:187-241).

This module is the driver's side of that contract, the same for the stock caller and every kit line:
  hits_of(tasks)          the (task, chain, sequence, hits file, pdb ids) rows the query names — `.a3m`: the records whose description
                          carries `mol:protein` (template_parser.py:606; name `<pdb>_<chain>/<start>-<end>`, :641-651); `.hhr`: the `>`
                          alignment headers (HHRParser); the pdb id = the name's first four characters, lowercase (:213-219).
  mmcif_dir_of(...)       the ONE directory the engine reads the hits' cifs from: `--template_mmcif_dir`, else the hits file's directory (or its `mmcif/`)
                          — unique over the templated chains, or a named refusal.
  model_root(...)         the OPENDDE_ROOT_DIR the model process gets: the weights root itself when its `search_database/mmcif` IS that
                          directory, else `<out_dir>/OVERLAY_NAME` — every entry of the weights root by symlink + `search_database/mmcif` ->
                          the cif directory (the weights root is never written).
  census(lines, hits)     the engine's own count, `Found <n> templates for sequence <seq>` (template_featurizer.py:314), beside the number
                          of hits the query names for that sequence — a record and one NOTE line per difference (a cif upstream's parser rejects,
                          a kalign failure, a date or duplicate prefilter, template_utils.py:332-381; chains of one sequence naming different hits
                          files); never an exit code: an input upstream accepts runs as upstream runs it.
The boot conditions (assets, kalign, a hits file on every protein chain, every named cif present) are frozen.problems'.
"""
from __future__ import annotations

import json
import logging
import os
import re

OVERLAY_NAME = "opendde_root"                       # <out_dir>/opendde_root: the model process's OPENDDE_ROOT_DIR when the cif directory is not the weights root's own
MMCIF_REL = os.path.join("search_database", "mmcif")   # opendde/config/data.py:23,40: prot_template_mmcif_dir under the root
OBSOLETE_REL = os.path.join("common", "obsolete_to_successor.json")   # config/data.py:44-46; template_utils.py:685 maps every hit id through it
FOUND_RE = re.compile(r"Found (\d+) templates for sequence (\S+)")    # template_featurizer.py:314, verbatim
SLOTS = 4                                           # the template slots the featurizer assembles per chain (template_featurizer.py:41 max_templates=4; dummies pad the rest)
BORN_WORD = "template_pair=born_rows"               # the row-sharded line's census word for template pair features born as this rank's rows (tp._inputs_word)
LOGGER_NAME = "opendde.data.template.template_featurizer"             # the module that logs the line (get_logger(__name__), :26)


class TemplateInputError(ValueError):
    """The query's template inputs cannot be used as named (one sentence; `pred` refuses on it before the model process starts)."""


def _protein_chains(tasks: list) -> list[tuple]:
    out = []
    for t in tasks or []:
        for i, seq in enumerate(t.get("sequences") or []):
            ch = seq.get("proteinChain") if isinstance(seq, dict) else None
            if isinstance(ch, dict):
                out.append((t.get("name"), i, ch))
    return out


def pdb_ids_of(hits_path: str) -> list[str]:
    """The pdb ids a hits file names, in file order: `.a3m` -> the `mol:protein` records; `.hhr` -> the `>` alignment headers; anything
    else -> TemplateInputError (the engine raises on it: template_featurizer.py:303-304)."""
    with open(hits_path, errors="replace") as fh:
        heads = [l[1:].strip() for l in fh if l.startswith(">")]
    if hits_path.endswith(".a3m"):
        heads = [h for h in heads if "mol:protein" in h]
    elif not hits_path.endswith(".hhr"):
        raise TemplateInputError(f"unsupported template hits format: {hits_path} (upstream reads .a3m or .hhr: template_featurizer.py:297-304)")
    return [h.split()[0][:4].lower() for h in heads if h.split()]


def hits_of(tasks: list) -> list[dict]:
    """One row per protein chain: {task, chain (index into sequences), sequence, path (templatesPath as given | None), pdb_ids}."""
    rows = []
    for name, i, ch in _protein_chains(tasks):
        p = ch.get("templatesPath")
        ids = pdb_ids_of(p) if (p and os.path.isfile(p)) else []
        rows.append({"task": name, "chain": i, "sequence": ch.get("sequence"), "path": p, "pdb_ids": ids})
    return rows


def mmcif_dir_of(rows: list[dict], explicit: str | None) -> str | None:
    """The directory the hits' cifs are read from: ``explicit`` (`--template_mmcif_dir`) when given, else the hits file's own directory —
    `<dir>/<pdb>.cif` beside the hits file — or that directory's `mmcif/` subdirectory when it exists (an upstream-style
    `search_database` tree); the hits files of every templated chain must share the one directory. None when no chain names a hit and
    nothing was given."""
    if explicit:
        return os.path.abspath(explicit)
    dirs = sorted({os.path.dirname(os.path.abspath(r["path"])) for r in rows if r["pdb_ids"]})
    if len(dirs) > 1:
        raise TemplateInputError("the templated chains' hits files live in different directories (" + ", ".join(dirs) + "): name the one mmCIF directory with --template_mmcif_dir")
    if not dirs:
        return None
    sub = os.path.join(dirs[0], "mmcif")
    return sub if os.path.isdir(sub) else dirs[0]


def model_root(weights_root: str, mmcif_dir: str | None, out_dir: str) -> str:
    """The OPENDDE_ROOT_DIR of the model process (module docstring). The overlay is rebuilt on every call; the weights root is only read."""
    if not mmcif_dir or os.path.realpath(os.path.join(weights_root, MMCIF_REL)) == os.path.realpath(mmcif_dir):
        return weights_root
    ov = os.path.join(out_dir, OVERLAY_NAME)
    if os.path.lexists(ov):
        _rm_tree_of_links(ov)
    os.makedirs(os.path.join(ov, "search_database"))
    for entry in sorted(os.listdir(weights_root)):
        if entry != "search_database":
            os.symlink(os.path.join(os.path.abspath(weights_root), entry), os.path.join(ov, entry))
    sd = os.path.join(weights_root, "search_database")
    if os.path.isdir(sd):                                                        # the weights root's other search databases stay visible (only mmcif is the item's)
        for entry in sorted(os.listdir(sd)):
            if entry != "mmcif":
                os.symlink(os.path.join(os.path.abspath(sd), entry), os.path.join(ov, "search_database", entry))
    os.symlink(os.path.abspath(mmcif_dir), os.path.join(ov, MMCIF_REL))
    return ov


def _rm_tree_of_links(path: str) -> None:
    for cur, dirs, files in os.walk(path, topdown=False):
        for f in files:
            os.unlink(os.path.join(cur, f))
        for d in dirs:
            p = os.path.join(cur, d)
            os.unlink(p) if os.path.islink(p) else os.rmdir(p)
    os.rmdir(path)


def obsolete_map(root: str | None) -> dict:
    """Upstream's obsolete -> successor pdb id map as the model process will read it ({} when the file is absent: template_utils.py:831)."""
    p = os.path.join(root, OBSOLETE_REL) if root else None
    if p and os.path.isfile(p):
        try:
            with open(p) as fh:
                m = json.load(fh)
            return m if isinstance(m, dict) else {}
        except (OSError, ValueError):
            return {}
    return {}


def missing_cifs(rows: list[dict], mmcif_dir: str | None, obsolete: dict) -> list[tuple]:
    """(task, chain, pdb id as named, cif path the engine opens) for every named hit whose cif is not in ``mmcif_dir``."""
    out = []
    for r in rows:
        for pid in r["pdb_ids"]:
            eff = str(obsolete.get(pid, obsolete.get(pid.upper(), pid))).lower()
            p = os.path.join(mmcif_dir, f"{eff}.cif") if mmcif_dir else f"<no mmCIF directory>/{eff}.cif"
            if not (mmcif_dir and os.path.isfile(p)):
                out.append((r["task"], r["chain"], pid, p))
    return out


def named(rows: list[dict]) -> dict:
    """{sequence: hits named} over the templated chains (the census key is the sequence: the engine's line names it, not the chain)."""
    out: dict = {}
    for r in rows:
        if r["pdb_ids"]:
            out.setdefault(r["sequence"], set()).add(len(r["pdb_ids"]))
    return out


def census(lines, rows: list[dict], form: str = "dense") -> dict:
    """The engine's `Found <n>` lines against the hits the query names, for the record: {ok, sequences: {seq: {named: [n, …], found: [n, …]}},
    real, slots, form, notes: [sentence, …]}. `ok` = every templated sequence was reported with exactly the count(s) its chains name; a
    difference is a NOTE (upstream's parser / date / duplicate prefilter dropped a hit, or chains of one sequence name different hits files) —
    the input is upstream's to accept, the run's exit code is never the census's. `real` = the template slots holding a real hit (the largest
    `Found <n>` over the templated sequences, capped at `slots` = SLOTS); `form` = how the template PAIR features were made: `dense` (upstream's
    featurizer as shipped, one model process) or `row_born` (census_ranks: rank 0 featurised, every rank bore only its rows)."""
    want = named(rows)
    found = _found(lines)
    notes = []
    rec = {}
    for seq, counts in sorted(want.items()):
        tag = seq if len(seq) <= 24 else seq[:24] + "…"
        got = found.get(seq, [])
        rec[seq] = {"named": sorted(counts), "found": got}
        if not got:
            notes.append(f"sequence {tag}: named {sorted(counts)} templates, the engine reported none (no `Found <n> templates` line)")
        elif sorted(set(got)) != sorted(counts):
            notes.append(f"sequence {tag}: named {sorted(counts)} templates, the engine found {got}")
    real = min(SLOTS, max([n for seq in want for n in found.get(seq, [])] or [0]))
    return {"ok": not notes, "sequences": rec, "real": real, "slots": SLOTS, "form": form, "notes": notes}


def _found(lines) -> dict:
    """{sequence: [n, …]} from the `Found <n> templates for sequence <seq>` lines, in order."""
    found: dict = {}
    for line in lines:
        m = FOUND_RE.search(line)
        if m:
            found.setdefault(m.group(2), []).append(int(m.group(1)))
    return found


def census_ranks(logs: dict, rows: list[dict]) -> dict:
    """The census of a ``--n_gpu P`` pass from the P rank transcripts ``{rank: text}``. Rank 0 alone runs the featurizer for every rank (the
    line's ``data_form=rank0_bcast``): it reads the hits, so its `Found <n>` lines are the counts the census keys on, and a rank other than 0
    that reports `Found <n>` lines ran a featurizer of its own — a NOTE, `ok` False. Every rank bears its own rows of the template pair features
    (ranks > 0 from the template coordinates rank 0 broadcast): `form` = `row_born` when every rank's rowpair census names its template rows as
    born (BORN_WORD), else `unverified` with a NOTE naming the ranks that did not."""
    texts = {int(r): (t or "") for r, t in logs.items()}
    unborn = [r for r, t in sorted(texts.items()) if BORN_WORD not in t]
    out = census(texts.get(0, "").splitlines(), rows, form="row_born" if texts and not unborn else "unverified")
    for r, t in sorted(texts.items()):
        got = _found(t.splitlines())
        if r != 0 and got:
            out["notes"].append(f"rank {r} reported {got}: a rank other than 0 ran the template featurizer (rank 0 alone featurises under --n_gpu P)")
            out["ok"] = False
    if 0 not in texts:
        out["notes"].append("rank 0 left no transcript: no `Found <n> templates` line read"); out["ok"] = False
    if unborn:
        out["notes"].append(f"ranks {unborn} did not report {BORN_WORD} in their rowpair census")
    return out


class FoundCollector(logging.Handler):
    """Collects the engine's `Found <n> templates` records in this process (the kit lines run the stock CLI in-process); attach() puts it
    on the featurizer's own logger, so upstream's root-logger handlers are untouched."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if FOUND_RE.search(msg):
            self.lines.append(msg)

    def attach(self):
        logging.getLogger(LOGGER_NAME).addHandler(self)
        return self

    def detach(self):
        logging.getLogger(LOGGER_NAME).removeHandler(self)
