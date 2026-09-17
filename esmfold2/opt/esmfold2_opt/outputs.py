"""Per-prediction outputs — ONE schema, upstream's own outputs, written by the kit's one writer.

Every arm of the package (``pred`` under a kit mode, ``pred --mode off`` and the served line) ends in the same files, through
``ef2_server.write_outputs`` (the kit's code, imported from the kit directory; never re-implemented):

    <out_dir>/cif_all/<id>__<fast|full>__s<seed>_x<k>.cif      upstream's mmCIF text (``result.complex.to_mmcif()``)
    <out_dir>/cif_all/<id>__<fast|full>__s<seed>_x<k>_pae.npz  upstream's ``pae`` / ``plddt`` as float16, the token features ``asym_id`` / ``mol_type``,
                                                                and upstream's ``pair_chains_iptm`` (float32) when the result carries it
    <out_dir>/<tag>_rows.jsonl                                  one row per prediction: upstream's ``iptm`` / ``ptm``, ``n_tokens``, ``msa_depths``, the file name

Nothing is derived from the confidences (no interface scores, no per-chain summaries); the result's ``pde``, ``distogram`` and output
embeddings are not written. The file names carry the kit server's variant name (``fast`` / ``full``; the package's ``full_msa`` /
``full_nomsa`` axis is the ``variant`` on the ACTIVE line), and the rows are one record per prediction
(``{pred_file, gpu, wall_s, utc, tag, mode}`` are bookkeeping, everything else is compared).

``write_outputs`` needs what the kit's own loop prepares for it (``spi_meta``): the MSA depths and the token features ``asym_id`` /
``mol_type`` / ``token_attention_mask`` from upstream's ``builder.prepare_input`` (``token_meta``). An input is any upstream input — one
chain or many; no chain roles are read.

Stock arm. The stock subprocess has nothing from the kit on its path, so it cannot call the kit's writer: it STAGES each result as upstream
produced it (``stage_result``: the mmCIF text, float32 pae / plddt, the token features, the result attributes the writer reads) and
the ``pred`` process that launched it writes the file set from the staged bytes through the same ``write_outputs``
(``finalise_staged``; a staged result presents the same attributes the live result does). Same function, same bytes, same rows.
"""
from __future__ import annotations

import hashlib                                   # digests computed here, not opt_core.gates.sha256_file: this module also runs in the stock subprocess (stock_fold.py), where the core is absent by design
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from . import modes

CIF_DIR = "cif_all"
ROWS_SUFFIX = "_rows.jsonl"
STAGE_DIRNAME = "staged"
PROGRESS_NAME = "_progress.json"                                          # the stock subprocess's pass record inside the stage directory (stock_fold.Outcome)
RESULT_ATTRS = ("iptm", "ptm", "pair_chains_iptm")                       # the upstream result attributes the kit's writer reads (ef2_server ROW_RESULT_FIELDS + NPZ_RESULT_FIELDS), staged by the stock arm
META_KEYS = ("pred_file", "gpu", "wall_s", "utc", "tag", "mode")        # the bookkeeping keys of a row (vary run to run by construction)


# ---------------------------------------------------------------------------------------------------------------------- the kit's writer
def kit_server(kit_home: Optional[str] = None):
    """The kit's server module (``driver/ef2_server.py``) imported from the kit directory — the one writer. Importing it puts the kit's
    driver directory on ``sys.path`` (the kit's own convention, ``ef2_server.py`` does the same); nothing is configured by the import."""
    import sys
    from . import stack
    for p in stack.kit_sys_path(kit_home):
        if p not in sys.path:
            sys.path.insert(0, p)
    import ef2_server  # noqa: F401  (the kit's module)
    return ef2_server


ROWS_TAG = "pred"                                        # the rows file is <out_dir>/pred_rows.jsonl (the kit writer's <tag>_rows.jsonl with its one tag)


def seed_word(seed) -> str:
    """The seed as the file stem spells it: the integer, or `none` for upstream's unseeded default (seed=None)."""
    return "none" if seed is None else str(int(seed))


def rows_path(out_dir: str, tag: str = ROWS_TAG) -> str:
    return os.path.join(out_dir, f"{tag}{ROWS_SUFFIX}")


def server_variant(variant: str) -> str:
    return modes.SERVER_VARIANT[variant]


def kit_item(item: dict) -> dict:
    """The item as the kit's ``write_outputs`` reads it (the package's id is the kit's complex_id)."""
    return {"complex_id": item["id"]}


def token_meta(feats) -> tuple:
    """``asym_id``, ``mol_type`` (per token, first batch row, trimmed to the token count) and the token count from upstream's
    ``prepare_input`` features — exactly as the kit's loop reads them."""
    asym = np.asarray(feats["asym_id"][0].cpu().numpy()); mol = np.asarray(feats["mol_type"][0].cpu().numpy())
    ntok = int(feats["token_attention_mask"][0].sum().item())
    return asym[:ntok], mol[:ntok], ntok


def spi_meta(depths: List[int], feats) -> tuple:
    """The ``meta`` tuple ``write_outputs`` takes: (depths, asym, mol, ntok)."""
    asym, mol, ntok = token_meta(feats)
    return (list(depths), asym, mol, ntok)


def write_rows(server, out_dir: str, tag: str, item: dict, variant: str, seed: int, res, meta: tuple, wall: float,
               gpu_name: str, mode: Optional[str] = None) -> List[dict]:
    """The kit's ``write_outputs`` on one ``fold()`` result (one or several samples): writes ``cif_all/<stem>.cif`` + ``_pae.npz`` and appends
    the rows to ``<tag>_rows.jsonl`` with the server's own two bookkeeping keys (``mode``, ``tag``) as its loop adds them. Returns the rows."""
    cif_dir = os.path.join(out_dir, CIF_DIR)
    os.makedirs(cif_dir, exist_ok=True)
    rows = server.write_outputs(res, kit_item(item), server_variant(variant), None if seed is None else int(seed), meta, cif_dir, True, gpu_name, float(wall))
    with open(rows_path(out_dir, tag), "a", encoding="utf-8") as fh:
        for row in rows:
            row.update(mode=mode, tag=tag)
            fh.write(json.dumps(row, default=str) + "\n")
    return rows


def written_files(out_dir: str, rows: List[dict]) -> List[str]:
    out = []
    for r in rows:
        if r.get("pred_file"):
            stem = os.path.join(out_dir, CIF_DIR, r["pred_file"])
            out += [stem, stem[:-4] + "_pae.npz"]
    return out


# -------------------------------------------------------------------------------------------------------------- the stock arm's staging
class _Arr:
    """A staged array with the two methods the kit's writer calls on a result tensor (``.numpy()``; ``.detach().cpu()`` pass-through)."""
    def __init__(self, a):
        self._a = np.asarray(a)

    def numpy(self):
        return self._a

    def detach(self):
        return self

    def cpu(self):
        return self


class _StagedComplex:
    def __init__(self, cif_text: str):
        self._cif = cif_text

    def to_mmcif(self) -> str:
        return self._cif


class StagedResult:
    """What upstream's result looked like, rebuilt from the staged bytes: ``pae`` / ``plddt`` (float32), ``complex.to_mmcif()`` and the
    result attributes the kit's writer reads (``RESULT_ATTRS``)."""
    def __init__(self, arrays: dict, attrs: dict, cif_text: str):
        self.pae = _Arr(arrays["pae"]); self.plddt = _Arr(arrays["plddt"])
        self.complex = _StagedComplex(cif_text)
        for k in RESULT_ATTRS:
            v = attrs.get(k)
            setattr(self, k, (np.asarray(v) if isinstance(v, list) else v))


def _attr_value(res, name: str):
    """A result attribute as JSON (scalars as numbers, tensors / arrays staged whole as nested lists)."""
    v = getattr(res, name, None)
    if v is None:
        return None
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    elif hasattr(v, "numpy"):
        v = v.numpy()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    return v


def stage_result(stage_dir: str, item: dict, variant: str, seed: int, res, depths: List[int], feats, wall: float, gpu_name: str) -> List[str]:
    """Stage one ``fold()`` result (one or several samples) for ``finalise_staged``: ``<stem>.cif``, ``<stem>.npz`` (pae, plddt float32;
    asym_id, mol_type), ``<stem>.json`` (attributes, depths, ntok, wall, gpu). Returns the stems."""
    os.makedirs(stage_dir, exist_ok=True)
    asym, mol, ntok = token_meta(feats)
    stems = []
    for k, r in enumerate(res if isinstance(res, list) else [res]):
        stem = f"{item['id']}__{server_variant(variant)}__s{seed_word(seed)}_x{k}"
        pae = r.pae.numpy().astype(np.float32); plddt = r.plddt.numpy().astype(np.float32)
        np.savez(os.path.join(stage_dir, stem + ".npz"), pae=pae, plddt=plddt, asym_id=asym, mol_type=mol)
        with open(os.path.join(stage_dir, stem + ".cif"), "w", encoding="utf-8") as fh:
            fh.write(r.complex.to_mmcif())
        meta = {"item": {"id": item["id"]}, "variant": variant, "seed": None if seed is None else int(seed),
                "sample_index": k, "n_samples": len(res) if isinstance(res, list) else 1, "depths": list(depths), "ntok": ntok,
                "wall_s": float(wall), "gpu": gpu_name, "attrs": {a: _attr_value(r, a) for a in RESULT_ATTRS}}
        with open(os.path.join(stage_dir, stem + ".json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        stems.append(stem)
    return stems


def finalise_staged(server, stage_dir: str, out_dir: str, tag: str, mode: Optional[str] = "off") -> List[dict]:
    """The stock arm's second half (in the ``pred`` process): every staged ``fold()`` result -> the kit's ``write_outputs``, grouped by
    (item, seed) so a multi-sample fold becomes one call with the samples in order. A group with fewer samples than its fold produced
    (the subprocess ended while staging it) is left out. Returns all rows; the stage directory is removed."""
    import shutil
    metas = {}
    for fn in sorted(os.listdir(stage_dir)):
        if fn.endswith(".json") and fn != PROGRESS_NAME:
            with open(os.path.join(stage_dir, fn), encoding="utf-8") as fh:
                metas[fn[:-5]] = json.load(fh)
    groups: Dict[tuple, List[str]] = {}
    for stem, m in metas.items():
        groups.setdefault((m["item"]["id"], m["seed"]), []).append(stem)
    rows: List[dict] = []
    for (item_id, seed), stems in groups.items():
        stems.sort(key=lambda s: metas[s]["sample_index"])
        m0 = metas[stems[0]]
        if len(stems) != int(m0["n_samples"]):
            continue
        results = []
        for stem in stems:
            with np.load(os.path.join(stage_dir, stem + ".npz")) as z:
                arrays = {k: z[k] for k in z.files}
            with open(os.path.join(stage_dir, stem + ".cif"), encoding="utf-8") as fh:
                cif_text = fh.read()
            results.append(StagedResult(arrays, metas[stem]["attrs"], cif_text))
        with np.load(os.path.join(stage_dir, stems[0] + ".npz")) as z:
            asym, mol = z["asym_id"], z["mol_type"]
        meta = (list(m0["depths"]), asym, mol, int(m0["ntok"]))
        item = {"id": item_id}
        res: Any = results if m0["n_samples"] > 1 else results[0]
        rows += write_rows(server, out_dir, tag, item, m0["variant"], seed, res, meta, m0["wall_s"], m0["gpu"], mode=mode)
    shutil.rmtree(stage_dir, ignore_errors=True)
    return rows


def listing(out_dir: str, tag: str = "pred") -> Dict[str, str]:
    """The file set of an output directory as ``{relative path: sha256}`` plus, per row, the sha256 of its comparable fields
    (every key but ``META_KEYS``) — what two arms are compared on."""
    out: Dict[str, str] = {}
    cif_dir = os.path.join(out_dir, CIF_DIR)
    if os.path.isdir(cif_dir):
        for fn in sorted(os.listdir(cif_dir)):
            with open(os.path.join(cif_dir, fn), "rb") as fh:
                out[os.path.join(CIF_DIR, fn)] = hashlib.sha256(fh.read()).hexdigest()
    rp = rows_path(out_dir, tag)
    if os.path.isfile(rp):
        with open(rp, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                key = f"row:{row.get('complex_id')}:{row.get('seed')}:{row.get('sample_index')}"
                out[key] = hashlib.sha256(json.dumps({k: v for k, v in row.items() if k not in META_KEYS}, sort_keys=True).encode()).hexdigest()
    return out
