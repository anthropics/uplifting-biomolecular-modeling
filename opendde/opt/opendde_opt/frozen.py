"""frozen — the frozen-weights boot gate of `pred`: a run never reaches a download. The stock's network fallbacks
(`stock/src`, every site) and what closes each:

  checkpoint   opendde/utils/download.py:283-316 `download_inference_cache` (called inside `InferenceRunner.__init__`): the released
               checkpoint is fetched (a cross-checked "managed asset", :254-280) when no `--load_checkpoint_path` was given; given + absent =
               FileNotFoundError (:309-313, a refusal), given + present = used as is. The package always passes
               `--load_checkpoint_path <root>/checkpoint/opendde.pt` (settings.cli_args) — only when the root is known: this gate refuses an
               unset root and an absent checkpoint before the stock starts.
  CCD caches   download.py:290-294 + :254-280: `common/components.cif` + `common/components.cif.rdkit_mol.pkl` under DATA_ROOT (= OPENDDE_ROOT_DIR,
               opendde/config/data.py:21-22, :48-51) are fetched when absent OR when present with a byte size other than upstream's manifest
               (opendde/config/dependency_url.py:36-57; stock/PINS.json "common".managed_assets) — unconditional, no switch: this gate
               refuses their absence, and their size when the tree is given (`problems(..., tree=)`), by name.
  templates    under `--use_template true` four sites: download.py:296-304 fetches
               `common/release_date_cache.json` + `common/obsolete_to_successor.json` when absent or of another size (same rule as the CCD
               caches) — refused by name; runner/template_search.py:187-241 SEARCHES templates (hmmsearch + the pdb_seqres download,
               :107-120) for every protein chain without a `templatesPath` file — refused by name (task, chain index); template_utils.py
               :407-427 fetches a hit's mmCIF from PDBe when `<root>/search_database/mmcif/<pdb>.cif` is absent (`fetch_remote`, no CLI
               flag; the id mapped through obsolete_to_successor.json first, :685) — every cif a hits file names must be in that directory
               of the model process's root (templates.model_root), refused by name otherwise; and runner/batch_inference.py:494-497
               resolves the kalign binary (`--kalign_binary_path`, else PATH) and raises without it — refused by name.
  protein MSA  runner/msa_search.py:181-225 `update_infer_json` -> the MMseqs2 service (opendde/data/msa/msa_service_client.py:28-30
               MMSEQS_SERVICE_HOST_URL, upstream's default public host) for every task whose protein chain carries neither
               `pairedMsaPath`/`unpairedMsaPath` nor a `msa.precomputed_msa_dir` (`need_msa_search` :13-48): under `--use_msa true` this
               gate refuses an input with such a chain by name (task, chain index).
  RNA MSA      runner/rna_msa_search.py:224-265 (three database downloads): under `--use_rna_msa true` only — refused here.

The gate is a list of named problems; `pred` prints them on one REFUSED line and exits without starting the stock (EXIT_NOT_ACTIVE).
"""
from __future__ import annotations

import json
import os

REQUIRED_FILES = ("checkpoint/opendde.pt", "common/components.cif", "common/components.cif.rdkit_mol.pkl")   # under OPENDDE_ROOT_DIR
TEMPLATE_FILES = ("common/release_date_cache.json", "common/obsolete_to_successor.json")                        # + these under --use_template true
SITES = {"checkpoint/opendde.pt": "stock downloads it: opendde/utils/download.py:283-316",
         "common/components.cif": "stock downloads it: opendde/utils/download.py:254-294",
         "common/components.cif.rdkit_mol.pkl": "stock downloads it: opendde/utils/download.py:254-294",
         "common/release_date_cache.json": "stock downloads it under --use_template true: opendde/utils/download.py:296-304",
         "common/obsolete_to_successor.json": "stock downloads it under --use_template true: opendde/utils/download.py:296-304"}


def managed_sizes(tree: str | None) -> dict:
    """{relative path under the root: byte size} of the files upstream size-checks at runner start (stock/PINS.json "common".managed_assets
    — upstream's own MANAGED_ASSETS identities); {} when no tree is given."""
    if not tree:
        return {}
    with open(os.path.join(tree, "stock", "PINS.json")) as fh:
        common = json.load(fh).get("common", {})
    return {f"common/{name}": int(v["bytes"]) for name, v in (common.get("managed_assets") or {}).items()}


TRUE_WORDS = ("1", "true", "t", "yes", "y", "on")           # click.BOOL's true spellings (case-insensitive): the stock CLI parses the flag, so the gate compares the parsed value


def _truthy(v) -> bool:
    """The boolean the stock CLI (click BOOL) reads from a flag's text; None/absent = False."""
    return v is not None and str(v).strip().lower() in TRUE_WORDS


def _flag(name: str, flags: dict, extra: list[str]) -> str:
    """The effective value of a stock boolean flag: ``flags`` (settings.effective: stated, else upstream's default), overridden by a trailing `--<name> <v>` in the pass-through args."""
    v = str(flags.get(name, "false")).lower()
    for i, tok in enumerate(extra):
        if tok == f"--{name}" and i + 1 < len(extra):
            v = str(extra[i + 1]).lower()
        elif tok.startswith(f"--{name}="):
            v = tok.split("=", 1)[1].lower()
    return v


def _chain_has_msa(chain: dict) -> bool:
    if chain.get("pairedMsaPath") or chain.get("unpairedMsaPath"):
        return True
    msa = chain.get("msa")
    return isinstance(msa, dict) and bool(msa.get("precomputed_msa_dir"))


def problems(root: str | None, input_path: str, flags: dict, extra: list[str], tree: str | None = None) -> list[str]:
    """Every named reason this `pred` would reach a stock download (empty = the run is frozen). With ``tree`` the CCD files' byte sizes are
    checked against upstream's manifest too (a present file of another size is replaced by a download upstream)."""
    out: list[str] = []
    use_template = _truthy(_flag("use_template", flags, extra))
    required = REQUIRED_FILES + (TEMPLATE_FILES if use_template else ())
    if not root:
        out.append("weights root absent: OPENDDE_ROOT_DIR is not set — see README Variables (the stock resolves its default root and downloads the checkpoint: opendde/utils/download.py:283-316)")
    else:
        for rel in required:
            if not os.path.isfile(os.path.join(root, rel)):
                out.append(f"{rel} absent under {root} ({SITES[rel]})")
        for rel, want in managed_sizes(tree).items():
            p = os.path.join(root, rel)
            if rel in required and os.path.isfile(p) and os.path.getsize(p) != want:
                out.append(f"{rel} under {root} is {os.path.getsize(p)} bytes, upstream's manifest says {want}: the stock replaces it by a download (opendde/utils/download.py:254-280)")
    if use_template:
        out += template_problems(root, input_path, extra)
    if _truthy(_flag("use_rna_msa", flags, extra)):
        out.append("use_rna_msa=true: the RNA MSA databases are downloads (runner/rna_msa_search.py:224-265)")
    if _truthy(_flag("use_msa", flags, extra)):
        try:
            tasks = json.load(open(input_path))
        except (OSError, ValueError) as e:
            return out + [f"input unreadable for the MSA check: {e!r}"]
        for t in (tasks if isinstance(tasks, list) else [tasks]):
            for i, seq in enumerate(t.get("sequences") or []):
                ch = seq.get("proteinChain") if isinstance(seq, dict) else None
                if ch is not None and not _chain_has_msa(ch):
                    out.append(f"task {t.get('name')!r} sequences[{i}]: proteinChain without pairedMsaPath/unpairedMsaPath/msa.precomputed_msa_dir under use_msa=true (the stock searches through the MMseqs2 service: runner/msa_search.py:181-225, msa_service_client.py:28-30)")
    return out


def _value(name: str, extra: list[str]) -> str | None:
    """The last `--<name> <v>` / `--<name>=<v>` in the pass-through args (None when absent)."""
    v = None
    for i, tok in enumerate(extra):
        if tok == f"--{name}" and i + 1 < len(extra):
            v = extra[i + 1]
        elif tok.startswith(f"--{name}="):
            v = tok.split("=", 1)[1]
    return v


def template_problems(root: str | None, input_path: str, extra: list[str]) -> list[str]:
    """Under `--use_template true`: every named reason the model process would search, download, fetch or raise for templates (module
    docstring, `templates`). ``root`` is the model process's OPENDDE_ROOT_DIR (templates.model_root): the cifs are checked where the engine
    opens them, `<root>/search_database/mmcif`."""
    import shutil
    from . import inputs as _inputs, templates as _templates
    out: list[str] = []
    kb = _value("kalign_binary_path", extra)
    if not (shutil.which(kb) if kb else None) and not shutil.which("kalign"):
        out.append(f"kalign absent: neither --kalign_binary_path ({kb!r}) nor `kalign` on PATH is executable (runner/batch_inference.py:494-497 raises at runner start)")
    try:
        rows = _templates.hits_of(_inputs.load_query(input_path))
    except (OSError, ValueError) as e:                                       # TemplateInputError is a ValueError: an unsupported hits format is named here
        return out + [f"template inputs unreadable: {e}"]
    for r in rows:
        if not r["path"]:
            out.append(f"task {r['task']!r} sequences[{r['chain']}]: proteinChain without templatesPath under use_template=true (the stock SEARCHES templates: hmmsearch + the pdb_seqres download, runner/template_search.py:187-241; a chain with no template names a hits file holding only its query)")
        elif not os.path.isfile(r["path"]):
            out.append(f"task {r['task']!r} sequences[{r['chain']}]: templatesPath {r['path']} absent (the stock SEARCHES templates for it: runner/template_search.py:187-241)")
    mmcif = os.path.join(root, _templates.MMCIF_REL) if root else None
    for task, chain, pid, path in _templates.missing_cifs(rows, mmcif, _templates.obsolete_map(root)):
        out.append(f"task {task!r} sequences[{chain}]: template {pid}: {path} absent (the stock fetches it from PDBe: template_utils.py:407-427)")
    return out
