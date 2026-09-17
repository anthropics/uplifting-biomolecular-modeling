"""The target of a generation run: ``--input`` is ONE target entry file (JSON, or YAML when PyYAML is importable) holding exactly one
top-level key — the item name — mapped to an entry in upstream's own ``configs/targets/targets_dict.yaml`` form::

    {"1jz7A_N200": {"source": "custom", "target_filename": "1jz7A_N200",     # inert when target_path is set, but interpolated by upstream's config:
                                                                              # binder_generate.yaml's oc.select fallback reads both (a shorter entry raises there)
                    "target_path": "1jz7A_N200/target.pdb",                # relative to the entry file's directory (or absolute); upstream receives the ABSOLUTE path
                    "target_input": "A544-663",                            # the crop: chain + residue range in the file's own numbering (atomworks contig)
                    "hotspot_residues": ["A590", "A602"],                   # <chain><resnum>; consumed by the binder task's TargetFeatures
                    "binder_length": [80, 80],                              # [low, high] — equal: a fixed binder length, no RNG draw (gen_dataset UniformInt)
                    "pdb_id": null}}

The entry is a convenience for naming a target that is not in upstream's table; it is OPTIONAL — without ``--input`` the target is whatever
the Hydra overrides after ``--`` (``++generation.task_name=<item>``, upstream's own ``configs/targets/targets_dict.yaml`` or a
``++generation.target_dict_cfg={…}`` merge) and the shipped configuration name. ``target_path`` is resolved against the entry file's own
directory when relative (a mounted input set carries ``<bin>/<item>.json`` beside ``<bin>/<item>/target.pdb``) and handed to upstream
ABSOLUTE (upstream opens it from its own working directory, the run directory). The entry reaches upstream as two Hydra tokens,
``++generation.task_name=<item>`` and ONE dictionary merge ``++generation.target_dict_cfg={<item>:{…}}`` (``overrides``, which says why) —
no file of upstream's is edited, and whatever keys the entry carries are the keys upstream receives (a ligand or motif entry keeps its own
keys). Values are rendered in Hydra's override grammar (strings quoted with escapes, lists bracketed, null); what upstream's configuration
makes of them is upstream's to say — the package checks the FILE's shape (one item, a mapping), never the values.
"""
from __future__ import annotations

import json
import os
from typing import List, Tuple

KEY_ORDER = ("source", "target_filename", "target_path", "target_input", "hotspot_residues", "binder_length", "pdb_id")   # targets_dict.yaml's own order first; other keys follow in the entry's order
TARGET_CFG = "generation.target_dict_cfg"
TASK_KEY = "generation.task_name"


class InputError(ValueError):
    """An entry file that is not one item mapped to a mapping (the file's shape; values are never judged here)."""


def _load(path: str) -> dict:
    if not os.path.isfile(path):
        raise InputError(f"--input {path}: no such file")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml                                        # PyYAML: present on the pinned stack (hydra's dependency); JSON needs nothing
        except ImportError:
            raise InputError(f"--input {path}: a YAML entry needs PyYAML in this interpreter; give the entry as .json") from None
        try:
            return yaml.safe_load(text)
        except Exception as e:
            raise InputError(f"--input {path}: not valid YAML ({e})") from None
    try:
        return json.loads(text)
    except ValueError as e:
        raise InputError(f"--input {path}: not valid JSON ({e})") from None


def load_entry(path: str) -> Tuple[str, dict]:
    """(item, entry) of the entry file at ``path``: exactly one item name mapped to a mapping; a relative ``target_path`` made absolute
    against the file's directory. Nothing else about the entry is checked."""
    doc = _load(os.path.abspath(path))
    if isinstance(doc, dict) and set(doc) == {"target_dict_cfg"} and isinstance(doc["target_dict_cfg"], dict):
        doc = doc["target_dict_cfg"]                              # upstream's file form (top key target_dict_cfg:) with one entry is accepted too
    if not isinstance(doc, dict) or len(doc) != 1:
        raise InputError(f"--input {path}: exactly one top-level key (the item name) is expected, found {sorted(map(str, doc)) if isinstance(doc, dict) else type(doc).__name__}")
    (item, entry), = doc.items()
    item = str(item)
    if not item.strip():
        raise InputError(f"--input {path}: the item name is empty")
    if not isinstance(entry, dict):
        raise InputError(f"--input {path}: the item's value is not a mapping of target entry keys")
    tp = entry.get("target_path")
    if isinstance(tp, str) and tp and not os.path.isabs(tp):     # relative to the entry file's directory; upstream receives the absolute path
        entry = dict(entry, target_path=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), tp)))
    return item, dict(entry)


def render(value) -> str:
    """One value in Hydra's override grammar: str -> 'quoted' (backslash and quote escaped), bool -> true|false, int/float -> literal,
    None -> null, list -> [a,b], dict -> {k:v}."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(render(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{render(v)}" for k, v in value.items()) + "}"
    raise InputError(f"value {value!r} of type {type(value).__name__} has no Hydra rendering")


def ordered_keys(entry: dict) -> List[str]:
    return [k for k in KEY_ORDER if k in entry] + [k for k in entry if k not in KEY_ORDER]


def overrides(item: str, entry: dict) -> List[str]:
    """The two Hydra tokens that hand the entry to upstream: ``++generation.task_name=<item>`` and ONE dictionary merge
    ``++generation.target_dict_cfg={<item>:{source:'…',target_filename:'…',target_path:'…',target_input:'…',hotspot_residues:[…],binder_length:[L,L],pdb_id:null}}``
    (targets_dict key order first). Hydra applies a dictionary value with ``OmegaConf.update(merge=True)``: the item is added to upstream's own
    ``target_dict_cfg`` table, nothing else in it changes. The dotted per-key form ``target_dict_cfg.<item>.<key>=…`` is NOT used: Hydra's KEY
    grammar cannot spell a path element that starts with a digit (``1gpbA_N200`` lexes as INT + ID -> "extraneous input … expecting EQUAL"),
    while a dictionary KEY inside a VALUE may (dictKey = (INT|ID|…)+, read back as the string)."""
    body = ",".join(f"{k}:{render(entry[k])}" for k in ordered_keys(entry))
    return [f"++{TASK_KEY}={item}", f"++{TARGET_CFG}={{{item}:{{{body}}}}}"]


def binder_length(entry: dict):
    """The fixed binder length (``binder_length`` [L, L]; the high end of a range), or None when the entry carries none."""
    bl = entry.get("binder_length")
    if isinstance(bl, (list, tuple)) and bl:
        return bl[-1]
    return bl if isinstance(bl, int) else None
