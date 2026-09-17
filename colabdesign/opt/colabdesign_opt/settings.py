"""The design's settings files, BindCraft's own: `--advanced FILE` (bindcraft.py's `--advanced`; default its
`settings_advanced/default_4stage_multimer.json`, vendored under stock/src/bindcraft/ and named in stock/PINS.json `settings`) and
`--filters FILE` (bindcraft.py's `--filters`; default its `settings_filters/default_filters.json`, vendored beside it; the design step only
names its columns in the failure census), READ AT RUN TIME. No value of either is typed anywhere in this kit: `load()` returns the advanced
file's own dict after checking the file exists (the default also at the path stock/PINS.json declares — a pin naming another path is refused
by name, never read), and the driver (bindcraft.py) hands that dict to BindCraft's own code. The target settings BindCraft takes from its
target json — starting PDB, chains, binder length, hotspots — are the case's (`design --starting-pdb/--chains/--binder-len/--target-hotspot-residues`).
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .names import BINDCRAFT_DIR, sha256_file

NAME = "default_4stage_multimer"                                                  # the default settings' name: the file's stem (bindcraft.py's default --advanced)
SETTINGS_RELPATH = os.path.join(BINDCRAFT_DIR, "settings_advanced", NAME + ".json")   # tree-relative
FILTERS_RELPATH = os.path.join(BINDCRAFT_DIR, "settings_filters", "default_filters.json")   # bindcraft.py's default --filters
PIN_KEY = "settings"                                                              # stock/PINS.json: {"file", ...}


class SettingsError(ValueError):
    """A settings file is absent, or stock/PINS.json names another path for the default, or the advanced file lacks a key BindCraft's design
    step reads: refused by name."""


def path(tree: str) -> str:
    return os.path.join(tree, SETTINGS_RELPATH)


def filters_path(tree: str) -> str:
    return os.path.join(tree, FILTERS_RELPATH)


def pinned_relpath(tree: str) -> str:
    """The tree-relative settings path stock/PINS.json declares (`settings.file`)."""
    with open(os.path.join(tree, "stock", "PINS.json"), "r", encoding="utf-8") as fh:
        pins = json.load(fh)
    rec = pins.get(PIN_KEY) or {}
    if not rec.get("file"):
        raise SettingsError(f"stock/PINS.json has no {PIN_KEY}.file")
    return rec["file"]


def load(tree: str, advanced: Optional[str] = None, filters: Optional[str] = None) -> dict:
    """{"name" (the file's stem), "file" (the default: tree-relative; a given file: absolute), "path", "sha256" (observed and recorded;
    compared against nothing), "advanced": <the json's own dict>, "filters": <the filters file's path>}. `advanced` / `filters` are
    bindcraft.py's `--advanced` / `--filters` files; None = its defaults in the vendored tree (the default advanced file is checked present at
    the path stock/PINS.json declares). A file that does not exist is refused by name."""
    if advanced is None:
        p, name, label = path(tree), NAME, SETTINGS_RELPATH
        if not os.path.isfile(p):
            raise SettingsError(f"pinned settings file absent: {SETTINGS_RELPATH} under {tree}")
        declared = pinned_relpath(tree)
        if os.path.normpath(declared) != os.path.normpath(SETTINGS_RELPATH):
            raise SettingsError(f"stock/PINS.json {PIN_KEY}.file = {declared!r}, expected {SETTINGS_RELPATH!r}")
    else:
        p = os.path.abspath(advanced); name, label = os.path.splitext(os.path.basename(p))[0], p
        if not os.path.isfile(p):
            raise SettingsError(f"--advanced {advanced}: no such file")
    f = filters_path(tree) if filters is None else os.path.abspath(filters)
    if filters is not None and not os.path.isfile(f):
        raise SettingsError(f"--filters {filters}: no such file")
    with open(p, "r", encoding="utf-8") as fh:
        adv = json.load(fh)
    return {"name": name, "file": label, "path": p, "sha256": sha256_file(p), "advanced": adv, "filters": f}


def record(s: dict) -> dict:
    """The settings block of the run record and the SETTINGS line: the name, the file, its observed sha256 (evidence — the values are the
    file's; a reader opens it)."""
    return {"name": s["name"], "file": s["file"], "sha256": s["sha256"]}


def iteration_counts(advanced: dict) -> dict:
    """The four phases' iteration counts as the file states them (soft / temp / hard = graded steps; greedy = forward-only rounds) — read,
    for the RUN line's step estimate and the census; BindCraft's own loop decides what actually runs (its pLDDT gates, optimise_beta)."""
    try:
        return {"soft": int(advanced["soft_iterations"]), "temp": int(advanced["temporary_iterations"]), "hard": int(advanced["hard_iterations"]),
                "greedy": int(advanced["greedy_iterations"])}
    except KeyError as e:
        raise SettingsError(f"settings key {e} absent from the advanced settings file") from None
