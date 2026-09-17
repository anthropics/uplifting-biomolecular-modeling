"""The strategy catalogue: ``STRATEGIES.json`` beside this module — the canonical strategy ids a LEVER line's ``strategy=`` field may name
(``{"canonical": [{"id", "family", "numerics_changing"}], "aliases": {alias: canonical id}}``) — and :func:`check`, the membership rule a
kit's catalogue test states its lever ids against. A development-time table: no run path reads it (the LEVER line itself checks the id's
FORM, :func:`opt_core.report.strategy_form`). Standard library at import.
"""
from __future__ import annotations

import json
import os

from .report import strategy_form

__all__ = ["FILE", "table", "check", "strategies", "check_strategy"]

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "STRATEGIES.json")
_TABLE = None


def table() -> dict:
    """``{"canonical": {id: row}, "aliases": {alias: replaced_by}}``, read once per process."""
    global _TABLE
    if _TABLE is None:
        with open(FILE, encoding="utf-8") as fh:
            doc = json.load(fh)
        _TABLE = {"canonical": {r["id"]: r for r in doc.get("canonical", [])}, "aliases": dict(doc.get("aliases", {}))}
    return _TABLE


def check(sid: str) -> str:
    """``sid`` itself when it is a canonical strategy id of the catalogue (family ids and the shared ``LOCAL.<name>`` ids it lists) or a
    kit-local ``LOCAL.<kit>.<name>`` (outside the catalogue by definition); a ValueError naming the canonical id for an alias, or naming the
    fact for an unknown id."""
    sid = strategy_form(str(sid))
    if sid.startswith("LOCAL.") and len([p for p in sid.split(".") if p]) >= 3:
        return sid
    t = table()
    if sid in t["canonical"]:
        return sid
    if sid in t["aliases"]:
        raise ValueError(f"strategy {sid!r} is an alias; the canonical id is {t['aliases'][sid]!r}")
    raise ValueError(f"strategy {sid!r} is not in the catalogue (STRATEGIES.json canonical ids) and not a LOCAL.<kit>.<name> id")


# --- the two names a kit's catalogue test may use for the same table (they lived in ``opt_core.report`` as a
# compatibility face; no run path calls them).
def strategies() -> dict:
    """``{"canonical": {id: row}, "aliases": {alias: replaced_by}}`` — :func:`table`."""
    return table()


def check_strategy(sid: str) -> str:
    """``sid`` itself when it is a canonical strategy id or a kit-local ``LOCAL.<kit>.<name>``; a ValueError otherwise —
    :func:`check`."""
    return check(sid)
