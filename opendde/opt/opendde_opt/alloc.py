"""The CUDA allocator policy of a kit process, through the core (``opt_core.mem.torch_alloc``): a big line's fixed policy
(``Line.allocator``: expandable segments, exported at activation under the core's named refusals — another value present, CUDA already
initialised) and the S-hook lines' ``alloc_auto`` lever.

``alloc_auto`` (lever on ``exact``/``fast``, ``pred`` route): the expandable-segments policy is exported for the kit process of a `pred`
call once its query JSON is read; a route that reads no query (``check``, the hook alone) keeps the default allocator by name. Placement
only: no kernel input changes, outputs unchanged by construction. The decision and its reason are the
lever's evidence (``kit_stats.alloc``: policy, decision, reason, n_items, and after CUDA work the allocator's own read-back ``effective``).

``token_floor`` is the query's per-item lower bound of the token count (one token per polymer residue of every chain copy; ligands, ions
and modified residues, tokenised per atom by the featurizer, add tokens and are not counted) — the fact the size-gated levers' engagement
predicates read (``ran.py``).
"""
from __future__ import annotations

from opt_core.mem import MemLeverRefused, torch_alloc

from . import ActivationError, modes

POLICY_OF = {modes.EXPANDABLE: "expandable"}                # the kit's Line.allocator value -> the core's policy word
STATE = {"decision": None}                                  # the one decision of this process (kit_stats.alloc)

_RESIDUE_KEYS = ("proteinChain", "dnaSequence", "rnaSequence")


def token_floor(jobs) -> dict:
    """Per query item, a lower bound of its token count: sum over polymer chains of len(sequence) × count. ``{name: floor}``."""
    out = {}
    for j in jobs or []:
        n = 0
        for ent in j.get("sequences") or []:
            for key in _RESIDUE_KEYS:
                if key in ent and isinstance(ent[key], dict):
                    seq = ent[key].get("sequence") or ""
                    n += len(seq) * int(ent[key].get("count") or 1)
        out[str(j.get("name"))] = n
    return out


def decide_auto(jobs) -> dict:
    """The alloc_auto decision for a `pred` call: export once the call's query items are read; no item read keeps the default allocator by name."""
    n_items = len(jobs or [])
    facts = {"policy": "expandable", "n_items": n_items}
    if not n_items:
        return {**facts, "decision": "default", "reason": "no query item read: default allocator"}
    return {**facts, "decision": "export", "reason": f"the pred call's query is read ({n_items} item{'s' if n_items != 1 else ''}): expandable segments"}


def apply(res: modes.Resolution, jobs=None) -> dict | None:
    """Export the process's allocator policy through the core, or record why not. Big lines: the line's fixed policy (refusal = the line is
    NOT active: ActivationError). S-hook lines with ``alloc_auto`` on the `pred` route: the decision above. Returns the facts."""
    if res.line is None:
        return None
    if res.line.allocator:                                                     # a memory line's fixed policy
        policy = POLICY_OF.get(res.line.allocator)
        if policy is None:
            raise ActivationError(f"line {res.line.name} names allocator {res.line.allocator!r}: no core policy for it")
        try:
            facts = torch_alloc.export(policy, lever=f"{res.line.name}.allocator")
        except MemLeverRefused as e:
            raise ActivationError(f"line {res.line.name}: {e}") from None
        STATE["decision"] = {"lever": "allocator", "decision": "export", **facts}
        return STATE["decision"]
    if "alloc_auto" not in res.line.levers:
        return None
    if jobs is None:
        STATE["decision"] = {"lever": "alloc_auto", "policy": "expandable", "decision": "default", "reason": "no query given: default allocator"}
        return STATE["decision"]
    d = decide_auto(jobs)
    if d["decision"] == "export":
        try:
            d.update(torch_alloc.export(d["policy"], lever="alloc_auto"))
        except MemLeverRefused as e:                                          # another value present / CUDA initialised: named, the lever is skipped (not silent)
            d["decision"], d["reason"] = "refused", str(e)
    STATE["decision"] = {"lever": "alloc_auto", **d}
    return STATE["decision"]


def kit_stats() -> dict | None:
    """The lever's block of the exit tally / manifest: the decision facts plus the allocator's own read-back once CUDA ran."""
    d = STATE["decision"]
    if d is None:
        return None
    out = dict(d)
    try:
        eff = torch_alloc.effective()
        out["effective"], out["effective_source"] = eff.get("expandable"), eff.get("source")
    except Exception as e:  # noqa: BLE001
        out["effective"], out["effective_source"] = None, f"unreadable:{type(e).__name__}"
    return out


def fallbacks(planned) -> list:
    """A refused export (the environment named another configuration, or CUDA was up before activation) is a named event; a `default`
    decision is the lever's declared envelope, not a fallback. An `export` whose read-back says not expandable after CUDA work is named too."""
    d = STATE["decision"]
    if d is None or "alloc_auto" not in (planned or ()):
        return []
    if d.get("decision") == "refused":
        return [f"alloc_auto: {d.get('reason')}"]
    if d.get("decision") == "export":
        try:
            eff = torch_alloc.effective().get("expandable")
        except Exception:  # noqa: BLE001
            eff = None
        if eff is False:
            src = torch_alloc.effective().get("source")
            return [f"alloc_auto: exported expandable_segments but the allocator's record reads not expandable after CUDA work (source {src})"]
    return []
