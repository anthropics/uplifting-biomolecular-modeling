"""The card gate — the one place GPU 0 is compared with the pinned card (``stock/PINS.json`` ``pinned_stack.gpu``: the name and the
``nvidia-smi`` ``memory.total``, 81559 MiB), before any pass. The comparison words the card and exits on nothing: the pinned MiB and a
name that contains the expected name (or is contained in it: ``H100`` names ``NVIDIA H100 80GB HBM3``) = ``pinned`` (silent); any other
reading = ``untested``; no reading at all (no ``nvidia-smi``, no card) = ``unread``. Off the pinned card the GPU verbs (``design | warm |
check``, cli.hardware_facts) print ONE ``[ef2inv-opt] NOTE hardware … — <word> card: memory limits may differ; proceeding``
line (``note``) and run unchanged, exit code unchanged; ``design`` records the line in ``opt_manifest.json`` ``stack.hardware_gate.note``.
Another card runs unchanged and its memory limits are its own (a CUDA
out-of-memory arm exits by name, exit 4; no route changes by card). A box
with no CUDA device is refused by name where torch is asked (``design``: ``no CUDA device``; the arm process:
stock_design), never by this comparison. ``card()`` reads ``nvidia-smi --query-gpu=name,memory.total``; torch's ``total_memory`` (81079 MiB on the pinned
card — a different accounting) is recorded beside it by the design script and never compared. ``EF2INV_GPU`` / ``EF2INV_GPU_MIB``
(``configs/h100.env``; ``MODEL_OPT_TARGET_GPU`` is honoured as the name) set the expected pair in place of the pin's; ``EF2INV_GPU_MIB``
may name several readings ``|``-separated — ``configs/a100.env``: ``81920|40960``, the A100's 80 GB and 40 GB parts, one card class of one
compute capability whose memory-dependent choices the kit sizes from the probed device, never from the config."""
from __future__ import annotations

import os
from typing import Optional, Tuple

from . import report as R

WORDS = ("pinned", "untested", "unread")
CONSEQUENCE = "memory limits may differ"


def card(probe=None) -> Tuple[Optional[str], Optional[int], str]:
    """(name, nvidia-smi memory.total MiB, the raw 'name, MiB' reading) of GPU 0 — the shared core's torch-free probe
    (opt_core.gates.nvidia_smi_probe; ``probe`` injects its dict in tests); (None, None, why) without a card."""
    if probe is None:
        try:
            from opt_core import gates as G
        except Exception as e:  # noqa: BLE001 — the core gate refuses an unimportable core by name before any verb reaches the card
            return None, None, f"opt_core not importable ({type(e).__name__})"
        probe = G.nvidia_smi_probe(keys=("name", "memory_mib", "probe"))
    d = probe
    name, mib = d.get("name"), d.get("memory_mib")
    if name is None:
        return None, None, str(d.get("probe") or "no GPU reading")
    return name, (int(mib) if mib is not None else None), f"{name}, {mib}"


def expected(pins: dict) -> Tuple[str, Tuple[int, ...]]:
    """(name, the accepted nvidia-smi memory.total readings) — ``EF2INV_GPU_MIB`` ``|``-separated, else the pin's one reading."""
    g = (pins.get("pinned_stack") or {}).get("gpu") or {}
    name = os.environ.get("EF2INV_GPU") or os.environ.get("MODEL_OPT_TARGET_GPU") or g.get("name") or "NVIDIA H100 80GB HBM3"
    raw = os.environ.get("EF2INV_GPU_MIB") or str(g.get("nvidia_smi_memory_total_mib") or 81559)
    mibs = tuple(int(x) for x in raw.split("|") if x.strip())
    return name, mibs


def gate(pins: dict, probe=None) -> dict:
    """The comparison: ``ok`` iff the card's MiB is one of the expected MiB readings and its name contains the expected name or is contained in it,
    case-insensitively (``H100`` and ``NVIDIA H100 80GB HBM3`` name each other; ``NVIDIA H100 PCIe`` does not name the latter); ``reason``
    names the mismatch ('' on the pinned card). Nothing here exits — ``word`` / ``note`` word the result and the verbs proceed."""
    name, mib, raw = card(probe)
    want_name, want_mibs = expected(pins)
    ok = name is not None and mib in want_mibs and (want_name.lower() in name.lower() or name.lower() in want_name.lower())
    want = "|".join(str(m) for m in want_mibs)
    reason = "" if ok else f"hardware '{raw}' is not '{want_name}, {want}' (name, nvidia-smi memory.total MiB)"
    return {"ok": ok, "name": name, "nvidia_smi_memory_total_mib": mib, "expected_name": want_name, "expected_mib": want_mibs[0] if len(want_mibs) == 1 else list(want_mibs), "reason": reason}


def word(g: dict) -> str:
    """``pinned`` (the pinned card) | ``untested`` (another reading) | ``unread`` (no nvidia-smi reading) for a ``gate`` result."""
    return "pinned" if g.get("ok") else ("unread" if g.get("name") is None else "untested")


def note(g: dict) -> str:
    """'' on the pinned card; else the ONE line that words the card — ``NOTE <reason> — <word> card: <CONSEQUENCE>; proceeding``
    (report.note_line) — which the GPU verbs log before any pass and then run unchanged."""
    w = word(g)
    return "" if w == "pinned" else R.note_line(g["reason"], f"{w} card: {CONSEQUENCE}")
