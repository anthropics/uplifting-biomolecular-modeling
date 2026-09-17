"""The bit-equality probe of an alternative formulation (a batched GEMM for a loop of GEMMs, a fused statement for an unfused one): run both
on the device on the job's real shapes, compare bit-exact, decide once, serve by the verdict — never assume.

Contract. A lever that re-expresses a stock computation in a faster FORM is exact only if the two forms agree bit for bit on the running stack at
the shapes the job will use — a fact of the card, the library version and the shape (which kernel cuBLAS picks), not of the source text.
:class:`IdentityProbe` records one ``cell(key, candidate, reference)`` per real shape (both callables run, their output trees compared with
:func:`opt_core.capture.graphs.bitwise_equal`; a candidate that raises is a failed cell named by the exception), then ``decide(policy)`` fixes a
STICKY verdict — ``"all"``: the candidate form serves every probed key iff every cell agreed; ``"per_cell"``: it serves exactly the keys that agreed
— and ``serve(key)`` answers the lever's per-call question from that verdict: an unprobed key is served by the reference form and COUNTED
(``unprobed``), a call before the verdict likewise (``undecided``); no cell is accepted after the verdict (no mid-job switching). The evidence is
one ``LEVER`` line (``probe=pass|fail|undecided cells=<n> bad=<n> maxabs=<g> policy=… served=… reference=… unprobed=…``), a manifest record with
every cell, and a gate. Torch is touched only through the callables' outputs (Python 3.8 floor).
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from .. import report
from ..oom import is_oom

__all__ = ["IdentityProbe", "ProbeClosed"]

IMPL = "identity_probe"


class ProbeClosed(RuntimeError):
    """A cell offered after :meth:`IdentityProbe.decide`: the verdict is sticky (no mid-job switching); probe every shape before deciding."""


class IdentityProbe:
    """One lever's bit-equality probe (module contract)."""

    def __init__(self, name: str, *, log: Optional[Callable[[str], None]] = None):
        self.name = str(name)
        self._log = log
        self._lock = threading.RLock()
        self.cells: List[Dict[str, Any]] = []
        self.policy: Optional[str] = None
        self.verdict: Optional[bool] = None            # policy "all": the one verdict; "per_cell": True iff at least one key serves
        self.ok_keys: set = set()
        self.stats_ = {"served": 0, "reference": 0, "unprobed": 0, "undecided": 0}

    def _say(self, line: str) -> None:
        if self._log is not None:
            self._log(f"[probe:{self.name}] {line}")

    @staticmethod
    def _key(key) -> str:
        if isinstance(key, (tuple, list)):
            return "x".join(str(k) for k in key)
        return str(key)

    def cell(self, key, candidate: Callable[[], Any], reference: Callable[[], Any]) -> bool:
        """Run ``reference()`` then ``candidate()`` (no arguments: the caller closes over the real operands), compare their output trees bit-exact, record
        the cell; returns whether they agreed. A candidate that raises is recorded as a failed cell (``error=<ExceptionType>``), not propagated."""
        from ..capture.graphs import bitwise_equal, max_abs_diff  # noqa: PLC0415
        k = self._key(key)
        with self._lock:
            if self.policy is not None:
                raise ProbeClosed(f"{self.name}: cell {k} offered after decide() — the verdict is sticky")
            ref = reference()
            rec: Dict[str, Any] = {"key": k, "ok": False, "max_abs_diff": None, "error": None}
            try:
                cand = candidate()
            except Exception as e:                      # noqa: BLE001 — a formulation that cannot run at this shape is a failed cell, by name; an OOM is not
                if is_oom(e):
                    raise
                rec["error"] = type(e).__name__
                self.cells.append(rec)
                self._say(f"cell {k}: candidate raised {rec['error']} -> not identical")
                return False
            rec["ok"] = bool(bitwise_equal(cand, ref))
            if not rec["ok"]:
                rec["max_abs_diff"] = max_abs_diff(cand, ref)
            self.cells.append(rec)
            return rec["ok"]

    def decide(self, policy: str = "all") -> bool:
        """Fix the sticky verdict. ``"all"``: serve the candidate for every probed key iff EVERY cell agreed (and there is at least one); ``"per_cell"``:
        serve exactly the keys whose cell agreed. Returns whether anything will be served. One line is logged."""
        if policy not in ("all", "per_cell"):
            raise ValueError("policy must be 'all' or 'per_cell'")
        with self._lock:
            if self.policy is not None:
                return bool(self.verdict)
            self.policy = policy
            good = {c["key"] for c in self.cells if c["ok"]}
            bad = [c for c in self.cells if not c["ok"]]
            if policy == "all":
                self.verdict = bool(self.cells) and not bad
                self.ok_keys = {c["key"] for c in self.cells} if self.verdict else set()
            else:
                self.ok_keys = good
                self.verdict = bool(good)
            worst = max((c["max_abs_diff"] or 0.0 for c in bad), default=0.0)
            if self.verdict and not bad:
                self._say(f"PROBE PASS -> candidate form serves all {len(self.cells)} cell(s)")
            elif self.verdict:
                self._say(f"PROBE PARTIAL ({len(bad)}/{len(self.cells)} cells not bit-identical, max|d| {worst:.3g}) -> candidate form serves {len(good)} cell(s), reference form elsewhere")
            else:
                self._say(f"PROBE FAIL ({len(bad)}/{len(self.cells)} cells not bit-identical, max|d| {worst:.3g}) -> reference form for the whole job (outputs stay exact)")
            return bool(self.verdict)

    def serve(self, key=None) -> bool:
        """The lever's per-call question: use the candidate form for ``key``? False (counted ``undecided``) before :meth:`decide`; False (counted
        ``unprobed``) for a key no cell probed; otherwise the verdict for that key. Every answer is counted ``served`` or ``reference``."""
        with self._lock:
            if self.policy is None:
                self.stats_["undecided"] += 1
                self.stats_["reference"] += 1
                return False
            k = self._key(key) if key is not None else None
            if k is not None and k not in {c["key"] for c in self.cells}:
                self.stats_["unprobed"] += 1
                self.stats_["reference"] += 1
                return False
            use = bool(self.verdict) and (k is None or k in self.ok_keys)
            self.stats_["served" if use else "reference"] += 1
            return use

    # ------------------------------------------------------------------------------------------------------------- census
    def word(self) -> str:
        if self.policy is None:
            return "undecided"
        bad = sum(1 for c in self.cells if not c["ok"])
        if not self.cells:
            return "no_cells"
        if not bad:
            return "pass"
        return "partial" if self.verdict else "fail"

    def stats(self) -> dict:
        with self._lock:
            bad = [c for c in self.cells if not c["ok"]]
            d = dict(self.stats_)
            d.update(name=self.name, probe=self.word(), policy=self.policy, cells=len(self.cells), bad=len(bad),
                     maxabs=max((c["max_abs_diff"] or 0.0 for c in bad), default=0.0), bad_keys=[c["key"] for c in bad],
                     errors=sorted({c["error"] for c in self.cells if c["error"]}))
            return d

    def fields(self) -> List[Tuple[str, Any]]:
        s = self.stats()
        return [("probe", s["probe"]), ("cells", s["cells"]), ("bad", s["bad"]), ("maxabs", f"{s['maxabs']:.3g}"), ("policy", s["policy"] or "none"),
                ("served", s["served"]), ("reference", s["reference"]), ("unprobed", s["unprobed"]), ("undecided", s["undecided"])]

    def line(self, tag: str, name: Optional[str] = None, strategy: Optional[str] = None) -> str:
        """``[<tag>] LEVER name=<name> state=on|skipped [reason=probe_fail|undecided|no_cells] impl=identity_probe origin=core [strategy=…] probe=… cells=…
        bad=… maxabs=… policy=… served=… reference=… unprobed=… undecided=…`` (:func:`opt_core.report.lever_line`)."""
        w = self.word()
        state = "on" if w in ("pass", "partial") else "skipped"
        reason = None if state == "on" else {"fail": "probe_fail", "undecided": "undecided", "no_cells": "no_cells"}[w]
        return report.lever_line(tag, name or self.name, state, *self.fields(), reason=reason, impl=IMPL, origin="core", strategy=strategy)

    def gate(self, *, require_pass: bool = False, max_unprobed: Optional[int] = None, allow_partial: bool = True) -> List[str]:
        """The fail-closed sentences (empty = pass): ``require_pass`` with a failed / undecided / empty probe (a lever REQUESTED as exact whose probe
        did not pass); a partial verdict when ``allow_partial`` is False; unprobed calls above ``max_unprobed`` when given."""
        s, out = self.stats(), []
        if require_pass and s["probe"] not in ("pass",) and not (allow_partial and s["probe"] == "partial"):
            out.append(f"{self.name}: probe {s['probe']} ({s['bad']}/{s['cells']} cells not bit-identical{', keys ' + ','.join(s['bad_keys']) if s['bad_keys'] else ''}, max|d| {s['maxabs']:.3g})")
        if not allow_partial and s["probe"] == "partial":
            out.append(f"{self.name}: partial verdict not allowed ({s['bad']}/{s['cells']} cells serve the reference form)")
        if max_unprobed is not None and s["unprobed"] > int(max_unprobed):
            out.append(f"{self.name}: unprobed={s['unprobed']} calls above the allowed {int(max_unprobed)} (shapes served by the reference form without a cell)")
        return out

    def record(self) -> dict:
        """The manifest block: verdict word, policy, every cell (key, ok, max_abs_diff, error), the serve counts."""
        s = self.stats()
        return {"name": self.name, "probe": s["probe"], "policy": self.policy, "cells": [dict(c) for c in self.cells],
                "served": s["served"], "reference": s["reference"], "unprobed": s["unprobed"], "undecided": s["undecided"]}
