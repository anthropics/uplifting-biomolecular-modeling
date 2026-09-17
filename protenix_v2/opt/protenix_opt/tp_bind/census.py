"""Stage marks of the row-sharded run for the shared core's memory census (``opt_core.mem.rowpair.census``: one ``TPCENSUS`` JSON line
per mark on a rank's stderr when ``OPT_CORE_TP_CENSUS=1``, nothing at all otherwise). The marks ride the unit's own instrumentation
points — its phase log (``ptx_tp.mirror.PhaseLog.phase``: trunk_start / trunk_cycle_end / trunk_end / diffusion_start / diffusion_end /
confidence_start / confidence_end per item) and the return of three seam calls (template embedder, MSA module, distogram rows) — which
``tp_route.install`` rebinds in every rank; no statement of the run changes.

    unit phase / seam return                     census stage(s)
    trunk_start                                  trunk_entry
    ptx_tp.msa.tp_template_embedder returns      after_template          (per recycling cycle)
    ptx_tp.msa.tp_msa_module returns             after_msa               (per recycling cycle)
    trunk_cycle_end                              after_pairstack         (per recycling cycle)
    trunk_end                                    no_gather               (z stays sharded into the heads)
    ptx_tp.confidence.tp_distogram_contact_rows  distogram
    diffusion_start / diffusion_end              diffusion_start / diffusion_peak
    confidence_start                             confidence_start        (outside the census vocabulary: flagged known=false by the reader)
    confidence_end                               confidence, done        (per item: the last stage of a fold)
"""
from __future__ import annotations

import functools
from typing import Dict, Tuple

try:                                                       # instrumentation only: a core without the census module leaves the marks unplaced
    from opt_core.mem.rowpair import census
except ImportError:                                        # pragma: no cover
    census = None

__all__ = ["PHASE_STAGES", "CALL_STAGES", "SITES", "install", "stages_for_phase"]

TAG = "protenix-opt"
PHASE_STAGES: Dict[str, Tuple[str, ...]] = {
    "trunk_start": ("trunk_entry",), "trunk_cycle_end": ("after_pairstack",), "trunk_end": ("no_gather",),
    "diffusion_start": ("diffusion_start",), "diffusion_end": ("diffusion_peak",),
    "confidence_start": ("confidence_start",), "confidence_end": ("confidence", "done"),
}
CALL_STAGES: Dict[Tuple[str, str], str] = {
    ("ptx_tp.msa", "tp_template_embedder"): "after_template",
    ("ptx_tp.msa", "tp_msa_module"): "after_msa",
    ("ptx_tp.confidence", "tp_distogram_contact_rows"): "distogram",
}
PHASE_SITE = ("ptx_tp.mirror", "PhaseLog.phase")
SITES: Tuple[Tuple[str, str], ...] = (PHASE_SITE,) + tuple(CALL_STAGES)


def stages_for_phase(name: str) -> Tuple[str, ...]:
    return PHASE_STAGES.get(name, ())


def _mark(stage: str, **extra) -> None:
    if census is not None:
        census.mark(stage, **extra)


def _phase_factory(original):
    @functools.wraps(original)
    def phase(self, name, *args, **kwargs):
        rec = original(self, name, *args, **kwargs)
        for stage in stages_for_phase(name):
            _mark(stage, phase=name)
        return rec
    return phase


def _after_factory(stage: str):
    def make(original):
        @functools.wraps(original)
        def marked(*args, **kwargs):
            out = original(*args, **kwargs)
            _mark(stage)
            return out
        return marked
    return make


def install():
    """Arm the marks (idempotent per site): the phase method and the three seam returns are rebound when their modules are imported."""
    from opt_core.autoload import patch_attr_at_import
    out = [patch_attr_at_import(PHASE_SITE[0], PHASE_SITE[1], _phase_factory, tag=TAG, name="tp_census:phase")]
    for (module, attr), stage in CALL_STAGES.items():
        out.append(patch_attr_at_import(module, attr, _after_factory(stage), tag=TAG, name=f"tp_census:{module}.{attr}"))
    return out
