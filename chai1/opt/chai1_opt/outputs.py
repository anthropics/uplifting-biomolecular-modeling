"""Per-prediction outputs — ONE schema for every arm: upstream's own files per (item, seed), every diffusion sample.

Per (input, seed) the fold's output directory holds upstream's ``pred.model_idx_<k>.cif`` and ``scores.model_idx_<k>.npz`` for every
diffusion sample k (``num_diffn_samples``, stock's default 5) and, when the input carries an MSA, upstream's coverage plot ``msa_depth.pdf``
(run_folding_on_context writes it once per call, chai1.py ``plot_msa``; no MSA rows, no file) — in ``trunk_<i>/`` sub-directories when
``num_trunk_samples`` > 1, run_inference's own layout; the package keeps exactly those, under the item's key (the FASTA stem):

    <out>/<tag>/<key>/seed_<s>/[trunk_<i>/]pred.model_idx_<k>.cif, scores.model_idx_<k>.npz     upstream's files, every sample (copied as written)
    <out>/<tag>/<key>/seed_<s>/[trunk_<i>/]msa_depth.pdf                                     upstream's MSA coverage plot, when it wrote one

Kit route: the kit's ``save_seed_outputs`` (``kit/chai_proto.py``) keeps the first sample's two files; ``install_extra_samples_hook`` wraps
it in the launcher process (driver.py) so ``keep_extra_samples`` adds every other sample and ``keep_side_files`` the coverage plot from the fold
directory before the driver removes it. Stock route: upstream writes its files into ``<key>/seed_<s>/`` itself and they stay where it wrote
them; nothing is added beside them. Nothing else is written under ``<out>/<tag>/``.
"""
from __future__ import annotations

import os
import shutil
from typing import List



UPSTREAM_SIDE_FILES = ("msa_depth.pdf",)                # written by run_folding_on_context beside the samples, once per call, when the MSA context is not empty (chai1.py: plot_msa)


def upstream_names(k: int) -> List[str]:
    return [f"pred.model_idx_{k}.cif", f"scores.model_idx_{k}.npz"]


def n_samples(cand) -> int:
    return int(cand.pae.shape[0])


def keep_extra_samples(cand, od, sd) -> List[str]:
    """Copy upstream's files for every sample but the first (``cand.cif_paths[1:]``: every diffusion sample of every trunk sample) from
    the fold directory ``od`` to the keep directory ``sd``, in upstream's own layout: the seed directory itself at one trunk sample,
    ``trunk_<i>/`` sub-directories at more (run_inference, chai1.py). Returns the kept names, relative to ``sd``."""
    kept = []
    for cif in list(cand.cif_paths)[1:]:
        src_dir = os.path.dirname(str(cif)); sub = os.path.relpath(src_dir, str(od))          # "." | "trunk_<i>"
        k = int(os.path.basename(str(cif))[len("pred.model_idx_"):-len(".cif")])
        dst_dir = os.path.normpath(os.path.join(str(sd), sub)); os.makedirs(dst_dir, exist_ok=True)
        for name in upstream_names(k):
            src = os.path.join(src_dir, name)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(dst_dir, name)); kept.append(name if sub == "." else f"{sub}/{name}")
    return kept


def keep_side_files(cand, od, sd) -> List[str]:
    """Copy upstream's per-call side files (``UPSTREAM_SIDE_FILES``: the MSA coverage plot) from every fold-call directory the samples came
    from (``od`` itself, or each ``od/trunk_<i>``) to the same place under the keep directory ``sd`` — present exactly when upstream wrote
    them. Returns the kept names, relative to ``sd``."""
    kept = []
    dirs = []
    for cif in list(cand.cif_paths):
        d = os.path.dirname(str(cif))
        if d not in dirs:
            dirs.append(d)
    for src_dir in dirs:
        sub = os.path.relpath(src_dir, str(od))                                                   # "." | "trunk_<i>"
        for name in UPSTREAM_SIDE_FILES:
            src = os.path.join(src_dir, name)
            if os.path.isfile(src):
                dst_dir = os.path.normpath(os.path.join(str(sd), sub)); os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(src, os.path.join(dst_dir, name)); kept.append(name if sub == "." else f"{sub}/{name}")
    return kept


def install_extra_samples_hook(chai_proto) -> None:
    """Wrap the kit's ``save_seed_outputs`` so that every sample and upstream's side file are kept (upstream's file set, nothing renamed)."""
    orig = chai_proto.save_seed_outputs
    if getattr(orig, "_chai1_opt_hook", False):
        return

    def save_seed_outputs(cand, od, sd):
        row = orig(cand, od, sd)
        extra = keep_extra_samples(cand, od, sd)
        if extra:
            row["n_samples"] = n_samples(cand); row["extra_sample_files"] = extra
        side = keep_side_files(cand, od, sd)
        if side:
            row["side_files"] = side
        return row

    save_seed_outputs._chai1_opt_hook = True
    save_seed_outputs._chai1_opt_orig = orig
    chai_proto.save_seed_outputs = save_seed_outputs
