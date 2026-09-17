"""The outputs of a generation run and their census. Upstream roots everything at its working directory (the run directory ``<out>``,
``stock_design.py``): ``<out>/inference/<config>_<item>[_<run>]/`` is the run's root (generate.py setup(): config =
``search_binder_local_pipeline``, item = ``generation.task_name``, run = ``run_name`` — the shipped ``search_binder_local`` unless a
``++run_name=`` token names one), holding one directory per design ``job_<j>_n_<N>_id_<k>_<tag>/`` with one ``job_<j>_n_<N>_id_<k>_<tag>.pdb``
inside (generate.py save_predictions: N = the residues of the sample = target + binder, k = 0 .. n-1 per length in generation order, tag =
the sampler's metadata tag — ``single_orig<k>`` under single-pass); chain A of each PDB = the target crop, chain B = the binder.
``DESIGN_GLOB`` counts ``job_*/*.pdb`` under the run's root, whatever N and tag read. ``census`` accounts for the run: designs_written (the
PDB files under the run root that this run wrote — files already there before the launch are counted apart, ``preexisting``) against
designs_expected (``nres.nsamples`` × ``nrepeat_per_sample``, the designs asked) — the run is ``incomplete`` when it wrote fewer (one PDB per
design under single-pass generation; a search algorithm that keeps several candidates per design, as the shipped best-of-n does, writes more), and the OUTPUTS / EXIT
lines and the manifest carry both counts. ``results_csv`` is the file whose presence makes upstream exit 0 without generating (generate.py
main: "Results already exist"); the run names it in one NOTE line before the launch and refuses nothing. Upstream's CLI writes
``<out>/logs/*.log`` unless ``--verbose`` is among its arguments (cli_runner.py LOG_DIR); the composed command carries ``--verbose``.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, List, Optional

CONFIG_NAME = "search_binder_local_pipeline"            # the config file's stem = upstream's config_name / base_config_name
INFERENCE_DIR = "inference"
DESIGN_GLOB = os.path.join("job_*", "*.pdb")


def run_root(out_dir: str, item: Optional[str], run_name: Optional[str]) -> str:
    """``<out>/inference/search_binder_local_pipeline[_<item>[_<run>]]`` — upstream's root_path (generate.py setup(): the run suffix applies
    only with a task name; no task name = the bare config name)."""
    name = CONFIG_NAME
    if item:
        name += f"_{item}"
        if run_name:
            name += f"_{run_name}"
    return os.path.join(out_dir, INFERENCE_DIR, name)


def results_csv(out_dir: str, job_id: int = 0) -> str:
    """``<out>/inference/results_search_binder_local_pipeline_<job>.csv``."""
    return os.path.join(out_dir, INFERENCE_DIR, f"results_{CONFIG_NAME}_{int(job_id)}.csv")


def design_files(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, DESIGN_GLOB)))


def newest_root(out_dir: str, since: float) -> Optional[str]:
    """The most recently modified run root under ``<out>/inference`` touched at or after ``since`` (unix s), or None."""
    roots = [p for p in glob.glob(os.path.join(out_dir, INFERENCE_DIR, CONFIG_NAME + "*")) if os.path.isdir(p) and os.path.getmtime(p) >= since - 1.0]
    return max(roots, key=os.path.getmtime) if roots else None


def census(out_dir: str, root: str, item: Optional[str], designs_expected: Optional[int], preexisting: Iterable[str] = (),
           root_source: str = "composed") -> Dict[str, object]:
    """``{item, root, root_source, designs_expected (int | None), designs_written, designs_preexisting, files, items_complete,
    item_failed: {item: reason}, incomplete: str | None, n_items, n_items_complete}`` over the run's root. ``designs_written`` counts the files
    not in ``preexisting``; the run is complete when it wrote at least ``designs_expected`` of them and at least one (more is upstream's search
    keeping several candidates per design); with ``designs_expected`` None (a design count the package could not read), when it wrote at least one."""
    before = set(preexisting)
    files = design_files(root)
    new = [f for f in files if f not in before]
    w = len(new)
    e = None if designs_expected is None else int(designs_expected)
    ok = (w >= e and w > 0) if e is not None else (w > 0)      # short = fewer PDBs than designs asked (or none); a search that keeps several candidates per design (the shipped best-of-n: one per replica) writes more, never fewer
    key = item or "none"
    if ok:
        failed, incomplete = {}, None
    elif not os.path.isdir(root):
        failed, incomplete = {key: "run root missing"}, "run root missing"
    else:
        failed, incomplete = {key: f"{w}/{e if e is not None else '?'} designs written"}, f"designs {w}/{e if e is not None else 'unknown'}"
    return {"item": item, "root": root, "root_source": root_source, "designs_expected": e, "designs_written": w, "designs_preexisting": len(files) - w,
            "files": [os.path.relpath(f, out_dir) for f in new], "items_complete": ok, "item_failed": failed, "incomplete": incomplete,
            "n_items": 1, "n_items_complete": int(ok)}
