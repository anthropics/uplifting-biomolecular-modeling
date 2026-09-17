"""``<out>/opt_manifest.json`` of a generation run: the core's document (``opt_core.manifest.build``: schema, package, mode, route,
activation, command, stack, exit, the run census) plus this engine's ``kit`` block — the designs, seed and batch as run, the item and its
target entry, the overrides as given, the weights directory and the pinned digests it was checked against (bytes), the env proof's path and
verdict, and the child's invocation record.
"""
from __future__ import annotations

import os
from typing import List, Mapping, Optional

from opt_core import manifest as core

from . import TAG, __version__

SCHEMA = "complexa_opt/2"
FILENAME = core.FILENAME


def build(*, mode: str, route: str, report: Optional[Mapping], command: List[str], stack_block: Mapping, exit_: Mapping, census: Mapping,
          kit: Mapping) -> dict:
    return core.build(package="complexa_opt", package_version=__version__, schema=SCHEMA, mode=mode, report=report, route=route, command=command,
                      stack=stack_block, exit_=exit_, kit=kit,
                      pass_={"pass_status": "complete" if census.get("items_complete") else "incomplete", "n_items": census.get("n_items"),
                             "n_items_complete": census.get("n_items_complete"), "items_complete": census.get("items_complete"),
                             "item_failed": census.get("item_failed"), "designs_written": census.get("designs_written"),
                             "designs_expected": census.get("designs_expected")},
                      top_level=("designs", "seed", "batch", "weights", "designs_written", "designs_expected"))


def write(out_dir: str, doc: Mapping) -> str:
    return core.write(os.path.join(out_dir, FILENAME), doc)


def line(path: str) -> str:
    return core.line(TAG, path)
