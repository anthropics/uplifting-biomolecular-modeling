"""Background CIF writers used by `run_seq_des` (in the kit's lever files fast/seq_des_utils.py; switches
`CALIBY_X_BG_CIF` and `CALIBY_X_CIF_WORKERS`), layered over the shared core's `opt_core.host.outputs.AsyncWriter`.
This module keeps the glue: which files a batch hands over (`Writers.submit_batch`), when every write is joined
(`Writers.join`, before `run_seq_des` returns and before the caller's OUTPUTS_WRITTEN stamp), and the census words
(`census_word`, read by `report.cif_writers_state`).

`CALIBY_X_BG_CIF` = ``fork`` | ``thread`` | unset/``0`` (unset: each CIF serialised and written synchronously in the
loop; `from_env` returns None). `CALIBY_X_CIF_WORKERS` = N >= 1 workers of that mode (unset/``0`` = 1: one background
writer serialising each batch's files while the GPU samples the next batch); a batch's ``(path, AtomArray)`` items
are split into at most N chunks of >= `CHUNK_MIN` files, one `AsyncWriter.submit` per chunk, at most 2N chunks
outstanding (the producer blocks in submit; a forked worker touches no CUDA state). Same serialiser (`to_cif_string`)
on the same objects, same paths, as the synchronous loop. A write that fails is counted by the core and, after the
join, named on stderr (``[CALIBY_X_BG_CIF] ... failed: the outputs of this run are incomplete``) and raised (the
core's RuntimeError): the process exits non-zero; nothing is rewritten and nothing is dropped silently. Malformed
switch values are a usage error (`env_words`).
"""
import sys
from typing import Any, Callable, List, Optional, Sequence, Tuple

from .env_words import choice_word, int_word

MODE_SWITCH, WORKERS_SWITCH = "CALIBY_X_BG_CIF", "CALIBY_X_CIF_WORKERS"
MODES = ("fork", "thread")
FAILURE_MARKER = f"[{MODE_SWITCH}]"
CHUNK_MIN = 8                       # files per chunk at least: below that a chunk costs more to hand over than to write
STRATEGY = "F6.output_overlap"      # opt_core STRATEGIES.json canonical id of the mechanism
STATE = {"mode": None, "workers": 0, "batches": 0, "chunks": 0, "files": 0, "failed": 0, "joined": False}


def settings() -> Tuple[Optional[str], int]:
    """(mode | None, workers) from the two switch words; None = the stock synchronous writes."""
    mode = choice_word(MODE_SWITCH, MODES + ("0",), default="0")
    workers = int_word(WORKERS_SWITCH, default=0)
    return (None if mode == "0" else mode), max(1, workers)


def chunks(items: Sequence[Any], workers: int) -> List[Sequence[Any]]:
    """``items`` split into k = min(workers, len // CHUNK_MIN) contiguous, balanced chunks (at least one): every chunk holds >= CHUNK_MIN items
    whenever there is more than one, every item exactly once, in order."""
    if not items:
        return []
    k = max(1, min(int(workers), len(items) // CHUNK_MIN))
    q, r = divmod(len(items), k)
    out, start = [], 0
    for i in range(k):
        size = q + (1 if i < r else 0)
        out.append(items[start:start + size])
        start += size
    return out


class Writers:
    """One design pass's background writers: an `AsyncWriter` of ``mode`` with ``workers`` workers running ``write(chunk)``."""

    def __init__(self, write: Callable[[Any], Any], mode: str, workers: int, tag: str = "caliby-opt") -> None:
        from opt_core.host.outputs import AsyncWriter
        self.mode, self.workers, self.tag = mode, int(workers), tag
        self.aw = AsyncWriter(write, workers=self.workers, max_pending=2 * self.workers, name="cif", mode=mode)
        STATE.update(mode=mode, workers=self.workers, joined=False)

    def lever_line(self) -> str:
        """``[<tag>] LEVER name=F6.output_overlap state=on impl=host.outputs origin=core part=writer ...`` (the core's evidence line)."""
        return self.aw.active_line(self.tag, STRATEGY)

    def submit_batch(self, items: Sequence[Any]) -> int:
        """Hand one batch's items over, chunked; returns the number of chunks submitted."""
        parts = chunks(items, self.workers)
        for part in parts:
            self.aw.submit(list(part))
        STATE["batches"] += 1
        STATE["chunks"] += len(parts)
        STATE["files"] += len(items)
        return len(parts)

    def join(self) -> None:
        """Wait for every write, close the workers; a failed write is the FAILURE_MARKER line on stderr and the core's RuntimeError."""
        try:
            self.aw.close()
        except RuntimeError:
            failed = int(self.aw.tally().get("failed", 0)) or 1
            STATE["failed"] = failed
            sys.stderr.write(f"{FAILURE_MARKER} {failed} background CIF write(s) failed ({self.mode} mode, {self.workers} worker(s)): "
                             f"the outputs of this run are incomplete\n")
            sys.stderr.flush()
            raise
        finally:
            STATE["joined"] = True


def from_env(write: Callable[[Any], Any], tag: str = "caliby-opt") -> Optional[Writers]:
    """The pass's writers from the switch words — announced by the core's LEVER line on stderr (`report.say`) — or None for the stock
    synchronous writes."""
    mode, workers = settings()
    if mode is None:
        STATE.update(mode="0", workers=0)
        return None
    writers = Writers(write, mode, workers, tag)
    from . import report                    # report imports this module lazily too (cif_writers_state): no import cycle at load
    report.say(writers.lever_line())
    return writers


def census_word() -> str:
    """The exit line's ``cif_writers=`` value: ``unused`` (run_seq_des never ran here: the ensemble route writes elsewhere) | ``sync``
    (`CALIBY_X_BG_CIF` unset: stock writes) | ``<mode>:<workers>w/<chunks>chunks/<files>files[,failed:<n>]``."""
    if STATE["mode"] is None:
        return "unused"
    if STATE["mode"] == "0":
        return "sync"
    word = f"{STATE['mode']}:{STATE['workers']}w/{STATE['chunks']}chunks/{STATE['files']}files"
    if STATE["failed"]:
        word += f",failed:{STATE['failed']}"
    return word
