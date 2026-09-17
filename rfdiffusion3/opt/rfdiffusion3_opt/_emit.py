"""The package's one printer: every `[rfdiffusion3-opt] …` line goes to stderr through `emit`, at a LINE START whatever precedes it on the
stream, flushed at once.

Upstream draws tqdm progress bars on the same stream (the sampler's step bar rewrites its line with `\r` and ends without a newline), and a
consumer reads the census by splitting the output on `\n` (and `\r`) and matching lines that start with the tag. `fresh(line)` is the
one spelling of that guarantee — a newline before the line — and `emit(line)` writes `fresh(line)` plus the terminating newline and flushes,
so a census line printed inside a handler is on the wire when the handler returns (a consumer that timestamps lines on arrival sees it then).
Lines the shared core prints on this package's behalf (the exit tally) are handed to it already `fresh`. Standard library only.
"""
from __future__ import annotations

import sys

FRESH = "\n"                                            # what precedes every census line: the line starts a line even after a `\r`-drawn progress bar


def fresh(line: str) -> str:
    """`line` preceded by the newline that puts it at a line start."""
    return FRESH + line


def emit(line: str, stream=None) -> None:
    """Write one census line — fresh, newline-terminated, flushed — to `stream` (default: the process's stderr as bound now)."""
    stream = sys.stderr if stream is None else stream
    stream.write(fresh(line) + "\n")
    stream.flush()
