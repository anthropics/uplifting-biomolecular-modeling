"""Inputs of `pred`: `colabfold_batch`'s own command line. Its first positional — one `.a3m` (ColabFold's own MSA serialization, nothing
searched), a `.fasta`, a `.csv`/`.tsv` of `id,sequence`, or a directory of such files — and its second, the results directory, found among
its options exactly as its parser finds them (split_argv over settings.options_table(), the pinned wheel's own parser) and handed to it VERBATIM; the job names colabfold
gives the queries in it: the completion census reads `<result_dir>/<jobname>.done.txt` — or `<jobname>.result.zip` under `--zip` — (colabfold/batch.py:1405-1412,
colabfold's own finished-job test; manifest.completion) and every output is named `<jobname>_...`. Both come from colabfold itself: `colabfold.input.get_queries` (input.py:267-405: the queries and their raw names — a file's
stem, a fasta header, a csv `id`) and the naming of `colabfold.batch.run` (batch.py:1393-1399: `safe_filename(raw name)`, or
`safe_filename(--jobname-prefix)_<n>` over the queries in `--sort-queries-by` order when that flag is passed); and the token count of a
query (`tokens`: the residues the model runs, every copy of a chain counted — the ACTIVE line's `tokens=<min>-<max>` and the sub-batch
lever's input size). The package runs in colabfold's interpreter; nothing here re-reads or re-writes an input.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

from . import settings


class UsageError(ValueError):
    """The command line does not carry colabfold_batch's two positionals where its parser would find them."""


def split_argv(tokens: Sequence[str]) -> Tuple[str, str, List[str], bool]:
    """`(input, results, options, dashdash)` from colabfold_batch's argv as the caller wrote it (the kit's own flags already taken out):
    the first two tokens its parser reads as positionals — a token that is not an option and not an option's value (settings.options_table():
    `valued` take one value, `--opt=value` carries its own, `bare` take none; everything after a bare `--` is positional) — and every other
    token in order (`options`, the `--` itself dropped: `dashdash` says it was there). Raises UsageError when fewer or more than two
    positionals are present, or when an option colabfold_batch does not define precedes them (its value cannot be told from a positional)."""
    toks = list(tokens)
    table = settings.options_table()
    pos: List[Tuple[int, str]] = []
    opaque = False                                                     # an option outside the table after both positionals: its tokens are colabfold's to judge
    i, dash = 0, None
    while i < len(toks):
        t = toks[i]
        if t == "--":
            dash = i
            pos += [(j, x) for j, x in enumerate(toks[i + 1:], i + 1)]
            break
        if t.startswith("-") and t != "-":
            name = t.split("=", 1)[0]
            if name in table["bare"] or "=" in t or opaque:
                i += 1
            elif name in table["optional_value"]:                      # nargs='?': a following non-option token is its value
                i += 2 if (i + 1 < len(toks) and not toks[i + 1].startswith("-")) else 1
            elif name in table["valued"]:
                i += 2
            elif len(pos) >= 2:
                opaque = True; i += 1
            else:
                raise UsageError(f"{t}: not an option of colabfold_batch this package knows — put <input> <results> before it")
            continue
        if not opaque:
            pos.append((i, t))
        i += 1
    if len(pos) != 2:
        raise UsageError("colabfold_batch's two positionals, <input> <results>, are required" + (f" (found {len(pos)}: {' '.join(x for _, x in pos)})" if pos else ""))
    idx = {j for j, _ in pos} | ({dash} if dash is not None else set())
    options = [x for j, x in enumerate(toks) if j not in idx]
    return pos[0][1], pos[1][1], options, dash is not None


def header_tokens(a3m_lines) -> Optional[int]:
    """Σ length × cardinality from a ColabFold a3m header `#<len,...>\\t<card,...>` (colabfold/input.py:81-82; the first line of query[2][0]);
    None without one."""
    if not a3m_lines:
        return None
    text = a3m_lines[0] if isinstance(a3m_lines, (list, tuple)) else a3m_lines
    if not isinstance(text, str):
        return None
    first = text.split("\n", 1)[0].strip()
    if not first.startswith("#") or "\t" not in first:
        return None
    lens, cards = first[1:].split("\t", 1)
    try:
        ls = [int(x) for x in lens.split(",")]; cs = [int(x) for x in cards.split(",")]
    except ValueError:
        return None
    if len(ls) != len(cs):
        return None
    return sum(l * c for l, c in zip(ls, cs))


def tokens(query) -> int:
    """The residues of the run for one colabfold query tuple `(jobname, query_sequence, a3m_lines, ...)` (batch.py:1172): the a3m header's
    Σ length × cardinality when the query carries one (every copy of a chain counted; the query row itself carries each unique chain once,
    input.py:75-86, so the length colabfold logs at batch.py:1415-1416 under-counts a homo-oligomer), else `len(\"\".join(query_sequence))`."""
    n = header_tokens(query[2]) if len(query) > 2 else None
    return n if n is not None else len("".join(query[1]))


def jobs(path: str, stock_flags: Optional[Sequence[str]] = None) -> List[str]:
    """The job names `colabfold_batch <path> ... <stock_flags>` will run, in its order. Raises what colabfold raises on an input it cannot
    read (OSError: not found; ValueError / AssertionError: unreadable), and ValueError when colabfold finds no query in it."""
    from colabfold.input import get_queries  # noqa: PLC0415 — colabfold's own reader (the stack interpreter carries colabfold)
    queries, _is_complex = get_queries(os.path.abspath(path), sort_queries_by=settings.flag_value(stock_flags, "--sort-queries-by"))
    if not queries:
        raise ValueError(f"{path}: colabfold finds no query in it")
    return job_names(queries, settings.flag_value(stock_flags, "--jobname-prefix"))


def job_names(queries: Sequence, jobname_prefix=None, safe_filename=None) -> List[str]:
    """The job names `colabfold.batch.run(queries, …, jobname_prefix=…)` gives the queries, in their order — its own naming (batch.py:1393-1399):
    `safe_filename(<raw name>)` per query, or `safe_filename(<prefix>)_<n>` with n the query's index zero-filled to `len(str(len(queries)))`
    (ten queries: `<prefix>_00 … <prefix>_09`). `safe_filename`: colabfold's (`colabfold.input.safe_filename`, the name `colabfold.batch`
    carries; imported when not handed in). One naming for `pred`'s completion census (`jobs`) and the run hook's (stack.hook_run)."""
    if safe_filename is None:
        from colabfold.input import safe_filename  # noqa: PLC0415 — colabfold's own namer
    qs = list(queries)
    if jobname_prefix is not None:                                    # batch.py:1393-1396: <safe prefix>_<n>, n zero-filled to the width of the query count
        fill = len(str(len(qs)))
        return [safe_filename(str(jobname_prefix)) + "_" + str(n).zfill(fill) for n in range(len(qs))]
    return [safe_filename(q[0]) for q in qs]                         # batch.py:1399
