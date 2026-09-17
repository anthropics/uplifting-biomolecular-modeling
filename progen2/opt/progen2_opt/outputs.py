"""Items in, blocks out — the one schema both arms of `sample` / `score` use for a multi-item job (``--input FILE --out_dir DIR``; never
re-implemented per arm).

Items (``load_items``): a JSON-lines file (or a JSON list), one object per item; the keys are the argparse dests of the stock script's own
per-item flags, a missing key takes the stock default:
  sample   {"item_id": str, "context": str, "max_length": int, "num_samples": int, "t": float, "p": float, "rng_seed": int}
           — sample.py's --context / --max-length / --num-samples / --t / --p / --rng-seed (``SAMPLE_DEFAULTS``: 1, 256, 1, 0.2, 0.95, 42).
  score    {"item_id": str, "context": str}   — likelihood.py's --context as given (the authors' convention: the string is scored as
           written; the stock wraps nothing); a missing context takes likelihood.py's own default (read from the stock file).
``item_id`` defaults to ``item<index>``; any other key is a caller's bookkeeping and is ignored, except the retired spelling ``seed``, refused
by name (sample.py's dest is ``rng_seed``).

Outputs (``Writer``): ``<out_dir>/items/<item_id>/block.txt`` and nothing else — the bytes both arms print identically: for `sample` the stock
CLI's block (the context line, then per sample an empty line, the index and the truncated completion, then `done.` — sample.py L196-202; the
in-process route prints the same block), for `score` likelihood.py's result lines: the two lines ``ll_sum=<float>`` / ``ll_mean=<float>`` (L298-299),
preceded under ``--sanity true`` by the sanity section's own prints (L213-273: the cross-entropy triple, ll_0..2, the three sequences,
ll_x_*) — i.e. its stdout without print_time's ``<desc>`` / ``<desc> took <t>s`` lines and the closing ``done.``.
``block_of_stdout`` extracts the block from a stock process's full stdout.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import List, Optional

SAMPLE_DEFAULTS = {"context": "1", "max_length": 256, "num_samples": 1, "t": 0.2, "p": 0.95, "rng_seed": 42}     # sample.py main() L114-121 (argparse dests)
SAMPLE_ARG_OF = {"context": "--context", "max_length": "--max-length", "num_samples": "--num-samples", "t": "--t", "p": "--p", "rng_seed": "--rng-seed"}
RETIRED_KEYS = {"seed": "rng_seed (sample.py's --rng-seed)"}                                                        # a key whose meaning moved: refused by name, never read as bookkeeping
ITEMS_DIRNAME, BLOCK_NAME = "items", "block.txt"
LL_RE = re.compile(r"^ll_(sum|mean)=(.+)$")
TOOK_RE = re.compile(r"^(.+) took \d+\.\d+s$")                                                                       # print_time.__exit__'s line (likelihood.py L31); its group = the desc __enter__ printed (L27)


def stock_likelihood_default_context(stock_dir: str) -> Optional[str]:
    """likelihood.py's own --context default, read from the stock file (never re-typed)."""
    try:
        src = open(os.path.join(stock_dir, "likelihood.py"), "r", encoding="utf-8").read()
    except OSError:
        return None
    m = re.search(r"--context',\s*type=str,\s*default='([^']*)'", src) or re.search(r'--context",\s*type=str,\s*default="([^"]*)"', src)
    return m.group(1) if m else None


def load_items(path: str, route: str, stock_dir: Optional[str] = None) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if text.lstrip().startswith("["):
        items = json.loads(text)
    else:
        items = [json.loads(l) for l in text.splitlines() if l.strip()]
    return load_items_from(items, route, stock_dir)


def default_item_id(i: int) -> str:
    """The id of the i-th item when the object names none (a single call's one item is item0000)."""
    return f"item{i:04d}"


def load_items_from(items: list, route: str, stock_dir: Optional[str] = None) -> List[dict]:
    """The items of a list of objects: sample items filled with sample.py's own defaults (SAMPLE_DEFAULTS), score items with likelihood.py's default context."""
    out = []
    seen = set()
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"item {i}: not an object")
        iid = str(it.get("item_id") or default_item_id(i))
        if iid in seen or "/" in iid or iid.startswith("."):
            raise ValueError(f"item {i}: bad or duplicate item_id {iid!r}")
        seen.add(iid)
        for k, now in RETIRED_KEYS.items():
            if k in it:
                raise ValueError(f"item {iid}: the key {k!r} is retired — the item keys are the stock flags' dests: use {now}")
        if route == "sample":
            row = {"item_id": iid}
            for k, d in SAMPLE_DEFAULTS.items():
                v = it.get(k, d)
                row[k] = type(d)(v) if not isinstance(v, type(d)) else v
            out.append(row)
        elif route == "score":
            ctx = it.get("context")
            if ctx is None:
                ctx = stock_likelihood_default_context(stock_dir) if stock_dir else None
                if ctx is None:
                    raise ValueError(f"item {iid}: no context and no stock dir to read likelihood.py's default from")
            out.append({"item_id": iid, "context": str(ctx)})
        else:
            raise ValueError(route)
    return out


def sample_argv(item: dict) -> List[str]:
    """sample.py's own per-item arguments for an item (every field explicit: the stock defaults are the item's own values)."""
    argv = []
    for k, flag in SAMPLE_ARG_OF.items():
        argv += [flag, str(item[k])]
    return argv


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def block_of_stdout(route: str, stdout: str) -> str:
    """The block the arms print identically, cut from a STOCK process's stdout. `sample` = the lines after the `sampling` marker
    (print_time's desc) up to its `sampling took` line — the context, then per sample a blank line, the index and the truncated
    completion (sample.py:196-202) — followed by the script's closing `done.` line: the grammar the in-process route's block
    has (`opt/serving/pipeline_v0_4/progen2_decode.py` Handle.unit: `context\\n` + `\\n<i>\\n<truncation>\\n` per sample + `done.\\n`), so the two arms'
    blocks are the same bytes when the completions are. `score` = likelihood.py's result lines: its stdout without print_time's
    `<desc>` / `<desc> took <t>s` lines (TOOK_RE; the volatile walls) and the closing `done.` — the section-7 pair `ll_sum=` / `ll_mean=`,
    preceded by the sanity section's prints when --sanity true (the in-process route prints the same lines from the same stock
    statements); empty when the pair is not there (a process that did not reach section 7)."""
    lines = stdout.splitlines()
    if route == "sample":
        try:
            start = max(i for i, l in enumerate(lines) if l.strip() == "sampling") + 1
        except ValueError:
            return ""
        end = next((i for i in range(start, len(lines)) if lines[i].startswith("sampling took")), len(lines))
        body = lines[start:end]
        if any(l.strip() == "done." for l in lines[end:]):
            body = body + ["done."]
        return "\n".join(body) + "\n"
    if route == "score":
        took = [TOOK_RE.match(l) for l in lines]
        descs = {m.group(1) for m in took if m}
        body = [l for l, m in zip(lines, took) if not m and l not in descs]
        if body and body[-1].strip() == "done.":
            body = body[:-1]
        return "\n".join(body) + "\n" if any(LL_RE.match(l) for l in body) else ""
    raise ValueError(route)


class Writer:
    """The one writer of a multi-item job: ``<out_dir>/items/<item_id>/block.txt`` per item, nothing else."""

    def __init__(self, out_dir: str, route: str):
        self.out_dir, self.route = os.path.abspath(out_dir), route
        os.makedirs(os.path.join(self.out_dir, ITEMS_DIRNAME), exist_ok=True)
        self.n_written = 0

    def item_dir(self, item_id: str) -> str:
        d = os.path.join(self.out_dir, ITEMS_DIRNAME, item_id)
        os.makedirs(d, exist_ok=True)
        return d

    def write_item(self, item: dict, block: str) -> str:
        p = os.path.join(self.item_dir(item["item_id"]), BLOCK_NAME)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(block)
        self.n_written += 1
        return p
