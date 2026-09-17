"""warm: the kit's bundled public tiles through pred; refusals named before any GPU work."""
import json
import os
import re

import pytest

from .. import stack, warm


def test_warm_folds_every_bundled_tile_one_per_size_class(tmp_path):
    """warm's input is the whole bundled public file: the four 1BRS tiles, 199 / 398 / 597 / 796 tokens — the three pair-row buckets the fused
    LayerNorm-linear / transition kernels specialise on (< 256, 256-632, >= 633) and, with the largest, a 16-divisible pair-row count."""
    p = warm.public_input(str(tmp_path))
    items = json.load(open(p))
    tokens = [sum(len(c["seq"]) for c in it["components"]) for it in items]
    assert os.path.basename(p) == warm.WARM_INPUT and [it["name"] for it in items] == [f"1brs_barnase_barstar_{k}to{k}" for k in (1, 2, 3, 4)] and tokens == [199, 398, 597, 796]
    assert sorted({0 if t * t < 65536 else (1 if t * t < 400000 else 2) for t in tokens}) == [0, 1, 2] and any(t * t % 16 == 0 for t in tokens) and any(t * t % 16 for t in tokens)   # pair-row counts on both sides of Triton's 16-divisibility specialisation


def test_warm_refuses_off_like_run_sh(tmp_path):
    with pytest.raises(warm.WarmError, match="warm --mode off is refused"):
        warm.run("off", out_dir=str(tmp_path))
    run_sh = open(os.path.join(stack.tree_root(), "run.sh")).read()
    assert re.search(r'case "\$CMD" in pred\|check\) ;; \*\)', run_sh)          # run.sh: --mode off has pred (plus check) only
