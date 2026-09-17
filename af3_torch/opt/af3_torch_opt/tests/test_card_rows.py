"""The kit's per-card kernel rows and deployment configs: the cc 8.0 (A100) rows sit beside the 9.0 / 10.x rows in the cell table the
model process reads no cell table of the kit's (triangle attention, TriMul and transition rows are the shared core providers' measured cells,
asked by the mode's tier word), and configs/a100.env mirrors configs/h100.env but for the card token."""
import json
import os
import re

from af3_torch_opt import stack

KERNELS = os.path.join(stack.forward_dir(), "af3t", "kernels")


PER_CARD_HEADER = "# Per-card differences vs configs/h100.env — this list is the whole of them; anything not listed runs identically to the H100:"


def test_a100_lists_its_per_card_differences():
    """The per-card list is explicit and names none in modes, lever sets or gates (the 8.0 rows are tuning data)."""
    a = open(os.path.join(stack.home(), "configs", "a100.env"), encoding="utf-8").read().split("\n")
    tail = a[a.index(PER_CARD_HEADER):]
    assert any(l.startswith("#   modes and lever sets: none") for l in tail) and any(l.startswith("#   size / token gates: none beyond the H100's") for l in tail)
    assert any("providers' `8.0` cells" in l for l in tail)


def test_a100_config_mirrors_h100():
    """configs/a100.env is configs/h100.env with the card token changed (its own file name, the AF3_TORCH_GPU default and its cc words)."""
    h = open(os.path.join(stack.home(), "configs", "h100.env"), encoding="utf-8").read().split("\n")
    a = open(os.path.join(stack.home(), "configs", "a100.env"), encoding="utf-8").read().split("\n")
    i = a.index(PER_CARD_HEADER)                                             # a100.env closes with the per-card differences list (comment lines only); above it: h100.env's mirror
    assert all(l.startswith("#") for l in a[i:] if l) and a[-1] == ""
    a = a[:i] + [""]
    assert len(h) == len(a)
    diff = [(x, y) for x, y in zip(h, a) if x != y]
    assert 1 <= len(diff) <= 4, diff
    for x, y in diff:
        assert x.replace("h100", "a100").replace("H100", "A100").split("#")[0] == y.split("#")[0], (x, y)
    assert any(l.startswith("export AF3_TORCH_GPU=${AF3_TORCH_GPU:-A100}") for l in a)
