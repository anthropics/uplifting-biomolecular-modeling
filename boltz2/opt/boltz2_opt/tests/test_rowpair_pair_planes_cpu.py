"""rowpair._pair_planes_to_host moves the featurizer's NAMED token-pair planes (PAIR_PLANES) to the host and nothing else: at N = 33 tokens the
per-token features res_type / profile / msa / template_restype (one-hot over const.num_tokens = 33) are shaped ``[1, N, N]`` too, and the model
reads them on the device (the template rows' ``torch.cat([a_tij, res_type_i, res_type_j])`` was the device error at n_gpu = 2 on a 33-token
input). CPU-only: a stand-in tensor that reports ``is_cuda`` and records its moves plays the device."""
import os
import re

import pytest

torch = pytest.importorskip("torch")

from .. import rowpair


class OnDevice:
    """A tensor stand-in that says it is on the GPU: ``is_cuda`` True, ``.detach().to('cpu')`` hands back the real CPU tensor and records the move."""

    def __init__(self, t, moves, name):
        self.t, self.moves, self.name = t, moves, name
        self.is_cuda = True

    def dim(self):
        return self.t.dim()

    @property
    def shape(self):
        return self.t.shape

    def detach(self):
        return self

    def to(self, *a, **k):
        self.moves.append(self.name); return self.t


def _feats(N, moves, S=4, T=2):
    dev = lambda name, *shape, dtype=torch.float32: OnDevice(torch.zeros(*shape, dtype=dtype), moves, name)   # noqa: E731
    return {
        "token_pad_mask": torch.ones(1, N),
        # per-token / per-template / MSA features the model reads ON THE DEVICE — at N = 33 each has two dims equal to N
        "res_type": dev("res_type", 1, N, 33), "profile": dev("profile", 1, N, 33), "msa": dev("msa", 1, S, N, 33), "template_restype": dev("template_restype", 1, T, N, 33),
        "deletion_mean": dev("deletion_mean", 1, N), "template_cb": dev("template_cb", 1, T, N, 3),
        # the featurizer's token-pair planes (rowpair.PAIR_PLANES): host-resident, rows to the device per block
        "token_bonds": dev("token_bonds", 1, N, N, 1), "type_bonds": dev("type_bonds", 1, N, N, dtype=torch.int64),
        "contact_conditioning": dev("contact_conditioning", 1, N, N, 6), "contact_threshold": dev("contact_threshold", 1, N, N),
        # the distogram loss's target (rowpair.PARK_UNREAD): emitted at inference, read by nothing there — parked on the host whole, by name
        "disto_target": dev("disto_target", 1, N, N, 1, 64),
        # a pair-SHAPED tensor the featurizer does not emit at inference: named in the census, left where it is
        "mystery_pair": dev("mystery_pair", 1, N, N, 2),
    }


@pytest.mark.parametrize("N", [33, 34, 6])
def test_only_the_named_pair_planes_leave_the_device(N):
    moves = []; feats = _feats(N, moves); before = dict(feats)
    rowpair._STATE["pair_shaped_on_device"] = set(); rowpair._STATE["parked_unread"] = set(); c0 = int(rowpair._STATE["calls"]["pair_planes_to_host"])
    host = rowpair._pair_planes_to_host(feats)
    assert set(host) == set(rowpair.PAIR_PLANES) and sorted(moves) == sorted(set(rowpair.PAIR_PLANES) | set(rowpair.PARK_UNREAD)), moves
    for k in rowpair.PAIR_PLANES:
        assert isinstance(feats[k], torch.Tensor) and not feats[k].is_cuda and host[k] is feats[k], k     # moved to the host, feats holds the host plane
    assert rowpair.PARK_UNREAD == ("disto_target",)
    for k in rowpair.PARK_UNREAD:                                                                        # parked on the host whole (256·N² B off every rank's device), not a row source
        assert isinstance(feats[k], torch.Tensor) and not feats[k].is_cuda and k not in host and tuple(feats[k].shape) == (1, N, N, 1, 64), k
    for k in ("res_type", "profile", "msa", "template_restype", "deletion_mean", "template_cb", "mystery_pair"):
        assert feats[k] is before[k] and feats[k].is_cuda, f"{k} must stay on the device (the model reads it there)"
    assert int(rowpair._STATE["calls"]["pair_planes_to_host"]) == c0 + len(rowpair.PAIR_PLANES) + len(rowpair.PARK_UNREAD)
    named = rowpair._STATE["pair_shaped_on_device"]
    assert "mystery_pair" in named and not (named & (set(rowpair.PAIR_PLANES) | set(rowpair.PARK_UNREAD))) and rowpair._STATE["parked_unread"] == {"disto_target"}
    if N == 33:
        assert {"res_type", "profile", "msa", "template_restype"} <= named, "at N = 33 the one-hot features are pair-shaped: named, never moved"
    if N == 34:
        assert named == {"mystery_pair"}
    rep_ = rowpair.report()
    assert rep_["parked_unread"] == ["disto_target"] and "mystery_pair" in rep_["pair_shaped_on_device"] and "offloaded_planes" not in rep_


def test_host_planes_already_on_the_host_are_kept_as_is():
    N = 33; feats = {"token_pad_mask": torch.ones(1, N), "res_type": torch.zeros(1, N, 33), "token_bonds": torch.zeros(1, N, N, 1), "type_bonds": torch.zeros(1, N, N, dtype=torch.int64)}
    ids = {k: id(v) for k, v in feats.items()}
    host = rowpair._pair_planes_to_host(feats)
    assert set(host) == {"token_bonds", "type_bonds"} and all(id(feats[k]) == ids[k] for k in feats), "CPU tensors: nothing copied, nothing moved"


def test_the_pair_plane_offload_is_a_statement_of_the_n_gpu_gt_1_line_only():
    """n_gpu = 1 never consults the host-resident set: _pair_planes_to_host is called from _trunk_sharded alone (the row-sharded trunk the
    adapter installs only at BOLTZ_TP > 1 — apply() refuses at n_gpu = 1)."""
    src = open(os.path.join(os.path.dirname(rowpair.__file__), "rowpair.py")).read()
    calls = [m.start() for m in re.finditer(r"_pair_planes_to_host\(", src)]
    defs = [m.start() for m in re.finditer(r"def _pair_planes_to_host\(", src)]
    assert len(defs) == 1 and len(calls) == 2, "one definition, one call site"
    call = [c for c in calls if c - 4 != defs[0]][0]
    enclosing = re.findall(r"^def (\w+)\(", src[:call], re.M)[-1]
    assert enclosing == "_trunk_sharded"
    for other in ("worker.py", "stack.py", "big.py", "pairblock.py", "transition.py", "rowpair_heads.py", "rowpair_msa.py"):
        p = os.path.join(os.path.dirname(rowpair.__file__), other)
        assert "_pair_planes_to_host" not in open(p).read(), other
    os.environ.pop(rowpair.ENV_P, None)
    if not rowpair._STATE["installed"]:
        with pytest.raises(rowpair.Refused):
            rowpair.apply()
