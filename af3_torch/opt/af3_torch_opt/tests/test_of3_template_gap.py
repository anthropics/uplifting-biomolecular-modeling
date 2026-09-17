"""The OpenFold3 empty-template restype (the reference fork's AF3_WEIGHT_PORT patch 04) in the kit's template embedder,
xfold/nn/template.py `template_restype_for_one_hot` inside `SingleTemplateEmbedding.construct_input`: under xfold.of3.OF3 an all-empty
(zero-padded) template slot embeds GAP one-hots — index 21 of the 31 classes of POLYMER_TYPES_WITH_UNKNOWN_AND_GAP — on every token pair
in place of the featuriser's padding value 0 (= ALA); a slot with any atom present, and every slot under the AlphaFold 3 layout, embeds
its restypes as featurised. On the REAL xfold modules, CPU fp32, random-initialised parameters (the equivalences are between inputs to
the SAME module, no weights). Needs torch + einops + triton (xfold.fastnn imports triton; its LayerNorm / attention / GLU routes here are
the torch implementations, xfold/fastnn/config.py); skipped otherwise — the package's other tests import neither."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("triton")


def _triton_import_stub():
    """On a box WITHOUT a GPU driver ``xfold.fastnn`` cannot import as is: a module-level ``@triton.autotune`` probes the active driver at
    import time (``triton.jit`` does not). The routes exercised here are fastnn's TORCH implementations, so on such a box the real triton is
    imported and ONLY ``triton.autotune`` / ``triton.heuristics`` are replaced by pass-through decorators (never done when CUDA is present)."""
    if torch.cuda.is_available():
        return "real"
    import triton
    triton.autotune = lambda *a, **kw: (lambda fn: fn)
    triton.heuristics = lambda *a, **kw: (lambda fn: fn)
    return "autotune_noop"


TRITON = _triton_import_stub()

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))          # the tree root: af3_torch/
KIT = os.path.join(HOME, "opt", "forward", "af3t", "af3_torch")
if KIT not in sys.path:
    sys.path.insert(0, KIT)

from xfold import features, of3                       # noqa: E402
from xfold.constants import residue_names             # noqa: E402
from xfold.nn import template as TPL                  # noqa: E402

N = 24              # tokens: two chains of 12
A = 24              # dense atom slots per token
C_PAIR = 128
GAP_IDX = 21


@pytest.fixture(scope="module")
def mods():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    single = TPL.SingleTemplateEmbedding()
    full = TPL.TemplateEmbedding(pair_channel=C_PAIR)
    for m in (single, full):
        m.eval()
        for prm in m.parameters():                     # LayerNorm inits are degenerate (ones/zeros): randomise everything (small)
            with torch.no_grad():
                prm.copy_(torch.randn_like(prm) * 0.1)
    g = torch.Generator().manual_seed(1)
    asym = torch.tensor([1] * (N // 2) + [2] * (N - N // 2))
    return dict(single=single, full=full,
                query=torch.randn(N, N, C_PAIR, generator=g),
                multichain_mask_2d=(asym[:, None] == asym[None, :]).to(torch.float32),
                padding_mask_2d=torch.ones(N, N),
                live_pos=torch.randn(N, A, 3, generator=g) * 5.0,
                live_aatype=torch.randint(0, 20, (N,), generator=g).to(torch.int32))


def _slot(aatype, atoms_present, pos=None):
    """One template slot's tensors (fresh copies: construct_input scales atom_positions in place). atoms_present: bool [N, A]."""
    mask = atoms_present.clone().to(torch.bool)
    positions = (pos.clone() if pos is not None else torch.zeros(N, A, 3)) * mask[..., None]
    return dict(aatype=aatype.clone().to(torch.int32), atom_positions=positions, atom_mask=mask)


def _templates(slots):
    return features.Templates(aatype=torch.stack([s["aatype"] for s in slots]),
                              atom_positions=torch.stack([s["atom_positions"] for s in slots]),
                              atom_mask=torch.stack([s["atom_mask"] for s in slots]))


def _empty(aatype_value=0):
    return _slot(torch.full((N,), aatype_value, dtype=torch.int32), torch.zeros(N, A, dtype=torch.bool))


def _live(mods, aatype=None):
    present = torch.zeros(N, A, dtype=torch.bool); present[:, :5] = True          # backbone + CB present on every token
    return _slot(mods["live_aatype"] if aatype is None else aatype, present, mods["live_pos"])


def _construct(mods, slot):
    with torch.no_grad():
        return mods["single"].construct_input(mods["query"].clone(), _templates([slot])[0], mods["multichain_mask_2d"])


def _embed(mods, slots):
    with torch.no_grad():
        return mods["full"](mods["query"].clone(), _templates(slots), mods["padding_mask_2d"], mods["multichain_mask_2d"])


def test_gap_index_is_the_polymer_one_hot_gap_class():
    """The fork's gap index (the protein one-letter alphabet's '-') is class 21 of the 31-class polymer one-hot the features use; 0 is ALA."""
    assert residue_names.PROTEIN_TYPES_ONE_LETTER_WITH_UNKNOWN_AND_GAP.index('-') == GAP_IDX
    assert residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP[GAP_IDX] == residue_names.GAP
    assert residue_names.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP[0] == "ALA"
    assert residue_names.POLYMER_TYPES_NUM_WITH_UNKNOWN_AND_GAP == 31


def test_restype_helper_whole_slot_predicate(monkeypatch):
    aatype = torch.randint(0, 31, (N,), generator=torch.Generator().manual_seed(2)).to(torch.int32)
    empty = torch.zeros(N, A, dtype=torch.bool)
    one_atom = empty.clone(); one_atom[7, 3] = True
    monkeypatch.setattr(of3, "OF3", True)
    got = TPL.template_restype_for_one_hot(aatype, empty)
    assert got.dtype == aatype.dtype and got.shape == aatype.shape and bool((got == GAP_IDX).all())     # empty slot: GAP on EVERY token
    assert torch.equal(TPL.template_restype_for_one_hot(aatype, one_atom), aatype)                      # any atom anywhere: the slot's own restypes
    assert torch.equal(TPL.template_restype_for_one_hot(aatype, empty.to(torch.float32)), got)          # a float mask reads the same
    got64 = TPL.template_restype_for_one_hot(aatype.to(torch.int64), empty)
    assert got64.dtype == torch.int64 and torch.equal(got64, got.to(torch.int64))                        # int64 restypes keep their dtype
    monkeypatch.setattr(of3, "OF3", False)
    assert torch.equal(TPL.template_restype_for_one_hot(aatype, empty), aatype)                         # AlphaFold 3 layout: never


def test_construct_input_embeds_gap_for_an_empty_slot(mods, monkeypatch):
    """Regression on the statement itself: under OF3 the featuriser's zero-padded slot (restype 0, no atoms) embeds exactly as an explicit
    GAP slot does; on the unpatched statement the first equality fails (ALA vs GAP one-hots through template_pair_embedding_2 / _3)."""
    monkeypatch.setattr(of3, "OF3", True)
    padded = _construct(mods, _empty(0))
    explicit_gap = _construct(mods, _empty(GAP_IDX))
    assert torch.equal(padded, explicit_gap)
    monkeypatch.setattr(of3, "OF3", False)
    ala = _construct(mods, _empty(0))
    assert torch.equal(_construct(mods, _empty(GAP_IDX)), explicit_gap)          # the GAP slot itself embeds the same under either layout
    assert not torch.allclose(ala, explicit_gap) and not torch.allclose(padded, ala)   # and the padding slot no longer embeds as ALA under OF3
    assert torch.isfinite(padded).all() and padded.shape == (N, N, mods["single"].num_channels)


def test_a_slot_with_atoms_keeps_its_restypes(mods, monkeypatch):
    """A template that is present (atoms on its tokens) embeds its featurised restypes under OF3 — restype 0 included: no substitution."""
    zeros = torch.zeros(N, dtype=torch.int32)
    monkeypatch.setattr(of3, "OF3", True)
    on = _construct(mods, _live(mods, zeros)); on_rand = _construct(mods, _live(mods))
    monkeypatch.setattr(of3, "OF3", False)
    off = _construct(mods, _live(mods, zeros)); off_rand = _construct(mods, _live(mods))
    assert torch.equal(on, off) and torch.equal(on_rand, off_rand)
    partial = torch.zeros(N, A, dtype=torch.bool); partial[3, 1] = True         # one atom on one token keeps the WHOLE slot as featurised
    monkeypatch.setattr(of3, "OF3", True)
    p_on = _construct(mods, _slot(zeros, partial, mods["live_pos"]))
    monkeypatch.setattr(of3, "OF3", False)
    assert torch.equal(p_on, _construct(mods, _slot(zeros, partial, mods["live_pos"])))


def _stray(aatype_value=0):
    """A synthetic control slot (not a featuriser output): ONE atom flag at dense index 10 on token 5, positions zero, nothing else. Not
    empty, so its restypes stay as featurised; yet index 10 is a column neither the pseudo-beta lookup (CB = 4, CA = 1 for GLY, 0 for GAP)
    nor the backbone-frame lookup (N, CA, C = 0, 1, 2; 0, 0, 0 for GAP) reads, so every geometric feature is zero exactly as for an empty
    slot — it differs from the empty slot ONLY through the restype one-hots, which is the statement under test."""
    present = torch.zeros(N, A, dtype=torch.bool); present[5, 10] = True
    return _slot(torch.full((N,), aatype_value, dtype=torch.int32), present)


def test_template_embedding_over_a_partially_empty_stack(mods, monkeypatch):
    """TemplateEmbedding.forward sums every slot (present or not): under OF3 a 4-slot stack with 2 templates present and 2 zero-padded
    embeds exactly as the same stack with the padded slots' restypes set to GAP, and NOT as the stack whose padded slots keep restype 0
    (the stray-atom slots: ALA one-hots, identical geometry). Every comparison is under one layout: the template stack's pairformer blocks
    read xfold.of3.OF3 themselves (nn/attention.py), so outputs across layouts differ for reasons outside this statement."""
    stack = lambda mk: [_live(mods), mk(), _live(mods, torch.flip(mods["live_aatype"], [0])), mk()]
    monkeypatch.setattr(of3, "OF3", True)
    out = _embed(mods, stack(lambda: _empty(0)))
    assert out.shape == (N, N, C_PAIR) and torch.isfinite(out).all()
    assert torch.equal(out, _embed(mods, stack(lambda: _empty(GAP_IDX))))         # padding embeds as GAP
    assert torch.equal(_embed(mods, stack(lambda: _stray(GAP_IDX))), out)         # control: the stray-atom slot's geometry IS the empty slot's
    assert not torch.allclose(out, _embed(mods, stack(lambda: _stray(0))))        # ... so this difference is ALA vs GAP one-hots alone
    monkeypatch.setattr(of3, "OF3", False)                                        # the AlphaFold 3 layout: padding embeds as restype 0
    af3 = _embed(mods, stack(lambda: _empty(0)))
    assert torch.equal(af3, _embed(mods, stack(lambda: _stray(0)))) and not torch.allclose(af3, _embed(mods, stack(lambda: _empty(GAP_IDX))))
