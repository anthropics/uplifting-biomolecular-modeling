"""The tp line's relative-position rows (``tp_rowpair.trunk.relpos_rows``: the core's clipped offset rows + the adapter's cyclic-chain wrap +
one-hot) equal upstream's ``relpos.relpos_complex`` (openfold3 0.5.0, ``core/utils/relpos.py``) row slab for row slab — on a complex that MIXES a
cyclic chain and a linear chain (the wrap applies inside the cyclic chain only), with row windows that cross the chain boundary, and on the
two degenerate forms (no cyclic token; the ``cyclic_mask`` feature absent). CPU, no process group (the statements are per-row pure functions)."""
import importlib.util

import pytest

def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):                        # a finder that refuses the name outright
        return False


HAVE_TORCH = _importable("torch")
HAVE_OF3 = HAVE_TORCH and _importable("openfold3")
HAVE_CORE = _importable("opt_core")
needs_stack = pytest.mark.skipif(not (HAVE_OF3 and HAVE_CORE), reason="needs torch + openfold3 + opt_core (the kit's stack)")

R_IDX, R_CHAIN = 32, 2                       # model_config: max_relative_idx 32, max_relative_chain 2


def make_batch(cyclic="A", with_feature=True):
    """Two protein chains: A (11 residues, residue_index 1..11) and B (9 residues), one token per residue plus two extra tokens on A's residue 4
    (an atomized residue: same residue_index, increasing token_index) — 22 tokens. ``cyclic``: which chain is cyclic ("A", "B", "" for none)."""
    import torch
    res_a = [1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 10, 11]
    res_b = list(range(1, 10))
    n_a, n_b = len(res_a), len(res_b)
    residue_index = torch.tensor(res_a + res_b)
    token_index = torch.arange(1, n_a + n_b + 1)
    asym_id = torch.tensor([1] * n_a + [2] * n_b)
    entity_id = torch.tensor([1] * n_a + [2] * n_b)
    sym_id = torch.tensor([1] * n_a + [1] * n_b)
    batch = {"residue_index": residue_index, "token_index": token_index, "asym_id": asym_id, "entity_id": entity_id, "sym_id": sym_id}
    if with_feature:
        batch["cyclic_mask"] = (asym_id == {"A": 1, "B": 2}.get(cyclic, -1))
    return {k: v[None] for k, v in batch.items()}        # batch dims (1,)


@needs_stack
@pytest.mark.parametrize("cyclic,with_feature", [("A", True), ("B", True), ("", True), ("", False)])
def test_relpos_rows_equal_upstream(cyclic, with_feature):
    import torch
    from openfold3.core.utils.relpos import relpos_complex
    from openfold3_ob0_opt.tp_rowpair import trunk as T
    batch = make_batch(cyclic, with_feature)
    ref = relpos_complex(batch, R_IDX, R_CHAIN)                       # [1, N, N, C]
    N = ref.shape[-2]
    windows = [(0, N), (0, 5), (5, 13), (11, 16), (13, N), (3, 4), (N - 1, N)]     # whole, inside A, across A's atomized residue, ACROSS the A|B boundary, inside B, single rows
    for g0, g1 in windows:
        rows = T.relpos_rows(batch, g0, g1, R_IDX, R_CHAIN, dtype=ref.dtype)
        assert rows.shape == (1, g1 - g0, N, ref.shape[-1]), (rows.shape, (g0, g1))
        assert torch.equal(rows, ref[:, g0:g1]), f"relpos rows {g0}:{g1} differ from relpos_complex (cyclic={cyclic!r}, feature={with_feature}): max |d| {float((rows - ref[:, g0:g1]).abs().max())}"
    if cyclic and with_feature:                                        # the wrap is real on this input: the cyclic chain's far pairs differ from the linear offsets
        lin = relpos_complex({k: v for k, v in batch.items() if k != "cyclic_mask"}, R_IDX, R_CHAIN)
        assert not torch.equal(lin, ref), "test input has no cyclic effect (the wrap would be untested)"


@needs_stack
def test_cyclic_bins_rows_refuses_foreign_lead_dims():
    import torch
    from openfold3_ob0_opt.tp_rowpair import trunk as T
    batch = make_batch("A", True)
    N = batch["asym_id"].shape[-1]
    d = torch.zeros(1, 4, N, dtype=torch.int64)
    with pytest.raises(T.TrunkRefused):
        T.cyclic_bins_rows_(d, batch["residue_index"], batch["cyclic_mask"].expand(3, N), batch["asym_id"], 0, 4, R_IDX, torch.ones(1, 4, N, dtype=torch.bool))
