"""`of3_offload.relpos_rows` (the O1 `input_rows` / `cond_once` levers' row-block relative-position features) against upstream 0.5.0's
`relpos_complex` — the integer / one-hot path, so the row blocks concatenated must EQUAL the full tensor exactly, cyclic-chain offsets
(upstream's `apply_cyclic_offsets`, new at 0.5.0) included (CPU; skipped without torch or the openfold3 wheel)."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openfold3", reason="the openfold3 wheel is not installed: relpos rows are checked against upstream's relpos_complex")

HOME = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OF3O = os.path.join(HOME, "opt", "forward", "offload", "of3o")
MAX_REL_IDX, MAX_REL_CHAIN = 32, 2


@pytest.fixture(scope="module")
def O():
    sys.path.insert(0, OF3O)
    try:
        import of3_offload
        yield of3_offload
    finally:
        sys.path.remove(OF3O)


def _batch(lead=(1,), cyclic_chain=None):
    """Three chains: A (9 tokens, entity 0), B (9 tokens, entity 0 — same entity as A, sym_id 1), a 5-token ligand-like chain C (entity 1) with
    two tokens sharing a residue (an atomized residue: token_index differs, residue_index repeats)."""
    res = list(range(9)) + list(range(9)) + [0, 1, 1, 2, 3]
    tok = list(range(23))
    asym = [0] * 9 + [1] * 9 + [2] * 5
    ent = [0] * 9 + [0] * 9 + [1] * 5
    sym = [0] * 9 + [1] * 9 + [0] * 5
    b = {"residue_index": torch.tensor(res), "token_index": torch.tensor(tok), "asym_id": torch.tensor(asym), "entity_id": torch.tensor(ent),
         "sym_id": torch.tensor(sym)}
    if cyclic_chain is not None:
        b["cyclic_mask"] = (b["asym_id"] == cyclic_chain)
    return {k: v.reshape(*lead, -1) for k, v in b.items()}


@pytest.mark.parametrize("lead", [(1,), (1, 1)])
@pytest.mark.parametrize("cyclic_chain", [None, 1])
@pytest.mark.parametrize("rows", [5, 9, 23])
def test_relpos_rows_equal_upstream_relpos_complex(O, lead, cyclic_chain, rows):
    from openfold3.core.utils.relpos import relpos_complex
    batch = _batch(lead, cyclic_chain)
    N = batch["asym_id"].shape[-1]
    ref = relpos_complex(batch=batch, max_relative_idx=MAX_REL_IDX, max_relative_chain=MAX_REL_CHAIN)
    blocks = [O.relpos_rows(batch, MAX_REL_IDX, MAX_REL_CHAIN, r0, min(N, r0 + rows)) for r0 in range(0, N, rows)]
    out = torch.cat(blocks, dim=-3)
    assert out.shape == ref.shape == (*lead, N, N, 2 * (2 * MAX_REL_IDX + 2) + 1 + 2 * MAX_REL_CHAIN + 2)
    assert torch.equal(out, ref)


def test_cyclic_offsets_change_the_features(O):
    """Sensitivity: with a cyclic chain the features differ from the linear ones inside that chain only."""
    from openfold3.core.utils.relpos import relpos_complex
    lin = relpos_complex(batch=_batch(), max_relative_idx=MAX_REL_IDX, max_relative_chain=MAX_REL_CHAIN)
    cyc = relpos_complex(batch=_batch(cyclic_chain=1), max_relative_idx=MAX_REL_IDX, max_relative_chain=MAX_REL_CHAIN)
    diff = (lin != cyc).any(dim=-1)[0]
    inside = torch.zeros(23, 23, dtype=torch.bool); inside[9:18, 9:18] = True
    assert diff[inside].any() and not diff[~inside].any()
