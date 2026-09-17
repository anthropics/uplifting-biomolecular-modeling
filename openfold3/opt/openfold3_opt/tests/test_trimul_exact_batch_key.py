"""`trimul_exact` (cells/trimul_exact.py, the exact line) keys a TriMul call class BY ITS LEADING BATCH EXTENT;
The confidence head runs its pairformer over the S = 5 diffusion samples as ONE batch below `per_sample_token_cutoff`
(`[5, N, N, 128]`); the trunk calls `[1, N, N, 128]`.  Every exact-class row's byte vouch and the kit's first-call `torch.equal` proof are batch-1
evidence, and the library op's batched call is not 5 calls of one sample byte for byte (identity item entity_protein_rna_1urn_u1a: tmk3_exact
served the head's calls on the trunk's proof -> pLDDT / PAE off stock, coordinates bitwise).  Rules under test: the class key / proof unit / census
cell of a batched call carry `|B<b>`; the kit's planning select() states `batch=` and the pinned core answers the stock op BY NAME for a batched
layout no cell vouches (`exact_batch_unvouched`) -> the line's own statement serves those calls, counted `batched_cell:<row>`; batch 1 is spelt and
decided exactly as before.  No CUDA: pure table resolution against the pinned core."""
import re

from openfold3_opt.cells import trimul_exact as TE, trimul_form as F
from opt_core.kernels import trimul as KT
import opt_core

OF3 = "H100:2.10.0+cu128/3.6.0/cueq0.10.0"          # this kit's image stack word (KT.stack_word() form) -- a column of the table
CELL = "9.0|bf16|C128|H128|N<=256|%s|fwd"          # the trunk's / the head's cell for a 101..256-token item (c_z 128, bf16)


def _plan(N, B, direction="outgoing", prec="bf16", statement="library", use_form=False, stack=OF3):
    return TE.plan_class(KT, F, cc=(9, 0), prec=prec, C=128, D=128, N=N, B=B, direction=direction, statement=statement, stack=stack, use_form=use_form)


def test_batch_extent_is_the_product_of_the_leading_dims():
    assert TE.batch_extent((118, 118, 128)) == 1                  # a 3-d pair tensor
    assert TE.batch_extent((1, 118, 118, 128)) == 1               # the trunk / MSA / template pair stacks
    assert TE.batch_extent((5, 118, 118, 128)) == 5               # the confidence head's batched samples (z4 = z.reshape(-1, N, N, c))
    assert TE.batch_extent((1, 5, 118, 118, 128)) == 5            # ... before the reshape
    assert TE.batch_extent((2, 3, 64, 64, 128)) == 6


def test_class_key_proof_unit_and_census_cell_carry_the_extent_above_batch_1():
    k = CELL % "out"
    assert TE.BATCH_TAG == "|B"
    assert TE.batch_key(k, 1) == k                                          # batch 1: the key as before, byte for byte
    assert TE.batch_key(k, 5) == k + "|B5"
    assert TE.batch_key(F.class_key(k, F.MODULE_TAG), 5) == k + "|module|B5"
    # the proof unit (cells/trimul_form.proof_unit) is taken per layout: a batch-1 proof never covers the batched class
    assert F.proof_unit(TE.batch_key(k, 5), "library", 118) == k + "|B5" != F.proof_unit(k, "library", 118)
    assert F.proof_unit(TE.batch_key(k + "|module", 5), F.MODULE_TAG, 118) == k + "|module|B5@118"


def test_the_pinned_core_reads_the_batch_extent_under_the_exact_word():
    assert tuple(int(x) for x in opt_core.__version__.split(".")[:4]) >= (0, 5, 212, 5)
    assert "batch" in KT.select.__code__.co_varnames and hasattr(KT, "exact_batch_vouched") and KT.BATCH_NOTE == "exact_batch_unvouched"


def test_trunk_class_serves_the_kernel_row_head_class_is_the_lines_statement():
    for d, dw in (("outgoing", "out"), ("incoming", "in")):
        hit, cell = _plan(118, 1, d)
        assert hit == ("row", "tmk3_exact", CELL % dw, "tmk3_exact"), hit       # the trunk ([1,118,118,128]): the vouched kernel row, bound by its row word
        assert cell == (CELL % dw, "tmk3_exact")
        hit5, cell5 = _plan(118, 5, d)
        assert hit5 == ("line", "batched_cell:cueq"), hit5                     # the head ([5,118,118,128]): the core names the library op -> the line's own statement
        assert cell5 == ((CELL % dw) + "|B5", "cueq")                          # the census names the batched class beside the batch-1 class
        sel = KT.select(9.0, "bf16", 128, 128, 118, d, word="exact", stack=OF3, has_cueq=True, batch=5)
        assert sel.row == "cueq" and "exact_batch_unvouched(tmk3_exact->cueq:B5)" in sel.reason, sel.reason


def test_sizes_the_table_already_gives_the_line_are_unchanged_but_named():
    assert _plan(60, 1)[0] == ("line", "cell:cueq")                            # <= 100 tokens: the library op on every tier (its own torch path)
    assert _plan(60, 5)[0] == ("line", "batched_cell:cueq")
    assert _plan(700, 1)[0] == ("line", "cell:cueq")                           # >= 513 tokens on c_z 128: the library op by cell
    assert _plan(700, 5)[0] == ("line", "batched_cell:cueq")
    for N in (101, 256, 257, 400, 512):                                        # 101..512: tmk3_exact at batch 1, the line at batch 5
        assert _plan(N, 1)[0][:2] == ("row", "tmk3_exact"), (N, _plan(N, 1))
        assert _plan(N, 5)[0] == ("line", "batched_cell:cueq"), (N, _plan(N, 5))
        assert _plan(N, 2)[0] == ("line", "batched_cell:cueq")


def test_tf32_head_class_and_module_statement_classes():
    hit, cell = _plan(118, 5, prec="tf32")                                     # an fp32-z / tf32 confidence head: the line, by the batch rule -- no failed proof needed
    assert hit == ("line", "batched_cell:cueq") and cell == ("9.0|tf32|C128|H128|N<=256|out|fwd|B5", "cueq"), (hit, cell)
    assert _plan(118, 5, statement=F.MODULE_TAG, use_form=False) == (("line", "form_off"), None)   # trimul_form off (as the lines ship): the module's statement is the line's
    h, c = _plan(118, 5, statement=F.MODULE_TAG, use_form=True)               # trimul_form on: the form row where its cell vouches N (proven per class|B5@N by trimul_exact), else declined by name
    assert h[0] in ("row", "line") and (c is None or c[0].endswith("|B5")), (h, c)
    if h[0] == "row":
        assert h[1] == F.ROW and h[3] == F.WORD


def test_census_fields_name_the_batched_class_and_reason():
    saved = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in TE.STATE.items()}
    try:
        TE.STATE["cells"].clear(); TE.STATE["line"].clear(); TE.STATE["served"].clear()
        TE.STATE["cells"][CELL % "out"] = "tmk3_exact"; TE.STATE["cells"][(CELL % "out") + "|B5"] = "cueq"
        TE.STATE["line"]["batched_cell:cueq"] = 8; TE.STATE["served"]["tmk3_exact"] = 1054
        f = TE.fields()
        assert "cells=9.0|bf16|C128|H128|N<=256|out|fwd:tmk3_exact|9.0|bf16|C128|H128|N<=256|out|fwd|B5:cueq" in f, f
        assert "line=batched_cell:cueq:8" in f and "rows=tmk3_exact:1054" in f, f
        assert not re.search(r"(^|[^_a-z',=])refus(ed|ing)", f)                # no `refused` marker word on the census line
    finally:
        TE.STATE.clear(); TE.STATE.update(saved)


def test_the_serving_closure_states_the_extent():
    import inspect
    src = inspect.getsource(TE.install)
    assert "key, cc, prec, C, D, N, B = _class(z, mod, direction, statement)" in src
    assert "_select(key, cc, prec, C, D, N, direction, statement, B)" in src
    assert "batch=int(B)" in inspect.getsource(TE.plan_class)
