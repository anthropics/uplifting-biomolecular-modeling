"""The fused triangle-attention block's EPILOGUE piece follows the provider's pair_fused cell — the kit reports, never decides: the LEVER
token `epilogue=` reads opt_core.attn.pair_fused.SERVED_PIECES (the provider's own per-call plan: where the cell for (c_z 128, H 4, D 32) on the
running capability is named off, the gate-transpose + GEMM statements piece serves BY NAME and is counted there)."""
import types

from atlasfold_opt.hooks import pair_cells as PC


class _Ledger:
    def __init__(self): self.facts = {}
    def set(self, k, v): self.facts[k] = v


def test_epilogue_word_names_the_statements_piece_when_the_cell_is_off():
    PF = types.SimpleNamespace(SERVED_PIECES={"epilogue=fpf>lnl(off:slower)@128x4x32": 1000, "epilogue=fpf>lnl(off:slower)@128x4x32|8.0": 184, "prologue=other": 5})
    assert PC.epilogue_word(PF) == "cell:off(slower)>lnl:1184"
    PF2 = types.SimpleNamespace(SERVED_PIECES={"epilogue=fpf>lnl(off:not-measured)@128x4x32": 2, "epilogue=fpf>lnl(off:slower)@128x4x32": 3})
    assert PC.epilogue_word(PF2) == "cell:off(not-measured+slower)>lnl:5"


def test_epilogue_word_is_fpf_when_the_fused_kernel_served_or_nothing_ran():
    assert PC.epilogue_word(types.SimpleNamespace(SERVED_PIECES={})) == "fpf"
    assert PC.epilogue_word(types.SimpleNamespace()) == "fpf"                       # an older provider without the census attribute: the fused kernel is all it has


def test_set_piece_words_puts_the_token_on_the_line_and_never_raises():
    L = _Ledger()
    PC.set_piece_words(L, types.SimpleNamespace(SERVED_PIECES={"epilogue=fpf>lnl(off:slower)@128x4x32": 7}))
    assert L.facts["epilogue"] == "cell:off(slower)>lnl:7"
    class Bad:
        @property
        def SERVED_PIECES(self): raise RuntimeError("boom")
    PC.set_piece_words(L, Bad())
    assert L.facts["epilogue"].startswith("unknown:")


def test_the_kit_carries_no_epilogue_verdict_of_its_own():
    """No kit table / floor / card row decides the epilogue: the block levers pass the provider its default plan (no `variant=` pin, no
    engage override) — the source of hooks/triatt_block(.py|_exact.py) names no epilogue variant."""
    import inspect
    from atlasfold_opt.hooks import triatt_block as TB, triatt_block_exact as TX
    for mod in (TB, TX):
        src = inspect.getsource(mod)
        assert "variant=" not in src.replace("variant=None", "") and "epilogue_v" not in src, mod.__name__
        assert "set_piece_words" in src
