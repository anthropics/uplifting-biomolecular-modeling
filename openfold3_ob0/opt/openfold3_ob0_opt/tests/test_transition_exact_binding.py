"""The `transition_exact` lever's binding contract (CPU, no GPU): the kit names the core's transition provider by the tier word `exact` under
this engine's form, carries no kernel / tile table / token floor of its own, routes a stock winner or a refusal to the module's own statement
by name, and — on the kit's stacks — the provider's table serves a kernel row at the representative pair-stack sizes."""
import types

import pytest

from openfold3_ob0_opt import of3_transition as T

STACKS = {"9.0": "H100:torch2.10.0+cu128/3.6.0", "8.0": "A100:torch2.10.0+cu128/3.6.0"}   # the kit's stacks (environment/Dockerfile: torch 2.10.0+cu128, triton 3.6.0)


def test_words_and_no_kit_tables():
    assert T.WORD == "exact" and T.FORM == "swiglu"
    for name in ("FLOOR_N_BY_CC", "floor_word", "ENV_MIN_N", "_of3_transition_kernel", "_launch", "_cell_row", "triton"):
        assert not hasattr(T, name), name
    src = open(T.__file__).read()
    assert "triton.jit" not in src and "import triton" not in src


class _Sel(types.SimpleNamespace):
    pass


class _TP:
    STOCK_ROWS = ("torch_swiglu", "engine_module", "compile")

    class Refusal(Exception):
        def __init__(self, kind):
            super().__init__(kind + " (fallback=torch_swiglu)"); self.kind = kind

    def __init__(self, answer):
        self.answer = answer; self.calls = 0

    def select(self, word, **kw):
        self.calls += 1
        assert word == "exact" and kw["form"] == T.FORM and kw["ln_given"] is True and kw["family"] == "pair" and kw["dtype"] == "bf16"
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _fresh():
    T._WINNERS.clear(); T.STATE["vouch"] = None


def test_winner_routes_stock_rows_and_refusals_to_the_module_by_name():
    _fresh()
    tp = _TP(_Sel(row="torch_swiglu", variant=None, stack=STACKS["9.0"], cell_key="k"))
    with pytest.raises(T.Fallback) as e:
        T._winner(tp, 128, 512, 200, "cuda:0", "eager", False)
    assert e.value.reason == "exact_winner:torch_swiglu:c128"
    with pytest.raises(T.Fallback):                                   # memoised: the provider is asked once per key
        T._winner(tp, 128, 512, 200, "cuda:0", "eager", False)
    assert tp.calls == 1
    _fresh()
    tp = _TP(_TP.Refusal("exact_vouch_below_64_tokens_on_A100:torch2.10.0+cu128/3.6.0"))
    with pytest.raises(T.Fallback) as e:
        T._winner(tp, 128, 512, 20, "cuda:0", "eager", False)
    assert e.value.reason == "exact:exact_vouch_below_64_tokens_on_A100:torch2.10.0+cu128/3.6.0"
    _fresh()
    tp = _TP(_Sel(row="v1", variant=None, stack=STACKS["9.0"], cell_key="9.0|bf16|pair_c128_n4|N<=400|eager|fwd"))
    assert T._winner(tp, 128, 512, 400, "cuda:0", "eager", False) is None and T.STATE["vouch"] == STACKS["9.0"]
    _fresh()


def test_census_words():
    line = T.census()
    assert line.startswith(" word=exact form=swiglu core=") and " vouch=" in line and " rows=" in line and " cells=" in line and "min_n=" not in line and "kernel=" not in line


@pytest.mark.parametrize("cc", ["9.0", "8.0"])
@pytest.mark.parametrize("n_tokens", [400, 800, 1200])
@pytest.mark.parametrize("hidden", [512, 256])
def test_provider_serves_a_kernel_row_at_the_representative_sizes_on_the_kits_stacks(cc, n_tokens, hidden):
    TP = pytest.importorskip("opt_core.kernels.transition")
    if T.FORM not in getattr(TP, "FORMS", ()):
        pytest.skip("this core has no form words")
    sel = TP.select("exact", c=128, hidden=hidden, n_tokens=n_tokens, dtype="bf16", family="pair", cc=cc, stack=STACKS[cc], ln_given=True, form=T.FORM)
    assert sel.row not in TP.STOCK_ROWS, (cc, n_tokens, hidden, sel.row)
    assert sel.row == "v1" and sel.variant == None
