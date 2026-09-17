"""`--backend shipped`: the stock route makes NO model call — the model exactly as from_pretrained loads it (stock_fold.BACKENDS, the one
table); the SETTINGS line names the switches as loaded (stock_fold.as_loaded_state)."""
import sys
import types
import unittest

from esmfold2_opt import stock_fold


class _Rec:
    """A recording stand-in for upstream's model class: counts every setter call."""
    calls = []

    def __init__(self): self.config = types.SimpleNamespace(lm_encoder=None)
    @classmethod
    def from_pretrained(cls, repo, **kw): return cls()
    def to(self, device): return self
    def eval(self): return self
    def set_kernel_backend(self, b): _Rec.calls.append(("set_kernel_backend", b))
    def set_chunk_size(self, c): _Rec.calls.append(("set_chunk_size", c))


def _install_fake_upstream():
    names = ["transformers", "transformers.models", "transformers.models.esmfold2", "transformers.models.esmfold2.modeling_esmfold2"]
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules[n] = types.ModuleType(n)
    sys.modules[names[-1]].ESMFold2Model = _Rec
    return saved


def _restore(saved):
    for n, m in saved.items():
        if m is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = m


class BackendShipped(unittest.TestCase):
    def test_shipped_is_in_the_one_table_and_makes_no_call(self):
        self.assertIn("shipped", stock_fold.BACKENDS); self.assertEqual(stock_fold.BACKENDS["shipped"], [])
        self.assertEqual(stock_fold.model_calls(None, "shipped"), [])
        self.assertEqual(stock_fold.model_calls(None, None), [])                                                                # bare --mode off: the model as loaded
        self.assertEqual(stock_fold.model_calls(None, "fused"), [("set_kernel_backend", "fused"), ("set_chunk_size", None)])  # unchanged

    def test_load_model_under_shipped_issues_zero_setter_calls(self):
        saved = _install_fake_upstream()
        try:
            _Rec.calls = []
            stock_fold.load_model("EvolutionaryScale/esmfold2-any", device="cpu", settings=None, det_level=0, backend="shipped")
            self.assertEqual(_Rec.calls, [])
        finally:
            _restore(saved)

    def test_as_loaded_state_reads_pair_and_opm_chunks_apart(self):
        class OuterProductMean:
            def __init__(self): self._chunk_size = None
        class PairBlock:
            def __init__(self): self._chunk_size = 64; self._kernel_backend = None
        class Model:
            def __init__(self): self.blocks = [PairBlock(), PairBlock()]; self.opm = [OuterProductMean()]
            def modules(self): return [self] + self.blocks + self.opm
        self.assertEqual(stock_fold.as_loaded_state(Model()), {"kernel_backend": "None", "chunk": "64", "opm_chunk": "None"})
        self.assertEqual(stock_fold.as_loaded_state(object()), {"kernel_backend": "unread", "chunk": "unread", "opm_chunk": "unread"})

    def test_settings_line_words(self):
        line = stock_fold.settings_line("shipped", [], {"kernel_backend": "None", "chunk": "64", "opm_chunk": "None"})
        self.assertEqual(line, "SETTINGS backend=shipped model_calls=none kernel_backend=None chunk=64 opm_chunk=None")
        line = stock_fold.settings_line(None, [], {"kernel_backend": "None", "chunk": "64", "opm_chunk": "None"})
        self.assertIn("backend=shipped model_calls=none", line)                                                             # bare --mode off: the model as loaded
        line = stock_fold.settings_line("fused", [("set_kernel_backend", "fused"), ("set_chunk_size", None)], {"kernel_backend": "fused", "chunk": "None", "opm_chunk": "None"})
        self.assertIn("backend=fused model_calls=set_kernel_backend('fused') set_chunk_size(None)", line)


if __name__ == "__main__":
    unittest.main()
