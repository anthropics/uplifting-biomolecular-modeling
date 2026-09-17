"""gen/specdec/core.py StateStack on a fake two-device-free recurrent model: snapshot / per-layer record / commit restore the slots' values in
place (the tensors stay the same objects), flattened or not; the multi-token hook steps a fake cascade once per position."""
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from evo2_opt.gen.specdec import core as C


class _Params:
    def __init__(self, inner):
        self.seqlen_offset = 0
        self.fir_state_dict = {}
        if inner:
            self.fir_inner_state_dict = {}
        self.state_dict = {}


class _Filter:
    def __init__(self, idx, inner_len):
        self.layer_idx = idx
        self.fir_inner_filter_length = inner_len


class _Block:
    def __init__(self, idx, inner_len):
        self.filter = _Filter(idx, inner_len)


class _SH:
    """blocks 0 (hcs: fir + inner fir), 1 (mha), 2 (hcl: fir + iir state)."""
    def __init__(self):
        self.blocks = [_Block(0, 7), _Block(1, None), _Block(2, None)]

    def block_idx_to_name(self, i):
        return {0: "hcs", 1: "mha", 2: "hcl"}[i]


def _ipd():
    ipd = {"hcs": _Params(True), "hcm": _Params(True), "hcl": _Params(False), "mha": _Params(False)}
    ipd["hcs"].fir_state_dict[0] = torch.zeros(1, 6, 2); ipd["hcs"].fir_inner_state_dict[0] = torch.zeros(1, 2, 6)
    ipd["hcl"].fir_state_dict[2] = torch.zeros(1, 6, 2); ipd["hcl"].state_dict[2] = torch.zeros(1, 2, 4)
    return ipd


@unittest.skipIf(torch is None, "torch is not installed in this interpreter")
class TestStateStack(unittest.TestCase):
    def _slots(self, ipd):
        return [ipd["hcs"].fir_state_dict[0], ipd["hcs"].fir_inner_state_dict[0], ipd["hcl"].fir_state_dict[2], ipd["hcl"].state_dict[2]]

    def _run(self, flatten):
        sh, ipd = _SH(), _ipd()
        st = C.StateStack(sh, ipd, depth=3)
        self.assertEqual(len(st.slots), 4)
        if flatten:
            st.flatten()
            self.assertEqual(st.events["flatten"], 1)
        live = self._slots(ipd)
        st.snapshot(0)                                        # entry 0 = all zeros
        with st.record():
            for e in (1, 2, 3):                               # position e-1 consumed: layer slots hold the value e
                for t in live:
                    t.fill_(float(e))
                st.record_layer(0, e); st.record_layer(2, e)
        ids = [id(t) for t in self._slots(ipd)]
        st.commit(2)
        self.assertTrue(all(bool((t == 2.0).all()) for t in self._slots(ipd)))
        st.commit(0)
        self.assertTrue(all(bool((t == 0.0).all()) for t in self._slots(ipd)))
        st.commit(3)
        self.assertTrue(all(bool((t == 3.0).all()) for t in self._slots(ipd)))
        self.assertEqual(ids, [id(t) for t in self._slots(ipd)])            # copied INTO the live tensors, never rebound
        self.assertEqual(st.events["rebind_restore"], 0)
        self.assertGreater(st.nbytes(), 0)

    def test_per_slot(self):
        self._run(flatten=False)

    def test_flattened(self):
        self._run(flatten=True)

    def test_depth_exceeded_is_named(self):
        st = C.StateStack(_SH(), _ipd(), depth=1)
        with self.assertRaises(RuntimeError):
            st.record_layer(0, 2)

    def test_multi_token_hook_steps_each_position(self):
        sh, ipd = _SH(), _ipd()
        filt = sh.blocks[0].filter
        seen = []

        def stock_forward(u, inference_params=None, padding_mask=None):
            seen.append(u.shape[1]); return u * 2, inference_params
        fwd = C._make_filter_forward(stock_forward, filt, sh)
        u = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
        y, _ = fwd(u, ipd["hcs"], None)                      # cached (layer 0 has a fir state) and 5 positions: five single-position stock calls
        self.assertEqual(seen, [1, 1, 1, 1, 1])
        self.assertTrue(torch.equal(y, u * 2))
        seen.clear(); fwd(u, None, None)                      # no inference params: the stock call as is
        self.assertEqual(seen, [5])


if __name__ == "__main__":
    unittest.main()
