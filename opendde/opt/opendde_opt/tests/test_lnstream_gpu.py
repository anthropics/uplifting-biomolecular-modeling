"""GPU checks of the `lnstream` build (skipped without CUDA / nvcc): the current-stream extension's four forward entry points are
bitwise upstream's legacy build on random tensors of the model's widths and dtypes, and 20 of its launches captured in a CUDA graph on a side
stream replay to the eager output (the launches are on the capturing stream) and follow a changed static input. These checks live here, in the
kit's tests; the lever (opendde_opt/lnstream.py) carries no test hook."""
import os
import shutil

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or shutil.which("nvcc") is None, reason="needs CUDA + nvcc")



@pytest.fixture(scope="module", autouse=True)
def _upstream_modules_leave_with_this_module():
    """The real upstream modules these tests import (`opendde.*`) are dropped from sys.modules when the module's tests are done: a later test's
    synthetic process must not find an imported-but-unwrapped upstream (the house levers' `fallbacks()` read sys.modules)."""
    import sys
    before = {k for k in sys.modules if k.split(".")[0] == "opendde"}
    yield
    for k in [k for k in sys.modules if k.split(".")[0] == "opendde" and k not in before]:
        del sys.modules[k]
    for k in [k for k in sys.modules if k.rsplit(".", 1)[-1] == "_tiny_sampler"]:
        del sys.modules[k]

@pytest.fixture(scope="module")
def builds():
    os.environ.setdefault("LAYERNORM_TYPE", "fast_layernorm")
    lm = pytest.importorskip("opendde.model.layer_norm.layer_norm")
    from opendde_opt import lnstream
    legacy = lm._load_fast_layer_norm_cuda_v2()                     # upstream's own build (legacy-stream launches)
    assert legacy is not None, "upstream's fused LayerNorm extension did not load"
    cs = lnstream.build(lm)                                          # the lever's current-stream build
    return legacy, cs


def test_every_forward_entry_point_is_bitwise_the_legacy_build(builds):
    legacy, cs = builds
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(5)
    checked = 0
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for cols in (128, 384, 449, 768):
            x = torch.randn(96, cols, generator=g).to(dev, dtype)
            w = (torch.rand(cols, generator=g) + 0.5).to(dev, dtype)
            b = torch.randn(cols, generator=g).to(dev, dtype)
            for name, args in (("forward_none_affine", (x, [cols], 1e-5)), ("forward_with_weight_affine", (x, [cols], w, 1e-5)),
                               ("forward_with_bias_affine", (x, [cols], b, 1e-5)), ("forward_with_both_affine", (x, [cols], w, b, 1e-5))):
                if not hasattr(legacy, name):
                    continue
                out_l, out_c = getattr(legacy, name)(*args), getattr(cs, name)(*args)
                for a, c in zip(out_l, out_c):
                    assert torch.equal(torch.nan_to_num(a, 7.0), torch.nan_to_num(c, 7.0)), (name, dtype, cols)
                checked += 1
    assert checked >= 12


def test_twenty_launches_capture_and_replay_to_eager(builds):
    _, cs = builds
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(11)
    x = torch.randn(64, 384, generator=g).to(dev)
    w = (torch.rand(384, generator=g) + 0.5).to(dev)
    b = torch.randn(384, generator=g).to(dev)

    def f(t):
        return cs.forward_with_both_affine(t, [384], w, b, 1e-5)[0]
    ref = f(x)
    for _ in range(19):
        ref = f(ref)
    torch.cuda.synchronize()
    static_in = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y = f(static_in)
        for _ in range(19):
            y = f(y)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s, capture_error_mode="thread_local"):
        y = f(static_in)
        for _ in range(19):
            y = f(y)
    y.fill_(float("nan"))                                            # only a replay that runs the kernels restores the output
    graph.replay(); torch.cuda.synchronize()
    assert torch.equal(y, ref)
    static_in.copy_(x * 2.0)                                          # a changed static input changes the replayed output
    graph.replay(); torch.cuda.synchronize()
    assert not torch.equal(y, ref)
