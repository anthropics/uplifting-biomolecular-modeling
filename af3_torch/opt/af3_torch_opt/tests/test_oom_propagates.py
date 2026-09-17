"""OOM propagates; no fallback applied. The kit adapter's kernel-error route — a lever whose kernel raises is disabled for
the process and the stock path serves the call (opt/forward/af3t/kernels/af3_kernels.py `_kernel_error`) — re-raises an out-of-memory error
before anything else, through the shared core's one recogniser (`from opt_core.oom import is_oom`: the kit pins opt_core; one helper, no kit-local spelling).
Proven here without a GPU: the served entry point `_transition_forward` is executed from the carried source with its callee patched to raise a
torch-shaped OutOfMemoryError, and the handler census (which handlers re-raise, how many) is read from the file's AST."""
import ast
import os
import sys
import types

import pytest
from opt_core.oom import is_oom

from .conftest import HOME

ADAPTER = os.path.join(HOME, "opt", "forward", "af3t", "kernels", "af3_kernels.py")
KERNEL_ERROR_HANDLERS = 8          # trimul, triattn, transition, glu_proj, selfattn (apb), apb pair logits, msaattn pair side (pwa_lnl), msaattn msa side (pwa_msa)


def _source():
    return open(ADAPTER, encoding="utf-8").read()


def _reraises_oom_first(handler: ast.ExceptHandler) -> bool:
    """`if is_oom(e) or _capturing(): raise` — the OOM recogniser first, then the capture predicate (an error under the whole-step CUDA-graph
    capture belongs to the capture's owner, DiffusionHead.forward_graphed, not to the kernel: the lever is not disabled for it)."""
    first = handler.body[0]
    if not (isinstance(first, ast.If) and isinstance(first.body[0], ast.Raise) and first.body[0].exc is None):
        return False
    test = first.test
    if not (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or) and len(test.values) == 2):
        return False
    oom, cap = test.values
    return (isinstance(oom, ast.Call) and getattr(oom.func, "id", None) == "is_oom" and [getattr(a, "id", None) for a in oom.args] == [handler.name]
            and isinstance(cap, ast.Call) and getattr(cap.func, "id", None) == "_capturing" and not cap.args)


def test_every_kernel_error_route_reraises_oom_first():
    """The handler census: every broad handler of the adapter that routes to `_kernel_error` starts with `if is_oom(e): raise` (8 of them);
    the one recogniser is the core's (`from opt_core.oom import is_oom`, imported once); no handler names an OOM class of its own."""
    src = _source(); tree = ast.parse(src)
    routed, guarded = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None and ast.unparse(node.type) in ("Exception", "BaseException"):
            body = ast.unparse(ast.Module(body=node.body, type_ignores=[]))
            if "_kernel_error(" in body:
                routed.append(node.lineno)
                if _reraises_oom_first(node):
                    guarded.append(node.lineno)
    assert routed == guarded and len(routed) == KERNEL_ERROR_HANDLERS, (routed, guarded)
    assert src.count("from opt_core.oom import is_oom") == 1 and "OutOfMemoryError" not in src and "out of memory" not in src.lower().replace("opt_core.oom", "")
    cap = src[src.index("def _capturing("):src.index("def _kernel_error(")]
    assert "torch.cuda.is_current_stream_capturing()" in cap and "except Exception:" in cap and "return False" in cap


def test_the_whole_step_capture_is_thread_local_and_the_dtk_swap_defers_to_it():
    """Source contract (the capture needs a GPU): DiffusionHead.forward_graphed captures with capture_error_mode='thread_local' — another
    thread's CUDA call cannot invalidate the capture (torch's default 'global' mode counts those) — and the dtk swap, like the adapter's served
    sites, re-raises an error met under capture to the capture's owner instead of stepping aside for the process."""
    dh = open(os.path.join(HOME, "opt", "forward", "af3t", "af3_torch", "xfold", "nn", "diffusion_head.py"), encoding="utf-8").read()
    fg = dh[dh.index("    def forward_graphed("):dh.index("    def forward(\n", dh.index("    def forward_graphed("))]
    assert fg.count("torch.cuda.CUDAGraph()") == 1 and 'with torch.cuda.graph(graph, capture_error_mode="thread_local"):' in fg and "with torch.cuda.graph(graph):" not in fg
    assert "self.use_step_graph = False" in fg and "graph_failures" in fg, "a capture that fails still steps the step graph aside by name"
    fw = open(os.path.join(HOME, "opt", "af3_torch_opt", "forward.py"), encoding="utf-8").read()
    dtk = fw[fw.index("        def forward(self, act, mask, single_cond, pair_cond, pair_logits=None):"):fw.index("    return XfoldDTK")]
    assert "if _is_oom(e, torch) or self.stock is None or torch.cuda.is_current_stream_capturing():" in dtk and dtk.index("is_current_stream_capturing()") < dtk.index("self.dead = ")


class _FakeDtype:
    pass


class _FakeTensor:
    is_cuda = True
    shape = (4096, 128)
    dtype = _FakeDtype

    def numel(self):
        return 4096 * 128

    def dim(self):
        return 2


class _FakeModule:                                                             # the attributes of xfold's Transition the adapter reads before serving (transition2.weight [C, HID])
    transition2 = types.SimpleNamespace(weight=types.SimpleNamespace(shape=(128, 512)))


def _transition_forward_from_source(served, capturing=False):
    """`_transition_forward` compiled from the carried file into a namespace holding the adapter's own collaborators as fakes: the stock path
    (`_ORIG['transition']`), the census (`_count` / `_fallback`), the kernel-error route (`_kernel_error`, recording), the core's `is_oom`, the tier word, and a
    transition PROVIDER stand-in (`_TRANS_PROV['mod']`) whose face `transition` is the callee under test (its selection = a kernel row)."""
    src = _source(); tree = ast.parse(src); lines = src.splitlines(keepends=True)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("_plus", "_transition_forward")]   # _plus: the adapter's residual statement helper its fallback returns go through
    calls = {"kernel_error": [], "fallback": [], "count": []}
    TP = types.ModuleType("fake_transition_provider")
    TP.Refusal = type("Refusal", (Exception,), {}); TP.STOCK_ROWS = ("torch_swiglu", "engine_module", "compile"); TP.transition = served
    ns = {"_DEAD": {}, "_ORIG": {"transition": lambda self, x: "stock"}, "_autocast_bf16": lambda: True, "_TIER": {"word": "fast"},
          "torch": types.SimpleNamespace(bfloat16=_FakeDtype), "_PROV_DTYPE": {_FakeDtype: "bf16"}, "_TRANS_FAMILY": {384: "single", 64: "rows"},
          "_transition_pack": lambda m, TP_: object(), "_transition_decide": lambda TP_, C, HID, dt, N, rows, residual, capture: types.SimpleNamespace(row="v2"),
          "_transition_refused": lambda TP_, C, HID, dt, N, residual, capture, r: str(r),
          "_count": lambda lever, key: calls["count"].append((lever, key)), "_fallback": lambda lever, reason: calls["fallback"].append((lever, reason)),
          "_kernel_error": lambda lever, e: calls["kernel_error"].append((lever, repr(e))), "is_oom": is_oom, "_capturing": lambda: capturing,
          "_TRANS_PROV": {"mod": TP, "stack": None, "cc": "9.0", "refused": None, "sel": {}}}
    exec(compile("".join("".join(lines[n.lineno - 1: n.end_lineno]) for n in nodes), ADAPTER, "exec"), ns)
    return ns["_transition_forward"], calls


def test_oom_propagates_out_of_the_served_adapter_and_other_errors_keep_their_route():
    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch"})     # torch's class, by name (opt_core.oom recognises it without torch)

    def oom(*a, **k):
        raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    fwd, calls = _transition_forward_from_source(oom)
    with pytest.raises(OutOfMemoryError):
        fwd(_FakeModule(), _FakeTensor())
    assert calls["kernel_error"] == [] and calls["fallback"] == []          # not counted, not rerouted: the caller sees the OOM

    def drift(*a, **k):
        raise RuntimeError("triton: invalid launch configuration")

    fwd2, calls2 = _transition_forward_from_source(drift)
    assert fwd2(_FakeModule(), _FakeTensor()) == "stock"                    # a non-OOM kernel error: the delivered route — lever disabled, stock serves
    assert [lever for lever, _ in calls2["kernel_error"]] == ["transition"]

    def invalidated(*a, **k):
        raise RuntimeError('info.status != cudaStreamCaptureStatusInvalidated INTERNAL ASSERT FAILED at "CUDACachingAllocator.cpp":2213')

    fwd3, calls3 = _transition_forward_from_source(invalidated, capturing=True)
    with pytest.raises(RuntimeError, match="cudaStreamCaptureStatusInvalidated"):
        fwd3(_FakeModule(), _FakeTensor())                                   # under the whole-step capture: the capture's owner sees it; the lever is NOT disabled
    assert calls3["kernel_error"] == [] and calls3["fallback"] == []

    def served(x, W, **kw):                                                 # the face serving: (out, selection); the adapter returns out and counts served:c=128
        return "fused", types.SimpleNamespace(row="v2")

    fwd4, calls4 = _transition_forward_from_source(served)
    assert fwd4(_FakeModule(), _FakeTensor()) == "fused" and ("transition", "served:c=128") in calls4["count"] and calls4["fallback"] == []
