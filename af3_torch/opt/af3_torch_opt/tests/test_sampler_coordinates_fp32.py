"""The diffusion sampler's coordinate arithmetic is fp32 under the runner's bfloat16 autocast (README 'Known upstream issues' `XFOLD-004`):
the per-step rigid augmentation returns, bit for bit, what it returns with autocast off, and the denoised update widens the decoder's
bfloat16 update to fp32 before scaling — in the port's `DiffusionHead` and in the two kit statements that restate it (the api's
external-randomness sampler, the `--n_gpu` denoiser). The behavioural test runs the REAL `random_augmentation` on CPU (torch's CPU autocast
lowers the einsum like the CUDA one); it needs torch + einops + triton importable (xfold.fastnn imports triton) and is skipped otherwise.
The statement tests read the kit's source (af3_torch_opt.stack.forward_dir) and need nothing."""
import ast
import os
import sys

import pytest

from af3_torch_opt import stack

ROWPAIR_XFOLD = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rowpair_xfold.py")   # read as source: the module imports torch (the model process's), these statement tests do not

PORT_DH = "af3t/af3_torch/xfold/nn/diffusion_head.py"
API = "af3t/af3_torch/af3_torch_api.py"
WIDENED_UPDATE = "skip_scaling * positions_noisy + out_scaling * position_update.to(torch.float32)"


def _src(rel):
    return open(os.path.join(stack.forward_dir(), *rel.split("/")), encoding="utf-8").read()


def _func(src, qual):
    tree = ast.parse(src); node = tree
    for part in qual.split("."):
        node = next(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
    return node


def _is_autocast_disabled(with_node):
    for item in with_node.items:
        c = item.context_expr
        if isinstance(c, ast.Call) and ast.unparse(c.func) == "torch.autocast" and any(k.arg == "enabled" and ast.unparse(k.value) == "False" for k in c.keywords):
            return True
    return False


def _einsums(node):
    return {id(n) for n in ast.walk(node) if isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.einsum"}


def _function_source(path, name):
    """The source text of function `name` in the module file at `path`, parsed with ast — nothing imported."""
    src = open(path).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} is not defined in {path}")


def test_every_rotation_einsum_sits_in_an_autocast_disabled_block():
    for rel, qual in ((PORT_DH, "random_augmentation"), (API, "run_diffusion")):
        fn = _func(_src(rel), qual)
        inside = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.With) and _is_autocast_disabled(n):
                inside |= _einsums(n)
        every = _einsums(fn)
        assert every and every <= inside, (rel, qual, len(every), len(inside))       # the rigid transform's einsum(s), all of them, run with autocast off


def test_denoised_update_is_widened_in_the_port_and_its_restatement():
    ret = [n for n in ast.walk(_func(_src(PORT_DH), "DiffusionHead.forward")) if isinstance(n, ast.Return)][-1]
    assert WIDENED_UPDATE in ast.unparse(ret), ast.unparse(ret)
    sharded = _function_source(ROWPAIR_XFOLD, "_dh_forward_sharded")
    assert WIDENED_UPDATE in sharded, "rowpair_xfold._dh_forward_sharded restates DiffusionHead.forward's return statement"


def test_random_augmentation_is_fp32_under_autocast():
    torch = pytest.importorskip("torch"); pytest.importorskip("einops"); pytest.importorskip("triton")
    if not torch.cuda.is_available():                                   # xfold.fastnn's module-level @triton.autotune probes a driver at import; its routes are unused here
        import triton
        triton.autotune = lambda *a, **kw: (lambda fn: fn); triton.heuristics = lambda *a, **kw: (lambda fn: fn)
    kit = os.path.join(stack.forward_dir(), "af3t", "af3_torch")
    if kit not in sys.path:
        sys.path.insert(0, kit)
    from xfold.nn import diffusion_head as DH
    g = torch.Generator().manual_seed(0)
    positions = (torch.rand((37, 24, 3), generator=g) - 0.5) * 90.0          # absolute coordinates up to 45 Å, fp32
    mask = (torch.rand((37, 24), generator=g) > 0.2).to(torch.float32)
    torch.manual_seed(5); plain = DH.random_augmentation(positions=positions, mask=mask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        lowered = torch.einsum('...i,ij->...j', positions, torch.eye(3))
        if lowered.dtype != torch.bfloat16:
            pytest.skip("this torch build's CPU autocast does not lower einsum; the CUDA policy the runner uses does")
        torch.manual_seed(5); under = DH.random_augmentation(positions=positions, mask=mask)
    assert under.dtype == torch.float32 and torch.equal(under, plain)         # the same fp32 statement ran: no coordinate was rounded
