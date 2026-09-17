"""The batched line's per-batch sampling is ONE named function, genie3_opt.sampling.sample_batch, called once per batch by the
real driver g3batch.py: the initial `torch.randn_like` over gt_atom_positions,
then per step the denoiser call (graph wrapper or eager forward), the step's `torch.randn_like(xs)` noise on every step but the last, and
g3fast.ddim_math. Source facts (no torch here): the function's statement sequence, no exception handler / timer / print in the module, and the
driver's single call site inside its batch loop with the step loop gone from the driver."""
import ast, os

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(PKG, "sampling.py"), encoding="utf-8").read()
DRV = open(os.path.join(PKG, "g3batch.py"), encoding="utf-8").read()


def test_sample_batch_is_the_step_loop_verbatim():
    fn = next(n for n in ast.parse(SRC).body if isinstance(n, ast.FunctionDef) and n.name == "sample_batch")
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]   # minus the docstring
    assert [type(s).__name__ for s in body] == ["Assign", "For", "Return"]
    assert ast.unparse(body[0]) == "xs = torch.randn_like(bd['gt_atom_positions'])"
    loop = body[1]
    assert ast.unparse(loop.target) == "k" and ast.unparse(loop.iter) == "range(len(tab.steps))"
    assert [ast.unparse(s) for s in loop.body] == [
        "s_vec = SB[k]",
        "out_xl = gd(xs, s_vec) if gd is not None else model(batch=bd, xl=xs, t=s_vec / tab.n_timestep)['xl']",
        "noise = None if tab.is_last[k] else torch.randn_like(xs)",
        "xs = ddim_math(tab, k, s_vec, xs, out_xl, mask, noise)"]
    assert ast.unparse(body[2]) == "return xs"
    assert [a.arg for a in fn.args.args] == ["gd", "model", "bd", "tab", "SB", "mask", "ddim_math"]


def test_the_module_measures_nothing():
    tree = ast.parse(SRC)
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.ExceptHandler, ast.Try))]
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not ({"time", "perf_counter", "print", "synchronize", "Event", "logging"} & names), names
    assert [ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))] == ["from __future__ import annotations", "import torch"]


def test_the_driver_calls_it_once_per_batch_and_keeps_no_step_loop():
    assert DRV.count("SMP.sample_batch(") == 1 and "from genie3_opt import sampling as SMP" in DRV
    code = {ast.unparse(n) for n in ast.walk(ast.parse(DRV)) if isinstance(n, (ast.For, ast.Call))}
    assert not [c for c in code if c.startswith("for k in range(len(tab.steps))") or c == "torch.randn_like(xs)"]   # the loop and its per-step draw live in sampling.py only (the driver's docstring still describes them)
    call = DRV[DRV.index("SMP.sample_batch("):DRV.index(")", DRV.index("SMP.sample_batch(")) + 1]
    assert call == "SMP.sample_batch(gd, model, bd, tab, SB, mask, G.ddim_math)"
