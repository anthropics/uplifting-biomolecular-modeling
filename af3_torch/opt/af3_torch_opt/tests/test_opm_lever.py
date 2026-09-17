"""Lever `opm` (fast / big): the MSA module's OuterProductMean on the kit's af3t_opm kernels (opt/forward/af3t/kernels/af3t_msa.py +
third_party/af3t_opm.py). CPU, no torch: the mode table carries it in fast / big and never in exact, the ablation switch drops it by name,
the registry's records are complete, the adapter's kernel-error route re-raises an out-of-memory error first, and its Evoformer block forward
RESTATES xfold's EvoformerBlock.forward statement for statement except the first (`pair += self.outer_product_mean(msa, msa_mask)`,
served by the kernels' residual epilogue) — compared against the kit's source live, never a stored hash."""
import ast
import os
import textwrap

import pytest

from af3_torch_opt import modes, registry

from .conftest import HOME

ADAPTER = os.path.join(HOME, "opt", "forward", "af3t", "kernels", "af3t_msa.py")
KERNELS = os.path.join(HOME, "opt", "forward", "af3t", "kernels", "third_party", "af3t_opm.py")
PAIRFORMER = os.path.join(HOME, "opt", "forward", "af3t", "af3_torch", "xfold", "nn", "pairformer.py")
API = os.path.join(HOME, "opt", "forward", "af3t", "af3_torch", "af3_torch_api.py")


def _func(path, qual):
    src = open(path, encoding="utf-8").read(); tree = ast.parse(src)
    node = tree
    for part in qual.split("."):
        node = next(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
    return node


def _stmts(node):
    body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant)) else node.body
    return [ast.unparse(s) for s in body]


def test_mode_table_carries_opm_in_fast_and_big_only():
    sets = modes.kit_lever_sets()
    assert "opm" in sets["fastest"] and sets["fastest"].index("opm") == sets["fastest"].index("pwa_lnl") + 1
    for mode in ("fast", "big"):
        assert "opm" in modes.resolve(mode)["levers"], mode
    for mode in ("off", "exact"):
        assert "opm" not in modes.resolve(mode)["levers"], mode
    assert "opm" not in registry.EXACT and "opm" in registry.NOT_BITWISE


@pytest.mark.parametrize("mode", ["fast", "big"])
def test_ablation_switch_drops_opm_by_name(mode, monkeypatch):
    monkeypatch.delenv(modes.ENV_LEVERS_OFF, raising=False)
    base = modes.resolve(mode)
    r = modes.resolve(mode, environ={modes.ENV_LEVERS_OFF: "opm"})
    assert "opm" not in r["levers"] and r["levers_off"] == ("opm",)
    assert r["levers"] == tuple(l for l in base["levers"] if l != "opm")


def test_registry_records():
    row = registry.LEVERS["opm"]
    assert row["kind"] == "kernel" and row["family"] == "F5" and "tier 2" in row["numerics"]
    assert registry.EVIDENCE["opm"] == "census" and registry.IMPL["opm"] == ("af3t_opm", "kit") and registry.STRATEGY["opm"] == "F5.fpf_msa_kernels"
    assert all(os.path.isfile(os.path.join(HOME, "opt", "forward", p)) for p in row["touches"]), row["touches"]
    assert not any("opm" in r["levers"] for r in registry.KERNEL_ROUTES.values())          # the kernels are the kit's own (no core route)


def test_api_enables_the_adapter():
    src = open(API, encoding="utf-8").read()
    assert '_MSA_LEVERS = ("opm",)' in src and "import af3t_msa as M" in src and "M.enable(ml)" in src


def test_kernel_error_route_reraises_oom_first():
    """The adapter's one broad handler around the kernels starts with `if is_oom(e) or _capturing(): raise` (the core's recogniser, imported once)."""
    src = open(ADAPTER, encoding="utf-8").read(); tree = ast.parse(src)
    assert "from opt_core.oom import is_oom" in src
    handlers = [h for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler) and "_kernel_error" in ast.unparse(h)]
    assert len(handlers) == 1
    first = handlers[0].body[0]
    assert isinstance(first, ast.If) and isinstance(first.body[0], ast.Raise) and first.body[0].exc is None
    assert ast.unparse(first.test) == f"is_oom({handlers[0].name}) or _capturing()"


def test_evoformer_forward_restates_the_stock_block():
    """_evoformer_forward == xfold EvoformerBlock.forward but its first statement (the OPM residual served by the kernels) — same statements, same order."""
    stock = _stmts(_func(PAIRFORMER, "EvoformerBlock.forward"))
    mine = _stmts(_func(ADAPTER, "_evoformer_forward"))
    assert stock[0] == "pair += self.outer_product_mean(msa, msa_mask)"
    assert mine[0] == "pair = _opm_forward(self.outer_product_mean, msa, msa_mask, residual=pair)"
    assert mine[1:] == stock[1:]


def test_opm_forward_serves_the_stock_statement_outside_the_cell():
    """Outside the served cell (and once dead) the adapter calls the saved stock forward and does the block's add itself: `residual += upd`."""
    src = ast.unparse(_func(ADAPTER, "_opm_forward"))
    assert '_ORIG["opm"](self, msa, mask)' in src.replace("'", '"') and "residual += upd" in src
    ksrc = open(KERNELS, encoding="utf-8").read()
    assert "def _ln_proj2_kernel" in ksrc and "def _opm_out_kernel" in ksrc and "RESIDUAL: tl.constexpr" in ksrc
