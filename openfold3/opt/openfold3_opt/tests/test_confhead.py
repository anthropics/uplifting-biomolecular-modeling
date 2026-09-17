"""The confidence-phase levers of the single-GPU big line: the row-block confidence head (opt/openfold3_opt/confhead.py, lever `confhead`
of the resident line) and their size gate (modes.conf_gate, modes.CONF_MIN_TOKENS): the
line forms above and below the gate, the hook chain and its variables, the evidence fields, and the row-block head's statement against the
full-slab head (CPU torch; skipped without torch). No openfold3, no GPU."""
import os
import subprocess
import sys

import pytest

from openfold3_opt import hooks, modes, registry, report, stack
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()


def _resolve(line, n):
    assert line == "resident"
    return modes.resolve("big", HOME, environ={}, n_tokens=n)


def test_gate_constants_and_words():
    assert modes.CONF_MIN_TOKENS == 2048 and modes.CONF_HOOK == "confhead" and modes.CONF_GATE_NAME == "conf"
    assert set(modes.CONF_LEVERS) == {"confhead", "conf_chunked"} and set(modes.CONF_LEVERS) <= set(registry.LEVERS)
    assert set(modes.CONF_ENVS_BELOW) <= set(modes.CONF_ENVS) and modes.ENV_CONFHEAD in modes.CONF_ENVS
    ln = modes.LINES[("big", "resident")]
    assert modes.conf_gate(modes.LINES[("fast", None)], 5000) is None                                   # a line without the levers: no gate
    g = modes.conf_gate(ln, None)
    assert g["served"] and g["reason"] == "n_tok_unknown" and g["fragment"] == "conf=n_tok_unknown min_tokens=2048"
    g = modes.conf_gate(ln, 2047)
    assert not g["served"] and g["reason"] == "gated:lt_min" and g["fragment"] == "conf=gated:lt_min min_tokens=2048"   # opt_core.attn.size_gate's event word
    g = modes.conf_gate(ln, 2048)
    assert g["served"] and g["reason"] == "served"


@pytest.mark.parametrize("n", [None, 2048, 2956, 4076])
def test_resident_at_and_above_the_gate_runs_the_row_block_head_first_in_the_chain(n):
    r = _resolve("resident", n)
    assert r.hooks == ["confhead", "offload", "cells", "trunk_kernels", "fast_inference"] and r.entry_hook == os.path.join(modes.hook_dir(HOME, "confhead"), modes.HOOK_FILE)   # the row-block head's hook is the entry hook
    assert r.exports[modes.OFFLOAD_KIT_LEVERS_ENV] == modes.hook_dir(HOME, "cells")                          # the port's hook chains the cells' ...
    assert r.exports[modes.PAIRFUSED_CHAIN_ENV] == modes.hook_dir(HOME, "trunk_kernels")                    # ... which chains the trunk-kernels add-on's ...
    assert r.exports[modes.KIT_LEVERS_ENV] == modes.hook_dir(HOME, "fast_inference")                        # ... which chains the kit's
    assert r.exports["OPENFOLD3_OPT_CONFHEAD"] == "1" and r.exports["OF3O_CONF_MODE"] == "chunked" and r.exports["OF3O_CONF_TM_BACKEND"] == "blockreduce"
    assert r.exports[modes.CONFHEAD_CHAIN_ENV] == modes.hook_dir(HOME, "offload")                          # confhead chains the port's hook
    assert "confhead" in r.levers and "conf_chunked" in r.levers
    assert r.conf_gate == f"conf={'served' if n else 'n_tok_unknown'} min_tokens=2048" and r.size_gate is None and not r.conflicts


@pytest.mark.parametrize("line", ["resident"])
def test_below_the_gate_the_line_is_the_port_line_with_the_stock_confidence_path(line):
    """Under CONF_MIN_TOKENS the resolution is the line without the confidence levers: no confhead hook, the port's stock scorer, the levers absent from the request and their switches required absent — byte for byte the pre-lever line."""
    r = _resolve(line, 1536)                                                     # below the confidence gate (2048), above the offload port's item gate (modes.OF3O_GATE_DEFAULT)
    assert modes.OF3O_GATE_DEFAULT <= 1536 < modes.CONF_MIN_TOKENS
    assert r.hooks == ["offload", "cells", "trunk_kernels", "fast_inference"] and r.entry_hook == os.path.join(modes.hook_dir(HOME, "offload"), modes.HOOK_FILE)   # below the gate the port's hook is the entry hook
    assert r.exports[modes.OFFLOAD_KIT_LEVERS_ENV] == modes.hook_dir(HOME, "cells") and r.exports[modes.PAIRFUSED_CHAIN_ENV] == modes.hook_dir(HOME, "trunk_kernels")
    assert r.exports["OF3O_CONF_MODE"] == "stock" and "OF3O_CONF_TM_BACKEND" not in r.exports and "OPENFOLD3_OPT_CONFHEAD" not in r.exports
    assert modes.CONFHEAD_CHAIN_ENV not in r.exports and r.exports[modes.KIT_LEVERS_ENV] == modes.hook_dir(HOME, "fast_inference")
    below = {k for k in modes.LINES[("big", line)].env if k in modes.CONF_ENVS and k not in modes.CONF_ENVS_BELOW}   # the confhead switch + the TM backend
    assert not set(r.levers) & set(modes.CONF_LEVERS) and below and below <= set(r.unsets)
    assert r.conf_gate == "conf=gated:lt_min min_tokens=2048" and r.conf_reason == "gated:lt_min"
    want = {k: v for k, v in modes.LINES[("big", line)].env.items() if k not in modes.CONF_ENVS}
    assert {k: v for k, v in r.exports.items() if k not in modes.CONF_ENVS and k not in modes.KIT_LEVERS_ENVS} == want   # every other switch as written


def test_a_preset_confhead_switch_below_the_gate_is_a_conflict_not_an_override():
    r = modes.resolve("big", HOME, environ={"OPENFOLD3_OPT_CONFHEAD": "1"}, n_tokens=1536)
    assert r.conflicts and "OPENFOLD3_OPT_CONFHEAD" in " ".join(r.conflicts)


def test_hook_attribution_knows_the_package_hook():
    r = _resolve("resident", 4076)
    assert modes.hook_spellings(r) == ["opt/openfold3_opt/hooks/confhead", "of3o", "opt/openfold3_opt/hooks/cells", "of3t_hook", "of3_levers"]
    assert modes.describe_line(r).endswith("@ opt/openfold3_opt/hooks/confhead(confhead)>of3o(offload)>opt/openfold3_opt/hooks/cells(cells)>of3t_hook(trunk_kernels)>of3_levers(fast_inference)")
    assert "OPENFOLD3_OPT_CONFHEAD=1" in modes.describe_line(r) and modes.CONFHEAD_CHAIN_ENV + "=" not in modes.describe_line(r)   # chain variables are the hook order, not spelled
    assert modes.hook_dir(HOME, "confhead") in modes.hook_dirs_of(HOME, "confhead") and hooks.expected(r) == ["confhead", "offload", "cells", "trunk_kernels", "fast_inference"]
    assert modes.hook_dir(HOME, "confhead") in [os.path.abspath(d) if os.path.isabs(d) else d for d in __import__("openfold3_opt.env", fromlist=["kit_dirs"]).kit_dirs(HOME)]
    src = open(os.path.join(modes.hook_dir(HOME, "confhead"), modes.HOOK_FILE), encoding="utf-8").read()
    assert 'os.environ.get("OPENFOLD3_OPT_CONFHEAD"' in src and "OPENFOLD3_OPT_CONFHEAD_CHAIN" in src and "class _PostImportFinder" in src and "confhead.install()" in src
    from openfold3_opt import _autoload
    assert {"OPENFOLD3_OPT_CONFHEAD", "OPENFOLD3_OPT_CONFHEAD_CHAIN"} <= set(_autoload.DECLARED)            # the hook's variables are the package's (a child interpreter inherits them: never a mistyped-switch refusal)


def test_hook_installs_the_finder_only_when_its_switch_is_set(tmp_path):
    """In a fresh interpreter: with OPENFOLD3_OPT_CONFHEAD=1 the hook puts its finder on sys.meta_path and executes the chained directory's
    sitecustomize.py; without the switch it installs nothing (the line below the gate never runs it, the PYTHONPATH route may)."""
    chained = tmp_path / "next"
    chained.mkdir()
    (chained / "sitecustomize.py").write_text("import os\nos.environ['CHAIN_RAN'] = '1'\n")
    hook = os.path.join(modes.hook_dir(HOME, "confhead"), modes.HOOK_FILE)
    code = ("import json, os, runpy, sys\n"
            f"runpy.run_path({hook!r}, run_name='openfold3_opt_hook_confhead')\n"
            "print(json.dumps({'finders': [type(f).__name__ for f in sys.meta_path], 'chain': os.environ.get('CHAIN_RAN'), 'path1': sys.path[1]}))\n")
    for on in ("1", ""):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OF3", "OPENFOLD3_OPT"))}
        env.update({"OPENFOLD3_OPT_CONFHEAD": on, "OPENFOLD3_OPT_CONFHEAD_CHAIN": str(chained)})
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-800:]
        out = __import__("json").loads(r.stdout.strip().splitlines()[-1])
        assert (out["finders"][0] == "_PostImportFinder") == (on == "1") and out["chain"] == "1" and out["path1"] == str(chained), (on, out)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OF3", "OPENFOLD3_OPT"))}
    env.update({"OPENFOLD3_OPT_CONFHEAD": "1", "OPENFOLD3_OPT_CONFHEAD_CHAIN": str(tmp_path / "nowhere")})
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "no sitecustomize.py there" in r.stderr                                     # a chained directory without a hook: refused by name, never skipped


def test_lever_lines_carry_the_gate_decision_and_the_activation_line_the_fragment():
    rep = {"active": True, "mode": "big", "line": "resident", "levers_requested": ["fast_init", "trimul_hostsnap"], "levers_applied": ["fast_init"],
           "levers_unavailable": [], "conf_gate": "conf=gated:lt_min min_tokens=2048", "conf_reason": "gated:lt_min", "n_tokens": 1012, "core_routes": [], "n_gpu": 1}
    by = {l.split("name=")[1].split()[0]: l for l in report.lever_lines(rep)}
    assert "state=off reason=conf_gate impl=confhead origin=kit strategy=F7.chunked_eval mode=big line=resident tier=tolerance gate=gated:lt_min n_tokens=1012 min_tokens=2048" in by["confhead"]
    assert "state=off reason=conf_gate impl=offload origin=kit strategy=F7.chunked_eval " in by["conf_chunked"] and by["conf_chunked"].endswith("gate=gated:lt_min n_tokens=1012 min_tokens=2048")
    assert " conf=gated:lt_min min_tokens=2048 n_tokens=1012 n_gpu=1 " in report.activation_line(dict(rep, line_spelling="x", hooks_spelling="h")) + " "   # before the n_gpu tokens that end the line
    rep2 = dict(rep, levers_requested=["confhead", "conf_chunked"], levers_applied=["confhead", "conf_chunked"], conf_gate="conf=served min_tokens=2048", conf_reason="served", n_tokens=4076)
    by2 = {l.split("name=")[1].split()[0]: l for l in report.lever_lines(rep2)}
    assert by2["confhead"].startswith("[openfold3-opt] LEVER name=confhead state=on impl=confhead origin=kit strategy=F7.chunked_eval mode=big line=resident tier=tolerance gate=served n_tokens=4076 min_tokens=2048"), by2["confhead"]


def test_registry_rows():
    ch, cc = registry.LEVERS["confhead"], registry.LEVERS["conf_chunked"]
    assert ch.kit == "confhead" and ch.kit in modes.PACKAGE_HOOKS and ch.env_keys == ("OPENFOLD3_OPT_CONFHEAD",) and ch.tier == registry.TOLERANCE and ch.target == registry.T_HEADS
    assert ch.marker == "[openfold3-opt/confhead] installed" and ch.modes == ("big/resident",)
    assert set(cc.modes) == {"big/resident"}
    assert registry.STRATEGY == {"confhead": "F7.chunked_eval", "conf_chunked": "F7.chunked_eval"}      # catalogue membership: test_strategy_catalogue.py
    assert "confhead" in stack._PROBES and "logits_host" not in stack._PROBES and stack.lever_applied("confhead") is None   # pending: the heads module is not imported here (or torch is absent)
    assert registry.T_HEADS in hooks.targets_imported.__code__.co_consts or registry.T_HEADS in " ".join(map(str, hooks.targets_imported.__code__.co_consts))


def test_rowblock_logits_matches_full_head():                                          # ported from the memory mode's original test
    """RowBlockLogits (a torch.Tensor wrapper subclass): the per-row-block head equals the full-slab stock statement (CPU torch; skipped without
    torch) — PAE (asymmetric) and PDE (symmetric: f(z) + f(z)^T) on random data; the port's confidence path's ops covered: the tree maps'
    `t[i]` / `t[j:j+1]` on the lead dims (views), the scorer's `.reshape(S, n, n, -1)` and `pde_l[sl, i0:i1]` (the block), the ellipsis form,
    `.float()` / `.to(device=)` lazy, detach / clone lazy, a single-row select; every op outside those (column / bin indexing, steps, newaxis,
    advanced indexing, reductions, arithmetic, cat / stack, transposes, numpy, writes, iteration, truthiness) and a released object refuse BY
    NAME — at the Python level (RowBlockLogits.__getitem__ / __torch_function__), so on any torch version."""
    torch = pytest.importorskip("torch")
    import math
    from openfold3_opt import confhead
    torch.manual_seed(0)

    class Head(torch.nn.Module):
        def __init__(self, c_z=16, c_out=64):
            super().__init__()
            self.c_out = c_out
            self.layer_norm = torch.nn.LayerNorm(c_z)
            self.linear = torch.nn.Linear(c_z, c_out)

        def full(self, z, symmetric):
            x = self.linear(self.layer_norm(z))
            return x + x.transpose(-2, -3) if symmetric else x

    pae, pde = Head().double(), Head().double()
    B, S, N = 1, 3, 12
    z = torch.randn(B, S, N, N, 16, dtype=torch.float64)
    full_pae, full_pde = pae.full(z, False), pde.full(z, True)
    lp = confhead.RowBlockLogits.of(z, pae, False, "pae", autocast=(False, None))
    ld = confhead.RowBlockLogits.of(z, pde, True, "pde", autocast=(False, None))
    assert isinstance(lp, torch.Tensor) and tuple(lp.shape) == (B, S, N, N, 64) and lp.dtype == torch.float64 and lp.ndim == 5
    assert repr(lp).startswith("RowBlockLogits(pae")

    def slice_batch(t, i):                               # of3o_confidence.get_confidence_scores_chunked's tree maps
        return t[i] if isinstance(t, torch.Tensor) and t.ndim >= 1 else t

    def slice_sample(t, j):
        return t[j:j + 1] if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[0] != 1 else t

    lb = slice_batch(ld, 0)
    assert type(lb) is confhead.RowBlockLogits and tuple(lb.shape) == (S, N, N, 64)
    ls_ = slice_sample(lb, 1)
    assert type(ls_) is confhead.RowBlockLogits and tuple(ls_.shape) == (1, N, N, 64)
    lead = ls_.shape[:-3]; Sp = int(math.prod(lead)); n = ls_.shape[-3]           # pair_row_pass's reads
    fl = ls_.reshape(Sp, n, n, -1)
    assert type(fl) is confhead.RowBlockLogits and tuple(fl.shape) == (1, N, N, 64)
    for i0, i1 in ((0, 5), (5, 10), (10, 12)):
        blk = fl[slice(0, Sp), i0:i1]
        assert type(blk) is torch.Tensor and tuple(blk.shape) == (1, i1 - i0, N, 64)
        assert torch.equal(blk, full_pde[0, 1:2, i0:i1])
    fa = lp.reshape(B * S, N, N, -1)
    assert torch.equal(fa[slice(0, B * S), 3:7], full_pae.reshape(B * S, N, N, 64)[:, 3:7])
    blk2 = lp[..., 3:7, :, :]
    assert tuple(blk2.shape) == (B, S, 4, N, 64) and torch.equal(blk2, full_pae[:, :, 3:7])
    lf = ld.float()
    assert type(lf) is confhead.RowBlockLogits and lf.dtype == torch.float32
    b32 = lf.reshape(B * S, N, N, -1)[slice(0, B * S), 0:2]
    assert b32.dtype == torch.float32 and torch.allclose(b32, full_pde.reshape(B * S, N, N, 64)[:, 0:2].float())
    assert type(ld.to(device="cpu")) is confhead.RowBlockLogits and type(ld.detach()) is confhead.RowBlockLogits and type(ld.clone()) is confhead.RowBlockLogits
    one = lb[1]
    assert type(one) is confhead.RowBlockLogits and tuple(one.shape) == (N, N, 64)
    row = one[4]
    assert type(row) is torch.Tensor and tuple(row.shape) == (N, 64) and torch.equal(row, full_pde[0, 1, 4])
    # refused BY NAME at the Python level (RowBlockLogits.__getitem__ / __torch_function__), whatever the torch version lowers them to:
    # a pair-column / bin slice or select, a step, newaxis, advanced indexing, reductions, arithmetic, concatenation, transposes, numpy,
    # in-place writes, iteration, truthiness
    refused = {"col slice": lambda: fl[:, :, 0:3], "bin select": lambda: fl[..., 0], "col select": lambda: fl[0, 0, 1], "step": lambda: fl[:, 0:4:2],
               "newaxis": lambda: fl[None], "list index": lambda: fl[[0]], "tensor index": lambda: fl[torch.tensor([0])], "bool index": lambda: fl[True],
               "sum": lambda: fl.sum(), "torch.sum": lambda: torch.sum(fl), "mean": lambda: fl.mean(-1), "add": lambda: fl + 1, "mul": lambda: 2 * fl,
               "cat": lambda: torch.cat([fl, fl]), "stack": lambda: torch.stack([fl, fl]), "transpose": lambda: fl.transpose(-1, -2), "permute": lambda: fl.permute(0, 2, 1, 3),
               "mT": lambda: fl.mT, "numpy": lambda: fl.numpy(), "setitem": lambda: fl.__setitem__((0, 0), 0.0), "iter": lambda: next(iter(fl)),
               "bool": lambda: bool(fl), "softmax": lambda: torch.softmax(fl, -1), "einsum": lambda: torch.einsum("srnc->srn", fl),
               "bad reshape": lambda: fl.reshape(Sp, n * n, -1), "matmul": lambda: fl @ torch.ones(64, 1, dtype=fl.dtype)}
    for label, fn in refused.items():
        with pytest.raises(RuntimeError, match="RowBlockLogits"):
            fn()
            raise AssertionError(label)                              # unreachable when the op refused
    # metadata reads answer from the wrapper (no data): every one the tree maps and the scorer issue
    assert tuple(fl.shape) == (1, N, N, 64) and fl.dtype == torch.float64 and fl.device.type == "cpu" and fl.ndim == 4 and not fl.is_cuda
    assert tuple(fl.size()) == (1, N, N, 64) and fl.size(0) == 1 and fl.dim() == 4 and len(fl) == 1 and fl.numel() == N * N * 64 and fl.is_floating_point()
    # whole-axis `:` on the pair-column / bin axes and on the row axis stays lazy; contiguous / cpu are the same lazy object
    assert type(fl[:, :, :, :]) is confhead.RowBlockLogits and type(fl[0:1, :]) is confhead.RowBlockLogits and type(fl.contiguous()) is confhead.RowBlockLogits and type(fl.cpu()) is confhead.RowBlockLogits
    assert torch.equal(fl[..., 2:4, :, :], full_pde[0, 1:2, 2:4]) and torch.equal(fl[0, 5], full_pde[0, 1, 5]) and torch.equal(fl[-1, -1], full_pde[0, 1, N - 1])
    assert torch.equal(torch.reshape(lp, (B * S, N, N, 64))[:, 0:1], full_pae.reshape(B * S, N, N, 64)[:, 0:1]) and torch.equal(lp.view(B * S, N, N, -1)[:, 1:2], full_pae.reshape(B * S, N, N, 64)[:, 1:2])
    assert lp.to(torch.float32).dtype == torch.float32 and lp.to("cpu", torch.float32).dtype == torch.float32 and lp.to(z).dtype == z.dtype and lp.half().dtype == torch.float16
    assert confhead.STATE["blocks"] >= 7
    ld.release()
    with pytest.raises(RuntimeError, match="released"):
        fl[slice(0, 1), 0:2]
