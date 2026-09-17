"""Unit tests for ef2_sampler_graph (one GPU; a random-initialised DiffusionStructureHead, no weights; synthetic fold features).

    python k/test_ef2_sampler_graph.py          # standalone: last line "RESULT passed=N failed=M"
    pytest -q k/test_ef2_sampler_graph.py
The structure module's atom->token mean uses scatter atomics in stock (not run-to-run reproducible); the tests install the det recipe's
deterministic segment mean (the same function text as ef2inv_opt.det) on the fork module for their duration, as `--det 1` does on every arm.
"""
import os
import sys
import types

try:
    import pytest
except ImportError:                      # images without pytest: the same tiny stand-in k/run_tests_nopytest.py registers
    import importlib.machinery
    pytest = types.ModuleType("pytest")
    pytest.__spec__ = importlib.machinery.ModuleSpec("pytest", None)

    class _Mark:
        def skipif(self, cond, reason=""):
            def deco(f):
                f._skip = bool(cond) or getattr(f, "_skip", False)
                f._skip_reason = reason if cond else getattr(f, "_skip_reason", "")
                return f
            return deco
    pytest.mark = _Mark()
    sys.modules["pytest"] = pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ef2_sampler_graph as sg  # noqa: E402
import ef2_pairbias_attn as pba  # noqa: E402
from transformers.models.esmfold2 import modeling_esmfold2_common as C  # noqa: E402
from transformers.models.esmfold2.configuration_esmfold2 import ESMFold2Config  # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


def _det_scatter(atom_features, atom_to_token_idx, n_tokens, atom_mask=None):
    oh = torch.nn.functional.one_hot(atom_to_token_idx.long(), n_tokens).to(atom_features.dtype)
    if atom_mask is not None:
        oh = oh * atom_mask.unsqueeze(-1).to(oh.dtype)
    summed = torch.einsum("...at,...ad->...td", oh, atom_features)
    cnt = oh.sum(-2).clamp(min=1).unsqueeze(-1)
    return summed / cnt


class _Det:
    """The det recipe's scatter on the fork module (module-global rebinding; restored on exit)."""
    def __enter__(self):
        self.orig = C.scatter_atom_to_token
        C.scatter_atom_to_token = _det_scatter
        return self

    def __exit__(self, *a):
        C.scatter_atom_to_token = self.orig


class _Holder(torch.nn.Module):
    def __init__(self, sh):
        super().__init__()
        self.structure_head = sh


def _head(seed=0, token_blocks=2, atom_blocks=1):
    torch.manual_seed(seed)
    cfg = ESMFold2Config()
    cfg.structure_head.diffusion_module.token_num_blocks = token_blocks
    cfg.structure_head.diffusion_module.atom_num_blocks = atom_blocks
    sh = C.DiffusionStructureHead(cfg).cuda().float().eval().requires_grad_(False)
    for blk in sh.diffusion_module.token_transformer.attn_blocks:
        torch.nn.init.normal_(blk.pair_bias_proj.weight, std=0.5)
    return _Holder(sh)


def _features(n_tok=29, atoms_per_tok=5, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    A = n_tok * atoms_per_tok
    dmc = ESMFold2Config().structure_head.diffusion_module
    tok_idx = torch.arange(n_tok, device=dev).repeat_interleave(atoms_per_tok).unsqueeze(0)
    kw = dict(
        z_trunk=torch.randn(1, n_tok, n_tok, dmc.c_z, device=dev, generator=g),
        s_inputs=torch.randn(1, n_tok, dmc.c_s_inputs, device=dev, generator=g),
        s_trunk=None,
        relative_position_encoding=torch.randn(1, n_tok, n_tok, dmc.c_z, device=dev, generator=g),
        ref_pos=torch.randn(1, A, 3, device=dev, generator=g) * 3.0,
        ref_charge=torch.zeros(1, A, device=dev),
        ref_mask=torch.ones(1, A, device=dev),
        ref_element=torch.nn.functional.one_hot(torch.randint(0, C.MAX_ATOMIC_NUMBER, (1, A), device=dev, generator=g), C.MAX_ATOMIC_NUMBER).float(),
        ref_atom_name_chars=torch.nn.functional.one_hot(torch.randint(0, C.CHAR_VOCAB_SIZE, (1, A, C.MAX_CHARS), device=dev, generator=g), C.CHAR_VOCAB_SIZE).float(),
        ref_space_uid=tok_idx.clone(),
        tok_idx=tok_idx,
        asym_id=torch.zeros(1, n_tok, dtype=torch.long, device=dev),
        residue_index=torch.arange(n_tok, device=dev).unsqueeze(0),
        entity_id=torch.zeros(1, n_tok, dtype=torch.long, device=dev),
        token_index=torch.arange(n_tok, device=dev).unsqueeze(0),
        sym_id=torch.zeros(1, n_tok, dtype=torch.long, device=dev),
        token_attention_mask=torch.ones(1, n_tok, device=dev),
        num_diffusion_samples=1,
        num_sampling_steps=12,
        return_atom_repr=False,
    )
    return kw


def _sample(holder, kw, seed=5):
    torch.manual_seed(seed)
    with torch.no_grad():
        out = holder.structure_head.sample(**kw)
    return {k: v.clone() for k, v in out.items() if torch.is_tensor(v)}


def _equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


@needs_cuda
def test_graph_replay_is_bitwise_and_scoped_to_one_sample_call():
    with _Det():
        h = _head(); kw = _features()
        ref = _sample(h, kw)
        assert _equal(ref, _sample(h, kw)), "eager sampler not reproducible under the det scatter: test premise broken"
        st = sg.enable(h)
        got1 = _sample(h, kw)
        n_steps = st.stats["eager"] + st.stats["replays"]
        assert st.stats["captures"] == 1 and st.stats["eager"] == 2 and st.stats["replays"] == n_steps - 2, sg.describe(h)
        assert st.stats["mismatch"] == 0 and st.stats["capture_failed"] == 0, sg.describe(h)
        assert _equal(ref, got1), {k: float((ref[k] - got1[k]).abs().max()) for k in ref}
        assert h.structure_head.diffusion_module._sg_scope is None, "the graph must be dropped when sample() returns"
        got2 = _sample(h, kw)                                   # a second sample() call captures afresh (new argument objects)
        assert st.stats["captures"] == 2 and _equal(ref, got2), sg.describe(h)
        sg.disable(h)
        assert "sample" not in vars(h.structure_head) and "forward" not in vars(h.structure_head.diffusion_module)
        assert _equal(ref, _sample(h, kw))


@needs_cuda
def test_a_capture_out_of_memory_is_retried_once_after_empty_cache_and_any_other_failure_is_named():
    """An out-of-memory error raised while capturing (the private pool cannot use the ordinary pool's cached segments) → empty_cache and
    ONE retry: capture_retries=1, the capture then succeeds and replays bitwise; a second OOM, or any other exception, is capture_failed with
    last_failure named and the sample() continues eager (the same digits)."""
    with _Det():
        h = _head(); kw = _features()
        ref = _sample(h, kw)
        st = sg.enable(h)
        dm = h.structure_head.diffusion_module
        inner = vars(dm)["forward"]._ef2_prev                                   # the module's own forward under the lever's wrapper
        budget = {"oom": 1}
        def flaky(*a, **k):
            if torch.cuda.is_current_stream_capturing() and budget["oom"] > 0:
                budget["oom"] -= 1
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB (test)")
            return inner(*a, **k)
        vars(dm)["forward"]._ef2_prev = flaky
        got = _sample(h, kw)
        assert st.stats["capture_retries"] == 1 and st.stats["captures"] == 1 and st.stats["capture_failed"] == 0 and st.last_failure is None, sg.describe(h)
        assert _equal(ref, got), {k: float((ref[k] - got[k]).abs().max()) for k in ref}
        budget["oom"] = 2                                                       # both attempts run out: named, eager, the same digits
        got2 = _sample(h, kw)
        assert st.stats["capture_retries"] == 2 and st.stats["capture_failed"] == 1 and "OutOfMemoryError" in (st.last_failure or "") and _equal(ref, got2), sg.describe(h)
        budget["oom"] = 0
        def broken(*a, **k):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("not an OOM (test)")
            return inner(*a, **k)
        vars(dm)["forward"]._ef2_prev = broken
        got3 = _sample(h, kw)
        assert st.stats["capture_retries"] == 2 and st.stats["capture_failed"] == 2 and "not an OOM" in st.last_failure and _equal(ref, got3), sg.describe(h)   # no retry for other failures
        assert "capture_retries=2" in sg.describe(h) and "capture_failed=2" in sg.describe(h)
        vars(dm)["forward"]._ef2_prev = inner
        sg.disable(h)


@needs_cuda
def test_one_step_sample_and_atom_repr_calls_stay_eager():
    with _Det():
        h = _head(); kw = _features()
        st = sg.enable(h)
        kw1 = dict(kw, num_sampling_steps=1)
        ref = None
        _sample(h, kw1)
        assert st.stats["captures"] == 0 and st.stats["replays"] == 0 and st.stats["eager"] >= 1, sg.describe(h)
        kw2 = dict(kw, return_atom_repr=True)                  # atom intermediates requested: the last step asks for them -> eager by name
        sg.disable(h); ref = _sample(h, kw2); sg.enable(h)
        got = _sample(h, kw2)
        assert _equal(ref, got), {k: float((ref[k] - got[k]).abs().max()) for k in ref}
        sg.disable(h)


@needs_cuda
def test_composes_with_pairbias_hoist_in_either_order():
    with _Det():
        h = _head(); kw = _features()
        ref = _sample(h, kw)
        for order in (("pba", "sg"), ("sg", "pba")):
            for name in order:
                (pba.enable(h, "hoist") if name == "pba" else sg.enable(h))
            got = _sample(h, kw)
            assert _equal(ref, got), (order, {k: float((ref[k] - got[k]).abs().max()) for k in ref})
            assert h._sg_state.stats["replays"] > 0 and h._pba_state.stats["served"] > 0, (sg.describe(h), pba.describe(h))
            for name in order:                                   # disable in the SAME order (inner wrapper removed first) ...
                (pba.disable(h) if name == "pba" else sg.disable(h))
            assert "sample" not in vars(h.structure_head), order
            assert _equal(ref, _sample(h, kw)), order
        pba.enable(h, "hoist"); sg.enable(h); sg.disable(h); pba.disable(h)   # ... and in reverse order
        assert "sample" not in vars(h.structure_head) and "forward" not in vars(h.structure_head.diffusion_module)
        assert _equal(ref, _sample(h, kw))


if __name__ == "__main__":
    names = [n for n in dir(sys.modules[__name__]) if n.startswith("test_")]
    passed = failed = 0
    for n in names:
        fn = getattr(sys.modules[__name__], n)
        marks = [m.kwargs.get("reason") for m in getattr(fn, "pytestmark", []) if m.name == "skipif" and m.args and m.args[0]]
        if getattr(fn, "_skip", False):
            marks.append(getattr(fn, "_skip_reason", ""))
        if marks:
            print(f"SKIP {n}: {marks[0]}"); continue
        try:
            fn(); passed += 1; print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            failed += 1; print(f"FAIL {n}: {type(e).__name__}: {e}")
    print(f"RESULT passed={passed} failed={failed}")
    sys.exit(1 if failed else 0)
