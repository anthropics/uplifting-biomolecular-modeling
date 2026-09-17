"""0.5.213: the triangle-attention bias operand staged ONCE per gathered plane (``ROWPAIR_TRIATT_STAGE=once``:
``triatt.StagedBias`` / ``StageSlot`` / ``stage_bias_once``; ``flash_triattn_serve.triangle_attention_prepped`` beside ``triangle_attention``) and the
tier door's policy word above the int32 bias bound (``ROWPAIR_TRIATT_BIG=qblocks|native|auto``), on CPU with a recording stand-in for the carried
kernel (the kernels themselves need CUDA: tests/gpu/test_rowpair_triatt_stage_gpu.py):

* the words: defaults ``per_call`` / ``qblocks`` (= opt_core <= 0.5.212 byte for byte), env override, refusal by name of unknown words;
* ``stage=once`` on the flash word: the plane is prepared ONCE per plane (one ``prep_bias`` for any number of row windows), every window launches the
  prepared entry, outputs ``torch.equal`` the per-call path's; a second plane re-stages; leaving the armed span releases the operand; census words
  ``triatt_stage=once:flash_triattn``, ``triatt_stage_gib``, ``triatt_stage_planes``;
* outside an armed span (a kit calling the core directly) and under ``per_call`` the recorded kernel calls are exactly today's (whole-plane fp32
  contiguous bias per call);
* a tier word whose row is not a staged kind (here: the CPU box resolves ``cueq``) steps aside BY NAME to the per-call statements (census
  ``triatt_stage_aside=unstaged_row:…``) and records ``triatt_cells=nearest:<cell>|measured:<cell>``;
* above the (patched-small) bound: ``ROWPAIR_TRIATT_BIG=native|auto`` with a tier word that does not resolve to triattn_native here -> ``qblocks`` by name
  (census ``triatt_big=qblocks:tier_row=…``); ``stage=once`` there prepares per query block once and serves every window from the prepared blocks,
  outputs equal to the per-call q-block path;
* ``native_staged_bytes`` = the sealed package's m1 staging size (16.5 S^2 bytes at H=4).
"""
from __future__ import annotations

import contextlib
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused                            # noqa: E402
from opt_core.mem.rowpair import triatt as RA                             # noqa: E402
from opt_core.mem.rowpair.evidence import schedule, reset_schedule        # noqa: E402

H, D, S = 2, 8, 24
FP32_TOO = ("bfloat16", "float16", "float32")


def _eager_core(q, k, v, biases):
    a = torch.einsum("...qd,...kd->...qk", q * (D ** -0.5), k)
    for b in biases:
        a = a + b
    return torch.einsum("...qk,...kd->...qd", torch.softmax(a, dim=-1), v)


def _same(a, b) -> bool:
    """Bitwise equality with NaN == NaN (a fully-masked row's softmax is NaN in the materialised statement)."""
    return tuple(a.shape) == tuple(b.shape) and a.dtype == b.dtype and bool(((a == b) | (a.isnan() & b.isnan())).all())


class _Calls(object):
    def __init__(self):
        self.n = 0

    def __call__(self, q, k, v, biases):
        self.n += 1
        return _eager_core(q, k, v, biases)


def _plane(g, transposed=False):
    """The tp line's gathered bias: channel-last [S, S, H] -> the kit's [1, 1, H, S, S] VIEW (starting: movedim; ending: + transpose)."""
    tb_cl = torch.randn(S, S, H, generator=g)
    v = tb_cl.movedim(-1, 0)
    if transposed:
        v = v.transpose(-1, -2)
    return v[None, None]


def _windows(g, n=3, rows=4):
    out = []
    for _ in range(n):
        q, k, v = (torch.randn(1, rows, H, S, D, generator=g) for _ in range(3))
        keep = torch.rand(1, rows, 1, 1, S, generator=g) > 0.2
        keep[:, -1] = False                                                  # a fully-masked (dead) row in every window
        out.append((q, k, v, torch.zeros(1, rows, 1, 1, S).masked_fill(~keep, float("-inf"))))
    return out


class _FakeFlash(object):
    """Stand-in for the carried kernel module reached through flash_triattn_serve.kernel_module(): the per-call entry, plus the module's OWN
    staged entries (``prep_bias`` / ``flash_triangle_attention_prepped``, which the serve layer's prep_bias / flash_prepped defer to when a module
    carries them), all recording their calls and computing the materialised statement in fp32."""
    BuildFailed = ()

    def __init__(self):
        self.calls, self.preps, self.prepped_calls = [], [], []

    def flash_supported(self, q, k, v, bias, mask=None):
        return True, "ok"

    def _attn(self, q, k, v, bias32, mask, scale):
        a = torch.einsum("...qd,...kd->...qk", q * float(scale), k) + bias32.to(q.dtype)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        return torch.einsum("...qk,...kd->...qd", torch.softmax(a, dim=-1), v)

    def flash_triangle_attention(self, q, k, v, bias, mask=None, scale=None, **kw):
        self.calls.append(dict(q=tuple(q.shape), bias=tuple(bias.shape), bias_dtype=bias.dtype, bias_contig=bool(bias.is_contiguous())))
        return self._attn(q, k, v, bias.float(), mask, scale)

    def prep_bias(self, bias, *, head_dim, dtype, device=None):
        self.preps.append(dict(shape=tuple(bias.shape), dtype=bias.dtype, contig=bool(bias.is_contiguous())))
        b32 = bias.float().contiguous()
        return types.SimpleNamespace(bias32=b32, nbytes=int(b32.numel()) * 4, D=int(head_dim), dtype=dtype,
                                     dims=lambda: (int(b32.shape[0]), int(b32.shape[2]), int(b32.shape[3]), int(b32.shape[4])))

    def flash_triangle_attention_prepped(self, q, k, v, prep, mask=None, scale=None, **kw):
        self.prepped_calls.append(dict(q=tuple(q.shape), bias=tuple(prep.bias32.shape)))
        return self._attn(q, k, v, prep.bias32, mask, scale)


@pytest.fixture()
def fake(monkeypatch):
    from opt_core.kernels import flash_triattn_serve as F1
    f = _FakeFlash()
    monkeypatch.setattr(F1, "_KMOD", f)
    monkeypatch.setattr(RA, "_lever_ready", lambda F1_, word, q: None)
    for e in (RA.ENV_TRIATT_STAGE, RA.ENV_TRIATT_BIG, RA.ENV_TRIATT_INT32_GUARD, RA.ENV_TRIATT_FLASH_QBLOCK, RA.ENV_TRIATT_CORE):
        monkeypatch.delenv(e, raising=False)
    RA.STAGE.release()
    reset_schedule()
    yield f
    RA.STAGE.release()


def test_words_defaults_env_and_refusals(monkeypatch):
    for e in (RA.ENV_TRIATT_STAGE, RA.ENV_TRIATT_BIG):
        monkeypatch.delenv(e, raising=False)
    assert RA.stage_word() == RA.STAGE_DEFAULT == "per_call" and RA.STAGE_WORDS == ("per_call", "once")
    assert RA.big_word() == RA.BIG_DEFAULT == "qblocks" and RA.BIG_WORDS == ("qblocks", "native", "auto")
    assert RA.stage_word("once") == "once" and RA.big_word("AUTO") == "auto"
    monkeypatch.setenv(RA.ENV_TRIATT_STAGE, "once"); monkeypatch.setenv(RA.ENV_TRIATT_BIG, "native")
    assert RA.stage_word() == "once" and RA.big_word() == "native"
    assert RA.stage_word("per_call") == "per_call"                            # the argument wins over the env
    for bad in ("yes", "1", "stage"):
        with pytest.raises(RowpairRefused, match=RA.ENV_TRIATT_STAGE):
            RA.stage_word(bad)
        with pytest.raises(RowpairRefused, match=RA.ENV_TRIATT_BIG):
            RA.big_word(bad)
    # the slot: idle = per_call; armed per plane; nested arming restores
    sl = RA.StageSlot()
    assert not sl.once()
    with sl.plane("once"):
        assert sl.once()
        with sl.plane("per_call"):
            assert not sl.once()
        assert sl.once()
    assert not sl.once() and sl.staged is None
    assert RA.native_staged_bytes(4, 24035) == 4 * 188 * 4 * 189 * 4096 * 4          # [B*H, nq, 4*(nk+1), 4096] fp32, nq = nk = ceil(S/128)
    assert RA.native_staged_bytes(4, 16304) == 4 * 128 * 4 * 129 * 4096 * 4 and 16.0 < RA.native_staged_bytes(4, 16304) / 16304 ** 2 < 16.6


@pytest.mark.parametrize("inference_mode", [False, True])           # the kits predict under torch.inference_mode(): inference tensors carry no version counter
def test_stage_once_flash_word_prepares_once_per_plane_and_is_bitwise_per_call(fake, inference_mode):
    with (torch.inference_mode() if inference_mode else contextlib.nullcontext()):
        _stage_once_flash_word_body(fake)


def _stage_once_flash_word_body(fake):
    g = torch.Generator().manual_seed(5)
    tb = _plane(g)
    tbT = tb[0, 0].transpose(-1, -2)[None, None]                              # the ending orientation's view of another plane object (same storage, other strides)
    wins = _windows(g)
    stock = _Calls()
    L = RA.core_ledger(); L.clear()
    core = RA.attention_core(stock, kernel="flash_triattn", stock_qblock=8, serve_dtypes=FP32_TOO)
    # per_call (idle slot = a kit calling the core outside triatt_update_): today's calls -- the whole-plane fp32 CONTIGUOUS bias per window
    ref = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
    assert len(fake.calls) == 3 and not fake.preps and all(c["bias"] == (1, 1, H, S, S) and c["bias_dtype"] == torch.float32 and c["bias_contig"] for c in fake.calls)
    assert stock.n == 0 and L.served == 3
    # per_call inside an armed per_call span (triatt_update_'s default): the same calls
    fake.calls.clear()
    with RA.STAGE.plane("per_call"):
        ref2 = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
    assert len(fake.calls) == 3 and not fake.preps and all(_same(a, b) for a, b in zip(ref, ref2))
    assert schedule().get("triatt_stage") == "per_call", schedule()
    # once: ONE prep for the plane (read through the view's own strides: no copy handed), every window on the prepared entry, bitwise per_call
    fake.calls.clear()
    with RA.STAGE.plane("once"):
        outs = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
        assert len(fake.preps) == 1 and fake.preps[0]["shape"] == (1, 1, H, S, S) and fake.preps[0]["contig"] is False, fake.preps
        assert not fake.calls and len(fake.prepped_calls) == 3, (fake.calls, fake.prepped_calls)
        assert all(_same(a, b) for a, b in zip(outs, ref))
        sb = RA.STAGE.staged
        assert isinstance(sb, RA.StagedBias) and sb.kernel == "flash_triattn" and sb.n_staged == 1 and sb.n_served == 3 and sb.nbytes == H * S * S * 4
        assert RA.stage_bias_once(tb, "flash_triattn") is sb                   # the same view of the same plane for the same kernel: the held staging
        assert schedule().get("triatt_stage") == "once:flash_triattn" and schedule().get("triatt_stage_planes") == 1 and "triatt_stage_aside" not in schedule(), schedule()
        # another plane (the ending orientation's transposed view): re-staged once, served from then on
        outsT = [core(q, k, v, [mb, tbT]) for (q, k, v, mb) in wins]
        assert len(fake.preps) == 2 and RA.STAGE.staged is not sb and RA.STAGE.staged.n_served == 3 and RA.STAGE.restaged == 1
        refT = [_FakeFlash()._attn(q, k, v, tbT.float(), (mb == 0).expand(1, q.shape[1], 1, 1, S), D ** -0.5) for (q, k, v, mb) in wins]
        assert all(_same(a, b) for a, b in zip(outsT, refT))
    assert RA.STAGE.staged is None and not RA.STAGE.once()                    # released with the plane
    assert schedule().get("triatt_stage_served") == 6 and stock.n == 0 and L.served == 3 + 3 + 6


def test_stage_once_tier_word_with_an_unstaged_row_steps_aside_by_name(fake, monkeypatch):
    """The tier door below the bound under stage=once: kernels.triattn resolves the word at the window's call class (census triatt_cells=); a
    row that is not a staged kind (this CPU box: not triattn_native / flash) takes the per-call provider door exactly as before, named once."""
    import opt_core.attn.pair_fused as PF
    pf_calls = []

    def provider(q, k, v, bias, mask5=None, *, core=None, scale=None):
        pf_calls.append((tuple(q.shape), tuple(bias.shape), bool(bias.is_contiguous())))
        return _FakeFlash()._attn(q, k, v, bias.float(), mask5, scale)
    monkeypatch.setattr(PF, "core_attention", provider)
    g = torch.Generator().manual_seed(6)
    tb = _plane(g)
    wins = _windows(g, n=2)
    core = RA.attention_core(_Calls(), kernel="tier:big", stock_qblock=8, serve_dtypes=FP32_TOO)
    ref = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
    assert len(pf_calls) == 2 and all(c[2] for c in pf_calls)                 # per call: the contiguous whole-plane copy, as before
    del pf_calls[:]
    with RA.STAGE.plane("once"):
        outs = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
    assert len(pf_calls) == 2 and not fake.preps and all(_same(a, b) for a, b in zip(outs, ref))
    sch = schedule()
    assert str(sch.get("triatt_stage_aside", "")).startswith("unstaged_row:"), sch
    assert str(sch.get("triatt_cells", "")).split(":")[0] in ("nearest", "measured", "refused"), sch
    assert "triatt_tier_row" in sch or str(sch.get("triatt_cells")).startswith("refused"), sch


@pytest.mark.parametrize("bigword", ["native", "auto", "qblocks"])
def test_big_words_above_the_bound_on_a_box_without_triattn_native_take_qblocks_by_name(fake, monkeypatch, bigword):
    """ROWPAIR_TRIATT_BIG with a tier word above the int32 bias bound (patched small): native / auto need kernels.triattn to resolve the word to
    triattn_native at the class -- not on this box -> qblocks BY NAME (census triatt_big=qblocks:tier_row=...); qblocks = today's word. With
    stage=once the query blocks are prepared ONCE per plane and every window is served from them; outputs equal the per-call q-block path."""
    from opt_core.kernels import flash_triattn_serve as F1
    monkeypatch.setattr(RA, "INT32_MAX", H * S * S - 1)                       # H*S*S "above the bound"; rows per launch = (H*S*S-1)//(H*S*D) = 2 -> two launches per 4-row window
    monkeypatch.setenv(RA.ENV_TRIATT_FLASH_QBLOCK, "8")
    monkeypatch.setenv(RA.ENV_TRIATT_BIG, bigword)
    g = torch.Generator().manual_seed(7)
    tb = _plane(g, transposed=True)
    wins = _windows(g, n=2)
    nb = len(F1.query_blocks(S, 8))
    core = RA.attention_core(_Calls(), kernel="tier:big", stock_qblock=8, serve_dtypes=FP32_TOO)
    ref = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
    assert len(fake.calls) == 2 * 2 * nb and not fake.preps                   # per call: every launch preps its own block (2 windows x 2 row launches x nb blocks)
    sch = schedule()
    assert str(sch.get("triatt_big", "")).startswith("qblocks:"), sch
    if bigword != "qblocks":
        assert "tier_row=" in sch["triatt_big"], sch
    assert sch.get("triatt_core_tier_big") == "flash_triattn:bias_elems>int32", sch
    fake.calls.clear()
    with RA.STAGE.plane("once"):
        outs = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
        sb = RA.STAGE.staged
        assert sb is not None and sb.kernel == "flash_qblocks" and sb.n_staged == 1 and len(sb.ops) == nb
    assert len(fake.preps) == nb and not fake.calls and len(fake.prepped_calls) == 2 * 2 * nb, (len(fake.preps), len(fake.calls), len(fake.prepped_calls))
    assert all(_same(a, b) for a, b in zip(outs, ref))
    exp = [RA.attend_query_blocks(_eager_core, q, k, v, [mb, tb], None) for (q, k, v, mb) in wins]
    assert all(torch.allclose(a, b, rtol=0, atol=1e-5, equal_nan=True) for a, b in zip(outs, exp))


def test_triatt_update_default_is_per_call_and_the_env_arms_once(monkeypatch):
    """triatt_update_(stage=None) reads ROWPAIR_TRIATT_STAGE: unset = per_call (the slot armed per plane in per_call mode = today's per-window
    staging); the CPU rank-thread line runs the torch word either way (nothing to stage) and the shard update is unchanged."""
    from opt_core.mem.rowpair.dist import Layout
    from opt_core.testing import run_ranks
    monkeypatch.delenv(RA.ENV_TRIATT_STAGE, raising=False)
    N_, C_ = 16, 8
    g = torch.Generator().manual_seed(9)
    z = torch.randn(N_, N_, C_, generator=g)
    w_b = torch.randn(H, C_, generator=g) * 0.3
    w_q, w_k, w_v = (torch.randn(C_, C_, generator=g) * 0.3 for _ in range(3))
    seen = []

    def fns_of():
        def ln(x):
            return torch.nn.functional.layer_norm(x, (C_,))

        def bias(x):
            return x @ w_b.t()

        def attend(x, mask_rows, tb_full, rows):
            seen.append((RA.stage_slot().mode, RA.stage_slot().armed, RA.stage_slot() is not RA.STAGE))
            q, k, v = x @ w_q, x @ w_k, x @ w_v                                # [rows, N, C]: one "head" of width C over all keys, bias head 0
            a = torch.einsum("inc,imc->inm", q, k) + tb_full[..., 0][None] + (mask_rows[:, None, :] - 1.0) * 1e9
            return torch.softmax(a, -1) @ v
        return RA.TriAttFns(ln, bias, attend)

    def dense():
        f = fns_of()
        x = f.ln(z)
        return z + f.attend(x, torch.ones(N_, N_), f.bias(x), (0, N_))

    def rank(r, P):
        lay = Layout(N_, P, r, 4)
        zs = z[lay.r0:lay.r1].clone()
        return RA.triatt_update_(fns_of(), zs, None, lay, rows=4)

    for env in (None, "once"):
        if env:
            monkeypatch.setenv(RA.ENV_TRIATT_STAGE, env)
        d = dense()
        del seen[:]
        shards = run_ranks(2, rank)
        assert torch.allclose(torch.cat(shards, 0), d, atol=1e-5)
        assert seen and all(m == (env or "per_call") and armed and own for m, armed, own in seen), seen     # every rank thread arms its OWN slot
        assert not RA.STAGE.armed and RA.STAGE.staged is None


# --------------------------------------------------------------------------- 0.5.220.x: the staged members table + the sealed package's bytes
def test_native_stage_member_table_per_card_and_band():
    """Which sealed triattn_native member the stage-once glue drives, per (cc, dtype, head_dim, S): the package router's own bands (pure table)."""
    f = RA.native_stage_member
    # cc 9.0 (H100), bf16, D32: m1 for 512<=S<=3072 and S>4096, the high band cuda_c for 3072<S<=4096, nothing staged below 512 (k13 / cuda)
    for S, want in [(128, None), (511, None), (512, "cuda_b"), (2048, "cuda_b"), (3072, "cuda_b"), (3073, "cuda_c"), (3762, "cuda_c"), (4096, "cuda_c"),
                    (4097, "cuda_b"), (6072, "cuda_b"), (8192, "cuda_b"), (16304, "cuda_b"), (24035, "cuda_b")]:
        assert f((9, 0), "bf16", 32, S) == want, (S, f((9, 0), "bf16", 32, S), want)
        assert f("9.0", "bf16", 32, S) == want and f(9.0, "bf16", 32, S) == want
    assert f((9, 0), "bf16", 32, 256, strided=False) is None                 # k13 (contiguous, S < 512): per call by name
    for D_ in (16, 64, 128):                                                  # the Triton member's classes on 9.0: no staged entry
        assert f((9, 0), "bf16", D_, 8192) is None, D_
    assert f((9, 0), "fp16", 32, 8192) is None
    # cc 8.0 (A100): the sm_80 member serves bf16 D16/32/64 at every S (its extension is a member of the active package for cc 8.0)
    for S in (128, 2048, 3762, 6072, 8192, 24035):
        for D_ in (16, 32, 64):
            assert f((8, 0), "bf16", D_, S) == "cuda_80", (S, D_)
    assert f((8, 0), "fp16", 32, 2048) is None and f((8, 0), "bf16", 128, 2048) is None
    for cc in ((8, 6), (8, 9), (10, 0), (12, 0)):                             # other cards: the package has no CUDA member -> nothing staged
        assert f(cc, "bf16", 32, 4096) is None, cc
    assert set(RA.NATIVE_STAGED_MEMBERS) == {"cuda_b", "cuda_c", "cuda_80"}


SEALED_TRIATTN_NATIVE_PKG_DIGEST = ("9bbbf02a0a1f963fe17bbb465c279806fe2a59c41a447ee8d19e7381b4f3c7ce", 181)   # sha256 over (relpath, sha256(bytes)) of kernels/triattn/triattn_native/pkg/**, file count


def _pkg_tree_digest():
    import hashlib
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opt_core", "kernels", "triattn", "triattn_native", "pkg")
    h = hashlib.sha256(); n = 0
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for f in sorted(fns):
            if f.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(dp, f)
            with open(path, "rb") as fh:
                b = fh.read()
            h.update(os.path.relpath(path, root).replace(os.sep, "/").encode() + b"\0" + hashlib.sha256(b).hexdigest().encode() + b"\n"); n += 1
    return h.hexdigest(), n


def test_sealed_triattn_native_packages_are_byte_identical():
    """The stage-once glue drives the sealed triattn_native members through their public entry points: NO payload byte changes (pkg/** digest pinned;
    a new package generation lands beside with its own pin)."""
    assert _pkg_tree_digest() == SEALED_TRIATTN_NATIVE_PKG_DIGEST, _pkg_tree_digest()


# --------------------------------------------------------------------------- 0.5.220.4: the per-plane key-mask decision (no per-window host readback)
class _FakeMember(object):
    """A triattn_native member's public staged-call surface in plain torch (CPU): stage_mask -> keyany (keys some row attends), stage_bias folds
    ~keyany as -inf columns (what the sealed members do), triangle_attention(..., bias_staged=) = the per-call statement when None."""
    calls = {"stage_bias": 0, "fwd": 0}

    @staticmethod
    def geometry(D, small=False):
        return {"R": 4, "small_max": 0, "small_max_rows": 0}

    @staticmethod
    def stage_mask(mask5, B, N, S, R=None, device=None):
        m = mask5.reshape(B, N, S) if mask5.dtype == torch.bool else (mask5.reshape(B, N, S) != 0)
        return (m.any(dim=1), None, None)

    @staticmethod
    def stage_bias(bias5, scale, keyany=None):
        _FakeMember.calls["stage_bias"] += 1
        st = bias5[:, 0].float() / 1.0
        if keyany is not None:
            st = st.masked_fill(~keyany.reshape(keyany.shape[0], 1, 1, -1), float("-inf"))
        return st

    @staticmethod
    def triangle_attention_sm80(q, k, v, bias, mask=None, scale=None, *, bias_staged=None):
        _FakeMember.calls["fwd"] += 1
        B, N, Hh, S_, Dd = q.shape
        if bias_staged is None:
            keyany = _FakeMember.stage_mask(mask, B, N, S_)[0] if mask is not None else None
            bias_staged = _FakeMember.stage_bias(bias, scale, keyany)
        a = torch.einsum("bnhqd,bnhkd->bnhqk", q.float() * float(scale), k.float()) + bias_staged[:, None]
        if mask is not None:
            keep = (mask if mask.dtype == torch.bool else mask != 0).reshape(B, N, 1, 1, S_)
            dead = ~keep.any(dim=-1, keepdim=True)
            a = torch.where(keep | dead, a, torch.full_like(a, float("-inf")))
            a = torch.where(dead.expand_as(a), torch.zeros_like(a), a)
        return torch.einsum("bnhqk,bnhkd->bnhqd", torch.softmax(a, -1), v.float()).to(q.dtype)


def _native_cpu_rig(monkeypatch):
    """Route a tier word's windows to the fake member: kernels.triattn 'resolves' triattn_native, the per-call provider door and the staged member are the fake."""
    import types
    import opt_core.attn.pair_fused as PF
    monkeypatch.setattr(PF, "resolve_tier_core", lambda *a, **k: types.SimpleNamespace(row="triattn_native", measured=True, cell="fake"))
    monkeypatch.setattr(RA, "_native_member", lambda q5, k5, v5, tb5, mask5: ("cuda_80", _FakeMember, object()))
    monkeypatch.setattr(PF, "core_attention", lambda q, k, v, bias, mask5=None, *, core=None, scale=None: _FakeMember.triangle_attention_sm80(q, k, v, bias if bias.dim() == 5 else bias.unsqueeze(0), mask5, scale))
    monkeypatch.setattr(RA, "_lever_ready", lambda F1_, word, q: None)
    syncs = {"n": 0}
    real = RA._host_bools

    def counting(t):
        syncs["n"] += 1
        return real(t)
    monkeypatch.setattr(RA, "_host_bools", counting)
    return syncs


def _native_plane_and_windows(seed, Hh=2, Dd=16, S_=64, rows=4, dead_tail=True):
    g = torch.Generator().manual_seed(seed)
    tb = (torch.randn(1, 1, Hh, S_, S_, generator=g) * 0.7).to(torch.bfloat16)
    keep = torch.rand(S_, S_, generator=g) > 0.2                                 # the plane mask [R=S, N=S] (True = attend)
    keep[0::rows, :] = True                                                      # the first row of every window attends every (unpadded) key: all live windows share ONE attended-key set
    keep[:, S_ - 5:] = False                                                     # padded keys, every row
    if dead_tail:
        keep[S_ - rows:, :] = False                                              # the last window: all-dead rows -> its attended-key set is EMPTY (differs -> ONE re-stage by name)
    wins = []
    for i0 in range(0, S_, rows):
        q, k, v = ((torch.randn(1, rows, Hh, S_, Dd, generator=g) * 0.8).to(torch.bfloat16) for _ in range(3))
        mb = torch.zeros(1, rows, 1, 1, S_).masked_fill(~keep[i0:i0 + rows][None, :, None, None, :], float("-inf")).to(torch.bfloat16)
        wins.append((q, k, v, mb))
    return tb, keep, wins


@pytest.mark.parametrize("inference_mode", [False, True])
def test_stage_once_key_plan_one_readback_per_plane_and_bitwise(monkeypatch, inference_mode):
    """triatt_update_'s way: plane('once', mask=<plane mask>, rows=) + window(w) per window -> the attended-key decision of all 16 windows is
    made once per plane (ONE host readback), the all-dead tail window re-stages by name, and every output is bitwise the per-call door's."""
    syncs = _native_cpu_rig(monkeypatch)
    with (torch.inference_mode() if inference_mode else contextlib.nullcontext()):
        tb, keep, wins = _native_plane_and_windows(21)
        core = RA.attention_core(_Calls(), kernel="tier:big", stock_qblock=64, serve_dtypes=FP32_TOO + ("bfloat16",))
        ref = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
        assert syncs["n"] == 0
        for plane_i in range(2):                                                 # two planes (orientations) inside one armed span each
            with RA.STAGE.plane("once", mask=keep, rows=4) as slot:
                outs = []
                for w, (q, k, v, mb) in enumerate(wins):
                    slot.window(w)
                    outs.append(core(q, k, v, [mb, tb]))
                sb = RA.STAGE.staged
                assert sb is not None and sb.kernel == "triattn_native" and sb.n_served == len(wins), schedule()
                assert sb.n_staged == 2 and slot.restaged == 1, (sb.n_staged, slot.restaged)      # window 0 + the dead-tail window
                assert slot.keysyncs == 1, slot.keysyncs
                sch = schedule()
                assert sch.get("triatt_stage_keyplan") == "plan" and sch.get("triatt_stage_keysyncs") == 1 and sch.get("triatt_stage_member") == "cuda_80", sch
            assert all(_same(a, b) for a, b in zip(outs, ref)), [float((a.float() - b.float()).abs().max()) for a, b in zip(outs, ref)]
        assert syncs["n"] == 2, syncs                                                # ONE readback per plane


def test_stage_once_key_plan_fallbacks_by_name(monkeypatch):
    """No plane mask given (a kit's own loop on <= 0.5.220.3's signature) -> per-window compares as before (counted, 'percall'); a plane mask that
    does not match the windows' own masks -> the plan retires BY NAME ('mismatch') at its one readback and windows compare per call; both bitwise."""
    syncs = _native_cpu_rig(monkeypatch)
    tb, keep, wins = _native_plane_and_windows(22)
    core = RA.attention_core(_Calls(), kernel="tier:big", stock_qblock=64, serve_dtypes=FP32_TOO + ("bfloat16",))
    ref = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
    with RA.STAGE.plane("once") as slot:                                             # no mask= / rows=: 0.5.218.0's per-window decision
        outs = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
        assert slot.keysyncs == len(wins) - 1 and schedule().get("triatt_stage_keyplan") == "percall", (slot.keysyncs, schedule())
        assert slot.restaged == 1
    assert all(_same(a, b) for a, b in zip(outs, ref))
    wrong = keep.clone(); wrong[0, 0] = ~wrong[0, 0]; wrong[0, 1] = ~wrong[0, 1]      # a plane mask whose window-0 slice differs from the window's own mask
    if bool((wrong[0:4].any(0) == keep[0:4].any(0)).all()):
        wrong[1:4, 7] = False; wrong[0, 7] = not bool(keep[0:4, 7].any())
    reset_schedule()
    with RA.STAGE.plane("once", mask=wrong, rows=4) as slot:
        outs = []
        for w, (q, k, v, mb) in enumerate(wins):
            slot.window(w); outs.append(core(q, k, v, [mb, tb]))
        sch = schedule()
        assert sch.get("triatt_stage_keyplan") == "mismatch", sch
        assert slot.keysyncs == 1 + (len(wins) - 1), slot.keysyncs                    # the plan's one readback + per-window compares after it retired
    assert all(_same(a, b) for a, b in zip(outs, ref))


def test_window_sigs_and_key_decision_table():
    keep = torch.ones(10, 12, dtype=torch.bool); keep[8:, :] = False; keep[4:8, 3] = False; keep[0:4, 3] = True
    blocks = list(RA.zblocks(10, 4))                                                # (0,4) (4,8) (8,10): uniform with a short tail
    sig = RA._window_sigs(keep, blocks)
    assert sig.shape == (3, 12) and bool(sig[0].all()) and not bool(sig[1][3]) and not bool(sig[2].any())
    sig2 = RA._window_sigs(keep.float(), [(0, 4), (4, 10)])                          # ragged blocks: per-window statements, same sets
    assert bool((sig2[1] == (keep[4:10].any(0))).all())


# --------------------------------------------------------------------------- 0.5.220.5: the no_room guard (by name)
def test_stage_need_bytes_size_law():
    GiB = 2.0 ** 30
    assert RA.native_staged_bytes(4, 8152) == 65536 * 4 * 64 * 65                          # 1.016 GiB: the measured triatt_stage_gib 1.02 at 8,152 tokens H4
    assert abs(RA.native_staged_bytes(4, 16304) / GiB - 4.03) < 0.01                         # measured 4.03
    assert RA.stage_need_bytes("triattn_native", 4, 8192, member="cuda_80") == 4 * 4 * 8192 * 8192 + 2 * 4 * 8192 * 8192
    f = RA.stage_need_bytes("flash_triattn", 4, 70320)
    assert abs((6 * 4 * 70320 * 70320) / GiB - 110.5) < 0.1 and f > 6 * 4 * 70320 * 70320
    q = RA.stage_need_bytes("flash_qblocks", 4, 70320, qblock=2048)
    assert q == 6 * 4 * 70320 * 70320 + 4 * 4 * (70320 // 32 + 1) + 4 * 4 * 2048 * 70320
    assert abs(6 * 4 * 2048 * 70320 / GiB - 3.22) < 0.01                                     # ONE query block's prepared bytes at 70,320 tokens (2048-query block, H4)
    assert RA.stage_room_margin(10 << 30) == 1 << 30 and RA.stage_room_margin(100 << 30) == (100 << 30) // 20


@pytest.mark.parametrize("case", ["room", "no_room", "room_off"])
def test_stage_once_no_room_guard_by_name(monkeypatch, case):
    """Before a plane is staged once its size law + margin is checked against the device's free bytes (a fake reading here): room -> staged as
    before; no room -> the plane runs per call BY NAME (triatt_stage_aside=no_room=<windows>, need/free census, nothing staged) and the next plane
    asks again; ROWPAIR_TRIATT_STAGE_ROOM=0 -> no check. Outputs bitwise the per-call door's in every case."""
    syncs = _native_cpu_rig(monkeypatch)
    need = RA.stage_need_bytes("triattn_native", 2, 64, member="cuda_80")
    free = {"room": need + RA.stage_room_margin(need), "no_room": need + RA.stage_room_margin(need) - 1, "room_off": 0}[case]
    readings = {"n": 0}

    def fake_free(device):
        readings["n"] += 1
        return free
    monkeypatch.setattr(RA, "_free_device_bytes", fake_free)
    if case == "room_off":
        monkeypatch.setenv(RA.ENV_TRIATT_STAGE_ROOM, "0")
    tb, keep, wins = _native_plane_and_windows(23)
    core = RA.attention_core(_Calls(), kernel="tier:big", stock_qblock=64, serve_dtypes=FP32_TOO + ("bfloat16",))
    ref = [core(q, k, v, [mb, tb]) for (q, k, v, mb) in wins]
    for plane_i in range(2):
        reset_schedule()
        with RA.STAGE.plane("once", mask=keep, rows=4) as slot:
            outs = []
            for w, (q, k, v, mb) in enumerate(wins):
                slot.window(w); outs.append(core(q, k, v, [mb, tb]))
            sch = schedule(); sb = RA.STAGE.staged
            if case == "no_room":
                assert sb is not None and sb.ops is None and sb.n_staged == 0 and slot.no_room == sb.key, (sch, slot.no_room)
                assert sch.get("triatt_stage_aside") == "no_room=%d" % len(wins), sch                 # every window of the plane, counted; single token
                assert sch.get("triatt_stage_need_gib") is not None and sch.get("triatt_stage_free_gib") is not None
                assert readings["n"] == plane_i + 1                                                    # ONE reading per plane (not per window); the next plane asks again
            else:
                assert sb is not None and sb.n_staged == 2 and sb.n_served == len(wins) and "triatt_stage_aside" not in sch, sch
                assert readings["n"] == (0 if case == "room_off" else plane_i + 1)
        assert all(_same(a, b) for a, b in zip(outs, ref))
