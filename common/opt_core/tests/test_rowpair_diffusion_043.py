"""opt_core.mem.rowpair.diffusion / sampler_hook — CPU tests: the diffusion pair conditioning, the diffusion transformer over a
row-sharded conditioned pair, the atom-attention band and the replicated-draw guard at P in {2, 3, 4} under gloo (the launcher is the
launcher: ``launch.run_sharded(P, entry, backend="gloo", cpu_ok=True)``) against an INDEPENDENT dense reference — a synthetic 2-block
diffusion transformer + pair conditioning with random weights (fp32), 3 steps, 2 samples, whole-tensor torch statements that never touch
opt_core. P = 1 refuses by name (the kit's ``--n_gpu 1`` path runs the engine's own module). Pass class: fp32 max|diff| <= 1e-5 vs dense
(and the family's ``_ok`` class); torch.equal REPORTED per quantity; data movement (band, extras, noise broadcast) bit-exact.

Run: ``python -m pytest tests/test_rowpair_diffusion_043.py -q -rfE`` (torch required for the ``mp`` / tensor tests).
"""
from __future__ import annotations

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                      # the opt_core checkout under test

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import diffusion as DF  # noqa: E402
from opt_core.mem.rowpair import sampler_hook as SH  # noqa: E402
from opt_core.mem.rowpair.dist import Layout  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    HAVE_TORCH = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not importable")
TOL32 = 1e-5                                                    # the seam's own bar (fp32 CPU, same statements, M differs only)

if HAVE_TORCH:
    from tests.test_rowpair_logic import _cmp, _ok, _tol       # noqa: E402  the family's ONE sharded-vs-dense comparison


# ================================================================================================================ pure (no group)

def _engine_tokens():
    """Engine names derived from the release tree beside common/ (empty when the core is checked out alone) — the hygiene suite's one reader."""
    try:
        from tests.test_instances_backend_hygiene import engine_tokens
    except Exception:  # noqa: BLE001
        return []
    return engine_tokens()


def _mentions(token: str, text: str) -> bool:
    """The hygiene suite's ONE word matcher (a token inside another word is not a mention)."""
    try:
        from tests.test_instances_backend_hygiene import mentions
    except Exception:  # noqa: BLE001
        return re.search(r"(?<![a-z0-9_])" + re.escape(token.lower()) + r"(?![a-z0-9_])", text.lower()) is not None
    return mentions(token, text)

def test_module_is_engine_free_and_imports_lazily():
    for mod in (DF, SH):
        src = open(mod.__file__).read().lower()
        for name in ("openfold", "boltz", "protenix", "alphafold") + tuple(_engine_tokens()):
            # the name as a WORD (not inside another identifier: a kit named like a substring of an ordinary word is not a mention)
            assert not _mentions(name, src), (mod.__name__, name)
        assert not re.search(r"^\s*(import torch|from torch\b)", src, re.M), mod.__name__   # torch only through ._torch / the caller's tensors
        assert "torch.distributed" not in src, mod.__name__                 # communication only through .dist / .shard / .bcast


def test_row_blocks_come_from_the_one_chooser(monkeypatch):
    """`DiffusionSchedule` sizes its four row blocks through `shard.choose_block_rows`: given (0 = whole shard) | this statement's row-count
    pin (`env:<NAME>`) | the work budget (multiples of ROW_MULTIPLE) | the chooser's default target; always clamped to Rmax."""
    for e in (DF.ENV_COND_ROWS, DF.ENV_BIAS_ROWS, DF.ENV_Q_ROWS, DF.ENV_BAND_ROWS, DF.ENV_BIAS_CACHE, DF.ENV_WORK_GB, DF.ENV_BIAS_CACHE_GB):
        monkeypatch.delenv(e, raising=False)
    ch = DF._choose_rows
    assert ch(1000, 250, 96, rows=16, env=DF.ENV_COND_ROWS, budget_bytes=10 ** 12) == (16, "given")
    assert ch(1000, 250, 96, rows=0, env=None, budget_bytes=None) == (96, "given")                 # 0 = all local rows
    assert ch(1000, 250, 96, rows=50, env=None, budget_bytes=None)[0] == 50                          # given is taken as given (no rounding)
    assert ch(1000, 250, 96, rows=None, env=None, budget_bytes=40_500) == (40, "budget")             # 40.5 rows -> 40 (multiple of 4)
    assert ch(1000, 250, 96, rows=None, env=None, budget_bytes=10) == (4, "budget+align")            # never below the multiple (a target, not a refusal)
    r, s = ch(1000, 250, 96, rows=None, env=None, budget_bytes=10 ** 12)
    assert r == 96 and s.startswith("budget") and "n_max" in s                                       # clipped to Rmax
    r, s = ch(4 * 10 ** 6, 10 ** 6, 5000, rows=4000, env=None, budget_bytes=None)
    assert r == (2 ** 31 - 1) // 10 ** 6 and "cap" in s                                              # the int32 launch rule clips every source
    monkeypatch.setenv(DF.ENV_COND_ROWS, "24")
    assert ch(1000, 250, 96, rows=None, env=DF.ENV_COND_ROWS, budget_bytes=10 ** 12) == (24, "env:" + DF.ENV_COND_ROWS)
    assert ch(1000, 250, 96, rows=8, env=DF.ENV_COND_ROWS, budget_bytes=None) == (8, "given")
    monkeypatch.setenv(DF.ENV_COND_ROWS, "0")
    assert ch(1000, 250, 96, rows=None, env=DF.ENV_COND_ROWS, budget_bytes=10) == (96, "env:" + DF.ENV_COND_ROWS)
    monkeypatch.setenv(DF.ENV_COND_ROWS, "-3")
    with pytest.raises(RowpairRefused, match=DF.ENV_COND_ROWS):
        ch(1000, 250, 96, rows=None, env=DF.ENV_COND_ROWS, budget_bytes=None)
    monkeypatch.setenv(DF.ENV_COND_ROWS, "many")
    with pytest.raises(RowpairRefused, match=DF.ENV_COND_ROWS):
        ch(1000, 250, 96, rows=None, env=DF.ENV_COND_ROWS, budget_bytes=None)
    r, s = ch(1000, 250, 96, rows=None, env=None, budget_bytes=None)                                # no budget anywhere: the chooser's default MiB target
    assert s.startswith("default") and 1 <= r <= 96


def test_schedule_decide_is_recorded_and_deterministic(monkeypatch):
    from opt_core.mem.rowpair import evidence as EV
    for e in (DF.ENV_COND_ROWS, DF.ENV_BIAS_ROWS, DF.ENV_Q_ROWS, DF.ENV_BAND_ROWS, DF.ENV_BIAS_CACHE, DF.ENV_WORK_GB, DF.ENV_BIAS_CACHE_GB):
        monkeypatch.delenv(e, raising=False)
    lay = Layout(4000, 4, 1, 128)                                           # no group needed to DECIDE (budget given -> no collective)
    kw = dict(c_z=128, c_in=128 + 139, c_cond=128, H=16, S=1, n_blocks=24, c_pair=16)
    a = DF.DiffusionSchedule.decide(lay, budget_bytes=40 * 10 ** 9, **kw)
    b = DF.DiffusionSchedule.decide(lay, budget_bytes=40 * 10 ** 9, record=False, **kw)
    assert a.fields() == b.fields()                                          # a pure function of (layout, dims, budget): identical on every rank
    f = dict(a.fields())
    srcs = dict(kv.split(":", 1) for kv in f["diff_rows_source"].split(","))
    assert srcs["work"] == "given" and srcs["bias_cache"] == "given" and all(srcs[k].startswith("budget") for k in ("cond", "bias", "q", "band")), f
    assert 4 <= a.cond_rows <= lay.Rmax and a.cond_rows % 4 == 0 and a.q_rows % 4 == 0
    assert f["diff_bias_cache"] == "on"                                      # 24*16*1024*4000*4 = 6.3 GB <= 0.3 * 40 GB
    assert DF.DiffusionSchedule.decide(lay, budget_bytes=10 ** 9, record=False, **kw).bias_cache is False
    sched = dict(EV.schedule_fields())
    for k in ("diff_cond_rows", "diff_bias_rows", "diff_q_rows", "diff_band_rows", "diff_bias_cache", "diff_rows_source", "diff_replicated"):
        assert k in sched, (k, sched)
    # default budgets (no env, no budget_bytes): the 8 GB / 24 GB pins, printed as 'default' — no free-bytes collective needed
    d = DF.DiffusionSchedule.decide(lay, record=False, **kw)
    srcs = dict(kv.split(":", 1) for kv in dict(d.fields())["diff_rows_source"].split(","))
    assert srcs["work"] == "default" and srcs["bias_cache"] == "default" and all(srcs[k].startswith("budget") for k in ("cond", "bias", "q", "band")), d.fields()
    assert d.budget_bytes == int(8e9)
    monkeypatch.setenv(DF.ENV_Q_ROWS, "0")
    monkeypatch.setenv(DF.ENV_BIAS_CACHE, "0")
    e = DF.DiffusionSchedule.decide(lay, record=False, cond_rows=32, **kw)
    assert e.q_rows == lay.Rmax and e.sources["q"] == "env:" + DF.ENV_Q_ROWS and e.cond_rows == 32 and e.sources["cond"] == "given" and e.bias_cache is False
    monkeypatch.setenv(DF.ENV_BIAS_CACHE, "sometimes")
    with pytest.raises(RowpairRefused, match=DF.ENV_BIAS_CACHE):
        DF.DiffusionSchedule.decide(lay, record=False, **kw)


@needs_torch
def test_p1_statements_refuse_by_name():
    """The structural n_gpu=1 rule: every sharded statement refuses a P=1 layout BY NAME (no monkeypatch, no group) — the engine's own
    diffusion module runs at --n_gpu 1. The draw guard and the hook calls are identities without a group."""
    import torch
    lay = Layout(64, 1, 0, 16)
    z = torch.zeros(1, 64, 64, 4)
    fns = DF.DiTBlockFns(norm=None, kv=None, attn=None, update=None, bias=None)
    cases = {"pair_cond_rows": lambda: DF.pair_cond_rows(lambda zr, g0, g1: zr, z, lay, c_out=4),
             "pair_bias_rows": lambda: DF.pair_bias_rows(lambda zr: zr, z, lay),
             "dit_block_sharded": lambda: DF.dit_block_sharded(fns, torch.zeros(1, 64, 4), None, torch.zeros(4, 64, 64), lay),
             "diffusion_transformer_sharded": lambda: DF.diffusion_transformer_sharded([fns], z, None, z, lay),
             "pair_band_rows": lambda: DF.pair_band_rows(lambda zr: zr, z, lay, None),
             "gather_band_rows": lambda: DF.gather_band_rows(z, lay, 3, [0])}
    for name, call in cases.items():
        with pytest.raises(RowpairRefused) as ei:
            call()
        assert name in str(ei.value) and "n_gpu=1" in str(ei.value), (name, str(ei.value))
    t = torch.arange(5.0)
    assert DF.sync_replicated(t, "noise", mode="guard") is t and DF.sync_replicated(t, "noise", mode="bcast") is t
    with pytest.raises(RowpairRefused, match="mode"):
        DF.sync_replicated(t, "noise", mode="sometimes")


@needs_torch
def test_band_plan_and_lookup_dense_equivalence():
    """The band geometry is index arithmetic: on ONE process, band + extras assembled by hand from a dense zp must reproduce zp[q, k]."""
    import torch
    g = torch.Generator().manual_seed(5)
    N, C = 40, 3
    eng = _Engine(N, seed=5, nq=4, nk=10)
    zp = torch.randn((1, N, N, C), generator=g)
    ref = zp[0][eng.q_idx[0].unsqueeze(-1), eng.k_idx[0].unsqueeze(-2)]              # [nb, nq, nk, C]
    for max_w, emax in ((None, 8), (2, N), (0, N)):
        plan = DF.band_plan(eng.q_idx, eng.k_idx, eng.pair_valid, N, max_w=max_w, max_extra_rows=emax, record=False)
        W = plan.W
        band = torch.zeros(1, N, 2 * W + 1, C)
        for i in range(N):
            for t in range(2 * W + 1):
                j = i + t - W
                if 0 <= j < N:
                    band[0, i, t] = zp[0, i, j]
        extras = torch.stack([zp[0, i] for i in plan.extra_rows]).unsqueeze(0) if plan.extra_rows else None
        got = DF.band_lookup(band, extras, plan)[0]
        v = eng.pair_valid[0]
        assert torch.equal(got[v], ref[v]), (max_w, emax)
        if max_w is None:
            assert plan.extra_rows == [] or plan.extra_rows == [0], plan.extra_rows
        else:
            assert len(plan.extra_rows) > 0
    with pytest.raises(RowpairRefused, match="full pair rows needed"):
        DF.band_plan(eng.q_idx, eng.k_idx, eng.pair_valid, N, max_w=0, max_extra_rows=1, record=False)


def test_sampler_hook_protocol():
    assert SH.make("", {}) is None and SH.make(None, {}) is None
    with pytest.raises(ValueError, match="module>:<factory"):
        SH.make("no_colon_here", {})
    info = SH.build_info(n_token=3, n_atom=5, atom_to_token=[0, 0, 1, 2, 2], asym_id=[0, 0, 1], entity_id=[0, 0, 1], sym_id=[0, 0, 0],
                         num_steps=4, num_samples=2, seed=42, mode="rowpair")
    assert info["protocol"] == SH.PROTOCOL_VERSION and info["mode"] == "rowpair" and info["num_steps"] == 4
    hook = SH.make("opt_core.mem.rowpair.sampler_hook:Identity", info)
    assert isinstance(hook, SH.Identity) and SH.configured("ROWPAIR_TEST_NO_SUCH_ENV") is False
    if not HAVE_TORCH:
        return
    import torch
    x = torch.randn(1, 2, 5, 3)
    ctx = dict(step=-1, t_hat=1.0, sigma_next=0.5, gamma=0.0, aug_rot=None, aug_trans=None, aug_center=None)
    assert SH.call_init_noise(hook, x, ctx) is x and SH.call_init_noise(hook, x, dict(ctx, step=0), per_step=True) is x
    assert SH.call_project(hook, x, dict(ctx, step=0)) is x and hook.calls == [("init_noise", -1), ("init_noise", 0), ("project", 0)]
    assert SH.call_init_noise(None, x, ctx) is x and SH.call_project(None, x, ctx) is x

    class NoPerStep(SH.Identity):
        per_step_noise = False
    h2 = NoPerStep(info)
    assert SH.call_init_noise(h2, x, dict(ctx, step=3), per_step=True) is x and h2.calls == []        # not asked -> not called

    class Bad(SH.Identity):
        def project(self, x, ctx):
            return x[..., :2]
    with pytest.raises(TypeError, match="project"):
        SH.call_project(Bad(info), x, ctx)
    rc = SH.make("opt_core.mem.rowpair.sampler_hook:RecenterExample", dict(info, atom_mask=torch.tensor([1., 1., 1., 1., 0.])))
    y = SH.call_project(rc, x, dict(ctx, step=1))
    cen = (y[..., :4, :]).mean(dim=-2)
    assert float(cen.abs().max()) < 1e-6 and torch.equal(y[..., 4, :], x[..., 4, :])                  # padded atom left alone


# ================================================================================================================ the synthetic engine
class _Engine(object):
    """A synthetic diffusion module with random fp32 weights, built from a seeded CPU generator (identical on every rank): pair
    conditioning ``linear(LN(cat[z, relpos]))`` + 2 masked SwiGLU transitions; a 2-block diffusion transformer (AdaLN, 4-head attention
    with pair bias + key mask + sigmoid gate + output projection, AdaLN-zero gate, conditioned SwiGLU transition); the atom-attention pair
    projection ``linear(LN(z_cond))`` and an AF3-style atom block geometry (``nq`` query atoms per block, ``nk`` keys centred on it).
    ``dense_*`` are the whole-tensor references; ``*_rows`` / ``blocks_fns`` are the callables an adapter would bind."""

    c_z, c_rel, c, c_s, c_a, H, S, C_pair = 8, 6, 8, 12, 32, 4, 2, 3

    def __init__(self, N, seed=7, nq=8, nk=24, max_apt=3):
        import torch
        import torch.nn.functional as F
        self.N, self.F = N, F
        g = torch.Generator().manual_seed(seed)
        rnd = lambda *shape: torch.randn(shape, generator=g, dtype=torch.float32)  # noqa: E731
        lin = lambda i, o: (rnd(i, o) / i ** 0.5, 0.1 * rnd(o))                     # noqa: E731  (W, b)
        c_z, c_rel, c, c_s, c_a, H = self.c_z, self.c_rel, self.c, self.c_s, self.c_a, self.H
        self.c_h = c_a // H

        def ln(cdim):
            m = torch.nn.LayerNorm(cdim)
            with torch.no_grad():
                m.weight.copy_(1.0 + 0.1 * rnd(cdim))
                m.bias.copy_(0.1 * rnd(cdim))
            return m.requires_grad_(False)
        # inputs
        self.z = rnd(1, N, N, c_z)
        self.res_idx = torch.arange(N)
        self.mask = (torch.rand((N, N), generator=g) > 0.1).float()                # a FULL [N, N] pair mask (an INPUT; statements slice rows)
        self.token_mask = torch.ones(N)
        self.token_mask[N - 3:] = 0.0                                             # padded tokens: masked keys
        # conditioning
        self.ln_in = ln(c_z + c_rel)
        self.W_in, self.b_in = lin(c_z + c_rel, c)
        self.trans = [(ln(c), lin(c, 4 * c)[0], lin(2 * c, c)[0]) for _ in range(2)]
        # transformer blocks
        self.blocks = []
        for _ in range(2):
            blk = dict(ln_a=torch.nn.LayerNorm(c_a, elementwise_affine=False), ln_s=ln(c_s), W_s1=lin(c_s, c_a), W_s2=lin(c_s, c_a)[0],
                       Wq=lin(c_a, c_a), Wk=lin(c_a, c_a)[0], Wv=lin(c_a, c_a)[0], Wg=lin(c_a, c_a), Wo=lin(c_a, c_a)[0],
                       ln_z=ln(c), Wb=lin(c, H)[0], W_ada=lin(c_s, c_a),
                       t_ln_a=torch.nn.LayerNorm(c_a, elementwise_affine=False), t_ln_s=ln(c_s), t_W_s1=lin(c_s, c_a), t_W_s2=lin(c_s, c_a)[0],
                       t_W1=lin(c_a, 4 * c_a)[0], t_W2=lin(2 * c_a, c_a)[0], t_W_ada=lin(c_s, c_a))
            self.blocks.append(blk)
        # atom pair projection + atom block geometry
        self.ln_p = ln(c)
        self.W_p = lin(c, self.C_pair)[0]
        apt = torch.randint(1, max_apt + 1, (N,), generator=g)
        apt[0] = 1
        tok_of_atom = torch.repeat_interleave(torch.arange(N), apt)               # token-ordered atom -> token map
        A = int(tok_of_atom.numel())
        nb = -(-A // nq)
        pad = nb * nq - A
        atom_valid = torch.cat([torch.ones(A, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
        tok_pad = torch.cat([tok_of_atom, torch.zeros(pad, dtype=torch.long)])   # padded atoms read token row 0 (the AF3 engines' convention)
        q_atoms = torch.arange(nb * nq).view(nb, nq)
        centre = q_atoms[:, :1] + nq // 2
        k_atoms = centre - nk // 2 + torch.arange(nk)[None, :]                    # [nb, nk] keys centred on the query block
        k_ok = (k_atoms >= 0) & (k_atoms < nb * nq)
        k_atoms = k_atoms.clamp(0, nb * nq - 1)
        self.q_idx = tok_pad[q_atoms].unsqueeze(0)                                # [1, nb, nq]
        self.k_idx = torch.where(k_ok, tok_pad[k_atoms], torch.zeros_like(k_atoms)).unsqueeze(0)   # [1, nb, nk]
        qv = atom_valid[q_atoms]
        kv = k_ok & atom_valid[k_atoms]
        self.pair_valid = (qv[:, :, None] & kv[:, None, :]).unsqueeze(0)         # [1, nb, nq, nk]
        self._g = g

    # ------------------------------------------------------------------------------------------------ shared elementary statements
    def relpos_rows(self, g0, g1):
        """[1, rows, N, c_rel]: one-hot of the clipped residue offset for GLOBAL rows [g0, g1) (lazy per block)."""
        import torch
        d = (self.res_idx[None, :] - self.res_idx[g0:g1, None]).clamp(-2, 3) + 2   # [rows, N] in [0, 5]
        return self.F.one_hot(d, self.c_rel).float().unsqueeze(0)

    def _swiglu(self, x, W1, W2):
        a, b = (x @ W1).chunk(2, dim=-1)
        return (self.F.silu(a) * b) @ W2

    def _trans(self, k, x):
        lnm, W1, W2 = self.trans[k]
        return self._swiglu(lnm(x), W1, W2)

    def _adaln(self, ln_a, ln_s, W_s1, W_s2, a, s):
        import torch
        return ln_a(a) * torch.sigmoid(ln_s(s) @ W_s1[0] + W_s1[1]) + ln_s(s) @ W_s2

    def _cond_trans(self, blk, a, s):
        import torch
        x = self._adaln(blk["t_ln_a"], blk["t_ln_s"], blk["t_W_s1"], blk["t_W_s2"], a, s)
        return torch.sigmoid(s @ blk["t_W_ada"][0] + blk["t_W_ada"][1]) * self._swiglu(x, blk["t_W1"], blk["t_W2"])

    # ------------------------------------------------------------------------------------------------ DENSE references (whole tensors)
    def dense_cond(self, z):
        import torch
        N = self.N
        x = torch.cat([z, self.relpos_rows(0, N)], dim=-1)                        # [1, N, N, c_z + c_rel]
        x = self.ln_in(x) @ self.W_in + self.b_in
        for k in range(2):
            x = x + self._trans(k, x) * self.mask[None, :, :, None]
        return x

    def dense_block(self, blk, a, s, zc):
        import torch
        x = self._adaln(blk["ln_a"], blk["ln_s"], blk["W_s1"], blk["W_s2"], a, s)   # [1, S, N, c_a]
        B_, S_, N, _ = x.shape
        q = (x @ blk["Wq"][0] + blk["Wq"][1]).view(B_, S_, N, self.H, self.c_h).permute(0, 1, 3, 2, 4) / self.c_h ** 0.5
        k = (x @ blk["Wk"]).view(B_, S_, N, self.H, self.c_h).permute(0, 1, 3, 2, 4)
        v = (x @ blk["Wv"]).view(B_, S_, N, self.H, self.c_h).permute(0, 1, 3, 2, 4)
        bias = (blk["ln_z"](zc) @ blk["Wb"]).permute(0, 3, 1, 2).unsqueeze(1)     # [1, 1, H, N, N]
        logits = q @ k.transpose(-1, -2) + bias + ((self.token_mask - 1.0) * 1e9)[None, None, None, None, :]
        o = torch.softmax(logits, dim=-1) @ v                                       # [1, S, H, N, c_h]
        o = o.permute(0, 1, 3, 2, 4).reshape(B_, S_, N, self.c_a)
        o = (torch.sigmoid(x @ blk["Wg"][0] + blk["Wg"][1]) * o) @ blk["Wo"]
        a = a + torch.sigmoid(s @ blk["W_ada"][0] + blk["W_ada"][1]) * o
        return a + self._cond_trans(blk, a, s)

    def dense_transformer(self, a, s, zc):
        for blk in self.blocks:
            a = self.dense_block(blk, a, s, zc)
        return a

    def dense_pair_proj(self, zc):
        return self.ln_p(zc) @ self.W_p                                             # [1, N, N, C_pair]

    def step_inputs(self, step):
        import torch
        g = torch.Generator().manual_seed(100 + step)                               # replicated per-step activations (identical on every rank)
        return (torch.randn((1, self.S, self.N, self.c_a), generator=g), torch.randn((1, self.S, self.N, self.c_s), generator=g))

    # ------------------------------------------------------------------------------------------------ the callables an adapter binds
    def embed_rows(self, z_rows, g0, g1):
        import torch
        x = torch.cat([z_rows, self.relpos_rows(g0, g1)], dim=-1)
        return self.ln_in(x) @ self.W_in + self.b_in

    @property
    def transitions_rows(self):
        return [lambda x, g0, g1, k=k: self._trans(k, x) * self.mask[g0:g1][None, :, :, None] for k in range(2)]   # rows sliced FIRST: no [N, N] view

    def proj_rows(self, z_rows):
        return self.ln_p(z_rows) @ self.W_p

    @property
    def blocks_fns(self):
        import torch
        out = []
        for blk in self.blocks:
            def norm(a, s, blk=blk):
                return self._adaln(blk["ln_a"], blk["ln_s"], blk["W_s1"], blk["W_s2"], a, s)

            def kv(x, blk=blk):
                B_, S_, N, _ = x.shape
                k = (x @ blk["Wk"]).view(B_, S_, N, self.H, self.c_h).permute(0, 1, 3, 2, 4)
                v = (x @ blk["Wv"]).view(B_, S_, N, self.H, self.c_h).permute(0, 1, 3, 2, 4)
                mask_bias = ((self.token_mask - 1.0) * 1e9)[None, None, None, None, :]
                return k, v, mask_bias

            def attn(x_q, kvm, bias_q, rows, blk=blk):
                k, v, mask_bias = kvm
                B_, S_, nq, _ = x_q.shape
                q = (x_q @ blk["Wq"][0] + blk["Wq"][1]).view(B_, S_, nq, self.H, self.c_h).permute(0, 1, 3, 2, 4) / self.c_h ** 0.5
                logits = q @ k.transpose(-1, -2) + bias_q.unsqueeze(1) + mask_bias    # bias_q [1, H, q, N] -> [1, 1, H, q, N]
                o = torch.softmax(logits, dim=-1) @ v
                o = o.permute(0, 1, 3, 2, 4).reshape(B_, S_, nq, self.c_a)
                return (torch.sigmoid(x_q @ blk["Wg"][0] + blk["Wg"][1]) * o) @ blk["Wo"]

            def update(a, o_rows, s, rows, blk=blk):
                r0, r1 = rows
                a_r, s_r = a[..., r0:r1, :], s[..., r0:r1, :]
                a_r = a_r + torch.sigmoid(s_r @ blk["W_ada"][0] + blk["W_ada"][1]) * o_rows
                return a_r + self._cond_trans(blk, a_r, s_r)

            def bias(z_rows, blk=blk):
                return blk["ln_z"](z_rows) @ blk["Wb"]                                 # [1, rows, N, H]
            out.append(DF.DiTBlockFns(norm=norm, kv=kv, attn=attn, update=update, bias=bias))
        return out


def _pair_guard(N):
    """A dispatch-mode context recording every aten op output with >= 2 dims equal to N (a whole pair-shaped tensor) while active."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Guard(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.offenders = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for leaf in tree_flatten(out)[0]:
                if isinstance(leaf, torch.Tensor) and sum(1 for d in leaf.shape if int(d) == N) >= 2:
                    self.offenders.append((str(func), tuple(leaf.shape)))
            return out
    return Guard()


# ================================================================================================================ multi-process entry
def _entry_diffusion(N: int, B: int, steps: int = 3):
    import torch
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair import evidence as EV
    from opt_core.mem.rowpair.shard import shard_rows
    torch.set_num_threads(1)
    P, r = D.world()
    lay = D.Layout.checked(N, P, r, B)
    eng = _Engine(N, seed=7)
    res = {"rank": r, "P": P, "N": N, "R": lay.R, "checks": {}, "metrics": {}, "bitwise": {}}
    chk, met, bit = res["checks"], res["metrics"], res["bitwise"]

    def cmp(name, got, dense, tol=TOL32):
        m = _cmp(got.contiguous(), dense.contiguous())
        met[name] = m["maxabs_vs_dense"]
        bit[name] = m["bitwise_vs_dense"]
        chk[name] = bool(_ok(m, _tol(got.dtype, float(dense.abs().max()))) and m["maxabs_vs_dense"] <= tol)

    with torch.no_grad():
        z_loc = shard_rows(eng.z, lay, dim=-3)                                     # [1, R, N, c_z] (a view of the replicated test input)
        sched = DF.DiffusionSchedule.decide(lay, c_z=eng.c_z, c_in=eng.c_z + eng.c_rel, c_cond=eng.c, H=eng.H, S=eng.S, n_blocks=2,
                                            c_pair=eng.C_pair, cond_rows=16, bias_rows=8, q_rows=12, band_rows=20, bias_cache=True)
        chk["schedule_given_recorded"] = dict(EV.schedule_fields()).get("diff_rows_source") == "work:default,cond:given,bias:given,q:given,band:given,bias_cache:given"
        guard = _pair_guard(N)
        with guard:
            zc_loc = DF.pair_cond_rows(eng.embed_rows, z_loc, lay, c_out=eng.c, rows=sched.cond_rows, transitions=eng.transitions_rows)
            zc_loc_again = DF.pair_cond_rows(eng.embed_rows, z_loc, lay, c_out=eng.c, rows=0, transitions=eng.transitions_rows)   # one block
            cache = DF.PairBiasCache(enabled=sched.bias_cache)
            outs, outs_plain, ins = [], [], []
            for step in range(steps):
                a_in, s_in = eng.step_inputs(step)
                ins.append((a_in, s_in))
                outs.append(DF.diffusion_transformer_sharded(eng.blocks_fns, a_in, s_in, zc_loc, lay, schedule=sched, bias_cache=cache))
                outs_plain.append(DF.diffusion_transformer_sharded(eng.blocks_fns, a_in, s_in, zc_loc, lay, q_rows=0, bias_rows=0, bias_cache=None))
            one_block_rows = DF.dit_block_sharded(eng.blocks_fns[0], ins[0][0], ins[0][1],
                                                  DF.pair_bias_rows(eng.blocks_fns[0].bias, zc_loc, lay, rows=5), lay, q_rows=7, gather=False)
            plan = DF.band_plan(eng.q_idx, eng.k_idx, eng.pair_valid, N, max_w=None, max_extra_rows=8)
            band, extras = DF.pair_band_rows(eng.proj_rows, zc_loc, lay, plan, rows=sched.band_rows)
            got_band = DF.band_lookup(band, extras, plan)
            plan2 = DF.band_plan(eng.q_idx, eng.k_idx, eng.pair_valid, N, max_w=plan.w_need - 2, max_extra_rows=N)     # a cap below the need: SOME rows travel whole
            band2, extras2 = DF.pair_band_rows(eng.proj_rows, zc_loc, lay, plan2, rows=0)
            got_band2 = DF.band_lookup(band2, extras2, plan2)
            xp_loc = eng.proj_rows(zc_loc)                                                                            # the projected shard [1, R, N, C]
            band3, extras3 = DF.gather_band_rows(xp_loc, lay, plan2.W, plan2.extra_rows, rows=9)
        res["pair_shaped_allocations"] = guard.offenders[:5]
        chk["never_whole_pair"] = len(guard.offenders) == 0
        chk["bias_cache_hits"] = cache.hits == (steps - 1) * 2 and cache.misses == 2
        met["bias_cache_hits"] = cache.hits

        # ------------------------------------------------------------------ dense references (the dense module recomputes its conditioning every step)
        r0, r1 = lay.r0, lay.r1
        zc_dense_steps = [eng.dense_cond(eng.z) for _ in range(steps)]
        zc_dense = zc_dense_steps[0]
        chk["dense_cond_step_invariant"] = all(torch.equal(zc_dense, x) for x in zc_dense_steps[1:])
        cmp("z_cond_rows", zc_loc, zc_dense[:, r0:r1])
        bit["cond_once_vs_recompute"] = chk["cond_once_vs_recompute"] = bool(torch.equal(zc_loc, zc_loc_again))
        for step in range(steps):
            a_in, s_in = ins[step]
            dense_out = eng.dense_transformer(a_in, s_in, zc_dense)
            cmp(f"step{step}_a", outs[step], dense_out)
            cmp(f"step{step}_a_plain", outs_plain[step], dense_out)
            bit[f"step{step}_cache+qblocks_vs_plain"] = bool(torch.equal(outs[step], outs_plain[step]))
        dense_blk0 = eng.dense_block(eng.blocks[0], ins[0][0], ins[0][1], zc_dense)
        cmp("one_block_rows", one_block_rows, dense_blk0[..., r0:r1, :])
        zp = eng.dense_pair_proj(zc_dense)                                                                            # [1, N, N, C]
        ref = zp[0][eng.q_idx[0].unsqueeze(-1), eng.k_idx[0].unsqueeze(-2)].unsqueeze(0)                              # [1, nb, nq, nk, C]
        v = eng.pair_valid
        cmp("band_lookup_valid", got_band[v], ref[v])
        cmp("band_lookup_forced_extras_valid", got_band2[v], ref[v])
        met["band_W"], met["band_extra_rows"], met["band2_extra_rows"] = plan.W, len(plan.extra_rows), len(plan2.extra_rows)
        chk["band2_has_extras"] = 0 < len(plan2.extra_rows) < N // 2 and extras2 is not None and plan.extra_rows == []   # uncapped: nothing travels whole
        bit["gather_band_rows_vs_fused"] = chk["gather_band_rows_vs_fused"] = bool(torch.equal(band3, band2) and torch.equal(extras3, extras2))
        ref_ex = torch.stack([zp[0, i] for i in plan2.extra_rows]).unsqueeze(0)
        cmp("extras_rows", extras2, ref_ex)
        W = plan.W
        band_ref = torch.zeros_like(band)
        for i in range(N):
            lo, hi = max(0, i - W), min(N, i + W + 1)
            band_ref[0, i, lo - i + W:hi - i + W] = zp[0, i, lo:hi]
        cmp("band_rows", band, band_ref)
        # ------------------------------------------------------------------ replicated draws: guard / bcast / off
        same = torch.randn((2, 7, 3), generator=torch.Generator().manual_seed(1000))
        mine = torch.randn((2, 7, 3), generator=torch.Generator().manual_seed(1000 + r))
        chk["guard_passes_identical"] = DF.sync_replicated(same, "init_noise", mode="guard") is same
        try:
            DF.sync_replicated(mine, "per_rank_noise", mode="guard")
            chk["guard_refuses_divergent"] = False
        except RowpairRefused as e:
            chk["guard_refuses_divergent"] = "per_rank_noise" in str(e) and "differs across ranks" in str(e)
        got = DF.sync_replicated(mine.clone(), "per_rank_noise", mode="bcast")
        bit["bcast_is_rank0s_everywhere"] = chk["bcast_is_rank0s_everywhere"] = bool(torch.equal(got, same))
        chk["off_is_identity"] = DF.sync_replicated(mine, "x", mode="off") is mine
        chk["noise_sync_recorded"] = EV.schedule().get("diff_noise_sync") == "off" and "guard:" in str(EV.schedule().get("diff_noise_sync_calls"))
        fields = dict(EV.schedule_fields())
        chk["band_facts_recorded"] = "diff_band_W" in fields and "diff_band_extra_rows" in fields
    res["ok"] = all(chk.values())
    import torch.distributed as tdist                                        # EVERY rank's verdict reaches rank 0 (run_sharded returns rank 0's value)
    allres = [None] * P
    tdist.all_gather_object(allres, res)
    return {"P": P, "ranks": allres, "rows": [x["R"] for x in allres], "ok": all(x["ok"] for x in allres)}


def _mp(P, entry, *args):
    from opt_core.mem.rowpair import launch
    prev = os.environ.get("ROWPAIR_TEST_DEVICE")
    os.environ["ROWPAIR_TEST_DEVICE"] = "cpu"
    try:
        return launch.run_sharded(P, entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=900)
    finally:
        if prev is None:
            os.environ.pop("ROWPAIR_TEST_DEVICE", None)
        else:
            os.environ["ROWPAIR_TEST_DEVICE"] = prev


@needs_torch
@pytest.mark.parametrize("P,N,B,uneven", [(2, 200, 64, True), (3, 384, 128, False), (4, 400, 64, True)])
def test_mp_diffusion(P, N, B, uneven):
    res = _mp(P, _entry_diffusion, N, B, 3)
    print("RESULT " + json.dumps(res, sort_keys=True, default=str))
    assert len(res["ranks"]) == P and [x["rank"] for x in res["ranks"]] == list(range(P))          # every rank reported
    assert (len(set(res["rows"])) > 1) is uneven, res["rows"]                                       # the uneven grid really is uneven
    bad = {x["rank"]: [k for k, v in x["checks"].items() if not v] for x in res["ranks"] if not x["ok"]}
    assert not bad, (bad, [(x["rank"], x["metrics"], x["pair_shaped_allocations"]) for x in res["ranks"]])


if __name__ == "__main__":                                                       # python tests/test_rowpair_diffusion_043.py [P N B]
    P, N, B = (int(v) for v in (sys.argv[1:4] + ["2", "200", "64"][len(sys.argv) - 1:])[:3])
    out = _mp(P, _entry_diffusion, N, B, 3)
    print("RESULT " + json.dumps(out, sort_keys=True, default=str))
    sys.exit(0 if out["ok"] else 1)
