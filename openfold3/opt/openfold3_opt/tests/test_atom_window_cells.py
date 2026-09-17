"""The `atom_window` cell's addressing and dispatch contracts, CPU only: (1) the core kernel's `window_starts` + the kernel's key / pair validity
rule select, block by block and position by position, the keys of upstream's shifted-window gather and its block pair mask — against a
transliteration of `atom_attention_block_utils.get_block_indices` / `get_pair_atom_block_mask` here, and against upstream itself when
`openfold3` imports — over a sweep of atom counts (fewer atoms than one key window, counts not a multiple of the query block, ragged last
block, interior mask zeros, several samples); (2) `plan()` names its refusals; (3) the block patch dispatches by instance: a block without
`use_cross_attention`, a call outside a rollout, and a lever that is off all reach the wrapped method untouched."""
import math

import pytest

torch = pytest.importorskip("torch", reason="needs torch")
from openfold3_opt.cells import atom_window as AWC  # noqa: E402  (the adapter: binds the switch names, re-exports the tree's record)
from opt_core.of3_sampler import rollout_memo as RM  # noqa: E402
from opt_core.of3_sampler import atom_window as AWI  # noqa: E402  (the implementation)

aw = pytest.importorskip("opt_core.kernels.atom_window", reason="needs opt_core's atom_window kernel module (Triton importable)")
NQ, NK = 32, 128


def ref_block_indices(atom_mask, n_query, n_key):
    """Transliteration of upstream's get_block_indices: (safe_indices [S, NB, NK], invalid_mask [S, NB, NK])."""
    S, A = atom_mask.shape
    nb = math.ceil(A / n_query)
    centers = (n_query // 2 + torch.arange(nb) * n_query)[None].expand(S, -1)
    n_atom = atom_mask.sum(-1, keepdim=True).expand(-1, nb)
    init = (centers[..., None] + torch.arange(-n_key // 2, n_key // 2)[None, :]).int()
    underflow = torch.relu(-init[..., 0]); overflow = torch.relu(init[..., -1] - (n_atom - 1))
    shift = torch.where(underflow > 0, underflow, -overflow)
    final = init + shift[..., None]
    n_atom = n_atom.unsqueeze(-1)
    invalid = (final < 0) | (final >= n_atom)
    safe = torch.clamp(final, torch.zeros_like(n_atom), n_atom - 1)
    return safe.long(), invalid


def ref_pair_mask(atom_mask, n_query, n_key):
    """Transliteration of upstream's block pair mask: q mask (padded to NB*NQ, blocked) x gathered k mask with invalid keys zeroed -> [S, NB, NQ, NK] bool."""
    S, A = atom_mask.shape
    nb = math.ceil(A / n_query)
    safe, invalid = ref_block_indices(atom_mask, n_query, n_key)
    qm = torch.nn.functional.pad(atom_mask, (0, nb * n_query - A)).reshape(S, nb, n_query)
    km = torch.gather(atom_mask[:, None, :].expand(-1, nb, -1), 2, safe) * (~invalid).float()
    return (qm[..., :, None] * km[..., None, :]) > 0.5


def kernel_side(atom_mask):
    """The kernel's addressing: ks -> k_idx [NB, NK]; key valid = 0 <= k < A, k < n_real, mask[k]; pair = mask[q] & key valid."""
    S, A = atom_mask.shape
    n_real = int(atom_mask[0].sum().item())
    ks = aw.window_starts(A, torch.tensor(float(n_real)), NQ, NK, atom_mask.device).long()
    nb = ks.numel()
    k_idx = ks[:, None] + torch.arange(NK)[None, :]
    k_in = (k_idx >= 0) & (k_idx < A)
    am_k = torch.zeros((S, nb, NK)); am_k[:, k_in] = atom_mask[:, k_idx[k_in]]
    key_valid = k_in[None] & (k_idx[None] < n_real) & (am_k > 0.5)
    q_rows = torch.arange(nb * NQ).reshape(nb, NQ); q_in = q_rows < A
    am_q = torch.zeros((S, nb, NQ)); am_q[:, q_in] = atom_mask[:, q_rows[q_in]]
    pair = (am_q[..., :, None] > 0.5) & key_valid[..., None, :]
    return k_idx, key_valid, pair


def _cases():
    out = []
    for A in list(range(1, 300)) + [383, 384, 385, 511, 512, 513, 1000, 1023, 1024, 1025, 3236, 3260, 6472, 6473]:
        for n_real in sorted({A, max(1, A - 1), max(1, A - 24), max(1, A - 31), max(1, (A // 32) * 32), max(1, A // 2)}):
            out.append((A, n_real, 1 if A > 600 else 2, ()))
    out += [(200, 200, 2, (5, 77, 130)), (130, 120, 2, (0,)), (64, 64, 2, (63,)), (257, 250, 2, (128, 129))]
    return out


def _mask(A, n_real, S, holes):
    m = torch.zeros((S, A)); m[:, :n_real] = 1.0
    for h in holes:
        if h < n_real:
            m[:, h] = 0.0
    return m


def _agree(atom_mask, safe, invalid, pair_ref):
    k_idx, key_valid, pair = kernel_side(atom_mask)
    S = atom_mask.shape[0]
    valid_ref = ~invalid & (torch.gather(atom_mask[:, None, :].expand(-1, safe.shape[1], -1), 2, safe) > 0.5)
    return (torch.equal(key_valid, valid_ref)                                                        # the same key positions are valid
            and bool(((k_idx[None].expand_as(safe) == safe) | ~valid_ref).all())                      # wherever a key is valid the kernel addresses the same atom
            and torch.equal(pair, pair_ref))                                                          # the same query x key pairs enter the softmax


def test_window_addressing_equals_the_transliterated_upstream_rule():
    cases = _cases()
    assert len(cases) > 1700
    bad = []
    for A, n_real, S, holes in cases:
        m = _mask(A, n_real, S, holes)
        safe, invalid = ref_block_indices(m, NQ, NK)
        if not _agree(m, safe, invalid, ref_pair_mask(m, NQ, NK)):
            bad.append((A, n_real, S, holes))
    assert not bad, bad[:10]


def test_window_addressing_equals_upstream_when_importable():
    abu = pytest.importorskip("openfold3.core.utils.atom_attention_block_utils", reason="upstream openfold3 not importable here")
    bad = []
    for A, n_real, S, holes in _cases()[::3] + _cases()[-8:]:
        m = _mask(A, n_real, S, holes)
        safe, invalid = abu.get_block_indices(atom_mask=m, n_query=NQ, n_key=NK, device=m.device)
        _, _, pair = abu.convert_single_rep_to_blocks(torch.zeros((S, A, 4)), NQ, NK, m)
        if not _agree(m, safe.long(), invalid, pair > 0.5):
            bad.append((A, n_real, S, holes))
    assert not bad, bad[:10]


def test_weight_tf32_is_round_to_nearest_ties_away():
    w = torch.tensor([1.0, 1.0 + 2 ** -11, 1.0 + 2 ** -10, 1.0 + 3 * 2 ** -12, -1.0 - 2 ** -11, 3.0e38, 1e-40], dtype=torch.float32)
    r = aw.weight_tf32(w)
    assert r.dtype == torch.float32 and r.shape == w.shape
    assert r[0] == 1.0 and r[1] == 1.0 + 2 ** -10 and r[2] == 1.0 + 2 ** -10 and r[3] == 1.0 + 2 ** -10 and r[4] == -(1.0 + 2 ** -10)   # half-ulp ties away from zero
    assert ((r.view(torch.int32) & 0x1FFF) == 0).all()                                                                                     # the 13 dropped mantissa bits are clear
    assert aw.weight_tf32(w) is r                                                                                                           # memoised per parameter
    w.add_(0.0)                                                                                                                             # in-place modification -> re-made
    assert aw.weight_tf32(w) is not r


class _Lin:
    def __init__(self, cin, cout, bias=True):
        self.in_features, self.out_features = cin, cout
        self.weight = torch.zeros(cout, cin); self.bias = torch.zeros(cout) if bias else None


def _blk(C=128, H=4, nq=32, nk=128, cz=16, cross=True):
    ns = lambda **kw: type("NS", (), kw)()   # noqa: E731
    mha = ns(no_heads=H, linear_q=_Lin(C, C), linear_k=_Lin(C, C, False), linear_v=_Lin(C, C, False), linear_g=_Lin(C, C, False), linear_o=_Lin(C, C))
    apb = ns(mha=mha, n_query=nq, n_key=nk, inf=1e9, layer_norm_a_q=ns(eps=1e-5), layer_norm_a_k=ns(eps=1e-5), linear_z=_Lin(cz, H, False), linear_ada_out=_Lin(C, C))
    return ns(attention_pair_bias=apb, use_cross_attention=cross)


def test_plan_names_its_refusals():
    blk = _blk()
    S, A, C, cz = 2, 100, 128, 16
    nb = math.ceil(A / 32)
    a = torch.zeros(1, S, A, C); s = torch.zeros(1, A, C); z = torch.zeros(1, nb, 32, 128, cz); mask = torch.ones(1, S, A)
    assert AWC.plan(blk, a, s, z, mask, {})[1] == "device_cpu"                                       # everything else in domain: the first miss is the device
    assert AWC.plan(blk, a, s, z, mask, {"chunk": 4})[1] == "kwargs:chunk"
    assert AWC.plan(blk, a.double(), s, z, mask, {})[1] in ("device_cpu", "dtype_float64")
    # the domain words past the device check, exercised through a tensor subclass that reports is_cuda
    class _A(torch.Tensor):
        @property
        def is_cuda(self):
            return True
    ac = torch.zeros(1, S, A, C).as_subclass(_A)
    assert AWC.plan(blk, ac, None, z, mask, {})[1] == "no_s_z_or_mask"
    assert AWC.plan(blk, torch.zeros(2, S, A, C).as_subclass(_A), s, z, mask, {})[1] == "lead_dims"
    assert AWC.plan(blk, ac, torch.zeros(1, S, A, C), z, mask, {})[1] == "s_has_sample_dim"
    assert AWC.plan(blk, ac, s, z, torch.ones(1, S, A + 1), {})[1] == "mask_shape"
    assert AWC.plan(_blk(H=5), torch.zeros(1, S, A, 125).as_subclass(_A), torch.zeros(1, A, 125), z, mask, {})[1] == "head_geom"
    assert AWC.plan(_blk(C=256, H=4), torch.zeros(1, S, A, 256).as_subclass(_A), torch.zeros(1, A, 256), z, mask, {})[1] == "head_geom"   # C > 128: the weight tile does not fit
    assert AWC.plan(_blk(nq=16, nk=64), a.as_subclass(_A), s, z, mask, {})[1] == "window_geom"                                              # only the tested (32, 128) geometry is served
    assert AWC.plan(_blk(nq=24), ac, s, z, mask, {})[1] == "window_geom"
    assert AWC.plan(blk, ac, s, torch.zeros(1, nb + 1, 32, 128, cz), mask, {})[1] == "z_shape"
    assert AWC.plan(blk, ac, s, torch.zeros(1, nb, 32, 128, cz + 1), mask, {})[1] == "z_channels"
    assert AWC.plan(blk, ac, s, z, mask, {}) == (True, None)


def test_dispatch_by_instance_and_rollout_state():
    calls = []

    class Blk:
        use_cross_attention = True

        def forward(self, a, s, z, mask=None, _mask_trans=True, use_lma=False):
            calls.append((tuple(a.shape), _mask_trans)); return "wrapped"
    mod = type("M", (), {"DiffusionTransformerBlock": Blk})
    prev = dict(AWC.STATE)
    try:
        AWC.STATE["state"] = "on"
        assert AWI._patch_block(mod) is True and AWI._patch_block(mod) is False                     # idempotent, marked
        b = Blk(); a = torch.zeros(1, 2, 40, 128)
        assert b.forward(a, None, None) == "wrapped" and AWC.STATE["outside"] >= 1                   # outside a rollout: the wrapped method, counted `outside`
        t = Blk(); t.use_cross_attention = False
        n0 = AWC.STATE["calls"]
        assert t.forward(a, None, None) == "wrapped" and AWC.STATE["calls"] == n0                     # a token block: not this lever's call at all
        AWC.STATE["state"] = "off"
        assert b.forward(a, None, None, _mask_trans=False) == "wrapped" and calls[-1] == ((1, 2, 40, 128), False)   # off: kwargs pass through untouched
        AWC.STATE["state"] = "on"
        RM._EPOCH["depth"] += 1                                                                      # inside a rollout: the plan runs and refuses the cpu tensor by name
        try:
            with torch.no_grad():
                assert b.forward(a, torch.zeros(1, 40, 128), torch.zeros(1, 2, 32, 128, 16), mask=torch.ones(1, 2, 40)) == "wrapped"
            assert AWC.STATE["fallback"].get("module_layout", 0) + AWC.STATE["fallback"].get("device_cpu", 0) >= 1
        finally:
            RM._EPOCH["depth"] -= 1
    finally:
        for k in ("state", "calls", "outside", "served"):
            AWC.STATE[k] = prev[k]
        AWC.STATE["fallback"].clear()
