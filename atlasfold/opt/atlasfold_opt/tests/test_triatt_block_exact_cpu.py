"""triatt_block_exact — the exact-class construction of the fused triangle-attention surround (CPU: rows / registry / installers, the per-call
gate words incl. the card words, the floor word, install -> the stock statement serving a CPU call by name with the stock output unchanged)."""
import pytest

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import triatt_block_exact as X, card_rows as CR, pair_cells as PC


def test_rows_and_registry():
    ex, fa, bg = (modes.MODES[m] for m in ("exact", "fast", "big"))
    assert "triatt_block_exact" in ex and "triatt_block_exact" in fa and "triatt_block_exact" in bg              # exact ⊂ fast ⊂ big
    assert ex.index("triattn_exact") < ex.index("triatt_block_exact")
    assert fa.index("triattn_exact") < fa.index("triatt_block_exact") < fa.index("triatt_block")                 # beneath the flash-cored block in fast
    row = registry.LEVERS["triatt_block_exact"]
    assert row["cls"] == "exact" and "ln=stock" in row["provider"]
    assert {"backend_torch", "dtype:", "c:", "below_min_tokens", "cpu", "arch:", "no-cell:", *CR.WORDS} <= set(row["expected"])
    for w in ("heads:", "mask-shape", "mask-batch", "bias:"):                                                    # defects refuse the gate
        assert not PC.expected(w, row["expected"]), w
    from atlasfold_opt.hooks import installers
    assert installers()["triatt_block_exact"] is X.install
    assert PC.PIECES["triatt_block_exact"] == PC.PIECES["triatt_block"]                                          # the same core table rows


def test_floor_word(monkeypatch):
    monkeypatch.delenv(X.MIN_TOKENS_ENV, raising=False); assert X.min_tokens() == 512
    monkeypatch.setenv(X.MIN_TOKENS_ENV, "640"); assert X.min_tokens() == 640
    monkeypatch.setenv(X.MIN_TOKENS_ENV, "junk"); assert X.min_tokens() == 512
    monkeypatch.setenv(X.MIN_TOKENS_ENV, "0"); assert X.min_tokens() == 1


def test_card_rows_rule_matches_reference_rule():
    rb = X.CARD_ROWS[(8, 0)]
    assert rb == (5857, 3 * CR.PIECE_ROWS_SM80, CR.PIECE_ROWS_SM80) and (9, 0) not in X.CARD_ROWS and X.PROVEN_CC == ((9, 0), (8, 0))
    served = {N: CR.outside(N * N, rb) for N in (76, 77, 512, 1024, 1025, 1200, 1280, 1449, 1536, 1773, 1774, 2048)}
    assert served == {76: "below_min_rows", 77: None, 512: None, 1024: "trailing_piece", 1025: "trailing_piece", 1200: None, 1280: None,
                      1449: "trailing_piece", 1536: None, 1773: None, 1774: "above_max_rows", 2048: "above_max_rows"}
    assert CR.outside(2 * 1280 * 1280, rb) == "above_max_rows"                                                  # B=2 at the 1280 bucket on cc 8.0: stock by name
    assert CR.outside(4096 * 4096, None) is None and CR.facts(None) == {}                                        # no rule (cc 9.0): nothing kept out, nothing printed


def _site():
    torch = pytest.importorskip("torch")
    try:
        import atlasfold.model.network.primitives.triangle_update as tu
    except Exception:
        pytest.skip("stock atlasfold not importable here")
    return torch, tu


def test_gate_words_cpu():
    torch, tu = _site()
    m = tu.TriangleAttentionStartingNode(128, 4).eval()
    zb = torch.zeros(1, 16, 16, 128, dtype=torch.bfloat16); mk = torch.ones(1, 16, 16, dtype=torch.bool)
    assert X.refusal(m, zb, mk, "torch", 512, (9, 0)) == "backend_torch"
    assert X.refusal(m, zb.float(), mk, "cuequiv", 512, (9, 0)) == "dtype:float32"
    assert X.refusal(m, zb, mk, "cuequiv", 512, (9, 0)) == "below_min_tokens"
    big = torch.zeros(1, 512, 512, 128, dtype=torch.bfloat16); bigm = torch.ones(1, 512, 512, dtype=torch.bool)
    assert X.refusal(m, big, bigm, "cuequiv", 512, (9, 0)) == "cpu"                                              # everything structural passes: the device is left
    assert X.refusal(m, big, bigm, "cuequiv", 512, (9, 0), ceiling=512 * 512 - 1) == "above_proven_rows" and X.MAX_ROWS == 1280 * 1280
    assert X.refusal(m, big, bigm[:, :5], "cuequiv", 512, (9, 0)) == "mask-shape"
    m64 = tu.TriangleAttentionEndingNode(64, 4).eval()
    assert X.refusal(m64, torch.zeros(1, 16, 16, 64, dtype=torch.bfloat16), mk, "cuequiv", 8, (9, 0)) == "c:64"   # the template pair stack
    m.train(); assert X.refusal(m, zb, mk, "cuequiv", 8, (9, 0)) == "training"


def test_install_runs_stock_by_name_on_cpu():
    torch, tu = _site()
    origs = {c: getattr(tu, c).forward for c, _ in X.CLASSES}
    try:
        ins = X.install("exact", "[t]", {"mode": "exact", "det": 1})
        assert ins.applied, ins.reason
        g = torch.Generator().manual_seed(0)
        for cls_name, ending in X.CLASSES:
            m = getattr(tu, cls_name)(128, 4).eval()
            with torch.no_grad():
                for p in m.parameters():
                    p.copy_((torch.randn(p.shape, generator=g) * 0.1))
            m = m.to(torch.bfloat16)
            z = torch.randn(1, 16, 16, 128, generator=g).to(torch.bfloat16); mk = torch.ones(1, 16, 16, dtype=torch.bool)
            with torch.no_grad():
                out = m(z, mk, kernel_backend="torch")                                                           # backend_torch -> the stock statement
                ref = origs[cls_name](m, z, mk, kernel_backend="torch")
            assert torch.equal(out, ref)
        L = ins.facts["ledger"]
        assert L.served == 0 and set(L.fallbacks) == {"backend_torch"} and ins.gates[0]().ok
        line = ins.lines[0]()
        assert "name=LOCAL.atlasfold.triatt_block_exact" in line and "ln=stock" in line and "min_tokens=512" in line
    finally:
        for c, f in origs.items():
            getattr(tu, c).forward = f
