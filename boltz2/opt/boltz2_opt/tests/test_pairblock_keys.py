"""The fused triangle-attention block's served (C, H, D) keys per variant: every variant serves the C=128 stacks (bitwise
prologue / epilogue); the template Pairformer's (64,4,32) — opt_core's qualified TOLERANCE-class cells — is served by the Tier-2
variants only, under the companion word BOLTZ_PAIRBLOCK_C64=1 (registry lever pairblock_c64), with its own core gate; `cueq` (exact) hands it back
by name (kept_out:64x4x32)."""
from .. import pairblock as PB, modes, registry

ON = {"BOLTZ_PAIRBLOCK_C64": "1"}


def test_exact_variant_never_serves_the_template_key():
    for env in ({}, ON):
        for cc in ("9.0", "8.0", None):
            assert PB.keys_for("cueq", env, cc) == ((128, 4, 32),) and PB.levers_for("cueq", env, cc) == ["pairblock"]
            assert PB.c64_word("cueq", env, cc) == "kept_out:64x4x32"


def test_tier2_variants_serve_the_template_key_only_under_the_word():
    for v in ("flash", "k2b"):
        assert PB.KEY_C64 not in PB.keys_for(v, {}) and "pairblock_c64" not in PB.levers_for(v, {})
        assert PB.KEY_C64 not in PB.keys_for(v, {"BOLTZ_PAIRBLOCK_C64": "0"})
        for cc in ("9.0", "8.0"):
            assert PB.keys_for(v, ON, cc) == ((128, 4, 32), (64, 4, 32))
            assert PB.levers_for(v, ON, cc) == ["pairblock", "flash_triattn", "pairblock_c64"]
    assert PB.levers_for(None) == [] and PB.levers_for("nope") == []


def test_no_kit_token_gate_on_either_key():
    assert not hasattr(PB, "C64_CORE_MIN_TOKENS") and not hasattr(PB, "MIN_TOKENS_DEFAULT")   # the named core serves every token count on both keys: its own measured table decides per card and N


def test_the_hand_back_word_is_declared_and_the_lever_is_tabled():
    assert "kept_out:64x4x32" in PB.EXPECTED and "pairblock_c64" in PB.LEVERS
    L = registry.LEVERS["pairblock_c64"]
    assert L["switch"] == PB.C64_ENV == "BOLTZ_PAIRBLOCK_C64" and L["value"] == "1" and L["tier"] == 2
    assert modes.LEVER_ATTACH["pairblock_c64"] == "pairblock" and modes.NEEDS["pairblock_c64"] == ("pairblock",)
    assert "pairblock_c64" not in modes.MODES["exact"]["levers"] and "BOLTZ_PAIRBLOCK_C64" not in modes.env_row("exact")   # tolerance class: never in the exact row
    for m in ("fast", "big"):
        assert "pairblock_c64" in modes.MODES[m]["levers"] and modes.env_row(m)["BOLTZ_PAIRBLOCK_C64"] == "1"
    assert "pairblock_c64" not in modes.CARD_DROPS["8.0"].get("exact", {}) and set(modes.CARD_DROPS["10.0"]["fast"]) >= {"pairblock_c64"}


def test_default_variant_is_tier2_hosts_the_core_and_hands_the_block_its_own_keyed_core_word():
    """0.3.23: BOLTZ_PAIRBLOCK=default = the block's own keyed attention core (opt_core.attn.pair_fused core="default": its table names, per
    (cc, dtype, head_dim, heads, keys), the flash_triattn cell or the triattn provider's `fast` tier word) — a Tier-2 variant that installs the
    block and the hosted-core lever like flash / k2b, serves the template key under BOLTZ_PAIRBLOCK_C64=1, feeds the module's LayerNorm, and
    hands the block the core word the core itself defines (never a row by name)."""
    from opt_core.attn import pair_fused as PF
    assert "default" in PB.VARIANTS and "default" in PB.TIER2_VARIANTS and PB.tier2("default") and not PB.tier2("cueq")
    assert PB.levers_of("default") == ("pairblock", "flash_triattn") == PB.levers_of("k2b") and PB.LN_OF["default"] == "stock"
    assert PB.keys_for("default", {"BOLTZ_PAIRBLOCK_C64": "1"}) == PB.KEYS_BASE + (PB.KEY_C64,) and PB.keys_for("default", {}) == PB.KEYS_BASE
    assert PB.PF_DEFAULT_CORE == PF.DEFAULT_CORE == "default" and PF.is_core_ok(PB.PF_DEFAULT_CORE) if hasattr(PF, "is_core_ok") else PB.PF_DEFAULT_CORE == PF.DEFAULT_CORE
    assert PB.variant({"BOLTZ_PAIRBLOCK": "default"}) == "default" and PB.levers_for("default", {"BOLTZ_PAIRBLOCK_C64": "1"})[-1] == "pairblock_c64"
    C, H, Dd = PB.KEY_C64
    for cc in ((8, 0), (9, 0)):                                        # class truth of what the word resolves to for the template key and the C=128 key alike (keyed on head_dim / heads)
        for n in (400, 800, 1200):
            c = PF.default_core_for(cc, "bf16", Dd, H, n)
            assert c == "flash_triattn" or c.startswith(PF.TIER_CORE_PREFIX), (cc, n, c)
    assert isinstance(PB.pf_cores(), dict) and PB.pf_cores_word() in ("-",) + tuple([PB.pf_cores_word()])


def test_the_block_s_line_renders_against_the_installed_core_with_the_kit_s_tally_under_its_own_key(monkeypatch):
    """0.3.24: the adapter's ONE line goes through the core's emit_line, whose head carries the core's own tokens (opt_core >= 0.5.98:
    `cores=` = the block's served attention cores) — the kit passes its per-stack tally as `kit_served=` and the `default` variant's census as
    `pf_cores=`, never a key the core's head owns: rendering the line with a served tally must not raise (a duplicate keyword there killed the
    worker at attach) and shows both kit tokens and exactly one `cores=`."""
    from opt_core.attn import pair_fused as PF
    L = PF.ledger(PB.NAME, expected=PB.EXPECTED, min_tokens=300)
    monkeypatch.setitem(PB._STATE, "ledger", L); monkeypatch.setitem(PB._STATE, "variant", "default")
    monkeypatch.setitem(PB._STATE, "served_by", {"pairformer": 48, "msa_pair": 4}); monkeypatch.setitem(PB._STATE, "core_cells", {"128x4x32": ("k2b", "9.0|bf16|D32|H4|N<=400|fwd")}); monkeypatch.setitem(PB._STATE, "keys", list(PB.KEYS_BASE))
    line = PB.line()
    assert isinstance(line, str) and line.startswith(f"[{PB.TAG}") and " kit_served=msa_pair:4,pairformer:48" in line and " pf_cores=" in line and " variant=default" in line, line
    assert len([t for t in line.split() if t.startswith("cores=")]) == 1, "`cores=` is the core's head token alone"
    head = set(PF._evidence_head()) if hasattr(PF, "_evidence_head") else set()
    import inspect, re
    kit_keys = {"variant", "impl", "ln", "pf_cores", "kit_served", "core_cells", "c64"}
    assert not (kit_keys & head), f"the kit's keys never shadow the core's head tokens: {kit_keys & head}"
    src = inspect.getsource(PB.line)
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())                # the call's code, comments aside
    assert "kit_served=" in code and not re.search(r"[,(]\s*cores=", code), "the kit never passes the core's `cores=` key"

