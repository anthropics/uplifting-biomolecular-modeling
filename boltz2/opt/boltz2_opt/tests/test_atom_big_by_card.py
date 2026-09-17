"""The memory row's fused atom kernels, by card. fast's atom_fused / atom_gemm with their static buffers released per sample() (BOLTZ_ATOM_RELEASE=sample)
ride `big` where that placement is peak-neutral — sm_90: PEAK alloc +0.01 GiB, NVML +8 MiB at 1200 / 2000 tokens for -1.0 / -1.1 s of sampler — and leave
BY NAME on sm_80, where the kernels hold +3.5 GB of device memory outside torch's allocator (NVML 13279 -> 16797 MiB at 800 tokens, 19961 -> 23479 at 1200;
reserved unchanged): CARD_DROPS["8.0"]["big"], reason measured_memory_nvml_+3.5GB:sm_80. The one measured memory cost decides per card; the exact key gather
serves the stock atom layers wherever the fused kernels are off; the xP line keeps the stock atom layers on every card (not measured there)."""
from boltz2_opt import modes, report

REASON = "measured_memory_nvml_+3.5GB:sm_80"
UNITS = {"atom_fused", "atom_gemm"}


def _big(cc, memory_mib, n_gpu=1):
    modes.set_card(cc, memory_mib); modes.set_n_gpu(n_gpu)
    try:
        return modes.resolve("big")
    finally:
        modes.set_card(None); modes.set_n_gpu(1)


def test_no_row_level_entry_the_card_table_decides():
    assert not (UNITS & set(modes.MODES["big"]["off"])) and UNITS <= set(modes.MODES["big"]["levers"]) and "atom_fused" in modes.NEEDS["atom_gemm"]
    assert modes.CARD_DROPS["8.0"]["big"]["atom_fused"] == {"reason": REASON} and "atom_fused" not in modes.CARD_DROPS["8.0"].get("fast", {}) \
        and not any("atom_fused" in d for d in modes.CARD_DROPS.get("10.0", {}).values()), "the memory row on sm_80 only: fast carries the kernels on every card; other cards carry them on the memory row as fast does"


def test_sm90_big_carries_the_fused_atom_kernels_with_their_release_word():
    r = _big("9.0", 81559)
    assert [l for l in r["levers"] if l in UNITS] == ["atom_fused", "atom_gemm"] and "atom" in r["attach"] and not r.get("card_off")
    assert r["env"]["BOLTZ_ATOM"] == "fused" and r["env"]["BOLTZ_ATOM_GEMM"] == modes.env_row("fast")["BOLTZ_ATOM_GEMM"] and r["env"]["BOLTZ_ATOM_RELEASE"] == "sample"
    assert "atom_keys_gather" not in r["levers"] and r["off"]["atom_keys_gather"].startswith("no_call_site:"), \
        "on the composed row the fused kernels serve every atom layer and the DiT hoist is off by rule: the exact key gather has no caller and leaves by name (CONDITIONAL_RIDERS)"
    assert "card_off=" not in report.off_token(r) and "atom_keys_gather:no_call_site:" in report.off_token(r)


def test_sm80_big_names_the_fused_atom_kernels_off_with_the_measured_reason():
    for memory_mib in (81920, 40960):
        r = _big("8.0", memory_mib)
        assert not (UNITS & set(r["levers"])) and r["card_off"]["atom_fused"] == REASON == r["card_off"]["atom_gemm"], "atom_gemm rides atom_fused (NEEDS): both leave by the card's reason"
        assert r["env"]["BOLTZ_ATOM"] == "keys" and "atom_keys_gather" in r["levers"] and "atom" in r["attach"] and not ({"BOLTZ_ATOM_GEMM", "BOLTZ_ATOM_RELEASE"} & set(r["env"])), \
            "the lever leaves with its words (registry `words`); the stock atom layers run and the exact key gather serves them"
        assert f"atom_fused:{REASON},atom_gemm:{REASON}" in report.off_token(r) and report.off_token(r).startswith(" card_off=atom_fused:")


def test_other_cards_carry_the_kernels_as_fast_does():
    r = _big("10.0", 183359)
    assert [l for l in r["levers"] if l in UNITS] == ["atom_fused", "atom_gemm"] and not (UNITS & set(r.get("card_off") or {})) and r["env"]["BOLTZ_ATOM"] == "fused"


def test_the_xP_line_keeps_the_stock_atom_layers_on_every_card():
    for cc, memory_mib in (("9.0", 81559), ("8.0", 81920), ("10.0", 183359)):
        for n_gpu in (2, 4, 8):
            r = _big(cc, memory_mib, n_gpu)
            assert not (UNITS & set(r["levers"])) and "atom_keys_gather" in r["levers"] and r["env"]["BOLTZ_ATOM"] == "keys" and not ({"BOLTZ_ATOM_GEMM", "BOLTZ_ATOM_RELEASE"} & set(r["env"]))
            if cc == "8.0":
                assert r["card_off"]["atom_fused"] == REASON and not (UNITS & set(r.get("tp_off") or {})), "the card drop names them first; the xP table has nothing left to take"
            else:
                assert r["tp_off"]["atom_fused"].startswith("xP_line_not_measured:") and r["tp_off"]["atom_gemm"] == r["tp_off"]["atom_fused"] and modes.TP_DROPS["big"]["atom_gemm"] == modes.TP_DROPS["big"]["atom_fused"]
