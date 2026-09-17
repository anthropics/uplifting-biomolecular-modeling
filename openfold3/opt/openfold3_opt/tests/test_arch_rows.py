"""GPU-class support rows (opt_core.arch): every registry lever of this kit is declared once — tested on sm90 (H100, the class every lever
has test records on) plus sm80 (A100-80GB) for the levers of the exact, fast and big·tp lines (registry.A100_TESTED), floor sm80, no
exclusion known — and README §Applicability's GPU-classes column is registry.card_table()."""
from opt_core import arch

from openfold3_opt import registry


def test_every_lever_is_declared_once_with_the_kit_facts():
    sup = registry.declare_arch()
    assert set(sup) == set(registry.LEVERS) | set(registry.ARCH_LINES)
    for name, row in sup.items():
        assert row.certified == registry.tested_sm(name) and "sm90" in row.certified and row.min_sm == "sm80" and row.exclude == (), (name, row)
        assert ("sm80" in row.certified) == (name in registry.A100_TESTED), (name, row)
    assert set(registry.A100_TESTED) <= set(registry.LEVERS) | set(registry.ARCH_LINES)
    assert {"trimul_v4", "triatt_block", "pair_transition", "cuda_graphs", "tp_shard_s", "ds4sci_line"} <= set(registry.A100_TESTED) and not {"confhead", "of3tp_shard_s", "pair_offload_stream"} & set(registry.A100_TESTED)
    assert registry.declare_arch() == sup                                              # idempotent (arch.declare: same content twice is a no-op)
    assert arch.supports(registry.ARCH_PREFIX + "tp_shard_s", "sm90").word == "supported"
    assert arch.CARD_SM["H200"] == "sm90" and arch.CARD_SM["A100"] == "sm80" and arch.CARD_SM["B200"] == "sm100" and arch.CARD_SM["B300"] == "sm103"
    for card, word in (("sm80", "supported"), ("sm100", "uncertified"), ("sm103", "uncertified")):
        assert arch.supports(registry.ARCH_PREFIX + "cuda_graphs", card).word == word, (card, arch.supports(registry.ARCH_PREFIX + "cuda_graphs", card))
    assert arch.supports(registry.ARCH_PREFIX + "confhead", "sm80").word == "uncertified"        # a big·resident lever: sm80 is its untested floor
    assert "img_freeze_ds4sci_sm80_100" in sup["ds4sci_line"].note and "80GB" in sup["tp_shard_s"].note


def test_card_table_is_levers_by_class():
    t = registry.card_table()
    assert set(t) == set(registry.LEVERS) | set(registry.ARCH_LINES)
    assert t["tp_shard_s"] == {"sm80": "supported", "sm90": "supported", "sm100": "uncertified", "sm103": "uncertified"}, t["tp_shard_s"]
    assert t["confhead"] == {"sm80": "uncertified", "sm90": "supported", "sm100": "uncertified", "sm103": "uncertified"}, t["confhead"]
