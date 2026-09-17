"""GPU-class support rows (opt_core.arch): every registry lever of this kit is declared once — tested on sm90 (H100, the class every lever
has test records on) plus sm80 (A100-80GB, configs/a100.env) for the levers registry.A100_TESTED names (the exact, fast, big·resident and big·tp
lines: every lever of this kit), floor sm80, no exclusion known — and README §Applicability's GPU-classes column is registry.card_table()."""
from opt_core import arch

from openfold3_ob0_opt import registry


NO_A100_RECORD = ("writer_overlap", "hostfeat", "ckpt_mmap", "trimul_provider", "trimul_form")   # the levers tested on H100 (sm90) only: an A100 card line names each `uncertified:sm80(...)`, by name


def test_every_lever_is_declared_once_with_the_kit_facts():
    sup = registry.declare_arch()
    assert set(sup) == set(registry.LEVERS)
    for name, row in sup.items():
        assert row.certified == registry.tested_sm(name) and "sm90" in row.certified and ("sm80" in row.certified) == (name in registry.A100_TESTED) and row.min_sm == "sm80" and row.exclude == (), (name, row)
    assert set(registry.A100_TESTED) <= set(registry.LEVERS)                          # a record names a lever of this kit
    assert set(registry.A100_TESTED) | set(NO_A100_RECORD) == set(registry.LEVERS) and not set(NO_A100_RECORD) & set(registry.A100_TESTED)   # every lever of this kit has one, except the named ones measured on sm90 only
    assert registry.declare_arch() == sup                                              # idempotent (arch.declare: same content twice is a no-op)
    assert arch.supports(registry.ARCH_PREFIX + "tp_shard_s", "sm90").word == "supported"
    assert arch.CARD_SM["H200"] == "sm90" and arch.CARD_SM["A100"] == "sm80" and arch.CARD_SM["B200"] == "sm100" and arch.CARD_SM["B300"] == "sm103"
    for card, word in (("sm80", "supported"), ("sm100", "uncertified"), ("sm103", "uncertified")):
        assert arch.supports(registry.ARCH_PREFIX + "cuda_graphs", card).word == word, (card, arch.supports(registry.ARCH_PREFIX + "cuda_graphs", card))
    assert "80GB" in sup["tp_shard_s"].note


def test_card_table_is_levers_by_class():
    t = registry.card_table()
    assert set(t) == set(registry.LEVERS)
    assert t["tp_shard_s"] == {"sm80": "supported", "sm90": "supported", "sm100": "uncertified", "sm103": "uncertified"}, t["tp_shard_s"]
    assert t["confhead"] == {"sm80": "supported", "sm90": "supported", "sm100": "uncertified", "sm103": "uncertified"}, t["confhead"]
