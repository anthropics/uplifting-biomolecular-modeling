"""Every strategy id this kit's LEVER lines name is in the shared core's strategy catalogue (``opt_core.strategies``: the canonical ids of
STRATEGIES.json, or a kit-local ``LOCAL.openfold3_ob0.<name>``; an alias or an unknown id fails naming the fact). A development-time check —
no run reads the catalogue; the LEVER line itself checks the id's form."""
from openfold3_ob0_opt import registry
from opt_core import strategies
from opt_core.mem import ngpu


def test_every_strategy_id_is_in_the_catalogue():
    for lever, sid in registry.STRATEGY.items():
        assert strategies.check(sid) == sid, lever
    assert strategies.check(ngpu.TP_LEVER) == ngpu.TP_LEVER                     # the tp line's tp_shard_s LEVER line (report.lever_lines)
