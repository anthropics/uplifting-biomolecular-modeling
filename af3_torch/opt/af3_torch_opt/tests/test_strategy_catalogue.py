"""Every strategy id this kit's LEVER lines name is in the shared core's strategy catalogue (``opt_core.strategies``: the canonical ids of
STRATEGIES.json, or a kit-local ``LOCAL.af3_torch.<name>``; an alias or an unknown id fails naming the fact). A development-time check —
no run reads the catalogue; the LEVER line itself checks the id's form."""
from af3_torch_opt import registry
from opt_core import strategies


def test_every_strategy_id_is_in_the_catalogue():
    """registry.STRATEGY holds canonical ids, the catalogue's shared ``LOCAL.<name>`` ids, and this kit's ``LOCAL.af3_torch.<name>`` ids."""
    for lever, sid in registry.STRATEGY.items():
        assert strategies.check(sid) == sid, lever
