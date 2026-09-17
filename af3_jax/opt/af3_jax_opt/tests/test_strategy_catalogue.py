"""Every strategy id this kit's LEVER lines name is in the shared core's strategy catalogue (``opt_core.strategies``: the canonical ids of
STRATEGIES.json, or a kit-local ``LOCAL.af3_jax.<name>``; an alias or an unknown id fails naming the fact). A development-time check —
no run reads the catalogue; the LEVER line itself checks the id's form."""
from af3_jax_opt import registry as kit_registry
from opt_core import strategies


def test_every_strategy_id_is_in_the_catalogue():
    for lv, row in kit_registry.LEVERS.items():
        if row.get("strategy"):
            assert strategies.check(row["strategy"]) == row["strategy"], lv
