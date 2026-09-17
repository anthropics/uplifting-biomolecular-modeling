"""Every strategy id this kit's LEVER lines name is in the shared core's strategy catalogue (``opt_core.strategies``: the canonical ids of
STRATEGIES.json, or a kit-local ``LOCAL.boltz2.<name>``; an alias or an unknown id fails naming the fact). A development-time check —
no run reads the catalogue; the LEVER line itself checks the id's form."""
from boltz2_opt import registry
from opt_core import strategies


def test_every_strategy_id_is_in_the_catalogue():
    for name, L in registry.LEVERS.items():
        assert strategies.check(L["strategy"]) == L["strategy"], name
