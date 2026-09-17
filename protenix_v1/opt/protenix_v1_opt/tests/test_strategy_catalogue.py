"""Every strategy id this kit's LEVER lines name is in the shared core's strategy catalogue (``opt_core.strategies``: the canonical ids of
STRATEGIES.json, or a kit-local ``LOCAL.protenix_v1.<name>``; an alias or an unknown id fails naming the fact). A development-time check —
no run reads the catalogue; the LEVER line itself checks the id's form."""
from opt_core import strategies
from protenix_v1_opt import big as B


def test_every_strategy_id_is_in_the_catalogue():
    for lever, row in B.TABLE.items():
        assert strategies.check(row["strategy"]) == row["strategy"], lever
