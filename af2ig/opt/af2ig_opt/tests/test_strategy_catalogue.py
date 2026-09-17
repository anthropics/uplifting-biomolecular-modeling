"""Every strategy id this kit's LEVER lines name is in the shared core's strategy catalogue (``opt_core.strategies``: the canonical ids of
STRATEGIES.json, or a kit-local ``LOCAL.af2ig.<name>``; an alias or an unknown id fails naming the fact). A development-time check —
no run reads the catalogue; the LEVER line itself checks the id's form."""
import unittest

from af2ig_opt import registry
from opt_core import strategies


class TestStrategyCatalogue(unittest.TestCase):
    def test_every_strategy_id_is_in_the_catalogue(self):
        for lv, lever in registry.LEVERS.items():
            self.assertEqual(strategies.check(lever.strategy), lever.strategy, lv)


if __name__ == "__main__":
    unittest.main()
